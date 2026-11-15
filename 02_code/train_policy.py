"""
Train and score the five policies (Marshal, 13.11.26); made for the university's GPU pool, runs on any computer.

One run trains one policy with one seed on a dataset from build_dataset.py, keeps the epoch with the best score on the
validation episodes, and writes into ``04_runs/<batch>_<model>_seed<N>/``:

- metrics.json: the validation, test (and held-out sheet) scores of the best epoch, per sheet, and the settings;
- history.csv: the loss and validation scores after every epoch;
- checkpoints/best.pt: the weights, with everything the robot needs to use them (not in git).

All five policies are scored the same way, so they can be compared: the predicted move (for the number-predicting
policies, the direction of the predicted step, as the recording tool names it), the forearm label, and stop.

Run from 02_code:
    python train_policy.py --check                                  # before a night: GPU, data, weights, every model
    python train_policy.py --model factorised_classifier --seed 1 --batch 2026-11-13_overnight
    python train_policy.py --summary --batch 2026-11-13_overnight   # one table of every run of the batch
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
import torch
from torch.nn import functional

from config import POLICY_MODEL_PARAMETERS
from policy_data import PolicyData, merged_training_parameters, picture_batch
from policy_models import POLICY_CLASSES, build_policy_model


LOADED_DATA: Dict[Tuple, PolicyData] = {}


def merged_model_parameters(
        model_name: str,
        model_overrides: Optional[Dict] = None,
        training_overrides: Optional[Dict] = None
) -> Tuple[Dict, Dict]:
    """
    A policy's own settings and its training settings, each with any overrides.

    The training settings are ``TRAINING_PARAMETERS``, then the policy's "training" entry in
    ``POLICY_MODEL_PARAMETERS``, then ``training_overrides``.
    """

    model_parameters = dict(POLICY_MODEL_PARAMETERS[model_name])
    model_parameters.update(model_overrides or {})
    training = dict(model_parameters.pop("training", {}))
    training.update(training_overrides or {})
    return model_parameters, merged_training_parameters(training)


def choose_device(name: Optional[str] = None) -> torch.device:
    """The named device, else a CUDA GPU, else Apple's GPU, else the CPU."""

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(
        training_parameters: Dict,
        model_parameters: Dict
) -> PolicyData:
    """The dataset, read once per process and kept for later runs with the same settings (Optuna trials)."""

    key = (str(training_parameters["processed_folder"]), training_parameters["dataset_name"], tuple(training_parameters["cache_size"]),
           training_parameters["held_out_sheet"], model_parameters.get("chunk_length", 8), model_parameters.get("history_length", 3))
    if key not in LOADED_DATA:
        LOADED_DATA[key] = PolicyData(training_parameters, chunk_length = key[4], history_length = key[5])
    data = LOADED_DATA[key]
    # Settings that only change weighting or sampling are read from here, not frozen into the loaded data.
    data.parameters = training_parameters
    return data


def class_counts(parameters: Dict) -> Dict[str, int]:
    """How many values each output has."""

    return {"move": len(parameters["move_classes"]), "forearm": len(parameters["forearm_classes"]),
            "lean": len(parameters["lean_classes"]), "exact": len(parameters["exact_columns"])}


def seed_everything(seed: int) -> None:
    """Seed Python, numpy and torch, so a run can be repeated."""

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def chunk_targets(
        tensors: Dict[str, torch.Tensor],
        indices: torch.Tensor,
        chunk_length: int
) -> Dict[str, torch.Tensor]:
    """
    The targets of the next ``chunk_length`` steps from each picture.

    Steps after the episode's end are "stopped": stop is true, the move is stop, and the exact step is not scored.
    """

    chunk = tensors["chunk"][indices][:, :chunk_length]
    is_valid = chunk >= 0
    safe = chunk.clamp(min = 0)
    is_stop = torch.where(is_valid, tensors["is_stop"][safe], torch.ones_like(is_valid))
    moves =torch.where(is_valid, tensors["moves"][safe], torch.full_like(safe, -1))
    return {"is_valid": is_valid, "is_stop": is_stop, "moves": moves, "exact_scaled": tensors["exact_scaled"][safe]}


def fill_stop_moves(
        moves: torch.Tensor,
        stop_index: int
) -> torch.Tensor:
    """Steps after the episode's end (move -1) become the stop move."""

    return torch.where(moves < 0, torch.full_like(moves, stop_index), moves)


def model_inputs(
        model: torch.nn.Module,
        model_name: str,
        tensors: Dict[str, torch.Tensor],
        indices: torch.Tensor,
        is_training: bool,
        parameters: Dict,
        device: torch.device
) -> Dict[str, torch.Tensor]:
    """The pictures (and, for the history policy, the earlier pictures and the previous move) of one batch."""

    if model_name == "history_classifier":
        history = tensors["history"][indices][:, :model.history_length]
        return {"pictures": picture_batch(tensors["pictures"], history, is_training, parameters, device),
                "previous_move": tensors["previous_move"][indices]}
    return {"pictures": picture_batch(tensors["pictures"], indices, is_training, parameters, device)}


def batch_loss(
        model: torch.nn.Module,
        model_name: str,
        tensors: Dict[str, torch.Tensor],
        indices: torch.Tensor,
        parameters: Dict,
        move_weights: torch.Tensor,
        device: torch.device
) -> torch.Tensor:
    """
    The training loss of one batch, each sample weighted by its episode's score weight.

    Classifiers: weighted cross-entropy on the move, cross-entropy on the forearm and lean labels, binary cross-entropy
    on "at goal". Regressor: Huber loss on the scaled exact step of moving steps and binary cross-entropy on stop.
    Chunk transformer: the same over every step of the chunk, plus cross-entropy on each step's move. Diffusion: the
    noise-prediction error.
    """

    weights = tensors["score_weights"][indices]
    loss_weights = parameters["loss_weights"]
    stop_index = parameters["move_classes"].index("stop")
    inputs = model_inputs(model, model_name, tensors, indices, True, parameters, device)

    def weighted(per_sample: torch.Tensor) -> torch.Tensor:
        return (per_sample * weights).sum() / weights.sum()

    if model.kind == "diffusion":
        targets = chunk_targets(tensors, indices, model.chunk_length)
        stop_value = torch.where(targets["is_stop"], torch.ones_like(weights)[:, None], -torch.ones_like(weights)[:, None])
        exact = torch.where(targets["is_stop"][..., None], torch.zeros_like(targets["exact_scaled"]), targets["exact_scaled"])
        return model.training_loss(inputs["pictures"], torch.cat([exact, stop_value[..., None]], dim = 2), weights)

    outputs = model(**inputs)
    if model.kind == "classifier":
        move = functional.cross_entropy(outputs["move"], tensors["moves"][indices], weight = move_weights, reduction = "none")
        forearm = functional.cross_entropy(outputs["forearm"], tensors["forearm_labels"][indices], reduction = "none")
        lean = functional.cross_entropy(outputs["lean"], tensors["lean_labels"][indices], reduction = "none")
        at_goal = functional.binary_cross_entropy_with_logits(outputs["at_goal"], tensors["at_goal"][indices], reduction = "none")
        return (loss_weights["move"] * weighted(move) + loss_weights["forearm"] * weighted(forearm)
                + loss_weights["lean"] * weighted(lean) + loss_weights["at_goal"] * weighted(at_goal))

    if model.kind == "regressor":
        is_moving = (~tensors["is_stop"][indices]).float()
        exact = functional.huber_loss(outputs["exact"], tensors["exact_scaled"][indices], reduction = "none",
                                      delta = model.huber_delta).mean(dim = 1)
        stop = functional.binary_cross_entropy_with_logits(outputs["stop"], tensors["is_stop"][indices].float(), reduction = "none")
        exact_loss = (exact * is_moving * weights).sum() / (is_moving * weights).sum().clamp(min = 1e-6)
        return loss_weights["exact"] * exact_loss + loss_weights["stop"] * weighted(stop)

    # Chunk transformer.
    targets = chunk_targets(tensors, indices, model.chunk_length)
    is_moving = (targets["is_valid"] & ~targets["is_stop"]).float()
    exact = (outputs["exact"] - targets["exact_scaled"]).abs().mean(dim = 2)
    exact_loss = (exact * is_moving * weights[:, None]).sum() / (is_moving * weights[:, None]).sum().clamp(min = 1e-6)
    stop = functional.binary_cross_entropy_with_logits(outputs["stop"], targets["is_stop"].float(), reduction = "none").mean(dim = 1)
    moves = fill_stop_moves(targets["moves"], stop_index)
    move = functional.cross_entropy(outputs["move"].flatten(0, 1), moves.flatten(), weight = move_weights,
                                    reduction = "none").reshape(moves.shape).mean(dim = 1)
    return loss_weights["exact"] * exact_loss + loss_weights["stop"] * weighted(stop) + loss_weights["move"] * weighted(move)


def moves_from_exact(
        forward_and_left: torch.Tensor,
        is_stop: torch.Tensor,
        parameters: Dict
) -> torch.Tensor:
    """The move name of a step as the recording tool names it: the nearest direction, or stop."""

    move_classes = parameters["move_classes"]
    names = list(parameters["action_angles_deg"])
    angles = torch.tensor([parameters["action_angles_deg"][name] for name in names], device = forward_and_left.device, dtype = torch.float32)
    picture_angle = torch.rad2deg(torch.atan2(forward_and_left[:, 1], forward_and_left[:, 0]))
    difference = torch.remainder(picture_angle[:, None] - angles[None, :] + 180.0, 360.0) - 180.0
    nearest = difference.abs().argmin(dim = 1)
    class_of_name = torch.tensor([move_classes.index(name) for name in names], device = forward_and_left.device)
    moves = class_of_name[nearest]
    return torch.where(is_stop, torch.full_like(moves, move_classes.index("stop")), moves)


def forearm_from_exact(
        forearm_change_deg: torch.Tensor,
        parameters: Dict
) -> torch.Tensor:
    """The forearm label of a step from its forearm change, with the recording tool's threshold."""

    classes = parameters["forearm_classes"]
    threshold = parameters["forearm_label_deg"]
    labels = torch.full_like(forearm_change_deg, classes.index("none"), dtype = torch.long)
    labels = torch.where(forearm_change_deg >= threshold, torch.full_like(labels, classes.index("turn_left")), labels)
    return torch.where(forearm_change_deg <= -threshold, torch.full_like(labels, classes.index("turn_right")), labels)


@torch.no_grad()
def predict(
        model: torch.nn.Module,
        model_name: str,
        data: PolicyData,
        tensors: Dict[str, torch.Tensor],
        indices: numpy.ndarray,
        parameters: Dict,
        device: torch.device,
        seed: int = 0
) -> Dict[str, numpy.ndarray]:
    """
    Each picture's predicted move and forearm label, and the exact step for the number-predicting policies.

    For the chunk and diffusion policies only the first step of the chunk counts here, as on the robot.
    """

    model.eval()
    exact_mean = torch.from_numpy(data.exact_mean).to(device)
    exact_std = torch.from_numpy(data.exact_std).to(device)
    generator = torch.Generator(device = device).manual_seed(seed) if model.kind == "diffusion" else None
    moves, forearms, exacts = [], [], []
    for start in range(0, len(indices), parameters["evaluation_batch_size"]):
        batch = torch.from_numpy(indices[start:start + parameters["evaluation_batch_size"]]).to(device)
        inputs = model_inputs(model, model_name, tensors, batch, False, parameters, device)
        if model.kind == "classifier":
            outputs = model(**inputs)
            moves.append(outputs["move"].argmax(dim = 1))
            forearms.append(outputs["forearm"].argmax(dim = 1))
            continue
        if model.kind == "diffusion":
            first = model.sample(inputs["pictures"], generator = generator)[:, 0]
            exact_scaled, is_stop = first[:, :-1], first[:, -1] > 0
        else:
            outputs = model(**inputs)
            exact_scaled = outputs["exact"] if model.kind == "regressor" else outputs["exact"][:, 0]
            stop_logit = outputs["stop"] if model.kind == "regressor" else outputs["stop"][:, 0]
            is_stop = stop_logit > 0
        exact = exact_scaled.float() * exact_std + exact_mean
        moves.append(moves_from_exact(exact[:, :2], is_stop, parameters))
        forearms.append(forearm_from_exact(exact[:, 2], parameters))
        exacts.append(exact)
    result = {"move": torch.cat(moves).cpu().numpy(), "forearm": torch.cat(forearms).cpu().numpy()}
    if exacts:
        result["exact"] = torch.cat(exacts).cpu().numpy()
    return result


def scores_of(
        predictions: Dict[str, numpy.ndarray],
        data: PolicyData,
        indices: numpy.ndarray,
        parameters: Dict
) -> Dict:
    """
    How well the predictions match the operators, in total and per sheet.

    "move_balanced_accuracy" is the mean, over the moves that occur, of the share of each move named right; it is not
    fooled by a policy that always says forward. "stop_recall" is the share of stop pictures called stop.
    """

    move_classes = parameters["move_classes"]
    stop_index = move_classes.index("stop")
    true_moves = data.moves[indices]
    true_forearm = data.forearm_labels[indices]
    sheets = numpy.array([data.sheets[index] for index in indices])

    def group_scores(mask: numpy.ndarray) -> Dict:
        if not mask.any():
            return {}
        truth, guess = true_moves[mask], predictions["move"][mask]
        recalls = {move_classes[move]: float((guess[truth == move] == move).mean()) for move in numpy.unique(truth)}
        said_stop = guess == stop_index
        group = {
            "pictures": int(mask.sum()),
            "move_accuracy": float((guess == truth).mean()),
            "move_balanced_accuracy": float(numpy.mean(list(recalls.values()))),
            "forearm_accuracy": float((predictions["forearm"][mask] == true_forearm[mask]).mean()),
            "stop_recall": recalls.get("stop"),
            "stop_precision": float((truth[said_stop] == stop_index).mean()) if said_stop.any() else None,
            "move_recall": recalls
        }
        if "exact" in predictions:
            is_moving = truth != stop_index
            if is_moving.any():
                error = predictions["exact"][mask][is_moving] - data.exact[indices][mask][is_moving]
                group["tip_error_mm"] = float(numpy.hypot(error[:, 0], error[:, 1]).mean() * 1000.0)
                group["forearm_error_deg"] = float(numpy.abs(error[:, 2]).mean())
        return group

    result = {"all": group_scores(numpy.ones(len(indices), dtype = bool))}
    result["per_sheet"] = {sheet: group_scores(sheets == sheet) for sheet in sorted(set(sheets.tolist()))}
    hard = [result["per_sheet"][sheet] for sheet in parameters["hard_sheets"] if result["per_sheet"].get(sheet)]
    groups = hard or [result["all"]]
    result["hard_sheet_score"] = float(numpy.mean([0.5 * (group["move_balanced_accuracy"] + group["forearm_accuracy"]) for group in groups]))
    return result


def learning_rate_factor(
        step: int,
        warmup_steps: int,
        total_steps: int
) -> float:
    """Linear warm-up, then a cosine fall to zero."""

    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def plain_settings(settings: Dict) -> Dict:
    """Settings with folders as text, so they can be written as JSON."""

    return {key: (str(value) if isinstance(value, Path) else value) for key, value in settings.items()}


def train_one_run(
        model_name: str,
        seed: int,
        batch_name: str,
        training_overrides: Optional[Dict] = None,
        model_overrides: Optional[Dict] = None,
        run_label: Optional[str] = None,
        device_name: Optional[str] = None,
        is_done_skipped: bool = False,
        trial = None
) -> Dict:
    """
    Train one policy with one seed, keep its best epoch, score it, and write its run folder.

    Parameters
    ----------
    model_name : str
        One of the five policies.
    seed : int
        Seed for the weights' start, the sampling and the augmentation.
    batch_name : str
        The run folder is ``04_runs/<batch_name>_<run_label>_seed<seed>``.
    training_overrides, model_overrides : dict, optional
        Overrides for the training settings and the policy's own settings.
    run_label : str, optional
        Name in the run folder instead of the model name (Optuna trials, the tuned model).
    device_name : str, optional
        "cuda", "mps" or "cpu"; chosen automatically when not given.
    is_done_skipped : bool, default False
        Return the stored result when the run folder already holds a finished run (to restart a night).
    trial : optuna.Trial, optional
        Reports the validation score after each epoch, so Optuna can stop a poor trial early.

    Returns
    -------
    dict
        The content of metrics.json.
    """

    model_parameters, parameters = merged_model_parameters(model_name, model_overrides, training_overrides)
    run_folder = parameters["runs_folder"] / f"{batch_name}_{run_label or model_name}_seed{seed}"
    metrics_path = run_folder / "metrics.json"
    if is_done_skipped and metrics_path.exists():
        stored = json.loads(metrics_path.read_text(encoding = "utf-8"))
        if stored.get("status") == "done":
            print(f"{run_folder.name}: already done, skipped")
            return stored

    seed_everything(seed)
    device = choose_device(device_name)
    data = load_data(parameters, model_parameters)
    tensors = data.tensors(device)
    model = build_policy_model(model_name, class_counts(parameters), model_parameters, parameters["is_pretrained"]).to(device)
    move_weights = torch.from_numpy(data.move_class_weights()).to(device)

    train_indices = data.indices("train")
    validation_indices = data.indices("validation")
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("the dataset has no train or no validation pictures")
    sampling = torch.from_numpy(data.sampling_weights(train_indices))
    train_tensor = torch.from_numpy(train_indices)
    steps_per_epoch = max(1, len(train_indices) // parameters["batch_size"])
    if parameters.get("max_batches_per_epoch"):
        steps_per_epoch = min(steps_per_epoch, parameters["max_batches_per_epoch"])
    total_steps = steps_per_epoch * parameters["epochs"]
    warmup_steps = max(1, int(steps_per_epoch * parameters["warmup_epochs"]))

    optimiser = torch.optim.AdamW(model.parameters(), lr = parameters["learning_rate"], weight_decay = parameters["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: learning_rate_factor(step, warmup_steps, total_steps))
    is_mixed_precision = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled = is_mixed_precision)

    (run_folder / "checkpoints").mkdir(parents = True, exist_ok = True)
    history_path = run_folder / "history.csv"
    history_columns = ["epoch", "train_loss", "hard_sheet_score", "move_accuracy", "move_balanced_accuracy", "forearm_accuracy",
                       "stop_recall", "learning_rate", "minutes"]
    with open(history_path, "w", encoding = "utf-8", newline = "") as file:
        csv.writer(file).writerow(history_columns)
    metrics_path.write_text(json.dumps({"status": "running", "model": model_name, "seed": seed}), encoding = "utf-8")

    started = time.time()
    best_score, best_epoch, epochs_without_gain, stopped_because = -1.0, 0, 0, "all epochs"
    print(f"{run_folder.name}: {len(train_indices)} train and {len(validation_indices)} validation pictures on {device}", flush = True)
    for epoch in range(1, parameters["epochs"] + 1):
        model.train()
        drawn = train_tensor[torch.multinomial(sampling, steps_per_epoch * parameters["batch_size"], replacement = True)]
        losses = []
        for step in range(steps_per_epoch):
            indices = drawn[step * parameters["batch_size"]:(step + 1) * parameters["batch_size"]].to(device)
            with torch.autocast(device_type = "cuda", dtype = torch.float16, enabled = is_mixed_precision):
                loss = batch_loss(model, model_name, tensors, indices, parameters, move_weights, device)
            optimiser.zero_grad(set_to_none = True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))

        validation = scores_of(predict(model, model_name, data, tensors, validation_indices, parameters, device, seed), data,
                               validation_indices, parameters)
        minutes = (time.time() - started) / 60.0
        row = [epoch, numpy.mean(losses), validation["hard_sheet_score"], validation["all"]["move_accuracy"],
               validation["all"]["move_balanced_accuracy"], validation["all"]["forearm_accuracy"], validation["all"]["stop_recall"],
               scheduler.get_last_lr()[0], minutes]
        with open(history_path, "a", encoding = "utf-8", newline = "") as file:
            csv.writer(file).writerow([f"{value:.5f}" if isinstance(value, float) else value for value in row])
        print(f"  epoch {epoch}: loss {numpy.mean(losses):.4f}, validation hard-sheet score {validation['hard_sheet_score']:.3f}, "
              f"move {validation['all']['move_balanced_accuracy']:.3f}, forearm {validation['all']['forearm_accuracy']:.3f} "
              f"({minutes:.1f} min)", flush = True)

        if validation["hard_sheet_score"] > best_score:
            best_score, best_epoch, epochs_without_gain = validation["hard_sheet_score"], epoch, 0
            torch.save({"model_name": model_name, "model_state": model.state_dict(), "model_parameters": model_parameters,
                        "training_parameters": plain_settings(parameters), "exact_mean": data.exact_mean.tolist(),
                        "exact_std": data.exact_std.tolist(), "epoch": epoch, "validation": validation,
                        "picture_preparation": "cv2.INTER_AREA to cache_size, BGR to RGB, centre crop to crop_size, ImageNet normalisation"},
                       run_folder / "checkpoints" / "best.pt")
        else:
            epochs_without_gain += 1
        if trial is not None:
            trial.report(validation["hard_sheet_score"], epoch)
            if trial.should_prune():
                stopped_because = "pruned by Optuna"
                break
        if epochs_without_gain >= parameters["patience_epochs"]:
            stopped_because = f"no gain for {parameters['patience_epochs']} epochs"
            break
        if minutes >= parameters["max_minutes_per_run"]:
            stopped_because = f"time limit of {parameters['max_minutes_per_run']} minutes"
            break

    checkpoint = torch.load(run_folder / "checkpoints" / "best.pt", map_location = device, weights_only = False)
    model.load_state_dict(checkpoint["model_state"])
    results = {"status": "done", "model": model_name, "run_label": run_label or model_name, "seed": seed, "batch": batch_name,
               "best_epoch": best_epoch, "epochs_run": epoch, "stopped_because": stopped_because,
               "minutes": round((time.time() - started) / 60.0, 2), "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "train_pictures": int(len(train_indices)), "validation": checkpoint["validation"]}
    for split in ("test", "held_out"):
        split_indices = data.indices(split)
        if len(split_indices):
            results[split] = scores_of(predict(model, model_name, data, tensors, split_indices, parameters, device, seed), data,
                                       split_indices, parameters)
    results["model_parameters"] = model_parameters
    results["training_parameters"] = plain_settings(parameters)
    metrics_path.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    if not parameters["is_checkpoint_kept"]:
        (run_folder / "checkpoints" / "best.pt").unlink(missing_ok = True)
    print(f"{run_folder.name}: best epoch {best_epoch}, validation hard-sheet score {best_score:.3f} ({stopped_because})", flush = True)
    return results


def check_setup(
        device_name: Optional[str] = None,
        training_overrides: Optional[Dict] = None
) -> None:
    """
    Check, before a night of training, everything that would otherwise fail at 2 a.m.: the GPU, the dataset and its
    picture cache, the ImageNet weights, and one training step and one prediction of every policy.
    """

    device = choose_device(device_name)
    print(f"torch {torch.__version__}, device {device}", flush = True)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU {properties.name}, {properties.total_memory / 1e9:.1f} GB", flush = True)
    for model_name in POLICY_CLASSES:
        model_parameters, parameters = merged_model_parameters(model_name, training_overrides = training_overrides)
        data = load_data(parameters, model_parameters)
        tensors = data.tensors(device)
        model = build_policy_model(model_name, class_counts(parameters), model_parameters, parameters["is_pretrained"]).to(device)
        indices = torch.from_numpy(data.indices("train")[:8]).to(device)
        model.train()
        loss = batch_loss(model, model_name, tensors, indices, parameters, torch.from_numpy(data.move_class_weights()).to(device), device)
        loss.backward()
        predictions = predict(model, model_name, data, tensors, data.indices("validation")[:8], parameters, device)
        print(f"  {model_name}: loss {float(loss.detach()):.3f}, {len(predictions['move'])} predictions: OK", flush = True)
    counts = {split: len(data.indices(split)) for split in ("train", "validation", "test", "held_out")}
    print(f"dataset {parameters['dataset_name']}: {counts}; everything ready", flush = True)


def write_summary(
        batch_name: str,
        training_overrides: Optional[Dict] = None
) -> Path:
    """
    One table of every finished run of a batch, averaged over seeds, written to ``04_runs/<batch>_summary.md``.

    Returns
    -------
    Path
        The summary file.
    """

    parameters = merged_training_parameters(training_overrides)
    runs = []
    for metrics_path in sorted(parameters["runs_folder"].glob(f"{batch_name}_*/metrics.json")):
        stored = json.loads(metrics_path.read_text(encoding = "utf-8"))
        if stored.get("status") == "done":
            runs.append(stored)
    by_label: Dict[str, List[Dict]] = {}
    for run in runs:
        by_label.setdefault(run["run_label"], []).append(run)

    def mean_and_spread(values: List[Optional[float]]) -> str:
        values = [value for value in values if value is not None]
        if not values:
            return "-"
        return f"{numpy.mean(values):.3f}" + (f" ± {numpy.std(values):.3f}" if len(values) > 1 else "")

    sheets = sorted({sheet for run in runs for sheet in run["validation"]["per_sheet"]})
    lines = [f"# Training batch {batch_name}", "",
             "Validation scores of the best epoch, mean ± spread over seeds. Hard-sheet score: mean over sheets 03, 04 and 06 "
             "of (move balanced accuracy + forearm accuracy) / 2. Test scores are for the report only; nothing is chosen by them.",
             "", "| Policy | Seeds | Hard-sheet score | Move balanced accuracy | Forearm accuracy | Stop recall | Test hard-sheet score |",
             "|---|---|---|---|---|---|---|"]
    for label, label_runs in sorted(by_label.items(), key = lambda item: -numpy.mean([run["validation"]["hard_sheet_score"] for run in item[1]])):
        lines.append(f"| {label} | {len(label_runs)} | {mean_and_spread([run['validation']['hard_sheet_score'] for run in label_runs])} | "
                     f"{mean_and_spread([run['validation']['all']['move_balanced_accuracy'] for run in label_runs])} | "
                     f"{mean_and_spread([run['validation']['all']['forearm_accuracy'] for run in label_runs])} | "
                     f"{mean_and_spread([run['validation']['all']['stop_recall'] for run in label_runs])} | "
                     f"{mean_and_spread([run.get('test', {}).get('hard_sheet_score') for run in label_runs])} |")
    lines += ["", "Move balanced accuracy per sheet (validation):", "", "| Policy | " + " | ".join(sheets) + " |",
              "|---|" + "---|" * len(sheets)]
    for label, label_runs in sorted(by_label.items()):
        cells = [mean_and_spread([run["validation"]["per_sheet"].get(sheet, {}).get("move_balanced_accuracy") for run in label_runs])
                 for sheet in sheets]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| Run | Best epoch | Epochs | Stopped because | Minutes | Device |", "|---|---|---|---|---|---|"]
    for run in runs:
        lines.append(f"| {run['run_label']} seed {run['seed']} | {run['best_epoch']} | {run['epochs_run']} | {run['stopped_because']} | "
                     f"{run['minutes']} | {run.get('gpu') or run['device']} |")
    summary_path = parameters["runs_folder"] / f"{batch_name}_summary.md"
    summary_path.parent.mkdir(parents = True, exist_ok = True)
    summary_path.write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return summary_path


def main() -> None:
    """Command line: check the set-up, train one run, or summarise a batch."""

    parser = argparse.ArgumentParser(description = "Train and score the policies.")
    parser.add_argument("--model", choices = sorted(POLICY_CLASSES), help = "the policy to train")
    parser.add_argument("--seed", type = int, default = 1)
    parser.add_argument("--batch", default = f"{date.today().isoformat()}_trial", help = "name shared by the runs of one night")
    parser.add_argument("--device", help = "cuda, mps or cpu (default: the best available)")
    parser.add_argument("--epochs", type = int, help = "override the number of epochs")
    parser.add_argument("--held-out-sheet", help = "leave this sheet (e.g. path_04) out of training and score it alone")
    parser.add_argument("--no-pretrained", action = "store_true", help = "start ResNet-18 from random weights")
    parser.add_argument("--skip-done", action = "store_true", help = "skip a run whose folder already holds a finished run")
    parser.add_argument("--check", action = "store_true", help = "check GPU, data, weights and every policy, then stop")
    parser.add_argument("--summary", action = "store_true", help = "write the summary table of --batch")
    arguments = parser.parse_args()

    overrides = {}
    if arguments.epochs:
        overrides["epochs"] = arguments.epochs
    if arguments.held_out_sheet:
        overrides["held_out_sheet"] = arguments.held_out_sheet
    if arguments.no_pretrained:
        overrides["is_pretrained"] = False
    if arguments.check:
        check_setup(arguments.device, overrides)
    elif arguments.summary:
        print(write_summary(arguments.batch, overrides).read_text(encoding = "utf-8"))
    elif arguments.model:
        train_one_run(arguments.model, arguments.seed, arguments.batch, training_overrides = overrides,
                      device_name = arguments.device, is_done_skipped = arguments.skip_done)
    else:
        sys.exit("name a --model, or use --check or --summary")


if __name__ == "__main__":
    main()
