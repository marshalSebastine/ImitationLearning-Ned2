"""
Train a picture-to-move model end to end and score it on held-out sheets or episodes (9.11.26, Model and training).

Unlike training_probes.py, the whole network learns here, from the pictures themselves, with augmentation. A run is a
named entry of TRAINING_RUNS in config.py: the network (a small network from scratch, or a pretrained one that is
fine-tuned), the target design (the same heads and losses as the probes), the augmentation and the training schedule.
Every run is scored with the shared offline measures, fold by fold, so its numbers sit beside the probes' numbers.

Augmentation, and why each kind keeps the label right:
    colour      brightness, contrast, saturation and gamma: the light changes, the right move does not;
    blur, noise the camera's focus and the dim room change the picture, not the move;
    shift       a few pixels only: the taped camera creeps a few pixels between days (camera_shift_px), while a large
                shift would move the line relative to the puck and change the right move;
    mirror      left and right swapped in the picture AND in the label (left moves become right moves, twists change
                sign); off by default, because the camera does not sit on the rod's axis, so a mirrored picture shows
                a view the camera can never have (tested as its own run).

Run from 02_code, after training_dataset.py:
    ..\\.venv\\Scripts\\python.exe train_policy.py resnet18_finetune_classes
    ..\\.venv\\Scripts\\python.exe train_policy.py resnet18_finetune_classes --folds without_path_04 without_path_05
Results: 04_runs/<date>_<run name>/metrics.json and fold_<name>.json; weights in checkpoints/ (not in git).
Settings: TRAINING_RUN_PARAMETERS and TRAINING_RUNS in config.py.
"""

import argparse
import datetime
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
import timm
import torch

from config import TRAINING_DATA_PARAMETERS, TRAINING_RUN_PARAMETERS, TRAINING_RUNS, TRAINING_TARGET_PARAMETERS
from training_metrics import summarised_folds
from training_probes import ProbeHead, design_outputs, fold_measures, head_losses, predictions_from_outputs, training_arrays
from training_targets import (MIRRORED_DIRECTION, class_weights, episode_folds, inner_validation_mask, leave_one_sheet_out_folds,
                              load_training_frames)



class SmallConvolutionalBackbone(torch.nn.Module):

    def __init__(self, channels: List[int]):
        """
        Initialise a small network trained from scratch: one block per entry of ``channels``.

        Each block is a 3 x 3 convolution with stride 2, batch normalisation and a ReLU, followed by a second 3 x 3
        convolution, so five blocks shrink a 160 x 120 picture to 5 x 4 cells.

        Parameters
        ----------
        channels : list of int
            The width of each block, for example ``[32, 64, 96, 128, 192]``.
        """

        super().__init__()
        layers = []
        width = 3
        for block_width in channels:
            layers += [
                torch.nn.Conv2d(width, block_width, 3, stride = 2, padding = 1, bias = False),
                torch.nn.BatchNorm2d(block_width),
                torch.nn.ReLU(inplace = True),
                torch.nn.Conv2d(block_width, block_width, 3, padding = 1, bias = False),
                torch.nn.BatchNorm2d(block_width),
                torch.nn.ReLU(inplace = True)
            ]
            width = block_width
        self.blocks = torch.nn.Sequential(*layers)
        self.num_features = width


    def forward_features(self, pictures: torch.Tensor) -> torch.Tensor:

        return self.blocks(pictures)



class PictureToMoveModel(torch.nn.Module):

    def __init__(
            self,
            run_settings: Dict,
            output_sizes: Dict[str, int]
    ):
        """
        Initialise a backbone with the shared prediction head on top.

        Parameters
        ----------
        run_settings : dict
            "backbone" ("small_cnn" or a timm model name), "is_pretrained", "small_cnn_channels", "pooling"
            ("average" for one summary, or "grid" for the cells of "pooling_grid", which keeps where things are),
            "hidden_size" and "dropout".
        output_sizes : dict of str to int
            From ``design_outputs``.
        """

        super().__init__()
        if run_settings["backbone"] == "small_cnn":
            self.backbone = SmallConvolutionalBackbone(run_settings["small_cnn_channels"])
        else:
            self.backbone = timm.create_model(run_settings["backbone"], pretrained = run_settings["is_pretrained"], num_classes = 0)
        self.pooling = run_settings["pooling"]
        self.pooling_grid = tuple(run_settings["pooling_grid"])
        cells = self.pooling_grid[0] * self.pooling_grid[1] if self.pooling == "grid" else 1
        self.head = ProbeHead(self.backbone.num_features * cells, output_sizes, run_settings["hidden_size"], run_settings["dropout"])


    def forward(self, pictures: torch.Tensor) -> Dict[str, torch.Tensor]:

        feature_map = self.backbone.forward_features(pictures)
        if self.pooling == "grid":
            pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, self.pooling_grid).flatten(1)
        else:
            pooled = feature_map.mean(dim = (2, 3))
        return self.head(pooled)



def merged_run_settings(run_name: str) -> Dict:
    """Return ``TRAINING_RUN_PARAMETERS`` with the named run's settings merged over it."""

    settings = dict(TRAINING_RUN_PARAMETERS)
    settings.update(TRAINING_RUNS[run_name])
    settings["name"] = run_name
    return settings



def gaussian_kernel(sigma: float) -> torch.Tensor:
    """Return a normalised one-dimensional Gaussian kernel of about three sigmas each way."""

    radius = max(1, int(math.ceil(3.0 * sigma)))
    positions = torch.arange(-radius, radius + 1, dtype = torch.float32)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / kernel.sum()



def augmented_pictures(
        pictures: torch.Tensor,
        settings: Dict,
        randomness_generator: numpy.random.Generator,
        is_training: bool
) -> Tuple[torch.Tensor, numpy.ndarray]:
    """
    Turn a batch of cached pictures into normalised network input, augmented when training.

    Parameters
    ----------
    pictures : torch.Tensor
        (batch, height, width, 3), uint8 RGB.
    settings : dict
        The run settings: "augmentation" (the strength of each kind) and "normalisation".
    randomness_generator : numpy.random.Generator
        Source of every random choice, so a run can be repeated.
    is_training : bool
        Augment only when True.

    Returns
    -------
    tuple of torch.Tensor and numpy.ndarray
        (batch, 3, height, width) float input, and which pictures were mirrored.
    """

    batch = pictures.permute(0, 3, 1, 2).float().div(255.0)
    count = batch.shape[0]
    is_mirrored = numpy.zeros(count, dtype = bool)
    augmentation = settings["augmentation"]
    if is_training:
        brightness = torch.from_numpy(randomness_generator.uniform(1 - augmentation["brightness"], 1 + augmentation["brightness"], count)).float().view(-1, 1, 1, 1)
        contrast = torch.from_numpy(randomness_generator.uniform(1 - augmentation["contrast"], 1 + augmentation["contrast"], count)).float().view(-1, 1, 1, 1)
        saturation = torch.from_numpy(randomness_generator.uniform(1 - augmentation["saturation"], 1 + augmentation["saturation"], count)).float().view(-1, 1, 1, 1)
        gamma = torch.from_numpy(numpy.exp(randomness_generator.uniform(-augmentation["gamma_log"], augmentation["gamma_log"], count))).float().view(-1, 1, 1, 1)
        batch = batch * brightness
        if augmentation["channel_gain"] > 0:
            # a small gain per colour channel imitates the bluish daylight cast and the violet puck in shadow
            gains = randomness_generator.uniform(1 - augmentation["channel_gain"], 1 + augmentation["channel_gain"], (count, 3))
            batch = batch * torch.from_numpy(gains).float().view(count, 3, 1, 1)
        mean_grey = batch.mean(dim = (1, 2, 3), keepdim = True)
        batch = (batch - mean_grey) * contrast + mean_grey
        grey = batch.mean(dim = 1, keepdim = True)
        batch = (batch - grey) * saturation + grey
        batch = batch.clamp(0.0, 1.0) ** gamma
        for index in numpy.flatnonzero(randomness_generator.random(count) < augmentation["blur_probability"]):
            kernel = gaussian_kernel(float(randomness_generator.uniform(0.3, augmentation["blur_sigma_most"])))
            size = len(kernel)
            picture = batch[index:index + 1].reshape(3, 1, *batch.shape[2:])
            picture = torch.nn.functional.conv2d(torch.nn.functional.pad(picture, (size // 2, size // 2, 0, 0), mode = "replicate"), kernel.view(1, 1, 1, -1))
            picture = torch.nn.functional.conv2d(torch.nn.functional.pad(picture, (0, 0, size // 2, size // 2), mode = "replicate"), kernel.view(1, 1, -1, 1))
            batch[index] = picture.reshape(3, *batch.shape[2:])
        if augmentation["noise_most"] > 0:
            noise_level = torch.from_numpy(randomness_generator.uniform(0, augmentation["noise_most"], count)).float().view(-1, 1, 1, 1)
            batch = batch + noise_level * torch.from_numpy(randomness_generator.standard_normal(batch.shape).astype(numpy.float32))
        shift = augmentation["shift_px"]
        if shift > 0:
            padded = torch.nn.functional.pad(batch, (shift, shift, shift, shift), mode = "replicate")
            height, width = batch.shape[2:]
            offsets = randomness_generator.integers(0, 2 * shift + 1, size = (count, 2))
            batch = torch.stack([padded[index, :, offsets[index, 0]:offsets[index, 0] + height, offsets[index, 1]:offsets[index, 1] + width]
                                 for index in range(count)])
        if augmentation["mirror_probability"] > 0:
            is_mirrored = randomness_generator.random(count) < augmentation["mirror_probability"]
            if is_mirrored.any():
                batch[torch.from_numpy(is_mirrored)] = batch[torch.from_numpy(is_mirrored)].flip(-1)
        batch = batch.clamp(0.0, 1.0)
    if settings["input_channels"] == "line_and_puck":
        return line_and_puck_maps(batch), is_mirrored
    mean = torch.tensor(settings["normalisation"]["mean"]).view(1, 3, 1, 1)
    standard_deviation = torch.tensor(settings["normalisation"]["std"]).view(1, 3, 1, 1)
    return (batch - mean) / standard_deviation, is_mirrored



def line_and_puck_maps(batch: torch.Tensor) -> torch.Tensor:
    """
    Turn RGB pictures into three maps that keep only what the push depends on: the dark line, the red puck and grey.

    "dark" is how much darker a pixel is than its picture's median grey (so the same line looks the same in bright
    and dim light); "red" is how much the red channel exceeds the larger of green and blue; "grey" is the grey
    picture with its own median taken off. All three are centred and scaled to about unit spread.

    Parameters
    ----------
    batch : torch.Tensor
        (batch, 3, height, width), RGB in 0 to 1.

    Returns
    -------
    torch.Tensor
        (batch, 3, height, width).
    """

    grey = batch.mean(dim = 1, keepdim = True)
    median = grey.flatten(1).median(dim = 1).values.view(-1, 1, 1, 1)
    dark = torch.relu(median - grey) / (median + 1e-3)
    red = torch.relu(batch[:, 0:1] - torch.maximum(batch[:, 1:2], batch[:, 2:3]))
    centred_grey = (grey - median) / (median + 1e-3)
    return torch.cat([dark * 4.0 - 0.5, red * 6.0 - 0.3, centred_grey * 2.0], dim = 1)



def batch_targets(
        arrays: Dict[str, numpy.ndarray],
        rows: numpy.ndarray,
        is_mirrored: numpy.ndarray,
        mirror_class_index: numpy.ndarray,
        direction_class_weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Return the targets of a batch, mirrored where the picture was mirrored.

    A mirrored picture swaps left and right: the left and right classes swap, the left component of the step, the
    forearm and hand changes and the left components of the chunk change sign, and a twist to the left becomes a twist
    to the right.
    """

    direction = arrays["direction"][rows].copy()
    twist = arrays["twist"][rows].copy()
    exact = arrays["exact"][rows].copy()
    chunk = arrays["chunk"][rows].copy()
    if is_mirrored.any():
        is_labelled = is_mirrored & (direction >= 0)
        direction[is_labelled] = mirror_class_index[direction[is_labelled]]
        twist[is_mirrored] = 2 - twist[is_mirrored]
        exact[is_mirrored, 1] *= -1.0
        exact[is_mirrored, 2] *= -1.0
        exact[is_mirrored, 4] *= -1.0
        chunk[is_mirrored, 1::2] *= -1.0
    return {
        "direction": torch.from_numpy(direction),
        "twist": torch.from_numpy(twist),
        "exact": torch.from_numpy(exact),
        "chunk": torch.from_numpy(chunk),
        "stop": torch.from_numpy(arrays["stop"][rows]),
        "sample_weight": torch.from_numpy(arrays["sample_weight"][rows]),
        "direction_class_weights": direction_class_weights
    }



def model_outputs(
        model: PictureToMoveModel,
        pictures: torch.Tensor,
        rows: numpy.ndarray,
        settings: Dict
) -> Dict[str, torch.Tensor]:
    """Return the model's outputs for the given rows, without augmentation, in batches, on the processor."""

    model.eval()
    device = next(model.parameters()).device
    parts = []
    with torch.inference_mode():
        for start in range(0, len(rows), settings["evaluation_batch_size"]):
            batch_rows = rows[start:start + settings["evaluation_batch_size"]]
            inputs, _ = augmented_pictures(pictures[batch_rows], settings, numpy.random.default_rng(0), is_training = False)
            parts.append({name: value.cpu() for name, value in model(inputs.to(device)).items()})
    return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}



def training_device(settings: Dict) -> torch.device:
    """Return the device to train on: with "device" "auto", the graphics card when PyTorch sees one, else the processor."""

    if settings["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(settings["device"])



def train_fold(
        settings: Dict,
        pictures: torch.Tensor,
        arrays: Dict[str, numpy.ndarray],
        train_rows: numpy.ndarray,
        validation_rows: numpy.ndarray,
        class_names: List[str],
        seed: int
) -> Tuple[PictureToMoveModel, List[Dict]]:
    """
    Train one model on one fold, keeping the weights of the epoch with the lowest validation loss.

    Returns
    -------
    tuple of PictureToMoveModel and list of dict
        The model and the loss per epoch.
    """

    torch.manual_seed(seed)
    randomness_generator = numpy.random.default_rng(seed)
    device = training_device(settings)
    output_sizes = design_outputs(settings["design"], len(class_names), settings["chunk_steps"], settings.get("is_stop_separate", False))
    model = PictureToMoveModel(settings, output_sizes).to(device)
    backbone_parameters = list(model.backbone.parameters())
    head_parameters = list(model.head.parameters())
    backbone_rate = settings["learning_rate"] * (settings["backbone_learning_rate_factor"] if settings["is_pretrained"] else 1.0)
    optimiser = torch.optim.AdamW([{"params": backbone_parameters, "lr": backbone_rate}, {"params": head_parameters, "lr": settings["learning_rate"]}],
                                  weight_decay = settings["weight_decay"])
    rows_per_epoch = max(settings["batch_size"], int(round(settings["train_share_per_epoch"] * len(train_rows))))
    steps_per_epoch = math.ceil(rows_per_epoch / settings["batch_size"])
    total_steps = steps_per_epoch * settings["epochs"]
    warm_up_steps = steps_per_epoch * settings["warm_up_epochs"]
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / max(1, warm_up_steps))
                                                 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(1, total_steps)))))
    class_index = {name: index for index, name in enumerate(class_names)}
    mirror_class_index = numpy.array([class_index[MIRRORED_DIRECTION[name]] for name in class_names])
    direction_class_weights = torch.from_numpy(class_weights(arrays["direction"][train_rows], len(class_names), settings["class_weight_power"]))
    head_settings = dict(settings)

    history = []
    best_loss = math.inf
    best_state = None
    epochs_without_gain = 0
    for epoch in range(settings["epochs"]):
        model.train()
        started = time.perf_counter()
        # neighbouring pictures are about 1 cm apart and look alike, so an epoch may use a random share of them
        order = randomness_generator.permutation(train_rows)[:rows_per_epoch]
        epoch_loss = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            inputs, is_mirrored = augmented_pictures(pictures[rows], settings, randomness_generator, is_training = True)
            targets = {name: value.to(device) for name, value in batch_targets(arrays, rows, is_mirrored, mirror_class_index, direction_class_weights).items()}
            loss = head_losses(model(inputs.to(device)), targets, head_settings)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            epoch_loss += float(loss) * len(rows)
        # the exam model ("split" "all") keeps no check pictures, so it has no validation loss and keeps its last weights
        if len(validation_rows) == 0:
            history.append({"epoch": epoch + 1, "train_loss": round(epoch_loss / len(order), 4), "validation_loss": None,
                            "seconds": round(time.perf_counter() - started, 1)})
            print(f"{settings['name']} | epoch {epoch + 1} | train {history[-1]['train_loss']} | {history[-1]['seconds']} s", flush = True)
            continue
        outputs = model_outputs(model, pictures, validation_rows, settings)
        validation_targets = batch_targets(arrays, validation_rows, numpy.zeros(len(validation_rows), dtype = bool), mirror_class_index, direction_class_weights)
        validation_loss = float(head_losses(outputs, validation_targets, head_settings))
        history.append({"epoch": epoch + 1, "train_loss": round(epoch_loss / len(order), 4), "validation_loss": round(validation_loss, 4),
                        "seconds": round(time.perf_counter() - started, 1)})
        print(f"{settings['name']} | epoch {epoch + 1} | train {history[-1]['train_loss']} | validation {history[-1]['validation_loss']} | {history[-1]['seconds']} s", flush = True)
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= settings["patience"]:
                break
    # validation loss is a poor guide to robot success (robomimic, Mandlekar and others 2021), so a fixed number of
    # epochs with the last weights is the default for the final model; the best-validation weights stay an option
    if not settings.get("is_last_checkpoint_kept") and best_state is not None:
        model.load_state_dict(best_state)
    return model, history



def milliseconds_per_picture(
        model: PictureToMoveModel,
        settings: Dict
) -> float:
    """Return the median time of one picture through the model on this laptop, as the robot loop would run it."""

    model.eval()
    height, width = settings["picture_size"][1], settings["picture_size"][0]
    picture = torch.rand(1, 3, height, width)
    timings = []
    with torch.inference_mode():
        for repeat in range(23):
            started = time.perf_counter()
            model(picture)
            if repeat >= 3:
                timings.append(1000.0 * (time.perf_counter() - started))
    return round(float(numpy.median(timings)), 1)



def run_training(
        run_name: str,
        fold_names: Optional[List[str]] = None,
        overrides: Optional[Dict] = None
) -> Dict:
    """
    Train and score one named run on every fold of its split (or on the named folds), and save the results.

    Parameters
    ----------
    run_name : str
        A key of ``TRAINING_RUNS``.
    fold_names : list of str, optional
        Only these folds, for example ``["without_path_04"]``.
    overrides : dict, optional
        Settings changed for this call only, for a quick check (for example "epochs" 1 and "train_row_limit" 512);
        the results then go to a folder whose name ends in "_check".

    Returns
    -------
    dict
        The settings, the measures per fold and their summary.
    """

    settings = merged_run_settings(run_name)
    settings.update(overrides or {})
    settings["device"] = str(training_device(settings))
    torch.set_num_threads(settings["threads"])
    # the pictures always have one size, so the graphics card may pick its fastest convolution once (not bit-for-bit
    # repeatable, unlike the processor runs)
    torch.backends.cudnn.benchmark = settings["device"] == "cuda"
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    settings["chunk_steps"] = target_parameters["chunk_steps"]
    frames = load_training_frames(target_parameters)
    cache = numpy.load(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / f"images_{settings['picture_cache']}.npy", mmap_mode = "r")
    pictures = torch.from_numpy(numpy.array(cache[frames["frame_index"].to_numpy()]))
    if settings["backbone"] != "small_cnn":
        data_configuration = timm.data.resolve_data_config({}, model = timm.create_model(settings["backbone"], pretrained = False, num_classes = 0))
        settings["normalisation"] = {"mean": list(data_configuration["mean"]), "std": list(data_configuration["std"])}

    arrays, class_names, is_moving = training_arrays(frames, settings, target_parameters)
    is_exam_model = settings["split"] == "all"
    if is_exam_model:
        # the exam model learns from every sheet and every episode; its test is the robot, not a held-out sheet
        folds = [{"name": "all_sheets", "train": numpy.ones(len(frames), dtype = bool), "test": numpy.zeros(len(frames), dtype = bool)}]
    elif settings["split"] == "sheets":
        folds = leave_one_sheet_out_folds(frames)
    else:
        folds = episode_folds(frames, target_parameters["episode_fold_count"], target_parameters["seed"])
    if fold_names:
        folds = [fold for fold in folds if fold["name"] in fold_names]

    run_date = settings["date"] or datetime.date.today().isoformat()
    run_folder = Path(settings["output_root"]) / f"{run_date}_{run_name}{'_check' if overrides else ''}"
    (run_folder / "checkpoints").mkdir(parents = True, exist_ok = True)
    fold_results = []
    for fold_number, fold in enumerate(folds):
        is_validation = (numpy.zeros(len(frames), dtype = bool) if is_exam_model
                         else inner_validation_mask(frames, fold["train"], target_parameters["inner_validation_share"], target_parameters["seed"] + fold_number))
        train_rows = numpy.flatnonzero(fold["train"] & ~is_validation)
        validation_rows = numpy.flatnonzero(is_validation)
        test_rows = numpy.flatnonzero(fold["test"])
        if settings.get("train_row_limit"):
            limit_generator = numpy.random.default_rng(settings["seed"])
            train_rows = numpy.sort(limit_generator.choice(train_rows, size = min(len(train_rows), settings["train_row_limit"]), replace = False))
            validation_rows = validation_rows[:settings["train_row_limit"] // 4]
            test_rows = test_rows[:settings["train_row_limit"]]
        started = time.perf_counter()
        model, history = train_fold(settings, pictures, arrays, train_rows, validation_rows, class_names, settings["seed"] + fold_number)
        if is_exam_model:
            measures = {"fold": fold["name"], "train_rows": len(train_rows), "epochs_run": len(history), "minutes": round((time.perf_counter() - started) / 60.0, 1)}
            fold_results.append(measures)
            (run_folder / f"fold_{fold['name']}.json").write_text(json.dumps({"measures": measures, "history": history}, indent = 1), encoding = "utf-8")
            model = model.to("cpu")
            torch.save(model.state_dict(), run_folder / "checkpoints" / f"{fold['name']}.pt")
            print(f"{run_name} | {fold['name']} | trained on {len(train_rows)} pictures in {measures['minutes']} min", flush = True)
            continue
        predictions = predictions_from_outputs(model_outputs(model, pictures, test_rows, settings), class_names, settings["stop_target"])
        validation_predictions = predictions_from_outputs(model_outputs(model, pictures, validation_rows, settings), class_names, settings["stop_target"])
        measures = fold_measures(predictions, frames, test_rows, arrays, is_moving, validation_predictions, validation_rows)
        measures.update({"fold": fold["name"], "epochs_run": len(history), "minutes": round((time.perf_counter() - started) / 60.0, 1)})
        fold_results.append(measures)
        (run_folder / f"fold_{fold['name']}.json").write_text(json.dumps({"measures": measures, "history": history}, indent = 1), encoding = "utf-8")
        numpy.savez_compressed(run_folder / f"predictions_{fold['name']}.npz", frame_index = frames["frame_index"].to_numpy()[test_rows], **predictions)
        # saved and timed on the processor, as the robot loop on the laptop runs it
        model = model.to("cpu")
        torch.save(model.state_dict(), run_folder / "checkpoints" / f"{fold['name']}.pt")
        print(f"{run_name} | {fold['name']} | {json.dumps({key: measures.get(key) for key in ('within_45_deg', 'class_accuracy', 'episodes_stopped_right', 'episodes_stopped_early', 'twist_sign_accuracy')})}", flush = True)

    result = {
        "name": run_name,
        "settings": {key: (str(value) if isinstance(value, Path) else value) for key, value in settings.items()},
        "class_names": class_names,
        "milliseconds_per_picture": milliseconds_per_picture(model, settings),
        "folds": fold_results,
        "summary": {} if is_exam_model else summarised_folds(fold_results)
    }
    (run_folder / "metrics.json").write_text(json.dumps(result, indent = 1), encoding = "utf-8")
    return result



def main() -> None:

    parser = argparse.ArgumentParser(description = "Train one named run of TRAINING_RUNS and score it fold by fold.")
    parser.add_argument("run_name", choices = sorted(TRAINING_RUNS))
    parser.add_argument("--folds", nargs = "*", default = None, help = "only these folds, for example without_path_04")
    parser.add_argument("--epochs", type = int, default = None, help = "quick check: this many epochs")
    parser.add_argument("--train-row-limit", type = int, default = None, help = "quick check: train on this many pictures")
    arguments = parser.parse_args()
    overrides = {key: value for key, value in (("epochs", arguments.epochs), ("train_row_limit", arguments.train_row_limit)) if value is not None}
    result = run_training(arguments.run_name, arguments.folds, overrides)
    print(json.dumps(result["summary"], indent = 1))



if __name__ == "__main__":
    main()
