"""
Quick model comparisons on frozen picture features (9.11.26, Model and training session).

A small head (one hidden layer, or none) is trained on the features of a frozen backbone (training_features.py) and
scored with the shared offline measures (training_metrics.py). Training a head takes seconds, so many questions can be
settled offline before any long training run: which backbone, which target design, whether a short picture history
helps, how to treat small motions, scores and class weights. Every variant is trained and tested on the same folds:
    sheets     leave one sheet out (train on five sheets, test on the sixth), the stand-in for an unseen track;
    episodes   whole episodes held out, five folds spread over all six sheets.
Target designs:
    classes    the move name (8 directions or stop) and the forearm twist (right, none, left) as classes;
    exact      the exact step (tip forward and left in cm, forearm, lean and hand changes) as numbers, plus stop;
    mixed      both heads at once; the direction is scored from the class and from the numbers;
    chunk      the tip's place after each of the next steps, plus the forearm change and stop.

Run from 02_code, after training_dataset.py and training_features.py:
    ..\\.venv\\Scripts\\python.exe training_probes.py backbones
    ..\\.venv\\Scripts\\python.exe training_probes.py all
Results: 04_runs/2026-11-09_frozen_feature_probes/<experiment>.json and results.csv (one row per variant).
Settings: TRAINING_PROBE_PARAMETERS and TRAINING_PROBE_EXPERIMENTS in config.py.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
import pandas
import torch

from config import (TRAINING_DATA_PARAMETERS, TRAINING_METRIC_PARAMETERS, TRAINING_PROBE_EXPERIMENTS, TRAINING_PROBE_PARAMETERS,
                    TRAINING_TARGET_PARAMETERS)
from training_metrics import (change_metrics, chunk_metrics, direction_metrics, stop_metrics, summarised_folds, tuned_stop_threshold,
                              twist_metrics)
from training_targets import (DIRECTION_ANGLES_DEG, DIRECTION_NAMES, chunk_targets, class_weights, direction_targets,
                              episode_folds, exact_step_targets, inner_validation_mask, leave_one_sheet_out_folds,
                              load_training_frames, previous_rows, score_weights, stop_targets, twist_targets)

TWIST_SIGNS = numpy.array([-1.0, 0.0, 1.0])



class ProbeHead(torch.nn.Module):

    def __init__(
            self,
            input_size: int,
            output_sizes: Dict[str, int],
            hidden_size: int,
            dropout: float,
            input_mean: Optional[numpy.ndarray] = None,
            input_standard_deviation: Optional[numpy.ndarray] = None
    ):
        """
        Initialise a head with one output layer per target, on a shared hidden layer (or straight on the features).

        Parameters
        ----------
        input_size : int
            Length of the feature vector.
        output_sizes : dict of str to int
            For example ``{"direction": 9, "twist": 3}``.
        hidden_size : int
            Width of the shared hidden layer; 0 for a linear head.
        dropout : float
            Dropout on the features and on the hidden layer.
        input_mean, input_standard_deviation : numpy.ndarray, optional
            The training rows' feature mean and spread; when given, the head standardises its own input, so the raw
            features can stay in half precision and a saved head carries its own standardisation.
        """

        super().__init__()
        mean = numpy.zeros(input_size) if input_mean is None else input_mean
        spread = numpy.ones(input_size) if input_standard_deviation is None else input_standard_deviation
        self.register_buffer("input_mean", torch.as_tensor(mean, dtype = torch.float32))
        self.register_buffer("input_standard_deviation", torch.as_tensor(spread, dtype = torch.float32))
        layers = [torch.nn.Dropout(dropout)]
        width = input_size
        if hidden_size > 0:
            layers += [torch.nn.Linear(input_size, hidden_size), torch.nn.GELU(), torch.nn.Dropout(dropout)]
            width = hidden_size
        self.shared = torch.nn.Sequential(*layers)
        self.outputs = torch.nn.ModuleDict({name: torch.nn.Linear(width, size) for name, size in output_sizes.items()})


    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:

        features = (features.float() - self.input_mean) / self.input_standard_deviation
        shared = self.shared(features)
        return {name: layer(shared) for name, layer in self.outputs.items()}



def merged_probe_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``TRAINING_PROBE_PARAMETERS`` with any overrides merged over it."""

    merged = dict(TRAINING_PROBE_PARAMETERS)
    merged.update(parameters or {})
    return merged



def load_features(
        frames: pandas.DataFrame,
        backbone: str,
        feature_kind: str,
        history_lags: List[int],
        is_difference_history: bool
) -> numpy.ndarray:
    """
    Return the feature vector of every row, with the features of earlier pictures of the same episode appended.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_training_frames``; "frame_index" points into the feature files.
    backbone : str
        A backbone name of ``TRAINING_FEATURE_PARAMETERS``.
    feature_kind : str
        "global" or "grid".
    history_lags : list of int
        Earlier pictures to append, in steps back (empty for one picture).
    is_difference_history : bool
        Append the change since the earlier picture instead of the earlier picture itself.

    Returns
    -------
    numpy.ndarray
        float16 (rows, features), as stored.
    """

    feature_path = Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "features" / f"{backbone}_{feature_kind}.npy"
    all_features = numpy.load(feature_path, mmap_mode = "r")
    current = numpy.asarray(all_features[frames["frame_index"].to_numpy()], dtype = numpy.float16)
    parts = [current]
    for lag in history_lags:
        earlier = current[previous_rows(frames, lag)]
        if is_difference_history:
            parts.append((current.astype(numpy.float32) - earlier.astype(numpy.float32)).astype(numpy.float16))
        else:
            parts.append(earlier)
    return numpy.concatenate(parts, axis = 1)



def design_outputs(
        design: str,
        class_count: int,
        chunk_steps: int,
        is_stop_separate: bool = False
) -> Dict[str, int]:
    """
    Return the output layers a target design needs.

    With ``is_stop_separate`` a class design gets its own stop output instead of deciding stop as one more class.
    """

    if design == "classes":
        outputs = {"direction": class_count, "twist": 3}
    elif design == "exact":
        outputs = {"exact": 5, "stop": 1}
    elif design == "mixed":
        outputs = {"direction": class_count, "exact": 5}
    elif design == "chunk":
        outputs = {"chunk": 2 * chunk_steps, "exact": 5, "stop": 1}
    else:
        raise ValueError(f"unknown design {design}")
    if is_stop_separate:
        outputs["stop"] = 1
    return outputs



def training_arrays(
        frames: pandas.DataFrame,
        parameters: Dict,
        target_parameters: Dict
) -> Tuple[Dict[str, numpy.ndarray], List[str], numpy.ndarray]:
    """
    Return every target and the sample weights of all rows, the direction class names and the moving rows.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_training_frames``.
    parameters : dict
        Supplies "score_weights", "is_stop_separate" and "stop_target" ("stop_row": only the episode's stop
        picture is a stop; "at_goal": every picture whose tip is within the at-goal distance of the goal).
    target_parameters : dict
        Supplies "small_motion_rule", "twist_threshold_deg" and "chunk_steps".

    Returns
    -------
    tuple of dict, list of str and numpy.ndarray
        The arrays by name, the class names, and whether each row's tip really moved (the rows the direction is
        scored on).
    """

    labels, class_names, is_label_used = direction_targets(frames, target_parameters["small_motion_rule"])
    labels = numpy.where(is_label_used, labels, -1)
    is_stop, is_at_goal = stop_targets(frames)
    if parameters.get("is_stop_separate"):
        labels = numpy.where(frames["action"].to_numpy() == "stop", -1, labels)
    if parameters.get("stop_target") == "goal_distance":
        # every picture knows how far its tip is from where the demonstration stopped, so the stop output can learn
        # from all pictures instead of one per episode; far distances are capped, they need not be told apart
        stop_target = numpy.minimum(frames["goal_distance_cm"].fillna(parameters["goal_distance_cap_cm"]).to_numpy(),
                                    parameters["goal_distance_cap_cm"]).astype(numpy.float32)
    elif parameters.get("stop_target") == "at_goal":
        stop_target = is_at_goal
    else:
        stop_target = is_stop
    arrays = {
        "direction": labels,
        "twist": twist_targets(frames, target_parameters["twist_threshold_deg"]),
        "exact": exact_step_targets(frames),
        "chunk": chunk_targets(frames, target_parameters["chunk_steps"]).reshape(len(frames), -1),
        "stop": stop_target,
        "is_stop_row": is_stop,
        "sample_weight": score_weights(frames, {int(score): weight for score, weight in parameters["score_weights"].items()})
    }
    is_moving = frames["action"].isin(DIRECTION_NAMES).to_numpy()
    return arrays, class_names, is_moving



def head_losses(
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        parameters: Dict
) -> torch.Tensor:
    """
    Return the summed loss of all outputs of a head on one batch.

    Classes use cross-entropy (with the class weights), numbers a Huber loss on scaled values (cm for the tip,
    "angle_scale_deg" degrees for the turns), stop a weighted binary cross-entropy. Each row's loss is multiplied by
    its sample weight.
    """

    total = torch.zeros(())
    weight = batch["sample_weight"]
    if "direction" in outputs:
        is_used = batch["direction"] >= 0
        if is_used.any():
            loss = torch.nn.functional.cross_entropy(outputs["direction"][is_used], batch["direction"][is_used],
                                                     weight = batch["direction_class_weights"], reduction = "none",
                                                     label_smoothing = parameters["label_smoothing"])
            total = total + (loss * weight[is_used]).mean()
    if "twist" in outputs:
        loss = torch.nn.functional.cross_entropy(outputs["twist"], batch["twist"], reduction = "none")
        total = total + parameters["twist_loss_weight"] * (loss * weight).mean()
    if "exact" in outputs:
        scale = torch.tensor([1.0, 1.0] + [parameters["angle_scale_deg"]] * 3)
        loss = torch.nn.functional.huber_loss(outputs["exact"] / scale, batch["exact"] / scale, reduction = "none",
                                              delta = parameters["huber_delta"]).sum(dim = 1)
        total = total + (loss * weight).mean()
    if "chunk" in outputs:
        loss = torch.nn.functional.huber_loss(outputs["chunk"], batch["chunk"], reduction = "none",
                                              delta = parameters["huber_delta"]).mean(dim = 1)
        total = total + (loss * weight).mean()
    if "stop" in outputs:
        if parameters.get("stop_target") == "goal_distance":
            loss = torch.nn.functional.huber_loss(outputs["stop"][:, 0], batch["stop"], reduction = "none", delta = parameters["huber_delta"])
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["stop"][:, 0], batch["stop"], reduction = "none",
                                                                        pos_weight = torch.tensor(parameters["stop_positive_weight"]))
        total = total + (loss * weight).mean()
    return total



def batch_dictionary(
        arrays: Dict[str, numpy.ndarray],
        rows: numpy.ndarray,
        direction_class_weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Return the tensors of the given rows, for one training or validation batch."""

    batch = {name: torch.from_numpy(values[rows]) for name, values in arrays.items()}
    batch["direction_class_weights"] = direction_class_weights
    return batch



def train_head(
        features: numpy.ndarray,
        arrays: Dict[str, numpy.ndarray],
        train_rows: numpy.ndarray,
        validation_rows: numpy.ndarray,
        output_sizes: Dict[str, int],
        direction_class_weights: numpy.ndarray,
        parameters: Dict,
        seed: int
) -> ProbeHead:
    """
    Train one head with early stopping on the validation rows.

    Parameters
    ----------
    features : numpy.ndarray
        Raw features of all rows (the head standardises them with the training rows' mean and spread).
    arrays : dict of str to numpy.ndarray
        Targets and sample weights of all rows.
    train_rows, validation_rows : numpy.ndarray
        Row positions.
    output_sizes : dict of str to int
        From ``design_outputs``.
    direction_class_weights : numpy.ndarray
        One weight per direction class.
    parameters : dict
        Head and training settings.
    seed : int
        Seed for the weights and the batch order.

    Returns
    -------
    ProbeHead
        The head with the weights of its best validation epoch.
    """

    torch.manual_seed(seed)
    randomness_generator = numpy.random.default_rng(seed)
    sample_rows = train_rows[::max(1, len(train_rows) // parameters["standardisation_rows"])]
    sample = features[sample_rows].astype(numpy.float64)
    head = ProbeHead(features.shape[1], output_sizes, parameters["hidden_size"], parameters["dropout"],
                     sample.mean(axis = 0), sample.std(axis = 0) + 1e-3)
    optimiser = torch.optim.AdamW(head.parameters(), lr = parameters["learning_rate"], weight_decay = parameters["weight_decay"])
    feature_tensor = torch.from_numpy(features)
    class_weight_tensor = torch.from_numpy(direction_class_weights)
    best_loss = math.inf
    best_state = None
    epochs_without_gain = 0
    for epoch in range(parameters["max_epochs"]):
        head.train()
        order = randomness_generator.permutation(train_rows)
        for start in range(0, len(order), parameters["batch_size"]):
            rows = order[start:start + parameters["batch_size"]]
            outputs = head(feature_tensor[rows])
            loss = head_losses(outputs, batch_dictionary(arrays, rows, class_weight_tensor), parameters)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        head.eval()
        with torch.no_grad():
            outputs = head(feature_tensor[validation_rows])
            validation_loss = float(head_losses(outputs, batch_dictionary(arrays, validation_rows, class_weight_tensor), parameters))
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {name: value.clone() for name, value in head.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= parameters["patience"]:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head



def predictions_of(
        head: ProbeHead,
        features: numpy.ndarray,
        rows: numpy.ndarray,
        design: str,
        class_names: List[str],
        stop_target: Optional[str] = None
) -> Dict[str, numpy.ndarray]:
    """
    Turn a head's outputs on the given rows into the quantities the measures need.

    Returns
    -------
    dict of str to numpy.ndarray
        As ``predictions_from_outputs``.
    """

    parts = []
    with torch.no_grad():
        for start in range(0, len(rows), 2048):
            parts.append(head(torch.from_numpy(features[rows[start:start + 2048]])))
    outputs = {name: torch.cat([part[name] for part in parts]) for name in parts[0]}
    return predictions_from_outputs(outputs, class_names, TRAINING_PROBE_PARAMETERS["stop_target"] if stop_target is None else stop_target)



def predictions_from_outputs(
        outputs: Dict[str, torch.Tensor],
        class_names: List[str],
        stop_target: str = "stop_row"
) -> Dict[str, numpy.ndarray]:
    """
    Turn a head's raw outputs into the quantities the measures need.

    Parameters
    ----------
    outputs : dict of str to torch.Tensor
        The head's output layers for a set of rows.
    class_names : list of str
        The direction classes, in output order.

    Returns
    -------
    dict of str to numpy.ndarray
        "angle_deg" (the class's nominal angle, or the predicted vector's angle), "angle_expected_deg" (the
        probability-weighted mean direction of a class head), "angle_vector_deg" (the vector's angle when a design
        has both), "stop_probability", "twist_deg" and "chunk_cm" where the design predicts them.
    """

    outputs = {name: value.detach() for name, value in outputs.items()}
    rows = numpy.arange(len(next(iter(outputs.values()))))
    result = {}
    nominal = numpy.array([DIRECTION_ANGLES_DEG.get(name, 0.0) for name in class_names])
    if "direction" in outputs:
        probabilities = torch.softmax(outputs["direction"], dim = 1).numpy()
        moving_columns = [index for index, name in enumerate(class_names) if name in DIRECTION_ANGLES_DEG]
        best_moving = numpy.array(moving_columns)[numpy.argmax(probabilities[:, moving_columns], axis = 1)]
        result["angle_deg"] = nominal[best_moving]
        weights = probabilities[:, moving_columns]
        radians = numpy.radians(nominal[moving_columns])
        result["angle_expected_deg"] = numpy.degrees(numpy.arctan2(weights @ numpy.sin(radians), weights @ numpy.cos(radians)))
        result["stop_probability"] = probabilities[:, class_names.index("stop")]
    if "exact" in outputs:
        exact = outputs["exact"].numpy()
        key = "angle_vector_deg" if "direction" in outputs else "angle_deg"
        result[key] = numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0]))
        result["twist_deg"] = exact[:, 2]
        result["lean_deg"] = exact[:, 3]
        result["hand_deg"] = exact[:, 4]
    if "twist" in outputs:
        result["twist_deg"] = TWIST_SIGNS[numpy.argmax(outputs["twist"].numpy(), axis = 1)] * 5.0
    if "stop" in outputs:
        if stop_target == "goal_distance":
            # a predicted distance d to the goal becomes exp(-d / 2 cm): 1 at the goal, 0.37 at 2 cm, 0.08 at 5 cm,
            # so the same tuned thresholds can be searched as for a stop probability
            result["stop_probability"] = numpy.exp(-numpy.clip(outputs["stop"][:, 0].numpy(), 0.0, None) / 2.0)
            result["goal_distance_cm"] = outputs["stop"][:, 0].numpy()
        else:
            result["stop_probability"] = torch.sigmoid(outputs["stop"][:, 0]).numpy()
    if "chunk" in outputs:
        chunk = outputs["chunk"].numpy().reshape(len(rows), -1, 2)
        result["chunk_cm"] = chunk
        result["angle_deg"] = numpy.degrees(numpy.arctan2(chunk[:, 0, 1], chunk[:, 0, 0]))
        # the predicted place a few steps ahead gives the path direction, less noisy than the first step
        ahead = min(TRAINING_METRIC_PARAMETERS["path_steps"], chunk.shape[1]) - 1
        result["angle_vector_deg"] = numpy.degrees(numpy.arctan2(chunk[:, ahead, 1], chunk[:, ahead, 0]))
    return result



def path_directions(
        frames: pandas.DataFrame,
        rows: numpy.ndarray,
        arrays: Dict[str, numpy.ndarray]
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return the direction the path takes from each row, and which rows it is scored on.

    The hand-guided steps wobble by about 21 degrees from one step to the next (label audit, 9.11.26), so a single
    step is a noisy target for a line follower. The path direction is the direction from the tip now to where it is
    "path_steps" steps later, in the current picture's frame; it is scored on rows other than the stop row whose tip
    moves at least "path_minimum_cm" over those steps.

    Returns
    -------
    tuple of numpy.ndarray
        The path angle in degrees and the boolean mask of scored rows, both for ``rows``.
    """

    steps = TRAINING_METRIC_PARAMETERS["path_steps"]
    chunk = arrays["chunk"][rows].reshape(len(rows), -1, 2)
    vector = chunk[:, min(steps, chunk.shape[1]) - 1, :]
    path_angle = numpy.degrees(numpy.arctan2(vector[:, 1], vector[:, 0]))
    is_path_row = (frames["action"].to_numpy()[rows] != "stop") & (numpy.linalg.norm(vector, axis = 1) >= TRAINING_METRIC_PARAMETERS["path_minimum_cm"])
    return path_angle, is_path_row



def fold_measures(
        predictions: Dict[str, numpy.ndarray],
        frames: pandas.DataFrame,
        rows: numpy.ndarray,
        arrays: Dict[str, numpy.ndarray],
        is_moving: numpy.ndarray,
        validation_predictions: Optional[Dict[str, numpy.ndarray]] = None,
        validation_rows: Optional[numpy.ndarray] = None
) -> Dict:
    """
    Return the shared offline measures of one fold's test rows.

    With the validation rows' predictions, the stop is also scored with a threshold tuned on the validation episodes
    (never on the test sheet), once for one picture and once for two pictures in a row ("tuned_" and "tuned_two_").
    """

    true_angle = frames["move_angle_deg"].to_numpy()[rows]
    path_angle, is_path_row = path_directions(frames, rows, arrays)
    measures = {"test_rows": int(len(rows)), "test_episodes": int(frames["episode_id"].iloc[rows].nunique())}
    for key, prefix in (("angle_deg", ""), ("angle_expected_deg", "expected_"), ("angle_vector_deg", "vector_")):
        if key in predictions:
            measures.update({f"{prefix}{name}": value for name, value in
                             direction_metrics(true_angle, predictions[key], is_moving[rows]).items()})
            measures.update({f"path_{prefix}{name}": value for name, value in
                             direction_metrics(path_angle, predictions[key], is_path_row).items()})
    if "stop_probability" in predictions:
        goal_distance = frames["goal_distance_cm"].fillna(0.0).to_numpy()
        episode_ids = frames["episode_id"].to_numpy()
        measures.update(stop_metrics(arrays["is_stop_row"][rows], predictions["stop_probability"], goal_distance[rows], episode_ids[rows]))
        # the rule the design review chose for every model: a fixed threshold, two pictures in a row, never tuned
        fixed_two = stop_metrics(arrays["is_stop_row"][rows], predictions["stop_probability"], goal_distance[rows], episode_ids[rows],
                                 consecutive_pictures = 2)
        measures.update({f"fixed_two_{name}": value for name, value in fixed_two.items() if name.startswith("episodes_")})
        if validation_predictions is not None and "stop_probability" in validation_predictions:
            for consecutive, prefix in ((1, "tuned_"), (2, "tuned_two_")):
                threshold = tuned_stop_threshold(arrays["is_stop_row"][validation_rows], validation_predictions["stop_probability"],
                                                 goal_distance[validation_rows], episode_ids[validation_rows], consecutive)
                tuned = stop_metrics(arrays["is_stop_row"][rows], predictions["stop_probability"], goal_distance[rows], episode_ids[rows],
                                     threshold = threshold, consecutive_pictures = consecutive)
                measures.update({f"{prefix}{name}": value for name, value in tuned.items() if name.startswith("episodes_")})
                measures[f"{prefix}stop_threshold"] = threshold
    if "twist_deg" in predictions:
        measures.update(twist_metrics(arrays["exact"][rows, 2], predictions["twist_deg"]))
    for column, name in ((3, "lean"), (4, "hand")):
        if f"{name}_deg" in predictions:
            measures.update(change_metrics(arrays["exact"][rows, column], predictions[f"{name}_deg"], name))
    if "chunk_cm" in predictions:
        measures.update(chunk_metrics(arrays["chunk"][rows].reshape(len(rows), -1, 2), predictions["chunk_cm"]))
    return measures



def subsampled_training_rows(
        frames: pandas.DataFrame,
        train_rows: numpy.ndarray,
        parameters: Dict,
        seed: int
) -> numpy.ndarray:
    """
    Keep only part of a fold's training rows, to see how the result grows with more data.

    "train_sheet_count" keeps that many of the training sheets (chosen by ``seed``); "train_episode_share" then keeps
    that share of each remaining sheet's episodes. Whole episodes are always kept or dropped together.

    Returns
    -------
    numpy.ndarray
        The kept row positions.
    """

    randomness_generator = numpy.random.default_rng(seed)
    training = frames.iloc[train_rows]
    sheets = sorted(training["sheet"].unique())
    sheet_count = parameters.get("train_sheet_count")
    if sheet_count:
        sheets = sorted(randomness_generator.permutation(sheets)[:sheet_count])
    kept_episodes = []
    for sheet in sheets:
        episodes = sorted(training.loc[training["sheet"] == sheet, "episode_id"].unique())
        randomness_generator.shuffle(episodes)
        kept_count = max(1, int(round(parameters.get("train_episode_share", 1.0) * len(episodes))))
        kept_episodes.extend(episodes[:kept_count])
    is_kept = training["episode_id"].isin(kept_episodes).to_numpy()
    return train_rows[is_kept]



def save_predictions(
        predictions: Dict[str, numpy.ndarray],
        frames: pandas.DataFrame,
        rows: numpy.ndarray,
        path: Path
) -> None:
    """Save a fold's predictions with the frame indices of their rows, for later error analysis."""

    path.parent.mkdir(parents = True, exist_ok = True)
    numpy.savez_compressed(path, frame_index = frames["frame_index"].to_numpy()[rows], **predictions)



def run_variant(
        variant: Dict,
        frames_cache: Dict,
        features_cache: Dict
) -> Dict:
    """
    Train and test one variant on every fold of its split.

    Parameters
    ----------
    variant : dict
        Overrides for ``TRAINING_PROBE_PARAMETERS`` and ``TRAINING_TARGET_PARAMETERS``; "name" names the variant.
    frames_cache, features_cache : dict
        Loaded tables and features, reused between variants.

    Returns
    -------
    dict
        The variant's settings, its measures per fold and their summary.
    """

    parameters = merged_probe_parameters(variant)
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in variant.items() if key in TRAINING_TARGET_PARAMETERS})
    frame_key = json.dumps({key: target_parameters[key] for key in ("kept_scores", "is_unscored_kept", "is_mistake_dropped", "is_flagged_dropped", "is_static_dropped", "is_audit_cleaning")}, sort_keys = True)
    if frame_key not in frames_cache:
        frames_cache[frame_key] = load_training_frames(target_parameters)
    frames = frames_cache[frame_key]
    feature_key = (frame_key, parameters["backbone"], parameters["feature_kind"], tuple(parameters["history_lags"]), parameters["is_difference_history"])
    if feature_key not in features_cache:
        features_cache.clear()
        features_cache[feature_key] = load_features(frames, parameters["backbone"], parameters["feature_kind"],
                                                    parameters["history_lags"], parameters["is_difference_history"])
    raw_features = features_cache[feature_key]

    arrays, class_names, is_moving = training_arrays(frames, parameters, target_parameters)
    labels = arrays["direction"]
    output_sizes = design_outputs(parameters["design"], len(class_names), target_parameters["chunk_steps"], parameters.get("is_stop_separate", False))
    folds = (leave_one_sheet_out_folds(frames) if parameters["split"] == "sheets"
             else episode_folds(frames, target_parameters["episode_fold_count"], target_parameters["seed"]))

    fold_results = []
    started = time.perf_counter()
    for fold_number, fold in enumerate(folds):
        is_validation = inner_validation_mask(frames, fold["train"], target_parameters["inner_validation_share"], target_parameters["seed"] + fold_number)
        train_rows = numpy.flatnonzero(fold["train"] & ~is_validation)
        if parameters.get("train_sheet_count") or parameters.get("train_episode_share", 1.0) < 1.0:
            train_rows = subsampled_training_rows(frames, train_rows, parameters, parameters["seed"] + fold_number)
        validation_rows = numpy.flatnonzero(is_validation)
        test_rows = numpy.flatnonzero(fold["test"])
        weights = class_weights(labels[train_rows], len(class_names), parameters["class_weight_power"])
        head = train_head(raw_features, arrays, train_rows, validation_rows, output_sizes, weights, parameters,
                          parameters["seed"] + fold_number)
        predictions = predictions_of(head, raw_features, test_rows, parameters["design"], class_names, parameters["stop_target"])
        validation_predictions = predictions_of(head, raw_features, validation_rows, parameters["design"], class_names, parameters["stop_target"])
        if parameters.get("is_prediction_saved"):
            save_predictions(predictions, frames, test_rows, Path(parameters["output_folder"]) / "predictions" / f"{variant['name']}_{fold['name']}.npz")
        measures = fold_measures(predictions, frames, test_rows, arrays, is_moving, validation_predictions, validation_rows)
        measures["fold"] = fold["name"]
        fold_results.append(measures)
    return {
        "name": variant["name"],
        "settings": {key: (str(value) if isinstance(value, Path) else value) for key, value in
                     ((key, parameters.get(key, target_parameters.get(key))) for key in sorted(set(parameters) | set(target_parameters)))},
        "class_names": class_names,
        "folds": fold_results,
        "summary": summarised_folds(fold_results),
        "seconds": round(time.perf_counter() - started, 1)
    }



def run_experiment(
        experiment_name: str,
        frames_cache: Dict,
        features_cache: Dict
) -> List[Dict]:
    """
    Run every variant of one experiment and save the results.

    Returns
    -------
    list of dict
        The results of ``run_variant`` for every variant.
    """

    experiment = TRAINING_PROBE_EXPERIMENTS[experiment_name]
    output_folder = Path(TRAINING_PROBE_PARAMETERS["output_folder"])
    output_folder.mkdir(parents = True, exist_ok = True)
    results = []
    for variant in experiment["variants"]:
        merged_variant = dict(experiment.get("shared", {}))
        merged_variant.update(variant)
        result = run_variant(merged_variant, frames_cache, features_cache)
        results.append(result)
        summary = result["summary"]
        print(f"{experiment_name} | {result['name']} | {result['seconds']} s | "
              f"within 45: {summary.get('within_45_deg_mean')} | class accuracy: {summary.get('class_accuracy_mean')} | "
              f"stopped right: {summary.get('episodes_stopped_right_mean')} | early: {summary.get('episodes_stopped_early_mean')}", flush = True)
        (output_folder / f"{experiment_name}.json").write_text(json.dumps({"question": experiment["question"], "results": results}, indent = 1), encoding = "utf-8")
        append_result_row(output_folder / "results.csv", experiment_name, result)
    return results



def append_result_row(
        table_path: Path,
        experiment_name: str,
        result: Dict
) -> None:
    """Add one variant's summary as a row to the results table, with the settings that tell variants apart."""

    row = {"experiment": experiment_name, "variant": result["name"]}
    for key in ("backbone", "feature_kind", "design", "split", "small_motion_rule", "history_lags", "class_weight_power", "kept_scores", "hidden_size"):
        row[key] = json.dumps(result["settings"].get(key)) if isinstance(result["settings"].get(key), list) else result["settings"].get(key)
    row.update(result["summary"])
    is_new = not table_path.exists()
    existing_columns = []
    if not is_new:
        with open(table_path, encoding = "utf-8", newline = "") as table_file:
            existing_columns = next(csv.reader(table_file), [])
    columns = existing_columns + [key for key in row if key not in existing_columns]
    if not is_new and columns != existing_columns:
        old_rows = pandas.read_csv(table_path).to_dict("records")
        with open(table_path, "w", encoding = "utf-8", newline = "") as table_file:
            writer = csv.DictWriter(table_file, fieldnames = columns)
            writer.writeheader()
            writer.writerows(old_rows)
    with open(table_path, "a", encoding = "utf-8", newline = "") as table_file:
        writer = csv.DictWriter(table_file, fieldnames = columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)



def baseline_measures(split: str) -> Dict:
    """
    Score three rules that need no model, on the same folds and measures as the probes.

    "always_forward" pushes straight ahead and never stops; "training_majority" always names the most common moving
    class of the training sheets; "repeat_previous_move" repeats the move the demonstration made one picture earlier.
    The last rule uses the previous move, which a picture-only model never sees, so it is a reference for how much
    the moves follow on from each other, not a rival model.

    Parameters
    ----------
    split : str
        "sheets" or "episodes".

    Returns
    -------
    dict
        Per rule, the summary over the folds and the per-fold measures.
    """

    frames = load_training_frames()
    is_moving = frames["action"].isin(DIRECTION_NAMES).to_numpy()
    true_angle = frames["move_angle_deg"].to_numpy()
    previous_angle = true_angle[previous_rows(frames, 1)]
    path_arrays = {"chunk": chunk_targets(frames, TRAINING_METRIC_PARAMETERS["path_steps"]).reshape(len(frames), -1)}
    folds = (leave_one_sheet_out_folds(frames) if split == "sheets"
             else episode_folds(frames, TRAINING_TARGET_PARAMETERS["episode_fold_count"], TRAINING_TARGET_PARAMETERS["seed"]))
    results = {}
    for rule in ("always_forward", "training_majority", "repeat_previous_move"):
        fold_results = []
        for fold in folds:
            rows = numpy.flatnonzero(fold["test"])
            if rule == "always_forward":
                predicted = numpy.zeros(len(rows))
            elif rule == "training_majority":
                majority = frames.loc[fold["train"] & is_moving, "action"].value_counts().idxmax()
                predicted = numpy.full(len(rows), float(DIRECTION_ANGLES_DEG[majority]))
            else:
                predicted = previous_angle[rows]
            measures = direction_metrics(true_angle[rows], predicted, is_moving[rows])
            path_angle, is_path_row = path_directions(frames, rows, path_arrays)
            measures.update({f"path_{name}": value for name, value in direction_metrics(path_angle, predicted, is_path_row).items()})
            measures["fold"] = fold["name"]
            fold_results.append(measures)
        results[rule] = {"summary": summarised_folds(fold_results), "folds": fold_results}
    return results



def main(experiment_names: List[str]) -> None:

    torch.set_num_threads(TRAINING_PROBE_PARAMETERS["threads"])
    if experiment_names == ["baselines"]:
        output_folder = Path(TRAINING_PROBE_PARAMETERS["output_folder"])
        output_folder.mkdir(parents = True, exist_ok = True)
        baselines = {split: baseline_measures(split) for split in ("sheets", "episodes")}
        (output_folder / "baselines.json").write_text(json.dumps(baselines, indent = 1), encoding = "utf-8")
        for split, rules in baselines.items():
            for rule, result in rules.items():
                print(split, rule, {key: value for key, value in result["summary"].items() if key.endswith("_mean")})
        return
    names = list(TRAINING_PROBE_EXPERIMENTS) if experiment_names in ([], ["all"]) else experiment_names
    frames_cache = {}
    features_cache = {}
    for name in names:
        run_experiment(name, frames_cache, features_cache)



if __name__ == "__main__":
    main(sys.argv[1:])
