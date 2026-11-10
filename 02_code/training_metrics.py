"""
Offline measures that every model is scored with, whatever it predicts (9.11.26, Model and training session).

A model that names a move class and a model that predicts the exact step must be compared on the same scale, so every
prediction is turned into a direction on the table (in the picture's frame) and a decision to stop:
    direction   the angle between the predicted push direction and the demonstrated one, on rows where the tip
                really moved; the share within 30, 45 and 60 degrees; the class accuracy, with a predicted vector
                turned into the nearest class by the recording tool's own rule;
    stop        how often the model stops at the right picture, and how often it stops too early (more than
                "early_stop_cm" before the goal), the mistake that ends a run;
    twist       whether the forearm turns the way the demonstration turned it, on rows where it turned.
Thresholded shares are reported beside averages because a thresholded error tracks driving success better than a mean
error in the study of Codevilla and others (2018).

Settings: TRAINING_METRIC_PARAMETERS in config.py.
"""

from typing import Dict, List, Optional

import numpy

from config import TRAINING_METRIC_PARAMETERS
from training_targets import DIRECTION_ANGLES_DEG, DIRECTION_NAMES

# Reference
# Source: Codevilla, Lopez, Koltun and Dosovitskiy, "On Offline Evaluation of Vision-based Driving Models", ECCV 2018 -
#         offline prediction error against closed-loop driving; thresholded errors correlate better than mean errors.
# Link: https://arxiv.org/abs/1809.04843



def merged_metric_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``TRAINING_METRIC_PARAMETERS`` with any overrides merged over it."""

    merged = dict(TRAINING_METRIC_PARAMETERS)
    merged.update(parameters or {})
    return merged



def angle_difference_deg(
        first_deg: numpy.ndarray,
        second_deg: numpy.ndarray
) -> numpy.ndarray:
    """Return the absolute difference of two angles in degrees, between 0 and 180."""

    return numpy.abs((numpy.asarray(first_deg) - numpy.asarray(second_deg) + 180.0) % 360.0 - 180.0)



def nearest_direction(angles_deg: numpy.ndarray) -> numpy.ndarray:
    """
    Return the index in ``DIRECTION_NAMES`` of the fixed direction nearest to each angle, the recording tool's rule.

    Parameters
    ----------
    angles_deg : numpy.ndarray
        Angles in the picture's frame, 0 forward, positive to the left.

    Returns
    -------
    numpy.ndarray
        int64 indices.
    """

    nominal = numpy.array([DIRECTION_ANGLES_DEG[name] for name in DIRECTION_NAMES], dtype = float)
    differences = angle_difference_deg(numpy.asarray(angles_deg)[:, None], nominal[None, :])
    return numpy.argmin(differences, axis = 1)



def direction_metrics(
        true_angle_deg: numpy.ndarray,
        predicted_angle_deg: numpy.ndarray,
        is_moving: numpy.ndarray,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    Score predicted push directions against the demonstrated ones on the rows where the tip moved.

    Parameters
    ----------
    true_angle_deg : numpy.ndarray
        The demonstrated move's angle in the picture's frame (move_angle_deg).
    predicted_angle_deg : numpy.ndarray
        The model's direction: the nominal angle of its class, or the angle of its predicted vector.
    is_moving : numpy.ndarray
        True where the tip moved at least the small-motion distance and the row is not the stop row.
    parameters : dict, optional
        Overrides for ``TRAINING_METRIC_PARAMETERS`` ("angle_thresholds_deg").

    Returns
    -------
    dict
        "direction_rows", the median and mean angle error, the share within each threshold, the class accuracy and
        the balanced class accuracy (each true class counted equally).
    """

    parameters = merged_metric_parameters(parameters)
    is_moving = numpy.asarray(is_moving, dtype = bool)
    if not is_moving.any():
        return {"direction_rows": 0}
    errors = angle_difference_deg(true_angle_deg[is_moving], predicted_angle_deg[is_moving])
    true_class = nearest_direction(true_angle_deg[is_moving])
    predicted_class = nearest_direction(predicted_angle_deg[is_moving])
    recalls = [float((predicted_class[true_class == index] == index).mean()) for index in range(len(DIRECTION_NAMES))
               if (true_class == index).any()]
    result = {
        "direction_rows": int(is_moving.sum()),
        "angle_error_median_deg": round(float(numpy.median(errors)), 2),
        "angle_error_mean_deg": round(float(errors.mean()), 2),
        "class_accuracy": round(float((true_class == predicted_class).mean()), 4),
        "class_balanced_accuracy": round(float(numpy.mean(recalls)), 4)
    }
    for threshold in parameters["angle_thresholds_deg"]:
        result[f"within_{threshold}_deg"] = round(float((errors <= threshold).mean()), 4)
    return result



def stop_metrics(
        is_stop: numpy.ndarray,
        stop_probability: numpy.ndarray,
        goal_distance_cm: numpy.ndarray,
        episode_ids: numpy.ndarray,
        parameters: Optional[Dict] = None,
        threshold: Optional[float] = None,
        consecutive_pictures: int = 1
) -> Dict:
    """
    Score the stop decision per picture and per episode.

    Per episode the first picture whose stop probability passes the threshold is where the model would stop: the
    episode counts as stopped right when that is within "right_stop_cm" of the goal, too early when it is more than
    "early_stop_cm" before the goal, and missed when no picture passes.

    Parameters
    ----------
    is_stop : numpy.ndarray
        1 on the stop rows.
    stop_probability : numpy.ndarray
        The model's stop probability per row.
    goal_distance_cm : numpy.ndarray
        The tip's distance to the goal per row.
    episode_ids : numpy.ndarray
        The episode of each row, rows in step order within an episode.
    parameters : dict, optional
        Overrides for ``TRAINING_METRIC_PARAMETERS`` ("stop_threshold", "early_stop_cm", "right_stop_cm").
    threshold : float, optional
        The stop threshold; "stop_threshold" when not given.
    consecutive_pictures : int, optional
        The robot stops only after this many pictures in a row pass the threshold, as a run-time rule would.

    Returns
    -------
    dict
        Picture-level precision and recall, the average precision, and the episode-level shares.
    """

    parameters = merged_metric_parameters(parameters)
    threshold = parameters["stop_threshold"] if threshold is None else threshold
    is_stop = numpy.asarray(is_stop) > 0.5
    is_predicted = numpy.asarray(stop_probability) >= threshold
    true_positive = float((is_predicted & is_stop).sum())
    precision = true_positive / max(1.0, float(is_predicted.sum()))
    recall = true_positive / max(1.0, float(is_stop.sum()))
    order = numpy.argsort(-numpy.asarray(stop_probability))
    sorted_stop = is_stop[order]
    cumulative_true = numpy.cumsum(sorted_stop)
    precision_at_rank = cumulative_true / numpy.arange(1, len(sorted_stop) + 1)
    average_precision = float((precision_at_rank * sorted_stop).sum() / max(1, sorted_stop.sum()))

    stopped_right = stopped_early = missed = 0
    episode_count = 0
    for episode_id in numpy.unique(episode_ids):
        rows = numpy.flatnonzero(episode_ids == episode_id)
        episode_count += 1
        passing = rows[is_predicted[rows]]
        if consecutive_pictures > 1:
            run_length = numpy.zeros(len(rows), dtype = int)
            for position, row in enumerate(rows):
                run_length[position] = (run_length[position - 1] + 1 if position > 0 else 1) if is_predicted[row] else 0
            passing = rows[run_length >= consecutive_pictures]
        if len(passing) == 0:
            missed += 1
            continue
        distance = float(goal_distance_cm[passing[0]])
        if distance <= parameters["right_stop_cm"]:
            stopped_right += 1
        elif distance > parameters["early_stop_cm"]:
            stopped_early += 1
    return {
        "stop_precision": round(precision, 4),
        "stop_recall": round(recall, 4),
        "stop_average_precision": round(average_precision, 4),
        "episodes_stopped_right": round(stopped_right / max(1, episode_count), 4),
        "episodes_stopped_early": round(stopped_early / max(1, episode_count), 4),
        "episodes_never_stopped": round(missed / max(1, episode_count), 4)
    }



def tuned_stop_threshold(
        is_stop: numpy.ndarray,
        stop_probability: numpy.ndarray,
        goal_distance_cm: numpy.ndarray,
        episode_ids: numpy.ndarray,
        consecutive_pictures: int,
        parameters: Optional[Dict] = None
) -> float:
    """
    Return the stop threshold that stops most episodes at the right place, chosen on validation episodes only.

    Ties go to the threshold with fewer early stops, because a stop before the goal ends a run as a failure.

    Parameters
    ----------
    is_stop, stop_probability, goal_distance_cm, episode_ids : numpy.ndarray
        The validation rows, as for ``stop_metrics``.
    consecutive_pictures : int
        The run-time rule's number of pictures in a row.
    parameters : dict, optional
        Supplies "stop_threshold_candidates".

    Returns
    -------
    float
        The chosen threshold.
    """

    parameters = merged_metric_parameters(parameters)
    best_threshold = parameters["stop_threshold"]
    best_key = None
    for candidate in parameters["stop_threshold_candidates"]:
        result = stop_metrics(is_stop, stop_probability, goal_distance_cm, episode_ids, parameters, candidate, consecutive_pictures)
        key = (result["episodes_stopped_right"], -result["episodes_stopped_early"])
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = candidate
    return float(best_threshold)



def twist_metrics(
        true_change_deg: numpy.ndarray,
        predicted_change_deg: numpy.ndarray,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    Score the forearm twist: the sign on rows that twisted at least "twist_threshold_deg", and the mean error.

    Returns
    -------
    dict
        "twist_rows", "twist_sign_accuracy" and "twist_error_mean_deg".
    """

    parameters = merged_metric_parameters(parameters)
    true_change_deg = numpy.asarray(true_change_deg, dtype = float)
    predicted_change_deg = numpy.asarray(predicted_change_deg, dtype = float)
    is_twist = numpy.abs(true_change_deg) >= parameters["twist_threshold_deg"]
    if not is_twist.any():
        return {"twist_rows": 0}
    return {
        "twist_rows": int(is_twist.sum()),
        "twist_sign_accuracy": round(float((numpy.sign(predicted_change_deg[is_twist]) == numpy.sign(true_change_deg[is_twist])).mean()), 4),
        "twist_error_mean_deg": round(float(numpy.abs(predicted_change_deg - true_change_deg).mean()), 3)
    }



def change_metrics(
        true_change_deg: numpy.ndarray,
        predicted_change_deg: numpy.ndarray,
        name: str,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    Score a predicted joint change (the rod's lean or the hand's turn) like the twist: the sign where the
    demonstration changed by at least "<name>_threshold_deg", and the mean error.

    Returns
    -------
    dict
        "<name>_rows", "<name>_sign_accuracy" and "<name>_error_mean_deg".
    """

    parameters = merged_metric_parameters(parameters)
    true_change_deg = numpy.asarray(true_change_deg, dtype = float)
    predicted_change_deg = numpy.asarray(predicted_change_deg, dtype = float)
    is_changed = numpy.abs(true_change_deg) >= parameters[f"{name}_threshold_deg"]
    if not is_changed.any():
        return {f"{name}_rows": 0}
    return {
        f"{name}_rows": int(is_changed.sum()),
        f"{name}_sign_accuracy": round(float((numpy.sign(predicted_change_deg[is_changed]) == numpy.sign(true_change_deg[is_changed])).mean()), 4),
        f"{name}_error_mean_deg": round(float(numpy.abs(predicted_change_deg - true_change_deg).mean()), 3)
    }



def chunk_metrics(
        true_chunk_cm: numpy.ndarray,
        predicted_chunk_cm: numpy.ndarray
) -> Dict:
    """
    Score predicted tip places several steps ahead: the mean distance error after each step and at the last one.

    Parameters
    ----------
    true_chunk_cm, predicted_chunk_cm : numpy.ndarray
        (rows, steps_ahead, 2) in cm.

    Returns
    -------
    dict
        "chunk_error_step_<n>_cm" for every step ahead.
    """

    distances = numpy.linalg.norm(numpy.asarray(true_chunk_cm) - numpy.asarray(predicted_chunk_cm), axis = 2)
    return {f"chunk_error_step_{step + 1}_cm": round(float(distances[:, step].mean()), 3) for step in range(distances.shape[1])}



def summarised_folds(fold_results: List[Dict]) -> Dict:
    """
    Return the mean, the smallest and the largest value of every number over the folds.

    Parameters
    ----------
    fold_results : list of dict
        One dictionary of numbers per fold.

    Returns
    -------
    dict
        "<name>_mean", "<name>_minimum" and "<name>_maximum" for every numeric entry.
    """

    summary = {}
    names = [name for name, value in fold_results[0].items() if isinstance(value, (int, float))]
    for name in names:
        values = [result[name] for result in fold_results if isinstance(result.get(name), (int, float))]
        if not values:
            continue
        summary[f"{name}_mean"] = round(float(numpy.mean(values)), 4)
        summary[f"{name}_minimum"] = round(float(numpy.min(values)), 4)
        summary[f"{name}_maximum"] = round(float(numpy.max(values)), 4)
    return summary
