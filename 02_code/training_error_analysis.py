"""
Where a model goes wrong, situation by situation (9.11.26, Model and training session).

Reads the predictions a probe variant (training_probes.py, saved under 04_runs/<date>_frozen_feature_probes/predictions/)
or an end-to-end run (train_policy.py, saved as predictions_<fold>.npz in its run folder) made on its held-out pictures,
and scores them per situation: sheet, place in the episode (start, middle, near the goal), bends against straight
stretches, how far the forearm is turned, light, camera view (the taped camera moved once, on the evening of 8.11.26)
and score. The result says which situations to record more of, and which a model handles already.

Run from 02_code:
    ..\\.venv\\Scripts\\python.exe training_error_analysis.py <variant or run folder name>
Settings: TRAINING_ERROR_ANALYSIS_PARAMETERS in config.py.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy
import pandas

from config import TRAINING_ERROR_ANALYSIS_PARAMETERS, TRAINING_METRIC_PARAMETERS, TRAINING_PROBE_PARAMETERS, TRAINING_RUN_PARAMETERS
from training_metrics import angle_difference_deg
from training_targets import chunk_targets, load_training_frames



def prediction_files(name: str) -> List[Path]:
    """Return the prediction files of a probe variant or of an end-to-end run, found by name."""

    probe_folder = Path(TRAINING_PROBE_PARAMETERS["output_folder"]) / "predictions"
    files = sorted(probe_folder.glob(f"{name}_without_*.npz")) + sorted(probe_folder.glob(f"{name}_episodes_fold_*.npz"))
    if files:
        return files
    run_folders = sorted(Path(TRAINING_RUN_PARAMETERS["output_root"]).glob(f"*_{name}"))
    return sorted(run_folders[-1].glob("predictions_*.npz")) if run_folders else []



def situation_columns(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Return, for every row, the situation it is in.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_training_frames``, sorted by episode and step.
    parameters : dict
        The analysis settings.

    Returns
    -------
    pandas.DataFrame
        "place" (start, middle, near_goal, stop), "shape" (bend or straight, from the change of the tip's direction
        over "bend_window_steps" steps before and after), "forearm" (straight, turned or far_turned), "camera_view" (A
        before the camera moved, B after it settled, settling between), "lighting" and "score".
    """

    situations = pandas.DataFrame(index = frames.index)
    is_stop = frames["action"] == "stop"
    is_start = frames["row_in_episode"] < parameters["start_rows"]
    is_near_goal = frames["goal_distance_cm"].fillna(99.0) <= parameters["near_goal_cm"]
    situations["place"] = numpy.select([is_stop, is_start, is_near_goal], ["stop", "start", "near_goal"], default = "middle")

    window = parameters["bend_window_steps"]
    tip_x = frames["tip_x_m"].to_numpy()
    tip_y = frames["tip_y_m"].to_numpy()
    episode = frames["episode_id"].to_numpy()
    turning = numpy.zeros(len(frames))
    for position in range(len(frames)):
        before, after = position - window, position + window
        if before < 0 or after >= len(frames) or episode[before] != episode[position] or episode[after] != episode[position]:
            continue
        first = numpy.array([tip_x[position] - tip_x[before], tip_y[position] - tip_y[before]])
        second = numpy.array([tip_x[after] - tip_x[position], tip_y[after] - tip_y[position]])
        if numpy.linalg.norm(first) < 0.005 or numpy.linalg.norm(second) < 0.005:
            continue
        turning[position] = angle_difference_deg(numpy.degrees(numpy.arctan2(second[1], second[0])),
                                                 numpy.degrees(numpy.arctan2(first[1], first[0])))
    situations["shape"] = numpy.where(turning > parameters["bend_threshold_deg"], "bend", "straight")

    turn = frames["forearm_turn_deg"].abs().fillna(0.0)
    situations["forearm"] = numpy.select([turn < parameters["forearm_turned_deg"], turn < parameters["forearm_far_turned_deg"]],
                                         ["straight", "turned"], default = "far_turned")
    started = pandas.to_datetime(frames["started"])
    moved_from = pandas.Timestamp(parameters["camera_moved_from"])
    settled_at = pandas.Timestamp(parameters["camera_settled_at"])
    situations["camera_view"] = numpy.select([started < moved_from, started >= settled_at], ["A", "B"], default = "settling")
    situations["lighting"] = frames["lighting"].to_numpy()
    situations["score"] = frames["score"].astype(str).to_numpy()
    situations["sheet"] = frames["sheet"].to_numpy()
    return situations



def error_table(
        name: str,
        parameters: Dict = TRAINING_ERROR_ANALYSIS_PARAMETERS
) -> pandas.DataFrame:
    """
    Score one variant's or run's held-out predictions per situation.

    Returns
    -------
    pandas.DataFrame
        One row per situation value: pictures scored, the share whose predicted direction is within 45 degrees of the
        demonstrated step and of the path direction, and the same shares for "always forward" beside them.
    """

    frames = load_training_frames()
    index_of_frame = pandas.Series(numpy.arange(len(frames)), index = frames["frame_index"].to_numpy())
    situations = situation_columns(frames, parameters)
    chunk = chunk_targets(frames, TRAINING_METRIC_PARAMETERS["path_steps"])[:, -1, :]
    path_angle = numpy.degrees(numpy.arctan2(chunk[:, 1], chunk[:, 0]))
    has_path = (numpy.linalg.norm(chunk, axis = 1) >= TRAINING_METRIC_PARAMETERS["path_minimum_cm"]) & (frames["action"] != "stop").to_numpy()
    is_moving = frames["action"].isin(["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"]).to_numpy()

    parts = []
    for file in prediction_files(name):
        predictions = numpy.load(file)
        positions = index_of_frame.reindex(predictions["frame_index"]).to_numpy()
        is_known = ~numpy.isnan(positions)
        positions = positions[is_known].astype(int)
        predicted = predictions["angle_deg"][is_known]
        part = situations.iloc[positions].copy()
        part["step_hit"] = numpy.where(is_moving[positions], angle_difference_deg(predicted, frames["move_angle_deg"].to_numpy()[positions]) <= 45, numpy.nan)
        part["path_hit"] = numpy.where(has_path[positions], angle_difference_deg(predicted, path_angle[positions]) <= 45, numpy.nan)
        part["forward_path_hit"] = numpy.where(has_path[positions], angle_difference_deg(0.0, path_angle[positions]) <= 45, numpy.nan)
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"no saved predictions for {name}")
    scored = pandas.concat(parts)
    rows = []
    for column in ["sheet", "place", "shape", "forearm", "camera_view", "lighting", "score"]:
        for value, group in scored.groupby(column):
            rows.append({"situation": column, "value": value, "pictures": len(group),
                         "step_within_45": round(float(group["step_hit"].mean()), 3),
                         "path_within_45": round(float(group["path_hit"].mean()), 3),
                         "always_forward_path_within_45": round(float(group["forward_path_hit"].mean()), 3)})
    return pandas.DataFrame(rows)



def smoothing_table(
        name: str,
        windows: List[int]
) -> pandas.DataFrame:
    """
    Score the predicted direction averaged over the last few pictures of the same episode.

    Each earlier prediction is turned from its own picture's frame into the current picture's frame with the logged
    picture headings (the controller knows how the forearm turned; the model never sees it), then the directions are
    averaged as unit vectors. Scored against the path direction of the current picture. This approximates averaging
    at run time on the demonstrated path; on the robot the model's own path would differ.

    Parameters
    ----------
    name : str
        A probe variant or an end-to-end run with saved predictions.
    windows : list of int
        How many pictures to average (1: no averaging).

    Returns
    -------
    pandas.DataFrame
        One row per window: pictures scored and the share within 45 degrees of the path direction.
    """

    frames = load_training_frames()
    index_of_frame = pandas.Series(numpy.arange(len(frames)), index = frames["frame_index"].to_numpy())
    chunk = chunk_targets(frames, TRAINING_METRIC_PARAMETERS["path_steps"])[:, -1, :]
    path_angle = numpy.degrees(numpy.arctan2(chunk[:, 1], chunk[:, 0]))
    has_path = (numpy.linalg.norm(chunk, axis = 1) >= TRAINING_METRIC_PARAMETERS["path_minimum_cm"]) & (frames["action"] != "stop").to_numpy()
    heading = frames["picture_heading_deg"].to_numpy()
    episode = frames["episode_id"].to_numpy()
    predicted_table = numpy.full(len(frames), numpy.nan)
    for file in prediction_files(name):
        predictions = numpy.load(file)
        positions = index_of_frame.reindex(predictions["frame_index"]).to_numpy()
        is_known = ~numpy.isnan(positions)
        positions = positions[is_known].astype(int)
        predicted_table[positions] = heading[positions] + predictions["angle_deg"][is_known]
    rows = []
    for window in windows:
        hits = []
        for position in numpy.flatnonzero(has_path & ~numpy.isnan(predicted_table)):
            earlier = [position - back for back in range(window) if position - back >= 0 and episode[position - back] == episode[position]
                       and not numpy.isnan(predicted_table[position - back])]
            radians = numpy.radians(predicted_table[earlier])
            mean_table = numpy.degrees(numpy.arctan2(numpy.sin(radians).mean(), numpy.cos(radians).mean()))
            hits.append(angle_difference_deg(mean_table - heading[position], path_angle[position]) <= 45)
        rows.append({"window": window, "pictures": len(hits), "path_within_45": round(float(numpy.mean(hits)), 4)})
    return pandas.DataFrame(rows)



def main(names: List[str]) -> None:

    output_folder = Path(TRAINING_PROBE_PARAMETERS["output_folder"])
    for name in names:
        table = error_table(name)
        table.to_csv(output_folder / f"errors_{name}.csv", index = False)
        print(f"\n{name}")
        print(table.to_string(index = False))
        smoothing = smoothing_table(name, TRAINING_ERROR_ANALYSIS_PARAMETERS["smoothing_windows"])
        smoothing.to_csv(output_folder / f"smoothing_{name}.csv", index = False)
        print(smoothing.to_string(index = False))



if __name__ == "__main__":
    main(sys.argv[1:])
