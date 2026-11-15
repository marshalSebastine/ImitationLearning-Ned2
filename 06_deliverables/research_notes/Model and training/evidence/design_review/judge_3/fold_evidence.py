"""
Evidence check for judge 3 of the training design (9.11.26, read only).

Re-scores the saved held-out-sheet predictions of the end-to-end runs from their npz files and frames.csv, without
importing project code and without writing anything under D:/AI_Project. It first reproduces the stored fold measures
(within 45 degrees, path within 45 degrees, stop outcomes at 0.5) as a self-check, then adds what no fold file holds:

1. the final candidate's stop outcome per fold for every candidate threshold, one and two pictures in a row, pooled
   over the folds that exist, and a threshold chosen on the other held-out sheets (leave one sheet out);
2. always-forward on each fold's own test rows (the final candidate's rows are the audit-cleaned ones);
3. backing up: within 45 degrees on the demonstrated backward moves, and on the other moves;
4. the forearm twist gain (least squares through zero) and the sign by predicted size;
5. paired differences of two runs on the same rows, with a 95 % episode-bootstrap interval.

Writes fold_evidence.out.json and fold_evidence.out.txt next to this file.
"""

import json
import math
from pathlib import Path

import numpy
import pandas

ANALYSIS_PARAMETERS = {
    "runs_folder": Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs"),
    "frames_path": Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv"),
    "output_json_path": Path(__file__).with_name("fold_evidence.out.json"),
    "output_text_path": Path(__file__).with_name("fold_evidence.out.txt"),
    "direction_names": ["forward", "forward_left", "left", "back_left", "back", "back_right", "right", "forward_right"],
    "backward_names": ["back", "back_left", "back_right"],
    "right_stop_cm": 2.0,
    "early_stop_cm": 3.0,
    "path_steps": 3,
    "path_minimum_cm": 1.5,
    "angle_threshold_deg": 45.0,
    "twist_threshold_deg": 2.0,
    "twist_clip_deg": 15.0,
    "thresholds": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "bootstrap_draws": 2000,
    "bootstrap_seed": 91126,
}

RUN_FOLDS = {
    "final_candidate": ("2026-11-09_resnet18_final_candidate", ["01", "02", "03", "04", "05", "06"]),
    "finetune_exact": ("2026-11-09_resnet18_finetune_exact", ["03", "05", "06"]),
    "finetune_chunk": ("2026-11-09_resnet18_finetune_chunk", ["03", "05", "06"]),
    "finetune_classes": ("2026-11-09_resnet18_finetune_classes", ["03", "05", "06"]),
    "finetune_mirror": ("2026-11-09_resnet18_finetune_mirror", ["03", "05", "06"]),
    "small_cnn_classes": ("2026-11-09_small_cnn_scratch_classes", ["03", "05", "06"]),
}

PAIRED_COMPARISONS = [
    ("finetune_chunk", "angle_deg", "finetune_exact", "angle_deg"),
    ("finetune_exact", "angle_deg", "finetune_classes", "angle_deg"),
    ("finetune_exact", "angle_deg", "finetune_classes", "angle_expected_deg"),
    ("finetune_mirror", "angle_deg", "finetune_classes", "angle_deg"),
    ("finetune_classes", "angle_deg", "small_cnn_classes", "angle_deg"),
    ("final_candidate", "angle_deg", "finetune_exact", "angle_deg"),
]



def angle_difference_deg(
        first_deg: numpy.ndarray,
        second_deg: numpy.ndarray
) -> numpy.ndarray:
    """Return the absolute difference of two angles in degrees, between 0 and 180."""

    return numpy.abs((numpy.asarray(first_deg, dtype = float) - numpy.asarray(second_deg, dtype = float) + 180.0) % 360.0 - 180.0)



def path_angles(
        joined: pandas.DataFrame,
        parameters: dict
) -> tuple:
    """
    Return the direction from the tip now to where it is "path_steps" rows later, in the current picture's frame.

    Parameters
    ----------
    joined : pandas.DataFrame
        The test rows in the order the model scored them, with tip positions and picture headings.
    parameters : dict
        Supplies "path_steps".

    Returns
    -------
    tuple of numpy.ndarray
        The path angle in degrees and the path length in cm for every row.
    """

    tip_x = joined["tip_x_m"].to_numpy()
    tip_y = joined["tip_y_m"].to_numpy()
    heading = numpy.radians(joined["picture_heading_deg"].to_numpy())
    episode_ids = joined["episode_id"].to_numpy()
    angle = numpy.full(len(joined), numpy.nan)
    length = numpy.zeros(len(joined))
    starts = numpy.flatnonzero(numpy.r_[True, episode_ids[1:] != episode_ids[:-1]])
    ends = numpy.r_[starts[1:], len(joined)]
    for start, end in zip(starts, ends):
        for position in range(start, end):
            future = min(position + parameters["path_steps"], end - 1)
            moved_x = tip_x[future] - tip_x[position]
            moved_y = tip_y[future] - tip_y[position]
            forward = moved_x * math.cos(heading[position]) + moved_y * math.sin(heading[position])
            left = moved_x * math.cos(heading[position] + math.pi / 2.0) + moved_y * math.sin(heading[position] + math.pi / 2.0)
            angle[position] = math.degrees(math.atan2(left, forward))
            length[position] = 100.0 * math.hypot(forward, left)
    return angle, length



def stop_outcome_counts(
        stop_probability: numpy.ndarray,
        goal_distance_cm: numpy.ndarray,
        episode_ids: numpy.ndarray,
        threshold: float,
        consecutive_pictures: int,
        parameters: dict
) -> dict:
    """
    Count episodes stopped right, early, never and between 2 and 3 cm, as training_metrics.stop_metrics scores them.

    Returns
    -------
    dict
        Counts, and the distances to the goal of the early stops.
    """

    counts = {"episodes": 0, "right": 0, "early": 0, "never": 0, "between": 0, "early_distance_cm": []}
    for episode_id in numpy.unique(episode_ids):
        rows = numpy.flatnonzero(episode_ids == episode_id)
        counts["episodes"] += 1
        passing = stop_probability[rows] >= threshold
        run_length = numpy.zeros(len(rows), dtype = int)
        for position in range(len(rows)):
            run_length[position] = ((run_length[position - 1] + 1) if position > 0 else 1) if passing[position] else 0
        hits = numpy.flatnonzero(run_length >= consecutive_pictures)
        if len(hits) == 0:
            counts["never"] += 1
            continue
        distance = float(goal_distance_cm[rows[hits[0]]])
        if distance <= parameters["right_stop_cm"]:
            counts["right"] += 1
        elif distance > parameters["early_stop_cm"]:
            counts["early"] += 1
            counts["early_distance_cm"].append(round(distance, 1))
        else:
            counts["between"] += 1
    return counts



def episode_hits(
        error_deg: numpy.ndarray,
        mask: numpy.ndarray,
        episode_ids: numpy.ndarray,
        parameters: dict
) -> pandas.DataFrame:
    """Return per-episode counts of scored rows and rows within the angle threshold."""

    table = pandas.DataFrame({"episode_id": episode_ids[mask], "hit": (error_deg[mask] <= parameters["angle_threshold_deg"]).astype(int)})
    grouped = table.groupby("episode_id")["hit"].agg(["sum", "count"])
    return grouped



def bootstrap_difference(
        first: pandas.DataFrame,
        second: pandas.DataFrame,
        parameters: dict
) -> dict:
    """
    Return the pooled share difference (first minus second) and its 95 % episode-bootstrap interval.

    Both tables are indexed by episode and hold "sum" (hits) and "count" (scored rows) on the same rows.
    """

    episodes = first.index.intersection(second.index)
    first = first.loc[episodes]
    second = second.loc[episodes]
    generator = numpy.random.default_rng(parameters["bootstrap_seed"])
    first_sum, first_count = first["sum"].to_numpy(), first["count"].to_numpy()
    second_sum, second_count = second["sum"].to_numpy(), second["count"].to_numpy()
    point = first_sum.sum() / first_count.sum() - second_sum.sum() / second_count.sum()
    draws = generator.integers(0, len(episodes), size = (parameters["bootstrap_draws"], len(episodes)))
    differences = first_sum[draws].sum(axis = 1) / first_count[draws].sum(axis = 1) - second_sum[draws].sum(axis = 1) / second_count[draws].sum(axis = 1)
    return {"episodes": int(len(episodes)), "rows": int(first_count.sum()), "difference_points": round(100.0 * float(point), 1),
            "interval_points": [round(100.0 * float(numpy.percentile(differences, 2.5)), 1), round(100.0 * float(numpy.percentile(differences, 97.5)), 1)]}



def wilson_interval(
        successes: int,
        trials: int
) -> list:
    """Return the 95 % Wilson interval of a share."""

    if trials == 0:
        return [float("nan"), float("nan")]
    z_value = 1.959964
    share = successes / trials
    centre = (share + z_value * z_value / (2 * trials)) / (1 + z_value * z_value / trials)
    half_width = z_value * math.sqrt(share * (1 - share) / trials + z_value * z_value / (4 * trials * trials)) / (1 + z_value * z_value / trials)
    return [round(centre - half_width, 3), round(centre + half_width, 3)]



def score_fold(
        predictions: dict,
        frames: pandas.DataFrame,
        parameters: dict
) -> tuple:
    """
    Score one fold's saved predictions.

    Returns
    -------
    tuple
        The fold's measures (dict) and the joined test rows with their masks and errors (pandas.DataFrame).
    """

    index = pandas.DataFrame({"frame_index": predictions["frame_index"], "order": numpy.arange(len(predictions["frame_index"]))})
    joined = index.merge(frames, on = "frame_index", how = "left").sort_values("order").reset_index(drop = True)
    path_angle, path_length = path_angles(joined, parameters)
    action = joined["action"].to_numpy()
    episode_ids = joined["episode_id"].to_numpy()
    is_moving = numpy.isin(action, parameters["direction_names"])
    is_path_row = (action != "stop") & (path_length >= parameters["path_minimum_cm"])
    is_backward = numpy.isin(action, parameters["backward_names"])
    move_angle = joined["move_angle_deg"].to_numpy()
    joined["is_moving"] = is_moving
    joined["is_path_row"] = is_path_row
    joined["path_angle_deg"] = path_angle

    measures = {"test_rows": int(len(joined)), "test_episodes": int(len(numpy.unique(episode_ids))),
                "moving_rows": int(is_moving.sum()), "path_rows": int(is_path_row.sum()), "backward_rows": int(is_backward.sum())}
    forward_step_error = angle_difference_deg(move_angle, 0.0)
    forward_path_error = angle_difference_deg(path_angle, 0.0)
    measures["always_forward_within_45"] = round(float((forward_step_error[is_moving] <= 45.0).mean()), 4)
    measures["always_forward_path_within_45"] = round(float((forward_path_error[is_path_row] <= 45.0).mean()), 4)
    for key in ("angle_deg", "angle_expected_deg", "angle_vector_deg"):
        if key not in predictions:
            continue
        step_error = angle_difference_deg(move_angle, predictions[key])
        path_error = angle_difference_deg(path_angle, predictions[key])
        joined[f"predicted_{key}"] = numpy.asarray(predictions[key], dtype = float)
        joined[f"step_error_{key}"] = step_error
        joined[f"path_error_{key}"] = path_error
        measures[f"{key}_within_45"] = round(float((step_error[is_moving] <= 45.0).mean()), 4)
        measures[f"{key}_path_within_45"] = round(float((path_error[is_path_row] <= 45.0).mean()), 4)
        measures[f"{key}_backward_within_45"] = round(float((step_error[is_backward] <= 45.0).mean()), 4) if is_backward.any() else None
        other = is_moving & ~is_backward
        measures[f"{key}_other_moves_within_45"] = round(float((step_error[other] <= 45.0).mean()), 4)

    if "stop_probability" in predictions:
        probability = numpy.asarray(predictions["stop_probability"], dtype = float)
        distance = joined["goal_distance_cm"].fillna(0.0).to_numpy()
        grid = {}
        for threshold in parameters["thresholds"]:
            for consecutive in (1, 2):
                counts = stop_outcome_counts(probability, distance, episode_ids, threshold, consecutive, parameters)
                grid[f"t{threshold}_k{consecutive}"] = counts
        measures["stop_grid"] = grid

    if "twist_deg" in predictions and "angle_expected_deg" not in predictions:
        is_not_stop = action != "stop"
        predicted_twist = numpy.asarray(predictions["twist_deg"], dtype = float)[is_not_stop]
        true_twist = joined["forearm_change_deg"].fillna(0.0).clip(-parameters["twist_clip_deg"], parameters["twist_clip_deg"]).to_numpy()[is_not_stop]
        twisted = numpy.abs(true_twist) >= parameters["twist_threshold_deg"]
        confident = twisted & (numpy.abs(predicted_twist) >= 3.0)
        measures["twist_gain_through_zero"] = round(float((predicted_twist * true_twist).sum() / (predicted_twist * predicted_twist).sum()), 3)
        measures["twist_sign_on_twisted_rows"] = round(float((numpy.sign(predicted_twist[twisted]) == numpy.sign(true_twist[twisted])).mean()), 4)
        measures["twist_twisted_rows"] = int(twisted.sum())
        measures["twist_sign_where_predicted_3_deg_or_more"] = round(float((numpy.sign(predicted_twist[confident]) == numpy.sign(true_twist[confident])).mean()), 4) if confident.any() else None
        measures["twist_rows_predicted_3_deg_or_more"] = int(confident.sum())
        measures["twist_median_abs_predicted_on_twisted"] = round(float(numpy.median(numpy.abs(predicted_twist[twisted]))), 2)
        measures["twist_median_abs_demonstrated_on_twisted"] = round(float(numpy.median(numpy.abs(true_twist[twisted]))), 2)
    return measures, joined



def choose_threshold(
        grids: list,
        consecutive: int,
        parameters: dict
) -> float:
    """Return the threshold with the most right stops summed over the given folds; ties go to fewer early stops."""

    best_threshold, best_key = None, None
    for threshold in parameters["thresholds"]:
        right = sum(grid[f"t{threshold}_k{consecutive}"]["right"] for grid in grids)
        early = sum(grid[f"t{threshold}_k{consecutive}"]["early"] for grid in grids)
        key = (right, -early)
        if best_key is None or key > best_key:
            best_key, best_threshold = key, threshold
    return best_threshold



def leave_one_sheet_out(
        fold_grids: dict,
        parameters: dict
) -> dict:
    """Choose each fold's threshold on the other folds' held-out predictions and count the outcomes."""

    result = {}
    for consecutive in (1, 2):
        totals = {"episodes": 0, "right": 0, "early": 0, "never": 0, "between": 0}
        per_fold = {}
        for sheet, grid in fold_grids.items():
            others = [other_grid for other_sheet, other_grid in fold_grids.items() if other_sheet != sheet]
            threshold = choose_threshold(others, consecutive, parameters)
            counts = grid[f"t{threshold}_k{consecutive}"]
            per_fold[sheet] = {"threshold": threshold, **{key: counts[key] for key in totals}}
            for key in totals:
                totals[key] += counts[key]
        totals["right_wilson"] = wilson_interval(totals["right"], totals["episodes"])
        result[f"k{consecutive}"] = {"per_fold": per_fold, "total": totals}
    return result



def pooled_grid(
        fold_grids: dict,
        parameters: dict
) -> dict:
    """Sum the stop outcomes over folds for every fixed threshold and rule."""

    pooled = {}
    for threshold in parameters["thresholds"]:
        for consecutive in (1, 2):
            setting = f"t{threshold}_k{consecutive}"
            totals = {key: sum(grid[setting][key] for grid in fold_grids.values()) for key in ("episodes", "right", "early", "never", "between")}
            pooled[setting] = totals
    return pooled



def main(parameters: dict = None) -> None:
    parameters = dict(ANALYSIS_PARAMETERS, **(parameters or {}))
    columns = ["frame_index", "episode_id", "row_in_episode", "action", "goal_distance_cm", "move_angle_deg",
               "forearm_change_deg", "tip_x_m", "tip_y_m", "picture_heading_deg"]
    frames = pandas.read_csv(parameters["frames_path"], usecols = columns, low_memory = False)
    result = {"folds": {}, "stored_check": {}, "pooled": {}, "leave_one_sheet_out": {}, "paired": {}}
    joined_by_run = {}
    lines = []

    for run_name, (folder, sheets) in RUN_FOLDS.items():
        result["folds"][run_name] = {}
        joined_by_run[run_name] = {}
        for sheet in sheets:
            path = parameters["runs_folder"] / folder / f"predictions_without_path_{sheet}.npz"
            if not path.exists():
                continue
            predictions = dict(numpy.load(path))
            measures, joined = score_fold(predictions, frames, parameters)
            joined_by_run[run_name][sheet] = (predictions, joined)
            fold_path = parameters["runs_folder"] / folder / f"fold_without_path_{sheet}.json"
            if fold_path.exists():
                stored = json.loads(fold_path.read_text(encoding = "utf-8"))["measures"]
                check = {"within_45": (measures["angle_deg_within_45"], stored.get("within_45_deg")),
                         "path_within_45": (measures["angle_deg_path_within_45"], stored.get("path_within_45_deg")),
                         "direction_rows": (measures["moving_rows"], stored.get("direction_rows"))}
                if "stop_grid" in measures and "episodes_stopped_right" in stored:
                    counts = measures["stop_grid"]["t0.5_k1"]
                    check["stopped_right_0.5"] = (round(counts["right"] / counts["episodes"], 4), stored.get("episodes_stopped_right"))
                    check["stopped_early_0.5"] = (round(counts["early"] / counts["episodes"], 4), stored.get("episodes_stopped_early"))
                    if "tuned_two_stop_threshold" in stored:
                        tuned = stored["tuned_two_stop_threshold"]
                        tuned_counts = measures["stop_grid"][f"t{tuned}_k2"]
                        check["tuned_two_right"] = (round(tuned_counts["right"] / tuned_counts["episodes"], 4), stored.get("tuned_two_episodes_stopped_right"))
                        measures["tuned_two_counts"] = {key: tuned_counts[key] for key in ("right", "early", "never", "between")}
                        measures["tuned_two_threshold"] = tuned
                    if "tuned_stop_threshold" in stored:
                        tuned = stored["tuned_stop_threshold"]
                        tuned_counts = measures["stop_grid"][f"t{tuned}_k1"]
                        check["tuned_one_right"] = (round(tuned_counts["right"] / tuned_counts["episodes"], 4), stored.get("tuned_episodes_stopped_right"))
                        measures["tuned_one_counts"] = {key: tuned_counts[key] for key in ("right", "early", "never", "between")}
                        measures["tuned_one_threshold"] = tuned
                result["stored_check"][f"{run_name}_{sheet}"] = check
            result["folds"][run_name][sheet] = measures
            print(run_name, sheet, "scored", flush = True)

    for run_name in ("final_candidate", "finetune_exact", "finetune_chunk", "finetune_classes", "finetune_mirror", "small_cnn_classes"):
        fold_grids = {sheet: measures["stop_grid"] for sheet, measures in result["folds"][run_name].items() if "stop_grid" in measures}
        if not fold_grids:
            continue
        pooled = pooled_grid(fold_grids, parameters)
        result["pooled"][run_name] = {"sheets": sorted(fold_grids), "grid": pooled}
        tuned_totals = {}
        for label in ("tuned_one_counts", "tuned_two_counts"):
            folds_with = [measures[label] for measures in result["folds"][run_name].values() if label in measures]
            if folds_with:
                tuned_totals[label] = {key: sum(fold[key] for fold in folds_with) for key in ("right", "early", "never", "between")}
        result["pooled"][run_name]["tuned"] = tuned_totals
        if len(fold_grids) >= 3:
            result["leave_one_sheet_out"][run_name] = leave_one_sheet_out(fold_grids, parameters)

    for first_run, first_key, second_run, second_key in PAIRED_COMPARISONS:
        label = f"{first_run}.{first_key} minus {second_run}.{second_key}"
        result["paired"][label] = {}
        for sheet in sorted(set(joined_by_run[first_run]) & set(joined_by_run[second_run])):
            first_predictions, first_joined = joined_by_run[first_run][sheet]
            second_predictions, second_joined = joined_by_run[second_run][sheet]
            if f"step_error_{first_key}" not in first_joined or f"step_error_{second_key}" not in second_joined:
                continue
            shared = numpy.intersect1d(first_joined["frame_index"].to_numpy(), second_joined["frame_index"].to_numpy())
            first_part = first_joined[first_joined["frame_index"].isin(shared)].set_index("frame_index").loc[shared]
            second_part = second_joined[second_joined["frame_index"].isin(shared)].set_index("frame_index").loc[shared]
            # the masks come from the first run's own row order; a shared row keeps the first run's path direction
            moving = first_part["is_moving"].to_numpy() & second_part["is_moving"].to_numpy()
            path_rows = first_part["is_path_row"].to_numpy() & second_part["is_path_row"].to_numpy()
            episodes = first_part["episode_id"].to_numpy()
            entry = {}
            # one target for both runs: the demonstrated step, and the first run's path direction
            shared_path_angle = first_part["path_angle_deg"].to_numpy()
            errors = {
                "step": (first_part[f"step_error_{first_key}"].to_numpy(), second_part[f"step_error_{second_key}"].to_numpy(), moving),
                "path": (angle_difference_deg(shared_path_angle, first_part[f"predicted_{first_key}"].to_numpy()),
                         angle_difference_deg(shared_path_angle, second_part[f"predicted_{second_key}"].to_numpy()), path_rows),
            }
            for measure, (first_error, second_error, mask) in errors.items():
                first_hits = episode_hits(first_error, mask, episodes, parameters)
                second_hits = episode_hits(second_error, mask, episodes, parameters)
                entry[measure] = bootstrap_difference(first_hits, second_hits, parameters)
            entry["shared_rows"] = int(len(shared))
            result["paired"][label][sheet] = entry

    parameters_for_json = {key: (str(value) if isinstance(value, Path) else value) for key, value in parameters.items()}
    result["parameters"] = parameters_for_json
    parameters["output_json_path"].write_text(json.dumps(result, indent = 1), encoding = "utf-8")

    lines.append("STORED CHECK (recomputed, stored)")
    for name, check in result["stored_check"].items():
        lines.append(f"  {name}: " + "; ".join(f"{key} {value[0]} / {value[1]}" for key, value in check.items()))
    lines.append("")
    lines.append("PER FOLD (moving rows, model within 45, path rows, model path 45, always forward 45 and path 45, backward rows and within 45, other moves within 45)")
    for run_name, folds in result["folds"].items():
        for sheet, measures in folds.items():
            line = (f"  {run_name} {sheet}: moving {measures['moving_rows']}, within45 {measures['angle_deg_within_45']}, path rows {measures['path_rows']}, "
                    f"path45 {measures['angle_deg_path_within_45']}, forward {measures['always_forward_within_45']} / {measures['always_forward_path_within_45']}, "
                    f"backward {measures['backward_rows']} at {measures['angle_deg_backward_within_45']}, other {measures['angle_deg_other_moves_within_45']}")
            if "angle_expected_deg_within_45" in measures:
                line += (f"; expected: within45 {measures['angle_expected_deg_within_45']}, path45 {measures['angle_expected_deg_path_within_45']}, "
                         f"backward {measures['angle_expected_deg_backward_within_45']}")
            if "angle_vector_deg_within_45" in measures:
                line += f"; vector: within45 {measures['angle_vector_deg_within_45']}, path45 {measures['angle_vector_deg_path_within_45']}"
            if "twist_gain_through_zero" in measures:
                line += (f"; twist gain {measures['twist_gain_through_zero']}, sign {measures['twist_sign_on_twisted_rows']} of {measures['twist_twisted_rows']}, "
                         f"sign where predicted >= 3 deg {measures['twist_sign_where_predicted_3_deg_or_more']} of {measures['twist_rows_predicted_3_deg_or_more']}")
            lines.append(line)
    lines.append("")
    lines.append("POOLED STOP OUTCOMES (episodes: right / early / never / between 2 and 3 cm)")
    for run_name, pooled in result["pooled"].items():
        grid = pooled["grid"]
        lines.append(f"  {run_name} over sheets {pooled['sheets']}:")
        for setting in ("t0.5_k1", "t0.5_k2", "t0.3_k1", "t0.3_k2", "t0.4_k1", "t0.4_k2", "t0.6_k1", "t0.7_k1"):
            counts = grid[setting]
            lines.append(f"    fixed {setting}: {counts['right']} / {counts['early']} / {counts['never']} / {counts['between']} of {counts['episodes']}")
        for label, counts in pooled["tuned"].items():
            lines.append(f"    {label}: {counts['right']} / {counts['early']} / {counts['never']} / {counts['between']}")
        best = sorted(grid.items(), key = lambda item: (-item[1]["right"], item[1]["early"]))[:3]
        lines.append("    best fixed settings with hindsight: " + "; ".join(f"{setting} {counts['right']} / {counts['early']} / {counts['never']}" for setting, counts in best))
        if run_name in result["leave_one_sheet_out"]:
            for rule, entry in result["leave_one_sheet_out"][run_name].items():
                total = entry["total"]
                lines.append(f"    leave one sheet out, {rule}: {total['right']} / {total['early']} / {total['never']} / {total['between']} of {total['episodes']} "
                             f"(right Wilson {total['right_wilson']}); thresholds " + ", ".join(f"{sheet} {fold['threshold']}" for sheet, fold in entry["per_fold"].items()))
    lines.append("")
    lines.append("PER-FOLD STOP COUNTS, final candidate (right / early / never / between)")
    for sheet, measures in result["folds"]["final_candidate"].items():
        grid = measures["stop_grid"]
        parts = [f"{setting} {grid[setting]['right']}/{grid[setting]['early']}/{grid[setting]['never']}/{grid[setting]['between']}" for setting in ("t0.5_k1", "t0.5_k2", "t0.3_k1", "t0.3_k2")]
        parts.append(f"tuned one ({measures.get('tuned_one_threshold')}) " + "/".join(str(measures.get('tuned_one_counts', {}).get(key)) for key in ("right", "early", "never", "between")))
        parts.append(f"tuned two ({measures.get('tuned_two_threshold')}) " + "/".join(str(measures.get('tuned_two_counts', {}).get(key)) for key in ("right", "early", "never", "between")))
        lines.append(f"  {sheet} ({measures['test_episodes']} episodes): " + "; ".join(parts))
    lines.append("")
    lines.append("PAIRED DIFFERENCES ON SHARED ROWS (points, 95 % episode bootstrap)")
    for label, sheets in result["paired"].items():
        for sheet, entry in sheets.items():
            lines.append(f"  {label}, sheet {sheet}: step {entry['step']['difference_points']} {entry['step']['interval_points']} on {entry['step']['rows']} rows; "
                         f"path {entry['path']['difference_points']} {entry['path']['interval_points']} on {entry['path']['rows']} rows")
    parameters["output_text_path"].write_text("\n".join(lines), encoding = "utf-8")
    print("\n".join(lines))



if __name__ == "__main__":
    main()
