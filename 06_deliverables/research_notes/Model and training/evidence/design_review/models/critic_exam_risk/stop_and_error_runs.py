"""
Exam-risk checks on predictions that already exist (read only; nothing in D:/AI_Project is written).

For the exact-step ResNet-18 folds (final candidate: held-out sheets 01, 02; finetune_exact: 03, 05, 06) and, for the
direction checks, the class models on sheets 03, 05, 06, this script recomputes from the saved test predictions:

1. the stop outcome per episode for thresholds 0.05 to 0.9 and for 1 to 5 pictures in a row (the same rule as
   training_metrics.stop_metrics: first passing picture; right within 2 cm of the demonstrated stop, early more than
   3 cm before it, never when no picture passes);
2. where the early stops happen (distance to the goal, share of the episode's rows already driven);
3. the share of pictures more than 3 cm from the goal whose stop probability passes the threshold;
4. runs of consecutive moving pictures whose predicted direction is more than 45 degrees off the demonstrated step
   and off the 3-step path direction (a proxy for errors that come in streaks, which a closed loop cannot absorb);
5. the signed forearm twist error (a bias would add up over a track).
"""

import json
import math
import sys
from pathlib import Path

import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FRAMES_PATH = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")
OUTPUT_PATH = Path(__file__).with_suffix(".out.json")

DIRECTION_NAMES = ["forward", "forward_left", "left", "back_left", "back", "back_right", "right", "forward_right"]
PICTURE_LEFT_SIGN = 1
RIGHT_STOP_CM = 2.0
EARLY_STOP_CM = 3.0
PATH_STEPS = 3
PATH_MINIMUM_CM = 1.5
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TUNED_TWO_THRESHOLD = {"final_01": 0.4, "final_02": 0.1, "exact_03": 0.15, "exact_05": 0.3, "exact_06": 0.15}

EXACT_FOLDS = {
    "final_01": RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_01.npz",
    "final_02": RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_02.npz",
    "exact_03": RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_03.npz",
    "exact_05": RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_05.npz",
    "exact_06": RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_06.npz",
}
CLASS_FOLDS = {}
for run in ["resnet18_finetune_classes", "small_cnn_scratch_classes", "small_cnn_line_and_puck_classes"]:
    for sheet in ["03", "05", "06"]:
        CLASS_FOLDS[f"{run}_{sheet}"] = RUNS_FOLDER / f"2026-11-09_{run}" / f"predictions_without_path_{sheet}.npz"



def angle_difference_deg(first_deg, second_deg):
    return numpy.abs((numpy.asarray(first_deg) - numpy.asarray(second_deg) + 180.0) % 360.0 - 180.0)



def path_angles(frames):
    """Direction from the tip now to where it is PATH_STEPS rows later, in the current picture's frame."""

    tip_x = frames["tip_x_m"].to_numpy()
    tip_y = frames["tip_y_m"].to_numpy()
    heading = numpy.radians(frames["picture_heading_deg"].to_numpy())
    episode_ids = frames["episode_id"].to_numpy()
    angle = numpy.full(len(frames), numpy.nan)
    length = numpy.zeros(len(frames))
    starts = numpy.flatnonzero(numpy.r_[True, episode_ids[1:] != episode_ids[:-1]])
    ends = numpy.r_[starts[1:], len(frames)]
    for start, end in zip(starts, ends):
        for position in range(start, end):
            future = min(position + PATH_STEPS, end - 1)
            moved_x = tip_x[future] - tip_x[position]
            moved_y = tip_y[future] - tip_y[position]
            forward = moved_x * math.cos(heading[position]) + moved_y * math.sin(heading[position])
            left = (moved_x * math.cos(heading[position] + PICTURE_LEFT_SIGN * math.pi / 2)
                    + moved_y * math.sin(heading[position] + PICTURE_LEFT_SIGN * math.pi / 2))
            angle[position] = math.degrees(math.atan2(left, forward))
            length[position] = 100.0 * math.hypot(forward, left)
    return angle, length



def stop_outcomes(probability, distance_cm, episode_ids, threshold, consecutive):
    right = early = never = between = 0
    early_distances, early_progress = [], []
    episodes = pandas.unique(episode_ids)
    for episode_id in episodes:
        rows = numpy.flatnonzero(episode_ids == episode_id)
        passing = probability[rows] >= threshold
        run_length = numpy.zeros(len(rows), dtype = int)
        for position in range(len(rows)):
            run_length[position] = (run_length[position - 1] + 1 if position > 0 else 1) if passing[position] else 0
        hits = numpy.flatnonzero(run_length >= consecutive)
        if len(hits) == 0:
            never += 1
            continue
        distance = float(distance_cm[rows[hits[0]]])
        if distance <= RIGHT_STOP_CM:
            right += 1
        elif distance > EARLY_STOP_CM:
            early += 1
            early_distances.append(distance)
            early_progress.append(float(hits[0] + 1) / len(rows))
        else:
            between += 1
    return {"episodes": len(episodes), "right": right, "early": early, "never": never, "between_2_and_3_cm": between,
            "early_distance_cm": early_distances, "early_share_of_rows_driven": early_progress}



def error_runs(is_bad, episode_ids):
    """Lengths of runs of consecutive True values of is_bad within each episode."""

    runs = []
    longest_per_episode = []
    for episode_id in pandas.unique(episode_ids):
        values = is_bad[episode_ids == episode_id]
        current = longest = 0
        for value in values:
            if value:
                current += 1
                longest = max(longest, current)
            else:
                if current > 0:
                    runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)
        longest_per_episode.append(longest)
    runs = numpy.array(runs if runs else [0])
    longest_per_episode = numpy.array(longest_per_episode)
    return {
        "bad_share": round(float(is_bad.mean()), 4),
        "run_count": int((runs > 0).sum()),
        "run_length_median": float(numpy.median(runs)),
        "run_length_90th": float(numpy.percentile(runs, 90)),
        "run_length_max": int(runs.max()),
        "episodes": int(len(longest_per_episode)),
        "episodes_with_run_3_or_more": int((longest_per_episode >= 3).sum()),
        "episodes_with_run_5_or_more": int((longest_per_episode >= 5).sum()),
        "longest_run_median_per_episode": float(numpy.median(longest_per_episode)),
    }



def direction_checks(predictions, joined):
    is_moving = joined["action"].isin(DIRECTION_NAMES).to_numpy()
    episode_ids = joined["episode_id"].to_numpy()
    predicted = predictions["angle_deg"]
    step_error = angle_difference_deg(joined["move_angle_deg"].to_numpy(), predicted)
    path_error = angle_difference_deg(joined["path_angle_deg"].to_numpy(), predicted)
    is_path_row = (joined["action"].to_numpy() != "stop") & (joined["path_length_cm"].to_numpy() >= PATH_MINIMUM_CM)
    forward_step_error = angle_difference_deg(joined["move_angle_deg"].to_numpy(), 0.0)
    forward_path_error = angle_difference_deg(joined["path_angle_deg"].to_numpy(), 0.0)
    result = {}
    for name, error, mask in (("step", step_error, is_moving), ("path", path_error, is_path_row)):
        result[f"{name}_model"] = error_runs(error[mask] > 45.0, episode_ids[mask])
        result[f"{name}_model"]["over_90_share"] = round(float((error[mask] > 90.0).mean()), 4)
    for name, error, mask in (("step", forward_step_error, is_moving), ("path", forward_path_error, is_path_row)):
        result[f"{name}_always_forward"] = error_runs(error[mask] > 45.0, episode_ids[mask])
        result[f"{name}_always_forward"]["over_90_share"] = round(float((error[mask] > 90.0).mean()), 4)
    return result



def main():
    columns = ["frame_index", "episode_id", "row_in_episode", "action", "goal_distance_cm", "move_angle_deg",
               "forearm_change_deg", "tip_x_m", "tip_y_m", "picture_heading_deg"]
    frames = pandas.read_csv(FRAMES_PATH, usecols = columns, low_memory = False)
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    result = {"exact_folds": {}, "class_folds": {}}

    for name, path in list(EXACT_FOLDS.items()) + list(CLASS_FOLDS.items()):
        if not path.exists():
            result.setdefault("missing", []).append(str(path))
            continue
        predictions = dict(numpy.load(path))
        index = pandas.DataFrame({"frame_index": predictions["frame_index"], "order": numpy.arange(len(predictions["frame_index"]))})
        # the path direction must come from the test rows' own episode order, as in training_probes.path_directions
        joined = index.merge(frames, on = "frame_index", how = "left").sort_values("order").reset_index(drop = True)
        path_angle, path_length = path_angles(joined)
        joined["path_angle_deg"] = path_angle
        joined["path_length_cm"] = path_length
        fold = {"test_rows": int(len(joined)), "episodes": int(joined["episode_id"].nunique()),
                "direction": direction_checks(predictions, joined)}

        if name in EXACT_FOLDS:
            probability = predictions["stop_probability"]
            distance = joined["goal_distance_cm"].fillna(0.0).to_numpy()
            episode_ids = joined["episode_id"].to_numpy()
            grid = {}
            for threshold in THRESHOLDS:
                for consecutive in range(1, 6):
                    outcome = stop_outcomes(probability, distance, episode_ids, threshold, consecutive)
                    grid[f"t{threshold}_k{consecutive}"] = {key: outcome[key] for key in ("right", "early", "never", "between_2_and_3_cm")}
            fold["stop_grid"] = grid
            tuned = TUNED_TWO_THRESHOLD[name]
            fold["tuned_two_threshold"] = tuned
            fold["at_tuned_threshold"] = {}
            for consecutive in range(1, 6):
                outcome = stop_outcomes(probability, distance, episode_ids, tuned, consecutive)
                outcome["early_distance_cm"] = [round(value, 1) for value in outcome["early_distance_cm"]]
                outcome["early_share_of_rows_driven"] = [round(value, 2) for value in outcome["early_share_of_rows_driven"]]
                fold["at_tuned_threshold"][f"k{consecutive}"] = outcome
            far = distance > EARLY_STOP_CM
            fold["far_pictures"] = int(far.sum())
            fold["far_pictures_passing_tuned_threshold"] = int((probability[far] >= tuned).sum())
            fold["far_pictures_passing_0.5"] = int((probability[far] >= 0.5).sum())
            best = max(grid.items(), key = lambda item: (item[1]["right"], -item[1]["early"]))
            fold["best_on_test_sheet_itself"] = {"setting": best[0], **best[1]}
            # forearm twist: signed error and how it adds up per episode (open loop, on the demonstrated pictures)
            true_twist = joined["forearm_change_deg"].fillna(0.0).clip(-15.0, 15.0).to_numpy()
            predicted_twist = predictions["twist_deg"]
            signed = predicted_twist - true_twist
            per_episode_sum = pandas.Series(signed).groupby(episode_ids).sum()
            per_episode_true = pandas.Series(numpy.abs(true_twist)).groupby(episode_ids).sum()
            fold["twist"] = {
                "signed_error_mean_deg": round(float(signed.mean()), 3),
                "predicted_mean_deg": round(float(predicted_twist.mean()), 3),
                "true_mean_deg": round(float(true_twist.mean()), 3),
                "episode_sum_of_signed_error_median_deg": round(float(per_episode_sum.median()), 1),
                "episode_sum_of_signed_error_abs_90th_deg": round(float(per_episode_sum.abs().quantile(0.9)), 1),
                "episode_sum_of_abs_true_twist_median_deg": round(float(per_episode_true.median()), 1),
            }
            result["exact_folds"][name] = fold
        else:
            result["class_folds"][name] = fold
        print(name, "done", flush = True)

    # pooled stop outcomes over the five exact-step folds for each setting
    pooled = {}
    for setting in result["exact_folds"]["final_01"]["stop_grid"]:
        total = {"right": 0, "early": 0, "never": 0, "between_2_and_3_cm": 0}
        for fold in result["exact_folds"].values():
            for key in total:
                total[key] += fold["stop_grid"][setting][key]
        pooled[setting] = total
    result["pooled_stop_grid"] = pooled
    best = sorted(pooled.items(), key = lambda item: (-item[1]["right"], item[1]["early"]))[:8]
    result["pooled_best_settings"] = best
    OUTPUT_PATH.write_text(json.dumps(result, indent = 1), encoding = "utf-8")
    print("written", OUTPUT_PATH)



if __name__ == "__main__":
    main()
