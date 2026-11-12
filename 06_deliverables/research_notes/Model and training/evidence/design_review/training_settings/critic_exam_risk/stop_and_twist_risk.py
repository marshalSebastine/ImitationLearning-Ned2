"""
Exam-risk checks on the saved held-out-sheet predictions of the end-to-end runs (read only; nothing is trained).

1. Stop rules. For every held-out sheet the episode outcome (stopped right within 2 cm of the demonstrated stop, too
   early more than 3 cm before it, never) is recounted from the saved stop probabilities for fixed thresholds and for
   1, 2 or 3 pictures in a row, beside the rule tuned on the inner validation episodes (read from the fold files).
   For early stops: how far before the goal and how far along the episode the stop fired.
2. Twist. The predicted forearm change against the demonstrated one: bias, shrinkage (least-squares slope), and the
   open-loop sum over an episode, beside the demonstrated forearm range near the joint-4 limit.

Run with python -B from any folder. Writes nothing except its printout (redirect it).
"""

import json
from pathlib import Path

import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FRAMES_PATH = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")

RUN_FOLDS = {
    "final_candidate (exact, audit cleaning, last weights)": ("2026-11-09_resnet18_final_candidate", ["01", "02"]),
    "finetune_exact (exact, before cleaning)": ("2026-11-09_resnet18_finetune_exact", ["03", "05", "06"]),
    "finetune_classes (class head, square-root weights)": ("2026-11-09_resnet18_finetune_classes", ["03", "05", "06"]),
    "finetune_mirror (class head, half mirrored)": ("2026-11-09_resnet18_finetune_mirror", ["03", "05"]),
}

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RIGHT_STOP_CM = 2.0
EARLY_STOP_CM = 3.0
JOINT_4_LIMIT_DEG = 119.75


def episode_outcomes(
        stop_probability: numpy.ndarray,
        goal_distance_cm: numpy.ndarray,
        episode_ids: numpy.ndarray,
        step_fraction: numpy.ndarray,
        threshold: float,
        consecutive_pictures: int
) -> dict:
    """Return the counts of right, early, between and never, and the goal distance and progress of each early stop."""

    counts = {"right": 0, "early": 0, "between": 0, "never": 0, "episodes": 0}
    early_distances, early_progress = [], []
    for episode_id in numpy.unique(episode_ids):
        rows = numpy.flatnonzero(episode_ids == episode_id)
        counts["episodes"] += 1
        is_passing = stop_probability[rows] >= threshold
        run_length = numpy.zeros(len(rows), dtype = int)
        for position in range(len(rows)):
            run_length[position] = ((run_length[position - 1] + 1) if position > 0 else 1) if is_passing[position] else 0
        passing = rows[run_length >= consecutive_pictures]
        if len(passing) == 0:
            counts["never"] += 1
            continue
        distance = float(goal_distance_cm[passing[0]])
        if distance <= RIGHT_STOP_CM:
            counts["right"] += 1
        elif distance > EARLY_STOP_CM:
            counts["early"] += 1
            early_distances.append(distance)
            early_progress.append(float(step_fraction[passing[0]]))
        else:
            counts["between"] += 1
    counts["early_distance_cm"] = early_distances
    counts["early_progress"] = early_progress
    return counts


def main():
    frames = pandas.read_csv(FRAMES_PATH, low_memory = False, usecols = [
        "frame_index", "episode_id", "sheet", "row_in_episode", "step_fraction", "steps_to_end", "action",
        "goal_distance_cm", "forearm_change_deg", "forearm_turn_deg", "joint_4", "tip_forward_m", "tip_left_m"])
    frames = frames.set_index("frame_index")
    pooled = {}
    twist_rows = []
    for run_label, (run_folder, sheets) in RUN_FOLDS.items():
        print(f"\n######## {run_label}")
        for sheet in sheets:
            fold = f"without_path_{sheet}"
            predictions = dict(numpy.load(RUNS_FOLDER / run_folder / f"predictions_{fold}.npz"))
            fold_file = json.loads((RUNS_FOLDER / run_folder / f"fold_{fold}.json").read_text(encoding = "utf-8"))["measures"]
            rows = frames.loc[predictions["frame_index"]]
            stop_probability = predictions["stop_probability"].astype(float)
            goal_distance = rows["goal_distance_cm"].fillna(0.0).to_numpy()
            episode_ids = rows["episode_id"].to_numpy()
            step_fraction = rows["step_fraction"].fillna(0.0).to_numpy()
            print(f"\n== sheet {sheet}: {len(rows)} test rows, {len(numpy.unique(episode_ids))} episodes")
            check = episode_outcomes(stop_probability, goal_distance, episode_ids, step_fraction, 0.5, 1)
            print(f"   check against fold file at 0.5, one picture: right {check['right']} early {check['early']} never {check['never']}"
                  f" | fold file shares right {fold_file.get('episodes_stopped_right')} early {fold_file.get('episodes_stopped_early')}"
                  f" never {fold_file.get('episodes_never_stopped')}")
            for prefix, consecutive in (("tuned_", 1), ("tuned_two_", 2)):
                threshold = fold_file.get(prefix + "stop_threshold")
                if threshold is None:
                    continue
                result = episode_outcomes(stop_probability, goal_distance, episode_ids, step_fraction, threshold, consecutive)
                print(f"   inner-validation rule {prefix}({threshold}, {consecutive} in a row): right {result['right']} early {result['early']}"
                      f" between {result['between']} never {result['never']} of {result['episodes']}")
                key = (run_label, prefix)
                pooled.setdefault(key, {"right": 0, "early": 0, "between": 0, "never": 0, "episodes": 0})
                for name in ("right", "early", "between", "never", "episodes"):
                    pooled[key][name] += result[name]
            print("   threshold | 1 in a row: right/early/never | 2 in a row | 3 in a row")
            for threshold in THRESHOLDS:
                parts = []
                for consecutive in (1, 2, 3):
                    result = episode_outcomes(stop_probability, goal_distance, episode_ids, step_fraction, threshold, consecutive)
                    parts.append(f"{result['right']:3d}/{result['early']:3d}/{result['never']:3d}")
                    key = (run_label, "fixed", threshold, consecutive)
                    pooled.setdefault(key, {"right": 0, "early": 0, "between": 0, "never": 0, "episodes": 0})
                    for name in ("right", "early", "between", "never", "episodes"):
                        pooled[key][name] += result[name]
                print(f"   {threshold:9.2f} | {parts[0]:>27s} | {parts[1]:>11s} | {parts[2]:>11s}")
            early = episode_outcomes(stop_probability, goal_distance, episode_ids, step_fraction, 0.5, 1)
            if early["early_distance_cm"]:
                distances = numpy.array(early["early_distance_cm"])
                progress = numpy.array(early["early_progress"])
                print(f"   early stops at 0.5, one picture: {len(distances)}; goal distance median {numpy.median(distances):.1f} cm"
                      f" (min {distances.min():.1f}, max {distances.max():.1f}); share of the episode done median {numpy.median(progress):.2f}"
                      f" (min {progress.min():.2f}, max {progress.max():.2f}); within 6 cm of the goal: {(distances <= 6.0).sum()}")
            if "twist_deg" in predictions and run_label.startswith(("final", "finetune_exact")):
                is_not_stop = (rows["action"] != "stop").to_numpy()
                true_twist = rows["forearm_change_deg"].fillna(0.0).clip(-15.0, 15.0).to_numpy()
                predicted_twist = predictions["twist_deg"].astype(float)
                twist_rows.append((run_label, sheet, true_twist[is_not_stop], predicted_twist[is_not_stop], episode_ids[is_not_stop],
                                   rows["forearm_turn_deg"].to_numpy()[is_not_stop]))

    print("\n######## Pooled over the held-out sheets of each run")
    for key, result in pooled.items():
        if key[1] in ("tuned_", "tuned_two_") or (key[1] == "fixed" and key[2] in (0.3, 0.5, 0.7) ):
            print(f"   {key}: right {result['right']} early {result['early']} between {result['between']} never {result['never']} of {result['episodes']}")

    print("\n######## Twist (forearm change per step, degrees), rows that are not the stop row")
    for run_label, sheet, true_twist, predicted_twist, episode_ids, forearm_turn in twist_rows:
        slope = float(numpy.polyfit(true_twist, predicted_twist, 1)[0])
        bias = float((predicted_twist - true_twist).mean())
        ratio = float(numpy.abs(predicted_twist).mean() / max(1e-9, numpy.abs(true_twist).mean()))
        cumulative = []
        for episode_id in numpy.unique(episode_ids):
            is_episode = episode_ids == episode_id
            cumulative.append(float(predicted_twist[is_episode].sum() - true_twist[is_episode].sum()))
        cumulative = numpy.abs(numpy.array(cumulative))
        turn = numpy.abs(forearm_turn[~numpy.isnan(forearm_turn)])
        print(f"   {run_label[:24]} sheet {sheet}: rows {len(true_twist)}; slope of predicted on demonstrated {slope:.2f};"
              f" mean |predicted| / mean |demonstrated| {ratio:.2f}; mean signed error {bias:+.2f} deg per step;"
              f" open-loop sum over an episode, absolute difference median {numpy.median(cumulative):.0f} deg,"
              f" 90th percentile {numpy.percentile(cumulative, 90):.0f}; demonstrated |forearm| max {turn.max():.1f} deg,"
              f" rows beyond 110 deg {(turn > 110.0).sum()}")


if __name__ == "__main__":
    main()
