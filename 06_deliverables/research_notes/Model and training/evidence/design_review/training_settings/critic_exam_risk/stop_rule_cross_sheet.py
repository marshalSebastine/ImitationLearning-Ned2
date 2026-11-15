"""
Choose the stop rule without looking at the sheet it is used on (read only).

For each of the five held-out sheets with an exact-head end-to-end model (final candidate: 01, 02; finetune_exact:
03, 05, 06), the rule (threshold, pictures in a row) is chosen on the OTHER four sheets' saved predictions, by a
criterion, and then counted on the held-out sheet. Compared with the rule tuned on the inner validation episodes (the
section's rule, read from the fold files). Two criteria: right minus early (an early stop costs as much as a missed
right stop) and right minus twice early (an early stop ends the run, a late one may not). Run with python -B.
"""

import json
from pathlib import Path

import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FRAMES_PATH = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")
SHEET_RUNS = {"01": "2026-11-09_resnet18_final_candidate", "02": "2026-11-09_resnet18_final_candidate",
              "03": "2026-11-09_resnet18_finetune_exact", "05": "2026-11-09_resnet18_finetune_exact",
              "06": "2026-11-09_resnet18_finetune_exact"}
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RULES = [(threshold, consecutive) for threshold in THRESHOLDS for consecutive in (1, 2, 3)]


def outcomes(stop_probability, goal_distance_cm, episode_ids, threshold, consecutive_pictures):
    """Return numpy array (right, early, never) of episode counts."""

    right = early = never = 0
    for episode_id in numpy.unique(episode_ids):
        rows = numpy.flatnonzero(episode_ids == episode_id)
        is_passing = stop_probability[rows] >= threshold
        run_length = numpy.zeros(len(rows), dtype = int)
        for position in range(len(rows)):
            run_length[position] = ((run_length[position - 1] + 1) if position > 0 else 1) if is_passing[position] else 0
        passing = rows[run_length >= consecutive_pictures]
        if len(passing) == 0:
            never += 1
        elif goal_distance_cm[passing[0]] <= 2.0:
            right += 1
        elif goal_distance_cm[passing[0]] > 3.0:
            early += 1
    return numpy.array([right, early, never])


def main():
    frames = pandas.read_csv(FRAMES_PATH, low_memory = False, usecols = ["frame_index", "episode_id", "goal_distance_cm"]).set_index("frame_index")
    table = {}
    tuned = {}
    for sheet, run_folder in SHEET_RUNS.items():
        predictions = dict(numpy.load(RUNS_FOLDER / run_folder / f"predictions_without_path_{sheet}.npz"))
        rows = frames.loc[predictions["frame_index"]]
        stop_probability = predictions["stop_probability"].astype(float)
        goal_distance = rows["goal_distance_cm"].fillna(0.0).to_numpy()
        episode_ids = rows["episode_id"].to_numpy()
        table[sheet] = {rule: outcomes(stop_probability, goal_distance, episode_ids, *rule) for rule in RULES}
        fold_file = json.loads((RUNS_FOLDER / run_folder / f"fold_without_path_{sheet}.json").read_text(encoding = "utf-8"))["measures"]
        tuned[sheet] = outcomes(stop_probability, goal_distance, episode_ids, fold_file["tuned_two_stop_threshold"], 2)
    for label, early_cost in (("right minus early", 1.0), ("right minus twice early", 2.0)):
        total = numpy.zeros(3, dtype = int)
        print(f"\n== criterion: {label}")
        for sheet in SHEET_RUNS:
            others = [other for other in SHEET_RUNS if other != sheet]
            scores = {rule: sum(table[other][rule][0] - early_cost * table[other][rule][1] for other in others) for rule in RULES}
            best_rule = max(RULES, key = lambda rule: (scores[rule], -rule[1], rule[0]))
            result = table[sheet][best_rule]
            total += result
            print(f"   sheet {sheet}: rule chosen on the other sheets {best_rule}: right {result[0]} early {result[1]} never {result[2]}"
                  f" | inner-validation rule: right {tuned[sheet][0]} early {tuned[sheet][1]} never {tuned[sheet][2]}")
        tuned_total = sum(tuned.values())
        print(f"   pooled over 165 episodes: chosen on other sheets right {total[0]} early {total[1]} never {total[2]}"
              f" | inner-validation rule right {tuned_total[0]} early {tuned_total[1]} never {tuned_total[2]}")


if __name__ == "__main__":
    main()
