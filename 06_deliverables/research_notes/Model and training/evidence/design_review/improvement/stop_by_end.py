"""
Does the stop fail more where the line ends far from the paper edge? (read only)

For the end-to-end exact run (held-out sheets 03, 05, 06) and the frozen exact probe (all six held-out sheets), every
held-out episode is scored as in training_metrics.stop_metrics (first picture, or first two pictures in a row, at or
above the stop threshold; right within 2 cm of where the demonstration stopped, early more than 3 cm before, never)
and grouped by the end the puck finishes at. The finishing end is the other end from the corrected start end of the
audit's table (06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv); the distance from each
line end to the paper edge behind it is the audit's table 10.1. Nothing is written outside this script's folder.

Usage: python stop_by_end.py
"""

import json
import sys
from pathlib import Path

CODE_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/02_code"
sys.path.insert(0, CODE_FOLDER)

import numpy
import pandas

from training_targets import load_training_frames

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")

PARAMETERS = {
    "corrected_ends": Path("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv"),
    "edge_distance_cm": {("path_01", "A"): 3.7, ("path_01", "B"): 3.0, ("path_02", "A"): 6.2, ("path_02", "B"): 6.4,
                         ("path_03", "A"): 3.6, ("path_03", "B"): 2.5, ("path_04", "A"): 2.4, ("path_04", "B"): 2.8,
                         ("path_05", "A"): 5.9, ("path_05", "B"): 6.0, ("path_06", "A"): 4.4, ("path_06", "B"): 8.0},
    "right_stop_cm": 2.0,
    "early_stop_cm": 3.0,
    "fixed_threshold": 0.5,
    "output_folder": Path(__file__).resolve().parent,
}


def outcome_of(stop_probability, goal_distance_cm, threshold, consecutive_pictures, settings):
    """
    The stop outcome of one episode, rows in step order.
    """
    is_predicted = stop_probability >= threshold
    run_length = numpy.zeros(len(is_predicted), dtype = int)
    for position in range(len(is_predicted)):
        run_length[position] = (run_length[position - 1] + 1 if position > 0 else 1) if is_predicted[position] else 0
    passing = numpy.flatnonzero(run_length >= consecutive_pictures)
    if len(passing) == 0:
        return "never", numpy.nan
    distance = float(goal_distance_cm[passing[0]])
    if distance <= settings["right_stop_cm"]:
        return "right", distance
    if distance > settings["early_stop_cm"]:
        return "early", distance
    return "near miss", distance


def run_predictions(kind):
    """
    Return (fold name, prediction file, tuned two-in-a-row threshold) for a run.
    """
    if kind == "end_to_end_exact":
        folder = RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact"
        items = []
        for fold in ("without_path_03", "without_path_05", "without_path_06"):
            with open(folder / f"fold_{fold}.json", "r", encoding = "utf-8") as handle:
                threshold = json.load(handle)["measures"]["tuned_two_stop_threshold"]
            items.append((fold, folder / f"predictions_{fold}.npz", threshold))
        return items
    with open(RUNS_FOLDER / "2026-11-09_frozen_feature_probes" / "designs.json", "r", encoding = "utf-8") as handle:
        results = json.load(handle)["results"]
    result = [entry for entry in results if entry["name"] == "exact_step"][0]
    items = []
    for fold in result["folds"]:
        items.append((fold["fold"], RUNS_FOLDER / "2026-11-09_frozen_feature_probes" / "predictions" / f"exact_step_{fold['fold']}.npz",
                      fold["tuned_two_stop_threshold"]))
    return items


def main(parameters = None):
    """
    Print the stop outcomes per finishing end.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    frames = load_training_frames()
    corrected = pandas.read_csv(settings["corrected_ends"]).set_index("episode_id")["corrected_end"]
    frames["start_end_corrected"] = frames["episode_id"].map(corrected).fillna(frames["start_end"])
    frames["finish_end"] = frames["start_end_corrected"].map({"A": "B", "B": "A"})
    position_of = pandas.Series(numpy.arange(len(frames)), index = frames["frame_index"].to_numpy())
    all_rows = []
    for kind in ("end_to_end_exact", "probe_exact_step"):
        for fold, path, tuned_threshold in run_predictions(kind):
            predictions = numpy.load(path)
            positions = position_of.reindex(predictions["frame_index"]).to_numpy()
            is_known = ~numpy.isnan(positions)
            positions = positions[is_known].astype(int)
            probability = predictions["stop_probability"][is_known]
            part = frames.iloc[positions][["episode_id", "sheet", "row_in_episode", "goal_distance_cm", "finish_end", "start_kind"]].copy()
            part["stop_probability"] = probability
            for episode_id, group in part.groupby("episode_id"):
                group = group.sort_values("row_in_episode")
                fixed, fixed_distance = outcome_of(group["stop_probability"].to_numpy(), group["goal_distance_cm"].to_numpy(),
                                                   settings["fixed_threshold"], 1, settings)
                tuned, tuned_distance = outcome_of(group["stop_probability"].to_numpy(), group["goal_distance_cm"].to_numpy(),
                                                   tuned_threshold, 2, settings)
                sheet = group["sheet"].iloc[0]
                finish_end = group["finish_end"].iloc[0]
                all_rows.append({"kind": kind, "fold": fold, "episode_id": episode_id, "sheet": sheet,
                                 "finish_end": finish_end, "start_kind": group["start_kind"].iloc[0],
                                 "edge_distance_cm": settings["edge_distance_cm"].get((sheet, finish_end), numpy.nan),
                                 "fixed_outcome": fixed, "tuned_two_outcome": tuned,
                                 "tuned_two_threshold": tuned_threshold})
    table = pandas.DataFrame(all_rows)
    table.to_csv(settings["output_folder"] / "stop_by_end_episodes.csv", index = False)
    for kind, kind_table in table.groupby("kind"):
        print(f"=== {kind} ===")
        for (sheet, finish_end), group in kind_table.groupby(["sheet", "finish_end"]):
            fixed_counts = group["fixed_outcome"].value_counts().to_dict()
            tuned_counts = group["tuned_two_outcome"].value_counts().to_dict()
            print(f"{sheet} finishing at end {finish_end} (line end {group['edge_distance_cm'].iloc[0]} cm from the paper edge): "
                  f"{len(group)} episodes; fixed 0.5 {fixed_counts}; tuned two in a row {tuned_counts}")
        groups = kind_table.groupby(["sheet", "finish_end"]).agg(
            edge = ("edge_distance_cm", "first"),
            episodes = ("episode_id", "count"),
            never_fixed = ("fixed_outcome", lambda values: float((values == "never").mean())),
            right_fixed = ("fixed_outcome", lambda values: float((values == "right").mean())),
            never_tuned = ("tuned_two_outcome", lambda values: float((values == "never").mean())),
            right_tuned = ("tuned_two_outcome", lambda values: float((values == "right").mean())))
        print(groups.round(3).to_string())
        if len(groups) >= 4:
            ranks = groups[["edge", "never_fixed", "right_fixed", "never_tuned", "right_tuned"]].rank()
            print("Spearman rank correlation with the edge distance (groups):",
                  {column: round(float(numpy.corrcoef(ranks["edge"], ranks[column])[0, 1]), 3)
                   for column in ["never_fixed", "right_fixed", "never_tuned", "right_tuned"]})
        far = kind_table["edge_distance_cm"] >= 5.0
        for label, mask in (("line end 5 cm or more from the edge", far), ("line end under 5 cm from the edge", ~far)):
            part = kind_table[mask]
            print(f"{label}: {len(part)} episodes; fixed never {float((part['fixed_outcome'] == 'never').mean()):.3f}, "
                  f"right {float((part['fixed_outcome'] == 'right').mean()):.3f}; tuned two never "
                  f"{float((part['tuned_two_outcome'] == 'never').mean()):.3f}, right {float((part['tuned_two_outcome'] == 'right').mean()):.3f}")
        print()


if __name__ == "__main__":
    main()
