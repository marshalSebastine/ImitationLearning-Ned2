"""
Count what the training rows look like with and without the audit's cleaning (read only; prints to standard output).

Uses the worktree's own loader (training_targets.load_training_frames), so the counts are exactly the rows the runs
train on. Run with python -B so that no bytecode is written next to the project's code.
"""

import sys

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

import numpy
import pandas

from config import TRAINING_TARGET_PARAMETERS
from training_targets import class_weights, direction_targets, load_training_frames


def describe(label, frames):
    """Print row counts by action, sheet, score and stop share."""

    print(f"\n=== {label}: {len(frames)} rows in {frames['episode_id'].nunique()} episodes")
    actions = frames["action"].value_counts()
    print("actions:", ", ".join(f"{name} {count}" for name, count in actions.items()))
    stop_rows = int((frames["action"] == "stop").sum())
    print(f"stop rows {stop_rows} = {100.0 * stop_rows / len(frames):.2f} % of rows; negatives per stop row {(len(frames) - stop_rows) / stop_rows:.1f}")
    per_sheet = frames.groupby("sheet").agg(rows = ("episode_id", "size"), episodes = ("episode_id", "nunique"))
    per_sheet["share_of_rows"] = (100.0 * per_sheet["rows"] / len(frames)).round(1)
    per_sheet["share_of_episodes"] = (100.0 * per_sheet["episodes"] / frames["episode_id"].nunique()).round(1)
    print(per_sheet.to_string())
    print("scores:", frames.groupby("score")["episode_id"].agg(["size", "nunique"]).to_string().replace("\n", " | "))
    moving = frames["action"].isin(["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"])
    forward_share = (frames.loc[moving, "action"] == "forward").mean()
    back_share = frames.loc[moving, "action"].isin(["back", "back_left", "back_right"]).mean()
    print(f"moving rows {int(moving.sum())}; forward {100 * forward_share:.1f} %, any back class {100 * back_share:.1f} % of moving rows")
    labels, class_names, is_used = direction_targets(frames, "next_move")
    used = numpy.where(is_used, labels, -1)
    weights = class_weights(used, len(class_names), 0.5)
    print("square-root class weights (next_move rule):", ", ".join(f"{name} {weight:.2f}" for name, weight in zip(class_names, weights)))
    lighting = frames.groupby("lighting")["episode_id"].agg(["size", "nunique"])
    print("lighting note:", lighting.to_string().replace("\n", " | "))
    small = frames["action"].isin(["small_motion", "turn_left", "turn_right"])
    print(f"small_motion and turn rows {int(small.sum())}")
    twist = frames["forearm_change_deg"].abs()
    print(f"forearm change: median {twist.median():.2f} deg, p90 {twist.quantile(0.9):.2f}, max {twist.max():.2f}")
    length_cm = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    print(f"step length over moving rows: median {length_cm[moving].median():.2f} cm, p99 {length_cm[moving].quantile(0.99):.2f}, max {length_cm.max():.2f}")


def main():
    default_parameters = dict(TRAINING_TARGET_PARAMETERS)
    describe("worktree default (before the audit cleaning)", load_training_frames(default_parameters))
    cleaned_parameters = dict(TRAINING_TARGET_PARAMETERS)
    cleaned_parameters["is_audit_cleaning"] = True
    cleaned = load_training_frames(cleaned_parameters)
    describe("audit cleaning (the final candidate's rows)", cleaned)
    print(f"\nrelabelled stop rows: {int(cleaned['is_relabelled_stop'].sum())}")
    # how many rows per epoch each sheet contributes when the five other sheets train, and what equal weight per
    # sheet would mean
    counts = cleaned["sheet"].value_counts().sort_index()
    for held_out in counts.index:
        training = counts.drop(held_out)
        print(f"held out {held_out}: training rows per sheet " + ", ".join(f"{sheet} {100.0 * count / training.sum():.1f} %" for sheet, count in training.items()))


if __name__ == "__main__":
    main()
