"""
Direction accuracy of the exact-head and class-head end-to-end models on the held-out pictures recorded after a hand
correction (the correction replays, the closest thing to recovery states in the data) against the other pictures
(read only). A row counts as after a correction when an earlier row of its episode restarted the waypoint count
(waypoint 0 after row 0). Run with python -B.
"""

from pathlib import Path

import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FRAMES_PATH = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")
DIRECTION_NAMES = ["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"]
RUNS = {
    "final candidate (exact)": ("2026-11-09_resnet18_final_candidate", ["01", "02"]),
    "finetune_exact": ("2026-11-09_resnet18_finetune_exact", ["03", "05", "06"]),
    "finetune_classes": ("2026-11-09_resnet18_finetune_classes", ["03", "05", "06"]),
}


def angle_difference_deg(first, second):
    return numpy.abs((first - second + 180.0) % 360.0 - 180.0)


def main():
    frames = pandas.read_csv(FRAMES_PATH, low_memory = False, usecols = [
        "frame_index", "episode_id", "row_in_episode", "waypoint", "action", "move_angle_deg"])
    frames = frames.sort_values(["episode_id", "row_in_episode"])
    is_restart = (frames["waypoint"].fillna(-1) == 0) & (frames["row_in_episode"] > 0)
    frames["is_after_correction"] = is_restart.groupby(frames["episode_id"]).cumsum() > 0
    frames = frames.set_index("frame_index")
    for label, (run_folder, sheets) in RUNS.items():
        pooled = {True: [], False: []}
        back_pooled = {True: [0, 0], False: [0, 0]}
        for sheet in sheets:
            predictions = dict(numpy.load(RUNS_FOLDER / run_folder / f"predictions_without_path_{sheet}.npz"))
            rows = frames.loc[predictions["frame_index"]]
            is_moving = rows["action"].isin(DIRECTION_NAMES).to_numpy()
            after = rows["is_after_correction"].to_numpy()
            errors = angle_difference_deg(rows["move_angle_deg"].to_numpy(), predictions["angle_deg"])
            true_angle = rows["move_angle_deg"].to_numpy()
            is_backward = numpy.abs(true_angle) > 112.5
            predicted_backward = numpy.abs(predictions["angle_deg"]) > 112.5
            parts = []
            for flag in (True, False):
                selection = is_moving & (after == flag)
                pooled[flag].append(errors[selection])
                backward = selection & is_backward
                back_pooled[flag][0] += int((backward & predicted_backward).sum())
                back_pooled[flag][1] += int(backward.sum())
                parts.append(f"{'after correction' if flag else 'other'}: {selection.sum()} moving, within 45 {numpy.mean(errors[selection] <= 45):.3f}")
            print(f"{label} sheet {sheet}: " + " | ".join(parts))
        for flag in (True, False):
            forward_errors = []
            for sheet in sheets:
                predictions = dict(numpy.load(RUNS_FOLDER / run_folder / f"predictions_without_path_{sheet}.npz"))
                rows = frames.loc[predictions["frame_index"]]
                selection = rows["action"].isin(DIRECTION_NAMES).to_numpy() & (rows["is_after_correction"].to_numpy() == flag)
                forward_errors.append(numpy.abs(rows["move_angle_deg"].to_numpy()[selection]))
            forward_errors = numpy.concatenate(forward_errors)
            print(f"   always-forward on the same pictures ({'after correction' if flag else 'other'}): within 45 {numpy.mean(forward_errors <= 45):.3f}")
            errors = numpy.concatenate(pooled[flag])
            print(f"   {label} pooled, {'after correction' if flag else 'other'}: {len(errors)} moving pictures, within 45 {numpy.mean(errors <= 45):.3f};"
                  f" backward moves (any back component) recalled as backward {back_pooled[flag][0]} of {back_pooled[flag][1]}")


if __name__ == "__main__":
    main()
