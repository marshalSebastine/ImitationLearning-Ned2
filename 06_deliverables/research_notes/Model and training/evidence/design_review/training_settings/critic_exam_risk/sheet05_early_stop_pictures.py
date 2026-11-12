"""
List where the exact model (resnet18_finetune_exact, held-out sheet 05) first passes the stop threshold 0.5 in the
episodes it stops too early, with the picture file, so a few can be looked at (read only). Run with python -B.
"""

from pathlib import Path

import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FRAMES_PATH = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")


def main():
    frames = pandas.read_csv(FRAMES_PATH, low_memory = False, usecols = [
        "frame_index", "episode_id", "episode_folder", "row_in_episode", "step_fraction", "goal_distance_cm", "frame_path",
        "tip_x_m", "tip_y_m", "brightness_mean", "lighting"]).set_index("frame_index")
    for run_folder, sheet in (("2026-11-09_resnet18_finetune_exact", "05"), ("2026-11-09_resnet18_final_candidate", "02")):
        predictions = dict(numpy.load(RUNS_FOLDER / run_folder / f"predictions_without_path_{sheet}.npz"))
        rows = frames.loc[predictions["frame_index"]].copy()
        rows["stop_probability"] = predictions["stop_probability"]
        print(f"\n== {run_folder}, sheet {sheet}")
        for episode_id, episode in rows.groupby("episode_id", sort = False):
            passing = episode[episode["stop_probability"] >= 0.5]
            if len(passing) == 0:
                continue
            first = passing.iloc[0]
            if first["goal_distance_cm"] > 3.0:
                print(f"   {first['episode_folder'][-48:]} row {int(first['row_in_episode'])} of {len(episode)}"
                      f" | goal {first['goal_distance_cm']:.1f} cm | done {first['step_fraction']:.2f} | p {first['stop_probability']:.2f}"
                      f" | tip ({first['tip_x_m']:.3f}, {first['tip_y_m']:.3f}) | brightness {first['brightness_mean']:.0f} {first['lighting']}"
                      f" | {first['frame_path']}")


if __name__ == "__main__":
    main()
