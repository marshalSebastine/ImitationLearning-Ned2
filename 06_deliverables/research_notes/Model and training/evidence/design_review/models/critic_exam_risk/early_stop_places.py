"""
List where the deployed stop rule (fold's tuned threshold, two pictures in a row) first fires in the episodes it stops
too early, from the saved test predictions (read only). Writes early_stop_places.out.csv next to this file.
"""

from pathlib import Path

import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FRAMES_PATH = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")
OUTPUT_PATH = Path(__file__).with_name("early_stop_places.out.csv")
FOLDS = {
    "final_01": (RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_01.npz", 0.4),
    "final_02": (RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_02.npz", 0.1),
    "exact_03": (RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_03.npz", 0.15),
    "exact_05": (RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_05.npz", 0.3),
    "exact_06": (RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_06.npz", 0.15),
}



def main():
    columns = ["frame_index", "episode_id", "episode_folder", "row_in_episode", "step_fraction", "steps_to_end",
               "goal_distance_cm", "forearm_turn_deg", "start_end", "start_kind", "brightness_mean", "frame_path",
               "tip_x_m", "tip_y_m"]
    frames = pandas.read_csv(FRAMES_PATH, usecols = columns, low_memory = False)
    rows_out = []
    for name, (path, threshold) in FOLDS.items():
        predictions = dict(numpy.load(path))
        index = pandas.DataFrame({"frame_index": predictions["frame_index"], "order": numpy.arange(len(predictions["frame_index"])),
                                  "stop_probability": predictions["stop_probability"]})
        joined = index.merge(frames, on = "frame_index", how = "left").sort_values("order").reset_index(drop = True)
        for episode_id, group in joined.groupby("episode_id", sort = False):
            passing = (group["stop_probability"] >= threshold).to_numpy()
            run_length = numpy.zeros(len(group), dtype = int)
            for position in range(len(group)):
                run_length[position] = (run_length[position - 1] + 1 if position > 0 else 1) if passing[position] else 0
            hits = numpy.flatnonzero(run_length >= 2)
            if len(hits) == 0:
                continue
            first = group.iloc[hits[0]]
            if float(first["goal_distance_cm"]) > 3.0:
                start = group.iloc[0]
                rows_out.append({
                    "fold": name, "episode_folder": first["episode_folder"], "start_end": first["start_end"],
                    "start_kind": first["start_kind"], "row_in_episode": int(first["row_in_episode"]),
                    "rows_in_test_episode": len(group), "goal_distance_cm": round(float(first["goal_distance_cm"]), 1),
                    "distance_from_start_cm": round(100.0 * float(numpy.hypot(first["tip_x_m"] - start["tip_x_m"], first["tip_y_m"] - start["tip_y_m"])), 1),
                    "forearm_turn_deg": round(float(first["forearm_turn_deg"]), 1),
                    "stop_probability": round(float(first["stop_probability"]), 3),
                    "brightness_mean": round(float(first["brightness_mean"]), 1), "frame_path": first["frame_path"]})
    table = pandas.DataFrame(rows_out)
    table.to_csv(OUTPUT_PATH, index = False)
    with pandas.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 200):
        print(table.drop(columns = ["frame_path"]).to_string())
        print(table[["fold", "frame_path"]].to_string())



if __name__ == "__main__":
    main()
