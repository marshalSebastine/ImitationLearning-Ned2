"""
Stop outcome per held-out episode under the deployed rule (fold's tuned threshold, two pictures in a row), split by the
end the run started from, so that the goal's distance to the paper edge can be compared (read only).
"""

from pathlib import Path

import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FRAMES_PATH = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")
OUTPUT_PATH = Path(__file__).with_name("stop_outcome_by_end.out.csv")
FOLDS = {
    "final_01": (RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_01.npz", 0.4),
    "final_02": (RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_02.npz", 0.1),
    "exact_03": (RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_03.npz", 0.15),
    "exact_05": (RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_05.npz", 0.3),
    "exact_06": (RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_06.npz", 0.15),
}
# the audit's table 10.1: distance from each line end to the paper edge behind it, in cm
END_TO_EDGE_CM = {("01", "A"): 3.7, ("01", "B"): 3.0, ("02", "A"): 6.2, ("02", "B"): 6.4, ("03", "A"): 3.6, ("03", "B"): 2.5,
                  ("05", "A"): 5.9, ("05", "B"): 6.0, ("06", "A"): 4.4, ("06", "B"): 8.0}



def main():
    columns = ["frame_index", "episode_id", "start_end", "goal_distance_cm"]
    frames = pandas.read_csv(FRAMES_PATH, usecols = columns, low_memory = False)
    corrected = pandas.read_csv("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv")
    corrected = corrected.set_index("episode_id")["corrected_end"]
    frames["start_end"] = frames["episode_id"].map(corrected).fillna(frames["start_end"])
    rows_out = []
    for name, (path, threshold) in FOLDS.items():
        sheet = name[-2:]
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
                outcome = "never"
            else:
                distance = float(group["goal_distance_cm"].fillna(0.0).iloc[hits[0]])
                outcome = "right" if distance <= 2.0 else ("early" if distance > 3.0 else "between")
            start_end = str(group["start_end"].iloc[0])
            goal_end = {"A": "B", "B": "A"}.get(start_end, "?")
            rows_out.append({"fold": name, "episode_id": episode_id, "start_end": start_end, "goal_end": goal_end,
                             "goal_end_to_paper_edge_cm": END_TO_EDGE_CM.get((sheet, goal_end)), "outcome": outcome,
                             "last_stop_probability": round(float(group["stop_probability"].iloc[-1]), 3),
                             "max_stop_probability_last_3": round(float(group["stop_probability"].iloc[-3:].max()), 3)})
    table = pandas.DataFrame(rows_out)
    table.to_csv(OUTPUT_PATH, index = False)
    summary = table.groupby(["fold", "goal_end", "goal_end_to_paper_edge_cm", "outcome"]).size().unstack(fill_value = 0)
    print(summary.to_string())
    print()
    print(table.groupby(["fold", "goal_end"])["max_stop_probability_last_3"].median().to_string())



if __name__ == "__main__":
    main()
