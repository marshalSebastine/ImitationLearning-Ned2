"""
Verifier, check 8: false stops on real pictures by the wrist AT THAT ROW. Saved predictions of five fold-model sets;
rows more than 5 cm from the goal in start-mark episodes; a false stop = stop chance of 0.5 or more.
    D:/AI_Project/.venv/Scripts/python.exe v08_false_stops_by_wrist.py
"""
import sys
import numpy
import pandas
sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
from training_targets import load_training_frames

RUNS_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014"
RUNS = {"E": "2026-11-10_resnet18_chunk_stop_weight_25", "E2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2", "A": "2026-11-09_resnet18_final_candidate",
        "A2": "2026-11-09_resnet18_final_candidate_seed_2", "C": "2026-11-10_resnet18_final_candidate_chunk"}
pandas.set_option("display.width", 250)
frames = load_training_frames()
frames["offset"] = numpy.degrees(frames["joint_5"]) - 13.5
frames["is_start_mark"] = frames["sheet_orientation"].str.contains("START MARK", na = False)
for label, folder in RUNS.items():
    parts = [pandas.DataFrame({"frame_index": saved["frame_index"], f"stop_{label}": saved["stop_probability"]})
             for saved in (numpy.load(f"{RUNS_FOLDER}/{folder}/predictions_without_path_{n:02d}.npz") for n in range(1, 7))]
    frames = frames.merge(pandas.concat(parts), on = "frame_index", how = "left")
frames = frames.dropna(subset = ["stop_E"])
frames["false_stop"] = numpy.mean([(frames[f"stop_{label}"] >= 0.5) for label in RUNS], axis = 0)
frames["group"] = numpy.where(frames["offset"].abs() <= 5.7, "near", numpy.where(frames["offset"] < 0, "below", "above"))
far = frames[frames["is_start_mark"] & (frames["goal_distance_cm"] > 5.0) & (frames["action"] != "stop")]
print("start-mark rows more than 5 cm from the goal:", len(far))
print("\nfalse-stop share (%), mean over five model sets, by sheet and the row's wrist group (rows in the second table):")
print((100 * far.pivot_table(index = "sheet", columns = "group", values = "false_stop", aggfunc = "mean")).round(2).to_string())
print(far.pivot_table(index = "sheet", columns = "group", values = "false_stop", aggfunc = "size").to_string())
far = far.assign(half = numpy.where(far["step_fraction"] < 0.5, "first half", "second half"))
print("\nsame, split by the half of the episode:")
print((100 * far.pivot_table(index = ["sheet", "half"], columns = "group", values = "false_stop", aggfunc = "mean")).round(2).to_string())
print("\nall sheets:", (100 * far.groupby("group")["false_stop"].mean()).round(2).to_dict())
print("all sheets, second half only:", (100 * far[far["half"] == "second half"].groupby("group")["false_stop"].mean()).round(2).to_dict())
