"""
Plan review, key "evidence": details behind e02. For the sampled sheet-03 demonstrations and for the robot runs on
sheet 03 (runs 5, 9, 10), print the stop-and-centring analyst's per-picture offset of the puck from the line's middle
over the first pictures, so that the plan's block A rule ("within 12 mm of the line's middle to 10 cm past the bend")
and its "lost" rule (more than 12 mm outside the ink = more than 17 mm from the middle, two pictures in a row) can be
read against what is already measured. Reads CSV files only; nothing is changed.
"""

import numpy
import pandas

pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 400)
folder = "D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/stop_and_centring/"

print("===== demonstrations of sheet 03 (sampled six), first 40 pictures: offset_mm (12-picture measure), single_offset_mm, flags =====")
table = pandas.read_csv(folder + "demonstration_centring_03.csv")
for episode_id, rows in table.groupby("episode_id", sort = False):
    rows = rows.sort_values("step").reset_index(drop = True)
    travel_cm = 100.0 * numpy.concatenate([[0.0], numpy.cumsum(numpy.hypot(numpy.diff(rows["tip_x"]), numpy.diff(rows["tip_y"])))])
    rows["travel_cm"] = travel_cm.round(1)
    print()
    print(episode_id.split("/")[-1], "| start kind", rows["start_kind"].iloc[0], "| hand corrections", int(rows["hand_corrections"].iloc[0]), "| period", rows["camera_period"].iloc[0])
    part = rows[rows["travel_cm"] <= 20.0]
    print(part[["step", "travel_cm", "offset_mm", "is_clean", "is_aligned", "single_offset_mm", "is_single_clean", "action", "forearm_change_deg"]].round(1).to_string(index = False))

print()
print("===== robot runs: offsets over the first 40 pictures =====")
runs = pandas.read_csv(folder + "robot_runs_steps.csv")
for run_number in (5, 9, 10, 6, 3, 4, 7, 8, 1, 2):
    rows = runs[runs["run"] == run_number].sort_values("step").reset_index(drop = True)
    sizes = rows["offset_mm"].abs()
    over_12 = (sizes > 12.0).to_numpy()
    over_17 = (sizes > 17.0).to_numpy()
    first_two_12 = next((int(rows["step"].iloc[index]) for index in range(1, len(rows)) if over_12[index] and over_12[index - 1]), None)
    first_two_17 = next((int(rows["step"].iloc[index]) for index in range(1, len(rows)) if over_17[index] and over_17[index - 1]), None)
    print(f"run {run_number:2d}: pictures {len(rows)}, measured {int(sizes.notna().sum())}, within 7 mm {100 * (sizes <= 7).sum() / max(1, sizes.notna().sum()):.0f} %, "
          f"first two in a row over 12 mm from the middle at picture {first_two_12}, over 17 mm (lost by the plan's rule) at picture {first_two_17}")
for run_number in (5, 9, 10):
    rows = runs[runs["run"] == run_number].sort_values("step").reset_index(drop = True)
    print()
    print("run", run_number)
    print(rows[rows["step"] <= 40][["step", "offset_mm", "is_clean", "is_aligned", "single_offset_mm", "is_single_clean"]].round(1).to_string(index = False))
