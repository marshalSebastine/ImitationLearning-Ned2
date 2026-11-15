"""
Verifier, check 18: is the 'twist 30 to 77 degrees behind' still there when the robot and the demonstrations are
matched by how far the tip has TRAVELLED instead of by the place in the robot's frame? (The test's sheets lay 2 to 5 cm
from the recorded place, see v16.) Demonstrations: start-mark, kind STD, same corrected end as the run's plan. The
demonstrations begin 1 to 2 cm before the tip meets the puck, the runs begin at the puck, so the demonstrated twist is
read at the robot's travel plus 0, 1.5 and 3 cm. Twist is counted from each one's own start value.
    D:/AI_Project/.venv/Scripts/python.exe v18_twist_by_travel.py
"""
import numpy
import pandas

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "tip_x_m", "tip_y_m", "joint_4", "joint_5"])
corrected = pandas.read_csv("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv")
frames = frames.merge(corrected[["episode_id", "plan", "corrected_end", "start_kind"]], on = "episode_id").sort_values(["episode_id", "row_in_episode"])
frames["twist"] = numpy.degrees(frames["joint_4"])
frames["wrist"] = numpy.degrees(frames["joint_5"])
step = numpy.hypot(frames.groupby("episode_id")["tip_x_m"].diff().fillna(0.0), frames.groupby("episode_id")["tip_y_m"].diff().fillna(0.0))
frames["travel_cm"] = 100 * step.groupby(frames["episode_id"]).cumsum()
pandas.set_option("display.width", 220)
for run, folder, sheet, plan in ((5, "173549_E_sheet03_run1", "path_03", "03-01"), (9, "175203_E_sheet03_run2_no_hoodie", "path_03", "03-01"), (10, "175544_B_team5_sheet03", "path_03", "03-01"),
                                 (6, "174108_E_sheet04_run1", "path_04", "04-01"), (8, "174936_E_sheet06_run1", "path_06", "06-01"), (7, "174448_E_sheet05_run1", "path_05", "05-01"),
                                 (4, "173036_E_sheet02_run2", "path_02", "02-01")):
    end = corrected.loc[corrected["plan"] == plan, "corrected_end"].iloc[0]
    family = frames[(frames["sheet"] == sheet) & (frames["corrected_end"] == end) & (frames["start_kind"] == "STD") & frames["sheet_orientation"].str.contains("START MARK", na = False)]
    steps = pandas.read_csv(f"D:/AI_Project/03_data/model_runs/2026-11-11_{folder}/steps.csv")
    robot_travel = 100 * numpy.concatenate([[0.0], numpy.cumsum(numpy.hypot(numpy.diff(steps["tip_x"]), numpy.diff(steps["tip_y"])))])
    robot_twist = numpy.degrees(steps["joint_4"].to_numpy()) - numpy.degrees(steps["joint_4"].iloc[0])
    rows = []
    for picture in (2, 4, 6, 8, 10, 12, 14, 16):
        if picture >= len(steps):
            break
        row = {"picture": picture, "robot_travel_cm": robot_travel[picture], "robot_twist_change": robot_twist[picture]}
        for lead in (0.0, 1.5, 3.0):
            values, wrists = [], []
            for _, episode in family.groupby("episode_id"):
                values.append(numpy.interp(robot_travel[picture] + lead, episode["travel_cm"], episode["twist"] - episode["twist"].iloc[0]))
                wrists.append(numpy.interp(robot_travel[picture] + lead, episode["travel_cm"], episode["wrist"]))
            row[f"demo_twist_lead_{lead}"] = float(numpy.median(values))
            if lead == 1.5:
                row["demo_twist_least"], row["demo_twist_most"], row["demo_wrist_median"] = float(numpy.min(values)), float(numpy.max(values)), float(numpy.median(wrists))
        rows.append(row)
    print(f"\nrun {run} ({sheet}, plan {plan}, {family['episode_id'].nunique()} STD episodes of end {end}):")
    print(pandas.DataFrame(rows).round(1).to_string(index = False))
