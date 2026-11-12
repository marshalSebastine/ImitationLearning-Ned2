"""
Verifier, check 3b: the same comparison as v03 with the analyst's wider family (all start-mark episodes of the sheet
with the same corrected end as the run's plan; run 3, 'lying straight', takes every episode of sheet 02), to re-count
'493 of 554 pictures have 5 or more neighbours, 91.7 % of them with the neighbours' median wrist within 5.7 degrees'.
    D:/AI_Project/.venv/Scripts/python.exe v03b_runs_wide_family.py
"""
import math
import numpy
import pandas

RUNS_FOLDER = "D:/AI_Project/03_data/model_runs"
RUNS = [(1, "172021_check_no_puck", "path_01", "01-01"), (2, "172421_E_sheet01_run1", "path_01", "01-01"), (3, "172743_E_sheet02_run1", "path_02", None),
        (4, "173036_E_sheet02_run2", "path_02", "02-01"), (5, "173549_E_sheet03_run1", "path_03", "03-01"), (6, "174108_E_sheet04_run1", "path_04", "04-01"),
        (7, "174448_E_sheet05_run1", "path_05", "05-01"), (8, "174936_E_sheet06_run1", "path_06", "06-01"), (9, "175203_E_sheet03_run2_no_hoodie", "path_03", "03-01"),
        (10, "175544_B_team5_sheet03", "path_03", "03-01")]
frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "tip_x_m", "tip_y_m", "joint_4", "joint_5"])
corrected = pandas.read_csv("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv")
frames = frames.merge(corrected[["episode_id", "plan", "corrected_end"]], on = "episode_id", how = "left")
frames["wrist"] = numpy.degrees(frames["joint_5"])
frames["twist"] = numpy.degrees(frames["joint_4"])
frames["is_start_mark"] = frames["sheet_orientation"].str.contains("START MARK", na = False)
total = with_near = in_range = 0
for number, folder, sheet, plan in RUNS:
    steps = pandas.read_csv(f"{RUNS_FOLDER}/2026-11-11_{folder}/steps.csv")
    part = frames[frames["sheet"] == sheet]
    if plan is not None:
        end = corrected.loc[corrected["plan"] == plan, "corrected_end"].iloc[0]
        part = part[part["is_start_mark"] & (part["corrected_end"] == end)]
    xy = part[["tip_x_m", "tip_y_m"]].to_numpy()
    medians, gaps, outside, shares = [], [], [], []
    for _, step in steps.iterrows():
        near = part[numpy.hypot(xy[:, 0] - step["tip_x"], xy[:, 1] - step["tip_y"]) <= 0.02]
        if len(near) >= 5:
            medians.append((int(step["step"]), near["wrist"].median(), float(((near["wrist"] - 13.5).abs() <= 5.7).mean()), math.degrees(step["joint_4"]) - near["twist"].median(),
                            float(math.degrees(step["joint_4"]) < near["twist"].quantile(0.1) or math.degrees(step["joint_4"]) > near["twist"].quantile(0.9))))
    medians = numpy.array(medians)
    total += len(steps)
    with_near += len(medians)
    in_range += int((numpy.abs(medians[:, 1] - 13.5) <= 5.7).sum())
    first = medians[medians[:, 0] < 13]
    print("run %2d (%s, %d episodes in the family): %d of %d pictures with 5 or more neighbours; neighbours' median wrist within 5.7 of 13.5 in %.0f %%; first 13 pictures: median wrist %.1f to %.1f, "
          "rows in range %.0f to %.0f %%, largest twist gap %.0f degrees, outside the 10th-90th band in %d of %d" % (
              number, sheet, part["episode_id"].nunique(), len(medians), len(steps), 100 * (numpy.abs(medians[:, 1] - 13.5) <= 5.7).mean(), first[:, 1].min(), first[:, 1].max(),
              100 * first[:, 2].min(), 100 * first[:, 2].max(), numpy.abs(first[:, 3]).max(), int(first[:, 4].sum()), len(first)))
print("\nall ten runs: %d of %d pictures with 5 or more neighbours; neighbours' median wrist within 5.7 degrees of 13.5 in %.1f %% of them" % (with_near, total, 100 * in_range / with_near))
