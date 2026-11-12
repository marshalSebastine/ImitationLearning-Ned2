"""
Verifier, check 3c: run 7 (sheet 05, the run that finished the whole loop): how different was the demonstrated wrist
at the places it passed? Wide family as in v03b (start-mark episodes of sheet 05 with the plan's corrected end).
    D:/AI_Project/.venv/Scripts/python.exe v03c_run7_band.py
"""
import numpy
import pandas
frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "sheet", "sheet_orientation", "tip_x_m", "tip_y_m", "joint_4", "joint_5"])
corrected = pandas.read_csv("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv")
frames = frames.merge(corrected[["episode_id", "plan", "corrected_end"]], on = "episode_id")
end = corrected.loc[corrected["plan"] == "05-01", "corrected_end"].iloc[0]
part = frames[(frames["sheet"] == "path_05") & frames["sheet_orientation"].str.contains("START MARK", na = False) & (frames["corrected_end"] == end)]
part = part.assign(wrist = numpy.degrees(part["joint_5"]), twist = numpy.degrees(part["joint_4"]))
steps = pandas.read_csv("D:/AI_Project/03_data/model_runs/2026-11-11_174448_E_sheet05_run1/steps.csv")
xy = part[["tip_x_m", "tip_y_m"]].to_numpy()
rows = []
for _, step in steps.iterrows():
    near = part[numpy.hypot(xy[:, 0] - step["tip_x"], xy[:, 1] - step["tip_y"]) <= 0.02]
    alike = near[(near["twist"] - numpy.degrees(step["joint_4"])).abs() <= 20]
    if len(near) >= 5:
        rows.append({"picture": int(step["step"]), "neighbours": len(near), "p10": near["wrist"].quantile(0.1), "median": near["wrist"].median(), "p90": near["wrist"].quantile(0.9),
                     "far_share": float(((near["wrist"] - 13.5).abs() > 5.7).mean()), "alike": len(alike),
                     "alike_far_share": float(((alike["wrist"] - 13.5).abs() > 5.7).mean()) if len(alike) >= 5 else numpy.nan})
table = pandas.DataFrame(rows)
print("run 7: %d of %d pictures with 5 or more neighbours; neighbours' wrist: lowest 10th percentile %.1f, highest 90th percentile %.1f; median of the medians %.1f" % (
    len(table), len(steps), table["p10"].min(), table["p90"].max(), table["median"].median()))
print("pictures where more than a third of the neighbours had a wrist over 5.7 degrees from 13.5: %d; among neighbours twisted alike (within 20 degrees): %d of %d pictures" % (
    int((table["far_share"] > 1 / 3).sum()), int((table["alike_far_share"] > 1 / 3).sum()), int(table["alike_far_share"].notna().sum())))
print(table.iloc[::6].round(1).to_string(index = False))
