"""
Verifier, check 12: small facts read from the ten runs' steps.csv (measured by the driver's log).
    D:/AI_Project/.venv/Scripts/python.exe v12_run_logs.py
"""
import glob
import math
import numpy
import pandas

rows = []
for number, folder in enumerate(sorted(glob.glob("D:/AI_Project/03_data/model_runs/2026-11-11_17*")), start = 1):
    steps = pandas.read_csv(f"{folder}/steps.csv")
    reach = 100 * numpy.hypot(steps["tip_x"], steps["tip_y"])
    lean = steps["lean_deg"] if "lean_deg" in steps else pandas.Series(dtype = float)
    rows.append({"run": number, "pictures": len(steps), "wrist_least": math.degrees(steps["joint_5"].min()), "wrist_most": math.degrees(steps["joint_5"].max()),
                 "hand_least": math.degrees(steps["joint_6"].min()), "hand_most": math.degrees(steps["joint_6"].max()),
                 "lean_most": lean.max(), "reach_at_lean_most": reach[lean.idxmax()] if lean.notna().any() else numpy.nan, "lean_first": lean.iloc[0], "reach_most": reach.max(),
                 "model_lean_sum": steps["model_lean_deg"].sum(), "model_hand_least": steps["model_hand_deg"].min(), "model_hand_most": steps["model_hand_deg"].max(),
                 "model_hand_sum": steps["model_hand_deg"].sum(), "twist_first": math.degrees(steps["joint_4"].iloc[0]), "twist_least": math.degrees(steps["joint_4"].min()),
                 "twist_most": math.degrees(steps["joint_4"].max()), "refusals": int(steps["refusal"].notna().sum()), "last_action": steps["action"].iloc[-1]})
pandas.set_option("display.width", 260)
pandas.set_option("display.max_columns", 40)
print(pandas.DataFrame(rows).round(2).to_string(index = False))
