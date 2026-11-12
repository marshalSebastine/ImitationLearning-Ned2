"""
Review check (reads only): which settings did the ten robot runs of 11.11.26 log, and would the stall rule of the
driver as it is now have ended any of them? The rule is applied to the logged tips and forearm angles exactly as
ModelRun.run applies it (one entry per picture that was not the stopping one).
"""

import glob
import json
import math
from pathlib import Path

import pandas

import safe_import
from safe_import import model_mode
from config import MODEL_MODE_PARAMETERS

folders = sorted(glob.glob("D:/AI_Project/03_data/model_runs/2026-11-11_17*"))
first_summary = json.loads((Path(folders[0]) / "summary.json").read_text(encoding = "utf-8"))
print("settings of today that the robot runs of 11.11.26 did not log (they did not exist yet when the runs were driven):")
print("  ", sorted(set(MODEL_MODE_PARAMETERS) - set(first_summary["parameters"])))
print("summary keys of a run of 11.11.26:", sorted(first_summary))
print()
runner = model_mode.ModelRun(None, None, {})
print("stall rule now:", {name: runner.parameters[name] for name in ("stall_steps", "stall_extent_m", "stall_twist_deg", "stall_action")})
for folder in folders:
    summary = json.loads((Path(folder) / "summary.json").read_text(encoding = "utf-8"))
    steps = pandas.read_csv(Path(folder) / "steps.csv")
    tips_and_twists = []
    fired_at = None
    for _, row in steps.iterrows():
        if row["action"] == "stop":
            break
        tips_and_twists.append((row["tip_x"], row["tip_y"], math.degrees(row["joint_4"])))
        if runner.is_stalled(tips_and_twists):
            fired_at = int(row["step"])
            break
    name = Path(folder).name[18:]
    print(f"{name:28s} ended '{summary['ending']}' after {summary['pictures']:3d} pictures and {summary['moves']:3d} moves; "
          + (f"TODAY'S STALL RULE WOULD HAVE ENDED IT at picture {fired_at} (step numbers count from 0)" if fired_at is not None else "the stall rule would not have fired"))
