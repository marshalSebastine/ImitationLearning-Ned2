"""
Proofreader sitting_a, check 9: do the driver's guards let a run finish the sheets of sitting A where they were recorded?

Part A5 allows one full run of sheet 04 and the risk table promises "one dry run to a sheet's far end in sitting A",
which the table of sitting A does not hold. On paper instead: every recorded step of the standard-start demonstrations
of sheets 01, 03 and 04 from end A (the lay-outs sitting A tapes) is put through model_step from its recorded joints,
with the wrist and hand held as the driver holds them. A refused step means the driver would have ended a run there
("refused"), whatever the model does. Printed per episode: refused steps, where the first one lies, the largest reach
and the largest planned lean.

Arm model and frames.csv only. Run:
    D:/AI_Project/.venv/Scripts/python.exe -B p09_guards_on_recorded_paths.py > p09_guards_on_recorded_paths.out.txt
"""

import math
import sys
from pathlib import Path

import numpy
import pandas

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
sys.path.insert(0, str(CODE))

from config import MODEL_MODE_PARAMETERS      # noqa: E402
from model_mode import held_joints_at, model_step   # noqa: E402
from side_view_test import magnet_tip_and_lean      # noqa: E402

INDEX = Path("D:/AI_Project/03_data/processed/v01_training_index")
columns = ["episode_id", "row_in_episode", "tip_forward_m", "tip_left_m", "forearm_change_deg", "tip_x_m", "tip_y_m"] + [f"joint_{index}" for index in range(1, 7)]
frames = pandas.read_csv(INDEX / "frames.csv", usecols = columns, low_memory = False)
PLANS = {"01": ["01-01", "01-02", "01-03", "01-04"], "03": ["03-01", "03-02", "03-03", "03-04", "03-05"], "04": ["04-01", "04-02", "04-03", "04-04", "04-05"]}
settings = dict(MODEL_MODE_PARAMETERS, step_length_rule = "predicted")
print(f"guards: reach {[100 * value for value in settings['tip_reach_m']]} cm, x {[100 * value for value in settings['workspace_x_m']]} cm, y {[100 * value for value in settings['workspace_y_m']]} cm, "
      f"lean at most {settings['rod_lean_most_deg']} degrees, forearm within {settings['joint_4_most_deg']} degrees, wrist held at {settings['wrist_held_deg']}")
for sheet, plans in PLANS.items():
    print(f"\nsheet {sheet}, end A, standard starts:")
    for plan in plans:
        episode_id = next(name for name in frames["episode_id"].unique() if f"plan_{plan}_" in name)
        rows = frames[frames["episode_id"] == episode_id].sort_values("row_in_episode").reset_index(drop = True)
        refused, leans, held_out, first = [], [], 0, None
        for position in range(len(rows) - 1):
            row = rows.iloc[position]
            joints_recorded = [float(row[f"joint_{index}"]) for index in range(1, 7)]
            # the driver would stand at the same tip point with the wrist and hand held
            joints_held = held_joints_at(joints_recorded, 0.0, settings)
            if joints_held is None:
                held_out += 1
                first = first if first is not None else (position, "the recorded tip point cannot be reached with the wrist held")
                continue
            decision = {"tip_forward_cm": 100.0 * (0.0 if pandas.isna(row["tip_forward_m"]) else row["tip_forward_m"]),
                        "tip_left_cm": 100.0 * (0.0 if pandas.isna(row["tip_left_m"]) else row["tip_left_m"]),
                        "twist_deg": 0.0 if pandas.isna(row["forearm_change_deg"]) else float(row["forearm_change_deg"])}
            planned = model_step(joints_held, decision, None, settings)
            if planned["refusal"]:
                refused.append(position)
                first = first if first is not None else (position, planned["refusal"])
            elif planned["lean_deg"] is not None:
                leans.append(planned["lean_deg"])
        reach = 100.0 * numpy.hypot(rows["tip_x_m"], rows["tip_y_m"])
        text = "none refused" if not refused and not held_out else f"{len(refused)} steps refused, {held_out} tip points out of reach with the wrist held; first at picture {first[0]} of {len(rows)} ({first[1]})"
        print(f"   {plan}: {len(rows)} pictures; largest reach {reach.max():.1f} cm, tip x up to {100 * rows['tip_x_m'].max():.1f} cm, y {100 * rows['tip_y_m'].min():.1f} to {100 * rows['tip_y_m'].max():.1f} cm; "
              f"largest planned lean {max(leans) if leans else float('nan'):.1f} deg; {text}")
