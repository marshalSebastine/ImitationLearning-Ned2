"""
Robot-session critic: three stall rules on the ten logged runs of 11.11.26 (as stall_rule_on_the_ten_runs.py, with a
third rule that keeps healthy demonstrations alive under short steps, see stall_rule_detail.out.txt):
    the plan's   15 pictures, 4 cm, 15 degrees
    today's      30 pictures, 3 cm, 15 degrees
    middle       25 pictures, 4 cm, 15 degrees
Reads steps.csv only. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe stall_rule_variants_on_the_ten_runs.py > stall_rule_variants_on_the_ten_runs.out.txt
"""

from pathlib import Path

import numpy
import pandas

RUN_ROOT = Path("D:/AI_Project/03_data/model_runs")
RULES = {"plan 15/4cm/15deg": (15, 4.0, 15.0), "today 30/3cm/15deg": (30, 3.0, 15.0), "middle 25/4cm/15deg": (25, 4.0, 15.0)}


def first_stall(position, turn, pictures, extent_cm, turn_deg):
    for last in range(pictures - 1, len(position)):
        window = position[last - pictures + 1:last + 1]
        extent = numpy.hypot(window[:, 0].max() - window[:, 0].min(), window[:, 1].max() - window[:, 1].min())
        turned = turn[last - pictures + 1:last + 1].max() - turn[last - pictures + 1:last + 1].min()
        if extent < extent_cm and turned < turn_deg:
            return last
    return None


print("picture at which each rule would end the run (- = never): as driven (1 cm steps) | own length, open loop | fixed 0.65 cm, open loop")
for number, folder in enumerate(sorted(RUN_ROOT.glob("2026-11-11_17*")), start = 1):
    steps = pandas.read_csv(folder / "steps.csv")
    moving = steps[steps["action"] != "stop"]
    driven_position = 100.0 * moving[["tip_x", "tip_y"]].to_numpy()
    driven_turn = numpy.degrees(moving["joint_4"].to_numpy())
    heading = numpy.radians(moving["picture_heading_deg"].to_numpy())
    forward, left = moving["model_tip_forward_cm"].to_numpy(), moving["model_tip_left_cm"].to_numpy()
    vector = numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])
    length = numpy.linalg.norm(vector, axis = 1)
    unit = vector / numpy.maximum(length, 1e-9)[:, None]
    twist = numpy.clip(moving["model_twist_deg"].to_numpy(), -10.0, 10.0)
    turn = numpy.r_[0.0, numpy.cumsum(twist)][:-1]
    own = numpy.vstack([numpy.zeros((1, 2)), numpy.cumsum(unit * numpy.minimum(length, 1.0)[:, None], axis = 0)])[:-1]
    fixed = numpy.vstack([numpy.zeros((1, 2)), numpy.cumsum(unit * 0.65, axis = 0)])[:-1]
    cells = []
    for rule_name, (pictures, extent_cm, turn_deg) in RULES.items():
        results = [first_stall(position, angle, pictures, extent_cm, turn_deg) for position, angle in ((driven_position, driven_turn), (own, turn), (fixed, turn))]
        cells.append(f"{rule_name}: " + " | ".join("-" if value is None else str(value) for value in results))
    print(f"{number:2d} {folder.name[18:]:26s} ({len(steps):2d} pictures)   " + "     ".join(cells))
