"""
The verifier's own numbers for the ten robot runs of 11.11.26 in coverage terms (steps.csv only): reach, joint 4,
direction of the step from radial and in the picture, what the model asked for, and how many demonstrated moving
steps lie near each run step (within 10 degrees of joint 4, 15 degrees of direction from radial and 5 cm of reach,
the analyst's rule, recomputed with own code).

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v9_runs_coverage.py > v9_runs_coverage.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from training_targets import load_training_frames  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
ROBOT_RUNS = [(1, "2026-11-11_172021_check_no_puck"), (2, "2026-11-11_172421_E_sheet01_run1"), (3, "2026-11-11_172743_E_sheet02_run1"),
              (4, "2026-11-11_173036_E_sheet02_run2"), (5, "2026-11-11_173549_E_sheet03_run1"), (6, "2026-11-11_174108_E_sheet04_run1"),
              (7, "2026-11-11_174448_E_sheet05_run1"), (8, "2026-11-11_174936_E_sheet06_run1"), (9, "2026-11-11_175203_E_sheet03_run2_no_hoodie"),
              (10, "2026-11-11_175544_B_team5_sheet03")]


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


frames = load_training_frames({"is_audit_cleaning": True})
forward = 100.0 * frames["tip_forward_m"].fillna(0.0).to_numpy()
left = 100.0 * frames["tip_left_m"].fillna(0.0).to_numpy()
is_moving = (numpy.hypot(forward, left) >= 0.3) & (frames["action"] != "stop").to_numpy()
demo_x = 100.0 * frames["tip_x_m"].to_numpy()[is_moving]
demo_y = 100.0 * frames["tip_y_m"].to_numpy()[is_moving]
demo_reach = numpy.hypot(demo_x, demo_y)
demo_joint_4 = numpy.degrees(frames["joint_4"].to_numpy())[is_moving]
demo_direction = wrap(frames["picture_heading_deg"].to_numpy()[is_moving] + numpy.degrees(numpy.arctan2(left[is_moving], forward[is_moving]))
                      - numpy.degrees(numpy.arctan2(demo_y, demo_x)))

rows = []
for number, folder in ROBOT_RUNS:
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    steps = steps[steps["action"] == "model_step"] if "action" in steps else steps
    x = 100.0 * steps["tip_x"].to_numpy()
    y = 100.0 * steps["tip_y"].to_numpy()
    reach = numpy.hypot(x, y)
    joint_4 = numpy.degrees(steps["joint_4"].to_numpy())
    step_table = numpy.degrees(numpy.arctan2(steps["tip_next_y"] - steps["tip_y"], steps["tip_next_x"] - steps["tip_x"]))
    from_radial = wrap(step_table - numpy.degrees(numpy.arctan2(y, x)))
    in_picture = wrap(step_table - steps["picture_heading_deg"].to_numpy())
    asked = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"]).to_numpy()
    neighbours = numpy.array([int(((numpy.abs(demo_joint_4 - j) <= 10.0) & (numpy.abs(wrap(demo_direction - d)) <= 15.0) & (numpy.abs(demo_reach - r) <= 5.0)).sum())
                              for j, d, r in zip(joint_4, from_radial, reach)])
    few = numpy.nonzero(neighbours < 20)[0]
    rows.append({"run": number, "steps": len(steps), "reach_cm": f"{reach.min():.0f} to {reach.max():.0f}", "joint_4_deg": f"{joint_4.min():.0f} to {joint_4.max():.0f}",
                 "from_radial_deg": f"{from_radial.min():.0f} to {from_radial.max():.0f}", "beyond_90_from_radial": int((numpy.abs(from_radial) > 90).sum()),
                 "over_60_off_picture_top": int((numpy.abs(in_picture) > 60).sum()), "neighbours_median": int(numpy.median(neighbours)),
                 "steps_under_20_neighbours": int(len(few)), "first_such_step": int(few[0]) if len(few) else -1,
                 "asked_cm_median": round(float(numpy.median(asked)), 2), "asked_under_0.3_share": round(float((asked < 0.3).mean()), 2),
                 "asked_path_cm": round(float(asked.sum()), 1)})
print(pandas.DataFrame(rows).to_string(index = False))
