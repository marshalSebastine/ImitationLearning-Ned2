"""
Verifier: how far the tip went past the puck's start place along the first leg, measured in EACH EPISODE'S OWN FRAME
(no laying of demonstrations onto the run, so independent of the analyst's registration).

The puck positions are computed here again with 02_code/puck_tracking.PuckTracker on every picture of the first
stretch of each episode (the project's own tool; pictures and joints only, nothing connects to the robot).
  p0  = the puck's first position
  d   = the direction from p0 to the puck's position once it has moved 2 cm (the first leg)
  overshoot measure = the largest (tip - p0) . d while the tip travels its first 16 cm
Also printed: the tip's start distance from the puck, and the puck's own largest advance along d in that stretch.

Run from this folder:  python v05_overshoot_own_frame.py > v05_overshoot_own_frame.out.txt
"""

import os
import sys

import cv2
import numpy
import pandas

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
from puck_tracking import PuckTracker

cv2.setNumThreads(2)
INDEX = "D:/AI_Project/03_data/processed/v01_training_index/frames.csv"
RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = {5: "2026-11-11_173549_E_sheet03_run1", 9: "2026-11-11_175203_E_sheet03_run2_no_hoodie", 10: "2026-11-11_175544_B_team5_sheet03",
               6: "2026-11-11_174108_E_sheet04_run1"}
PLANS = {"path_03": ["03-01", "03-02", "03-03", "03-04", "03-05", "03-25", "03-26", "03-27"],
         "path_04": ["04-01", "04-02", "04-03", "04-04", "04-05", "04-13", "04-14", "04-15", "04-16", "04-17", "04-18"]}
JOINTS = [f"joint_{number}" for number in range(1, 7)]
TRAVEL_CM = 16.0
pandas.set_option("display.width", 250)


def measures(name, picture_paths, joints, tip_cm, joint_4_deg):
    moved = numpy.linalg.norm(numpy.diff(tip_cm, axis = 0), axis = 1)
    travelled = numpy.r_[0.0, numpy.cumsum(moved)]
    count = int(numpy.searchsorted(travelled, TRAVEL_CM, side = "right"))
    count = min(max(count, 2), len(tip_cm))
    tracker = PuckTracker()
    puck = numpy.full((count, 2), numpy.nan)
    for index in range(count):
        picture = cv2.imread(picture_paths[index])
        seen = tracker.see_puck(picture, list(joints[index]))
        if seen is not None and seen["table_xy"] is not None:
            puck[index] = 100.0 * numpy.asarray(seen["table_xy"])
    valid = numpy.flatnonzero(numpy.isfinite(puck[:, 0]))
    p0 = puck[valid[0]]
    puck_moved = numpy.linalg.norm(puck - p0, axis = 1)
    far = numpy.flatnonzero(puck_moved >= 2.0)
    if not len(far):
        return {"name": name, "note": "puck never moved 2 cm"}
    direction = puck[far[0]] - p0
    direction = direction / numpy.linalg.norm(direction)
    along_tip = (tip_cm[:count] - p0) @ direction
    along_puck = (puck - p0) @ direction
    turned = numpy.abs(joint_4_deg[:count] - joint_4_deg[0])
    return {"name": name, "rows_used": count, "puck_start_x": p0[0], "puck_start_y": p0[1],
            "first_leg_deg": float(numpy.degrees(numpy.arctan2(direction[1], direction[0]))),
            "tip_start_to_puck_cm": float(numpy.linalg.norm(tip_cm[0] - p0)),
            "tip_start_along_leg": float(along_tip[0]),
            "tip_furthest_along_leg": float(numpy.nanmax(along_tip)),
            "row_of_furthest": int(numpy.nanargmax(along_tip)),
            "joint_4_turn_at_furthest": float(turned[int(numpy.nanargmax(along_tip))]),
            "puck_furthest_along_leg": float(numpy.nanmax(along_puck))}


frames = pandas.read_csv(INDEX, low_memory = False, usecols = ["episode_id", "row_in_episode", "plan_number", "frame_path", "tip_x_m", "tip_y_m"] + JOINTS)
frames = frames.sort_values(["episode_id", "row_in_episode"])
for sheet, plans in PLANS.items():
    rows = []
    for plan in plans:
        group = frames[frames["plan_number"] == plan]
        rows.append(measures(f"demonstration {plan}", [os.path.join(RAW, name) for name in group["frame_path"]], group[JOINTS].to_numpy(dtype = float),
                             100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy(), numpy.degrees(group["joint_4"].to_numpy())))
    demonstrations = pandas.DataFrame(rows)
    run_rows = []
    for number, folder in RUN_FOLDERS.items():
        if (sheet == "path_04") != (number == 6):
            continue
        steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
        run_rows.append(measures(f"robot run {number}", [os.path.join(RUNS, folder, name) for name in steps["frame"]], steps[JOINTS].to_numpy(dtype = float),
                                 100.0 * steps[["tip_x", "tip_y"]].to_numpy(), numpy.degrees(steps["joint_4"].to_numpy())))
    runs = pandas.DataFrame(run_rows)
    print(f"=== {sheet}")
    print(pandas.concat([demonstrations, runs]).round(2).to_string(index = False))
    middle = demonstrations["tip_furthest_along_leg"].median()
    print(f"demonstrations: tip furthest along the first leg median {middle:.2f} cm ({demonstrations['tip_furthest_along_leg'].min():.2f} to {demonstrations['tip_furthest_along_leg'].max():.2f}); "
          f"puck furthest median {demonstrations['puck_furthest_along_leg'].median():.2f} ({demonstrations['puck_furthest_along_leg'].min():.2f} to {demonstrations['puck_furthest_along_leg'].max():.2f})")
    for _, row in runs.iterrows():
        print(f"{row['name']}: tip furthest {row['tip_furthest_along_leg']:.2f} cm, overshoot against the demonstrations' median {row['tip_furthest_along_leg'] - middle:+.2f} cm; "
              f"puck furthest {row['puck_furthest_along_leg']:.2f} cm against {demonstrations['puck_furthest_along_leg'].median():.2f}")
    print()
