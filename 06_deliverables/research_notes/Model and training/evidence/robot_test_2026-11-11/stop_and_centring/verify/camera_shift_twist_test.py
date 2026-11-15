"""
Verifier's check 15 (11.11.26): did the camera's view move between the two periods (before and after the evening of
8.11.26)? A test that does not use the line at all: twists in place.

In many demonstrations the forearm (joint 4) turns by 20 degrees or more while the magnet tip stays where it is
(within 2.5 mm); the puck then stands still on the table while the camera swings round the forearm's axis. With the
right camera model my fitted puck foot lands on the same table place in the first and the last picture of such a
twist; with a wrong principal point it wanders, the more the further the forearm turns. Per period the wander is
measured with the camera as fitted on 6.11.26, with the period's own correction of the analyst and with the other
period's. One twist per episode (its largest). Reads frames.csv and the raw pictures only.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy
import pandas

import my_measure

cv2.setNumThreads(2)
HERE = Path(__file__).resolve().parent
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
CAMERA_MOVED_AT = "2026-11-08T19:00:00"
# python camera_shift_twist_test.py [tip stays within, mm] [forearm turn at least, degrees]
TIP_STAYS_M = (float(sys.argv[1]) if len(sys.argv) > 1 else 2.5) / 1000.0
TURN_LEAST_DEG = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False)
shifts = json.loads((HERE.parent / "camera_shift.json").read_text(encoding = "utf-8"))
cameras = {"fitted": {"before_move": None, "after_move": None},
           "own": {period: my_measure.shifted_camera(shifts[period]["shift_px"], shifts[period]["roll_deg"]) for period in ("before_move", "after_move")}}
cameras["other"] = {"before_move": cameras["own"]["after_move"], "after_move": cameras["own"]["before_move"]}
# the shift alone and the roll alone of the period's own correction, to see which of the two does the work
cameras["own shift only"] = {period: my_measure.shifted_camera(shifts[period]["shift_px"], 0.0) for period in ("before_move", "after_move")}
cameras["own roll only"] = {period: my_measure.shifted_camera((0.0, 0.0), shifts[period]["roll_deg"]) for period in ("before_move", "after_move")}

rows = []
for episode_id, steps in frames.groupby("episode_id", sort = False):
    steps = steps.sort_values("row_in_episode").reset_index(drop = True)
    tips = steps[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    turn = numpy.degrees(steps["joint_4"].to_numpy(dtype = float))
    best = None
    for first in range(len(steps) - 1):
        last = first
        while last + 1 < len(steps) and float(numpy.linalg.norm(tips[last + 1] - tips[first])) < TIP_STAYS_M:
            last += 1
        if last > first and abs(turn[last] - turn[first]) >= TURN_LEAST_DEG and (best is None or abs(turn[last] - turn[first]) > abs(best[2])):
            best = (first, last, float(turn[last] - turn[first]))
    if best is None:
        continue
    first, last, turned = best
    period = "after_move" if str(steps["started"].iloc[0]) >= CAMERA_MOVED_AT else "before_move"
    pictures = [cv2.imread(str(RAW_FOLDER / steps["frame_path"].iloc[index])) for index in (first, last)]
    joints = [[steps[f"joint_{joint}"].iloc[index] for joint in range(1, 7)] for index in (first, last)]
    entry = {"episode_id": episode_id, "sheet": steps["sheet"].iloc[0], "period": period, "first": first, "last": last, "turned_deg": turned}
    is_complete = True
    for name, by_period in cameras.items():
        places = [my_measure.puck_foot(picture, joint_set, by_period[period]) for picture, joint_set in zip(pictures, joints)]
        if any(place is None or place["residual_px"] > 4.0 for place in places):
            is_complete = False
            break
        entry[f"wander_{name}_mm"] = 1000.0 * float(numpy.linalg.norm(places[1]["xy"] - places[0]["xy"]))
    if is_complete:
        rows.append(entry)
table = pandas.DataFrame(rows)
table.to_csv(HERE / "camera_shift_twist_test.csv", index = False, float_format = "%.2f")
print(f"twists in place of {TURN_LEAST_DEG:.0f} degrees or more with the tip within {1000 * TIP_STAYS_M:.1f} mm, one per episode, puck fitted in both pictures: {len(table)} episodes")
for period, part in table.groupby("period"):
    print(f"{period}: {len(part)} twists, forearm turned by a median {part['turned_deg'].abs().median():.0f} degrees")
    for name in cameras:
        print(f"  camera '{name}': the standing puck wanders by a median {part[f'wander_{name}_mm'].median():.1f} mm (75th percentile {part[f'wander_{name}_mm'].quantile(0.75):.1f} mm)")
    print(f"  own correction better than as fitted in {int((part['wander_own_mm'] < part['wander_fitted_mm'] - 0.3).sum())}, worse in {int((part['wander_own_mm'] > part['wander_fitted_mm'] + 0.3).sum())};"
          f" own better than the other period's in {int((part['wander_own_mm'] < part['wander_other_mm'] - 0.3).sum())}, worse in {int((part['wander_own_mm'] > part['wander_other_mm'] + 0.3).sum())}")
