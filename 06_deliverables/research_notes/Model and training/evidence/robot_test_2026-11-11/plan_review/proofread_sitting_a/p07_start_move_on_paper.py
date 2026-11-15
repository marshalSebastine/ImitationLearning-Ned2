"""
Proofreader sitting_a, check 7: the new start move on paper (arm model only, nothing moves).

How many recordings begin at the stored start joints; whether today's guards accept that pose; whether the place
5 cm above it can be reached with the wrist and hand held (the "travel 5 cm above, straight down" of step 1 item 2);
how high the block's lower edge then stands over a 2.9 cm puck; and where the tip ends after the runs of 11.11, to
see from where the arm would have to travel back.

Run:  D:/AI_Project/.venv/Scripts/python.exe -B p07_start_move_on_paper.py > p07_start_move_on_paper.out.txt
"""

import math
import sys
from pathlib import Path

import numpy
import pandas

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
sys.path.insert(0, str(CODE))

from config import MODEL_MODE_PARAMETERS, PUCK_PARAMETERS, PUSHER_PARAMETERS, RECORDING_PARAMETERS, pushing_height, table_height   # noqa: E402
from model_mode import held_joints_at, model_step, placed_point_problem                                                              # noqa: E402
from side_view_test import magnet_tip_and_lean                                                                                        # noqa: E402

INDEX = Path("D:/AI_Project/03_data/processed/v01_training_index")
frames = pandas.read_csv(INDEX / "frames.csv", usecols = ["episode_id", "row_in_episode"] + [f"joint_{index}" for index in range(1, 7)], low_memory = False)
first = frames.sort_values(["episode_id", "row_in_episode"]).groupby("episode_id").first()
degrees = numpy.degrees(first[[f"joint_{index}" for index in range(1, 7)]].to_numpy())
stored = numpy.array([-6.06, -41.87, -61.93, 0.0, 13.53, -6.15])
near = numpy.abs(degrees - stored).max(axis = 1) < 0.5
print(f"episodes in frames.csv: {len(first)}; first picture within 0.5 degrees of the stored start joints {stored.tolist()}: {int(near.sum())}")
print(f"   their median joints: {numpy.round(numpy.median(degrees[near], axis = 0), 2).tolist()}; largest difference among them {numpy.abs(degrees[near] - numpy.median(degrees[near], axis = 0)).max():.2f} degrees")

joints = [math.radians(value) for value in stored]
tip, lean = magnet_tip_and_lean(joints)
print(f"\nat the stored start joints: tip ({100 * tip[0]:.2f}, {100 * tip[1]:.2f}) cm, lean {lean:.1f} deg; tip {1000 * (tip[2] - table_height(tip[0], tip[1])):.1f} mm above the taught table plane "
      f"(the driver's pushing height puts it {1000 * PUCK_PARAMETERS['pusher_clearance_above_sheet_m']:.0f} mm above)")
print(f"   today's start check (placed_point_problem): {placed_point_problem(joints, dict(MODEL_MODE_PARAMETERS)) or 'accepted'}")
settled = held_joints_at(joints, 0.0)
print(f"   settled with the wrist and hand held: differs from the stored joints by at most {max(abs(math.degrees(a - b)) for a, b in zip(settled, joints)):.2f} degrees")
for lift in (0.03, 0.05, 0.06):
    lifted = held_joints_at(joints, lift)
    if lifted is None:
        print(f"   {100 * lift:.0f} cm above the start: OUT OF REACH with the wrist held")
        continue
    tip_up, lean_up = magnet_tip_and_lean(lifted)
    print(f"   {100 * lift:.0f} cm above the start: reachable; joints {[round(math.degrees(value), 1) for value in lifted]}; tip {1000 * (tip_up[2] - table_height(tip_up[0], tip_up[1])):.0f} mm above the table, "
          f"lean {lean_up:.1f} deg; clear of a {1000 * PUCK_PARAMETERS['height_m']:.0f} mm puck by {1000 * (tip_up[2] - table_height(tip_up[0], tip_up[1]) - PUCK_PARAMETERS['height_m']):.0f} mm")
first_step = model_step(joints, {"tip_forward_cm": 0.7, "tip_left_cm": 0.0, "twist_deg": 0.0})
print(f"   a first 1 cm step straight ahead from there: {first_step['refusal'] or 'accepted'}")
print(f"   recording tool's travel speed {RECORDING_PARAMETERS['travel_velocity_percent']} % above {100 * RECORDING_PARAMETERS['travel_minimum_height_m']:.0f} cm; the driver's moves use the link's default (20 %)")

print("\nwhere the ten runs of 11.11 ended (the tip, and the lifted tip the arm would travel back from):")
for folder in sorted(Path("D:/AI_Project/03_data/model_runs").glob("2026-11-11_17*")):
    steps = pandas.read_csv(folder / "steps.csv")
    last = steps.iloc[-1]
    end_joints = [float(last[f"joint_{index}"]) for index in range(1, 7)]
    lifted = held_joints_at(end_joints, MODEL_MODE_PARAMETERS["lift_at_end_m"])
    print(f"   {folder.name:42s} tip ({100 * last['tip_x']:5.1f}, {100 * last['tip_y']:6.1f}) cm, forearm {math.degrees(last['joint_4']):+6.1f} deg, "
          f"{100 * math.hypot(last['tip_x'] - tip[0], last['tip_y'] - tip[1]):4.1f} cm from the start; lift of 5 cm {'reachable' if lifted is not None else 'OUT OF REACH'}")
