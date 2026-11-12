"""
The verifier's own calculation of the arm pose a pre-turned start would need in run 3's lay-out (sheet 02 square,
portrait, from end B): the tip 5 cm (or 2.5 cm, touching) behind the puck on the line's backward extension, the
forearm twisted so that the picture's top points along the line. Arm model only; nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v16_pre_turned_start_pose.py > v16_pre_turned_start_pose.out.txt
"""

import math
import sys
from pathlib import Path

import numpy

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height  # noqa: E402
from record_episodes import forearm_camera_heading_rad  # noqa: E402  (a pure function; nothing connects at import)
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

puck = numpy.array([18.8, -1.5])            # run 3's puck at the start (verifier's v2_run3.py, measured)
frames = load_training_frames({"is_audit_cleaning": True})
reach_all = 100.0 * numpy.hypot(frames["tip_x_m"], frames["tip_y_m"]).to_numpy()
joint_4_all = numpy.degrees(frames["joint_4"].to_numpy())
for line_deg in (62.0, 68.0):
    for behind_cm in (5.0, 2.5):
        tip_xy = puck - behind_cm * numpy.array([math.cos(math.radians(line_deg)), math.sin(math.radians(line_deg))])
        x, y = tip_xy / 100.0
        target = numpy.array([x, y, pushing_height(x, y) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
        best = None
        for joint_4 in numpy.arange(-117.0, 117.1, 0.5):
            joints = solve_with_wrist_fixed(target, math.radians(joint_4), math.radians(MODEL_MODE_PARAMETERS["wrist_held_deg"]), [math.atan2(y, x), -0.8, -0.6])
            if joints is None:
                continue
            heading = math.degrees(forearm_camera_heading_rad(list(joints)))
            miss = abs((heading - line_deg + 180.0) % 360.0 - 180.0)
            if best is None or miss < best[0]:
                best = (miss, joint_4, heading, magnet_tip_and_lean(joints)[1])
        reach = float(numpy.hypot(*tip_xy))
        near = (numpy.abs(reach_all - reach) <= 3.0) & (numpy.abs(joint_4_all - best[1]) <= 15.0)
        print(f"line leaving at {line_deg:.0f} degrees, tip {behind_cm} cm behind the puck: tip ({tip_xy[0]:.1f}, {tip_xy[1]:.1f}) cm, reach {reach:.1f} cm;"
              f" joint 4 {best[1]:+.1f} degrees puts the picture's top at {best[2]:+.1f} (miss {best[0]:.1f}), rod lean {best[3]:.1f};"
              f" demonstration pictures within 3 cm of this reach and 15 degrees of this joint 4: {near.sum()} in {frames.loc[near, 'episode_id'].nunique()} episodes")
