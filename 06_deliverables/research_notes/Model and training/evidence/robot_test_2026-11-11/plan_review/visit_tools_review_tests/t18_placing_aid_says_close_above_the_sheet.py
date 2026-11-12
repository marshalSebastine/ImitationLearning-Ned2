"""
Review check: what the placing aid says when the tip hangs above the sheet, and what the run then does with that start.
The tip is held exactly over the recorded start, 3 cm and 5 cm too high (the arm model, no robot).
"""

import math

import numpy

import safe_import
from safe_import import model_mode
from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed

recorded = [math.radians(value) for value in MODEL_MODE_PARAMETERS["recorded_start_joints_deg"]]
recorded_tip = magnet_tip_and_lean(recorded)[0]
settings = dict(MODEL_MODE_PARAMETERS)
print("the run refuses a start whose tip is more than", 100 * settings["start_height_tolerance_m"], "cm from pushing height")
for lift_cm in (0.0, 1.5, 3.0, 5.0):
    target = numpy.array([recorded_tip[0], recorded_tip[1], pushing_height(recorded_tip[0], recorded_tip[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"] + lift_cm / 100])
    joints = solve_with_wrist_fixed(target, 0.0, math.radians(settings["wrist_held_deg"]), [recorded[0], recorded[1], recorded[2]])
    joints[5] = recorded[5]
    offset = model_mode.offset_from_recorded_start(joints)
    print(f"tip {lift_cm:.1f} cm above pushing height | the placing aid shows: {model_mode.placing_line(offset)!r}")
    print(f"      | the start line: {model_mode.start_offset_lines(offset)[0][:62]!r} ...  ({len(model_mode.start_offset_lines(offset))} line)")
    print(f"      | the run then says: {model_mode.placed_point_problem(joints, settings) or 'the start is fine'}")
