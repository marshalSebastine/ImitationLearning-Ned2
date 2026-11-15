"""
The verifier's own check of how well a plain picture rotation about the tip's pixel reproduces a real turn of the table
about the tip (fitted camera, driver's arm pose with the forearm straight): table points round the tip are turned on
the table and projected, and compared with their original pixels rotated about the tip's pixel.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v15_rotation_error.py > v15_rotation_error.out.txt
"""

import math
import sys
from pathlib import Path

import numpy

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height, table_height  # noqa: E402
from forearm_camera_model import project_points  # noqa: E402
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed  # noqa: E402

for reach_cm in (15.0, 25.0, 35.0, 45.0):
    x = reach_cm / 100.0
    target = numpy.array([x, 0.0, pushing_height(x, 0.0) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    joints = solve_with_wrist_fixed(target, 0.0, math.radians(MODEL_MODE_PARAMETERS["wrist_held_deg"]), [0.0, -0.8, -0.6])
    joints[5] = math.radians(MODEL_MODE_PARAMETERS["hand_held_deg"])
    tip = magnet_tip_and_lean(joints)[0]
    height = table_height(float(tip[0]), float(tip[1]))
    tip_pixel = project_points(numpy.array([tip[0], tip[1], height]), joints)[0]
    # table points on a grid round the tip that are in the picture above the robot's parts
    offsets = numpy.array([[dx, dy] for dx in numpy.arange(-0.14, 0.141, 0.01) for dy in numpy.arange(-0.14, 0.141, 0.01)])
    points = numpy.column_stack([tip[0] + offsets[:, 0], tip[1] + offsets[:, 1], numpy.full(len(offsets), height)])
    pixels = project_points(points, joints)
    visible = numpy.isfinite(pixels).all(axis = 1) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= 639) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= 330)
    distance_cm = 100.0 * numpy.hypot(offsets[:, 0], offsets[:, 1])
    line = f"reach {reach_cm:.0f} cm (tip pixel {tip_pixel[0]:.0f}, {tip_pixel[1]:.0f}):"
    for angle in (10.0, 30.0):
        turn = math.radians(angle)
        rotation = numpy.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
        turned = offsets @ rotation.T
        turned_pixels = project_points(numpy.column_stack([tip[0] + turned[:, 0], tip[1] + turned[:, 1], numpy.full(len(offsets), height)]), joints)
        best = None
        for sign in (1.0, -1.0):
            image_turn = sign * turn
            image_rotation = numpy.array([[math.cos(image_turn), -math.sin(image_turn)], [math.sin(image_turn), math.cos(image_turn)]])
            plain = (pixels - tip_pixel[None, :]) @ image_rotation.T + tip_pixel[None, :]
            error = numpy.hypot(*(plain - turned_pixels).T)
            ok = visible & numpy.isfinite(error)
            score = numpy.nanmedian(error[ok & (distance_cm <= 5.0)])
            if best is None or score < best[0]:
                best = (score, numpy.nanmedian(error[ok & (distance_cm >= 3.0) & (distance_cm <= 12.0)]), numpy.nanmax(error[ok & (distance_cm >= 11.0) & (distance_cm <= 13.0)]))
        line += f"  {angle:.0f} degrees: median error within 5 cm {best[0]:.1f} px, 3 to 12 cm {best[1]:.1f} px, largest at about 12 cm {best[2]:.1f} px;"
    print(line)
