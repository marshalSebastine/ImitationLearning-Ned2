"""
Verifier, check 10: limits with the wrist held, and the hand (joint 6). Calculated with the project's arm model
(side_view_test.magnet_tip_and_lean, solve_with_wrist_fixed), tip at the driver's height (config.pushing_height minus
the extension), straight ahead of the base (y = 0) unless said otherwise.
    D:/AI_Project/.venv/Scripts/python.exe v10_lean_reach_hand.py
"""
import math
import sys
import numpy
sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
import forearm_camera_model
from config import MODEL_MODE_PARAMETERS, PUCK_PARAMETERS, PUSHER_PARAMETERS, RECORDING_PARAMETERS, pushing_height, table_height
from side_view_test import JOINT_LIMITS_RAD, magnet_tip_and_lean, solve_with_wrist_fixed

wrist = math.radians(MODEL_MODE_PARAMETERS["wrist_held_deg"])
hand = math.radians(MODEL_MODE_PARAMETERS["hand_held_deg"])
print("config: wrist_pitch_max_rad %.3f rad = %.1f degrees; rod_lean_most_deg %.1f; clearance above the sheet %.1f mm; joint 5 limits %.1f to %.1f degrees" % (
    RECORDING_PARAMETERS["wrist_pitch_max_rad"], math.degrees(RECORDING_PARAMETERS["wrist_pitch_max_rad"]), MODEL_MODE_PARAMETERS["rod_lean_most_deg"],
    1000 * PUCK_PARAMETERS["pusher_clearance_above_sheet_m"], math.degrees(JOINT_LIMITS_RAD[4][0]), math.degrees(JOINT_LIMITS_RAD[4][1])))


def held(reach_m, twist_deg, wrist_rad = wrist, bearing_deg = 0.0):
    x, y = reach_m * math.cos(math.radians(bearing_deg)), reach_m * math.sin(math.radians(bearing_deg))
    target = numpy.array([x, y, pushing_height(x, y) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    for start in ([math.atan2(y, x), -0.8, -0.6], [math.atan2(y, x), -0.3, -0.9], [math.atan2(y, x), -1.2, 0.0], [math.atan2(y, x), 0.2, -1.2]):
        joints = solve_with_wrist_fixed(target, math.radians(twist_deg), wrist_rad, start)
        if joints is not None:
            joints[5] = hand
            return joints
    return None


print("\nrod lean (degrees) with the wrist held at 13.5, by reach and twist ('-' = no solution within the joint limits):")
twists = [0, 30, 60, 90, 117, -60, -117]
print("reach_cm " + " ".join("%7d" % twist for twist in twists))
for reach_cm in (12, 14, 20, 26, 30, 34, 38, 40, 42, 44, 45, 46, 47, 48, 49, 49.5, 50):
    cells = []
    for twist in twists:
        joints = held(reach_cm / 100.0, twist)
        cells.append("      -" if joints is None else "%7.1f" % magnet_tip_and_lean(joints)[1])
    print("%8.1f " % reach_cm + " ".join(cells))
print("\nreach at which the lean passes 58 degrees (0.25 cm steps):")
for twist in twists:
    reached = None
    for reach in numpy.arange(0.40, 0.5001, 0.0025):
        joints = held(float(reach), twist)
        if joints is None:
            reached = "no solution from %.2f cm" % (100 * reach)
            break
        if magnet_tip_and_lean(joints)[1] > 58.0:
            reached = "%.2f cm" % (100 * reach)
            break
    print("  twist %5d: %s" % (twist, reached))
print("\nsame to the sides (bearing of the tip from the base), twist 0: lean at 40 and 46 cm")
for bearing in (-40, -20, 0, 20, 40):
    values = [held(reach, 0, bearing_deg = bearing) for reach in (0.40, 0.46)]
    print("  bearing %4d: %s" % (bearing, ", ".join("-" if joints is None else "%.1f" % magnet_tip_and_lean(joints)[1] for joints in values)))

print("\nother held wrists at twist 0 (what a wrist sweep or a different constant would do): lean at reach 14 / 20 / 30 / 40 / 46 cm")
for wrist_deg in (-10, 0, 5, 13.5, 25, 35):
    cells = []
    for reach in (0.14, 0.20, 0.30, 0.40, 0.46):
        joints = held(reach, 0, math.radians(wrist_deg))
        cells.append("-" if joints is None else "%.1f" % magnet_tip_and_lean(joints)[1])
    print("  wrist %5.1f: %s" % (wrist_deg, " / ".join(cells)))

print("\nthe hand (joint 6): does turning it move the tip or the camera?")
joints = held(0.30, 20)
for hand_deg in (-6.1, 0.0, 29.6):
    turned = list(joints)
    turned[5] = math.radians(hand_deg)
    tip, lean = magnet_tip_and_lean(turned)
    centre, axes = forearm_camera_model.camera_pose_in_base(turned)
    print("  hand %5.1f: tip %s mm, lean %.3f, camera centre %s mm" % (hand_deg, numpy.round(1000 * tip, 3), lean, numpy.round(1000 * centre, 3)))
print("  a block face 15 mm off the rod's axis turns with the hand: 30 degrees of hand moves it %.1f mm sideways" % (15 * math.sin(math.radians(30))))
