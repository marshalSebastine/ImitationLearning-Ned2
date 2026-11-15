"""
Safety review of the visit tools, check 5: is the sign of "left" and of "further out" right in the offset of the
hand-placed start (offset_from_recorded_start, start_offset_lines, placing_line)?

1. A hand formula. With the forearm untwisted (joint 4 = 0) every link of the Ned2's chain in ned2_fk.py lies in the
   upright plane through the base's axis at the angle of joint 1, so the tip is at radius r and height z in that plane
   and at x = r cos(joint 1), y = r sin(joint 1) on the table. r and z are added up link by link below, straight from the
   chain's numbers, without calling the arm model. Joint 1 turned counter-clockwise seen from above is positive, so it
   moves the tip towards +y; with x pointing out from the base and z up, +y is to the LEFT of someone at the base who
   looks out along x.
2. The driver's function and its printed lines for the same joint sets.
3. Whether the arm model's +y is the robot's own +y: the robot's reported y of the recorded demonstrations against the
   model's tip y (frames.csv, read only), and how many demonstrations begin at the recorded start.

Nothing is read from the robot and nothing moves.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe s05_start_offset_signs_by_hand.py > s05_start_offset_signs_by_hand.out.txt
"""

import math
import sys
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))

import robot_link  # noqa: E402


class RefusedRobotLink:
    """Stands in the real RobotLink's place, so that nothing here can connect to the robot."""

    def __init__(self, *arguments, **named_arguments):

        raise AssertionError("a connection to the robot was attempted")


robot_link.RobotLink = RefusedRobotLink

import model_mode  # noqa: E402
from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, TRAINING_DATA_PARAMETERS, pushing_height  # noqa: E402
from side_view_test import magnet_tip_and_lean  # noqa: E402


def tip_by_hand(joints_deg):
    """
    The magnet tip for ``joints_deg`` with joint 4 = 0, added up in the arm's upright plane (radius, height).

    Each link's direction is an angle in that plane, measured from the horizontal outward direction: the upper arm's
    frame is turned by joint 2, the elbow's by a quarter turn plus joint 3, the wrist's by joint 5. The lengths are the
    origins of ned2_fk.CHAIN in metres; the forearm (0.17) and the rod (0.0285 plus the pusher) run along the "down"
    axis of their frames.
    """

    joint_1, joint_2, joint_3, joint_4, joint_5, _ = [math.radians(value) for value in joints_deg]
    assert abs(joint_4) < 1e-12, "the hand formula holds for an untwisted forearm only"
    radius, height = 0.0, 0.1013 + 0.065
    upper_arm = joint_2
    radius += 0.012 * math.cos(upper_arm) - 0.221 * math.sin(upper_arm)
    height += 0.012 * math.sin(upper_arm) + 0.221 * math.cos(upper_arm)
    elbow = upper_arm + math.pi / 2 + joint_3
    radius += 0.0325 * math.cos(elbow) + 0.065 * math.sin(elbow)
    height += 0.0325 * math.sin(elbow) - 0.065 * math.cos(elbow)
    radius += 0.17 * math.sin(elbow)
    height -= 0.17 * math.cos(elbow)
    wrist = elbow + joint_5
    radius += 0.00925 * math.cos(wrist) + 0.0197 * math.sin(wrist)
    height += 0.00925 * math.sin(wrist) - 0.0197 * math.cos(wrist)
    rod = 0.0285 + PUSHER_PARAMETERS["extension_below_rod_end_m"]
    radius += rod * math.sin(wrist)
    height -= rod * math.cos(wrist)
    return radius * math.cos(joint_1), radius * math.sin(joint_1), height, radius


recorded = MODEL_MODE_PARAMETERS["recorded_start_joints_deg"]
x, y, z, radius = tip_by_hand(recorded)
model_tip = magnet_tip_and_lean([math.radians(value) for value in recorded])[0]
print("1. the recorded start", recorded)
print(f"   by hand:   tip x {100 * x:.2f} cm, y {100 * y:.2f} cm, height {100 * z:.2f} cm, {100 * radius:.2f} cm from the base's axis")
print(f"   arm model: tip x {100 * model_tip[0]:.2f} cm, y {100 * model_tip[1]:.2f} cm, height {100 * model_tip[2]:.2f} cm; pushing height there "
      f"{100 * (pushing_height(model_tip[0], model_tip[1]) - PUSHER_PARAMETERS['extension_below_rod_end_m']):.2f} cm")

print("\n2. joint sets made by hand, the offset by hand and by the driver's function, and the driver's lines:")
for words, change in (("joint 1 turned 4 degrees counter-clockwise seen from above", [4, 0, 0, 0, 0, 0]), ("joint 1 turned 4 degrees clockwise", [-4, 0, 0, 0, 0, 0]),
                      ("joint 2 (shoulder) 3 degrees forward", [0, 3, 0, 0, 0, 0]), ("joint 2 (shoulder) 3 degrees back", [0, -3, 0, 0, 0, 0])):
    joints = [value + added for value, added in zip(recorded, change)]
    hand_x, hand_y, _, _ = tip_by_hand(joints)
    offset = model_mode.offset_from_recorded_start([math.radians(value) for value in joints])
    is_agreeing = abs(100 * (hand_x - x) - offset["tip_along_x_cm"]) < 0.01 and abs(100 * (hand_y - y) - offset["tip_along_y_cm"]) < 0.01
    print(f"   {words}:")
    print(f"      by hand along x {100 * (hand_x - x):+.2f} cm, along y {100 * (hand_y - y):+.2f} cm | the function {offset['tip_along_x_cm']:+.2f}, {offset['tip_along_y_cm']:+.2f} | "
          f"{'they agree' if is_agreeing else 'THEY DISAGREE'}")
    print("      " + model_mode.start_offset_lines(offset)[0])
    print("      " + model_mode.placing_line(offset))

print("\n3. the arm model against the robot's own reports in the recorded demonstrations (frames.csv):")
joint_names = [f"joint_{index}" for index in range(1, 7)]
frames = pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", usecols = ["episode_id", "row_in_episode", "x", "y"] + joint_names)
first_rows = frames.sort_values(["episode_id", "row_in_episode"]).groupby("episode_id").head(1)
first_deg = numpy.degrees(first_rows[joint_names].to_numpy())
is_at_start = (numpy.abs(first_deg - numpy.array(recorded)) <= 0.5).all(axis = 1)
print(f"   demonstrations: {len(first_rows)}; their first row within 0.5 degrees of the recorded start on all six joints: {int(is_at_start.sum())}")
sample = frames.sample(3000, random_state = 1)
sample_tips = numpy.array([magnet_tip_and_lean(list(row))[0] for row in sample[joint_names].to_numpy()])
print(f"   3000 recorded rows: the robot's reported y against the model's tip y, correlation {numpy.corrcoef(sample['y'], sample_tips[:, 1])[0, 1]:.4f}; "
      f"reported x against the model's tip x, {numpy.corrcoef(sample['x'], sample_tips[:, 0])[0, 1]:.4f}")
print("   (config.py, RECORDING_PARAMETERS \"picture_left_sign\": tested on the robot on 31.10.26 and 2.11.26, three \"left\" steps moved the tip towards +y)")
