"""
Proofreader sitting_a, check 6: can part A1's fixed list of answers go through today's loop, and what is still open?

A stand-in policy (as test_model_mode.HesitatingPolicy) gives ten steps each of 1, 2 and 3 mm straight ahead, then five
twists in place of 5 degrees and five of 10 degrees, then asks to stop. It is run through ModelRun on the SIMULATED
robot from the recorded start joints, with today's switches only (--step-length predicted is the one rule that can make
a step shorter than 1 cm or a twist in place). Two patterns for the twists, because the plan names none: all one way
(75 degrees in sum) and left and right in turn. Also: the same list under the fixed rule, to show what a forgotten
switch would do. Nothing connects to the robot; the run folders are written into this folder only.

Run:  D:/AI_Project/.venv/Scripts/python.exe -B p06_scripted_list_on_paper.py > p06_scripted_list_on_paper.out.txt
"""

import math
import shutil
import sys
from pathlib import Path

import numpy
import pandas

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from config import MODEL_MODE_PARAMETERS                      # noqa: E402
from model_mode import ModelRun, model_step                  # noqa: E402
from robot_link import SimulatedRobotLink                    # noqa: E402  (the class that needs no hardware)
from side_view_test import magnet_tip_and_lean               # noqa: E402

RECORDED_START_DEG = [-6.06, -41.87, -61.93, 0.0, 13.53, -6.15]   # the checker's reading of frames.csv, 149 episodes


class ScriptedPolicy:
    """A fixed list of answers; after the list it asks to stop."""

    run_name = "scripted_small_steps"
    checkpoint_paths = []

    def __init__(self, twists_deg):
        self.answers = ([(0.1, 0.0)] * 10 + [(0.2, 0.0)] * 10 + [(0.3, 0.0)] * 10 + [(0.0, value) for value in twists_deg])
        self.index = 0

    def decide(self, picture_bytes: bytes) -> dict:
        if self.index >= len(self.answers):
            return {"tip_forward_cm": 0.0, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0, "angle_deg": 0.0, "stop_probability": 1.0}
        forward, twist = self.answers[self.index]
        self.index += 1
        return {"tip_forward_cm": forward, "tip_left_cm": 0.0, "twist_deg": twist, "lean_deg": 0.0, "hand_deg": 0.0, "angle_deg": 0.0, "stop_probability": 0.0}


start_joints = [math.radians(value) for value in RECORDED_START_DEG]
tip, lean = magnet_tip_and_lean(start_joints)
print(f"recorded start joints {RECORDED_START_DEG} degrees: tip at ({100 * tip[0]:.2f}, {100 * tip[1]:.2f}) cm, {100 * math.hypot(tip[0], tip[1]):.2f} cm from the base, "
      f"tip height {1000 * tip[2]:.1f} mm in the robot's frame, rod lean {lean:.1f} degrees")

patterns = {
    "twists all one way (+5 x5, +10 x5 = +75 degrees)": [5.0] * 5 + [10.0] * 5,
    "twists all the other way (-75 degrees)": [-5.0] * 5 + [-10.0] * 5,
    "twists left and right in turn": [5.0, -5.0, 5.0, -5.0, 5.0, -10.0, 10.0, -10.0, 10.0, -10.0],
}
output = HERE / "simulated_runs"
for rule in ("predicted", "fixed"):
    for name, twists in patterns.items():
        if rule == "fixed" and "in turn" not in name:
            continue
        robot = SimulatedRobotLink()
        robot.move_joints(start_joints)
        run = ModelRun(robot, ScriptedPolicy(twists), {"run_folder": output, "step_length_rule": rule, "step_budget": 60}, key_reader = lambda: None, is_shown = False)
        summary = run.run(f"scripted_{rule}")
        steps = pandas.read_csv(Path(summary["folder"]) / "steps.csv")
        tips = 1000.0 * steps[["tip_x", "tip_y"]].to_numpy()
        moved = numpy.linalg.norm(numpy.diff(tips, axis = 0), axis = 1)
        groups = [moved[0:10].sum(), moved[10:20].sum(), moved[20:30].sum()] if len(moved) >= 30 else []
        print(f"\nstep rule '{rule}', {name}: ending {summary['ending']} after {summary['pictures']} pictures and {summary['moves']} moves")
        if groups:
            print(f"   logged tip travel in the three groups of ten: {groups[0]:.2f}, {groups[1]:.2f}, {groups[2]:.2f} mm (asked 10, 20, 30)")
        twisting = steps.iloc[30:40]
        if len(twisting):
            drift = numpy.linalg.norm(tips[30:min(41, len(tips))] - tips[30], axis = 1).max() if len(tips) > 30 else float("nan")
            print(f"   during the twists: forearm from {math.degrees(steps['joint_4'].iloc[30]):+.1f} to {math.degrees(steps['joint_4'].iloc[-1]):+.1f} degrees; "
                  f"the logged tip strays at most {drift:.2f} mm; rod lean planned up to {twisting['lean_deg'].max():.1f} degrees (limit {MODEL_MODE_PARAMETERS['rod_lean_most_deg']})")
        refused = steps[steps["action"] == "refused"]
        if len(refused):
            print(f"   REFUSED at picture {int(refused['step'].iloc[0])}: {refused['refusal'].iloc[0]}")
        kept = int(steps["is_direction_kept"].fillna(0).sum())
        print(f"   steps whose direction was 'kept' because the asked move was under {MODEL_MODE_PARAMETERS['smallest_direction_cm']} cm: {kept}")
shutil.rmtree(output, ignore_errors = True)

# how far does a twist in place carry the block's faces? (the tip stays; the block turns about the rod with the forearm)
print("\nthe joints during one twist in place of 10 degrees at the end of the 60 mm (on paper):")
robot = SimulatedRobotLink()
robot.move_joints(start_joints)
joints = list(start_joints)
for _ in range(30):
    pass
planned = model_step(joints, {"tip_forward_cm": 0.0, "tip_left_cm": 0.0, "twist_deg": 10.0}, None, {"step_length_rule": "predicted"})
print(f"   refusal: {planned['refusal']}; joints 1 to 3 change by {[round(math.degrees(after - before), 2) for after, before in zip(planned['joints_next'][:3], joints[:3])]} degrees, "
      f"joint 4 by {math.degrees(planned['joints_next'][3] - joints[3]):+.1f}")
