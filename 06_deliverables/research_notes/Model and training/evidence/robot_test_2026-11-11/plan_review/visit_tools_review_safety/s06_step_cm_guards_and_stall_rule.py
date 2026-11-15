"""
Safety review of the visit tools, check 6: --step-cm. Is a silly value refused, is every guard of a step (reach,
workspace, lean, twist per step, joint 4's limit) still applied to the shorter step, and what does the stall rule do
with very short steps? Also: the stall rule of the committed driver on the logged robot runs of 11.11.26 (the runs'
summaries show that the rule was added after them).

model_step is called on paper (nothing moves); the runs are simulated and logged in the computer's temporary folder;
the robot runs' logs under 03_data/model_runs are only read.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe s06_step_cm_guards_and_stall_rule.py > s06_step_cm_guards_and_stall_rule.out.txt
"""

import contextlib
import io
import json
import math
import sys
import tempfile
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
from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height  # noqa: E402
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed  # noqa: E402


class SteadyPolicy:
    """A stand-in for a model that always asks for 0.8 cm straight up the picture, no twist and no stop."""

    run_name = "steady_stand_in"
    checkpoint_paths = []

    def decide(self, picture_bytes):

        return {"tip_forward_cm": 0.8, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0, "angle_deg": 0.0, "stop_probability": 0.0}


def joints_behind(tip_xy, turn_deg = 0.0):
    """Joints with the tip at pushing height over ``tip_xy``, the forearm turned ``turn_deg``, the wrist and hand held."""

    target = numpy.array([tip_xy[0], tip_xy[1], pushing_height(*tip_xy) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    joints = solve_with_wrist_fixed(target, math.radians(turn_deg), math.radians(MODEL_MODE_PARAMETERS["wrist_held_deg"]), [math.atan2(tip_xy[1], tip_xy[0]), -0.8, -0.6])
    joints[5] = math.radians(MODEL_MODE_PARAMETERS["hand_held_deg"])
    return joints


print("1. the command line (model_mode.command_line; nothing is loaded or connected):")
for wrong in ("0", "-0.65", "0.09", "1.01", "1.5", "6.5", "65", "nan", "inf", "abc"):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            model_mode.command_line(["robot", "--step-cm", wrong])
        print(f"   --step-cm {wrong}: TAKEN")
    except SystemExit:
        print(f"   --step-cm {wrong}: refused")
try:
    with contextlib.redirect_stderr(io.StringIO()):
        model_mode.command_line(["robot", "--step-cm", "0.65", "--step-length", "predicted"])
    print("   --step-cm 0.65 --step-length predicted: TAKEN")
except SystemExit:
    print("   --step-cm 0.65 --step-length predicted: refused")
for fine in ("0.1", "0.65", "1.0"):
    print(f"   --step-cm {fine}: taken, step_m = {model_mode.command_line(['robot', '--step-cm', fine])[1]['step_m']}")

print("\n2. the guards of one step with the 1 cm step and with 0.65 cm (model_step on paper):")
cases = (("the workspace's far edge", [0.487, 0.0], 0.0, {"tip_forward_cm": 1.0, "tip_left_cm": 0.0, "twist_deg": 0.0}, {}),
         ("the workspace's left edge", [0.30, 0.247], 0.0, {"tip_forward_cm": 0.0, "tip_left_cm": 1.0, "twist_deg": 0.0}, {}),
         ("the workspace's right edge", [0.30, -0.297], 0.0, {"tip_forward_cm": 0.0, "tip_left_cm": -1.0, "twist_deg": 0.0}, {}),
         ("the least reach, towards the base", [0.123, 0.0], 0.0, {"tip_forward_cm": -1.0, "tip_left_cm": 0.0, "twist_deg": 0.0}, {}),
         ("the lean limit (set to 5 degrees to trip it)", [0.40, 0.0], 0.0, {"tip_forward_cm": 1.0, "tip_left_cm": 0.0, "twist_deg": 0.0}, {"rod_lean_most_deg": 5.0}),
         ("a twist of 40 degrees asked in one step", [0.25, 0.0], 0.0, {"tip_forward_cm": 1.0, "tip_left_cm": 0.0, "twist_deg": 40.0}, {}),
         ("the same with joint 4 at 112 degrees", [0.25, 0.0], 112.0, {"tip_forward_cm": 1.0, "tip_left_cm": 0.0, "twist_deg": 40.0}, {}))
for words, tip_xy, turn_deg, decision, extra in cases:
    for step_words, overrides in (("1 cm   ", {}), ("0.65 cm", {"step_m": 0.0065})):
        planned = model_mode.model_step(joints_behind(tip_xy, turn_deg), decision, None, dict(overrides, **extra))
        if planned["joints_next"] is None:
            print(f"   {words:46s} {step_words}: refused, {planned['refusal'][:62]}")
        else:
            moved_cm = 100 * float(numpy.linalg.norm(magnet_tip_and_lean(planned["joints_next"])[0][:2] - planned["tip_now"][:2]))
            print(f"   {words:46s} {step_words}: allowed, the tip moves {moved_cm:.2f} cm, the forearm turns {planned['twist_used_deg']:.1f} degrees to "
                  f"{math.degrees(planned['joints_next'][3]):.1f}")

print("\n3. a straight push with the default stall rule (it ends a run), 45 steps asked, simulated:")
with tempfile.TemporaryDirectory() as folder:
    for step_cm in (0.1, 0.11, 0.2, 0.65, 1.0):
        overrides = model_mode.command_line(["simulated", "--step-cm", str(step_cm)])[1]
        with contextlib.redirect_stdout(io.StringIO()):
            summary = model_mode.simulated_run(dict(overrides, run_folder = folder, step_budget = 45), SteadyPolicy(), f"step_{step_cm}")
        print(f"   fixed step {step_cm} cm: ended {summary['ending']} after {summary['pictures']} pictures and {summary['moves']} moves")
print(f"   (the rule: the tips of {MODEL_MODE_PARAMETERS['stall_steps']} pictures within {100 * MODEL_MODE_PARAMETERS['stall_extent_m']:g} cm and the forearm turned less than "
      f"{MODEL_MODE_PARAMETERS['stall_twist_deg']:g} degrees; {MODEL_MODE_PARAMETERS['stall_steps'] - 1} steps of 0.1 cm are 2.9 cm)")

print("\n4. the stall rule and the robot runs of 11.11.26 (read from 03_data/model_runs):")
run_folder = Path(MODEL_MODE_PARAMETERS["run_folder"])
stall_names = ["stall_steps", "stall_extent_m", "stall_twist_deg", "step_length_rule"]
rule = model_mode.ModelRun(None, None)
for folder in sorted(run_folder.iterdir()):
    if not folder.name.startswith("2026-11-11") or "simulated" in folder.name or not (folder / "summary.json").is_file():
        continue
    summary = json.loads((folder / "summary.json").read_text(encoding = "utf-8"))
    steps = pandas.read_csv(folder / "steps.csv")
    seen, fired_at = [], None
    for _, row in steps.iterrows():
        if row["action"] == "stop":
            break
        seen.append((row["tip_x"], row["tip_y"], math.degrees(row["joint_4"])))
        if rule.is_stalled(seen):
            fired_at = int(row["step"])
            break
    has_rule = [name for name in stall_names if name in summary["parameters"]]
    print(f"   {folder.name[18:]:26s} {len(steps):3d} pictures, ended {summary['ending']:8s} | stall settings in its summary: {has_rule or 'none'} | "
          f"today's rule would have ended it at picture: {fired_at}")
