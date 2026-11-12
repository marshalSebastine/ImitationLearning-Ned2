"""
Safety review of the visit tools, check 1: started WITHOUT any new flag, does a robot run of the edited driver ask the
robot for the same things, in the same order, as the committed driver (commit 164ee79, the driver before the visit
tools)?

Both drivers run robot_run exactly as the command line calls it (a label and overrides only) on a recording stand-in
link: the simulated robot with a counter camera (a new picture at every call), which notes every call with its joints.
The real robot is never reached: robot_link.RobotLink is replaced by a function that hands out the stand-in, the
keyboard and the picture window by stand-ins, and the logs go to the computer's temporary folder, not into 03_data.
The committed driver is read from git (git show, read only).

Compared per scenario: every call to the link with its joints (to 1e-10 rad), the prompts and the moment they come,
the pictures given to the window, the pictures given to the model, the saved frames, steps.csv (all but the clock
columns), the result, what was raised, the motors' state at the end, and every printed line except the new "start:"
lines.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe s01_without_flags_committed_against_edited.py > s01_without_flags_committed_against_edited.out.txt
"""

import builtins
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
COMMITTED = "164ee79e918112021905db4e87e09229f43f9b01"
sys.path.insert(0, str(ROOT / "02_code"))

import robot_link  # noqa: E402


class RecordingLink(robot_link.SimulatedRobotLink):
    """The simulated robot with a counter camera and a list of everything that is asked of it."""

    def __init__(self, failures = None):

        super().__init__()
        self.calls = []
        self.failures = dict(failures or {})
        self.counts = {}


    def note(self, name, *details):
        """Keep the call; raise the failure planned for this call, if any (``failures``: name -> (which call, what to raise))."""

        self.calls.append((name,) + details)
        self.counts[name] = self.counts.get(name, 0) + 1
        wanted = self.failures.get(name)
        if wanted is not None and self.counts[name] == wanted[0]:
            raise wanted[1]


    def set_free_motion(self, is_enabled):

        self.note("set_free_motion", bool(is_enabled))
        super().set_free_motion(is_enabled)


    def read_state(self):

        self.note("read_state")
        return super().read_state()


    def move_joints(self, joints, velocity_percent = None):

        self.note("move_joints", tuple(round(float(value), 10) for value in joints), velocity_percent)
        super().move_joints(joints, velocity_percent)


    def move_home(self, *arguments, **named_arguments):

        self.note("move_home")
        super().move_home(*arguments, **named_arguments)


    def read_picture_bytes(self):

        self.note("read_picture_bytes")
        self.picture_count += 1
        self.picture_bytes = self._SimulatedRobotLink__render_picture()
        return self.picture_bytes


    def close(self):

        self.note("close")



class NoKeyboard:
    """Stands in msvcrt's place: no key ever waits in the terminal."""

    def kbhit(self):

        return False


    def getch(self):

        raise AssertionError("no key waits")

    getwch = getch



def loaded(name, path):
    """Load the Python file at ``path`` as the module ``name``."""

    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


SCRATCH = tempfile.TemporaryDirectory()
committed_text = subprocess.run(["git", "-C", str(ROOT), "show", f"{COMMITTED}:02_code/model_mode.py"], capture_output = True, check = True).stdout
(Path(SCRATCH.name) / "model_mode_committed.py").write_bytes(committed_text)
committed = loaded("model_mode_committed", str(Path(SCRATCH.name) / "model_mode_committed.py"))
edited = loaded("model_mode_edited", str(ROOT / "02_code" / "model_mode.py"))

from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height  # noqa: E402
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed  # noqa: E402

RECORDED_START = [math.radians(value) for value in [-6.06, -41.87, -61.93, 0.0, 13.53, -6.15]]


def joints_behind(tip_xy, turn_deg = 0.0):
    """Joints with the tip at pushing height over ``tip_xy``, the forearm turned ``turn_deg``, the wrist and hand held."""

    target = numpy.array([tip_xy[0], tip_xy[1], pushing_height(*tip_xy) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    joints = solve_with_wrist_fixed(target, math.radians(turn_deg), math.radians(MODEL_MODE_PARAMETERS["wrist_held_deg"]),
                                    [math.atan2(tip_xy[1], tip_xy[0]), -0.8, -0.6])
    joints[5] = math.radians(MODEL_MODE_PARAMETERS["hand_held_deg"])
    return joints



class StandInPolicy:
    """A stand-in model with a fixed answer; it can stop after some pictures or fail at one."""

    run_name = "stand_in"
    checkpoint_paths = []

    def __init__(self, forward = 0.8, left = 0.0, twist = 0.0, stop_after = None, fail_at = None):

        self.forward, self.left, self.twist, self.stop_after, self.fail_at = forward, left, twist, stop_after, fail_at
        self.pictures = []


    def decide(self, picture_bytes):

        self.pictures.append(hashlib.sha256(picture_bytes).hexdigest()[:12])
        if self.fail_at is not None and len(self.pictures) == self.fail_at:
            raise ValueError("the model gave a number that is not finite")
        stop = 0.9 if self.stop_after is not None and len(self.pictures) > self.stop_after else 0.0
        return {"tip_forward_cm": self.forward, "tip_left_cm": self.left, "twist_deg": self.twist, "lean_deg": 0.0, "hand_deg": 0.0,
                "angle_deg": float(numpy.degrees(numpy.arctan2(self.left, self.forward))), "stop_probability": stop}



def one_run(module, scenario, run_folder):
    """robot_run of ``module`` as the command line calls it without new flags, on a recording link; everything seen is returned."""

    link = RecordingLink(scenario.get("link_failures"))
    link.move_joints(scenario.get("joints_placed", RECORDED_START))
    link.calls.clear()
    link.counts.clear()
    policy = scenario["policy"]() if scenario.get("policy") != "real" else None
    window_keys = list(scenario.get("window_keys", ["enter"]))
    key_calls = {"count": 0}
    prompts, shown, printed, moments = [], [], io.StringIO(), []

    def stand_in_input(prompt = ""):

        prompts.append(prompt)
        moments.append(len(link.calls))
        raising = scenario.get("input_raises", {}).get(len(prompts))
        if raising is not None:
            raise raising
        return ""

    def stand_in_window_key(wait_ms):

        key_calls["count"] += 1
        raising = scenario.get("window_key_raises", {}).get(key_calls["count"])
        if raising is not None:
            raise raising
        return window_keys.pop(0) if window_keys else None

    real_parts = (builtins.input, module.show_in_window, module.read_window_key, module.msvcrt, robot_link.RobotLink, module.ModelPolicy)
    builtins.input = stand_in_input
    module.show_in_window = lambda picture: shown.append((picture.shape, hashlib.sha256(picture.tobytes()).hexdigest()[:12]))
    module.read_window_key = stand_in_window_key
    module.msvcrt = NoKeyboard()
    robot_link.RobotLink = lambda *arguments, **named_arguments: link
    if policy is not None:
        module.ModelPolicy = lambda *arguments, **named_arguments: policy
    outcome = {"raised": None, "result": None}
    try:
        with contextlib.redirect_stdout(printed):
            try:
                outcome["result"] = module.robot_run(scenario.get("label", "model_1"), dict(scenario.get("overrides", {}), run_folder = run_folder))
            except BaseException as error:
                outcome["raised"] = f"{type(error).__name__}: {error}".rstrip(": ")
    finally:
        builtins.input, module.show_in_window, module.read_window_key, module.msvcrt, robot_link.RobotLink, module.ModelPolicy = real_parts
    outcome.update({"calls": link.calls, "prompts": prompts, "moments": moments, "shown": shown, "printed": printed.getvalue(),
                    "policy_pictures": None if policy is None else policy.pictures, "are_motors_off_at_the_end": link.is_free_motion,
                    "tip_height_cm_at_the_end": round(100 * float(magnet_tip_and_lean(link.joints)[0][2]), 1)})
    folders = sorted(Path(run_folder).iterdir()) if Path(run_folder).is_dir() else []
    outcome.update({"steps": None, "files": [], "frames": None})
    if folders:
        folder = folders[-1]
        outcome["files"] = sorted(str(path.relative_to(folder)).replace("\\", "/") for path in folder.rglob("*") if path.is_file())
        if (folder / "steps.csv").exists():
            with open(folder / "steps.csv", newline = "", encoding = "utf-8") as file:
                outcome["steps"] = [{key: value for key, value in row.items() if key not in ("unix_time", "decision_ms")} for row in csv.DictReader(file)]
        outcome["frames"] = [hashlib.sha256(path.read_bytes()).hexdigest()[:12] for path in sorted((folder / "frames").glob("*.jpg"))]
    return outcome


NEW_SUMMARY_KEYS = {"seconds_first_to_last_picture", "step_length_rule", "fixed_step_cm", "stall_action", "stall_logged_at_picture", "start_offset"}
NEW_LINE_STARTS = ("start: tip ", "the start is further off")


def without_clock(text):
    """``text`` with the run folders' dates and places taken out."""

    text = re.sub(r"\d{4}-\d{2}-\d{2}_\d{6}_", "<time>_", text)
    return re.sub(r"logged in .*?<time>_", "logged in <folder>/<time>_", text)


def differences_between(before, after):
    """What differs between the committed driver's run and the edited driver's, beyond the additions the task asked for."""

    differences = []
    if before["calls"] != after["calls"]:
        differences.append(f"the calls to the link differ: {len(before['calls'])} against {len(after['calls'])}")
        for index, (call_before, call_after) in enumerate(zip(before["calls"], after["calls"])):
            if call_before != call_after:
                differences.append(f"   first at call {index}: {call_before} against {call_after}")
                break
    for name, words in (("prompts", "the prompts differ"), ("moments", "the prompts come at other moments"), ("shown", "the pictures in the window differ"),
                        ("raised", "what is raised differs"), ("are_motors_off_at_the_end", "the motors' state at the end differs"),
                        ("policy_pictures", "the model was given other pictures"), ("frames", "the saved frames differ"), ("files", "the files differ"),
                        ("steps", "steps.csv differs"), ("tip_height_cm_at_the_end", "the tip ends at another height")):
        if before[name] != after[name]:
            differences.append(f"{words}: {str(before[name])[:120]} against {str(after[name])[:120]}")
    lines_before = without_clock(before["printed"]).splitlines()
    lines_after = [line for line in without_clock(after["printed"]).splitlines() if not line.startswith(NEW_LINE_STARTS)]
    if lines_before != lines_after:
        differences.append(f"the printed lines differ:\n      committed {lines_before}\n      edited    {lines_after}")
    if (before["result"] is None) != (after["result"] is None):
        differences.append("one returned a result, the other did not")
    elif before["result"] is not None:
        left_out = NEW_SUMMARY_KEYS | {"seconds", "started", "folder", "parameters"}
        kept_before = {key: value for key, value in before["result"].items() if key not in left_out}
        kept_after = {key: value for key, value in after["result"].items() if key not in left_out}
        if kept_before != kept_after:
            differences.append(f"the results differ: {kept_before} against {kept_after}")
        if not set(after["result"]) - set(before["result"]) <= NEW_SUMMARY_KEYS:
            differences.append(f"result keys nobody asked for: {set(after['result']) - set(before['result']) - NEW_SUMMARY_KEYS}")
        if "parameters" in before["result"]:
            # both drivers read today's config.py, so both summaries carry the 13 new settings; the rest must agree
            if {key: str(value) for key, value in before["result"]["parameters"].items() if key != "run_folder"} != \
                    {key: str(value) for key, value in after["result"]["parameters"].items() if key != "run_folder"}:
                differences.append("the settings written into the summary differ")
    return differences


SCENARIOS = {
    "step budget (6 steps)": {"policy": lambda: StandInPolicy(), "overrides": {"step_budget": 6}},
    "stopped by the stop rule": {"policy": lambda: StandInPolicy(stop_after = 4), "overrides": {"step_budget": 30}},
    "steering left with a twist of 14 degrees asked": {"policy": lambda: StandInPolicy(forward = 0.6, left = 0.5, twist = 14.0), "overrides": {"step_budget": 12}},
    "stalled (the model's own length, a hesitating model)": {"policy": lambda: StandInPolicy(forward = 0.02),
                                                             "overrides": {"step_budget": 20, "step_length_rule": "predicted", "stall_steps": 5}},
    "40 fixed steps straight out (runs into a guard)": {"policy": lambda: StandInPolicy(), "overrides": {"step_budget": 40}},
    "the model's own length": {"policy": lambda: StandInPolicy(forward = 0.4, left = 0.1, twist = 3.0), "overrides": {"step_budget": 8, "step_length_rule": "predicted"}},
    "the real model 1 on the simulated pictures": {"policy": "real", "overrides": {"step_budget": 12}},
    "Esc at the go-ahead window": {"policy": lambda: StandInPolicy(), "window_keys": ["x", "escape"]},
    "Ctrl+C at the go-ahead window": {"policy": lambda: StandInPolicy(), "window_key_raises": {1: KeyboardInterrupt()}},
    "the camera fails for the go-ahead picture": {"policy": lambda: StandInPolicy(), "link_failures": {"read_picture_bytes": (1, RuntimeError("camera gone"))}},
    "placed 5 cm too high": {"policy": lambda: StandInPolicy(), "joints_placed": edited.held_joints_at(RECORDED_START, 0.05)},
    "placed with the forearm beyond its limit": {"policy": lambda: StandInPolicy(), "joints_placed": joints_behind([0.25, 0.0], 118.5)},
    "Esc in the run": {"policy": lambda: StandInPolicy(), "window_keys": ["enter", None, None, None, "escape"], "overrides": {"step_budget": 30}},
    "Ctrl+C in the run (between two calls to the link)": {"policy": lambda: StandInPolicy(), "window_key_raises": {6: KeyboardInterrupt()}, "overrides": {"step_budget": 30}},
    "a step refused at the workspace edge": {"policy": lambda: StandInPolicy(), "joints_placed": joints_behind([0.47, 0.0]), "overrides": {"step_budget": 30}},
    "the robot refuses the third step": {"policy": lambda: StandInPolicy(), "link_failures": {"move_joints": (5, RuntimeError("collision detected\nmore"))},
                                         "overrides": {"step_budget": 30}},
    "the model fails at the third picture": {"policy": lambda: StandInPolicy(fail_at = 3), "overrides": {"step_budget": 30}},
    "the camera fails in the run": {"policy": lambda: StandInPolicy(), "link_failures": {"read_picture_bytes": (9, RuntimeError("camera gone"))}, "overrides": {"step_budget": 30}},
    "the lift after the run fails": {"policy": lambda: StandInPolicy(), "link_failures": {"move_joints": (6, RuntimeError("refused"))}, "overrides": {"step_budget": 3}},
    "the state cannot be read after placing": {"policy": lambda: StandInPolicy(), "link_failures": {"read_state": (1, RuntimeError("no answer"))}},
    "the motors cannot be switched back on at first": {"policy": lambda: StandInPolicy(), "link_failures": {"set_free_motion": (2, RuntimeError("no answer"))}},
    "Ctrl+C at the first prompt": {"policy": lambda: StandInPolicy(), "input_raises": {1: KeyboardInterrupt()}},
    "Ctrl+C at the second prompt": {"policy": lambda: StandInPolicy(), "input_raises": {2: KeyboardInterrupt()}},
}


if __name__ == "__main__":
    print(f"the committed driver: commit {COMMITTED[:7]}, sha256 {hashlib.sha256(committed_text).hexdigest()[:12]}; the edited driver: sha256 "
          f"{hashlib.sha256((ROOT / '02_code' / 'model_mode.py').read_bytes()).hexdigest()[:12]}")
    print("per scenario: the same or different; then what the edited driver did (the committed one did the same where it says so)\n")
    are_all_the_same = True
    for name, scenario in SCENARIOS.items():
        with tempfile.TemporaryDirectory() as folder_before, tempfile.TemporaryDirectory() as folder_after:
            before = one_run(committed, scenario, folder_before)
            after = one_run(edited, scenario, folder_after)
        differences = differences_between(before, after)
        are_all_the_same = are_all_the_same and not differences
        moves = sum(1 for call in after["calls"] if call[0] == "move_joints")
        new_lines = sum(1 for line in after["printed"].splitlines() if line.startswith(NEW_LINE_STARTS))
        ending = None if after["result"] is None else after["result"].get("ending")
        print(f"{'THE SAME ' if not differences else 'DIFFERENT'}  {name}: {len(after['calls'])} calls to the link, {moves} of them moves, {len(after['prompts'])} prompts, "
              f"ending {ending}, raised {after['raised']}, tip {after['tip_height_cm_at_the_end']} cm high at the end, motors off at the end {after['are_motors_off_at_the_end']}, "
              f"new printed lines {new_lines}")
        for difference in differences:
            print("     ", difference)
    print("\nALL THE SAME except the new 'start:' lines and the new summary keys" if are_all_the_same else "\nSOME DIFFER")
    print("(the tip is 1.3 cm high at pushing height at the recorded start, 6.3 cm after the lift at the end, 4.3 cm after the 3 cm of the settling)")
