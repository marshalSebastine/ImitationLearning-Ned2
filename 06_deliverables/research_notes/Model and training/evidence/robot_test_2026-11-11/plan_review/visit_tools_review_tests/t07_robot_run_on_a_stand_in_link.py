"""Review attack 6: the whole robot_run flow on a watched simulated link, with things going wrong at each new place."""

import builtins
import contextlib
import io
import math
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import numpy

import safe_import
from safe_import import model_mode, robot_link
from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed

recorded = [math.radians(value) for value in MODEL_MODE_PARAMETERS["recorded_start_joints_deg"]]


class Steady:

    run_name = "steady_stand_in"
    checkpoint_paths = []

    def decide(self, picture_bytes):

        return {"tip_forward_cm": 0.8, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0, "angle_deg": 0.0, "stop_probability": 0.0}


class Watched(robot_link.SimulatedRobotLink):

    def __init__(self, failing = None):

        super().__init__()
        self.asked = []
        self.failing = failing or {}
        self.counts = {}

    def note(self, name):

        self.asked.append(name)
        self.counts[name] = self.counts.get(name, 0) + 1
        if name in self.failing and self.counts[name] == self.failing[name][0]:
            raise self.failing[name][1]

    def set_free_motion(self, is_enabled):

        self.note("motors off" if is_enabled else "motors on")
        super().set_free_motion(is_enabled)

    def read_state(self):

        self.note("read state")
        return super().read_state()

    def read_picture_bytes(self):

        self.counts["picture"] = self.counts.get("picture", 0) + 1
        if "picture" in self.failing and self.counts["picture"] == self.failing["picture"][0]:
            self.asked.append("picture (fails)")
            raise self.failing["picture"][1]
        return super().read_picture_bytes()

    def move_joints(self, joints, velocity_percent = None):

        self.note("move")
        super().move_joints(joints, velocity_percent)

    def close(self):

        self.asked.append("close")


def drive(name, parameters, window_keys, failing = None, prompt_answers = None, console = True, terminal_keys = None, joints_placed = None):

    link = Watched(failing)
    robot_link.SimulatedRobotLink.move_joints(link, joints_placed or recorded)
    keys = list(window_keys)
    prompts, shown, printed = [], [], io.StringIO()
    answers = list(prompt_answers or [])
    looks = {"n": 0}
    terminal = list(terminal_keys or [])
    real = (builtins.input, model_mode.show_in_window, model_mode.read_window_key, model_mode.is_terminal_a_console, model_mode.enter_pressed)

    def answered(prompt = ""):
        prompts.append(prompt[:42])
        answer = answers.pop(0) if answers else ""
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def terminal_key():
        looks["n"] += 1
        answer = terminal.pop(0) if terminal else None
        if isinstance(answer, BaseException):
            raise answer
        return answer

    builtins.input = answered
    model_mode.show_in_window = shown.append
    model_mode.read_window_key = lambda wait_ms: keys.pop(0) if keys else None
    model_mode.is_terminal_a_console = lambda: console
    model_mode.enter_pressed = terminal_key
    result = None
    raised = None
    try:
        with contextlib.redirect_stdout(printed):
            result = model_mode.robot_run("review", parameters, stand_in_link = link, stand_in_policy = Steady())
    except BaseException as error:
        raised = f"{type(error).__name__}: {error}"
    finally:
        builtins.input, model_mode.show_in_window, model_mode.read_window_key, model_mode.is_terminal_a_console, model_mode.enter_pressed = real
    print(f"--- {name}")
    print("   ended:", (result or {}).get("ending"), "| raised:", raised)
    print("   asked:", link.asked)
    print("   prompts:", prompts)
    print("   shown:", len(shown), "| free motion at the end:", link.is_free_motion)
    interesting = [line for line in printed.getvalue().replace("\r", "\n").splitlines() if any(word in line for word in ("placing aid", "not started", "STILL", "camera", "error", "start is further", "nothing moved"))]
    print("   printed:", interesting[:6])
    return result, link, printed.getvalue()


with tempfile.TemporaryDirectory() as folder:
    ghost_path = Path(folder) / "ghost.jpg"
    cv2.imwrite(str(ghost_path), numpy.zeros((480, 640, 3), numpy.uint8))
    base = {"run_folder": folder, "step_budget": 2}
    aid = dict(base, is_place_help_shown = True, place_help_refresh_s = 0.0, place_help_key_poll_s = 0.0)

    drive("default run", base, ["enter"])
    drive("aid: Enter at the third look", aid, ["enter"], terminal_keys = [None, None, "enter"])
    drive("aid: a reading of the robot fails at the 2nd reading -> plain prompt", aid, ["enter"], failing = {"read state": (2, RuntimeError("TCP timeout"))}, terminal_keys = [None] * 50)
    drive("aid: Ctrl+C while guiding -> the motors question, then motors on", aid, ["enter"], terminal_keys = [None, None, KeyboardInterrupt()])
    drive("aid: Ctrl+C while guiding, and Ctrl+C again at the motors question", aid, ["enter"], terminal_keys = [None, KeyboardInterrupt()], prompt_answers = ["", KeyboardInterrupt()])
    drive("aid wanted but the terminal is no console", aid, ["enter"], console = False)
    drive("ghost: Enter at the third look at the window", dict(base, ghost_path = ghost_path), [None, None, "enter"])
    drive("ghost: Esc -> lift, not started", dict(base, ghost_path = ghost_path), [None, "escape"])
    # the pictures: number 1.. are read by the go-ahead loop
    drive("ghost: the camera read RAISES at the 3rd live picture", dict(base, ghost_path = ghost_path), [None] * 10 + ["enter"], failing = {"picture": (3, RuntimeError("camera stream lost"))})
    drive("no ghost: the one camera read of the still go-ahead RAISES (as on 11.11.26)", base, ["enter"], failing = {"picture": (1, RuntimeError("camera stream lost"))})
    drive("ghost + aid + stall log + step 0.65: everything at once", dict(aid, ghost_path = ghost_path, stall_action = "log", step_m = 0.0065), [None, "enter"], terminal_keys = [None, None, "enter"])
    drive("ghost given as text, not a Path", dict(base, ghost_path = str(ghost_path)), ["enter"])
    drive("ghost strength None (a mistake in config.py)", dict(base, ghost_path = ghost_path, ghost_strength = None), ["enter"])
    drive("recorded start with five joints (a mistake in config.py)", dict(base, recorded_start_joints_deg = [0.0] * 5), ["enter"])
    drive("warn limits None (a mistake in config.py)", dict(base, start_offset_warn_m = None), ["enter"])
    high = list(recorded)
    high[1] = math.radians(-30.0)
    drive("placed too high, with the aid and the ghost", dict(aid, ghost_path = ghost_path), ["enter"], terminal_keys = [None, None, "enter"], joints_placed = high)

print()
print("=== a missing reference picture, with NO stand-in link and NO stand-in model: nothing may be loaded or connected")
started = time.monotonic()
loaded = []
real_policy = model_mode.ModelPolicy
model_mode.ModelPolicy = lambda *arguments, **named: loaded.append("model loaded") or real_policy(*arguments, **named)
real_input = builtins.input
builtins.input = lambda prompt = "": (_ for _ in ()).throw(AssertionError("a prompt was reached"))
try:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        result = model_mode.robot_run("review", {"ghost_path": Path("D:/no/such/picture.jpg")})
    print("result:", result, "| model loaded:", loaded, "| seconds:", round(time.monotonic() - started, 3))
    print("printed:", printed.getvalue().strip())
except BaseException as error:
    print("RAISED:", type(error).__name__, error)
finally:
    builtins.input = real_input
    model_mode.ModelPolicy = real_policy
