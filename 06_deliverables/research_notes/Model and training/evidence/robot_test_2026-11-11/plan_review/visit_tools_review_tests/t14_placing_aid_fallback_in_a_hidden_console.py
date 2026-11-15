"""
Review check in a hidden console of its own, with the real input() and the real msvcrt: the placing aid fails in the
middle (a reading of the robot raises), and the plain prompt that follows must wait for a real Enter. Also: an Enter
pressed in the aid's first half second, which the aid throws away, must not answer anything later by itself.
No robot: pyniryo is replaced, the link is the simulated one. Keys are typed into this hidden console only.

    python console_fallback_check.py <result file>
"""

import builtins
import io
import math
import sys
import tempfile
import threading
import time
import traceback
import types
from pathlib import Path

lines = []
try:
    class Refused:

        def __init__(self, *arguments, **named_arguments):

            raise AssertionError("a connection to the robot was attempted")

    stand_in_pyniryo = types.ModuleType("pyniryo")
    stand_in_pyniryo.NiryoRobot = stand_in_pyniryo.JointsPosition = stand_in_pyniryo.PoseObject = Refused
    sys.modules["pyniryo"] = stand_in_pyniryo
    sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import typed_keys
    import model_mode
    import robot_link
    from config import MODEL_MODE_PARAMETERS

    robot_link.RobotLink = Refused

    class WatchedLink(robot_link.SimulatedRobotLink):

        def __init__(self, failing_reading = None):

            super().__init__()
            self.asked = []
            self.readings = 0
            self.failing_reading = failing_reading

        def set_free_motion(self, is_enabled):

            self.asked.append((round(time.monotonic() - START, 2), "motors off" if is_enabled else "motors on"))
            super().set_free_motion(is_enabled)

        def read_state(self):

            self.readings += 1
            self.asked.append((round(time.monotonic() - START, 2), "read state"))
            if self.readings == self.failing_reading:
                raise ConnectionError("the robot did not answer")
            return super().read_state()

        def move_joints(self, joints, velocity_percent = None):

            self.asked.append((round(time.monotonic() - START, 2), "move"))
            super().move_joints(joints, velocity_percent)

        def close(self):

            self.asked.append((round(time.monotonic() - START, 2), "close"))

    class SteadyPolicy:

        run_name = "steady_stand_in"
        checkpoint_paths = []

        def decide(self, picture_bytes):

            return {"tip_forward_cm": 0.8, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0, "angle_deg": 0.0, "stop_probability": 0.0}

    def typist(plan):

        for at, name in plan:
            time.sleep(max(0.0, at - (time.monotonic() - START)))
            typed_keys.press(name)

    def flow(title, settings, plan, failing_reading = None):

        global START
        link = WatchedLink(failing_reading)
        robot_link.SimulatedRobotLink.move_joints(link, [math.radians(value) for value in MODEL_MODE_PARAMETERS["recorded_start_joints_deg"]])
        prompts = []
        real_input = builtins.input
        model_mode.show_in_window = lambda picture: None
        model_mode.read_window_key = lambda wait_ms: "escape"

        def recorded_input(prompt = ""):

            prompts.append((round(time.monotonic() - START, 2), prompt[:34]))
            answer = real_input(prompt)
            prompts.append((round(time.monotonic() - START, 2), "  ... answered"))
            return answer

        builtins.input = recorded_input
        START = time.monotonic()
        thread = threading.Thread(target = typist, args = (plan,), daemon = True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                result = model_mode.robot_run("console_check", dict(settings, run_folder = folder), stand_in_link = link, stand_in_policy = SteadyPolicy())
            outcome = f"ending {result['ending']}"
        except BaseException as error:
            outcome = f"{type(error).__name__} came out of robot_run: {error}"
        finally:
            builtins.input = real_input
        thread.join(timeout = 5)
        lines.append(f"== {title}: {outcome}")
        lines.append(f"   prompts (seconds, text): {prompts}")
        lines.append(f"   asked of the link (seconds, what): {link.asked}")

    START = time.monotonic()
    aid = {"step_budget": 2, "is_place_help_shown": True}
    # Enter at 0.5 s switches the motors off; the third reading of the aid fails (about 1.5 s); Enter at 3.0 s must be the one that ends the placing
    flow("the aid fails at its third reading, then the plain prompt", aid, [(0.5, "enter"), (3.0, "enter")], failing_reading = 3)
    # the same, but letters and an Enter are typed while the aid is still running, before it fails: they must not answer the plain prompt
    flow("keys and an Enter typed in the aid's first half second, then the aid fails", aid, [(0.5, "enter"), (0.7, "a"), (0.8, "enter"), (3.5, "enter")], failing_reading = 3)
    # an Enter in the aid's first half second is thrown away; the next one, at 2.6 s, ends the placing
    flow("an Enter in the aid's first half second, a second one later", aid, [(0.5, "enter"), (0.8, "enter"), (2.6, "enter")])
    # the aid is wanted and --fresh-keys too, with two Enters typed before the first prompt
    for name in ("enter", "enter"):
        typed_keys.press(name)
    time.sleep(0.2)
    flow("two Enters typed before the first prompt, with the aid and fresh keys", dict(aid, is_typed_ahead_discarded = True), [(1.0, "enter"), (3.0, "enter")])
except BaseException:
    lines.append(traceback.format_exc())
finally:
    Path(sys.argv[1]).write_text("\n".join(lines), encoding = "utf-8")
