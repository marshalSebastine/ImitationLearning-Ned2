"""
Review attack 9: --fresh-keys when the terminal is not a console window (keys come through a pipe, as in Git Bash's own
window or an editor's output pane). A watched simulated link stands in for the robot; the real input() is used.

    python attack_fresh_keys_no_console.py child     run by the parent below with a pipe as its keyboard
"""

import subprocess
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] == "child":
    import math
    import tempfile

    import safe_import
    from safe_import import model_mode, robot_link
    from config import MODEL_MODE_PARAMETERS

    START = time.monotonic()
    asked = []

    class Watched(robot_link.SimulatedRobotLink):

        def set_free_motion(self, is_enabled):
            asked.append((round(time.monotonic() - START, 2), "motors off" if is_enabled else "motors on"))
            super().set_free_motion(is_enabled)

        def close(self):
            pass

    class Steady:
        run_name = "steady"
        checkpoint_paths = []

        def decide(self, picture_bytes):
            return {"tip_forward_cm": 0.8, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0, "angle_deg": 0.0, "stop_probability": 0.0}

    link = Watched()
    robot_link.SimulatedRobotLink.move_joints(link, [math.radians(value) for value in MODEL_MODE_PARAMETERS["recorded_start_joints_deg"]])
    model_mode.show_in_window = lambda picture: None
    model_mode.read_window_key = lambda wait_ms: "escape"
    # the model "loads" for two seconds; the Enter typed during that time is already waiting in the pipe
    time.sleep(2.0)
    START = time.monotonic()
    with tempfile.TemporaryDirectory() as folder:
        model_mode.robot_run("review", {"run_folder": folder, "is_typed_ahead_discarded": sys.argv[2] == "fresh", "is_place_help_shown": sys.argv[3] == "aid"}, stand_in_link = link, stand_in_policy = Steady())
    print("\nRESULT is the terminal a console:", model_mode.is_terminal_a_console(), "| asked (seconds after the first prompt could be shown):", asked)
else:
    for fresh in ("fresh", "plain"):
        for aid in ("no_aid", "aid"):
            child = subprocess.Popen([sys.executable, __file__, "child", fresh, aid], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)
            # one Enter pressed too early, while the model loads; a second and third one much later, as a person would
            child.stdin.write("\n")
            child.stdin.flush()
            time.sleep(4.0)
            child.stdin.write("\n")
            child.stdin.flush()
            time.sleep(1.0)
            child.stdin.write("\n\n")
            child.stdin.flush()
            output = child.communicate(timeout = 60)[0]
            lines = output.splitlines()
            print(f"--- is_typed_ahead_discarded = {fresh == 'fresh'}, placing aid wanted = {aid == 'aid'}")
            for line in lines:
                if "ignored" in line or "RESULT" in line or "placing aid" in line or "Traceback" in line or "Error" in line:
                    print("   ", line.strip()[:330])
