"""
Safety review of the visit tools, check 2: with the NEW flags (the placing aid, the live go-ahead window with the faint
reference picture), what is asked of the robot, in which order, and what is the person told, when a reading or a picture
fails, on Esc and on Ctrl+C?

robot_run of the edited driver runs on the recording stand-in link of check 1 (no robot can be reached). The terminal's
Enter and the picture window's keys are stand-ins; a failure is planned for one call of the link, or a KeyboardInterrupt
is raised where Ctrl+C would raise it.

Read in the output: "asked of the robot" must never show a move or a motor switch that the run without flags does not
have; "tip ... cm high at the end" tells whether the tip was lifted (1.3 = still at pushing height, 6.3 = lifted).

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe s02_new_flags_when_something_fails.py > s02_new_flags_when_something_fails.out.txt
"""

import builtins
import contextlib
import io
import tempfile
from pathlib import Path

import cv2
import numpy

import s01_without_flags_committed_against_edited as check_1

edited = check_1.edited


def grey_jpeg(grey):
    """A camera-sized JPEG of one shade of grey."""

    return cv2.imencode(".jpg", numpy.full((480, 640, 3), grey, dtype = numpy.uint8))[1].tobytes()


def one_run(scenario, run_folder):
    """robot_run with the new settings of ``scenario`` on a recording link; what was asked, said and left behind."""

    link = check_1.RecordingLink(scenario.get("link_failures"))
    link.move_joints(scenario.get("joints_placed", check_1.RECORDED_START))
    link.calls.clear()
    link.counts.clear()
    if "pictures" in scenario:
        # the camera's first answers are given by the scenario (bytes, or something to raise); then the simulated camera
        pictures = list(scenario["pictures"])
        simulated_camera = link.read_picture_bytes

        def planned_camera():

            if not pictures:
                return simulated_camera()
            link.note("read_picture_bytes")
            answer = pictures.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        link.read_picture_bytes = planned_camera
    window_keys = list(scenario.get("window_keys", ["enter"]))
    terminal_keys = list(scenario.get("terminal_keys", []))
    looks = {"window": 0, "terminal": 0}
    prompts, shown, printed = [], [], io.StringIO()

    def stand_in_input(prompt = ""):

        prompts.append(prompt)
        print(f"<PROMPT {len(prompts)}, after {len(link.calls)} calls to the link: {prompt.strip()[:58]} ...>")
        raising = scenario.get("input_raises", {}).get(len(prompts))
        if raising is not None:
            raise raising
        return ""

    def stand_in_window_key(wait_ms):

        looks["window"] += 1
        raising = scenario.get("window_key_raises", {}).get(looks["window"])
        if raising is not None:
            raise raising
        return window_keys.pop(0) if window_keys else None

    def stand_in_enter_pressed():

        looks["terminal"] += 1
        raising = scenario.get("terminal_key_raises", {}).get(looks["terminal"])
        if raising is not None:
            raise raising
        return terminal_keys.pop(0) if terminal_keys else None

    real_parts = (builtins.input, edited.show_in_window, edited.read_window_key, edited.msvcrt, edited.is_terminal_a_console, edited.enter_pressed)
    builtins.input = stand_in_input
    edited.show_in_window = lambda picture: shown.append(picture.shape)
    edited.read_window_key = stand_in_window_key
    edited.msvcrt = check_1.NoKeyboard()
    edited.is_terminal_a_console = lambda: True
    edited.enter_pressed = stand_in_enter_pressed
    outcome = {"raised": None, "result": None}
    try:
        with contextlib.redirect_stdout(printed):
            try:
                outcome["result"] = edited.robot_run("check", dict(scenario.get("overrides", {}), run_folder = run_folder), stand_in_link = link,
                                                     stand_in_policy = check_1.StandInPolicy())
            except BaseException as error:
                outcome["raised"] = f"{type(error).__name__}: {error}".rstrip(": ")
    finally:
        builtins.input, edited.show_in_window, edited.read_window_key, edited.msvcrt, edited.is_terminal_a_console, edited.enter_pressed = real_parts
    names = [call[0] + (" OFF" if call[1:2] == (True,) else " ON" if call[1:2] == (False,) else "") for call in link.calls]
    outcome.update({"asked": squeezed(names), "shown": len(shown), "printed": printed.getvalue(), "are_motors_off_at_the_end": link.is_free_motion,
                    "tip_height_cm_at_the_end": round(100 * float(check_1.magnet_tip_and_lean(link.joints)[0][2]), 1)})
    return outcome


def squeezed(names):
    """``names`` in order, with a call repeated in a row written once with its count."""

    counted = []
    for name in names:
        if counted and counted[-1][0] == name:
            counted[-1][1] += 1
        else:
            counted.append([name, 1])
    return ", ".join(name if count == 1 else f"{name} x{count}" for name, count in counted)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        ghost_path = Path(folder) / "ghost.jpg"
        ghost_path.write_bytes(grey_jpeg(0))
        aid = {"is_place_help_shown": True, "place_help_refresh_s": 0.0, "step_budget": 2}
        ghost = {"ghost_path": ghost_path, "step_budget": 2}
        scenarios = {
            "no new flag (for comparison)": {"overrides": {"step_budget": 2}},
            "placing aid: Enter at the fourth look at the keyboard": {"overrides": aid, "terminal_keys": [None, None, None, "enter"]},
            "placing aid: the third reading of the state fails": {"overrides": aid, "link_failures": {"read_state": (3, RuntimeError("no answer from the robot"))}},
            "placing aid: Ctrl+C while the arm is guided (between two readings)": {"overrides": aid, "terminal_key_raises": {4: KeyboardInterrupt()}},
            "placing aid: Ctrl+C raised inside a reading": {"overrides": aid, "link_failures": {"read_state": (2, KeyboardInterrupt())}},
            "live window: Enter after three live pictures": {"overrides": ghost, "window_keys": [None, None, "enter"]},
            "live window: Esc": {"overrides": ghost, "window_keys": [None, "x", "escape"]},
            "live window: the camera raises at the third live picture": {"overrides": ghost, "window_keys": [None] * 20,
                                                                         "pictures": [grey_jpeg(200), grey_jpeg(190), RuntimeError("Command GET_IMAGE_COMPRESSED KO")]},
            "live window: Ctrl+C": {"overrides": ghost, "window_keys": [None] * 20, "window_key_raises": {3: KeyboardInterrupt()}},
            "live window: one empty picture, then good ones": {"overrides": ghost, "window_keys": [None, None, "enter"], "pictures": [b"", grey_jpeg(100)]},
            "both aids together": {"overrides": dict(aid, **ghost), "terminal_keys": [None, None, "enter"], "window_keys": [None, "enter"]},
        }
        for name, scenario in scenarios.items():
            outcome = one_run(scenario, folder)
            result = None if outcome["result"] is None else {key: outcome["result"].get(key) for key in ("ending", "moves", "is_lifted")}
            print("=" * 118)
            print(name)
            print("   asked of the robot:", outcome["asked"])
            print(f"   result: {result} | raised: {outcome['raised']} | tip {outcome['tip_height_cm_at_the_end']} cm high at the end | "
                  f"motors off at the end: {outcome['are_motors_off_at_the_end']} | pictures given to the window: {outcome['shown']}")
            print("   printed:")
            for line in outcome["printed"].replace("\r", "\n<the line is written over> ").splitlines():
                if line.strip():
                    print("      |", line.rstrip()[:175])
