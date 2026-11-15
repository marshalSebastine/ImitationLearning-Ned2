"""
Tests of model_mode.py; no robot is needed (the simulated robot, the arm model and the recorded data stand in), and no
test can reach one: the real RobotLink is replaced below by a class that refuses to be made.

Run from 02_code:
    python test_model_mode.py
"""

import builtins
import contextlib
import io
import json
import math
import os
import signal
import tempfile
from pathlib import Path

import cv2
import numpy

import model_mode
import robot_link
from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height
from model_mode import (LinkWithInterruptKept, ModelRun, StopRule, command_line, enter_pressed, escape_pressed_in_terminal, escape_pressed_with_queue_emptied,
                        executor_check, ghost_blend, go_ahead_with_ghost, height_above_pushing_m, held_joints_at, model_step,
                        offset_from_recorded_start, picture_axes,
                        placed_point_problem, placing_help, placing_line, reference_picture, rehearsal_hand_path, rehearsal_run,
                        replay_check, robot_run, simulated_run, start_offset_lines, window_lines, with_interrupt_kept)
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed


class RefusedRobotLink:
    """Stands in the real RobotLink's place while the tests run, so that no test can connect to the robot."""

    def __init__(self, *arguments, **named_arguments):

        raise AssertionError("a test tried to connect to the real robot")


robot_link.RobotLink = RefusedRobotLink

# the placing aid cuts its line at the terminal's width; the tests fix that width, so that they give the same result in
# whatever window they are run
os.environ["COLUMNS"] = "120"


def joints_behind(tip_xy: list, turn_deg: float = 0.0) -> list:
    """Joints with the tip at pushing height over ``tip_xy``, the forearm turned ``turn_deg`` and the wrist and hand held."""

    target = numpy.array([tip_xy[0], tip_xy[1], pushing_height(*tip_xy) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    joints = solve_with_wrist_fixed(target, math.radians(turn_deg), math.radians(MODEL_MODE_PARAMETERS["wrist_held_deg"]),
                                    [math.atan2(tip_xy[1], tip_xy[0]), -0.8, -0.6])
    joints[5] = math.radians(MODEL_MODE_PARAMETERS["hand_held_deg"])
    return joints


def test_stop_rule_needs_two_in_a_row() -> None:

    rule = StopRule(0.5, 2)
    assert [rule.update(value) for value in [0.6, 0.2, 0.7, 0.4, 0.9, 0.5]] == [False, False, False, False, False, True]


def test_forward_and_left_follow_the_picture() -> None:

    joints = joints_behind([0.25, 0.0])
    up_xy, left_xy = picture_axes(joints)
    for decision, expected in (({"tip_forward_cm": 0.7, "tip_left_cm": 0.0, "twist_deg": 0.0}, up_xy),
                               ({"tip_forward_cm": 0.0, "tip_left_cm": 0.4, "twist_deg": 0.0}, left_xy)):
        planned = model_step(joints, decision)
        assert planned["refusal"] is None, planned["refusal"]
        moved = magnet_tip_and_lean(planned["joints_next"])[0][:2] - planned["tip_now"][:2]
        assert abs(numpy.linalg.norm(moved) - MODEL_MODE_PARAMETERS["step_m"]) < 1e-4
        assert moved @ expected / numpy.linalg.norm(moved) > 0.9999


def test_twist_is_clamped_and_wrist_and_hand_are_held() -> None:

    joints = joints_behind([0.28, 0.02], turn_deg = 110.0)
    planned = model_step(joints, {"tip_forward_cm": 0.8, "tip_left_cm": 0.1, "twist_deg": 25.0})
    assert planned["refusal"] is None, planned["refusal"]
    assert abs(math.degrees(planned["joints_next"][3]) - MODEL_MODE_PARAMETERS["joint_4_most_deg"]) < 1e-9
    assert planned["is_turn_clamped"]
    planned = model_step(joints_behind([0.28, 0.02]), {"tip_forward_cm": 0.8, "tip_left_cm": 0.1, "twist_deg": -25.0})
    assert abs(planned["twist_used_deg"] + MODEL_MODE_PARAMETERS["twist_step_most_deg"]) < 1e-9
    assert abs(math.degrees(planned["joints_next"][4]) - MODEL_MODE_PARAMETERS["wrist_held_deg"]) < 1e-9
    assert abs(math.degrees(planned["joints_next"][5]) - MODEL_MODE_PARAMETERS["hand_held_deg"]) < 1e-9


def test_short_direction_keeps_the_last_one_and_far_steps_are_refused() -> None:

    joints = joints_behind([0.25, 0.0])
    planned = model_step(joints, {"tip_forward_cm": 0.01, "tip_left_cm": 0.0, "twist_deg": 0.0}, numpy.array([0.0, 1.0]))
    assert planned["is_direction_kept"] and planned["direction_xy"] @ numpy.array([0.0, 1.0]) > 0.9999
    near_edge = joints_behind([0.485, 0.0])
    up_xy, _ = picture_axes(near_edge)
    planned = model_step(near_edge, {"tip_forward_cm": 100 * up_xy[0], "tip_left_cm": 0.0, "twist_deg": 0.0})
    assert planned["refusal"] is not None and planned["joints_next"] is None


def test_predicted_length_rule_moves_what_the_model_asks() -> None:

    joints = joints_behind([0.25, 0.0])
    rule = {"step_length_rule": "predicted", "step_length_gain": 1.0, "step_most_m": 0.01}
    for forward_cm, expected_m in ((0.3, 0.003), (2.0, 0.01)):
        planned = model_step(joints, {"tip_forward_cm": forward_cm, "tip_left_cm": 0.0, "twist_deg": 0.0}, None, rule)
        assert planned["refusal"] is None, planned["refusal"]
        moved = magnet_tip_and_lean(planned["joints_next"])[0][:2] - planned["tip_now"][:2]
        assert abs(numpy.linalg.norm(moved) - expected_m) < 1e-4
    # a twist in place: the forearm turns by what the model asks and the tip stays where it is
    planned = model_step(joints, {"tip_forward_cm": 0.0, "tip_left_cm": 0.0, "twist_deg": 8.0}, None, rule)
    assert planned["refusal"] is None, planned["refusal"]
    assert numpy.linalg.norm(magnet_tip_and_lean(planned["joints_next"])[0][:2] - planned["tip_now"][:2]) < 1e-4
    assert abs(math.degrees(planned["joints_next"][3] - joints[3]) - 8.0) < 1e-6
    # the fixed rule still steps 1 cm whatever length is asked
    planned = model_step(joints, {"tip_forward_cm": 0.3, "tip_left_cm": 0.0, "twist_deg": 0.0})
    assert abs(numpy.linalg.norm(magnet_tip_and_lean(planned["joints_next"])[0][:2] - planned["tip_now"][:2]) - MODEL_MODE_PARAMETERS["step_m"]) < 1e-4


class HesitatingPolicy:
    """A stand-in for a model that always asks for a tiny step, no twist and no stop."""

    run_name = "hesitating_stand_in"
    checkpoint_paths = []

    def decide(self, picture_bytes: bytes) -> dict:

        return {"tip_forward_cm": 0.02, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0,
                "angle_deg": 0.0, "stop_probability": 0.0}


def test_a_hesitating_model_ends_the_run_as_stalled() -> None:

    with tempfile.TemporaryDirectory() as folder:
        summary = simulated_run({"run_folder": folder, "step_length_rule": "predicted", "stall_steps": 5, "step_budget": 20}, HesitatingPolicy())
        assert summary["ending"] == "stalled" and summary["pictures"] == 5 and summary["is_lifted"], summary


def test_lift_keeps_the_tip_above_its_point() -> None:

    joints = joints_behind([0.30, -0.05], turn_deg = 30.0)
    lifted = held_joints_at(joints, 0.05)
    before, after = magnet_tip_and_lean(joints)[0], magnet_tip_and_lean(lifted)[0]
    assert numpy.linalg.norm(after[:2] - before[:2]) < 1e-4 and abs(after[2] - before[2] - 0.05) < 1e-4


def test_executor_gives_back_recorded_steps() -> None:

    results = executor_check({"executor_check_rows": 300})
    assert results["direction_error_deg"].median() < 1.0
    # the planned joint-4 change against the recorded one, on the steps whose twist needed no clamp
    assert results["twist_error_deg"].dropna().median() < 0.1
    assert results["refusal"].notna().mean() < 0.02


def test_replay_gives_back_saved_predictions() -> None:

    results = replay_check({"replay_rows": 40})
    assert results["angle_difference_deg"].max() < 0.01
    assert results["stop_difference"].max() < 1e-4


def test_simulated_run_takes_one_centimetre_steps() -> None:

    import pandas

    with tempfile.TemporaryDirectory() as folder:
        summary = simulated_run({"run_folder": folder, "step_budget": 10})
        steps = pandas.read_csv(f"{summary['folder']}/steps.csv")
        moving = steps[steps["action"] == "model_step"]
        assert summary["ending"] in ("stopped", "step_budget_reached", "refused") and summary["is_lifted"]
        assert summary["moves"] >= 1 and summary["moves"] == len(moving)
        # what was really done: each executed step's planned point is where the next picture finds the tip
        for position in moving.index:
            following = steps.loc[position + 1]
            assert abs(following["tip_x"] - steps.loc[position, "tip_next_x"]) < 1e-4
            assert abs(following["tip_y"] - steps.loc[position, "tip_next_y"]) < 1e-4
        lengths = numpy.hypot(moving["tip_next_x"] - moving["tip_x"], moving["tip_next_y"] - moving["tip_y"])
        assert numpy.allclose(lengths, MODEL_MODE_PARAMETERS["step_m"], atol = 1e-4)


def test_team_of_two_averages_its_members() -> None:

    from model_mode import ModelPolicy

    folder = MODEL_MODE_PARAMETERS["checkpoint_path"].parent.parent.parent
    paths = [folder / name / "checkpoints" / "all_sheets.pt" for name in ("2026-11-10_resnet18_model_1_all_sheets", "2026-11-10_resnet18_model_1_all_sheets_seed_2")]
    picture_bytes = open(f"{MODEL_MODE_PARAMETERS['run_folder']}/../raw/main_data/" + __import__("pandas").read_csv(
        __import__("config").TRAINING_DATA_PARAMETERS["processed_folder"] / "frames.csv", usecols = ["frame_path"], nrows = 1)["frame_path"][0], "rb").read()
    single = [ModelPolicy(checkpoint_path = path).decide(picture_bytes) for path in paths]
    team = ModelPolicy(checkpoint_path = paths).decide(picture_bytes)
    assert abs(team["tip_forward_cm"] - (single[0]["tip_forward_cm"] + single[1]["tip_forward_cm"]) / 2) < 1e-5
    assert abs(team["stop_probability"] - (single[0]["stop_probability"] + single[1]["stop_probability"]) / 2) < 1e-6


def test_two_picture_model_gets_the_picture_before_and_forgets_it_between_runs() -> None:

    import cv2
    import numpy
    from model_mode import ModelPolicy

    policy = ModelPolicy.__new__(ModelPolicy)
    policy.settings = dict(__import__("config").TRAINING_RUN_PARAMETERS, picture_count = 2)
    policy.picture_count = 2
    policy.earlier_pictures = []
    jpegs = [cv2.imencode(".jpg", numpy.full((480, 640, 3), shade, dtype = numpy.uint8))[1].tobytes() for shade in (40, 200)]
    first = policy.picture_input(jpegs[0])
    second = policy.picture_input(jpegs[1])
    assert first.shape == (1, 6, 120, 160)
    assert __import__("torch").allclose(first[:, :3], first[:, 3:]), "the first picture of a run must stand in for its previous one"
    assert __import__("torch").allclose(second[:, 3:], first[:, :3]), "the second picture must carry the first behind it"
    policy.forget_earlier_pictures()
    again = policy.picture_input(jpegs[1])
    assert __import__("torch").allclose(again[:, :3], again[:, 3:]), "a new run must not see the last run's picture"


class SteadyPolicy:
    """A stand-in for a model that always asks for 0.8 cm straight up the picture, no twist and no stop."""

    run_name = "steady_stand_in"
    checkpoint_paths = []

    def decide(self, picture_bytes: bytes) -> dict:

        return {"tip_forward_cm": 0.8, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0,
                "angle_deg": 0.0, "stop_probability": 0.0}


def refusal_of(arguments: list) -> str:
    """What the command line says when it refuses ``arguments``; fails when it takes them."""

    complaint = io.StringIO()
    try:
        with contextlib.redirect_stderr(complaint):
            command_line(arguments)
    except SystemExit:
        return complaint.getvalue()
    raise AssertionError(f"{arguments} was not refused")


def test_step_cm_flag_is_read_and_refused_where_it_does_not_fit() -> None:

    # no flag, no override: the settings of config.py stay as they are
    assert command_line(["simulated"])[1] == {} and command_line(["robot"])[1] == {}
    assert command_line(["simulated", "--step-cm", "0.65"])[1] == {"step_m": 0.0065}
    assert command_line(["simulated", "--step-cm", "0.65", "--step-length", "fixed"])[1] == {"step_m": 0.0065, "step_length_rule": "fixed"}
    # the two do not go together: with the predicted rule the model's own length is used
    complaint = refusal_of(["simulated", "--step-cm", "0.65", "--step-length", "predicted"])
    assert "--step-cm" in complaint and "predicted" in complaint, complaint
    # a typing mistake cannot ask for a long step, none at all or one backwards; nor for one so short that the stall
    # rule would end a run that drives straight on (below 0.104 cm; the allowed lengths begin at 0.3 cm)
    for wrong in ("6.5", "65", "0", "-0.65", "nan", "0.1", "0.29"):
        assert "--step-cm" in refusal_of(["simulated", "--step-cm", wrong]), wrong
    assert command_line(["simulated", "--step-cm", "0.3"])[1] == {"step_m": 0.003}
    # the refusal names the length as it was typed, not rounded onto the limit
    assert "--step-cm 0.2999999 is not a step length between 0.3 and 1 cm" in refusal_of(["simulated", "--step-cm", "0.2999999"])
    # the rehearsal is a part of its own and takes the robot part's flags
    arguments, overrides = command_line(["rehearsal", "--place-help"])
    assert arguments.part == "rehearsal" and overrides == {"is_place_help_shown": True}, overrides
    # the other new flags override their own settings only
    arguments, overrides = command_line(["robot", "--stall", "log", "--place-help", "--ghost", "first.jpg", "--fresh-keys", "--run-folder", "somewhere"])
    assert overrides == {"stall_action": "log", "is_place_help_shown": True, "ghost_path": Path("first.jpg"), "is_typed_ahead_discarded": True,
                         "run_folder": Path("somewhere")}, overrides
    assert arguments.label is None


def test_simulated_run_with_step_cm_logs_shorter_tip_moves() -> None:

    import pandas

    with tempfile.TemporaryDirectory() as folder:
        for flags, expected_cm in ((["--step-cm", "0.65"], 0.65), ([], 1.0)):
            overrides = command_line(["simulated"] + flags)[1]
            summary = simulated_run(dict(overrides, run_folder = folder, step_budget = 8), SteadyPolicy(), f"step_{expected_cm}")
            steps = pandas.read_csv(f"{summary['folder']}/steps.csv")
            assert summary["ending"] == "step_budget_reached" and summary["moves"] == 8 and len(steps) == 8, summary
            # what was really done: from each picture's tip to the next picture's tip, 6.5 mm or 10 mm within 0.1 mm
            tip_moves = numpy.hypot(numpy.diff(steps["tip_x"]), numpy.diff(steps["tip_y"]))
            assert len(tip_moves) == 7 and numpy.allclose(tip_moves, expected_cm / 100.0, atol = 1e-4), tip_moves
            assert numpy.allclose(steps["step_cm"], expected_cm, atol = 1e-3)
            # the length used is in the summary, with the time from the first to the last picture
            assert summary["step_length_rule"] == "fixed" and summary["fixed_step_cm"] == expected_cm, summary
            assert summary["pictures"] == 8 and 0.0 < summary["seconds_first_to_last_picture"] <= summary["seconds"], summary
            # from the first row's time to the last row's, not the whole run's seconds (which also count the lift)
            assert abs(summary["seconds_first_to_last_picture"] - (steps["unix_time"].iloc[-1] - steps["unix_time"].iloc[0])) <= 0.06, summary
            assert summary["start_offset"] is None


def test_stall_rule_can_log_and_let_the_run_go_on() -> None:

    settings = {"step_length_rule": "predicted", "stall_steps": 5, "step_budget": 20}
    assert command_line(["simulated", "--stall", "log"])[1] == {"stall_action": "log"}
    with tempfile.TemporaryDirectory() as folder:
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            summary = simulated_run(dict(settings, run_folder = folder, stall_action = "log"), HesitatingPolicy(), "stall_log")
        # the rule fires at the fifth picture as before, is logged once, and the run goes on to its step budget. The
        # summary holds that picture's step number, which counts from 0 as the column "step" and the frame files do:
        # 4 for the fifth picture; the printed line gives both numbers
        assert summary["ending"] == "step_budget_reached" and summary["pictures"] == 20 and summary["is_lifted"], summary
        assert summary["stall_logged_at_picture"] == 4 and summary["stall_action"] == "log", summary
        assert printed.getvalue().count("the stall rule fired") == 1, printed.getvalue()
        assert "[step 4] the stall rule fired (the run's picture number 5; steps are counted from 0)" in printed.getvalue(), printed.getvalue()
        # under the model's own length the summary names no fixed step
        assert summary["step_length_rule"] == "predicted" and summary["fixed_step_cm"] is None, summary
        # "end", the default, ends the run as stalled and names no picture
        summary = simulated_run(dict(settings, run_folder = folder, **command_line(["simulated", "--stall", "end"])[1]), HesitatingPolicy(), "stall_end")
        assert summary["ending"] == "stalled" and summary["pictures"] == 5 and summary["stall_logged_at_picture"] is None, summary


def recorded_start_joints() -> list:
    """The recorded start of config.py in radians."""

    return [math.radians(value) for value in MODEL_MODE_PARAMETERS["recorded_start_joints_deg"]]


def test_start_offset_is_zero_at_the_recorded_start_and_its_signs_are_right() -> None:

    offset = offset_from_recorded_start(recorded_start_joints())
    assert abs(offset["tip_distance_cm"]) < 0.05 and abs(offset["forearm_deg"]) < 0.05, offset
    # the recorded start's tip lies where the notes put it: x 14.0 cm, y -1.5 cm in the robot's frame
    recorded_tip = magnet_tip_and_lean(recorded_start_joints())[0]
    assert abs(recorded_tip[0] - 0.140) < 0.001 and abs(recorded_tip[1] + 0.015) < 0.001, recorded_tip
    # 1 cm further out, made with the helper: 1.0 cm, all of it along x
    offset = offset_from_recorded_start(joints_behind([recorded_tip[0] + 0.01, recorded_tip[1]]))
    assert abs(offset["tip_distance_cm"] - 1.0) < 0.05 and abs(offset["tip_along_x_cm"] - 1.0) < 0.05 and abs(offset["tip_along_y_cm"]) < 0.05, offset
    # Two hand-made joint sets settle which way "left" is. Joint 1 turned 4 degrees counter-clockwise seen from above
    # swings the tip to the left of someone at the base looking out along x (towards +y); 4 degrees clockwise swings it
    # to the right. Worked out by hand: the tip is 14.085 cm from the base's axis at -6.06 degrees, so at -2.06 degrees
    # it lies x +0.07 cm, y +0.98 cm from the recorded start, and at -10.06 degrees x -0.14 cm, y -0.97 cm.
    turned_left = offset_from_recorded_start([math.radians(value) for value in [-2.06, -41.87, -61.93, 0.0, 13.53, -6.15]])
    turned_right = offset_from_recorded_start([math.radians(value) for value in [-10.06, -41.87, -61.93, 0.0, 13.53, -6.15]])
    assert abs(turned_left["tip_along_x_cm"] - 0.07) < 0.02 and abs(turned_left["tip_along_y_cm"] - 0.98) < 0.02, turned_left
    assert abs(turned_right["tip_along_x_cm"] + 0.14) < 0.02 and abs(turned_right["tip_along_y_cm"] + 0.97) < 0.02, turned_right
    assert "1.0 cm to the left" in start_offset_lines(turned_left)[0] and "1.0 cm to the right" in start_offset_lines(turned_right)[0]
    # the forearm: joint 4 minus the recorded joint 4, with its sign
    for forearm_deg in (2.1, -4.0):
        twisted = offset_from_recorded_start([math.radians(value) for value in [-6.06, -41.87, -61.93, forearm_deg, 13.53, -6.15]])
        assert abs(twisted["forearm_deg"] - forearm_deg) < 1e-9, twisted


def test_start_offset_lines_warn_only_beyond_the_limits() -> None:

    close = {"tip_distance_cm": 0.78, "tip_along_x_cm": 0.6, "tip_along_y_cm": 0.5, "forearm_deg": 2.1}
    lines = start_offset_lines(close)
    assert len(lines) == 1, lines
    assert lines[0].startswith("start: tip 0.8 cm from the recorded start (0.6 cm further out, 0.5 cm to the left"), lines[0]
    assert "as seen from the robot's base looking out along x" in lines[0] and lines[0].endswith("forearm +2.1 degrees"), lines[0]
    # beyond 1 cm, or beyond 3 degrees: a second line says how to place the start again; nothing is refused
    far_tip = dict(close, tip_distance_cm = 1.3, tip_along_x_cm = -1.2, tip_along_y_cm = -0.5)
    for far in (far_tip, dict(close, forearm_deg = -3.4)):
        lines = start_offset_lines(far)
        assert len(lines) == 2, lines
        assert lines[1] == ("the start is further off than 1 cm or 3 degrees: to place it again, press Esc in the picture window "
                            "and start the same run again"), lines[1]
    assert "(1.2 cm closer to the base, 0.5 cm to the right" in start_offset_lines(far_tip)[0]
    assert start_offset_lines(dict(close, forearm_deg = -3.4))[0].endswith("forearm -3.4 degrees")
    # the limits are settings
    assert len(start_offset_lines(far_tip, {"start_offset_warn_m": 0.02})) == 1
    # the limits are judged on the numbers as they are printed: 1.04 cm reads "1.0 cm" and gets no warning, 1.06 cm
    # reads "1.1 cm" and gets one; the same for the forearm at 3 degrees
    for near, beyond, name in ((1.04, 1.06, "tip_distance_cm"), (-3.04, -3.06, "forearm_deg"), (3.04, 3.06, "forearm_deg")):
        assert len(start_offset_lines(dict(close, **{name: near}))) == 1, (name, near)
        assert len(start_offset_lines(dict(close, **{name: beyond}))) == 2, (name, beyond)
    assert start_offset_lines(dict(close, tip_distance_cm = 1.04))[0].startswith("start: tip 1.0 cm from the recorded start")


def test_placing_help_shows_each_reading_until_enter() -> None:

    recorded_tip = magnet_tip_and_lean(recorded_start_joints())[0]
    readings = [joints_behind([recorded_tip[0] + 0.02, recorded_tip[1] + 0.015]), joints_behind([recorded_tip[0] - 0.01, recorded_tip[1] - 0.005]),
                recorded_start_joints(), recorded_start_joints(), recorded_start_joints()]
    taken = []

    def read_joints() -> list:

        taken.append(readings[len(taken)])
        return taken[-1]

    def read_key():

        # the stand-in keyboard: Enter arrives after the third reading
        return "enter" if len(taken) >= 3 else None

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        is_ended_by_enter = placing_help(read_joints, read_key, {"place_help_refresh_s": 0.0})
    assert is_ended_by_enter is True and len(taken) == 3, taken
    assert "LET GO of the arm, then press Enter ..." in printed.getvalue()
    # the legend says that the line names where the tip is, not the way to move it, and which way left is
    for words in ("where the tip IS against the recorded start", "Move the tip the other way", "as seen from the robot's base looking out along x",
                  "too high / too low = more than 2 cm"):
        assert words in printed.getvalue(), words
    # one line that overwrites itself: each reading follows a carriage return, and the only line feed is the last one
    live = printed.getvalue().split("press Enter ...\n", 1)[1]
    shown = [part.strip() for part in live.split("\r") if part.strip()]
    assert live.count("\n") == 1 and live.endswith("\n") and len(shown) == 3, repr(live)
    # the first reading does not take Enter yet and says so; from the second one on the line says that Enter is awaited
    assert shown[0] == "tip  2.5 cm off: 2.0 too far out, 1.5 too far left, forearm   +0.0 deg, height  +0.0 cm: not close   (wait)", shown[0]
    assert shown[1] == "tip  1.1 cm off: 1.0 too close, 0.5 too far right, forearm   +0.0 deg, height  +0.0 cm: not close   Enter = placed", shown[1]
    assert shown[2] == "tip  0.0 cm off: 0.0 too far out, 0.0 too far left, forearm   +0.0 deg, height  +0.0 cm: close   Enter = placed", shown[2]
    assert all(len(line) <= 119 for line in shown), [len(line) for line in shown]

    # the height against pushing height is shown, and "close" is never said where the run would be refused for its
    # height: straight above the recorded start the tip is 0.0 cm off, yet 3 cm and 5 cm up the run does not start
    for lift_cm, last_word in ((0.0, "close"), (1.5, "close"), (3.0, "too high"), (5.0, "too high")):
        lifted = held_joints_at(recorded_start_joints(), lift_cm / 100.0)
        line = placing_line(offset_from_recorded_start(lifted), None, height_above_pushing_m(lifted))
        assert line == f"tip  0.0 cm off: 0.0 too far out, 0.0 too far left, forearm   +0.0 deg, height {lift_cm:+5.1f} cm: {last_word}", line
        assert (placed_point_problem(lifted, MODEL_MODE_PARAMETERS) is None) is (last_word == "close"), line
    close = {"tip_distance_cm": 0.4, "tip_along_x_cm": -0.3, "tip_along_y_cm": -0.2, "forearm_deg": -1.0}
    assert placing_line(close, None, -0.025) == "tip  0.4 cm off: 0.3 too close, 0.2 too far right, forearm   -1.0 deg, height  -2.5 cm: too low"
    assert placing_line(dict(close, forearm_deg = 4.0), None, 0.004).endswith("forearm   +4.0 deg, height  +0.4 cm: not close")
    # without a height the line ends with the offset's own word
    assert placing_line(close) == "tip  0.4 cm off: 0.3 too close, 0.2 too far right, forearm   -1.0 deg: close"

    # an Enter that is there from the start (the key held down too long at the prompt before) ends the wait at the
    # second reading, not at the first
    taken.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        assert placing_help(read_joints, lambda: "enter", {"place_help_refresh_s": 0.0}) is True and len(taken) == 2, taken

    # Ctrl+C goes on to the caller, which asks about the motors; the live line is ended first
    def interrupted_key():

        raise KeyboardInterrupt

    taken.clear()
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            placing_help(read_joints, interrupted_key, {"place_help_refresh_s": 0.0})
        raise AssertionError("Ctrl+C was swallowed by the placing aid")
    except KeyboardInterrupt:
        assert printed.getvalue().endswith("\n"), repr(printed.getvalue()[-40:])

    # a reading that fails ends the aid, which says so; the caller then asks for Enter in the plain way
    def failed_reading() -> list:

        raise RuntimeError("no answer from the robot")

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert placing_help(failed_reading, lambda: None, {"place_help_refresh_s": 0.0}) is False
    assert "the placing aid stopped (no answer from the robot)" in printed.getvalue()

    # how many readings come before Enter counts is a setting
    taken.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        assert placing_help(read_joints, lambda: "enter", {"place_help_refresh_s": 0.0, "place_help_readings_before_enter": 4}) is True and len(taken) == 4, taken

    # a terminal too narrow for the line cuts its end off (the last word and the Enter hint); that is said once, on a
    # line of its own, and the readings still overwrite one another
    taken.clear()
    printed = io.StringIO()
    os.environ["COLUMNS"] = "80"
    try:
        with contextlib.redirect_stdout(printed):
            placing_help(read_joints, read_key, {"place_help_refresh_s": 0.0})
    finally:
        os.environ["COLUMNS"] = "120"
    live = printed.getvalue().split("press Enter ...\n", 1)[1]
    assert live.count("this terminal is 80 columns wide and cuts the end of the line below off") == 1 and live.count("\n") == 2, repr(live)
    assert all(len(part) == 79 for part in live.split("\n")[1].split("\r") if part), [len(part) for part in live.split("\n")[1].split("\r")]


def test_ctrl_c_during_a_question_to_the_robot_is_held_back_until_the_answer_is_complete() -> None:

    handler_before = signal.getsignal(signal.SIGINT)
    notes = []

    def reading_with_ctrl_c() -> list:

        # Ctrl+C arrives while the robot's answer is awaited; the answer is complete when the note is made
        signal.raise_signal(signal.SIGINT)
        notes.append("the reading was finished")
        return recorded_start_joints()

    # the plain call is cut off in the middle of the reading, which leaves the link to the robot out of step
    try:
        reading_with_ctrl_c()
        raise AssertionError("Ctrl+C was not raised")
    except KeyboardInterrupt:
        assert notes == [], notes
    # held back, it is raised only after the reading; and Ctrl+C is as it was afterwards
    try:
        with_interrupt_kept(reading_with_ctrl_c)
        raise AssertionError("Ctrl+C was swallowed")
    except KeyboardInterrupt:
        assert notes == ["the reading was finished"], notes
    assert signal.getsignal(signal.SIGINT) is handler_before
    # without Ctrl+C the call's own answer comes back, and its own error
    assert with_interrupt_kept(lambda: "the answer") == "the answer"
    try:
        with_interrupt_kept(lambda: 1 / 0)
        raise AssertionError("the call's error was swallowed")
    except ZeroDivisionError:
        assert signal.getsignal(signal.SIGINT) is handler_before
    # the placing aid reads the robot's state in this way: the reading is finished, then Ctrl+C goes on to the caller,
    # which asks about the motors
    notes.clear()
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            placing_help(reading_with_ctrl_c, lambda: None, {"place_help_refresh_s": 0.0})
        raise AssertionError("Ctrl+C was swallowed by the placing aid")
    except KeyboardInterrupt:
        assert notes == ["the reading was finished"] and printed.getvalue().endswith("\n"), (notes, repr(printed.getvalue()[-40:]))
    assert signal.getsignal(signal.SIGINT) is handler_before


class TypedKeys:
    """
    A stand-in for msvcrt: the keys waiting in the terminal, handed out one code per getch. A reader that keeps
    looking whether a key waits without ever reading one would never end; here it fails the test instead.
    """

    def __init__(self, characters: str):

        self.characters = list(characters)
        self.looks_without_a_read = 0

    def kbhit(self) -> bool:

        self.looks_without_a_read += 1 if self.characters else 0
        if self.looks_without_a_read > 10_000:
            raise AssertionError("a key waits in the terminal and is looked at again and again without being read")
        return bool(self.characters)

    def getch(self) -> bytes:

        self.looks_without_a_read = 0
        return self.characters.pop(0).encode("latin-1")

    def getwch(self) -> str:

        self.looks_without_a_read = 0
        return self.characters.pop(0)


def test_only_enter_ends_the_wait_in_the_terminal() -> None:

    real_msvcrt = model_mode.msvcrt
    try:
        # other keys, Esc, and an arrow or function key (two codes, even when the second one equals Enter's) are not Enter
        for typed, expected in (("", None), ("\r", "enter"), ("ab\r", "enter"), ("x \x1b", None), ("\xe0H", None), ("\x00\r", None), ("\xe0\r\r", "enter")):
            model_mode.msvcrt = TypedKeys(typed)
            assert enter_pressed() == expected, repr(typed)
            assert not model_mode.msvcrt.characters, repr(typed)
        # Ctrl+C read as a key interrupts, as it interrupts input()
        model_mode.msvcrt = TypedKeys("\x03")
        try:
            enter_pressed()
            raise AssertionError("Ctrl+C was swallowed")
        except KeyboardInterrupt:
            pass
        # Esc in the terminal, which the live go-ahead window reads: found among other keys, not taken from the second
        # code of an arrow or function key, and Enter is not Esc
        for typed, expected in (("", None), ("\x1b", "escape"), ("a\x1bb\r", "escape"), ("\r", None), ("\xe0\x1b", None), ("\x00\x1b\x1b", "escape")):
            model_mode.msvcrt = TypedKeys(typed)
            assert escape_pressed_in_terminal() == expected, repr(typed)
            assert not model_mode.msvcrt.characters, repr(typed)
    finally:
        model_mode.msvcrt = real_msvcrt


def test_esc_ends_a_run_with_fresh_keys_also_when_more_keys_wait_in_the_terminal() -> None:

    real_parts = (model_mode.msvcrt, model_mode.read_window_key)
    try:
        # the key reader of a run started with --fresh-keys: Esc is found wherever it stands (pressed twice, before
        # Enter, in the picture window while a key waits in the terminal), and no key is left waiting
        for typed, window_keys, expected in (("", [], None), ("ab\r", [], None), ("a", ["x"], None), ("\xe0\x1b", [], None),
                                             ("\x1b", [], "escape"), ("\x1b\x1b", [], "escape"), ("\x1b\r", [], "escape"), ("a\x1bb", [], "escape"),
                                             ("a", ["escape"], "escape"), ("\x1b", ["escape"], "escape"), ("", ["x", "escape", "y"], "escape")):
            model_mode.msvcrt = TypedKeys(typed)
            keys = list(window_keys)
            model_mode.read_window_key = lambda wait_ms: keys.pop(0) if keys else None
            assert escape_pressed_with_queue_emptied() == expected, (typed, window_keys)
            assert not model_mode.msvcrt.characters and not keys, (typed, window_keys)
    finally:
        model_mode.msvcrt, model_mode.read_window_key = real_parts

    # in the whole flow: Esc pressed twice in the terminal during a run ends the run as "quit", with the lift
    keyboard = TypedKeys("")
    shown = []

    def window_with_a_typist(picture) -> None:

        # the stand-in window: when it shows its third picture (the run's second), Esc is pressed twice in the terminal
        shown.append(picture)
        if len(shown) == 3:
            keyboard.characters.extend("\x1b\x1b")

    real_parts = (model_mode.msvcrt, model_mode.is_terminal_a_console)
    model_mode.msvcrt, model_mode.is_terminal_a_console = keyboard, lambda: True
    try:
        with tempfile.TemporaryDirectory() as folder:
            run = stand_in_robot_run(recorded_start_joints(), {"run_folder": folder, "step_budget": 30, "is_typed_ahead_discarded": True}, ["enter"],
                                     show = window_with_a_typist)
    finally:
        model_mode.msvcrt, model_mode.is_terminal_a_console = real_parts
    assert run["result"]["ending"] == "quit" and run["result"]["pictures"] == 2 and run["result"]["is_lifted"], run["result"]
    assert not keyboard.characters and run["asked"][-3:] == ["read state", "move", "close"], run["asked"]


def grey_jpeg(grey: int) -> bytes:
    """A camera-sized JPEG of one shade of grey."""

    return cv2.imencode(".jpg", numpy.full((480, 640, 3), grey, dtype = numpy.uint8))[1].tobytes()


def test_ghost_blend_lays_the_reference_over_the_live_picture() -> None:

    live = numpy.full((480, 640, 3), 200, dtype = numpy.uint8)
    live[100:200, 300:400] = (10, 20, 30)
    ghost = numpy.zeros((480, 640, 3), dtype = numpy.uint8)
    # strength 0 gives back the live picture, strength 1 the reference
    assert numpy.array_equal(ghost_blend(live, ghost, 0.0), live)
    assert numpy.array_equal(ghost_blend(live, ghost, 1.0), ghost)
    blended = ghost_blend(live, ghost, 0.35)
    assert blended.shape == live.shape and blended.dtype == live.dtype
    assert abs(int(blended[0, 0, 0]) - 130) <= 1 and abs(int(blended[150, 350, 2]) - 20) <= 1, blended[0, 0]
    # a reference of another size is resized to the live picture's: here its left half black, its right half white
    small_ghost = numpy.zeros((240, 320, 3), dtype = numpy.uint8)
    small_ghost[:, 160:] = 255
    blended = ghost_blend(live, small_ghost, 0.5)
    assert blended.shape == live.shape
    assert abs(int(blended[50, 100, 0]) - 100) <= 1 and abs(int(blended[50, 500, 0]) - 228) <= 1, (blended[50, 100], blended[50, 500])
    # the live picture itself is left as it was
    assert live[0, 0, 0] == 200 and live[150, 350, 0] == 10


def test_reference_picture_is_none_when_missing_or_not_a_picture() -> None:

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "first.jpg").write_bytes(grey_jpeg(90))
        (Path(folder) / "not_a_picture.jpg").write_text("no picture in here", encoding = "utf-8")
        (Path(folder) / "empty.jpg").write_bytes(b"")
        assert reference_picture(Path(folder) / "first.jpg").shape == (480, 640, 3)
        assert reference_picture(str(Path(folder) / "first.jpg")).shape == (480, 640, 3)
        for unreadable in ("missing.jpg", "not_a_picture.jpg", "empty.jpg", "."):
            assert reference_picture(Path(folder) / unreadable) is None, unreadable


def test_go_ahead_with_ghost_shows_live_pictures_until_a_key() -> None:

    ghost = numpy.zeros((240, 320, 3), dtype = numpy.uint8)
    for last_key, expected in (("enter", True), ("escape", False)):
        # the stand-in camera: a brighter picture, one that cannot be decoded, then darker ones
        pictures = [grey_jpeg(200), b"not a picture", grey_jpeg(100), grey_jpeg(60), grey_jpeg(20)]
        handed_out, shown, waits = [], [], []

        def read_picture_bytes() -> bytes:

            handed_out.append(pictures[min(len(handed_out), len(pictures) - 1)])
            return handed_out[-1]

        def read_key(wait_ms: int):

            # the stand-in window: another key first, which must not end the wait; Enter while the picture could not be
            # read, which must not count; and the last key after four shown pictures
            waits.append(wait_ms)
            if len(shown) >= 4:
                return last_key
            return {1: "x", 2: "enter"}.get(len(waits))

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            is_go_ahead_given = go_ahead_with_ghost(read_picture_bytes, read_key, ghost, {"ghost_strength": 0.5}, show = shown.append)
        assert is_go_ahead_given is expected
        # live: a new picture was asked for at every turn, and every turn showed something
        assert len(handed_out) == 4 and len(shown) == 4, (len(handed_out), len(shown))
        live = [shown[0], shown[2], shown[3]]
        assert all(picture.shape == (480, 640, 3) for picture in live), [picture.shape for picture in live]
        # each good one under the faint reference (black here, at half strength): half as bright below the text band
        assert all(abs(int(picture[400, 320, 0]) - half) <= 2 for picture, half in zip(live, (100, 50, 30))), [int(picture[400, 320, 0]) for picture in live]
        # the picture that could not be read: the reference alone under a band that says so, so that a still picture
        # is not taken for a live one; the terminal says it once, with the reason
        assert shown[1].shape == (240, 320, 3) and int(shown[1][35:].max()) == 0 and int(shown[1][:35].max()) > 200, shown[1].shape
        assert printed.getvalue().count("a camera picture could not be read (what came is not a picture)") == 1, printed.getvalue()
        # about three pictures a second, and the person is told that the arm's motors are on
        assert all(abs(wait - 1000 * MODEL_MODE_PARAMETERS["ghost_refresh_s"]) <= 1 for wait in waits), waits
        assert "Do not touch the arm" in printed.getvalue()
        # a reference picture of another size than the camera's is stretched to fit, and that is said once
        assert printed.getvalue().count("note: the reference picture is 320 x 240, the camera's picture 640 x 480") == 1, printed.getvalue()


def test_live_go_ahead_waits_for_the_person_when_the_camera_fails() -> None:

    ghost = numpy.full((480, 640, 3), 90, dtype = numpy.uint8)

    def outcome(
            read_picture_bytes,
            window_keys: list,
            terminal_keys: tuple = ()
    ) -> tuple:
        """
        The live window with a stand-in camera and stand-in keys (when the window's keys run out, Esc ends the wait, so
        that no case can wait for ever): what it returned, the pictures shown, and what was printed.
        """

        shown, printed = [], io.StringIO()
        window_keys, terminal_keys = list(window_keys), list(terminal_keys)
        with contextlib.redirect_stdout(printed):
            result = go_ahead_with_ghost(read_picture_bytes, lambda wait_ms: window_keys.pop(0) if window_keys else "escape", ghost, {}, show = shown.append,
                                         read_terminal_key = lambda: terminal_keys.pop(0) if terminal_keys else None)
        return result, shown, printed.getvalue()

    # a camera request that fails (the robot answers "KO") does not end the wait: the window shows the reference alone,
    # Enter does not count under it, and with the next picture Enter starts the run
    calls = []

    def failing_at_the_second() -> bytes:

        calls.append("asked")
        if len(calls) == 2:
            raise RuntimeError("Command GET_IMAGE_COMPRESSED KO\nmore")
        return grey_jpeg(200)

    result, shown, text = outcome(failing_at_the_second, [None, "enter", "enter"])
    assert result is True and len(calls) == 3 and len(shown) == 3, (result, len(calls), len(shown))
    assert numpy.array_equal(shown[1][35:], ghost[35:]) and not numpy.array_equal(shown[0][35:], ghost[35:])
    assert text.count("a camera picture could not be read (RuntimeError: Command GET_IMAGE_COMPRESSED KO)") == 1 and "note: the reference" not in text, text

    # no picture ever (nothing comes, or every request fails): a window is there from the first turn, Enter never
    # counts, nothing ends the wait by itself, and Esc ends it
    def always_failing() -> bytes:

        raise OSError("the link to the robot is lost")

    for camera in (lambda: b"", always_failing):
        result, shown, text = outcome(camera, ["enter", None, "enter", None, None, "escape"])
        assert result is False and len(shown) == 6 and all(numpy.array_equal(picture[35:], ghost[35:]) for picture in shown), (result, len(shown))
        assert text.count("a camera picture could not be read") == 1, text

    # Esc in the terminal ends the wait as Esc in the picture window does
    result, shown, text = outcome(lambda: grey_jpeg(200), [None, None, None, "enter"], terminal_keys = (None, "escape"))
    assert result is False and len(shown) == 2, (result, len(shown))

    # Ctrl+C is taken as Esc. During a camera request it is held back until the answer is complete ...
    notes = []

    def camera_with_ctrl_c() -> bytes:

        signal.raise_signal(signal.SIGINT)
        notes.append("the request was finished")
        return grey_jpeg(200)

    handler_before = signal.getsignal(signal.SIGINT)
    result, shown, text = outcome(camera_with_ctrl_c, ["enter"])
    assert result is False and notes == ["the request was finished"] and shown == [] and "Ctrl+C: taken as Esc" in text, (result, notes, text)
    assert signal.getsignal(signal.SIGINT) is handler_before

    # ... and while the window waits for a key it ends the wait at once
    def interrupted_key(wait_ms: int):

        raise KeyboardInterrupt

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert go_ahead_with_ghost(lambda: grey_jpeg(200), interrupted_key, ghost, {}, show = lambda picture: None) is False
    assert "Ctrl+C: taken as Esc" in printed.getvalue()


def test_window_names_a_step_of_part_millimetres_in_full() -> None:

    row = {"step": 3, "action": "model_step", "model_stop_probability": 0.123, "model_twist_deg": -2.34, "model_angle_deg": 10.0, "step_cm": 0.65}
    assert window_lines(row, {"step_length_rule": "fixed"}) == ["step 3   model_step   stop chance 0.12 (stops at 0.50 twice)",
                                                                  "twist -2.3 deg   step 0.65 cm   Esc = end the run"]
    # a whole millimetre keeps its one decimal, as on 11.11.26, and so does the model's own length
    assert window_lines(dict(row, step_cm = 1.0), {"step_length_rule": "fixed"})[1] == "twist -2.3 deg   step 1.0 cm   Esc = end the run"
    assert window_lines(dict(row, step_cm = 0.312), {"step_length_rule": "predicted"})[1] == "twist -2.3 deg   step 0.3 cm   Esc = end the run"
    # the window draws these two lines on its dark band: for a 0.65 cm step the band is the one written with "0.65 cm"
    shown = []
    real_show = model_mode.show_in_window
    model_mode.show_in_window = shown.append
    try:
        for step_cm in (0.65, 0.7):
            ModelRun(None, SteadyPolicy(), {"step_m": step_cm / 100.0}).show(grey_jpeg(128), dict(row, step_cm = step_cm))
    finally:
        model_mode.show_in_window = real_show
    expected_band = numpy.zeros((58, 640, 3), dtype = numpy.uint8)
    for number, line in enumerate(window_lines(row, {"step_length_rule": "fixed"})):
        cv2.putText(expected_band, line, (10, 22 + 26 * number), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    assert numpy.array_equal(shown[0][:58], expected_band) and not numpy.array_equal(shown[1][:58], expected_band)


class WatchedSimulatedLink(robot_link.SimulatedRobotLink):
    """The simulated robot, keeping a list of what was asked of it (its pictures may be read without a note)."""

    def __init__(self):

        super().__init__()
        self.asked = []

    def set_free_motion(self, is_enabled: bool) -> None:

        self.asked.append("motors off" if is_enabled else "motors on")
        super().set_free_motion(is_enabled)

    def read_state(self) -> dict:

        self.asked.append("read state")
        return super().read_state()

    def move_joints(self, joints: list, velocity_percent = None) -> None:

        self.asked.append("move")
        super().move_joints(joints, velocity_percent)

    def close(self) -> None:

        self.asked.append("close")


class InterruptedMoveLink(WatchedSimulatedLink):
    """The watched simulated robot during whose chosen move Ctrl+C arrives; a move that is carried out to its end is noted."""

    def __init__(self, interrupted_move: int):

        super().__init__()
        self.interrupted_move = interrupted_move
        self.moves_begun = 0
        self.moves_finished = 0

    def move_joints(self, joints: list, velocity_percent = None) -> None:

        self.moves_begun += 1
        if self.moves_begun == self.interrupted_move:
            signal.raise_signal(signal.SIGINT)
        super().move_joints(joints, velocity_percent)
        self.moves_finished += 1


def test_ctrl_c_during_a_move_ends_a_run_with_fresh_keys_after_that_move() -> None:

    handler_before = signal.getsignal(signal.SIGINT)
    # the link of a run with --fresh-keys hands every call on and gives its answer back, and what is no call as it is
    inner = WatchedSimulatedLink()
    kept = LinkWithInterruptKept(inner)
    kept.move_joints(recorded_start_joints())
    assert kept.read_state()["joints"] == inner.read_state()["joints"] and kept.is_free_motion is False and inner.asked[:2] == ["move", "read state"], inner.asked
    real_console = model_mode.is_terminal_a_console
    model_mode.is_terminal_a_console = lambda: True
    try:
        with tempfile.TemporaryDirectory() as folder:
            for is_fresh, moves_finished in ((True, 5), (False, 4)):
                # Ctrl+C arrives during the run's second step, the link's fifth move (the placing by the stand-in hand and
                # the two settling moves come first)
                link = InterruptedMoveLink(interrupted_move = 5)
                run = stand_in_robot_run(recorded_start_joints(), {"run_folder": folder, "step_budget": 30, "is_typed_ahead_discarded": is_fresh}, ["enter"],
                                         link = link)
                # either way the run ends as "quit" with the lift, as the committed driver ends it on a stand-in link; but
                # only with fresh keys is the move under way carried out to its end first (on the real link an answer cut
                # in two leaves every later question waiting for ever)
                assert run["result"]["ending"] == "quit" and run["result"]["is_lifted"], run["result"]
                assert link.moves_begun == 6 and link.moves_finished == moves_finished + 1, (is_fresh, link.moves_begun, link.moves_finished)
    finally:
        model_mode.is_terminal_a_console = real_console
    assert signal.getsignal(signal.SIGINT) is handler_before


class FailingCameraLink(WatchedSimulatedLink):
    """The watched simulated robot whose camera request fails at chosen calls, as when the robot answers "KO"."""

    def __init__(self, failing_calls: set):

        super().__init__()
        self.failing_calls = failing_calls
        self.picture_calls = 0

    def read_picture_bytes(self) -> bytes:

        self.picture_calls += 1
        if self.picture_calls in self.failing_calls:
            raise RuntimeError("Command GET_IMAGE_COMPRESSED KO")
        return super().read_picture_bytes()


@contextlib.contextmanager
def stand_in_person(
        window_keys: list,
        at_prompt = None,
        show = None
):
    """
    In place of the person at the robot, while the block runs: prompts that are answered at once (``at_prompt``, when
    given, is called at each one); in place of the picture window a list of keys (``window_keys``, then none) and a
    list of the pictures shown (or ``show`` in the window's place); and no key waiting in the terminal, unless the
    test has put a stand-in keyboard there itself.

    Yields
    ------
    dict
        "prompts", "shown" (the pictures given to the window) and "printed" (an io.StringIO with what was printed).
    """

    keys = list(window_keys)
    seen = {"prompts": [], "shown": [], "printed": io.StringIO()}
    real_parts = (builtins.input, model_mode.show_in_window, model_mode.read_window_key, model_mode.msvcrt)
    waits_without_a_key = []

    def answered_at_once(prompt: str = "") -> str:

        seen["prompts"].append(prompt)
        if at_prompt is not None:
            at_prompt()
        return ""

    def window_key(wait_ms: int):

        if keys:
            return keys.pop(0)
        # no key is left. A run only looks whether a key is there (1 ms); a go-ahead window waits for one (0 = for
        # ever, the live window turn after turn) and would never get it: that fails the test instead of never ending
        waits_without_a_key.append(wait_ms)
        if wait_ms == 0 or sum(1 for wait in waits_without_a_key if wait > 1) > 100:
            raise AssertionError("the picture window waits for a key that the test does not give")
        return None

    builtins.input = answered_at_once
    model_mode.show_in_window = show or seen["shown"].append
    model_mode.read_window_key = window_key
    if not isinstance(model_mode.msvcrt, TypedKeys):
        model_mode.msvcrt = TypedKeys("")
    try:
        with contextlib.redirect_stdout(seen["printed"]):
            yield seen
    finally:
        builtins.input, model_mode.show_in_window, model_mode.read_window_key, model_mode.msvcrt = real_parts


def stand_in_robot_run(
        joints_placed: list,
        parameters: dict,
        window_keys: list,
        at_prompt = None,
        link = None,
        show = None
) -> dict:
    """
    robot_run without a robot: a watched simulated link (``link``, or a new one) whose arm a "hand" has already put at
    ``joints_placed``, the steady stand-in model, and stand_in_person in place of the person.

    Returns
    -------
    dict
        "result" (what robot_run returned), "asked" (what was asked of the link, in order), "prompts", "shown" (the
        pictures given to the window) and "printed".
    """

    link = link or WatchedSimulatedLink()
    link.move_joints(joints_placed)
    link.asked.clear()
    with stand_in_person(window_keys, at_prompt, show) as seen:
        result = robot_run("stand_in", parameters, stand_in_link = link, stand_in_policy = SteadyPolicy())
    return {"result": result, "asked": link.asked, "prompts": seen["prompts"], "shown": seen["shown"], "printed": seen["printed"].getvalue()}


PROMPTS_OF_11_11_26 = ["Put the puck at the line's start. Press Enter to switch the motors OFF (hold the arm: it will sag) ... ",
                       "Guide the magnet tip just behind the puck, facing along the line. LET GO of the arm, then press Enter ... "]


def test_robot_run_without_new_flags_asks_what_it_asked_before() -> None:

    with tempfile.TemporaryDirectory() as folder:
        run = stand_in_robot_run(recorded_start_joints(), {"run_folder": folder, "step_budget": 3}, ["enter"])
        # the two prompts word for word, then: motors off, the placed joints read, motors on, the two settling moves,
        # three steps (each one reading and one move), the lift, and the link closed
        assert run["prompts"] == PROMPTS_OF_11_11_26, run["prompts"]
        assert run["asked"] == ["motors off", "read state", "motors on", "move", "move"] + ["read state", "move"] * 3 + ["read state", "move", "close"], run["asked"]
        assert len(run["shown"]) == 4 and all(picture.shape == (480, 640, 3) for picture in run["shown"])
        result = run["result"]
        assert result["ending"] == "step_budget_reached" and result["moves"] == 3 and result["is_lifted"], result
        assert result["fixed_step_cm"] == 1.0 and result["stall_action"] == "end" and result["stall_logged_at_picture"] is None, result
        # the start's offset: one printed line, no warning at the recorded start, and the four numbers in the summary
        assert run["printed"].count("start: tip 0.0 cm from the recorded start") == 1 and "the start is further off" not in run["printed"], run["printed"]
        written = json.loads((Path(result["folder"]) / "summary.json").read_text(encoding = "utf-8"))
        assert written["start_offset"] == {"tip_distance_cm": 0.0, "tip_along_x_cm": 0.0, "tip_along_y_cm": 0.0, "forearm_deg": 0.0}, written["start_offset"]
        assert written["pictures"] == 3 and 0.0 < written["seconds_first_to_last_picture"] <= written["seconds"], written


def test_robot_run_with_the_placing_aid_and_the_ghost() -> None:

    recorded_tip = magnet_tip_and_lean(recorded_start_joints())[0]
    joints_placed = joints_behind([recorded_tip[0] + 0.012, recorded_tip[1]], turn_deg = 5.0)
    real_parts = (model_mode.is_terminal_a_console, model_mode.enter_pressed)
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "ghost.jpg").write_bytes(grey_jpeg(0))
        settings = {"run_folder": folder, "step_budget": 2, "step_m": 0.0065, "is_place_help_shown": True, "place_help_refresh_s": 0.0,
                    "ghost_path": Path(folder) / "ghost.jpg"}
        polls = []
        try:
            # the stand-in terminal: a console whose Enter arrives at the third look at the keyboard, after two readings
            model_mode.is_terminal_a_console = lambda: True
            model_mode.enter_pressed = lambda: polls.append("look") or ("enter" if len(polls) >= 3 else None)
            run = stand_in_robot_run(joints_placed, settings, [None, None, "enter"])
            # a terminal that is not a console: the aid says so and the plain prompt is used
            model_mode.is_terminal_a_console = lambda: False
            plain_run = stand_in_robot_run(joints_placed, dict(settings, ghost_path = None), ["escape"])
        finally:
            model_mode.is_terminal_a_console, model_mode.enter_pressed = real_parts
    # the aid takes the second prompt's place and only reads the state: two readings, then the placed joints as before
    assert run["prompts"] == PROMPTS_OF_11_11_26[:1], run["prompts"]
    assert run["asked"] == ["motors off", "read state", "read state", "read state", "motors on", "move", "move"] + ["read state", "move"] * 2 + ["read state", "move", "close"], run["asked"]
    assert "not close" in run["printed"] and "the start is further off than 1 cm or 3 degrees" in run["printed"], run["printed"]
    live = [part.strip() for part in run["printed"].split("\r") if part.strip().startswith("tip ")]
    assert len(live) == 2 and live[0] == "tip  1.2 cm off: 1.2 too far out, 0.0 too far left, forearm   +5.0 deg, height  +0.0 cm: not close   (wait)", live
    assert live[1].startswith("tip  1.2 cm off: 1.2 too far out, 0.0 too far left, forearm   +5.0 deg, height  +0.0 cm: not close   Enter = placed"), live
    # the go-ahead window was live: three pictures under the faint (black) reference, then the run's two pictures
    assert len(run["shown"]) == 5, len(run["shown"])
    assert all(abs(int(picture[400, 5, 0]) - 0.65 * 235) <= 3 for picture in run["shown"][:3]), [int(picture[400, 5, 0]) for picture in run["shown"]]
    assert all(abs(int(picture[400, 5, 0]) - 235) <= 3 for picture in run["shown"][3:]), [int(picture[400, 5, 0]) for picture in run["shown"]]
    result = run["result"]
    assert result["ending"] == "step_budget_reached" and result["moves"] == 2 and result["fixed_step_cm"] == 0.65, result
    assert abs(result["start_offset"]["tip_along_x_cm"] - 1.2) < 0.05 and abs(result["start_offset"]["forearm_deg"] - 5.0) < 0.05, result["start_offset"]
    assert plain_run["prompts"] == PROMPTS_OF_11_11_26 and "the placing aid reads Enter from a console window" in plain_run["printed"], plain_run["prompts"]
    assert plain_run["result"]["ending"] == "not_started", plain_run["result"]
    assert plain_run["asked"] == ["motors off", "read state", "motors on", "move", "move", "read state", "move", "close"], plain_run["asked"]


def test_keys_pressed_too_early_do_not_answer_the_first_prompt_with_fresh_keys() -> None:

    real_parts = (model_mode.msvcrt, model_mode.is_terminal_a_console)
    try:
        with tempfile.TemporaryDirectory() as folder:
            for is_console, is_discarded, expected_waiting in ((True, True, 0), (True, False, 2), (False, True, 2), (False, False, 2)):
                # two Enters pressed while the model was loading wait in the terminal when the robot part begins
                model_mode.is_terminal_a_console = lambda: is_console
                model_mode.msvcrt = TypedKeys("\r\r")
                waiting = []
                run = stand_in_robot_run(recorded_start_joints(), {"run_folder": folder, "is_typed_ahead_discarded": is_discarded}, ["escape"],
                                         at_prompt = lambda: waiting.append(len(model_mode.msvcrt.characters)))
                # with the setting, in a console window, they are thrown away before the first prompt, and the person is
                # told; without it nothing is touched, as on 11.11.26
                assert waiting[0] == expected_waiting and run["prompts"] == PROMPTS_OF_11_11_26, (waiting, run["prompts"])
                assert ("keys pressed before this prompt were ignored" in run["printed"]) is (is_console and is_discarded), run["printed"]
                # where the terminal is not a console window (Git Bash's own window, an editor's pane) early keys cannot
                # be thrown away; the setting then says so instead of promising it
                assert ("keys pressed early cannot be thrown away in this terminal" in run["printed"]) is (is_discarded and not is_console), run["printed"]
    finally:
        model_mode.msvcrt, model_mode.is_terminal_a_console = real_parts


def test_missing_ghost_is_reported_before_anything_is_asked_of_the_robot() -> None:

    with tempfile.TemporaryDirectory() as folder:
        run = stand_in_robot_run(recorded_start_joints(), {"run_folder": folder, "ghost_path": Path(folder) / "missing.jpg"}, ["enter"])
        assert run["result"]["ending"] == "not_started" and "missing.jpg" in run["result"]["reason"], run["result"]
        assert run["asked"] == [] and run["prompts"] == [] and run["shown"] == [], run
        assert "not started: the reference picture" in run["printed"]

        # the same without any stand-in for the robot and the model: the file is looked at before the model is loaded
        # and before the robot is connected (here the model's class fails when it is made, and RobotLink refuses too)
        def model_not_to_be_loaded(*arguments, **named_arguments):

            raise AssertionError("the model was loaded before the reference picture was looked at")

        real_policy = model_mode.ModelPolicy
        model_mode.ModelPolicy = model_not_to_be_loaded
        try:
            with stand_in_person([]) as seen:
                result = robot_run("no_stand_ins", {"run_folder": folder, "ghost_path": Path(folder) / "missing.jpg"})
        finally:
            model_mode.ModelPolicy = real_policy
        assert result["ending"] == "not_started" and "missing.jpg" in result["reason"] and seen["prompts"] == [], result


def test_wrong_new_settings_are_reported_before_anything_is_asked_of_the_robot() -> None:

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "ghost.jpg").write_bytes(grey_jpeg(0))
        with_ghost = {"run_folder": folder, "ghost_path": Path(folder) / "ghost.jpg"}
        # a stall action that is neither "end" nor "log" would silently end runs; a strength or refresh that is no number
        # would fail at the go-ahead, after the motors were off and on and with the arm held low
        for wrong, words in ((dict(run_folder = folder, stall_action = "Log"), "\"stall_action\" is 'Log'"),
                             (dict(with_ghost, ghost_strength = None), "\"ghost_strength\" (None)"),
                             (dict(with_ghost, ghost_refresh_s = "often"), "\"ghost_refresh_s\" ('often')")):
            run = stand_in_robot_run(recorded_start_joints(), wrong, ["enter"])
            assert run["result"]["ending"] == "not_started" and words in run["result"]["reason"], run["result"]
            assert run["asked"] == [] and run["prompts"] == [] and run["shown"] == [], run
            assert "not started: the setting" in run["printed"], run["printed"]


def test_start_warning_is_left_out_when_the_placed_start_is_refused() -> None:

    recorded_tip = magnet_tip_and_lean(recorded_start_joints())[0]
    # 2 cm further out than the recorded start and 3 cm above pushing height: beyond the warning limit, and refused for
    # its height
    too_high = held_joints_at(joints_behind([recorded_tip[0] + 0.02, recorded_tip[1]]), 0.03)
    with tempfile.TemporaryDirectory() as folder:
        run = stand_in_robot_run(too_high, {"run_folder": folder}, ["enter"])
    assert run["result"]["ending"] == "not_started" and "+3.0 cm from pushing height" in run["result"]["reason"], run["result"]
    # the offset's own line is printed; its advice (Esc in the picture window) is not, since no window comes
    assert "start: tip 2.0 cm from the recorded start" in run["printed"] and "the start is further off" not in run["printed"], run["printed"]
    assert run["shown"] == [] and run["asked"] == ["motors off", "read state", "motors on", "close"], run["asked"]


def test_live_window_in_the_whole_flow_when_the_camera_or_the_window_fails() -> None:

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "ghost.jpg").write_bytes(grey_jpeg(0))
        settings = {"run_folder": folder, "step_budget": 2, "ghost_path": Path(folder) / "ghost.jpg"}
        # the live window's second camera request fails: the window shows the reference alone (black here), Enter does
        # not count under it, and with the next picture the wait goes on and Enter starts the run
        run = stand_in_robot_run(recorded_start_joints(), settings, [None, "enter", None, "enter"], link = FailingCameraLink({2}))
        assert run["result"]["ending"] == "step_budget_reached" and run["result"]["moves"] == 2, run["result"]
        assert len(run["shown"]) == 6 and int(run["shown"][1][35:].max()) == 0 and int(run["shown"][0][400, 5, 0]) > 100, len(run["shown"])
        assert "a camera picture could not be read (RuntimeError: Command GET_IMAGE_COMPRESSED KO)" in run["printed"], run["printed"]
        assert run["asked"] == ["motors off", "read state", "motors on", "move", "move"] + ["read state", "move"] * 2 + ["read state", "move", "close"], run["asked"]

        # a camera that never answers: nothing ends the wait by itself; after Esc the arm is lifted as after every Esc
        run = stand_in_robot_run(recorded_start_joints(), settings, [None, "enter", "escape"], link = FailingCameraLink(set(range(1, 100))))
        assert run["result"]["ending"] == "not_started" and len(run["shown"]) == 3, (run["result"], len(run["shown"]))
        assert run["asked"] == ["motors off", "read state", "motors on", "move", "move", "read state", "move", "close"], run["asked"]

        # Esc pressed in the terminal while the window is live ends the wait at once, as Esc in the window does (here
        # it is pressed when the window shows its second picture); the arm is lifted as after every Esc
        keyboard = TypedKeys("")
        shown = []

        def window_with_a_typist(picture) -> None:

            shown.append(picture)
            if len(shown) == 2:
                keyboard.characters.append("\x1b")

        real_keyboard, model_mode.msvcrt = model_mode.msvcrt, keyboard
        try:
            run = stand_in_robot_run(recorded_start_joints(), settings, [None, None, None, None, "enter"], show = window_with_a_typist)
        finally:
            model_mode.msvcrt = real_keyboard
        assert run["result"]["ending"] == "not_started" and len(shown) == 2 and not keyboard.characters, (run["result"], len(shown))
        assert run["asked"] == ["motors off", "read state", "motors on", "move", "move", "read state", "move", "close"], run["asked"]

        # the picture window itself fails: no key can be given there, so nothing is lifted without the person; the run
        # ends as not started and says that the arm is still low
        def broken_window(picture) -> None:

            raise RuntimeError("the window could not be made")

        run = stand_in_robot_run(recorded_start_joints(), settings, ["enter"], show = broken_window)
        assert run["result"]["ending"] == "not_started", run["result"]
        assert run["result"]["reason"] == "the live window stopped (RuntimeError: the window could not be made)", run["result"]
        assert "THE ARM IS STILL LOW" in run["printed"], run["printed"]
        assert run["asked"] == ["motors off", "read state", "motors on", "move", "move", "close"], run["asked"]


def test_rehearsal_runs_the_robot_part_on_the_simulated_robot() -> None:

    # the simulated hand: poses that come nearer to the recorded start in equal steps and end on it
    path = rehearsal_hand_path(MODEL_MODE_PARAMETERS)
    distances = [offset_from_recorded_start(joints)["tip_distance_cm"] for joints in path]
    heights = [100.0 * height_above_pushing_m(joints) for joints in path]
    assert len(path) == MODEL_MODE_PARAMETERS["rehearsal_hand_readings"] == 12, len(path)
    assert abs(distances[0] - 2.69) < 0.05 and distances[-1] < 0.01, distances
    assert all(nearer < further for further, nearer in zip(distances, distances[1:])), distances
    assert abs(heights[0] - 3.0) < 0.1 and abs(heights[-1]) < 0.05, heights

    real_parts = (model_mode.is_terminal_a_console, model_mode.enter_pressed)
    looks = []
    with tempfile.TemporaryDirectory() as folder:
        try:
            # the stand-in terminal: a console whose Enter arrives when the simulated hand has come to rest (one look at
            # the keyboard before the first reading and one after each reading)
            model_mode.is_terminal_a_console = lambda: True
            model_mode.enter_pressed = lambda: looks.append("look") or ("enter" if len(looks) >= 14 else None)
            with stand_in_person(["enter"]) as seen:
                result = rehearsal_run("rehearsal_check", {"run_folder": folder, "is_place_help_shown": True, "place_help_refresh_s": 0.0, "step_budget": 3},
                                       stand_in_policy = SteadyPolicy())
        finally:
            model_mode.is_terminal_a_console, model_mode.enter_pressed = real_parts
        text = seen["printed"].getvalue()
        assert "REHEARSAL: no robot is connected" in text and "a simulated hand brings the tip to the recorded start" in text, text
        # the placing line changes as the hand comes nearer: too high at first, close at the end
        live = [part.strip() for part in text.split("\r") if part.strip().startswith("tip ")]
        assert len(live) == 13, len(live)
        assert live[0] == "tip  2.7 cm off: 1.0 too close, 2.5 too far left, forearm   +8.0 deg, height  +3.0 cm: too high   (wait)", live[0]
        assert live[-1].startswith("tip  0.0 cm off: 0.0 too far out, 0.0 too far left, forearm   +0.0 deg, height  +0.0 cm: close   Enter = placed"), live[-1]
        # then the robot part's own flow: its first question, the go-ahead, the run, and the log in the folder asked for
        assert seen["prompts"] == PROMPTS_OF_11_11_26[:1] and len(seen["shown"]) == 4, (seen["prompts"], len(seen["shown"]))
        assert result["ending"] == "step_budget_reached" and result["moves"] == 3 and result["is_lifted"], result
        assert result["start_offset"]["tip_distance_cm"] == 0.0 and Path(result["folder"]).parent == Path(folder), result

        # without the placing aid the simulated arm stands at the recorded start and the two questions of 11.11.26 come
        with stand_in_person(["escape"]) as seen:
            result = rehearsal_run("rehearsal_plain", {"run_folder": folder}, stand_in_policy = SteadyPolicy())
        assert result["ending"] == "not_started" and seen["prompts"] == PROMPTS_OF_11_11_26, (result, seen["prompts"])
        assert "start: tip 0.0 cm from the recorded start" in seen["printed"].getvalue() and "a simulated hand" not in seen["printed"].getvalue()

    # left to itself a rehearsal logs outside 03_data (looked at with a rehearsal that a wrong setting ends before it begins)
    with stand_in_person([]) as seen:
        result = rehearsal_run("rehearsal_not_begun", {"stall_action": "neither"}, stand_in_policy = SteadyPolicy())
    assert result["ending"] == "not_started", result
    assert f"The run is logged in {MODEL_MODE_PARAMETERS['rehearsal_run_folder']}." in seen["printed"].getvalue(), seen["printed"].getvalue()
    assert "03_data" not in Path(MODEL_MODE_PARAMETERS["rehearsal_run_folder"]).parts


if __name__ == "__main__":
    for test in [test_stop_rule_needs_two_in_a_row, test_forward_and_left_follow_the_picture,
                 test_twist_is_clamped_and_wrist_and_hand_are_held, test_short_direction_keeps_the_last_one_and_far_steps_are_refused,
                 test_predicted_length_rule_moves_what_the_model_asks, test_a_hesitating_model_ends_the_run_as_stalled,
                 test_lift_keeps_the_tip_above_its_point, test_executor_gives_back_recorded_steps,
                 test_replay_gives_back_saved_predictions, test_simulated_run_takes_one_centimetre_steps,
                 test_team_of_two_averages_its_members, test_two_picture_model_gets_the_picture_before_and_forgets_it_between_runs,
                 test_step_cm_flag_is_read_and_refused_where_it_does_not_fit,
                 test_simulated_run_with_step_cm_logs_shorter_tip_moves, test_stall_rule_can_log_and_let_the_run_go_on,
                 test_start_offset_is_zero_at_the_recorded_start_and_its_signs_are_right, test_start_offset_lines_warn_only_beyond_the_limits,
                 test_placing_help_shows_each_reading_until_enter,
                 test_ctrl_c_during_a_question_to_the_robot_is_held_back_until_the_answer_is_complete, test_only_enter_ends_the_wait_in_the_terminal,
                 test_esc_ends_a_run_with_fresh_keys_also_when_more_keys_wait_in_the_terminal,
                 test_ctrl_c_during_a_move_ends_a_run_with_fresh_keys_after_that_move,
                 test_ghost_blend_lays_the_reference_over_the_live_picture, test_reference_picture_is_none_when_missing_or_not_a_picture,
                 test_go_ahead_with_ghost_shows_live_pictures_until_a_key, test_live_go_ahead_waits_for_the_person_when_the_camera_fails,
                 test_window_names_a_step_of_part_millimetres_in_full, test_robot_run_without_new_flags_asks_what_it_asked_before,
                 test_robot_run_with_the_placing_aid_and_the_ghost, test_keys_pressed_too_early_do_not_answer_the_first_prompt_with_fresh_keys,
                 test_missing_ghost_is_reported_before_anything_is_asked_of_the_robot,
                 test_wrong_new_settings_are_reported_before_anything_is_asked_of_the_robot,
                 test_start_warning_is_left_out_when_the_placed_start_is_refused,
                 test_live_window_in_the_whole_flow_when_the_camera_or_the_window_fails,
                 test_rehearsal_runs_the_robot_part_on_the_simulated_robot]:
        test()
        print(f"{test.__name__}: passed")
    print("all tests passed")
