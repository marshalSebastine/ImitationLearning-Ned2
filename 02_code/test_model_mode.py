"""
Tests of model_mode.py; no robot is needed (the simulated robot, the arm model and the recorded data stand in).

Run from 02_code:
    python test_model_mode.py
"""

import math
import tempfile

import numpy

from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height
from model_mode import (StopRule, executor_check, held_joints_at, model_step, picture_axes, replay_check, simulated_run)
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed


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


if __name__ == "__main__":
    for test in [test_stop_rule_needs_two_in_a_row, test_forward_and_left_follow_the_picture,
                 test_twist_is_clamped_and_wrist_and_hand_are_held, test_short_direction_keeps_the_last_one_and_far_steps_are_refused,
                 test_predicted_length_rule_moves_what_the_model_asks, test_a_hesitating_model_ends_the_run_as_stalled,
                 test_lift_keeps_the_tip_above_its_point, test_executor_gives_back_recorded_steps,
                 test_replay_gives_back_saved_predictions, test_simulated_run_takes_one_centimetre_steps,
                 test_team_of_two_averages_its_members]:
        test()
        print(f"{test.__name__}: passed")
    print("all tests passed")
