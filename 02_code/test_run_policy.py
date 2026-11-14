"""
Tests of run_policy.py and the recording tool's policy mode, with tiny policies trained on made-up pictures and the
simulated robot; no robot, no recorded episode and no ImageNet download is used. About a minute.

Run from 02_code:
    python test_run_policy.py
"""

import csv
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy

from config import RECORDING_PARAMETERS
from record_episodes import EpisodeRecorder
from robot_link import SimulatedRobotLink
from run_policy import PolicyDriver
from test_policy_training import prepared_folder
from test_record_episodes import scripted_keys
from train_policy import train_one_run


def trained_checkpoint(
        folder: Path,
        model_name: str
) -> Path:
    """Train ``model_name`` for one short epoch on the made-up dataset and return its best.pt."""

    settings = prepared_folder(folder) if not (folder / "processed").exists() else None
    if settings is None:
        from test_policy_training import tiny_settings
        settings = tiny_settings(folder)
    train_one_run(model_name, 1, "policy_test", training_overrides = settings, device_name = "cpu")
    return folder / "runs" / f"policy_test_{model_name}_seed1" / "checkpoints" / "best.pt"


def camera_jpeg(shade: int = 0) -> bytes:
    """A camera-sized JPEG like the made-up training pictures."""

    picture = numpy.full((480, 640, 3), (130 + shade, 150, 175), dtype = numpy.uint8)
    cv2.circle(picture, (310, 300), 30, (40, 40, 200), -1)
    return cv2.imencode(".jpg", picture)[1].tobytes()


def test_commands_follow_the_gamepad_conventions() -> None:
    """Forward is stick up, left is stick left (negative), a turn_left of 6 degrees is a right-stick push of -0.4."""

    with tempfile.TemporaryDirectory() as folder:
        driver = PolicyDriver(trained_checkpoint(Path(folder), "factorised_classifier"), {"device": "cpu"})
        forward = driver.pad_command(0.01, 0.0, 0.0)
        assert math.isclose(forward["pad_move_y"], 1.0) and math.isclose(forward["pad_move_x"], 0.0, abs_tol = 1e-9), forward
        left = driver.pad_command(0.0, 0.01, 6.0)
        assert math.isclose(left["pad_move_x"], -1.0) and math.isclose(left["pad_twist"], -6.0 / RECORDING_PARAMETERS["pad_twist_deg"]), left
        small = driver.pad_command(0.002, 0.0, 0.0)
        assert math.isclose(small["pad_move_y"], 0.3), small
        decision = driver.decide(camera_jpeg(), 0)
        assert decision["action"] in ("pad", "stop") and decision["move"] in driver.training_parameters["move_classes"], decision


def test_history_policy_remembers_only_steps_that_happened() -> None:
    """Asking twice for the same step adds nothing to the history; going on to the next step adds the last picture."""

    with tempfile.TemporaryDirectory() as folder:
        driver = PolicyDriver(trained_checkpoint(Path(folder), "history_classifier"), {"device": "cpu"})
        driver.decide(camera_jpeg(0), 0)
        driver.decide(camera_jpeg(5), 0)
        assert driver.past_pictures == [], len(driver.past_pictures)
        first = driver.decide(camera_jpeg(10), 1)
        assert len(driver.past_pictures) == 1 and driver.previous_move != len(driver.training_parameters["move_classes"])
        for step in range(2, 6):
            driver.decide(camera_jpeg(step), step)
        assert len(driver.past_pictures) == driver.history_length - 1, len(driver.past_pictures)
        driver.start_episode()
        assert driver.past_pictures == [] and driver.pending is None and first["action"] in ("pad", "stop")


def test_policy_mode_drives_the_simulated_robot() -> None:
    """Enter carries out five proposals of a regressor policy, Space stops by hand; the rows say who chose each step."""

    with tempfile.TemporaryDirectory() as folder:
        checkpoint = trained_checkpoint(Path(folder), "exact_step_regressor")
        driver = PolicyDriver(checkpoint, {"device": "cpu"})
        driver.decide = lambda picture_bytes, step, original = driver.decide: {
            **original(picture_bytes, step), "action": "pad", "pad_command": driver.pad_command(0.01, 0.0, 0.0), "move": "forward"}
        raw_folder = Path(folder) / "raw_policy"
        robot_link = SimulatedRobotLink({"line_xy": [[0.156, 0.0], [0.398, 0.0]], "puck_xy": [0.25, 0.0]})
        keys = ["enter"] + ["enter"] * 5 + [" "] + [None] * 10 + ["y", "escape"]
        recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
            "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": [0.25, 0.0]},
            key_reader = scripted_keys(keys), policy_driver = driver)
        recorder.record_session("policy")

        episode_folder = next(raw_folder.glob("*_session_*/*_episode_*"))
        summary = json.loads((episode_folder / "episode.json").read_text(encoding = "utf-8"))
        with open(episode_folder / "steps.csv", encoding = "utf-8", newline = "") as file:
            rows = list(csv.DictReader(file))
        assert summary["status"] == "completed" and summary["mode"] == "policy", summary
        assert summary["policy"]["model_name"] == "exact_step_regressor" and summary["outcome"] == "puck_in_goal", summary
        assert [row["source"] for row in rows] == ["policy"] * 5 + ["human"], [row["source"] for row in rows]
        assert all(row["action"] == "forward" for row in rows[:5]) and rows[-1]["action"] == "stop", [row["action"] for row in rows]
        positions = [float(row["x"]) for row in rows]
        assert math.isclose(positions[5] - positions[0], 0.05, abs_tol = 0.004), positions


if __name__ == "__main__":
    for test in (test_commands_follow_the_gamepad_conventions, test_history_policy_remembers_only_steps_that_happened,
                 test_policy_mode_drives_the_simulated_robot):
        test()
        print(f"{test.__name__}: passed", flush = True)
    print("all tests passed")
