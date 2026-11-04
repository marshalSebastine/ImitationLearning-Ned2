"""
Check the recording tool against the simulated robot, without a window or a person: keys come from a script.

Run from 02_code (about half a minute; nothing touches the robot or 03_data):
    python test_record_episodes.py
"""

import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from config import RECORDING_PARAMETERS, WORKSPACE_PARAMETERS, picture_step_to_table, start_pose, step_budget_for
from record_episodes import EpisodeRecorder
from robot_link import SimulatedRobotLink, simulated_hand_path


def scripted_keys(keys: List[Optional[str]]) -> Callable[[int], Optional[str]]:
    """
    A key reader that hands out ``keys`` in order and then Esc and Enter in turn for ever.

    Esc quits a step loop; Enter answers the prompts that accept nothing else (the release of a held arm, the puck
    reset), so the tool always reaches its end, as it does with a person at the keys.
    """

    remaining = list(keys)
    fallback = ["escape", "enter"]

    def read_key(wait_ms: int) -> Optional[str]:
        if remaining:
            return remaining.pop(0)
        fallback.append(fallback.pop(0))
        return fallback[-1]

    return read_key


def read_episode(raw_folder: Path) -> tuple:
    """The only episode recorded under ``raw_folder``: its summary and its step rows."""

    episode_folder = next(raw_folder.glob("*_session_*/*_episode_*"))
    summary = json.loads((episode_folder / "episode.json").read_text(encoding = "utf-8"))
    with open(episode_folder / "steps.csv", encoding = "utf-8") as file:
        rows = list(csv.DictReader(file))
    return episode_folder, summary, rows


def test_picture_steps_point_the_right_way() -> None:
    """With the rod straight ahead of the base, picture-forward is table +x and picture-left is table +y."""

    forward = picture_step_to_table("forward", 0.25, 0.0)
    left = picture_step_to_table("left", 0.25, 0.0)
    assert math.isclose(forward[0], 0.01) and abs(forward[1]) < 1e-12, forward
    assert abs(left[0]) < 1e-12 and math.isclose(left[1], 0.01), left


def test_keyboard_episode(raw_folder: Path) -> None:
    """
    22 forward steps and a stop push the puck along a line pointing straight away from the base (where picture-forward
    is the line's direction); every step logs a new picture.
    """

    radial_line = [[0.156, 0.0], [0.398, 0.0]]
    robot_link = SimulatedRobotLink({"line_xy": radial_line, "puck_xy": radial_line[0]})
    # "u" after the tenth step flags step 9 as a mistake; "l" before the fifteenth step flags step 14 as line lost.
    keys = ["enter"] + ["w"] * 10 + ["u"] + ["w"] * 4 + ["l"] + ["w"] * 8 + [" ", "escape"]
    recorder = EpisodeRecorder(robot_link, operator = "Test Person", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": radial_line[0]},
        key_reader = scripted_keys(keys))
    recorder.record_session("keyboard")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    assert episode_folder.parent.name.endswith("_test_person"), episode_folder.parent.name
    assert summary["camera_mount"] and summary["mapping_version"] and summary["step_budget"] == 36, summary
    assert len(rows) == 23 and rows[-1]["action"] == "stop", len(rows)
    assert [row["is_mistake"] for row in rows] == ["1" if row["step"] == "9" else "0" for row in rows], "mistake flag"
    assert [row["is_lost"] for row in rows] == ["1" if row["step"] == "14" else "0" for row in rows], "lost flag"
    assert summary["mistake_steps"] == 1 and summary["lost_steps"] == 1, summary
    assert all(float(row["capture_time"]) >= float(row["move_return_time"]) for row in rows), "capture before the move returned"
    assert all(float(row["decision_time"]) >= float(row["capture_time"]) for row in rows), "decision before the picture"
    assert all((episode_folder / row["frame"]).exists() for row in rows)
    assert all(first["frame_md5"] != second["frame_md5"] for first, second in zip(rows, rows[1:])), "a stale picture was logged"
    tip_steps = [(float(second["x"]) - float(first["x"]), float(second["y"]) - float(first["y"])) for first, second in zip(rows, rows[1:])]
    assert all(math.isclose(step_x, 0.01, abs_tol = 0.000_5) and abs(step_y) < 0.000_5 for step_x, step_y in tip_steps), tip_steps
    assert (episode_folder.parent / "session_notes.md").read_text(encoding = "utf-8").count("completed") == 1


def test_hand_guided_episode(raw_folder: Path) -> None:
    """
    A wobbly hand path is replayed in 1 cm steps that stay within half a step of it and end with a stop at its end.
    Every replay step needs an Enter; a missing key must not move the robot. (The simulated puck's physics is too
    rough to check where the puck ends up; that is for the robot.)
    """

    hand_path = simulated_hand_path()
    robot_link = SimulatedRobotLink({"hand_path_xy": hand_path})
    keys = ["enter", "enter"] + [None] * (len(hand_path) + 5) + [" ", "enter", "enter"] + ["enter"] * 40 + ["escape"]
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True}, key_reader = scripted_keys(keys))
    recorder.record_session("hand_guided")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    assert (episode_folder / "guided_path.csv").exists()
    assert rows[-1]["action"] == "stop" and all(row["source"] == "replay" for row in rows), [row["action"] for row in rows]

    path_end = hand_path[-1]
    last = rows[-1]
    distance_to_path_end = math.hypot(float(last["x"]) - path_end[0], float(last["y"]) - path_end[1])
    assert distance_to_path_end < 0.01, distance_to_path_end
    assert len(rows) >= int((path_end[0] - start_pose()[0]) / 0.01), len(rows)
    distances_to_path = [min(math.hypot(float(row["x"]) - point[0], float(row["y"]) - point[1]) for point in hand_path) for row in rows]
    assert max(distances_to_path) < 0.005, max(distances_to_path)


def test_workspace_guard_protects_the_puck(raw_folder: Path) -> None:
    """
    Near the sheet's far edge, forward keys are refused once the predicted puck centre would leave the sheet plus its
    margin, before the rod itself reaches the workspace box; the refused keys log nothing and the episode goes on.
    """

    radial_line = [[0.156, 0.0], [0.398, 0.0]]
    far_edge_x = max(corner[0] for corner in WORKSPACE_PARAMETERS["sheet_corners_xy"])
    start = [far_edge_x - 0.03, 0.0]
    robot_link = SimulatedRobotLink({"line_xy": radial_line, "puck_xy": start})
    keys = ["enter"] + ["w"] * 12 + [" ", "escape"]
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": start},
        key_reader = scripted_keys(keys))
    recorder.record_session("keyboard")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    moving_rows = [row for row in rows if row["action"] != "stop"]
    assert 0 < len(moving_rows) < 12, len(moving_rows)
    last_x = float(rows[-1]["x"])
    puck_centre_x = last_x + RECORDING_PARAMETERS["puck_centre_ahead_of_tip_m"]
    assert puck_centre_x <= far_edge_x + RECORDING_PARAMETERS["sheet_margin_m"] + 1e-6, puck_centre_x


def test_step_budget_per_sheet() -> None:
    """The budget is 1.5 times the traced line length, and the fallback holds for an unknown sheet."""

    assert step_budget_for("path_01_straight_line") == 36
    assert step_budget_for("path_04_meander") == 116
    assert step_budget_for("path_99_unknown") == RECORDING_PARAMETERS["step_budget"]


if __name__ == "__main__":
    test_picture_steps_point_the_right_way()
    test_step_budget_per_sheet()
    for test in (test_keyboard_episode, test_hand_guided_episode, test_workspace_guard_protects_the_puck):
        with tempfile.TemporaryDirectory() as folder:
            test(Path(folder))
        print(f"{test.__name__}: passed")
    print("all tests passed")
