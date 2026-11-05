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

import numpy
from typing import Callable, List, Optional

from config import (RECORDING_PARAMETERS, WORKSPACE_PARAMETERS, action_step_to_table, camera_heading_rad, pushing_height,
                    start_pose, step_budget_for)
from record_episodes import EpisodeRecorder, camera_shift_px, forearm_camera_heading_rad, picture_heading_rad
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
    """
    Camera on the hand: with the hand untouched, the picture's top is table +x wherever the rod stands, and turned
    90 degrees it is +y. Camera on the forearm: the picture's top points from the base towards the rod. Straight
    ahead of the base both agree, and "forward" is +x and "left" is +y there.
    """

    for x, y in ((0.25, 0.0), (0.22, -0.11)):
        pose = [x, y, 0.03, 0.0, 1.57, 0.0]
        assert abs(math.degrees(camera_heading_rad(pose, "hand"))) < 0.1, camera_heading_rad(pose, "hand")
        assert math.isclose(math.degrees(camera_heading_rad(pose, "forearm")), math.degrees(math.atan2(y, x)), abs_tol = 1e-9)
    turned = [0.25, 0.0, 0.03, 0.0, 1.57, math.pi / 2]
    assert math.isclose(math.degrees(camera_heading_rad(turned, "hand")), 90.0, abs_tol = 0.1), camera_heading_rad(turned, "hand")
    # Forearm camera: untwisted, the picture's top points from the base towards the wrist (joint 1's direction); the
    # forearm twisted 90 degrees turns the picture by about 90 degrees, since the forearm stands nearly upright.
    pushing_joints = [0.3, -0.809, -0.588, 0.0, -0.173, 0.3]
    assert math.isclose(math.degrees(forearm_camera_heading_rad(pushing_joints, True)), math.degrees(0.3), abs_tol = 0.5)
    # Close to the base the forearm leans back towards the robot; the picture must not flip (it did on 4.11.26).
    near_base_joints = [0.24126, -0.73527, -1.01429, 0.00163, 0.16404, 0.23786]
    assert math.isclose(math.degrees(forearm_camera_heading_rad(near_base_joints, True)), math.degrees(0.24126), abs_tol = 2.0), \
        math.degrees(forearm_camera_heading_rad(near_base_joints, True))
    twisted = list(pushing_joints)
    twisted[3] = math.pi / 2
    turned_by = math.degrees(forearm_camera_heading_rad(twisted, True)) - math.degrees(0.3)
    assert 70.0 < abs(turned_by) < 110.0, turned_by
    assert math.isclose(math.degrees(forearm_camera_heading_rad(twisted, False)), math.degrees(0.3), abs_tol = 0.5)

    ahead = [0.25, 0.0, 0.03, 0.0, 1.57, 0.0]
    forward = action_step_to_table("forward", ahead, 0.0)
    left = action_step_to_table("left", ahead, 0.0)
    assert math.isclose(forward[0], 0.01) and abs(forward[1]) < 1e-9, forward
    assert abs(left[0]) < 1e-9 and math.isclose(left[1], 0.01), left


def test_camera_side_view_turn(raw_folder: Path) -> None:
    """
    Three "v" presses turn the forearm 30, 60 and 90 degrees, the first landing the tip back from the puck (away from
    it), the others not again; a fourth is refused (-90 to 90 allowed); while turned, forward goes where the turned
    picture's top points and left 90 degrees from it; three "b" turn back and forward is +x again; after five steps a
    further "b" turns to -30 and lands near the tip as it is then; every row logs the forearm angle.
    """

    radial_line = [[0.156, 0.0], [0.398, 0.0]]
    start_point = [0.25, 0.0]
    robot_link = SimulatedRobotLink({"line_xy": radial_line, "puck_xy": start_point})
    keys = ["enter", "v", "v", "v", "v", "w", "w", "a", "b", "b", "b"] + ["w"] * 5 + ["b", " ", "y", "escape"]
    turn_keys = dict(RECORDING_PARAMETERS["keys"], v = "turn_left", b = "turn_right")  # unbound by default since the camera left the hand
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": start_point,
        "keys": turn_keys}, key_reader = scripted_keys(keys))
    recorder.record_session("keyboard")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    actions = [row["action"] for row in rows]
    assert actions == ["turn_left"] * 3 + ["forward", "forward", "left"] + ["turn_right"] * 3 + ["forward"] * 5 + ["turn_right", "stop"], actions
    turns = [row["forearm_turn_deg"] for row in rows]
    assert turns == ["0.0", "30.0", "60.0", "90.0", "90.0", "90.0", "90.0", "60.0", "30.0"] + ["0.0"] * 6 + ["-30.0"], turns
    positions = [(float(row["x"]), float(row["y"])) for row in rows]
    retreat = RECORDING_PARAMETERS["turn_landing_retreat_m"]
    assert math.isclose(positions[1][0] - positions[0][0], -retreat, abs_tol = 0.008) and abs(positions[1][1] - positions[0][1]) < 0.008, \
        "the first turn should land the tip back from the puck, away from it"
    for index in (1, 2):
        assert math.hypot(positions[index + 1][0] - positions[index][0], positions[index + 1][1] - positions[index][1]) < 0.008, \
            "a turn straight after another should not retreat again"
    assert float(rows[15]["joint_4"]) < -0.4, rows[15]["joint_4"]
    # After five ordinary steps, the next turn must land near the tip as it is now, not where an earlier turn left it
    # (test 1 of 4.11.26 landed 14.9 cm away).
    landing_jump = math.hypot(positions[15][0] - positions[14][0], positions[15][1] - positions[14][1])
    assert landing_jump < 0.035, landing_jump

    def heading_deg(row) -> float:
        state = {"pose": [float(row[name]) for name in ("x", "y", "z", "roll", "pitch", "yaw")],
                 "joints": [float(row[f"joint_{index}"]) for index in range(1, 7)]}
        return math.degrees(picture_heading_rad(state, RECORDING_PARAMETERS))

    def direction_deg(index: int) -> float:
        return math.degrees(math.atan2(positions[index + 1][1] - positions[index][1], positions[index + 1][0] - positions[index][0]))

    for index in (3, 4):
        assert abs((direction_deg(index) - heading_deg(rows[index]) + 180) % 360 - 180) < 8.0, "turned, forward should go where the picture's top points"
        assert math.isclose(math.hypot(positions[index + 1][0] - positions[index][0], positions[index + 1][1] - positions[index][1]), 0.01, abs_tol = 0.002)
    assert abs((direction_deg(5) - heading_deg(rows[5]) - 90 + 180) % 360 - 180) < 8.0, "turned, left should go 90 degrees left of the picture's top"
    assert math.isclose(positions[10][0] - positions[9][0], 0.01, abs_tol = 0.002) and abs(positions[10][1] - positions[9][1]) < 0.002, "turned back, forward should go to +x"
    assert all(float(row["joint_4"]) > 1.0 for row in rows[3:6]), [row["joint_4"] for row in rows]


def test_replay_copies_forearm_turn(raw_folder: Path) -> None:
    """A forearm twisted by 30 degrees halfway along the guided path is proposed as one turn_left in the replay."""

    hand_path = simulated_hand_path()
    forearm = [0.0 if index < len(hand_path) // 2 else math.radians(30.0) for index in range(len(hand_path))]
    robot_link = SimulatedRobotLink({"hand_path_xy": hand_path, "hand_path_forearm_rad": forearm})
    keys = ["enter", "enter"] + [None] * (len(hand_path) + 5) + [" ", "enter", "enter"] + ["enter"] * 60 + ["escape"]
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "hand_turn_deg": 0},
        key_reader = scripted_keys(keys))
    recorder.record_session("hand_guided")

    episode_folder, summary, rows = read_episode(raw_folder)
    actions = [row["action"] for row in rows]
    assert actions.count("turn_left") == 1 and all(row["source"] == "replay" for row in rows), actions
    turn_index = actions.index("turn_left")
    assert 0 < turn_index < len(actions) - 1 and rows[turn_index + 1]["forearm_turn_deg"] == "30.0", (turn_index, rows[turn_index + 1]["forearm_turn_deg"])
    with open(episode_folder / "guided_path.csv", encoding = "utf-8") as file:
        assert "joint_4" in file.readline()


def test_exact_replay(raw_folder: Path) -> None:
    """
    A guided path of 7 cm straight out with a 30 degree forearm twist halfway is replayed waypoint by waypoint ("g"
    runs it through): the robot ends at the guided joints, the twist is labelled as one turn, the rest as forward
    moves, and the exact movement is stored beside each label.
    """

    from side_view_test import solve_least_lean
    from config import PUSHER_PARAMETERS

    joints_path = []
    for index in range(36):
        x = 0.20 + 0.002 * index
        target = numpy.array([x, 0.0, pushing_height(x, 0.0) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
        joints = solve_least_lean(target, math.radians(0.0 if index < 18 else 30.0), wrist_range_deg = (-30.0, 30.0))
        if index < 18:
            joints[5] = joints[0]
        joints_path.append(joints)
    hand_path = [[0.20 + 0.002 * index, 0.0] for index in range(36)]
    robot_link = SimulatedRobotLink({"hand_path_xy": hand_path, "hand_path_joints": joints_path, "puck_xy": [0.2185, 0.0]})
    keys = ["enter", "enter"] + [None] * 40 + [" ", "enter", "enter", "g"] + [None] * 30 + ["enter", "y", "escape"]
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": [0.2185, 0.0]},
        key_reader = scripted_keys(keys))
    recorder.record_session("hand_guided_exact")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    actions = [row["action"] for row in rows]
    assert actions[-1] == "stop" and actions.count("turn_left") == 1, actions
    assert all(row["source"] == "replay_exact" for row in rows[:-1]), [row["source"] for row in rows]
    assert sum(action == "forward" for action in actions) >= 4, actions
    with open(episode_folder / "guided_path.csv", encoding = "utf-8") as file:
        last_guided = list(csv.DictReader(file))[-1]
    final_joints = [float(rows[-1][f"joint_{index}"]) for index in range(1, 7)]
    guided_joints = [float(last_guided[f"joint_{index}"]) for index in range(1, 7)]
    assert max(abs(final - guided) for final, guided in zip(final_joints, guided_joints)) < 1e-4, (final_joints, guided_joints)
    assert all(row["tip_forward_m"] != "" for row in rows[1:-1]), "the exact movement is stored"


def test_exact_replay_with_hand_correction(raw_folder: Path) -> None:
    """
    Two replayed waypoints, then "h": a second guided path is recorded, the arm returns to where "h" was pressed, and
    the correction is replayed to its end; the episode counts one correction and keeps its reference picture and path.
    """

    from side_view_test import solve_least_lean
    from config import PUSHER_PARAMETERS

    joints_path = []
    for index in range(20):
        x = 0.20 + 0.002 * index
        target = numpy.array([x, 0.0, pushing_height(x, 0.0) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
        joints = solve_least_lean(target, 0.0, wrist_range_deg = (-30.0, 30.0))
        joints[5] = joints[0]
        joints_path.append(joints)
    hand_path = [[0.20 + 0.002 * index, 0.0] for index in range(20)]
    robot_link = SimulatedRobotLink({"hand_path_xy": hand_path, "hand_path_joints": joints_path, "puck_xy": [0.2185, 0.0]})
    guiding = ["enter"] + [None] * 24 + [" ", "enter"]
    keys = (["enter"] + guiding + ["enter", "enter", "enter", "h"] + guiding + ["enter", "g"] + [None] * 20
            + ["enter", "y", "escape"])
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": [0.2185, 0.0]},
        key_reader = scripted_keys(keys))
    recorder.record_session("hand_guided_exact")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed" and summary["hand_corrections"] == 1, summary
    assert not (episode_folder / "steps_in_progress.csv").exists(), "the in-progress log is removed when the episode is finished"
    assert (episode_folder / "correction_01_reference.jpg").exists() and (episode_folder / "correction_01_guided_path.csv").exists()
    assert rows[-1]["action"] == "stop" and all(row["source"] == "replay_exact" for row in rows[:-1]), [row["source"] for row in rows]
    assert len(rows) >= 4, len(rows)


def scripted_pad(states: List[dict]) -> Callable[[], dict]:
    """A pad reader that hands out ``states`` in order (sticks at rest, no buttons for missing names), then rest."""

    remaining = list(states)
    rest = {"move_x": 0.0, "move_y": 0.0, "twist": 0.0, "lean": 0.0, "buttons": set()}
    return lambda: dict(rest, **remaining.pop(0)) if remaining else dict(rest)


def test_gamepad_episode(raw_folder: Path) -> None:
    """
    Left stick up pushes along the picture's top (+x at the start), right stick twists the forearm by 15 degrees per
    full push, L1 turns the hand, square flags a mistake and cross stops; every step stores the stick values.
    """

    radial_line = [[0.156, 0.0], [0.398, 0.0]]
    robot_link = SimulatedRobotLink({"line_xy": radial_line, "puck_xy": [0.25, 0.0]})
    pad = scripted_pad([{}, {"move_y": 1.0}, {"move_y": 1.0}, {"move_y": 0.5, "move_x": 0.5}, {"twist": -1.0},
                        {"buttons": {"hand_left"}}, {"buttons": set()}, {"buttons": {"mistake"}}, {"buttons": set()},
                        {"move_y": 1.0}, {"buttons": {"stop"}}])
    keys = ["enter"] + [None] * 40 + ["y", "escape"]
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": [0.25, 0.0]},
        key_reader = scripted_keys(keys), pad_reader = pad)
    recorder.record_session("gamepad")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    sources = [row["source"] for row in rows]
    assert sources[:-1] == ["gamepad"] * (len(rows) - 1) and rows[-1]["action"] == "stop", sources
    assert rows[0]["action"] == "forward" and rows[0]["pad_move_y"] == "1.0", rows[0]
    positions = [(float(row["x"]), float(row["y"])) for row in rows]
    assert math.isclose(positions[1][0] - positions[0][0], 0.01, abs_tol = 0.002), positions[:2]
    assert rows[3]["action"] == "turn_left" and math.isclose(float(rows[4]["forearm_turn_deg"]), 15.0, abs_tol = 0.5), (rows[3]["action"], rows[4]["forearm_turn_deg"])
    # A row holds the arm's state before its own move, so the hand turn of row 4 shows in row 5's joint 6.
    assert float(rows[4]["pad_hand"]) == 1.0 and float(rows[5]["joint_6"]) < float(rows[4]["joint_6"]) - 0.2, (rows[4]["joint_6"], rows[5]["joint_6"])
    assert [row["is_mistake"] for row in rows].count("1") == 1, [row["is_mistake"] for row in rows]


def test_camera_shift_is_measured() -> None:
    """A picture moved 6 px right and 3 px down in its bottom strip is measured as such; an unmoved one as zero."""

    import cv2

    generator = numpy.random.default_rng(7)
    picture = cv2.GaussianBlur((generator.random((480, 640)) * 255).astype(numpy.uint8), (7, 7), 0)
    picture = cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR)
    moved = cv2.warpAffine(picture, numpy.float32([[1, 0, 6], [0, 1, 3]]), (640, 480), borderMode = cv2.BORDER_REPLICATE)
    region = RECORDING_PARAMETERS["camera_check_region"]
    as_jpeg = lambda image: cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    shift_x, shift_y = camera_shift_px(as_jpeg(picture), as_jpeg(moved), region)
    assert abs(shift_x - 6.0) < 0.5 and abs(shift_y - 3.0) < 0.5, (shift_x, shift_y)
    still_x, still_y = camera_shift_px(as_jpeg(picture), as_jpeg(picture), region)
    assert abs(still_x) < 0.1 and abs(still_y) < 0.1, (still_x, still_y)


def test_turn_hint(raw_folder: Path) -> None:
    """At the far end of sheet 04 with the forearm at 90 degrees the hint names a smaller turn that keeps the lean
    within the limit (calc: 40 cm needs 33 degrees at 90, 18 at 30)."""

    robot_link = SimulatedRobotLink()
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True}, key_reader = scripted_keys([]))
    recorder.forearm_turn_deg = 90.0
    hint = recorder._EpisodeRecorder__turn_hint([0.40, 0.0])
    assert "+30 degrees" in hint or "+60 degrees" in hint, hint
    assert " b " in hint, hint

    assert RECORDING_PARAMETERS["keys"]["y"] == "back_left" and RECORDING_PARAMETERS["keys"]["c"] == "back_right"
    untouched = [0.25, 0.0, 0.03, 0.0, 1.57, 0.0]
    back_left = action_step_to_table("back_left", untouched, 0.0)
    back_right = action_step_to_table("back_right", untouched, 0.0)
    assert back_left[0] < 0 and back_left[1] > 0, back_left
    assert back_right[0] < 0 and back_right[1] < 0, back_right
    assert math.isclose(math.hypot(*back_left), 0.01) and math.isclose(math.hypot(*back_right), 0.01)


def test_keyboard_episode(raw_folder: Path) -> None:
    """
    22 forward steps and a stop push the puck along a line pointing straight away from the base (where picture-forward
    is the line's direction); every step logs a new picture.
    """

    radial_line = [[0.156, 0.0], [0.398, 0.0]]
    robot_link = SimulatedRobotLink({"line_xy": radial_line, "puck_xy": radial_line[0]})
    # "u" after the tenth step flags step 9 as a mistake; "l" before the fifteenth step flags step 14 as line lost.
    keys = ["enter"] + ["w"] * 10 + ["u"] + ["w"] * 4 + ["l"] + ["w"] * 8 + [" ", "y", "escape"]
    recorder = EpisodeRecorder(robot_link, operator = "Test Person", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": radial_line[0],
        "hand_turn_deg": 0}, key_reader = scripted_keys(keys))
    recorder.record_session("keyboard")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    assert episode_folder.parent.name.endswith("_test_person"), episode_folder.parent.name
    assert summary["camera_mount"] and summary["mapping_version"] and summary["step_budget"] == 48, summary
    assert summary["stepping_rule_version"] == RECORDING_PARAMETERS["stepping_rule_version"], summary
    assert summary["outcome"] == "puck_in_goal" and summary["discard_reason"] is None, summary
    assert len(rows) == 23 and rows[-1]["action"] == "stop", len(rows)
    assert [row["is_mistake"] for row in rows] == ["1" if row["step"] == "9" else "0" for row in rows], "mistake flag"
    assert [row["is_lost"] for row in rows] == ["1" if row["step"] == "14" else "0" for row in rows], "lost flag"
    assert summary["mistake_steps"] == 1 and summary["lost_steps"] == 1, summary
    assert all(row["picture_heading_deg"] != "" for row in rows), "every row stores the picture's direction"
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
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "hand_turn_deg": 0},
        key_reader = scripted_keys(keys))
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
    # 5 mm until 4.11.26; 6 mm since the diagonals went from 45 to 60 degrees, which matches gentle curves more coarsely
    # (the replay strayed up to 5.5 mm in this test).
    assert max(distances_to_path) < 0.006, max(distances_to_path)


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
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": start,
        "hand_turn_deg": 0},
        key_reader = scripted_keys(keys))
    recorder.record_session("keyboard")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    moving_rows = [row for row in rows if row["action"] != "stop"]
    assert 0 < len(moving_rows) < 12, len(moving_rows)
    last_x = float(rows[-1]["x"])
    puck_centre_x = last_x + RECORDING_PARAMETERS["puck_centre_ahead_of_tip_m"]
    assert puck_centre_x <= far_edge_x + RECORDING_PARAMETERS["sheet_margin_m"] + 1e-6, puck_centre_x


def test_step_over_reverses_without_touching_the_puck(raw_folder: Path) -> None:
    """
    Three pushes, a step over the puck that leaves it where it is, three pulls towards the base, a step back over it
    and one push: the two step-overs each move the rod the full length and end at pushing height.
    """

    radial_line = [[0.156, 0.0], [0.398, 0.0]]
    start = [0.25, 0.0]
    robot_link = SimulatedRobotLink({"line_xy": radial_line, "puck_xy": start})
    keys = ["enter"] + ["w"] * 3 + ["r"] + ["s"] * 3 + ["f"] + ["w"] + [" ", "escape"]
    step_over_keys = dict(RECORDING_PARAMETERS["keys"], r = "step_over_forward", f = "step_over_back")  # unbound by default since 4.11
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": start, "keys": step_over_keys},
        key_reader = scripted_keys(keys))
    recorder.record_session("keyboard")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "completed", summary
    assert [row["action"] for row in rows] == (["forward"] * 3 + ["step_over_forward"] + ["back"] * 3
                                               + ["step_over_back", "forward", "stop"]), rows

    length = RECORDING_PARAMETERS["step_over_m"]
    forward_over = float(rows[4]["x"]) - float(rows[3]["x"])
    back_over = float(rows[8]["x"]) - float(rows[7]["x"])
    assert math.isclose(forward_over, length, abs_tol = 1e-4), forward_over
    assert math.isclose(back_over, -length, abs_tol = 1e-4), back_over
    for row_after_step_over in (rows[4], rows[8]):
        x, y = float(row_after_step_over["x"]), float(row_after_step_over["y"])
        assert math.isclose(float(row_after_step_over["z"]), pushing_height(x, y), abs_tol = 1e-4), row_after_step_over

    # Pushed 3 cm, pulled 3 cm less the 5 mm landing gap, pushed 1 cm less the same gap: 1 cm from the start in all.
    assert math.isclose(float(robot_link.puck_xy[0]), start[0] + 0.01, abs_tol = 0.005), robot_link.puck_xy


def test_discard_stores_its_reason(raw_folder: Path) -> None:
    """A discarded episode asks for its reason and stores no outcome; keys other than the listed ones are ignored."""

    robot_link = SimulatedRobotLink()
    keys = ["enter", "w", "x", "w", "t", "escape"]
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True},
        key_reader = scripted_keys(keys))
    recorder.record_session("keyboard")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert summary["status"] == "discarded", summary
    assert summary["discard_reason"] == "puck_tipped" and summary["outcome"] is None, summary


def test_hand_flick(raw_folder: Path) -> None:
    """With the flick keys bound, q and e flick the hand and leave the rod where it was; both are logged as their own actions."""

    radial_line = [[0.156, 0.0], [0.398, 0.0]]
    robot_link = SimulatedRobotLink({"line_xy": radial_line, "puck_xy": radial_line[0]})
    keys = ["enter", "q", "e", "w", " ", "y", "escape"]
    flick_keys = dict(RECORDING_PARAMETERS["keys"], q = "flick_left", e = "flick_right")  # unbound by default since test 2
    recorder = EpisodeRecorder(robot_link, operator = "test", recorder_parameters = {
        "raw_folder": str(raw_folder), "is_window_shown": False, "is_simulated": True, "start_point_xy": radial_line[0],
        "keys": flick_keys}, key_reader = scripted_keys(keys))
    recorder.record_session("keyboard")

    episode_folder, summary, rows = read_episode(raw_folder)
    assert [row["action"] for row in rows] == ["flick_left", "flick_right", "forward", "stop"], [row["action"] for row in rows]
    assert robot_link.hand_flicks == 2, robot_link.hand_flicks
    positions = [(float(row["x"]), float(row["y"])) for row in rows]
    assert all(math.hypot(position[0] - positions[0][0], position[1] - positions[0][1]) < 1e-9 for position in positions[:3]), positions


def test_step_budget_per_sheet() -> None:
    """The budget is 1.5 times the traced line length, and the fallback holds for an unknown sheet."""

    assert step_budget_for("path_01_straight_line") == 48
    assert step_budget_for("path_04_meander") == 154
    assert step_budget_for("path_99_unknown") == RECORDING_PARAMETERS["step_budget"]


if __name__ == "__main__":
    test_picture_steps_point_the_right_way()
    test_step_budget_per_sheet()
    test_camera_shift_is_measured()
    print("test_camera_shift_is_measured: passed")
    for test in (test_keyboard_episode, test_hand_guided_episode, test_workspace_guard_protects_the_puck, test_step_over_reverses_without_touching_the_puck,
                 test_discard_stores_its_reason, test_camera_side_view_turn, test_hand_flick, test_turn_hint,
                 test_replay_copies_forearm_turn, test_exact_replay, test_exact_replay_with_hand_correction, test_gamepad_episode):
        with tempfile.TemporaryDirectory() as folder:
            test(Path(folder))
        print(f"{test.__name__}: passed")
    print("all tests passed")
