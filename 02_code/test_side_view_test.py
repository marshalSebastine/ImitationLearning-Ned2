"""
Tests of side_view_test.py with a pretend robot built on the arm model; no robot is needed.

Run from 02_code:
    python test_side_view_test.py
"""

import tempfile
from typing import List, Optional

import numpy

from config import pushing_height, PUSHER_PARAMETERS, SIDE_VIEW_TEST_PARAMETERS
from side_view_test import SideViewTest, magnet_tip_and_lean, ned2_fk


class PretendRobotLink:

    def __init__(self, kinematics_error_m: float = 0.0):

        self.kinematics_error_m = kinematics_error_m
        self.joints = list(SIDE_VIEW_TEST_PARAMETERS["home_joints"])
        self.moves = []
        self.picture_count = 0
        self.is_closed = False


    def move_joints(self, joints: List[float]) -> None:

        self.joints = list(joints)
        self.moves.append(list(joints))


    def predicted_position(self, joints: List[float]) -> List[float]:

        position = list(ned2_fk.fk(list(joints))["tool"][0])
        position[0] += self.kinematics_error_m
        return position


    def read_state(self) -> dict:

        return {"pose": [0.0] * 6, "joints": list(self.joints)}


    def read_picture_bytes(self) -> bytes:

        self.picture_count += 1
        return f"picture {self.picture_count}".encode()


    def close(self) -> None:

        self.is_closed = True



def scripted_keys(keys: List[Optional[str]]):

    remaining = list(keys)
    return lambda wait_ms: remaining.pop(0)


def fast_parameters(folder: str) -> dict:

    return {"is_window_shown": False, "settle_s": 0.0, "output_folder": folder}


def test_view_part_holds_the_tip_and_goes_home() -> None:
    """Three distances: the tip comes down to pushing height, lean stays small, the arm ends at home and disconnected."""

    with tempfile.TemporaryDirectory() as folder:
        robot_link = PretendRobotLink()
        test = SideViewTest(robot_link, "test", fast_parameters(folder), scripted_keys(["enter"] * 6))
        rows = test.run("view")

    assert len(rows) == 3, rows
    for row, expected_lean in zip(rows, (1.5, 9.5, 14.1)):
        wanted_tip_z = pushing_height(row["distance_m"], SIDE_VIEW_TEST_PARAMETERS["line_y_m"]) - PUSHER_PARAMETERS["extension_below_rod_end_m"]
        assert abs(row["tip_z"] - wanted_tip_z) < 1e-3, row
        assert abs(row["lean_deg"] - expected_lean) < 1.0, row
        assert abs(row["turn_deg"] - 90.0) < 1e-6, row
    assert robot_link.moves[-1] == SIDE_VIEW_TEST_PARAMETERS["home_joints"] and robot_link.is_closed


def test_pull_part_steps_towards_the_base() -> None:
    """Each key pulls the tip 1 cm towards the base at constant height; x stops and the tip lifts before anything else."""

    with tempfile.TemporaryDirectory() as folder:
        robot_link = PretendRobotLink()
        keys = ["enter", "enter", "enter", "enter", "enter", "x"]
        test = SideViewTest(robot_link, "test", fast_parameters(folder), scripted_keys(keys))
        rows = test.run("pull", 0.25)

    assert [row["picture"] for row in rows] == ["pull_000.jpg", "pull_001.jpg", "pull_002.jpg", "pull_003.jpg"], rows
    x_values = numpy.array([row["tip_x"] for row in rows])
    assert numpy.allclose(numpy.diff(x_values), -0.01, atol = 1e-3), x_values
    assert numpy.allclose([row["tip_z"] for row in rows], rows[0]["tip_z"], atol = 1.5e-3), rows

    first_move_after_stop = robot_link.moves[len(robot_link.moves) - 5]
    assert magnet_tip_and_lean(first_move_after_stop)[0][2] > rows[-1]["tip_z"] + 0.04, "did not lift first"
    assert robot_link.is_closed


def test_nothing_moves_when_the_model_and_the_robot_disagree() -> None:

    with tempfile.TemporaryDirectory() as folder:
        robot_link = PretendRobotLink(kinematics_error_m = 0.01)
        test = SideViewTest(robot_link, "test", fast_parameters(folder), scripted_keys([]))
        try:
            test.run("view")
        except ValueError as error:
            assert "disagree" in str(error), error
        else:
            raise AssertionError("the plan check let a 10 mm disagreement through")
    assert robot_link.moves == []


def test_too_much_lean_is_refused_before_moving() -> None:
    """At 36 cm the calculated lean is about 26 degrees, above the limit, so the arm must stay where it is."""

    with tempfile.TemporaryDirectory() as folder:
        robot_link = PretendRobotLink()
        test = SideViewTest(robot_link, "test", fast_parameters(folder), scripted_keys([]))
        try:
            test.run("pull", 0.36)
        except ValueError as error:
            assert "lean" in str(error), error
        else:
            raise AssertionError("a 26 degree lean was allowed")
    assert robot_link.moves == []


def test_interrupt_while_down_lifts_before_going_home() -> None:
    """Ctrl-C with the tip down: the first move afterwards is straight up, then the forearm turns back, then home."""

    keys = ["enter"]

    def interrupted_key_reader(wait_ms: int) -> str:
        if keys:
            return keys.pop(0)
        raise KeyboardInterrupt()

    with tempfile.TemporaryDirectory() as folder:
        robot_link = PretendRobotLink()
        test = SideViewTest(robot_link, "test", fast_parameters(folder), interrupted_key_reader)
        try:
            test.run("view")
        except KeyboardInterrupt:
            pass

    turns = [round(numpy.degrees(joints[3])) for joints in robot_link.moves]
    tip_heights = [magnet_tip_and_lean(joints)[0][2] for joints in robot_link.moves]
    down_index = int(numpy.argmin(tip_heights))
    assert tip_heights[down_index + 1] > tip_heights[down_index] + 0.04, tip_heights
    assert turns[down_index:] == [90, 90, 60, 30, 0, 0], turns
    assert robot_link.moves[-1] == SIDE_VIEW_TEST_PARAMETERS["home_joints"] and robot_link.is_closed


if __name__ == "__main__":
    test_view_part_holds_the_tip_and_goes_home()
    test_pull_part_steps_towards_the_base()
    test_nothing_moves_when_the_model_and_the_robot_disagree()
    test_too_much_lean_is_refused_before_moving()
    test_interrupt_while_down_lifts_before_going_home()
    print("all tests passed")
