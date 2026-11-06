"""
Tests of calibrate_forearm_camera.py on the simulated robot, with faults injected; no robot is needed.

Every test runs the real routine on a short plan (two spots, three survey poses) and then checks, independently of the
routine's own checks, every move the simulated robot actually made against where the simulated puck actually was.

Run from 02_code (or with 02_code on PYTHONPATH); the run folders go to the system's temporary folder, or under
CALIBRATION_TEST_FOLDER when that environment variable is set:
    python test_calibrate_forearm_camera.py
"""

import copy
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy

from calibrate_forearm_camera import (CALIBRATION_CAPTURE_PARAMETERS, CalibrationPlanner, ForearmCameraCalibration,
                                      block_geometry, check_joint_path, puck_top_centre)
from config import PUCK_PARAMETERS, table_height
from robot_link import SimulatedRobotLink
from side_view_test import magnet_tip_and_lean, ned2_fk

SHORT_SURVEY = [
    {"twist_deg": -60.0, "ahead_m": 0.050, "left_m": 0.000, "clearance_m": 0.030},
    {"twist_deg": 0.0, "ahead_m": 0.030, "left_m": 0.000, "clearance_m": 0.030},
    {"twist_deg": 60.0, "ahead_m": 0.070, "left_m": -0.030, "clearance_m": 0.050}
]
TEST_PARAMETERS = dict(CALIBRATION_CAPTURE_PARAMETERS)
TEST_PARAMETERS.update({"spots_xy": [[0.22, 0.04], [0.37, -0.06]], "survey_poses": SHORT_SURVEY, "settle_s": 0.0})
PLAN_CACHE = {}



class FaultyRobotLink(SimulatedRobotLink):

    def __init__(
            self,
            fault: Optional[Dict] = None
    ):
        """
        The simulated robot, recording every move with where the simulated puck was, and failing as ``fault`` says.

        Parameters
        ----------
        fault : dict, optional
            "kind": "interrupt" (Ctrl-C after move "at_move"), "stuck" (from move "at_move" on, the first move of more
            than 10 degrees is acknowledged but not performed, as with "calibration needed" on 2.11.26), "camera"
            (picture reads fail after "after_pictures") or "model" (its own kinematics 10 mm off the arm model).
        """

        super().__init__({"puck_xy": [1.0, 1.0], "move_seconds": 0.0})
        self.fault = fault or {}
        self.executed = []
        self.picture_reads = 0
        self.is_stuck_triggered = False
        self.is_closed = False


    def move_joints(self, joints: List[float]) -> None:

        start = list(self.joints)
        target = [float(value) for value in joints]
        largest_change_deg = max(abs(math.degrees(end - begin)) for begin, end in zip(start, target))
        if (self.fault.get("kind") == "stuck" and not self.is_stuck_triggered and len(self.executed) + 1 >= self.fault["at_move"]
                and largest_change_deg > 10.0):
            self.is_stuck_triggered = True
            target = list(start)
        self.executed.append({"start": start, "end": target, "puck_xy": numpy.array(self.puck_xy, dtype = float)})
        super().move_joints(target)
        if self.fault.get("kind") == "interrupt" and len(self.executed) == self.fault["at_move"]:
            raise KeyboardInterrupt()


    def predicted_position(self, joints: List[float]) -> List[float]:

        position = super().predicted_position(joints)
        if self.fault.get("kind") == "model":
            position[0] += 0.010
        return position


    def read_picture_bytes(self) -> bytes:

        self.picture_reads += 1
        if self.fault.get("kind") == "camera" and self.picture_reads > self.fault["after_pictures"]:
            raise ConnectionError("simulated camera stream lost")
        return super().read_picture_bytes()


    def close(self) -> None:

        self.is_closed = True



def scripted_keys(keys: List[str]):
    """A key reader that answers the routine's questions in order, then always Enter."""

    remaining = list(keys)
    return lambda wait_ms: remaining.pop(0) if remaining else "enter"


def short_plan() -> Dict:
    """The two-spot, three-pose plan, solved once for all tests."""

    if "plan" not in PLAN_CACHE:
        PLAN_CACHE["plan"] = CalibrationPlanner(TEST_PARAMETERS).plan_capture(TEST_PARAMETERS["spots_xy"], say = lambda message: None)
    return copy.deepcopy(PLAN_CACHE["plan"])


def run_capture(
        robot_link: FaultyRobotLink,
        keys: List[str],
        plan: Optional[Dict] = None
) -> tuple:
    """Run the routine on ``robot_link`` with scripted keys; returns the summary and the run folder's tables."""

    root = os.environ.get("CALIBRATION_TEST_FOLDER")
    if root:
        Path(root).mkdir(parents = True, exist_ok = True)
    folder = Path(tempfile.mkdtemp(prefix = "calibration_test_", dir = root))
    capture = ForearmCameraCalibration(robot_link, "test", plan or short_plan(), folder, TEST_PARAMETERS,
                                       key_reader = scripted_keys(keys), is_window_shown = False, is_simulated = True)
    summary = capture.run()
    tables = {}
    for name in ("captures", "spots", "moves"):
        with open(folder / f"{name}.csv", newline = "", encoding = "utf-8") as file:
            tables[name] = list(csv.DictReader(file))
    return summary, tables


def lowest_clearance_over_puck(
        executed: List[Dict],
        sample_deg: float = 0.25
) -> Dict:
    """
    Independent re-check of what the simulated robot did: along every executed move (straight joint line, sampled
    every ``sample_deg``), the lowest point of the block above the top of the puck where the puck really was, wherever
    the block is within 4 cm of it sideways; also the lowest above the table. Moves that start at the hover (the rise
    from the puck) are reported apart: "hover_m" while still under 2 cm, "hover_rise_near_puck_m" above that.
    """

    footprint = TEST_PARAMETERS["block_footprint_radius_m"]
    puck_radius = PUCK_PARAMETERS["diameter_m"] / 2
    lowest = {"near_puck_m": math.inf, "table_m": math.inf, "hover_m": math.inf, "hover_rise_near_puck_m": math.inf}
    for move in executed:
        puck_xy = move["puck_xy"]
        if abs(puck_xy[0]) > 0.9:
            puck_top = None
        else:
            puck_top = puck_top_centre(list(puck_xy))
        largest_deg = max(abs(math.degrees(end - begin)) for begin, end in zip(move["start"], move["end"]))
        count = max(2, int(math.ceil(largest_deg / sample_deg)) + 1)
        start_geometry = block_geometry(move["start"], footprint)
        for step in range(count):
            joints = [begin + (end - begin) * step / (count - 1) for begin, end in zip(move["start"], move["end"])]
            geometry = block_geometry(joints, footprint)
            tip_xy = geometry["tip"][:2]
            lowest["table_m"] = min(lowest["table_m"], geometry["lowest_z"] - table_height(float(tip_xy[0]), float(tip_xy[1])))
            if puck_top is None:
                continue
            gap = float(numpy.linalg.norm(tip_xy - puck_top[:2])) - footprint - puck_radius
            if gap >= 0.04:
                continue
            clearance = geometry["lowest_z"] - float(puck_top[2])
            is_hover_rise = start_geometry["lowest_z"] - float(puck_top[2]) < 0.0125
            if is_hover_rise and clearance < 0.02:
                lowest["hover_m"] = min(lowest["hover_m"], clearance)
            elif is_hover_rise:
                lowest["hover_rise_near_puck_m"] = min(lowest["hover_rise_near_puck_m"], clearance)
            else:
                lowest["near_puck_m"] = min(lowest["near_puck_m"], clearance)
    return lowest


def is_at_home(robot_link: FaultyRobotLink) -> bool:

    return numpy.allclose(robot_link.joints, TEST_PARAMETERS["home_joints"], atol = 1e-6)


def first_move_after_fault_rises(
        robot_link: FaultyRobotLink,
        fault_index: int
) -> bool:
    """The first move after move ``fault_index`` goes up (or stays) and not sideways, or the arm was already high."""

    footprint = TEST_PARAMETERS["block_footprint_radius_m"]
    move = robot_link.executed[fault_index]
    start, end = block_geometry(move["start"], footprint), block_geometry(move["end"], footprint)
    sideways = float(numpy.linalg.norm(end["tip"][:2] - start["tip"][:2]))
    return end["lowest_z"] >= start["lowest_z"] - 1e-4 and (sideways < 0.002 or start["lowest_z"] - table_height(*start["tip"][:2]) > 0.10)


def test_full_capture_goes_home_and_keeps_clear() -> None:
    """All pictures taken, the puck measured where the tip hovered, every executed move clear of the real puck."""

    robot_link = FaultyRobotLink()
    summary, tables = run_capture(robot_link, ["enter", "enter", "enter"])

    assert summary["status"] == "completed", summary["status"]
    assert len(tables["captures"]) == 6 and summary["pictures"] == 6, summary
    assert is_at_home(robot_link) and robot_link.is_closed
    assert all(row["is_safe"] == "1" for row in tables["moves"]), [row for row in tables["moves"] if row["is_safe"] != "1"]
    for row in tables["spots"]:
        planned = numpy.array([float(row["planned_x"]), float(row["planned_y"])])
        measured = numpy.array([float(row["puck_top_x"]), float(row["puck_top_y"])])
        assert numpy.linalg.norm(planned - measured) < 1e-3, row
    for row in tables["captures"]:
        assert float(row["clearance_m"]) >= 0.0299, row
        assert float(row["joint_error_max_deg"]) < 0.01, row
    lowest = lowest_clearance_over_puck(robot_link.executed)
    assert lowest["near_puck_m"] >= 0.0199, lowest
    assert lowest["hover_m"] >= 0.0099, lowest
    assert lowest["table_m"] >= 0.0199, lowest
    print(f"full capture: lowest block above the real puck within 4 cm sideways: {100 * lowest['near_puck_m']:.2f} cm on the "
          f"survey and travel moves, {100 * lowest['hover_m']:.2f} cm at the hover, {100 * lowest['hover_rise_near_puck_m']:.2f} cm "
          f"on the rise once past 2 cm; above the table {100 * lowest['table_m']:.2f} cm; {len(robot_link.executed)} moves")


def test_escape_at_the_second_spot_rises_and_goes_home() -> None:

    robot_link = FaultyRobotLink()
    summary, tables = run_capture(robot_link, ["enter", "enter", "escape"])

    assert summary["status"] == "stopped_by_operator", summary["status"]
    assert summary["retreat"].startswith("completed"), summary["retreat"]
    assert len(tables["captures"]) == 3, len(tables["captures"])
    assert is_at_home(robot_link) and robot_link.is_closed
    hover_index = int(numpy.argmin([block_geometry(move["end"], 0.02)["lowest_z"] for move in robot_link.executed[-8:]])) + len(robot_link.executed) - 8
    assert first_move_after_fault_rises(robot_link, hover_index + 1), "the first move after Esc at the hover was not straight up"
    lowest = lowest_clearance_over_puck(robot_link.executed)
    assert lowest["near_puck_m"] >= 0.0199, lowest


def test_skip_leaves_the_spot_out() -> None:

    robot_link = FaultyRobotLink()
    summary, tables = run_capture(robot_link, ["enter", "s", "enter"])

    assert summary["status"] == "completed", summary["status"]
    assert [row["status"] for row in tables["spots"]] == ["skipped", "captured"], tables["spots"]
    assert {row["spot"] for row in tables["captures"]} == {"2"}, tables["captures"]
    assert is_at_home(robot_link)
    # After s the puck may still lie under the block: the rise from the hover is the same two short moves.
    rise_rows = [row for row in tables["moves"] if row["checked_at"] == "run" and row["spot"] == "1" and row["move"].startswith("hover_up")]
    assert [row["move"] for row in rise_rows] == ["hover_up_1", "hover_up_2"], rise_rows
    assert float(rise_rows[0]["max_rise_sideways_m"]) < 0.002, rise_rows[0]


def test_ctrl_c_during_the_survey_rises_first_then_goes_home() -> None:

    for at_move in (14, 17, 21):
        robot_link = FaultyRobotLink({"kind": "interrupt", "at_move": at_move})
        summary, tables = run_capture(robot_link, ["enter", "enter", "enter"])
        assert summary["status"] == "stopped_by_ctrl_c", summary["status"]
        assert summary["retreat"].startswith("completed"), summary["retreat"]
        assert is_at_home(robot_link) and robot_link.is_closed, at_move
        assert first_move_after_fault_rises(robot_link, at_move), f"Ctrl-C after move {at_move}: the next move was not straight up"
        lowest = lowest_clearance_over_puck(robot_link.executed)
        assert lowest["near_puck_m"] >= 0.0199, (at_move, lowest)


def test_lost_camera_ends_safely() -> None:
    """The camera stream dies during the survey of the first spot: an error, then rise, turn back, home."""

    robot_link = FaultyRobotLink({"kind": "camera", "after_pictures": 120})
    summary, tables = run_capture(robot_link, ["enter", "enter", "enter"])

    assert summary["status"].startswith("error: ConnectionError"), summary["status"]
    assert summary["retreat"].startswith("completed"), summary["retreat"]
    assert 0 < len(tables["captures"]) < 6, len(tables["captures"])
    assert is_at_home(robot_link) and robot_link.is_closed
    assert lowest_clearance_over_puck(robot_link.executed)["near_puck_m"] >= 0.0199


def test_a_robot_that_stops_moving_is_noticed() -> None:
    """A move acknowledged but not performed is caught from the joints read back; the retreat then works again."""

    robot_link = FaultyRobotLink({"kind": "stuck", "at_move": 14})
    summary, tables = run_capture(robot_link, ["enter", "enter", "enter"])

    assert summary["status"].startswith("error: UnsafeMove") and "short" in summary["status"], summary["status"]
    assert summary["retreat"].startswith("completed"), summary["retreat"]
    assert is_at_home(robot_link) and robot_link.is_closed
    assert lowest_clearance_over_puck(robot_link.executed)["near_puck_m"] >= 0.0199


def test_a_survey_pose_too_close_to_the_puck_is_refused_before_moving() -> None:
    """A survey pose replaced by one 1 cm over the puck: the descent is refused, the arm never goes there."""

    plan = short_plan()
    bad_joints = list(plan["spots"][0]["hover"])
    plan["spots"][0]["survey"][0]["joints"] = bad_joints
    robot_link = FaultyRobotLink()
    summary, tables = run_capture(robot_link, ["enter", "enter", "enter"], plan)

    assert summary["status"].startswith("error: UnsafeMove") and "refused before moving" in summary["status"], summary["status"]
    hover_visits = [move for move in robot_link.executed if numpy.allclose(move["end"], bad_joints)]
    assert len(hover_visits) == 1, "the arm went back down to 1 cm over the placed puck"
    assert is_at_home(robot_link)


def test_nothing_moves_when_the_model_and_the_robot_disagree() -> None:

    robot_link = FaultyRobotLink({"kind": "model"})
    summary, tables = run_capture(robot_link, ["enter"])

    assert summary["status"].startswith("error: UnsafeMove") and "disagree" in summary["status"], summary["status"]
    assert summary["retreat"].startswith("not needed"), summary["retreat"]
    assert robot_link.executed == [] and robot_link.is_closed


def test_the_rise_from_the_hover_must_be_straight() -> None:
    """A rise that drifts 5 mm sideways in the first centimetre is refused; the planned rise passes, in two moves."""

    plan = short_plan()
    spot = plan["spots"][0]
    puck_top = spot["puck_top"]
    planner = CalibrationPlanner(TEST_PARAMETERS)
    drifted = planner.solve_at_clearance(numpy.array(spot["spot_xy"]) + numpy.array([0.005, 0.0]), 0.0, float(puck_top[2]), 0.02,
                                         TEST_PARAMETERS["survey_max_lean_deg"])
    first = check_joint_path(spot["hover"], spot["hover_step"], [puck_top], TEST_PARAMETERS, puck_top, is_rise = True)
    second = check_joint_path(spot["hover_step"], spot["above_hover"], [puck_top], TEST_PARAMETERS, puck_top, is_rise = True)
    bad = check_joint_path(spot["hover"], drifted, [puck_top], TEST_PARAMETERS, puck_top, is_rise = True)
    assert first["is_safe"] and first["max_rise_sideways_m"] < 0.002, first
    assert second["is_safe"] and second["min_near_puck_clearance_m"] >= 0.0249, second
    assert not bad["is_safe"] and any("sideways" in reason for reason in bad["reasons"]), bad
    print(f"rise from the hover: first move at most {1000 * first['max_rise_sideways_m']:.2f} mm sideways, never down; "
          f"second move at least {100 * second['min_near_puck_clearance_m']:.2f} cm above the puck")


def test_an_arm_left_low_rises_straight_before_going_home() -> None:
    """
    Started with the block 5 mm over the table (the recording tool's start pose): Esc at the first question moves
    nothing; Enter makes the first move straight up, then home, the first spot, and Esc there brings it home.
    """

    planner = CalibrationPlanner(TEST_PARAMETERS)
    start_xy = numpy.array([0.1522, 0.0374])
    low = planner.solve_at_clearance(start_xy, 0.0, table_height(*start_xy), 0.005, TEST_PARAMETERS["survey_max_lean_deg"])
    robot_link = FaultyRobotLink()
    robot_link.move_joints(low)
    robot_link.executed = []
    summary, tables = run_capture(robot_link, ["escape"])

    assert summary["status"] == "stopped_by_operator", summary["status"]
    assert robot_link.executed == [] and robot_link.is_closed
    robot_link2 = FaultyRobotLink()
    robot_link2.move_joints(low)
    robot_link2.executed = []
    summary2, tables2 = run_capture(robot_link2, ["enter", "escape"])
    assert summary2["status"] == "stopped_by_operator", summary2["status"]
    assert first_move_after_fault_rises(robot_link2, 0), "the first move from low down was not straight up"
    assert lowest_clearance_over_puck(robot_link2.executed)["table_m"] >= 0.0049
    assert is_at_home(robot_link2)


if __name__ == "__main__":
    test_full_capture_goes_home_and_keeps_clear()
    test_escape_at_the_second_spot_rises_and_goes_home()
    test_skip_leaves_the_spot_out()
    test_ctrl_c_during_the_survey_rises_first_then_goes_home()
    test_lost_camera_ends_safely()
    test_a_robot_that_stops_moving_is_noticed()
    test_a_survey_pose_too_close_to_the_puck_is_refused_before_moving()
    test_nothing_moves_when_the_model_and_the_robot_disagree()
    test_the_rise_from_the_hover_must_be_straight()
    test_an_arm_left_low_rises_straight_before_going_home()
    print("all tests passed")
