"""
Survey the robot more deeply than measure_robot.py does.

Without moving: the hardware status, the tool, how fast the state and the camera
can be read, how many camera pictures are really new, and what happens when a
second program connects. Moving: how long small shifts of the tool tip take, how
accurately they land, how the speed cap changes that, and whether the arm returns
to where it started. Every move is a shift of at most two centimetres next to
the home pose, under the speed cap. Clear the table and keep the stop button in
reach, then run from this folder:
    ..\\.venv\\Scripts\\python.exe survey_robot.py
"""

import json
import sys
import threading
import time
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List

import cv2
import numpy
from pyniryo import JointsPosition, NiryoRobot, RobotAxis

from config import ROBOT_PARAMETERS

SURVEY_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "home_joints": [0.0, 0.5, -1.25, 0.0, 0.0, 0.0],
    "state_sample_seconds": 10.0,
    "camera_frame_count": 300,
    "shift_millimetres": [1, 2, 5, 10, 20],
    "moves_per_shift": 4,
    "velocity_percents_for_one_centimetre": [20, 40],
    "second_client_wait_seconds": 10.0,
    "result_path": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "robot_survey" / "2026-10-24_survey.json"
}


def report_progress(
        name: str,
        results: Dict[str, object],
        result_path: Path
) -> None:
    """Say which part finished and save everything so far, so a hang never loses data."""

    result_path.parent.mkdir(parents = True, exist_ok = True)
    result_path.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    print(f"[survey] {name} done", file = sys.stderr, flush = True)


def timed_samples(
        call: Callable[[], object],
        count: int
) -> List[float]:
    """
    Run ``call`` ``count`` times and return the seconds each run took.

    Parameters
    ----------
    call : callable
        A zero-argument call to time.
    count : int
        How many times to run it.
    """

    durations = []
    for _ in range(count):
        started = time.perf_counter()
        call()
        durations.append(time.perf_counter() - started)
    return durations


def survey_status(robot: NiryoRobot) -> Dict[str, object]:
    """Read everything the robot reports about itself without moving it."""

    status = robot.get_hardware_status()
    joints = robot.get_joints()
    pose = robot.get_pose()
    # forward kinematics is what a workspace box needs when actions are joint angles,
    # so its answer is checked against the pose the robot itself reports
    predicted = robot.forward_kinematics(JointsPosition(*joints))
    kinematics_error_mm = 1000 * float(numpy.linalg.norm(
        numpy.array([pose.x, pose.y, pose.z]) - numpy.array([predicted.x, predicted.y, predicted.z])))

    return {
        "needs_calibration": robot.need_calibration(),
        "learning_mode": robot.get_learning_mode(),
        "tool_id": str(robot.get_current_tool_id()),
        "collision_detected": bool(robot.collision_detected),
        "hardware_version": getattr(status, "hardware_version", None),
        "rpi_temperature": getattr(status, "rpi_temperature", None),
        "motor_names": list(getattr(status, "motor_names", [])),
        "motor_temperatures": list(getattr(status, "temperatures", [])),
        "motor_voltages": list(getattr(status, "voltages", [])),
        "hardware_errors": list(getattr(status, "hardware_errors", [])),
        "error_message": getattr(status, "error_message", None),
        "joints_rad": list(joints),
        "pose_m_rad": [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw],
        "forward_kinematics_error_mm": kinematics_error_mm
    }


def survey_state_rate(
        robot: NiryoRobot,
        seconds: float
) -> Dict[str, float]:
    """
    Poll the pose as fast as possible for ``seconds`` and report the achieved rate.

    The recorder will poll like this, so the gaps between samples say how
    evenly the robot's state can be logged.
    """

    stamps = []
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        robot.get_pose()
        stamps.append(time.perf_counter())
    gaps = numpy.diff(numpy.array(stamps))

    return {
        "samples_per_second": len(stamps) / seconds,
        "median_gap_ms": 1000 * float(numpy.median(gaps)),
        "max_gap_ms": 1000 * float(gaps.max()),
        "gaps_above_100_ms": int((gaps > 0.1).sum())
    }


def survey_camera(
        robot: NiryoRobot,
        frame_count: int
) -> Dict[str, float]:
    """
    Read ``frame_count`` pictures through the robot and report the real picture rate.

    A read returns the latest JPEG even when the camera has not produced a new
    one, so identical consecutive pictures are counted to find how many are new.
    """

    stamps, sizes, decode_seconds, unique_count = [], [], [], 0
    previous = None
    started = time.perf_counter()
    for _ in range(frame_count):
        compressed = robot.get_img_compressed()
        stamps.append(time.perf_counter())
        sizes.append(len(compressed))
        if compressed != previous:
            unique_count += 1
            previous = compressed
        decode_started = time.perf_counter()
        frame = cv2.imdecode(numpy.frombuffer(compressed, numpy.uint8), cv2.IMREAD_COLOR)
        decode_seconds.append(time.perf_counter() - decode_started)
    elapsed = stamps[-1] - started
    height, width = frame.shape[:2]

    return {
        "reads_per_second": frame_count / elapsed,
        "new_pictures_per_second": unique_count / elapsed,
        "median_read_ms": 1000 * float(numpy.median(numpy.diff(numpy.array(stamps)))),
        "median_decode_ms": 1000 * median(decode_seconds),
        "median_jpeg_kilobytes": median(sizes) / 1000,
        "width": width,
        "height": height
    }


def pose_xyz_mm(robot: NiryoRobot) -> numpy.ndarray:
    """The tool tip position in millimetres, as the robot reports it."""

    pose = robot.get_pose()
    return 1000 * numpy.array([pose.x, pose.y, pose.z])


def survey_shifts(
        robot: NiryoRobot,
        shifts_mm: List[int],
        moves_per_shift: int
) -> Dict[str, Dict[str, float]]:
    """
    Shift the tool tip out and back along x and y and time every move.

    Each shift is commanded along one axis of the robot's base frame with
    ``shift_pose``, the call the control loop will use. Blocking time and the
    landing error against the commanded distance are recorded per shift size.
    """

    results = {}
    for shift_mm in shifts_mm:
        durations, landing_errors_mm = [], []
        for move_index in range(moves_per_shift):
            axis = 0 if move_index < moves_per_shift / 2 else 1
            sign = 1 if move_index % 2 == 0 else -1
            before = pose_xyz_mm(robot)
            started = time.perf_counter()
            robot.shift_pose(RobotAxis.X if axis == 0 else RobotAxis.Y, sign * shift_mm / 1000)
            durations.append(time.perf_counter() - started)
            moved = pose_xyz_mm(robot) - before
            landing_errors_mm.append(abs(moved[axis] - sign * shift_mm))
        results[f"shift_{shift_mm:02d}_mm"] = {
            "median_blocking_s": median(durations),
            "median_landing_error_mm": median(landing_errors_mm)
        }
    return results


def survey_velocity_cap(
        robot: NiryoRobot,
        velocity_percents: List[int],
        restore_percent: int
) -> Dict[str, float]:
    """Time a one-centimetre shift out and back under each speed cap, then restore the cap."""

    results = {}
    for percent in velocity_percents:
        robot.set_arm_max_velocity(percent)
        durations = []
        for sign in (1, -1, 1, -1):
            started = time.perf_counter()
            robot.shift_pose(RobotAxis.X, sign * 0.01)
            durations.append(time.perf_counter() - started)
        results[f"one_centimetre_at_{percent}_percent_s"] = median(durations)
    robot.set_arm_max_velocity(restore_percent)
    return results


def survey_second_client(
        ip_address: str,
        wait_seconds: float
) -> Dict[str, object]:
    """
    Open a second connection while the first is still open and report what happens.

    The recorder and the control loop must know whether a second program can
    share the robot or whether one process has to own it. The attempt runs in
    a background thread with a time limit, because the robot has been seen to
    accept the connection and then never answer, which would hang the survey.
    """

    outcome = {"second_client_connected": False, "answered_within_limit": False,
               "first_client_still_works": None, "error": None}

    def attempt() -> None:
        try:
            second = NiryoRobot(ip_address)
            outcome["second_client_connected"] = True
            second.get_joints()
            outcome["answered_within_limit"] = True
            second.close_connection()
        except Exception as error:
            outcome["error"] = f"{type(error).__name__}: {error}"

    worker = threading.Thread(target = attempt, daemon = True)
    worker.start()
    worker.join(wait_seconds)
    if worker.is_alive():
        outcome["error"] = f"no answer within {wait_seconds:.0f} s; a second client hangs"
    return outcome


def survey_robot(parameters: dict = None) -> Dict[str, object]:
    """
    Run every survey part in a safe order and return the combined results.

    Parameters
    ----------
    parameters : dict, optional
        Overrides merged over ``SURVEY_PARAMETERS``.
    """

    settings = dict(SURVEY_PARAMETERS)
    settings.update(parameters or {})
    results = {"measured_on": time.strftime("%d.%m.%y %H:%M"), "velocity_percent": settings["velocity_percent"]}

    result_path = settings["result_path"]
    robot = NiryoRobot(settings["ip_address"])
    try:
        robot.calibrate_auto()
        robot.set_arm_max_velocity(settings["velocity_percent"])
        robot.update_tool()
        results["status"] = survey_status(robot)
        report_progress("status", results, result_path)
        results["state_rate"] = survey_state_rate(robot, settings["state_sample_seconds"])
        report_progress("state rate", results, result_path)
        results["camera"] = survey_camera(robot, settings["camera_frame_count"])
        report_progress("camera", results, result_path)

        robot.move(JointsPosition(*settings["home_joints"]))
        home_xyz = pose_xyz_mm(robot)
        results["shifts"] = survey_shifts(robot, settings["shift_millimetres"], settings["moves_per_shift"])
        report_progress("shifts", results, result_path)
        results["velocity_cap"] = survey_velocity_cap(
            robot, settings["velocity_percents_for_one_centimetre"], settings["velocity_percent"])
        robot.move(JointsPosition(*settings["home_joints"]))
        results["return_to_home_error_mm"] = float(numpy.linalg.norm(pose_xyz_mm(robot) - home_xyz))
        report_progress("velocity cap and return home", results, result_path)

        # last, because a second client may break the first connection
        results["second_client"] = survey_second_client(settings["ip_address"], settings["second_client_wait_seconds"])
        try:
            robot.get_joints()
            results["second_client"]["first_client_still_works"] = True
        except Exception as error:
            results["second_client"]["first_client_still_works"] = False
            results["second_client"]["first_client_error"] = f"{type(error).__name__}: {error}"
        report_progress("second client", results, result_path)
    finally:
        try:
            robot.close_connection()
        except Exception:
            pass

    return results


if __name__ == "__main__":
    print(json.dumps(survey_robot(), indent = 2))
