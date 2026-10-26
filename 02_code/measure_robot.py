"""
Measure how fast the robot answers and how long small moves take.

These numbers set the recording rate and the control period of the later phases.
The base turns by tiny steps at the capped speed; the rest of the arm stays in
the home pose. Clear the table and watch before running:
    ..\\.venv\\Scripts\\python.exe measure_robot.py
"""

import time
from statistics import median
from typing import Callable, Dict

from pyniryo import JointsPosition, NiryoRobot

from config import ROBOT_PARAMETERS

HOME_JOINTS = [0.0, 0.5, -1.25, 0.0, 0.0, 0.0]
STEP_SIZES_RAD = [0.01, 0.02, 0.05, 0.10]
SAMPLE_COUNT = 20
MOVES_PER_STEP = 6


def median_seconds(
        call: Callable[[], object],
        count: int
) -> float:
    """
    Time ``call`` ``count`` times and return the median duration in seconds.

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
    return median(durations)


def measure_robot(
        ip_address: str = ROBOT_PARAMETERS["ip_address"],
        velocity_percent: int = ROBOT_PARAMETERS["arm_max_velocity_percent"]
) -> Dict[str, float]:
    """
    Measure read latencies and the blocking time of small base moves.

    Parameters
    ----------
    ip_address : str
        The robot's address on the cable.
    velocity_percent : int
        Speed cap for the test moves, as a percentage of the robot's maximum.

    Returns
    -------
    dict of str to float
        Median seconds per call, keyed by what was measured.
    """

    robot = NiryoRobot(ip_address)
    results = {}

    try:
        robot.calibrate_auto()
        robot.set_arm_max_velocity(velocity_percent)
        robot.move(JointsPosition(*HOME_JOINTS))

        results["get_joints"] = median_seconds(robot.get_joints, SAMPLE_COUNT)
        results["get_pose"] = median_seconds(robot.get_pose, SAMPLE_COUNT)
        results["get_image"] = median_seconds(robot.get_img_compressed, SAMPLE_COUNT)

        # each step size is driven out and back, so the base never drifts from home
        for step in STEP_SIZES_RAD:
            durations = []
            for move_index in range(MOVES_PER_STEP):
                target = HOME_JOINTS.copy()
                target[0] = step if move_index % 2 == 0 else 0.0
                started = time.perf_counter()
                robot.move(JointsPosition(*target))
                durations.append(time.perf_counter() - started)
            results[f"move_base_{step:.2f}_rad"] = median(durations)

        robot.move(JointsPosition(*HOME_JOINTS))
    finally:
        robot.close_connection()

    return results


if __name__ == "__main__":
    for name, seconds in measure_robot().items():
        print(f"{name:22s} {seconds * 1000:7.1f} ms")
