"""
Time the ways of making a one-centimetre tip step, to choose the control loop's move call.

The survey showed that shifting the tip with the robot's pose command takes about two
seconds even for one centimetre, while a small joint move takes under one. This
script times, for the same one-centimetre step along x, out and back:
    pose command, joint-interpolated (what shift_pose does);
    pose command, linear in space;
    inverse kinematics on the robot, then a joint move (the IK call timed on its own).
The arm stays next to the home pose under the speed cap. Clear the table and keep the
stop button in reach, then run from this folder:
    ..\\.venv\\Scripts\\python.exe measure_step_methods.py
"""

import json
import time
from pathlib import Path
from statistics import median
from typing import Callable, Dict

from pyniryo import JointsPosition, NiryoRobot, PoseObject, RobotAxis

from config import ROBOT_PARAMETERS

STEP_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "home_joints": [0.0, 0.5, -1.25, 0.0, 0.0, 0.0],
    "step_metres": 0.01,
    "repeats": 4,
    "result_path": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "robot_survey" / "2026-10-24_survey.json"
}


def shifted_target(
        robot: NiryoRobot,
        step_metres: float
) -> PoseObject:
    """The current pose moved by ``step_metres`` along x, everything else unchanged."""

    pose = robot.get_pose()
    return PoseObject(pose.x + step_metres, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)


def time_method(
        move_once: Callable[[float], None],
        step_metres: float,
        repeats: int
) -> float:
    """Median seconds of ``move_once`` over ``repeats`` out-and-back steps of ``step_metres``."""

    durations = []
    for index in range(repeats):
        sign = 1 if index % 2 == 0 else -1
        started = time.perf_counter()
        move_once(sign * step_metres)
        durations.append(time.perf_counter() - started)
    return median(durations)


def measure_step_methods(parameters: dict = None) -> Dict[str, float]:
    """
    Time each step method and append the medians to the survey results file.

    Parameters
    ----------
    parameters : dict, optional
        Overrides merged over ``STEP_PARAMETERS``.
    """

    settings = dict(STEP_PARAMETERS)
    settings.update(parameters or {})
    step, repeats = settings["step_metres"], settings["repeats"]
    kinematics_seconds = []

    def by_inverse_kinematics(signed_step: float) -> None:
        started = time.perf_counter()
        answer = robot.inverse_kinematics(shifted_target(robot, signed_step))
        kinematics_seconds.append(time.perf_counter() - started)
        robot.move(answer if isinstance(answer, JointsPosition) else JointsPosition(*answer))

    robot = NiryoRobot(settings["ip_address"])
    try:
        if robot.need_calibration():
            robot.calibrate_auto()
        robot.set_arm_max_velocity(settings["velocity_percent"])
        robot.move(JointsPosition(*settings["home_joints"]))

        results = {
            "pose_command_joint_interpolated_s": time_method(
                lambda signed: robot.shift_pose(RobotAxis.X, signed), step, repeats),
            "pose_command_linear_s": time_method(
                lambda signed: robot.move(shifted_target(robot, signed), linear = True), step, repeats),
            "inverse_kinematics_then_joint_move_s": time_method(by_inverse_kinematics, step, repeats),
            "inverse_kinematics_call_s": median(kinematics_seconds)
        }
        robot.move(JointsPosition(*settings["home_joints"]))
    finally:
        robot.close_connection()

    path = settings["result_path"]
    survey = json.loads(path.read_text(encoding = "utf-8")) if path.exists() else {}
    survey["step_methods_one_centimetre"] = results
    path.write_text(json.dumps(survey, indent = 2), encoding = "utf-8")
    return results


if __name__ == "__main__":
    for name, seconds in measure_step_methods().items():
        print(f"{name:40s} {seconds:6.2f} s")
