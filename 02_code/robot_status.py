"""
Ask the robot how it is, without moving it: hardware errors, calibration and collision flags, where the arm is, and
whether its inverse kinematics can solve the recording tool's start pose right now.

Written 3.11.26 after the recording tool's first move was refused with "INVERSE_KINEMATICS KO" (error -232) at a pose
that had worked the day before. Run from 02_code, nothing else connected:
    ..\\.venv\\Scripts\\python.exe robot_status.py [start_x start_y]
"""

import math
import sys

from pyniryo import NiryoRobot, PoseObject

from config import RECORDING_PARAMETERS, ROBOT_PARAMETERS, WORKSPACE_PARAMETERS, camera_heading_rad, pushing_height


def start_pose_for(
        puck_x: float,
        puck_y: float
) -> list:
    """The rod's pose behind the puck, as the recording tool computes it."""

    orientation = [0.0, WORKSPACE_PARAMETERS["tool_pitch_rad"], 0.0]
    heading = camera_heading_rad([puck_x, puck_y, 0.0] + orientation)
    offset = RECORDING_PARAMETERS["puck_centre_ahead_of_tip_m"]
    x = puck_x - offset * math.cos(heading)
    y = puck_y - offset * math.sin(heading)
    return [x, y, pushing_height(x, y)] + orientation


def try_inverse_kinematics(
        robot: NiryoRobot,
        label: str,
        pose: list
) -> None:
    """Print the joints the robot finds for ``pose``, or the error it raises."""

    rounded = [round(value, 3) for value in pose]
    try:
        joints = [round(float(value), 3) for value in robot.inverse_kinematics(PoseObject(*pose))]
        print(f"IK {label} {rounded}: joints {joints}")
    except Exception as error:
        print(f"IK {label} {rounded}: REFUSED ({str(error).splitlines()[0]})")


def report(ip_address: str = ROBOT_PARAMETERS["ip_address"]) -> None:

    robot = NiryoRobot(ip_address)
    try:
        status = robot.get_hardware_status()
        print(f"hardware errors per motor: {list(status.hardware_errors)}")
        print(f"error message: '{status.error_message}', calibration needed: {status.calibration_needed}, "
              f"calibration in progress: {status.calibration_in_progress}")
        print(f"raspberry pi temperature: {status.rpi_temperature}")
        for name in ("get_learning_mode", "collision_detected"):
            try:
                value = getattr(robot, name)
                print(f"{name}: {value() if callable(value) else value}")
            except Exception as error:
                print(f"{name}: not available ({error})")
        joints = [round(float(value), 3) for value in robot.get_joints()]
        pose = robot.get_pose()
        print(f"joints now: {joints}")
        print(f"pose now: x {pose.x:.3f} y {pose.y:.3f} z {pose.z:.3f} roll {pose.roll:.2f} pitch {pose.pitch:.2f} yaw {pose.yaw:.2f}")

        puck_x, puck_y = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) == 3 else RECORDING_PARAMETERS.get(
            "start_point_xy", WORKSPACE_PARAMETERS["line_start_xy"])
        start = start_pose_for(puck_x, puck_y)
        above = list(start)
        above[2] += RECORDING_PARAMETERS["approach_height_m"]
        try_inverse_kinematics(robot, "start", start)
        try_inverse_kinematics(robot, "above start", above)
        lifted = [pose.x, pose.y, pose.z + RECORDING_PARAMETERS["approach_height_m"], pose.roll, pose.pitch, pose.yaw]
        try_inverse_kinematics(robot, "5 cm above where the arm is", lifted)
    finally:
        robot.close_connection()


if __name__ == "__main__":
    report()
