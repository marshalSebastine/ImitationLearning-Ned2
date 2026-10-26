"""
Connect to the Ned2, print its state and disconnect. Nothing moves.

Run from this folder with the project environment:
    ..\\.venv\\Scripts\\python.exe check_hardware.py
"""

from typing import Callable

from pyniryo import NiryoRobot

from config import ROBOT_PARAMETERS


def describe(
        label: str,
        read: Callable[[], object]
) -> None:
    """
    Print one reading, or the error it raised, so one failing call does not hide the others.

    Parameters
    ----------
    label : str
        What is being read, as shown on the line.
    read : callable
        A zero-argument call that returns the value.
    """

    try:
        print(f"{label}: {read()}")
    except Exception as error:
        print(f"{label}: not available ({error})")


def report_robot_state(ip_address: str = ROBOT_PARAMETERS["ip_address"]) -> None:
    """
    Connect to the robot, print every state reading we rely on, and close the connection.

    Parameters
    ----------
    ip_address : str
        The robot's address on the cable or the hotspot.
    """

    robot = NiryoRobot(ip_address)
    print(f"Connected to {ip_address}")

    try:
        describe("Calibration needed", robot.need_calibration)
        describe("Learning mode", robot.get_learning_mode)
        describe("Joints (rad)", robot.get_joints)
        describe("Pose (m, rad)", robot.get_pose)
        describe("Current tool id", robot.get_current_tool_id)
        describe("Hardware status", robot.get_hardware_status)
    finally:
        robot.close_connection()
        print("Disconnected")


if __name__ == "__main__":
    report_robot_state()
