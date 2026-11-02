"""
Run the robot's AUTOMATIC calibration and report the result. Never a manual calibration.

Needed after a power-up, and whenever the robot reports "calibration needed" (then it accepts moves without
performing them, seen on 2.11.26 after the arm had been turned by hand with the power off). The arm moves on its own
to find its joint stops: clear the table around the robot, puck away, nobody touching the arm, stop button in reach.
Run from 02_code, nothing else connected:
    ..\\.venv\\Scripts\\python.exe calibrate_robot.py
"""

from pyniryo import NiryoRobot

from config import ROBOT_PARAMETERS


def calibrate_robot(ip_address: str = ROBOT_PARAMETERS["ip_address"]) -> bool:
    """Calibrate automatically if needed; returns True when the robot no longer needs calibration."""

    robot = NiryoRobot(ip_address)
    try:
        if not robot.need_calibration():
            print("no calibration needed")
            return True
        print("automatic calibration running; the arm moves by itself")
        robot.calibrate_auto()
        is_calibrated = not robot.need_calibration()
        print("calibration done" if is_calibrated else "calibration FAILED: the robot still needs calibration")
        print(f"joints (rad): {[round(value, 3) for value in robot.get_joints()]}")
        return is_calibrated
    finally:
        robot.close_connection()


if __name__ == "__main__":
    calibrate_robot()
