"""
Move to a joint pose, hold it for a while and poll the robot's error flags, to find a pose that overloads a joint.

Written 2.11.26: with bracket F fitted, joint 5 (wrist) raised hardware error 32 twice while the arm stood at the
home pose, 10 to 15 s after a clean test, never during the low poses. This script holds a pose for
"hold_seconds" and prints the error codes, the joint angles and the wrist's drift every "poll_seconds"; at the first
error it switches the motors off (learning mode) so nothing keeps pushing, and asks for Enter before re-enabling them.
Run from 02_code, nothing else connected, robot clear, stop button in reach; the pose is six joint angles in radians:
    ..\\.venv\\Scripts\\python.exe hold_test.py 0.0 0.5 -1.25 0.0 -0.5 0.0
"""

import sys
import time
from typing import List

from pyniryo import JointsPosition, NiryoRobot

from config import RECORDING_PARAMETERS, ROBOT_PARAMETERS

HOLD_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "joints": RECORDING_PARAMETERS["home_joints"],
    "hold_seconds": 60,
    "poll_seconds": 5
}


def hold_test(parameters: dict = None) -> List[int]:
    """Hold the pose and return the last error codes seen."""

    settings = dict(HOLD_PARAMETERS)
    settings.update(parameters or {})
    errors = []

    robot = NiryoRobot(settings["ip_address"])
    try:
        errors = list(robot.get_hardware_status().hardware_errors)
        if any(errors):
            print(f"the robot already reports hardware errors {errors}; restart it first")
            return errors
        robot.set_arm_max_velocity(settings["velocity_percent"])
        robot.move(JointsPosition(*settings["joints"]))
        commanded_wrist = settings["joints"][4]
        print(f"holding {settings['joints']} for {settings['hold_seconds']} s")

        started = time.monotonic()
        while time.monotonic() - started < settings["hold_seconds"]:
            time.sleep(settings["poll_seconds"])
            errors = list(robot.get_hardware_status().hardware_errors)
            joints = [round(value, 3) for value in robot.get_joints()]
            drift_deg = 57.3 * (joints[4] - commanded_wrist)
            print(f"{time.monotonic() - started:4.0f} s: errors {errors}, wrist drift {drift_deg:+.1f} deg, joints {joints}")
            if any(errors):
                print("ERROR raised while holding; motors OFF now so nothing keeps pushing")
                robot.set_learning_mode(True)
                input("Hold the arm, then press Enter to switch the motors back ON. ")
                robot.set_learning_mode(False)
                break
        else:
            print("held without error")
    finally:
        robot.close_connection()

    return errors


if __name__ == "__main__":
    hold_test({"joints": [float(value) for value in sys.argv[1:7]]} if len(sys.argv) == 7 else None)
