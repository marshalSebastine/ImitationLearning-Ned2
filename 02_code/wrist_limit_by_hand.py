"""
Measure by hand the wrist angle at which the camera bracket touches the forearm. The motors are off throughout.

Written 2.11.26 for bracket F. The script switches the motors off (one person holds the arm at the elbow first),
prints the wrist joint (joint 5) twice a second while the other person tilts the wrist slowly towards the forearm,
and records the angle when Enter is pressed at the moment the bracket just touches. Then it asks for the arm to be
released and switches the motors back on. Put the measured value minus a margin into
``RECORDING_PARAMETERS["wrist_pitch_max_rad"]``. Run from 02_code, nothing else connected, in a terminal:
    ..\\.venv\\Scripts\\python.exe wrist_limit_by_hand.py
"""

import threading
import time

from pyniryo import NiryoRobot

from config import ROBOT_PARAMETERS

MARGIN_RAD = 0.05


def print_wrist_until(
        robot: NiryoRobot,
        is_finished: threading.Event
) -> None:
    """Print the wrist angle twice a second until ``is_finished`` is set."""

    while not is_finished.is_set():
        joints = robot.get_joints()
        print(f"\r  wrist (joint 5): {joints[4]:+.3f} rad = {57.3 * joints[4]:+.1f} deg      ", end = "", flush = True)
        time.sleep(0.5)
    print()


def wrist_limit_by_hand(ip_address: str = ROBOT_PARAMETERS["ip_address"]) -> float:
    """Return the wrist angle read when the bracket touched."""

    robot = NiryoRobot(ip_address)
    touched = float("nan")
    try:
        input("Hold the arm at the elbow. Press Enter to switch the motors OFF (the arm will sag). ")
        robot.set_learning_mode(True)
        print("MOTORS OFF. Tilt the wrist slowly towards the forearm. Press Enter the moment the bracket touches.")

        is_finished = threading.Event()
        printer = threading.Thread(target = print_wrist_until, args = (robot, is_finished))
        printer.start()
        input()
        is_finished.set()
        printer.join()
        touched = robot.get_joints()[4]
        print(f"bracket touches at wrist = {touched:+.3f} rad ({57.3 * touched:+.1f} deg); "
              f"safe limit with {57.3 * MARGIN_RAD:.0f} degrees margin: {touched - MARGIN_RAD:+.3f} rad")

        input("Tilt the wrist back away from the forearm, keep holding the arm, then press Enter: motors ON. ")
        robot.set_learning_mode(False)
        print("motors ON; you can let go")
    finally:
        robot.close_connection()

    return touched


if __name__ == "__main__":
    wrist_limit_by_hand()
