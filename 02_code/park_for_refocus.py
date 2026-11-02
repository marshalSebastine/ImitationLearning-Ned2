"""
Park the rod at pushing height over a text page, switch the motors off so the arm rests on its rod, and wait.

For the one planned refocus of the camera (camera-mount checklist step 10, done on 2.11.26 with bracket F): with the
arm parked, the live picture with its sharpness number is shown in a window; a person turns the focus ring in small
steps until the number peaks on the text page. Press Enter in the terminal when done: the motors come back on, the
arm lifts and goes home. Run from 02_code, nothing else connected, stop button in reach:
    ..\\.venv\\Scripts\\python.exe park_for_refocus.py 0.22 0.0
"""

import sys
import threading

import cv2
import numpy
from pyniryo import JointsPosition, NiryoRobot, PoseObject

from config import RECORDING_PARAMETERS, ROBOT_PARAMETERS, WORKSPACE_PARAMETERS, pushing_height

PARK_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "home_joints": RECORDING_PARAMETERS["home_joints"],
    "approach_height_m": RECORDING_PARAMETERS["approach_height_m"],
    "tip_xy": [0.22, 0.0]
}


def sharpness_of(picture: numpy.ndarray) -> float:
    """Variance of the Laplacian, the number the other camera scripts show: higher is sharper."""

    return float(cv2.Laplacian(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def show_live_until(
        robot: NiryoRobot,
        is_finished: threading.Event
) -> None:
    """Show the live picture with its sharpness in a window until ``is_finished`` is set."""

    best = 0.0
    while not is_finished.is_set():
        picture = cv2.imdecode(numpy.frombuffer(robot.get_img_compressed(), numpy.uint8), cv2.IMREAD_COLOR)
        sharpness = sharpness_of(picture)
        best = max(best, sharpness)
        cv2.putText(picture, f"sharpness {sharpness:.0f}   best so far {best:.0f}   MOTORS OFF, turn the ring in small steps",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("refocus", picture)
        cv2.waitKey(30)
    cv2.destroyAllWindows()


def park_for_refocus(parameters: dict = None) -> None:
    """Go to the parking pose, rest the arm on its rod with the motors off, show the live number, then recover."""

    settings = dict(PARK_PARAMETERS)
    settings.update(parameters or {})
    x, y = settings["tip_xy"]
    z = pushing_height(x, y)
    pitch = WORKSPACE_PARAMETERS["tool_pitch_rad"]

    robot = NiryoRobot(settings["ip_address"])
    try:
        robot.set_arm_max_velocity(settings["velocity_percent"])
        robot.move(JointsPosition(*settings["home_joints"]))
        robot.move(PoseObject(x, y, z + settings["approach_height_m"], 0.0, pitch, 0.0))
        robot.move(PoseObject(x, y, z, 0.0, pitch, 0.0), linear = True)

        input("Parked at pushing height. Clear your hands. Press Enter to switch the motors OFF (the arm will rest on its rod). ")
        robot.set_learning_mode(True)
        print("MOTORS OFF. Turn the focus ring in small steps and watch the number in the window.")

        is_finished = threading.Event()
        viewer = threading.Thread(target = show_live_until, args = (robot, is_finished))
        viewer.start()
        input("Press Enter here when the number has peaked and your hands are off the camera: motors back ON, lift, home. ")
        is_finished.set()
        viewer.join()

        robot.set_learning_mode(False)
        print("motors ON")
        robot.move(PoseObject(x, y, z + settings["approach_height_m"], 0.0, pitch, 0.0), linear = True)
        robot.move(JointsPosition(*settings["home_joints"]))
    finally:
        robot.close_connection()


if __name__ == "__main__":
    park_for_refocus({"tip_xy": [float(sys.argv[1]), float(sys.argv[2])]} if len(sys.argv) == 3 else None)
