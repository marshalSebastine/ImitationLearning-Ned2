"""
Lift the camera to known heights above a text page on the sheet and show the sharpness number at each height.

The no-touch focus sweep of the camera-mount checklist (step 4), done with the robot instead of a hand-held page:
a printed text page lies flat on the sheet at the line's start, the robot goes to the start pose above it and then
stops at each height in "tip_heights_above_paper_m", showing the live picture with its sharpness number. The lens
sits about "lens_above_tip_m" higher than the rod's end (an estimate from 25.10.26, not a caliper measurement), so
the lens-to-paper distance shown is that estimate plus the tip height. Nothing touches the camera or its focus ring.

Keys in the window: n = next height, q = finish (the robot lifts and goes home). Run from 02_code, nothing else
connected to the robot, stop button in reach; optionally give the rod end's x and y in metres instead of the taught
start pose (added 2.11.26, so a new camera mount can be checked away from the near pose where it may touch the forearm):
    ..\\.venv\\Scripts\\python.exe focus_sweep.py
    ..\\.venv\\Scripts\\python.exe focus_sweep.py 0.22 0.0
"""

import sys
from typing import Dict, List

import cv2
import numpy
from pyniryo import JointsPosition, NiryoRobot, PoseObject

from config import RECORDING_PARAMETERS, ROBOT_PARAMETERS, pushing_height, start_pose

SWEEP_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "home_joints": RECORDING_PARAMETERS["home_joints"],
    "approach_height_m": RECORDING_PARAMETERS["approach_height_m"],
    "tip_heights_above_paper_m": [0.0, 0.02, 0.04, 0.07],
    "lens_above_tip_m": 0.08,
    # None = the taught start pose; otherwise the rod end's [x, y] in metres, at pushing height, rod vertical.
    "tip_xy": None
}


def sharpness_of(picture: numpy.ndarray) -> float:
    """Variance of the Laplacian, the number view_camera.py and record_episodes.py show: higher is sharper."""

    return float(cv2.Laplacian(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def hardware_errors_of(robot: NiryoRobot) -> List[int]:
    """The robot's per-motor hardware error codes (0 = none; 32 on a Dynamixel joint was an overload on 25.10 and 2.11)."""

    return list(robot.get_hardware_status().hardware_errors)


def check_hardware(
        robot: NiryoRobot,
        label: str
) -> bool:
    """Print the joints' error codes after a move; False (and a loud line) when any joint reports an error."""

    errors = hardware_errors_of(robot)
    joints = [round(value, 3) for value in robot.get_joints()]
    print(f"{label}: hardware errors {errors}, joints {joints}")
    if any(errors):
        print(f"STOP: joint error at {label}; lifting and going home")
        return False
    return True


def show_until_key(
        robot: NiryoRobot,
        caption: str
) -> str:
    """Show the live picture with its sharpness and ``caption`` until n or q is pressed; returns the key."""

    while True:
        picture = cv2.imdecode(numpy.frombuffer(robot.get_img_compressed(), numpy.uint8), cv2.IMREAD_COLOR)
        cv2.putText(picture, f"sharpness {sharpness_of(picture):.0f}   {caption}   n = next, q = finish", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("focus sweep", picture)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("n"), ord("q")):
            return chr(key)


def focus_sweep(parameters: dict = None) -> List[Dict[str, float]]:
    """
    Visit each height above the paper and return the sharpness read at the moment the key was pressed.

    Parameters
    ----------
    parameters : dict, optional
        Overrides merged over ``SWEEP_PARAMETERS``.
    """

    settings = dict(SWEEP_PARAMETERS)
    settings.update(parameters or {})
    x, y, z, roll, pitch, yaw = start_pose()
    if settings["tip_xy"] is not None:
        x, y = settings["tip_xy"]
        z = pushing_height(x, y)
    readings = []

    robot = NiryoRobot(settings["ip_address"])
    try:
        robot.set_arm_max_velocity(settings["velocity_percent"])
        if any(hardware_errors_of(robot)):
            print(f"the robot already reports hardware errors {hardware_errors_of(robot)}; restart it first")
            return readings
        robot.move(JointsPosition(*settings["home_joints"]))
        robot.move(PoseObject(x, y, z + settings["approach_height_m"], roll, pitch, yaw))
        if not check_hardware(robot, "above the paper"):
            robot.move(JointsPosition(*settings["home_joints"]))
            return readings

        for height in settings["tip_heights_above_paper_m"]:
            robot.move(PoseObject(x, y, z + height, roll, pitch, yaw), linear = True)
            if not check_hardware(robot, f"tip {100 * height:.0f} cm above the paper"):
                break
            lens_distance_cm = 100 * (height + settings["lens_above_tip_m"])
            key = show_until_key(robot, f"lens to paper about {lens_distance_cm:.0f} cm (estimate)")
            picture = cv2.imdecode(numpy.frombuffer(robot.get_img_compressed(), numpy.uint8), cv2.IMREAD_COLOR)
            readings.append({"tip_height_m": height, "lens_distance_cm_estimate": round(lens_distance_cm, 1),
                             "sharpness": round(sharpness_of(picture), 1)})
            print(f"tip {100 * height:.0f} cm above the paper, lens about {lens_distance_cm:.0f} cm: sharpness {readings[-1]['sharpness']:.0f}")
            if key == "q":
                break

        robot.move(PoseObject(x, y, z + settings["approach_height_m"], roll, pitch, yaw), linear = True)
        robot.move(JointsPosition(*settings["home_joints"]))
    finally:
        robot.close_connection()
        cv2.destroyAllWindows()

    return readings


if __name__ == "__main__":
    focus_sweep({"tip_xy": [float(sys.argv[1]), float(sys.argv[2])]} if len(sys.argv) == 3 else None)
