"""
Move the robot to the start pose and the goal pose on its own and photograph the view.

This is the first time the robot, not a hand, puts the rod behind the puck. It goes
from the home pose to a point above the start, lowers to pushing height, takes a
picture, lifts, does the same at the goal (no puck there), and returns home. The
puck must sit at the start of the line. Speed is capped; clear the area and keep
the stop button in reach:
    ..\\.venv\\Scripts\\python.exe check_start_view.py
"""

import math
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy
from pyniryo import JointsPosition, NiryoRobot, PoseObject

from config import ROBOT_PARAMETERS, goal_pose, start_pose

START_VIEW_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "home_joints": [0.0, 0.5, -1.25, 0.0, 0.0, 0.0],
    "approach_height_m": 0.05,
    "tool_pitch_rad": None,
    "picture_name": "",
    "picture_folder": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "workspace"
}


def save_picture(
        robot: NiryoRobot,
        path: Path
) -> None:
    """Save what the camera sees now."""

    frame = cv2.imdecode(numpy.frombuffer(robot.get_img_compressed(), numpy.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(path), frame)


def visit(
        robot: NiryoRobot,
        pose: list,
        approach_height_m: float,
        picture_path: Path
) -> Dict[str, float]:
    """Come down onto ``pose`` from above, photograph, lift again; return where the tip landed."""

    x, y, z, roll, pitch, yaw = pose
    robot.move(PoseObject(x, y, z + approach_height_m, roll, pitch, yaw))
    robot.move(PoseObject(x, y, z, roll, pitch, yaw), linear = True)
    time.sleep(0.5)
    save_picture(robot, picture_path)
    landed = robot.get_pose()
    joints = [round(math.degrees(j)) for j in robot.get_joints()]
    robot.move(PoseObject(x, y, z + approach_height_m, roll, pitch, yaw), linear = True)
    return {"x": landed.x, "y": landed.y, "z": landed.z, "pitch_deg": math.degrees(landed.pitch), "joints_deg": joints}


def check_start_view(parameters: dict = None) -> Dict[str, Dict[str, float]]:
    """
    Visit the start pose and the goal pose, saving a picture at each.

    Parameters
    ----------
    parameters : dict, optional
        Overrides merged over ``START_VIEW_PARAMETERS``.
    """

    settings = dict(START_VIEW_PARAMETERS)
    settings.update(parameters or {})
    folder = settings["picture_folder"]
    folder.mkdir(parents = True, exist_ok = True)
    day = time.strftime("%Y-%m-%d")
    results = {}

    # an optional lean: a tool pitch below 1.57 tips the rod's top back towards the robot
    start, goal = start_pose(), goal_pose()
    if settings["tool_pitch_rad"] is not None:
        start[4] = goal[4] = settings["tool_pitch_rad"]
    name = settings["picture_name"]

    robot = NiryoRobot(settings["ip_address"])
    try:
        robot.set_arm_max_velocity(settings["velocity_percent"])
        robot.move(JointsPosition(*settings["home_joints"]))
        results["start"] = visit(robot, start, settings["approach_height_m"], folder / f"{day}_start_view{name}.jpg")
        print("start:", results["start"])
        results["goal"] = visit(robot, goal, settings["approach_height_m"], folder / f"{day}_goal_view{name}.jpg")
        print("goal: ", results["goal"])
        robot.move(JointsPosition(*settings["home_joints"]))
    finally:
        robot.close_connection()
    return results


if __name__ == "__main__":
    check_start_view()
