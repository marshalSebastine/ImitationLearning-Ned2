"""
Find the wrist turn at which the camera sees the puck and the line ahead of it.

The camera sits beside the wrist, so which way it looks depends on how the wrist
joint (joint 6) is turned about the rod. Starting from where the arm is, with the rod
already pointing down above the start of the line, this script turns only joint 6 to
each angle in the list and saves a picture at each, named by the angle. The rod's end
does not move. Clear the area and keep the stop button in reach:
    ..\\.venv\\Scripts\\python.exe check_camera_view.py
"""

import math
from pathlib import Path
from typing import List

import cv2
import numpy
from pyniryo import JointsPosition, NiryoRobot

from config import ROBOT_PARAMETERS

VIEW_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "wrist_angles_deg": [60, 0, -60, -120],
    "picture_folder": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "workspace"
}


def save_picture(
        robot: NiryoRobot,
        path: Path
) -> None:
    """Save what the camera sees now."""

    frame = cv2.imdecode(numpy.frombuffer(robot.get_img_compressed(), numpy.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(path), frame)


def check_camera_view(parameters: dict = None) -> List[Path]:
    """
    Take one picture per wrist angle, turning only joint 6, and return the file paths.

    Parameters
    ----------
    parameters : dict, optional
        Overrides merged over ``VIEW_PARAMETERS``.
    """

    settings = dict(VIEW_PARAMETERS)
    settings.update(parameters or {})
    folder = settings["picture_folder"]
    folder.mkdir(parents = True, exist_ok = True)
    saved = []

    robot = NiryoRobot(settings["ip_address"])
    try:
        robot.set_arm_max_velocity(settings["velocity_percent"])
        joints = list(robot.get_joints())
        print("starting joints (deg):", [round(math.degrees(j)) for j in joints])
        for angle_deg in settings["wrist_angles_deg"]:
            joints[5] = math.radians(angle_deg)
            robot.move(JointsPosition(*joints))
            pose = robot.get_pose()
            path = folder / f"view_wrist_{angle_deg:+04d}.jpg"
            save_picture(robot, path)
            saved.append(path)
            print(f"wrist {angle_deg:+4d} deg: tip x = {pose.x:.3f}, y = {pose.y:.3f}, z = {pose.z:.3f}; saved {path.name}")
    finally:
        robot.close_connection()
    return saved


if __name__ == "__main__":
    check_camera_view()
