"""
Read where the tool tip is each time Enter is pressed, with the arm in learning mode.

Used once in phase 2, step 1. One person holds the arm at the elbow; this script
switches the motors off, and for each named spot the rod's tip is touched to the paper
by hand, in whatever arm pose is comfortable, and Enter is pressed. Only the tip's
position matters: the start pose, pushing height and orientation are computed from
these points later. The positions are printed and written to
03_data/raw/workspace/<date>_positions.json. Nothing moves on its own. The motors come
back on at the end, so keep holding the arm until the script says they are on. Run
from this folder in a terminal, because it waits for Enter:
    ..\\.venv\\Scripts\\python.exe teach_positions.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy
from pyniryo import NiryoRobot

from config import ROBOT_PARAMETERS

TEACH_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "spots": [
        ("line_start", "the rod's tip touching the paper exactly where the black line BEGINS (near end)"),
        ("line_end", "the rod's tip touching the paper exactly where the black line ENDS (far end)"),
        ("corner_near_left", "the rod's tip on the sheet corner nearest the robot, on the left"),
        ("corner_near_right", "the rod's tip on the sheet corner nearest the robot, on the right"),
        ("corner_far_right", "the rod's tip on the far right corner"),
        ("corner_far_left", "the rod's tip on the far left corner")
    ],
    "result_folder": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "workspace"
}


def read_position(robot: NiryoRobot) -> Dict[str, List[float]]:
    """The tip pose (metres and radians) and the joint angles, as the robot reports them."""

    pose = robot.get_pose()
    return {
        "pose": [round(value, 4) for value in (pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)],
        "joints": [round(value, 4) for value in robot.get_joints()]
    }


def save_picture(
        robot: NiryoRobot,
        path: Path
) -> None:
    """Save what the camera sees at this moment, so the view can be judged afterwards."""

    frame = cv2.imdecode(numpy.frombuffer(robot.get_img_compressed(), numpy.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(path), frame)


def teach_positions(parameters: dict = None) -> Dict[str, Dict[str, List[float]]]:
    """
    Walk through every spot, reading the position after each Enter.

    Parameters
    ----------
    parameters : dict, optional
        Overrides merged over ``TEACH_PARAMETERS``.
    """

    settings = dict(TEACH_PARAMETERS)
    settings.update(parameters or {})
    positions = {}
    folder = settings["result_folder"]
    folder.mkdir(parents = True, exist_ok = True)
    day = time.strftime("%Y-%m-%d")

    robot = NiryoRobot(settings["ip_address"])
    try:
        input("Hold the arm at the elbow. Press Enter to switch the motors OFF (the arm will sag). ")
        robot.set_learning_mode(True)
        print("Motors are off. Guide the arm by hand.\n")

        for name, description in settings["spots"]:
            input(f"Place {description}, then press Enter. ")
            positions[name] = read_position(robot)
            save_picture(robot, folder / f"{day}_{name}.jpg")
            x, y, z = positions[name]["pose"][:3]
            print(f"  {name}: x = {x:.3f} m, y = {y:.3f} m, z = {z:.3f} m  (picture saved)\n")

        input("Keep holding the arm. Press Enter to switch the motors back ON. ")
        robot.set_learning_mode(False)
        print("Motors are on; the arm holds its position. You can let go.")
    finally:
        robot.close_connection()

    path = folder / f"{day}_positions.json"
    path.write_text(json.dumps(positions, indent = 2), encoding = "utf-8")
    print(f"\nSaved to {path}")
    return positions


if __name__ == "__main__":
    teach_positions()
