"""
Push the puck along the straight line with fixed steps, no learning involved.

A mechanics test for phase 2: the robot comes down behind the puck at the start
pose, then steps along the push direction in fixed increments, saving a picture
and the time per step, until the rod reaches the goal pose, the step budget runs
out, or the next step would leave the workspace box. The puck must sit at the
start of the line. Speed is capped; clear the area and keep the stop button in
reach:
    ..\\.venv\\Scripts\\python.exe push_straight.py
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy
from pyniryo import JointsPosition, NiryoRobot, PoseObject

from config import ROBOT_PARAMETERS, SHEET_PARAMETERS, WORKSPACE_PARAMETERS, goal_pose, pushing_height, start_pose

PUSH_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "home_joints": [0.0, 0.5, -1.25, 0.0, 0.0, 0.0],
    "step_m": 0.01,
    "max_steps": SHEET_PARAMETERS["step_budget"],
    "approach_height_m": 0.05,
    "tool_pitch_rad": WORKSPACE_PARAMETERS["tool_pitch_rad"],
    "result_folder": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "workspace" / "push_straight"
}


def inside_box(
        x: float,
        y: float
) -> bool:
    """True when (x, y) lies inside the workspace box from config."""

    box_x, box_y = WORKSPACE_PARAMETERS["box_x"], WORKSPACE_PARAMETERS["box_y"]
    return box_x[0] <= x <= box_x[1] and box_y[0] <= y <= box_y[1]


def save_picture(
        robot: NiryoRobot,
        path: Path
) -> None:
    """Save what the camera sees now."""

    frame = cv2.imdecode(numpy.frombuffer(robot.get_img_compressed(), numpy.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(path), frame)


def push_straight(parameters: dict = None) -> List[Dict[str, float]]:
    """
    Run the scripted push and return one record per step.

    Parameters
    ----------
    parameters : dict, optional
        Overrides merged over ``PUSH_PARAMETERS``.
    """

    settings = dict(PUSH_PARAMETERS)
    settings.update(parameters or {})
    folder = settings["result_folder"]
    folder.mkdir(parents = True, exist_ok = True)
    stamp = time.strftime("%Y-%m-%d_%H%M")
    direction_x, direction_y = WORKSPACE_PARAMETERS["push_direction_xy"]
    pitch = settings["tool_pitch_rad"]
    x, y, z, roll, _, yaw = start_pose()
    goal_x = goal_pose()[0]
    records = []

    robot = NiryoRobot(settings["ip_address"])
    try:
        robot.set_arm_max_velocity(settings["velocity_percent"])
        robot.move(JointsPosition(*settings["home_joints"]))
        robot.move(PoseObject(x, y, z + settings["approach_height_m"], roll, pitch, yaw))
        robot.move(PoseObject(x, y, z, roll, pitch, yaw), linear = True)
        save_picture(robot, folder / f"{stamp}_step_00.jpg")

        for step in range(1, settings["max_steps"] + 1):
            next_x = x + settings["step_m"] * direction_x
            next_y = y + settings["step_m"] * direction_y
            if next_x > goal_x + 1e-6:
                print("goal reached: the rod is where it stands when the puck is at the end of the line")
                break
            if not inside_box(next_x, next_y):
                print("stopped: the next step would leave the workspace box")
                break
            started = time.perf_counter()
            robot.move(PoseObject(next_x, next_y, pushing_height(next_x, next_y), roll, pitch, yaw), linear = True)
            seconds = time.perf_counter() - started
            x, y = next_x, next_y
            save_picture(robot, folder / f"{stamp}_step_{step:02d}.jpg")
            landed = robot.get_pose()
            records.append({"step": step, "x": round(landed.x, 4), "y": round(landed.y, 4), "z": round(landed.z, 4), "seconds": round(seconds, 2)})
            print(f"step {step:2d}: x = {landed.x:.3f} m, {seconds:.2f} s")

        robot.move(PoseObject(x, y, z + settings["approach_height_m"], roll, pitch, yaw), linear = True)
        robot.move(JointsPosition(*settings["home_joints"]))
    finally:
        robot.close_connection()

    (folder / f"{stamp}_steps.json").write_text(json.dumps(records, indent = 2), encoding = "utf-8")
    if records:
        print(f"{len(records)} steps, median {sorted(r['seconds'] for r in records)[len(records) // 2]:.2f} s per step")
    return records


if __name__ == "__main__":
    push_straight()
