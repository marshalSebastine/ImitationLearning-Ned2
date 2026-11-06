"""
Find how close to the base the rod may stand with bracket F, using the robot's inverse kinematics only. Nothing moves.

For rod positions along a line away from the base, at pushing height with the rod vertical, the robot is asked which
joint angles it would use; the wrist angle (joint 5) is printed, and the first position whose wrist angle is below
``wrist_pitch_max_rad`` is the nearest usable one. Written 2.11.26 after the bracket pressed on the forearm at the
near pose (wrist about 0 rad). Run from 02_code, nothing else connected:
    ..\\.venv\\Scripts\\python.exe find_wrist_limit.py
"""

from typing import List

import numpy
from pyniryo import NiryoRobot, PoseObject

from config import RECORDING_PARAMETERS, ROBOT_PARAMETERS, WORKSPACE_PARAMETERS, pushing_height

LIMIT_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "x_values_m": list(numpy.round(numpy.arange(0.13, 0.43, 0.01), 3)),
    "y_m": 0.0,
    "heights_above_pushing_m": [0.0, 0.05]
}


def find_wrist_limit(parameters: dict = None) -> List[dict]:
    """Print the wrist angle the robot would use at each x and return the rows."""

    settings = dict(LIMIT_PARAMETERS)
    settings.update(parameters or {})
    limit = RECORDING_PARAMETERS["wrist_pitch_max_rad"]
    pitch = WORKSPACE_PARAMETERS["tool_pitch_rad"]
    rows = []

    robot = NiryoRobot(settings["ip_address"])
    try:
        for height in settings["heights_above_pushing_m"]:
            print(f"\nrod vertical, {100 * height:.0f} cm above pushing height, y = {settings['y_m']} m (limit {limit} rad):")
            for x in settings["x_values_m"]:
                z = pushing_height(x, settings["y_m"]) + height
                try:
                    joints = robot.inverse_kinematics(PoseObject(x, settings["y_m"], z, 0.0, pitch, 0.0))
                    wrist = joints[4]
                    verdict = "ok" if wrist <= limit else "WRIST TOO HIGH"
                except Exception as error:
                    wrist, verdict = float("nan"), f"no solution ({error})"
                rows.append({"x_m": float(x), "height_m": height, "wrist_rad": wrist, "verdict": verdict})
                print(f"  x = {x:.2f} m: wrist {wrist:+.3f} rad  {verdict}")
    finally:
        robot.close_connection()

    return rows


if __name__ == "__main__":
    find_wrist_limit()
