"""
Check with a ruler that the robot puts the rod where it says, high above the sheet, before any work near the paper.

Written 4.11.26 evening: after a calibration, a taught start read 8.4 cm from the base axis with the rod leaning 21
degrees while the person held it upright at about 15 cm, which fits the elbow (joint 3) reading about 21 degrees off.
The arm goes to its parking pose, then to "check_point_xy" with the rod upright and the magnet tip "check_height_m"
above the paper, and waits. The person measures along the table from the centre of the base (the axis the whole arm
turns about) to the point under the tip, and the tip's height above the paper; Enter sends the arm back to its parking
pose. Run from 02_code, nothing else connected, nobody near the arm while it moves, stop button in reach:
    ..\\.venv\\Scripts\\python.exe position_check.py
"""

import math

from config import PUSHER_PARAMETERS, RECORDING_PARAMETERS, WORKSPACE_PARAMETERS, pushing_height
from robot_link import RobotLink

POSITION_CHECK_PARAMETERS = {
    "check_point_xy": [0.20, 0.0],
    "check_height_m": 0.10,
    "home_joints": RECORDING_PARAMETERS["home_joints"]
}


def position_check(parameters: dict = None) -> None:
    """Move to the check point, wait for the measurement, go back to the parking pose."""

    settings = dict(POSITION_CHECK_PARAMETERS)
    if parameters:
        settings.update(parameters)
    x, y = settings["check_point_xy"]
    rod_end_z = pushing_height(x, y) + settings["check_height_m"]

    robot_link = RobotLink()
    try:
        robot_link.move_home(settings["home_joints"])
        robot_link.move_to([x, y, rod_end_z, 0.0, WORKSPACE_PARAMETERS["tool_pitch_rad"], 0.0], is_linear = False)
        state = robot_link.read_state()
        print(f"the robot says: rod end at x {100 * state['pose'][0]:.1f} cm, y {100 * state['pose'][1]:.1f} cm; joints "
              f"{[round(value, 3) for value in state['joints']]}")
        print(f"EXPECTED by ruler: the point under the magnet tip {100 * math.hypot(x, y):.1f} cm straight out from the centre of "
              f"the base, the tip about {100 * settings['check_height_m']:.0f} cm above the paper "
              f"(block {100 * PUSHER_PARAMETERS['extension_below_rod_end_m']:.1f} cm below the rod's end)")
        input("Measure both, then press Enter here to send the arm back to its parking pose. ")
        robot_link.move_home(settings["home_joints"])
    finally:
        robot_link.close()


if __name__ == "__main__":
    position_check()
