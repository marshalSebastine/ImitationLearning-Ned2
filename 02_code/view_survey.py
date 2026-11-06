"""
Drive the rod to a few known points over the sheet at pushing height and save the camera's picture at each one.

Written 4.11.26 to compare camera placements with the same arm poses: first with the camera in bracket F3 on the
wrist, then taped to the upper arm. The pictures go to 03_data/raw/camera_check/<date>_<label>_<point>.jpg with the
sharpness number printed, and a JSON file next to them records the poses. Put the puck on the first point if you
want it in the pictures. Run from 02_code, nothing else connected, stop button in reach:
    ..\\.venv\\Scripts\\python.exe view_survey.py <label>
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy

from config import RECORDING_PARAMETERS, WORKSPACE_PARAMETERS, pushing_height
from robot_link import RobotLink

SURVEY_PARAMETERS = {
    # Rod points in metres: sheet 04's start of 2.11 and 3.11, the refocus parking point, and a far point.
    "points_xy": {"near_17cm": [0.1707, 0.0374], "mid_26cm": [0.26, 0.0], "far_35cm": [0.35, 0.0]},
    "height_above_pushing_m": 0.01,
    "approach_height_m": RECORDING_PARAMETERS["approach_height_m"],
    "home_joints": RECORDING_PARAMETERS["home_joints"],
    "frame_folder": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "camera_check"
}


def sharpness_of(picture: numpy.ndarray) -> float:
    """Variance of the Laplacian, the number the other camera scripts show: higher is sharper."""

    return float(cv2.Laplacian(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def fresh_picture_bytes(robot_link: RobotLink) -> bytes:
    """The second new JPEG after now, so the picture shows the arm where it is and not where it was."""

    previous = robot_link.read_picture_bytes()
    new_pictures = 0
    while new_pictures < 2:
        time.sleep(0.02)
        picture_bytes = robot_link.read_picture_bytes()
        if picture_bytes != previous:
            new_pictures += 1
            previous = picture_bytes
    return previous


def view_survey(
        label: str,
        parameters: dict = None
) -> List[Dict]:
    """Visit every point, save a picture at each and return what was recorded."""

    settings = dict(SURVEY_PARAMETERS)
    if parameters:
        settings.update(parameters)
    folder = settings["frame_folder"]
    folder.mkdir(parents = True, exist_ok = True)
    stamp = time.strftime("%Y-%m-%d_%H%M")
    records = []

    robot_link = RobotLink()
    try:
        robot_link.move_home(settings["home_joints"])
        for point_name, (x, y) in settings["points_xy"].items():
            pose = [x, y, pushing_height(x, y) + settings["height_above_pushing_m"], 0.0, WORKSPACE_PARAMETERS["tool_pitch_rad"], 0.0]
            above = list(pose)
            above[2] += settings["approach_height_m"]
            try:
                robot_link.move_to(above, is_linear = False)
                robot_link.move_to(pose, is_linear = False)
            except Exception as error:
                print(f"{point_name}: not reached ({str(error).splitlines()[0]})")
                continue

            picture_bytes = fresh_picture_bytes(robot_link)
            path = folder / f"{stamp}_{label}_{point_name}.jpg"
            path.write_bytes(picture_bytes)
            picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
            state = robot_link.read_state()
            record = {"point": point_name, "picture": path.name, "sharpness": round(sharpness_of(picture)),
                      "pose": [round(value, 4) for value in state["pose"]], "joints": [round(value, 3) for value in state["joints"]]}
            records.append(record)
            print(f"{point_name}: saved {path.name}, sharpness {record['sharpness']}, joints {record['joints']}")
            robot_link.move_to(above, is_linear = False)
        robot_link.move_home(settings["home_joints"])
    finally:
        robot_link.close()

    (folder / f"{stamp}_{label}_survey.json").write_text(json.dumps(records, indent = 2), encoding = "utf-8")
    return records


if __name__ == "__main__":
    view_survey(sys.argv[1] if len(sys.argv) > 1 else "survey")
