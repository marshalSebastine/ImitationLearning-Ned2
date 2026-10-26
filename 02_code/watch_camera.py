"""
Save the newest camera picture to one file, over and over, so it can be looked at from outside.

A stand-in for the live viewer when someone else needs to see the picture: every
``interval_seconds`` the latest picture is written to ``latest_path`` with its sharpness
in the file name's neighbour text file. Nothing moves. Stop it with Ctrl+C or by ending
the process:
    ..\\.venv\\Scripts\\python.exe watch_camera.py
"""

import time
from pathlib import Path

import cv2
import numpy
from pyniryo import NiryoRobot

from config import ROBOT_PARAMETERS

WATCH_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "interval_seconds": 1.0,
    "latest_path": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "camera_check" / "latest.jpg"
}


def watch_camera(parameters: dict = None) -> None:
    """Loop forever: read a picture, save it as ``latest_path``, note its sharpness beside it."""

    settings = dict(WATCH_PARAMETERS)
    settings.update(parameters or {})
    latest, note = settings["latest_path"], settings["latest_path"].with_suffix(".txt")

    robot = NiryoRobot(settings["ip_address"])
    try:
        while True:
            frame = cv2.imdecode(numpy.frombuffer(robot.get_img_compressed(), numpy.uint8), cv2.IMREAD_COLOR)
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(latest), frame)
            note.write_text(f"{time.strftime('%H:%M:%S')} sharpness {cv2.Laplacian(grey, cv2.CV_64F).var():.0f} brightness {grey.mean():.0f}\n")
            time.sleep(settings["interval_seconds"])
    finally:
        robot.close_connection()


if __name__ == "__main__":
    watch_camera()
