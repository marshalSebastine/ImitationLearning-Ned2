"""
Save one picture from the robot's camera under a dated name and print its sharpness. Nothing moves.

For checking a camera mount or the focus without a window: the picture goes to
03_data/raw/camera_check/<date>_<label>.jpg. Run from this folder, nothing else connected to the robot:
    ..\\.venv\\Scripts\\python.exe grab_frame.py <label>
"""

import sys
import time
from pathlib import Path

import cv2
import numpy
from pyniryo import NiryoRobot

from config import ROBOT_PARAMETERS

FRAME_FOLDER = Path(__file__).resolve().parent.parent / "03_data" / "raw" / "camera_check"


def sharpness_of(picture: numpy.ndarray) -> float:
    """Variance of the Laplacian, the number the other camera scripts show: higher is sharper."""

    return float(cv2.Laplacian(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def grab_frame(label: str = "frame") -> Path:
    """Read a fresh picture (the second new one after connecting), save it and return its path."""

    robot = NiryoRobot(ROBOT_PARAMETERS["ip_address"])
    try:
        previous = bytes(robot.get_img_compressed())
        new_pictures = 0
        while new_pictures < 2:
            time.sleep(0.02)
            picture_bytes = bytes(robot.get_img_compressed())
            if picture_bytes != previous:
                new_pictures += 1
                previous = picture_bytes
    finally:
        robot.close_connection()

    FRAME_FOLDER.mkdir(parents = True, exist_ok = True)
    path = FRAME_FOLDER / f"{time.strftime('%Y-%m-%d_%H%M')}_{label}.jpg"
    path.write_bytes(previous)
    picture = cv2.imdecode(numpy.frombuffer(previous, numpy.uint8), cv2.IMREAD_COLOR)
    print(f"saved {path}  sharpness {sharpness_of(picture):.0f}")
    return path


if __name__ == "__main__":
    grab_frame(sys.argv[1] if len(sys.argv) > 1 else "frame")
