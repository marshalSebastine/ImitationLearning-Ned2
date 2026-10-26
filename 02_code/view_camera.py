"""
Show the forearm camera live in a window, so the lens can be focused and the view checked.

Reads the camera through the robot (config "source": "robot") or from the laptop's USB
("usb"). Press q in the window to quit, s to save the current frame to
03_data/raw/camera_check/. Nothing moves. Run from this folder:
    ..\\.venv\\Scripts\\python.exe view_camera.py
"""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy
from pyniryo import NiryoRobot

from config import CAMERA_PARAMETERS, ROBOT_PARAMETERS

FRAME_FOLDER = Path(__file__).resolve().parent.parent / "03_data" / "raw" / "camera_check"


def decode_robot_frame(compressed: bytes) -> numpy.ndarray:
    """Turn the JPEG bytes the robot sends into an image; pyniryo's own decoder breaks on numpy 2."""

    return cv2.imdecode(numpy.frombuffer(compressed, numpy.uint8), cv2.IMREAD_COLOR)


def sharpness_of(frame: numpy.ndarray) -> float:
    """Variance of the Laplacian: higher means sharper, which helps focusing by hand."""

    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def show_frames(read_frame: Callable[[], Optional[numpy.ndarray]]) -> None:
    """Loop: read a frame, draw the sharpness number, show it, handle the q and s keys."""

    saved_count = 0
    while True:
        frame = read_frame()
        if frame is None:
            print("No frame from the camera. Is it plugged in?")
            break

        cv2.putText(frame, f"sharpness {sharpness_of(frame):.0f}   q quit   s save", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("robot camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            FRAME_FOLDER.mkdir(parents = True, exist_ok = True)
            saved_count += 1
            path = FRAME_FOLDER / f"view_{saved_count:02d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")
    cv2.destroyAllWindows()


def view_camera(parameters: dict = CAMERA_PARAMETERS) -> None:
    """
    Open the camera named in ``parameters`` and show it until q is pressed.

    Parameters
    ----------
    parameters : dict
        ``source`` is "robot" (through the robot) or "usb" (``usb_index`` on the laptop).
    """

    if parameters["source"] == "robot":
        robot = NiryoRobot(ROBOT_PARAMETERS["ip_address"])
        try:
            show_frames(lambda: decode_robot_frame(robot.get_img_compressed()))
        finally:
            robot.close_connection()
    else:
        capture = cv2.VideoCapture(parameters["usb_index"], cv2.CAP_DSHOW)
        try:
            show_frames(lambda: capture.read()[1] if capture.read()[0] else None)
        finally:
            capture.release()


if __name__ == "__main__":
    view_camera()
