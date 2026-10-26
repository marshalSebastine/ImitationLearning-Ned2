"""
Find the camera and save one frame from it. Nothing moves.

Two kinds of camera are possible, and this tries both:
- Niryo's own Vision Set, which streams through the robot;
- a USB webcam plugged into the laptop, read with OpenCV.

Run from this folder:
    ..\\.venv\\Scripts\\python.exe check_camera.py
"""

from pathlib import Path

import cv2
import numpy
from pyniryo import NiryoRobot

from config import ROBOT_PARAMETERS

FRAME_FOLDER = Path(__file__).resolve().parent.parent / "03_data" / "raw" / "camera_check"
USB_CAMERA_INDICES = [0, 1, 2]


def save_frame(image, name: str) -> None:
    """Write one frame to the camera-check folder and print its size."""

    FRAME_FOLDER.mkdir(parents = True, exist_ok = True)
    path = FRAME_FOLDER / f"{name}.jpg"
    cv2.imwrite(str(path), image)
    height, width = image.shape[:2]
    print(f"{name}: {width} x {height} pixels, saved to {path}")


def decode_robot_frame(compressed: bytes):
    """Turn the JPEG bytes the robot sends into an image; pyniryo's own decoder breaks on numpy 2."""

    return cv2.imdecode(numpy.frombuffer(compressed, numpy.uint8), cv2.IMREAD_COLOR)


def try_robot_camera(ip_address: str = ROBOT_PARAMETERS["ip_address"]) -> bool:
    """Grab one frame through the robot (Vision Set). Returns True if a frame came back."""

    robot = NiryoRobot(ip_address)
    try:
        save_frame(decode_robot_frame(robot.get_img_compressed()), "robot_camera")
        return True
    except Exception as error:
        print(f"robot camera: none ({error})")
        return False
    finally:
        robot.close_connection()


def try_usb_cameras() -> bool:
    """Grab one frame from each USB camera index that opens. Returns True if any did."""

    found = False
    for index in USB_CAMERA_INDICES:
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        is_open, frame = capture.read()
        capture.release()
        if is_open:
            save_frame(frame, f"usb_camera_{index}")
            found = True
        else:
            print(f"usb camera {index}: none")
    return found


if __name__ == "__main__":
    has_robot_camera = try_robot_camera()
    has_usb_camera = try_usb_cameras()
    if not (has_robot_camera or has_usb_camera):
        print("No camera found. Is one plugged in?")
