"""
Measure what the wrist camera can see when the arm's own joints are used to aim it.

Three tests, chosen by name on the command line, all without touching the hardware:

``intrinsics``   reads the camera calibration stored on the robot and prints the field of
                 view it implies (no motion).
``wrist_tilt``   goes to the start pose lifted clear of the paper, then changes ONLY the
                 wrist joint (joint 5) by a few angles and saves a picture at each; if the
                 picture tilts with the joint, the camera sits on the wrist link.
``lean_sweep``   at the start pose, leans the rod by a list of angles (wrist joint, tip put
                 back on the start point each time) and, at each lean, asks for the puck to
                 be placed at several spots around the rod before taking a picture; this
                 measures in which directions, and how far, the line stays visible.
``forearm_roll`` goes to the one spot on the line where the forearm stands vertical, then
                 turns only the forearm joint (joint 4) and saves a picture at each angle,
                 printing how far the rod tilts; the puck should lie beside the rod.

Pictures and a JSON log go to 03_data/raw/workspace/camera_tests/. Clear the area, keep
the stop button in reach, and keep hands away while the arm moves:
    ..\\.venv\\Scripts\\python.exe camera_view_tests.py lean_sweep
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy
from pyniryo import JointsPosition, NiryoRobot, PoseObject

from config import ROBOT_PARAMETERS, WORKSPACE_PARAMETERS, pushing_height, start_pose

TEST_PARAMETERS = {
    "ip_address": ROBOT_PARAMETERS["ip_address"],
    "velocity_percent": ROBOT_PARAMETERS["arm_max_velocity_percent"],
    "home_joints": [0.0, 0.5, -1.25, 0.0, 0.0, 0.0],
    "approach_height_m": 0.05,
    "settle_seconds": 0.7,
    "wrist_tilt_lift_m": 0.03,
    "wrist_tilt_angles_deg": [-15, 0, 15],
    "lean_angles_deg": [0, 10, 20, 30],
    "lean_puck_spots": ["ahead", "left", "behind"],
    "lean_max_forearm_twist_deg": 5.0,
    "forearm_roll_x_m": 0.20,
    "forearm_roll_lift_m": 0.03,
    "forearm_roll_angles_deg": [-90, -45, 0, 45, 90],
    "picture_folder": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "workspace" / "camera_tests"
}


def save_picture(
        robot: NiryoRobot,
        path: Path,
        settle_seconds: float
) -> None:
    """Wait for the arm to settle, drain the stream, then save the newest picture."""

    time.sleep(settle_seconds)
    for _ in range(3):
        data = robot.get_img_compressed()
    frame = cv2.imdecode(numpy.frombuffer(data, numpy.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(path), frame)


def read_state(robot: NiryoRobot) -> Dict[str, object]:
    """The tip pose and the joint angles, in degrees, as plain numbers for the log."""

    pose = robot.get_pose()
    joints = [round(math.degrees(angle), 2) for angle in robot.get_joints()]
    return {
        "x": round(pose.x, 4), "y": round(pose.y, 4), "z": round(pose.z, 4),
        "roll_deg": round(math.degrees(pose.roll), 1), "pitch_deg": round(math.degrees(pose.pitch), 1),
        "yaw_deg": round(math.degrees(pose.yaw), 1), "joints_deg": joints
    }


def go_to_start(
        robot: NiryoRobot,
        settings: dict,
        lift_m: float = 0.0
) -> List[float]:
    """From home, come down onto the start pose (raised by ``lift_m``) and return the joints."""

    x, y, z, roll, pitch, yaw = start_pose()
    robot.move(JointsPosition(*settings["home_joints"]))
    robot.move(PoseObject(x, y, z + settings["approach_height_m"], roll, pitch, yaw))
    robot.move(PoseObject(x, y, z + lift_m, roll, pitch, yaw), linear = True)
    return list(robot.get_joints())


def camera_intrinsics(robot: NiryoRobot) -> Dict[str, float]:
    """The stored camera matrix and the horizontal and vertical field of view it implies."""

    matrix, distortion = robot.get_camera_intrinsics()
    matrix = numpy.asarray(matrix, dtype = float).reshape(3, 3)
    focal_x, focal_y = matrix[0, 0], matrix[1, 1]
    width, height = 640, 480
    result = {
        "focal_x_px": round(focal_x, 1), "focal_y_px": round(focal_y, 1),
        "centre_x_px": round(matrix[0, 2], 1), "centre_y_px": round(matrix[1, 2], 1),
        "horizontal_fov_deg": round(math.degrees(2 * math.atan(width / (2 * focal_x))), 1),
        "vertical_fov_deg": round(math.degrees(2 * math.atan(height / (2 * focal_y))), 1),
        "distortion": [round(float(value), 4) for value in numpy.ravel(distortion)]
    }
    print(json.dumps(result, indent = 2))
    return result


def wrist_tilt(
        robot: NiryoRobot,
        settings: dict,
        log: list
) -> None:
    """Change only joint 5 from the lifted start pose and photograph each angle."""

    joints = go_to_start(robot, settings, lift_m = settings["wrist_tilt_lift_m"])
    base_wrist = joints[4]
    for angle_deg in settings["wrist_tilt_angles_deg"]:
        joints[4] = base_wrist + math.radians(angle_deg)
        robot.move(JointsPosition(*joints))
        path = settings["picture_folder"] / f"{settings['day']}_wrist_tilt_{angle_deg:+03d}.jpg"
        save_picture(robot, path, settings["settle_seconds"])
        state = read_state(robot)
        log.append({"test": "wrist_tilt", "joint_5_change_deg": angle_deg, "picture": path.name, **state})
        print(f"joint 5 {angle_deg:+3d} deg: pitch {state['pitch_deg']:6.1f}, tip z {state['z']:.3f}; saved {path.name}")
    joints[4] = base_wrist
    robot.move(JointsPosition(*joints))


def lean_sweep(
        robot: NiryoRobot,
        settings: dict,
        log: list
) -> None:
    """
    Lean the rod by the wrist joint, put the tip back on the start point, photograph the
    puck at several spots around the rod.

    The lean is made with joint 5 alone, so it stays in the arm's plane and the forearm
    twist (joint 4) stays where it is; the tip is then returned to the start point with
    the orientation read back from the robot, which keeps the twist from changing.
    """

    x, y, z, roll, pitch, yaw = start_pose()
    above = PoseObject(x, y, z + settings["approach_height_m"], roll, pitch, yaw)
    robot.move(JointsPosition(*settings["home_joints"]))
    for lean_deg in settings["lean_angles_deg"]:
        # lean with the tip well above the paper, so the swing of the rod's end touches nothing
        robot.move(above)
        joints = list(robot.get_joints())
        joints[4] = joints[4] + math.radians(lean_deg)
        robot.move(JointsPosition(*joints))
        leaned = robot.get_pose()
        robot.move(PoseObject(x, y, z, leaned.roll, leaned.pitch, leaned.yaw), linear = True)
        state = read_state(robot)
        twist_deg = state["joints_deg"][3]
        print(f"lean {lean_deg:2d} deg: pitch {state['pitch_deg']:6.1f}, forearm twist {twist_deg:6.1f} deg, "
              f"tip ({state['x']:.3f}, {state['y']:.3f}, {state['z']:.3f})")
        if abs(twist_deg) > settings["lean_max_forearm_twist_deg"]:
            print("   forearm twist too large for a clean comparison; skipping this lean")
            continue
        for spot in settings["lean_puck_spots"]:
            input(f"   Place the puck {spot.upper()} of the rod, touching it, take your hands away and press Enter: ")
            path = settings["picture_folder"] / f"{settings['day']}_lean_{lean_deg:02d}_{spot}.jpg"
            save_picture(robot, path, settings["settle_seconds"])
            log.append({"test": "lean_sweep", "lean_deg": lean_deg, "puck_spot": spot, "picture": path.name, **state})
            print(f"   saved {path.name}")
        input("   Take the puck away, keep hands clear and press Enter for the next lean: ")
        robot.move(PoseObject(x, y, z + settings["approach_height_m"], leaned.roll, leaned.pitch, leaned.yaw), linear = True)


def forearm_roll(
        robot: NiryoRobot,
        settings: dict,
        log: list
) -> None:
    """Turn only joint 4 above the spot where the forearm is vertical and photograph each angle."""

    _, _, _, roll, pitch, yaw = start_pose()
    x = settings["forearm_roll_x_m"]
    y = WORKSPACE_PARAMETERS["line_start_xy"][1]
    z = pushing_height(x, y) + settings["forearm_roll_lift_m"]
    robot.move(JointsPosition(*settings["home_joints"]))
    robot.move(PoseObject(x, y, z + settings["approach_height_m"], roll, pitch, yaw))
    robot.move(PoseObject(x, y, z, roll, pitch, yaw), linear = True)
    joints = list(robot.get_joints())
    print("joints at the vertical-forearm spot (deg):", [round(math.degrees(j), 1) for j in joints])
    base_roll = joints[3]
    for angle_deg in settings["forearm_roll_angles_deg"]:
        joints[3] = base_roll + math.radians(angle_deg)
        robot.move(JointsPosition(*joints))
        path = settings["picture_folder"] / f"{settings['day']}_forearm_roll_{angle_deg:+04d}.jpg"
        save_picture(robot, path, settings["settle_seconds"])
        state = read_state(robot)
        tilt_deg = abs(90.0 - abs(state["pitch_deg"]))
        log.append({"test": "forearm_roll", "joint_4_change_deg": angle_deg, "rod_tilt_deg": round(tilt_deg, 1),
                    "picture": path.name, **state})
        print(f"joint 4 {angle_deg:+4d} deg: rod tilt about {tilt_deg:4.1f} deg, tip ({state['x']:.3f}, {state['y']:.3f}, "
              f"{state['z']:.3f}); saved {path.name}")
    joints[3] = base_roll
    robot.move(JointsPosition(*joints))


def run_test(
        test_name: str,
        parameters: dict = None
) -> List[dict]:
    """
    Run one named test and return its log entries.

    Parameters
    ----------
    test_name : str
        One of ``intrinsics``, ``wrist_tilt``, ``lean_sweep``, ``forearm_roll``.
    parameters : dict, optional
        Overrides merged over ``TEST_PARAMETERS``.
    """

    settings = dict(TEST_PARAMETERS)
    settings.update(parameters or {})
    settings["day"] = time.strftime("%Y-%m-%d")
    settings["picture_folder"].mkdir(parents = True, exist_ok = True)
    tests = {"wrist_tilt": wrist_tilt, "lean_sweep": lean_sweep, "forearm_roll": forearm_roll}
    log = []

    robot = NiryoRobot(settings["ip_address"])
    try:
        robot.set_arm_max_velocity(settings["velocity_percent"])
        if test_name == "intrinsics":
            log.append({"test": "intrinsics", **camera_intrinsics(robot)})
        else:
            tests[test_name](robot, settings, log)
            robot.move(JointsPosition(*settings["home_joints"]))
    finally:
        robot.close_connection()

    log_path = settings["picture_folder"] / f"{settings['day']}_{test_name}_log.json"
    existing = json.loads(log_path.read_text(encoding = "utf-8")) if log_path.exists() else []
    log_path.write_text(json.dumps(existing + log, indent = 2), encoding = "utf-8")
    print("log written to", log_path)
    return log


if __name__ == "__main__":
    run_test(sys.argv[1] if len(sys.argv) > 1 else "intrinsics")
