"""
Record demonstration episodes: one fresh picture and one 1 cm step at a time.

Every step runs the same loop the trained model will later run: wait until the move has finished, take the first
fresh picture, ask a decider for an action, log the picture with the action and the robot's state, then move the
pusher's tip 1 cm in that direction, measured from the pusher's narrow face (config.action_step_to_table). An episode ends
with an explicit "stop" step.

Two modes:
    keyboard     the decider is a person pressing keys while looking at the picture on screen only;
    hand_guided  a person first guides the arm by hand (motors off) and pushes the puck to the goal while the tool
                 logs the tip's path; the person then puts the puck back, and the robot replays that path in 1 cm
                 steps, snapped back onto the path at every step. Each replay step runs only when Enter is pressed;
                 a direction key overrides it, and the episode can be discarded if the puck does not follow. Only
                 the replay becomes training data.

Keys (RECORDING_PARAMETERS in config): q w e = forward-left, forward, forward-right; a s d = left, back, right; z c = back-left, back-right; r f = step over the puck forward, back;
Space = stop; x = discard the episode; u = the previous step was a mistake (kept in the file, flagged); l = the line
is lost in this picture (the next step is flagged); Enter = accept a proposed step, or go on; Esc = quit. After each
episode the tool asks whether the puck reached the goal, or why the episode was discarded. Keys pressed
while the robot moves are thrown away, so a key always refers to the picture on screen.

Output, one folder per episode, never edited afterwards:
    03_data/raw/<yyyy-mm-dd>_session_NN_<operator>/session_notes.md
    03_data/raw/<yyyy-mm-dd>_session_NN_<operator>/<path_id>_episode_NNN/frames/NNN.jpg, steps.csv, episode.json
    (hand_guided also guided_path.csv)

Run from 02_code, robot connected, nothing else connected to it, stop button in reach:
    python record_episodes.py --operator marshal --mode keyboard
Without the robot (simulated sheet, puck and camera):
    python record_episodes.py --operator marshal --mode keyboard --simulated
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy

from config import (PUCK_PARAMETERS, PUSHER_PARAMETERS, RECORDING_PARAMETERS, ROBOT_PARAMETERS, SHEET_PARAMETERS, WORKSPACE_PARAMETERS,
                    action_step_to_table, camera_heading_rad, pushing_height, step_budget_for)
from puck_detector import detect_puck
from puck_tracking import PuckTracker
from side_view_test import magnet_tip_and_lean, ned2_fk, solve_least_lean

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
WINDOW_NAME = "record_episodes"
ORIENTATION_WINDOW_NAME = "orientation"
# What the orientation window lists under its map in each view (Emranul, 5.11.26: "make the orientation better for
# my comprehension"; in hand guiding not all instruction keys could be seen): the control, what it does, and the
# colour of the control's name (BGR; the gamepad's four face buttons in their own colours).
LEGEND_TEXT_COLOUR = (40, 40, 40)
ORIENTATION_LEGENDS = {
    "keyboard": [
        ("w q e a d s y c", "step 1 cm along the arrows", LEGEND_TEXT_COLOUR),
        ("v / b", "turn the forearm left / right", LEGEND_TEXT_COLOUR),
        ("n / m", "lean the rod left / right", LEGEND_TEXT_COLOUR),
        ("Space", "stop: the episode is done", LEGEND_TEXT_COLOUR),
        ("u", "the last step was a mistake", LEGEND_TEXT_COLOUR),
        ("l", "line lost", LEGEND_TEXT_COLOUR),
        ("x", "discard the episode", LEGEND_TEXT_COLOUR),
        ("Esc", "quit: the robot goes home", LEGEND_TEXT_COLOUR),
    ],
    "gamepad": [
        ("Left stick", "move the tip along the arrows", LEGEND_TEXT_COLOUR),
        ("Right stick < >", "turn the forearm (like v / b)", LEGEND_TEXT_COLOUR),
        ("Right stick up/down", "bend the wrist: the rod leans", LEGEND_TEXT_COLOUR),
        ("L2 / R2", "tip down / up", LEGEND_TEXT_COLOUR),
        ("L1 / R1", "turn the hand left / right", LEGEND_TEXT_COLOUR),
        ("Cross", "stop: the episode is done", (200, 120, 40)),
        ("Circle", "discard the episode", (50, 50, 210)),
        ("Square", "the last step was a mistake", (170, 90, 200)),
        ("Triangle", "line lost", (60, 160, 60)),
        ("OPTIONS", "quit: the robot goes home", LEGEND_TEXT_COLOUR),
        ("Keyboard", "Enter start, y / n goal, Esc end", LEGEND_TEXT_COLOUR),
    ],
    "replay": [
        ("Enter", "do the step the replay proposes", LEGEND_TEXT_COLOUR),
        ("w q e a d s y c", "do that step instead (arrows)", LEGEND_TEXT_COLOUR),
        ("v / b", "turn the forearm instead", LEGEND_TEXT_COLOUR),
        ("Space", "stop: the episode is done", LEGEND_TEXT_COLOUR),
        ("u", "the last step was a mistake", LEGEND_TEXT_COLOUR),
        ("l", "line lost", LEGEND_TEXT_COLOUR),
        ("x", "discard the episode", LEGEND_TEXT_COLOUR),
        ("Esc", "quit: the robot goes home", LEGEND_TEXT_COLOUR),
    ],
    "exact_replay": [
        ("Enter", "go to the next waypoint", LEGEND_TEXT_COLOUR),
        ("g", "run through the rest by itself", LEGEND_TEXT_COLOUR),
        ("any key", "pause that run", LEGEND_TEXT_COLOUR),
        ("h", "correct by hand (motors off)", LEGEND_TEXT_COLOUR),
        ("u", "the last step was a mistake", LEGEND_TEXT_COLOUR),
        ("l", "line lost", LEGEND_TEXT_COLOUR),
        ("x", "discard the episode", LEGEND_TEXT_COLOUR),
        ("Esc", "quit: the robot goes home", LEGEND_TEXT_COLOUR),
    ],
    "guiding": [
        ("Motors off", "push the puck by hand along the line", LEGEND_TEXT_COLOUR),
        ("Space", "puck at the goal: guiding done", LEGEND_TEXT_COLOUR),
        ("x", "discard this guiding", LEGEND_TEXT_COLOUR),
        ("Then", "let go of the arm, Enter = motors on", LEGEND_TEXT_COLOUR),
        ("Then", "put the puck back as asked, Enter", LEGEND_TEXT_COLOUR),
    ],
}
ORIENTATION_TITLES = {"keyboard": "KEYBOARD", "gamepad": "GAMEPAD", "replay": "HAND-GUIDED REPLAY",
                      "exact_replay": "EXACT REPLAY", "guiding": "HAND GUIDING, MOTORS OFF"}
# Three times per step: when the move returned, when the logged picture was read, when the action was chosen.
# is_mistake is set afterwards by the "u" key on the step it refers to; is_lost by the "l" key on the step whose
# picture shows no line. Training drops flagged steps (and the steps that undo a mistake); the episode survives.
STEP_COLUMNS = ["step", "unix_time", "move_return_time", "capture_time", "decision_time", "frame", "frame_md5", "sharpness",
                "action", "source", "is_mistake", "is_lost", "forearm_turn_deg", "picture_heading_deg",
                "waypoint", "tip_forward_m", "tip_left_m", "forearm_change_deg",
                "rod_lean_deg", "lean_change_deg", "lean_label", "hand_change_deg", "hand_label", "forearm_label",
                "goal_distance_cm", "is_at_goal", "pad_move_x", "pad_move_y", "pad_twist", "pad_lean", "pad_hand", "pad_up", "pad_down", "tip_height_above_push_m",
                "x", "y", "z", "roll", "pitch", "yaw",
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def rod_lean_forward_deg(
        joints: List[float],
        heading_rad: float
) -> float:
    """
    The rod's lean towards the picture's top, in degrees from straight down: positive when the tip (lower end) is
    further towards the picture's top than the rod's upper end, negative towards the picture's bottom.
    """

    rod_direction = numpy.array(ned2_fk.fk(list(joints))["joint_6"][1])[:, 2]
    forward = numpy.array([math.cos(heading_rad), math.sin(heading_rad)])
    return math.degrees(math.atan2(float(rod_direction[:2] @ forward), -float(rod_direction[2])))


def lean_and_hand_labels(
        joints_now: List[float],
        joints_next: List[float],
        heading_rad: float,
        parameters: Dict = RECORDING_PARAMETERS
) -> Dict:
    """
    The two labels beside the move name for the step from ``joints_now`` to ``joints_next`` (6.11.26).

    Parameters
    ----------
    joints_now, joints_next : list of float
        The six joints before and after the step, in radians.
    heading_rad : float
        Where the picture's top points on the table at ``joints_now`` (picture_heading_rad).
    parameters : dict, optional
        Supplies "lean_label_deg" and "hand_label_deg".

    Returns
    -------
    dict
        "rod_lean_deg" (the rod's lean towards the picture's top at ``joints_now``), "lean_change_deg" (positive: the
        tip leans further towards the picture's top), "lean_label" ("lean_forward", "lean_back" or "none"),
        "hand_change_deg" (joint 6, positive as the joint counts), "hand_label" ("hand_plus", "hand_minus" or "none";
        the joint's sign, not yet checked against the picture's left and right) and "forearm_label" ("turn_left",
        "turn_right" or "none", with the sign of the "turn_left" move).
    """

    lean_now = rod_lean_forward_deg(joints_now, heading_rad)
    lean_change = rod_lean_forward_deg(joints_next, heading_rad) - lean_now
    hand_change = math.degrees(joints_next[5] - joints_now[5])
    forearm_change = math.degrees(joints_next[3] - joints_now[3])
    lean_label = "none" if abs(lean_change) < parameters["lean_label_deg"] else ("lean_forward" if lean_change > 0 else "lean_back")
    hand_label = "none" if abs(hand_change) < parameters["hand_label_deg"] else ("hand_plus" if hand_change > 0 else "hand_minus")
    forearm_label = "none" if abs(forearm_change) < parameters["forearm_label_deg"] else ("turn_left" if forearm_change > 0 else "turn_right")
    return {"rod_lean_deg": round(lean_now, 2), "lean_change_deg": round(lean_change, 2), "lean_label": lean_label,
            "hand_change_deg": round(hand_change, 2), "hand_label": hand_label, "forearm_label": forearm_label}


def goal_labels(
        rows: List[Dict],
        parameters: Dict = RECORDING_PARAMETERS
) -> None:
    """
    Fill "goal_distance_cm" and "is_at_goal" in the step rows of one episode (7.11.26).

    The goal is the magnet tip's place at the episode's last stop step (the arm model on that row's joints); each row's
    distance is the straight line from its own tip to there. Without a stop step the episode never reached a goal,
    so both columns stay empty.

    Parameters
    ----------
    rows : list of dict
        The episode's step rows with "action" and "joint_1" to "joint_6"; changed in place.
    parameters : dict, optional
        Supplies "at_goal_within_cm".
    """

    def tip_of(row: Dict) -> numpy.ndarray:
        return magnet_tip_and_lean([float(row[f"joint_{number}"]) for number in range(1, 7)])[0][:2]

    stops = [row for row in rows if row["action"] == "stop"]
    if not stops:
        for row in rows:
            row["goal_distance_cm"], row["is_at_goal"] = "", ""
        return
    goal = tip_of(stops[-1])
    for row in rows:
        distance_cm = 100 * float(numpy.linalg.norm(tip_of(row) - goal))
        row["goal_distance_cm"] = round(distance_cm, 2)
        row["is_at_goal"] = int(distance_cm <= parameters["at_goal_within_cm"])


def read_window_key(wait_ms: int) -> Optional[str]:
    """
    Read one key from the recording window.

    Parameters
    ----------
    wait_ms : int
        How long to wait in milliseconds; 0 waits until a key is pressed.

    Returns
    -------
    str or None
        The key as a one-character string, "enter" or "escape", or None when no key came.
    """

    # Wait in half-second slices so the window can be checked between them: three times on 4.11.26 the picture
    # window vanished (hidden or closed) while the tool waited for a key, and the keys then went nowhere.
    deadline = None if wait_ms == 0 else time.monotonic() + wait_ms / 1000.0
    while True:
        slice_ms = 500 if deadline is None else max(1, min(500, int(1000 * (deadline - time.monotonic()))))
        code = cv2.waitKey(slice_ms)
        if code >= 0:
            break
        if deadline is not None and time.monotonic() >= deadline:
            return None
        ensure_window_is_shown()

    code = code & 0xFF
    if code in (10, 13):
        return "enter"
    if code == 27:
        return "escape"
    return chr(code).lower()


def forearm_picture_up_in_forearm_frame() -> numpy.ndarray:
    """
    The forearm camera's picture-up axis in the forearm's own frame (joint 4's frame of the arm model): at a reference
    pose with the tip 25 cm out and the forearm untwisted (it then leans slightly outwards), the part of straight up
    at right angles to the forearm, which points up and away from the base.
    """

    reference_joints = [0.0, -0.809, -0.588, 0.0, -0.173, 0.0]
    rotation = numpy.array(ned2_fk.fk(reference_joints)["joint_4"][1])
    forearm = rotation[:, 2]
    straight_up = numpy.array([0.0, 0.0, 1.0])
    picture_up = straight_up - (straight_up @ forearm) * forearm
    return rotation.T @ (picture_up / numpy.linalg.norm(picture_up))


FOREARM_PICTURE_UP_IN_FOREARM_FRAME = forearm_picture_up_in_forearm_frame()


def forearm_camera_heading_rad(
        joints: List[float],
        turns_with_joint_4: bool = RECORDING_PARAMETERS["forearm_camera_turns_with_joint_4"]
) -> float:
    """
    The table direction the top of the picture points to, for a camera on the forearm cover looking down the forearm.

    The camera is fixed to the forearm, so its picture-up axis is a fixed direction in the forearm's own frame: the
    one that points up and away from the base at a reference pose (tip 25 cm out, forearm untwisted, leaning slightly
    outwards). The first version (4.11.26 evening) worked it out as "the part of straight up at right angles to the
    forearm" at every pose, which flips by 180 degrees when the forearm passes upright: closer than about 21 cm to the
    base the forearm leans back towards the robot, and w went backwards at the start of sheet 04 (Emranul). Without
    the twist (``turns_with_joint_4`` False) the forearm frame is taken untwisted. A table point moves straight up the
    picture when it moves at right angles to the picture's right axis, so the heading is the table direction at right
    angles to that axis, on the picture-up side; with the forearm untwisted it is the direction from the base
    towards the rod.
    """

    if turns_with_joint_4:
        rotation = numpy.array(ned2_fk.fk(list(joints))["joint_4"][1])
    else:
        untwisted = list(joints)
        untwisted[3] = 0.0
        rotation = numpy.array(ned2_fk.fk(untwisted)["joint_4"][1])
    looking = rotation[:, 2]
    picture_up = rotation @ FOREARM_PICTURE_UP_IN_FOREARM_FRAME
    straight_up = numpy.array([0.0, 0.0, 1.0])
    picture_right = numpy.cross(looking, picture_up)
    top = numpy.cross(straight_up, numpy.array([picture_right[0], picture_right[1], 0.0]))
    if numpy.linalg.norm(top) < 1e-9:
        top = picture_up
    if top @ picture_up < -1e-9 or (abs(top @ picture_up) <= 1e-9 and top @ looking < 0.0):
        top = -top
    return math.atan2(top[1], top[0]) + RECORDING_PARAMETERS["picture_top_offset_rad"]


def camera_shift_px(
        reference_bytes: bytes,
        picture_bytes: bytes,
        region: List[float]
) -> Tuple[float, float]:
    """
    How far, in pixels (x right, y down), the content of ``region`` = [top, bottom, left, right] as fractions of the
    picture has moved from the reference picture to this one, by phase correlation on the grey levels.
    """

    def strip(jpeg: bytes) -> numpy.ndarray:
        grey = cv2.cvtColor(cv2.imdecode(numpy.frombuffer(jpeg, numpy.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        height, width = grey.shape
        top, bottom, left, right = region
        return grey[int(top * height):int(bottom * height), int(left * width):int(right * width)].astype(numpy.float32)

    reference, current = strip(reference_bytes), strip(picture_bytes)
    window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
    (shift_x, shift_y), _ = cv2.phaseCorrelate(reference, current, window)
    return float(shift_x), float(shift_y)


def picture_heading_rad(
        state: Dict[str, List[float]],
        recorder_parameters: Dict
) -> float:
    """The table direction the top of the picture points to now, for the configured camera position."""

    if recorder_parameters["camera_heading_model"] == "forearm":
        return forearm_camera_heading_rad(state["joints"], recorder_parameters["forearm_camera_turns_with_joint_4"])
    return camera_heading_rad(state["pose"], "hand")


LAST_SHOWN = {"picture": None}


def picture_axes_on_table(
        state: Dict,
        recorder_parameters: Dict
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    The picture's up and left directions on the table: where the gamepad's left stick moves the tip when pushed up
    or left.

    Parameters
    ----------
    state : dict
        The robot's state, as for picture_heading_rad.
    recorder_parameters : dict
        The recorder's settings ("picture_left_sign" and those of picture_heading_rad).

    Returns
    -------
    tuple of numpy.ndarray
        Unit vectors [x, y] in the robot's base frame: the picture's up, then its left.
    """

    heading_rad = picture_heading_rad(state, recorder_parameters)
    left_rad = heading_rad + recorder_parameters["picture_left_sign"] * math.pi / 2
    return numpy.array([math.cos(heading_rad), math.sin(heading_rad)]), numpy.array([math.cos(left_rad), math.sin(left_rad)])


def monitor_work_area(
        point_x: int,
        point_y: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    The usable area of the monitor that holds a screen point (Emranul's laptop drives three monitors, some at
    negative coordinates), so that a window can be placed on the same monitor as another.

    Parameters
    ----------
    point_x, point_y : int
        The point in screen pixels.

    Returns
    -------
    tuple of int or None
        Left, top, right and bottom of the monitor's work area (without the task bar); None off Windows.
    """

    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None
    if not hasattr(ctypes, "windll"):
        return None

    class MonitorInfo(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    user32 = ctypes.windll.user32
    user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
    user32.MonitorFromPoint.restype = wintypes.HANDLE
    monitor = user32.MonitorFromPoint(wintypes.POINT(int(point_x), int(point_y)), 2)  # 2: the nearest monitor
    information = MonitorInfo()
    information.cbSize = ctypes.sizeof(MonitorInfo)
    if not user32.GetMonitorInfoW(monitor, ctypes.byref(information)):
        return None
    work = information.rcWork
    return work.left, work.top, work.right, work.bottom


def show_in_window(picture: numpy.ndarray) -> None:
    """Show ``picture`` in the recording window, bring the window to the front, and remember it for re-showing."""

    LAST_SHOWN["picture"] = picture
    cv2.imshow(WINDOW_NAME, picture)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass


def status_bar_image(
        text: str,
        width: int
) -> numpy.ndarray:
    """
    A black bar with ``text`` in white, wrapped onto as many lines as it needs to fit ``width`` pixels, so that a
    long hint (the gamepad's controls) is not cut off at the picture's edge (5.11.26).

    Parameters
    ----------
    text : str
        The status text.
    width : int
        The bar's width in pixels, the picture's width.

    Returns
    -------
    numpy.ndarray
        The bar: 32 pixels high for one line, 22 more for every further line.
    """

    lines = [""]
    for word in text.split(" "):
        candidate = f"{lines[-1]} {word}".strip()
        if lines[-1] and cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] > width - 16:
            lines.append(word)
        else:
            lines[-1] = candidate
    status_bar = numpy.zeros((10 + 22 * len(lines), width, 3), dtype = numpy.uint8)
    for index, line in enumerate(lines):
        cv2.putText(status_bar, line, (8, 22 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return status_bar


def ensure_window_is_shown() -> None:
    """Re-open the recording window with the last picture if it was closed or is no longer visible."""

    if LAST_SHOWN["picture"] is None:
        return
    try:
        is_visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        is_visible = False
    if not is_visible:
        print("the picture window had gone; opened it again")
        show_in_window(LAST_SHOWN["picture"])


def sharpness_of(picture: numpy.ndarray) -> float:
    """Variance of the Laplacian, the same number view_camera.py shows: higher is sharper (about 100 to 150 when in focus)."""

    return float(cv2.Laplacian(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def rotation_from_roll_pitch_yaw(
        roll: float,
        pitch: float,
        yaw: float
) -> numpy.ndarray:
    """The rotation matrix of pyniryo's pose angles: extrinsic rotations about x (roll), then y (pitch), then z (yaw)."""

    cos_r, sin_r = math.cos(roll), math.sin(roll)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rotation_x = numpy.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
    rotation_y = numpy.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
    rotation_z = numpy.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    return rotation_z @ rotation_y @ rotation_x


def roll_pitch_yaw_from_rotation(rotation: numpy.ndarray) -> Tuple[float, float, float]:
    """The pose angles (roll, pitch, yaw) of a rotation matrix, the inverse of rotation_from_roll_pitch_yaw."""

    pitch = math.asin(max(-1.0, min(1.0, -rotation[2, 0])))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return roll, pitch, yaw


def rotation_about_axis(
        axis: numpy.ndarray,
        angle: float
) -> numpy.ndarray:
    """Rodrigues' rotation matrix for ``angle`` radians about the unit vector ``axis``."""

    axis = axis / numpy.linalg.norm(axis)
    cross = numpy.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return numpy.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def hand_orientation(
        turn_deg: float,
        picture_left_sign: int = RECORDING_PARAMETERS["picture_left_sign"]
) -> List[float]:
    """
    The orientation [roll, pitch, yaw] of the vertical rod with the hand turned by ``turn_deg`` about the vertical.

    A positive turn carries the picture's top towards picture-left of the untouched hand (``picture_left_sign`` sets
    which way round that is on the table). The untouched hand is [0, tool_pitch_rad, 0], camera looking along +x.
    """

    untouched = rotation_from_roll_pitch_yaw(0.0, WORKSPACE_PARAMETERS["tool_pitch_rad"], 0.0)
    turned = rotation_about_axis(numpy.array([0.0, 0.0, 1.0]), picture_left_sign * math.radians(turn_deg)) @ untouched
    return list(roll_pitch_yaw_from_rotation(turned))


def drain_window_keys(most_keys: int = 50) -> int:
    """
    Throw away keys pressed while the robot was moving, so that the next key read refers to the picture on screen.

    Returns
    -------
    int
        How many keys were thrown away; a double press during a move shows up here.
    """

    drained = 0
    while drained < most_keys and cv2.waitKey(1) >= 0:
        drained += 1
    return drained



class EpisodeRecorder:

    def __init__(
            self,
            robot_link,
            operator: str,
            path_id: str = SHEET_PARAMETERS["path_id"],
            recorder_parameters: Optional[dict] = None,
            key_reader: Callable[[int], Optional[str]] = read_window_key,
            key_drainer: Optional[Callable[[], int]] = None,
            pad_reader: Optional[Callable[[], Dict]] = None
    ):
        """
        Prepare a recording session on an open robot link.

        Parameters
        ----------
        robot_link : RobotLink or SimulatedRobotLink
            The robot to drive.
        operator : str
            Who records, written into every episode (the course grades each student's share).
        path_id : str
            The sheet's path_id, e.g. "path_01_straight_line"; it starts every episode's name.
        recorder_parameters : dict, optional
            Overrides merged over RECORDING_PARAMETERS, plus "start_point_xy" (the puck's start), "is_new_session",
            "is_window_shown", "raw_folder", "is_simulated", "step_budget" (default: step_budget_for(path_id)) and the
            episode metadata "camera_mount", "mapping_version", "sheet_orientation", "start_index",
            "deliberate_offset" and "lighting_note".
        key_reader : callable, optional
            Returns the next key; replaced in tests by a scripted sequence.
        key_drainer : callable, optional
            Throws away keys pressed during a move. Defaults to the window's drain when ``key_reader`` reads the
            window, and to nothing otherwise (a scripted sequence must not be eaten).
        pad_reader : callable, optional
            Returns the gamepad's state (GamepadReader.read) for the gamepad mode; replaced in tests by a script.
        """

        self.robot_link = robot_link
        self.operator = operator
        self.path_id = path_id
        self.key_reader = key_reader
        if key_drainer is None:
            key_drainer = drain_window_keys if key_reader is read_window_key else (lambda: 0)
        self.key_drainer = key_drainer
        self.pad_reader = pad_reader

        self.__initialize_variables()
        self.update_recorder_parameters(recorder_parameters)
        self.session_folder = self.__open_session_folder()


    def __initialize_variables(self):

        self.recorder_parameters = dict(RECORDING_PARAMETERS)
        self.recorder_parameters.update({
            "start_point_xy": WORKSPACE_PARAMETERS["line_start_xy"],
            "step_budget": step_budget_for(self.path_id),
            "is_new_session": False,
            "is_window_shown": True,
            "is_simulated": False,
            "sheet_orientation": "",
            "start_index": 0,
            "deliberate_offset": {"lateral_cm": 0.0, "angle_deg": 0.0},
            "lighting_note": "",
            "plan_number": "",
            "laptop": platform.node()
        })
        self.last_picture_md5 = None
        self.is_free_motion = False
        self.forearm_turn_deg = 0.0
        self.tip_xy = None
        self.is_arm_model_checked = False
        self.is_lost_pending = False
        self.current_rows = []
        self.puck_tracker = PuckTracker()
        self.puck_sighting = None
        self.is_tracking_error_reported = False
        self.is_tracking_failing = False
        self.camera_shift_px = 0.0


    def update_recorder_parameters(self, recorder_parameters: Optional[dict]):

        if recorder_parameters:
            self.recorder_parameters.update(recorder_parameters)
        # The puck is placed on the table only with a camera calibrated for the mount in use.
        self.puck_tracker.update_tracking_parameters({"camera_mount": self.recorder_parameters["camera_mount"]})


    def record_session(self, mode: str) -> List[Dict]:
        """
        Record episodes in ``mode`` ("keyboard" or "hand_guided") until the operator quits; returns their summaries.
        """

        summaries = []
        print(f"camera mount: {self.recorder_parameters['camera_mount']}, stepping rule: {self.recorder_parameters['stepping_rule_version']}")
        self.__append_session_notes(f"\n## {time.strftime('%H:%M')}: {self.operator}, {mode}, {self.path_id}"
                                    f"{', simulated robot' if self.recorder_parameters['is_simulated'] else ''}\n\n")
        try:
            self.robot_link.move_home(self.recorder_parameters["home_joints"])
            while True:
                self.__show_message(f"Put the puck at the start of the line. Enter = next {mode} episode, Esc = finish.")
                if self.__wait_for(["enter", "escape"]) == "escape":
                    break
                self.__simulate_puck_reset()

                summary = self.__record_episode(mode)
                summaries.append(summary)
                if summary["status"] == "quit" or self.recorder_parameters.get("is_single_episode"):
                    break
        finally:
            # Nothing may move while a person holds a torque-free arm: if the motors are off (an exception during
            # guiding, Ctrl-C included), they stay off until the person confirms that the arm is released.
            if self.is_free_motion:
                self.__release_arm_then_motors_on()
            self.robot_link.move_home(self.recorder_parameters["home_joints"])
            self.robot_link.close()
            if self.recorder_parameters["is_window_shown"]:
                cv2.destroyAllWindows()

        return summaries


    def __record_episode(self, mode: str) -> Dict:
        """Record one episode from the start pose and write its files; returns its summary."""

        episode_folder = self.__next_episode_folder()
        (episode_folder / "frames").mkdir(parents = True)
        summary = {
            "path_id": self.path_id,
            "episode": episode_folder.name,
            "mode": mode,
            "operator": self.operator,
            "started": datetime.datetime.now().isoformat(timespec = "seconds"),
            "is_simulated": self.recorder_parameters["is_simulated"],
            "code_commit": self.__code_commit(),
            "laptop": self.recorder_parameters["laptop"],
            "camera_mount": self.recorder_parameters["camera_mount"],
            "mapping_version": self.recorder_parameters["mapping_version"],
            "stepping_rule_version": self.recorder_parameters["stepping_rule_version"],
            "sheet_orientation": self.recorder_parameters["sheet_orientation"],
            "plan_number": self.recorder_parameters["plan_number"] or None,
            "start_index": self.recorder_parameters["start_index"],
            "deliberate_offset": self.recorder_parameters["deliberate_offset"],
            "lighting_note": self.recorder_parameters["lighting_note"],
            "start_pose": self.__start_pose(),
            "step_m": self.recorder_parameters["step_m"],
            "step_budget": self.recorder_parameters["step_budget"],
            "max_turn_lean_deg": self.recorder_parameters["max_turn_lean_deg"],
            "actions": self.recorder_parameters["actions"],
            "turn_actions": self.recorder_parameters["turn_actions"],
            "step_over_actions": self.recorder_parameters["step_over_actions"],
            "step_over_m": self.recorder_parameters["step_over_m"],
            "picture_left_sign": self.recorder_parameters["picture_left_sign"]
        }

        try:
            self.__lower_to_start()
        except Exception as error:
            # Seen on 31.10.26: a start 12 cm from the base axis cannot be reached with the rod vertical. Say so, lift
            # what was reached and go back to the prompt instead of crashing with the robot low over the sheet.
            print(f"cannot reach the start pose ({error}). Move the sheet so the line start is at least about 16 cm "
                  f"from the base, teach it again with teach_sheet_start.py, and restart with the new --start-x and --start-y.")
            summary["status"] = "start_unreachable"
            self.__lift()
            return self.__finish_episode(episode_folder, summary, [])

        if not self.__check_camera_shift(episode_folder, summary):
            summary["status"] = "discarded_camera_moved"
            self.__lift()
            return self.__finish_episode(episode_folder, summary, [])

        if mode in ("hand_guided", "hand_guided_exact"):
            guided_path = self.__record_guided_path(episode_folder)
            if guided_path is None:
                summary["status"] = "discarded_while_guiding"
                self.__lift()
                self.__ask_how_episode_ended(summary)
                return self.__finish_episode(episode_folder, summary, [])
            self.__lift()
            self.__show_message("Motors on. Put the puck back at the start of the line, then press Enter.")
            self.__wait_for(["enter"])
            self.__simulate_puck_reset()
            self.__lower_to_start()
            if mode == "hand_guided_exact":
                self.exact_waypoints = self.__exact_waypoints(guided_path)
                self.exact_index = 0
                self.is_auto_running = False
                self.reguide_count = 0
                self.__say(f"exact replay: {len(self.exact_waypoints)} waypoints from {len(guided_path)} guided points")
                decider = self.__exact_replay_decider
            else:
                decider = self.__make_replay_decider(guided_path[:, [0, 1, 5]])
        elif mode == "gamepad":
            if self.pad_reader is None:
                from gamepad import GamepadReader
                self.pad_reader = GamepadReader().read
            self.pad_wrist_offset_rad = 0.0
            self.pad_height_m = 0.0
            self.pad_buttons_before = set(self.pad_reader()["buttons"])
            decider = self.__gamepad_decider
        else:
            decider = self.__keyboard_decider

        status, rows = self.__run_steps(episode_folder, decider)
        summary["status"] = status
        if mode == "hand_guided_exact":
            summary["hand_corrections"] = getattr(self, "reguide_count", 0)
        self.__lift()
        self.__ask_how_episode_ended(summary)
        return self.__finish_episode(episode_folder, summary, rows)


    def __ask_how_episode_ended(self, summary: Dict) -> None:
        """Store whether the puck reached the goal, or why the episode was discarded; a quit episode asks nothing."""

        summary.update({"outcome": None, "discard_reason": None})
        if summary["status"] in ("completed", "step_budget_reached"):
            summary["outcome"] = self.__ask_choice("Puck in the goal? y = yes, n = no", self.recorder_parameters["outcome_keys"])
        elif summary["status"] in ("discarded", "discarded_while_guiding"):
            summary["discard_reason"] = self.__ask_choice("Why discard? t tipped, o off sheet, r robot, h hand, m other",
                                                          self.recorder_parameters["discard_reason_keys"])


    def __ask_choice(
            self,
            message: str,
            choices: Dict[str, str]
    ) -> str:
        """Show ``message`` and return the meaning of the key pressed; Esc returns "not_recorded"."""

        self.key_drainer()
        self.__show_message(message)
        return choices.get(self.__wait_for(list(choices) + ["escape"]), "not_recorded")


    def __run_steps(
            self,
            episode_folder: Path,
            decider: Callable[[bytes, Dict, int], Tuple[str, str]]
    ) -> Tuple[str, List[Dict]]:
        """
        The step loop shared by every mode: fresh picture, decision, log, 1 cm move.

        Returns
        -------
        tuple of str and list of dict
            The episode's status ("completed", "discarded", "quit" or "step_budget_reached") and the logged steps.
        """

        rows = []
        self.current_rows = rows
        self.is_lost_pending = False
        self.forearm_turn_deg = 0.0
        self.tip_xy = None
        self.is_arm_model_checked = False
        # Where the puck is believed to be, for the landing after a forearm turn: at the start point, then moved by
        # every step that reaches it (see __track_puck). Since 4.11.26 after test 3; a running count of how far the
        # block stood back drifted to 10 cm when sideways and back steps were taken in a turned picture.
        self.puck_xy_estimate = numpy.array(self.recorder_parameters["start_point_xy"], dtype = float)
        self.puck_sighting = None
        push = math.radians(self.recorder_parameters["start_push_direction_deg"])
        self.last_step_direction = numpy.array([math.cos(push), math.sin(push)])
        move_return_time = time.time()
        while not self.recorder_parameters["is_step_budget_enforced"] or len(rows) < self.recorder_parameters["step_budget"]:
            self.current_step = len(rows)
            # Kept up to date at every decision (5.11.26: a lost robot connection at the end of episode 041 lost all
            # 177 labels, because steps.csv was only written when the episode finished).
            if rows:
                self.__write_steps(episode_folder, rows, "steps_in_progress.csv")
            picture_bytes = self.__fresh_picture()
            capture_time = time.time()
            state = self.robot_link.read_state()
            self.puck_sighting = self.__see_puck(picture_bytes, state)
            action, source = decider(picture_bytes, state, len(rows))
            decision_time = time.time()

            if action in ("discard", "quit"):
                return ("discarded" if action == "discard" else "quit"), rows

            if action == "reguide":
                # Correction by hand in the exact replay (5.11.26, Emranul): instead of fixing a drifted puck with the
                # keys, guide again from here with the motors off; the correction is then replayed like the first path.
                self.__reguide(episode_folder, picture_bytes, state)
                move_return_time = time.time()
                continue

            if action == "pad":
                # Gamepad mode (5.11.26): one step from the pad's sticks and hand buttons; see __pad_target.
                try:
                    joints_next, tip_next, step_xy = self.__pad_target(state, self.pad_command)
                except Exception as error:
                    self.__say(f"refused: {str(error).splitlines()[0]}")
                    continue
                is_inside = (self.__is_step_allowed(tip_next[0], tip_next[1], step_xy[0], step_xy[1]) if self.recorder_parameters["pad_check_puck_on_sheet"]
                             else self.__is_inside_workspace(tip_next[0], tip_next[1]))
                if float(numpy.hypot(*step_xy)) > 1e-9 and not is_inside:
                    self.__say("refused: that step would carry the tip out of the workspace (table edge)")
                    continue
                tip_now = [tip_next[0] - step_xy[0], tip_next[1] - step_xy[1]]
                label, motion = self.__label_motion(state, joints_next, tip_now, tip_next)
                motion.update({name: round(value, 3) for name, value in self.pad_command.items()})
                motion["tip_height_above_push_m"] = round(self.pad_height_next_m, 4)
                times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
                row = self.__log_step(episode_folder, len(rows), picture_bytes, state, label, source, times)
                row.update(motion)
                rows.append(row)
                try:
                    self.robot_link.move_joints(joints_next)
                except Exception as error:
                    self.__say(f"the robot refused that step ({str(error).splitlines()[0]})")
                    rows.pop()
                    (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                    self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                    continue
                self.forearm_turn_deg = math.degrees(joints_next[3])
                self.pad_height_m = self.pad_height_next_m
                self.tip_xy = list(tip_next)
                self.__track_puck(label, tip_next, step_xy[0], step_xy[1])
                move_return_time = time.time()
                continue

            if action == "waypoint":
                # Exact replay (4.11.26, Emranul): move to the next guided waypoint joint by joint. The row's action is
                # the nearest fixed move; the exact movement is stored beside it and in the next row's joints.
                joints_next = [float(value) for value in self.exact_waypoints[self.exact_index]]
                label, motion = self.__label_motion(state, joints_next)
                motion["waypoint"] = self.exact_index
                times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
                row = self.__log_step(episode_folder, len(rows), picture_bytes, state, label, source, times)
                row.update(motion)
                rows.append(row)
                try:
                    self.robot_link.move_joints(joints_next)
                except Exception as error:
                    self.__say(f"the robot refused waypoint {self.exact_index} ({str(error).splitlines()[0]}); Enter tries again")
                    rows.pop()
                    (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                    self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                    self.is_auto_running = False
                    continue
                self.exact_index += 1
                self.forearm_turn_deg = math.degrees(joints_next[3])
                self.tip_xy = [float(value) for value in magnet_tip_and_lean(joints_next)[0][:2]]
                move_return_time = time.time()
                continue

            if action in self.recorder_parameters["turn_actions"]:
                least, most = self.recorder_parameters["min_turn_deg"], self.recorder_parameters["max_turn_deg"]
                turn_deg = max(least, min(most, self.forearm_turn_deg + self.recorder_parameters["turn_actions"][action]))
                if turn_deg == self.forearm_turn_deg:
                    self.__say(f"refused: the forearm cannot turn further that way (allowed {least} to {most} degrees)")
                    continue
                # The tip now: tracked while the forearm is turned (the rod leans), the rod's end when it is straight
                # (rod upright above the tip). Test 1 of 4.11.26 (episode 033, step 72) landed 14.9 cm away because a
                # tip remembered from an earlier turn back to straight was used after 33 ordinary steps.
                tip_xy = list(self.tip_xy) if self.forearm_turn_deg and self.tip_xy is not None else list(state["pose"][:2])
                # Land back from the puck, straight away from it, only by what the block does not already stand back
                # (4.11.26: two turns in a row put the tip 5 cm away; a turn after backing off moved it further still).
                away_from_puck = numpy.array(tip_xy) - self.puck_xy_estimate
                distance_to_puck = float(numpy.linalg.norm(away_from_puck))
                clearance_m = max(0.0, distance_to_puck - self.recorder_parameters["puck_centre_ahead_of_tip_m"])
                retreat_m = max(0.0, self.recorder_parameters["turn_landing_retreat_m"] - clearance_m)
                retreat_direction = away_from_puck / distance_to_puck if distance_to_puck > 1e-9 else numpy.array([-1.0, 0.0])
                if clearance_m > self.recorder_parameters["puck_estimate_lost_m"]:
                    # Test 5 of 4.11.26: the believed puck position fell 22 cm behind the real puck, so no turn stepped
                    # back any more. Further off than this, assume the block touches the puck ahead of the last step.
                    self.__say(f"the believed puck position is {100 * clearance_m:.0f} cm off; stepping back opposite to the last step instead")
                    retreat_m = self.recorder_parameters["turn_landing_retreat_m"]
                    retreat_direction = -self.last_step_direction
                    self.puck_xy_estimate = numpy.array(tip_xy) + self.recorder_parameters["puck_centre_ahead_of_tip_m"] * self.last_step_direction
                    clearance_m = 0.0
                landing_xy = list(numpy.array(tip_xy) + retreat_m * retreat_direction)
                if not self.__is_inside_workspace(landing_xy[0], landing_xy[1]):
                    self.__say("refused: the landing point behind the puck would be outside the workspace")
                    continue
                try:
                    plan = self.__turn_plan(self.forearm_turn_deg, turn_deg, tip_xy, landing_xy, state["joints"])
                    self.__check_arm_model(plan[-1])
                except Exception as error:
                    self.__say(f"refused: {error}; nothing moved")
                    continue
                times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
                rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
                try:
                    for joints in plan:
                        self.robot_link.move_joints(joints)
                except Exception as error:
                    self.__say(f"the robot refused that turn ({error}); choose another step")
                    rows.pop()
                    (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                    self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                    continue
                self.forearm_turn_deg = turn_deg
                self.tip_xy = landing_xy if turn_deg else None
                self.__say(f"forearm turned {turn_deg:+.0f} degrees, rod leaning {magnet_tip_and_lean(plan[-1])[1]:.1f} degrees, tip moved "
                      f"{100 * retreat_m:.1f} cm back from the puck ({100 * (clearance_m + retreat_m):.1f} cm clear); the picture's top now points "
                      f"{math.degrees(picture_heading_rad(self.robot_link.read_state(), self.recorder_parameters)):+.0f} degrees from the table's x axis")
                move_return_time = time.time()
                continue

            if action in self.recorder_parameters["flick_actions"]:
                # q / e (4.11.26 evening): a fast turn of the hand about the rod and straight back; the rod stays put.
                times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
                rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
                counter_clockwise_rad = self.recorder_parameters["picture_left_sign"] * math.radians(self.recorder_parameters["flick_actions"][action])
                try:
                    # Joint 6 turns the hand clockwise seen from above as it grows (with the rod pointing down).
                    self.robot_link.turn_hand_and_back(-counter_clockwise_rad, self.recorder_parameters["flick_velocity_percent"])
                except Exception as error:
                    self.__say(f"the robot refused that flick ({error}); choose another step")
                    rows.pop()
                    (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                    self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                    continue
                move_return_time = time.time()
                continue

            if action in self.recorder_parameters["lean_actions"]:
                target_pose = self.__leaned_pose(state["pose"], self.recorder_parameters["lean_actions"][action])
                if target_pose is None:
                    self.__say(f"refused: already leaning {self.recorder_parameters['max_lean_deg']} degrees that way")
                    continue
                times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
                rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
                try:
                    self.robot_link.move_to(target_pose, is_linear = False)
                except Exception as error:
                    self.__say(f"the robot refused that lean ({error}); choose another step")
                    rows.pop()
                    (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                    self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                    continue
                move_return_time = time.time()
                continue

            if action != "stop":
                # While the forearm is turned the rod leans, so the commanded magnet tip is tracked instead of the
                # rod's end. The keys are measured in the PICTURE's frame, read from the camera's current direction
                # (Emranul, 4.11.26: he steers from the picture, so "forward" must be the top of the picture also
                # when the forearm is turned; the model sees nothing but the picture either).
                x, y = self.tip_xy if self.forearm_turn_deg else state["pose"][:2]
                step_x, step_y = action_step_to_table(action, state["pose"], self.recorder_parameters["hand_turn_deg"],
                                                      heading_rad = picture_heading_rad(state, self.recorder_parameters))
                target_x, target_y = x + step_x, y + step_y
                # A step-over leaves the puck where it is, so only the rod's landing point needs checking.
                is_allowed = (self.__is_inside_workspace(target_x, target_y) if self.__is_step_over(action)
                              else self.__is_step_allowed(target_x, target_y, step_x, step_y))
                if not is_allowed:
                    self.__say(f"refused: {action} would carry the rod or the puck out of the workspace; choose another step")
                    continue

            times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
            rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
            if action == "stop":
                return "completed", rows

            # The rod is asked to be vertical at every step (the start pose's orientation); if the bracket's wrist
            # limit forbids that near the base, RobotLink leans it just enough and says so (2.11.26). A lean from the
            # n / m test keys therefore lasts one step only. With the forearm turned, the step is solved as joints
            # (least lean, magnet tip on the target) and the turn lasts.
            target_pose = [target_x, target_y, pushing_height(target_x, target_y)] + self.__turned_orientation(self.recorder_parameters["hand_turn_deg"])
            try:
                if self.forearm_turn_deg:
                    self.robot_link.move_joints(self.__turned_joints([target_x, target_y], self.forearm_turn_deg, 0.0))
                    self.tip_xy = [target_x, target_y]
                elif self.__is_step_over(action):
                    self.__step_over(target_pose)
                else:
                    self.robot_link.move_to(target_pose)
            except Exception as error:
                # The robot refused the step (seen on 31.10.26 near the base: "cannot be reached with a linear
                # trajectory"). The step did not happen, so its row and picture are taken back and the person
                # chooses another step from the same picture; the episode goes on.
                self.__say(f"the robot refused {action} ({str(error).splitlines()[0]}); choose another step")
                if self.forearm_turn_deg and "lean" in str(error):
                    self.__say(self.__turn_hint([target_x, target_y]))
                rows.pop()
                (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                continue
            self.__track_puck(action, [target_x, target_y], step_x, step_y)
            move_return_time = time.time()

        return "step_budget_reached", rows


    def __exact_waypoints(self, guided_path: numpy.ndarray) -> List[numpy.ndarray]:
        """
        The joint positions the exact replay visits: the first guided point, then every point where the tip has moved
        "exact_waypoint_m" or any joint has turned "exact_waypoint_joint_deg" since the last waypoint, and the last.
        ``guided_path`` columns: tip x, tip y, joints 1 to 6.
        """

        waypoints = [guided_path[0, 2:8]]
        last = guided_path[0]
        for point in guided_path[1:]:
            tip_moved = float(numpy.hypot(point[0] - last[0], point[1] - last[1]))
            joint_turned = float(numpy.max(numpy.abs(point[2:8] - last[2:8])))
            if tip_moved >= self.recorder_parameters["exact_waypoint_m"] or joint_turned >= math.radians(self.recorder_parameters["exact_waypoint_joint_deg"]):
                waypoints.append(point[2:8])
                last = point
        if last is not guided_path[-1]:
            waypoints.append(guided_path[-1, 2:8])
        return waypoints


    def __check_camera_shift(
            self,
            episode_folder: Path,
            summary: Dict
    ) -> bool:
        """
        With the arm at the start position, compare the picture's bottom strip (the robot's hand and the block, the
        same in every picture taken at this pose) with a reference picture, and measure by phase correlation how
        many pixels it has shifted (5.11.26, Emranul: the camera is taped to the forearm cover and must not creep).

        The first episode for a camera mount and start position saves its picture as the reference. The shift is
        written into episode.json; above "camera_shift_warning_px" the person is asked whether to go on.

        Returns
        -------
        bool
            False when the person chose to discard the episode because the camera moved.
        """

        picture_bytes = self.__fresh_picture()
        (episode_folder / "camera_check.jpg").write_bytes(picture_bytes)
        start_x, start_y = self.recorder_parameters["start_point_xy"]
        reference_folder = PROJECT_FOLDER / self.recorder_parameters["raw_folder"] / "camera_check" / "camera_reference"
        reference_path = reference_folder / f"{self.recorder_parameters['camera_mount']}_x{1000 * start_x:.0f}_y{1000 * start_y:.0f}mm.jpg"
        if not reference_path.exists():
            reference_folder.mkdir(parents = True, exist_ok = True)
            reference_path.write_bytes(picture_bytes)
            summary["camera_shift_px"] = 0.0
            self.camera_shift_px = 0.0
            self.__say(f"camera check: no reference for this mount and start yet; this picture is the reference now ({reference_path.name})")
            return True

        shift_x, shift_y = camera_shift_px(reference_path.read_bytes(), picture_bytes, self.recorder_parameters["camera_check_region"])
        shift = math.hypot(shift_x, shift_y)
        summary["camera_shift_px"] = round(shift, 2)
        self.camera_shift_px = shift
        summary["camera_reference"] = reference_path.name
        limit = self.recorder_parameters["camera_shift_warning_px"]
        self.__say(f"camera check: the hand at the bottom of the picture is {shift:.1f} px from the reference "
                   f"(x {shift_x:+.1f}, y {shift_y:+.1f}; warning above {limit})")
        if shift <= limit:
            return True
        self.__show_picture(picture_bytes, f"CAMERA MOVED by {shift:.1f} px since the reference: check the mount. Enter = record anyway, x = discard")
        self.__say("CAMERA MOVED: check the camera on the forearm cover. Enter = record anyway, x = discard this episode")
        return self.__wait_for(["enter", "x"]) == "enter"


    def __reguide(
            self,
            episode_folder: Path,
            picture_bytes: bytes,
            state: Dict
    ) -> None:
        """
        Correct by hand from where the replay stands: keep the picture as a reference, record a new guided path with
        the motors off, then ask for the puck to be put back as in the reference picture, return to the starting
        joints from above, and make the new path the rest of the replay. Nothing is logged as a step meanwhile, so
        every logged picture still comes from the robot moving on its own. A discarded correction returns to the same
        place and the old replay goes on.
        """

        self.reguide_count = getattr(self, "reguide_count", 0) + 1
        number = self.reguide_count
        (episode_folder / f"correction_{number:02d}_reference.jpg").write_bytes(picture_bytes)
        start_joints = [float(value) for value in state["joints"]]
        self.__say(f"correction {number} by hand: start joints {[round(value, 3) for value in start_joints]}")

        new_path = self.__record_guided_path(episode_folder, f"correction_{number:02d}_guided_path.csv")
        # The hand has moved the arm: the remembered forearm turn and tip no longer hold, so lift straight up from
        # where the arm is now (a turn-back lift would fly to the old tip, low over the sheet).
        self.forearm_turn_deg = 0.0
        self.tip_xy = None
        self.__lift()
        if new_path is not None:
            # 6.11.26 (Marshal): the arm first goes back to where the correction began, with the puck off the sheet;
            # the camera then sees from the same place as the reference picture, so a circle can show the puck's spot.
            self.__show_message("Take the puck OFF the sheet (away from the block), then Enter: the arm goes back to where the correction began")
            self.__wait_for(["enter"])
        self.__return_to_joints_from_above(start_joints)
        if new_path is None:
            self.__say(f"correction {number} discarded; the replay goes on where it was")
            return
        self.__guide_puck_back(picture_bytes)
        self.exact_waypoints = self.__exact_waypoints(new_path)
        self.exact_index = 0
        self.is_auto_running = False
        self.__say(f"correction {number}: {len(self.exact_waypoints)} waypoints from {len(new_path)} guided points; Enter or g replays it")


    def __guide_puck_back(self, reference_bytes: bytes) -> None:
        """
        Show the live picture with a circle where the puck was in the reference picture, until Enter. The arm stands
        where the reference was taken, so the circle is in the right place; the found puck is marked too, with its
        distance from the circle. Without a puck in the reference, the reference picture is shown instead.
        """

        reference = cv2.imdecode(numpy.frombuffer(reference_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        target = detect_puck(reference)
        if target is None:
            self.__show_picture(reference_bytes, "REFERENCE: put the puck back where it is in this picture, then Enter")
            self.__say("no puck found in the reference picture: put it back as the window shows, then Enter")
            self.__wait_for(["enter"])
            return
        centre = (int(round(target["centre_u"])), int(round(target["centre_v"])))
        radius = int(round(target["equivalent_radius_px"]))
        self.__say("put the puck into the green circle in the live picture, then Enter")
        while True:
            picture = cv2.imdecode(numpy.frombuffer(self.robot_link.read_picture_bytes(), numpy.uint8), cv2.IMREAD_COLOR)
            shown = picture.copy()
            cv2.circle(shown, centre, radius + 3, (0, 220, 0), 2)
            seen = detect_puck(picture)
            if seen is None:
                text = "put the puck into the GREEN circle (no puck seen yet), then Enter"
            else:
                distance = math.hypot(seen["centre_u"] - centre[0], seen["centre_v"] - centre[1])
                cv2.drawMarker(shown, (int(round(seen["centre_u"])), int(round(seen["centre_v"]))), (0, 230, 255), cv2.MARKER_CROSS, 16, 2)
                text = f"puck {distance:.0f} px from the GREEN circle's centre{' (good)' if distance <= self.recorder_parameters['puck_back_good_px'] else ''}, then Enter"
            if self.recorder_parameters["is_window_shown"]:
                show_in_window(numpy.vstack([shown, status_bar_image(text, shown.shape[1])]))
            if self.key_reader(100) == "enter":
                return


    def __return_to_joints_from_above(self, joints: List[float]) -> None:
        """
        Go to ``joints`` from above: the rod's end over that place, lifted by "approach_height_m" (5 cm; the puck,
        2.9 cm tall, may already be back in front of it), then down joint by joint.
        """

        from side_view_test import ned2_fk as arm_model

        position, orientation = arm_model.tool_pose(list(joints))
        lifted = [float(position[0]), float(position[1]), float(position[2]) + self.recorder_parameters["approach_height_m"]] + [float(value) for value in orientation]
        try:
            self.robot_link.move_to(lifted, is_linear = False)
        except Exception as error:
            # Above the target in joint space instead, so the last move is a short one straight down onto the puck's place.
            try:
                self.robot_link.move_joints(self.__joints_lifted(list(joints)))
                self.__say(f"the lifted approach was refused ({str(error).splitlines()[0]}); came above it with joints 2 and 3 instead")
            except Exception:
                self.__say(f"the lifted approach was refused ({str(error).splitlines()[0]}); going down joint by joint from here")
        self.robot_link.move_joints(list(joints))
        self.forearm_turn_deg = math.degrees(joints[3])
        self.tip_xy = [float(value) for value in magnet_tip_and_lean(list(joints))[0][:2]]


    def __gamepad_decider(
            self,
            picture_bytes: bytes,
            state: Dict,
            step: int
    ) -> Tuple[str, str]:
        """
        Wait for the gamepad: a stick out of its dead zone or a hand button gives a step ("pad", with the command in
        ``self.pad_command``); the stop, discard and quit buttons end the episode; mistake and lost flag as the u / l
        keys do. Buttons count when pressed, not while held. The keyboard's Esc and x still work.
        """

        self.__show_orientation(state, view = "gamepad")
        self.__show_picture(picture_bytes, f"step {step}: left stick = move, right stick = twist / bend wrist, "
                                           "L1 R1 = turn hand, L2 R2 = down / up, "
                                           "cross = stop, circle = discard, square = mistake, triangle = lost")
        self.__drain_keys(step)
        while True:
            key = self.key_reader(int(1000 * self.recorder_parameters["pad_poll_s"]))
            if key == "escape":
                return "quit", "human"
            if self.recorder_parameters["keys"].get(key) == "discard":
                return "discard", "human"
            pad = self.pad_reader()
            pressed = set(pad["buttons"]) - self.pad_buttons_before
            self.pad_buttons_before = set(pad["buttons"])
            if "stop" in pressed:
                return "stop", "human"
            if "discard" in pressed:
                return "discard", "human"
            if "quit" in pressed:
                return "quit", "human"
            for flag in ("mistake", "lost"):
                if flag in pressed:
                    self.__handle_flag_key(flag, picture_bytes, step)
            hand = (1.0 if "hand_left" in pressed else 0.0) - (1.0 if "hand_right" in pressed else 0.0)
            up, down = pad.get("up", 0.0), pad.get("down", 0.0)
            if any(abs(pad[name]) > 0.0 for name in ("move_x", "move_y", "twist", "lean")) or hand or up or down:
                self.pad_command = {"pad_move_x": pad["move_x"], "pad_move_y": pad["move_y"], "pad_twist": pad["twist"],
                                    "pad_lean": pad["lean"], "pad_hand": hand, "pad_up": up, "pad_down": down}
                return "pad", "gamepad"


    def __pad_target(
            self,
            state: Dict,
            command: Dict
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        The joints, the magnet tip and the tip's table step for one gamepad command.

        The left stick moves the tip in the picture's frame (up = towards the picture's top), by "pad_step_m" times how
        far it is pushed (at least "pad_min_step_m"). The right stick twists the forearm (right = like b) and adds to a
        wrist offset (up = bend one way) kept for the episode; L1 / R1 turn the hand. The arm's first three joints are
        solved to keep the tip on its target at pushing height with the rod leaning as little as the twist allows,
        plus the wrist offset. Raises when no joint angles within the limits reach the target, or the rod would lean
        more than "pad_max_lean_deg".
        """

        settings = self.recorder_parameters
        joints_now = [float(value) for value in state["joints"]]
        tip_now = list(self.tip_xy) if self.tip_xy is not None else (
            list(state["pose"][:2]) if settings["is_simulated"] else [float(value) for value in magnet_tip_and_lean(joints_now)[0][:2]])

        push = math.hypot(command["pad_move_x"], command["pad_move_y"])
        step_xy = [0.0, 0.0]
        if push > 0.0:
            length = max(settings["pad_min_step_m"], settings["pad_step_m"] * min(1.0, push))
            up_xy, left_xy = picture_axes_on_table(state, settings)
            forward_part, left_part = command["pad_move_y"] / push, -command["pad_move_x"] / push
            step_xy = [float(value) for value in length * (forward_part * up_xy + left_part * left_xy)]
        tip_next = [tip_now[0] + step_xy[0], tip_now[1] + step_xy[1]]

        twist_rad = joints_now[3] - math.radians(command["pad_twist"] * settings["pad_twist_deg"])
        self.pad_wrist_offset_rad = getattr(self, "pad_wrist_offset_rad", 0.0) + math.radians(command["pad_lean"] * settings["pad_lean_deg"])
        hand_rad = joints_now[5] - settings["picture_left_sign"] * math.radians(command["pad_hand"] * settings["pad_hand_deg"])

        # L2 / R2: the tip's height above pushing height, never below it (the tip must not press into the table).
        height_change = (command.get("pad_up", 0.0) - command.get("pad_down", 0.0)) * settings["pad_height_step_m"]
        self.pad_height_next_m = min(settings["pad_max_height_m"], max(0.0, getattr(self, "pad_height_m", 0.0) + height_change))
        height = pushing_height(tip_next[0], tip_next[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"] + self.pad_height_next_m
        target = numpy.array([tip_next[0], tip_next[1], height])
        least = solve_least_lean(target, twist_rad, wrist_range_deg = (max(-90.0, math.degrees(joints_now[4] - self.pad_wrist_offset_rad) - 10.0),
                                                                        min(90.0, math.degrees(joints_now[4] - self.pad_wrist_offset_rad) + 10.0)))
        if least is None:
            least = solve_least_lean(target, twist_rad, wrist_range_deg = (-90.0, 90.0))
        if least is None:
            raise RuntimeError(f"no arm position reaches ({tip_next[0]:.3f}, {tip_next[1]:.3f}) with the forearm at {math.degrees(twist_rad):+.0f} degrees")
        joints = least
        if abs(self.pad_wrist_offset_rad) > 1e-9:
            from side_view_test import solve_with_wrist_fixed
            joints = solve_with_wrist_fixed(target, twist_rad, least[4] + self.pad_wrist_offset_rad, least[:3])
            if joints is None:
                self.pad_wrist_offset_rad -= math.radians(command["pad_lean"] * settings["pad_lean_deg"])
                raise RuntimeError("the wrist cannot bend that far there")
        joints = list(joints)
        joints[5] = hand_rad
        lean_deg = magnet_tip_and_lean(joints)[1]
        if lean_deg > settings["pad_max_lean_deg"]:
            self.pad_wrist_offset_rad -= math.radians(command["pad_lean"] * settings["pad_lean_deg"])
            raise RuntimeError(f"the rod would lean {lean_deg:.0f} degrees, above the gamepad limit of {settings['pad_max_lean_deg']:.0f}")
        return joints, tip_next, step_xy


    def __exact_replay_decider(
            self,
            picture_bytes: bytes,
            state: Dict,
            step: int
    ) -> Tuple[str, str]:
        """
        Exact replay: Enter moves to the next guided waypoint; g runs through the rest without waiting (any key
        pauses); u / l flag as usual; x discards; Esc quits. No direction keys: the replay copies the hand.
        """

        total = len(self.exact_waypoints)
        proposal = "waypoint" if self.exact_index < total else "stop"
        self.__show_orientation(state, view = "exact_replay")
        if self.is_auto_running and proposal == "waypoint":
            self.__show_picture(picture_bytes, f"exact replay running: waypoint {self.exact_index + 1} of {total}. Any key = pause")
            if self.key_reader(1) is None:
                return proposal, "replay_exact"
            self.is_auto_running = False
            self.key_drainer()
        self.__show_picture(picture_bytes, f"exact replay: waypoint {min(self.exact_index + 1, total)} of {total}, next = {proposal}. "
                                           f"Enter = next, g = run on, h = correct by hand, u = mistake, l = lost, x = discard, Esc = quit")
        self.__drain_keys(step)
        while True:
            key = self.key_reader(0)
            if key == "escape":
                return "quit", "human"
            if key == "g" and proposal == "waypoint":
                self.is_auto_running = True
                return proposal, "replay_exact"
            if key == "h":
                return "reguide", "human"
            action = self.recorder_parameters["keys"].get(key)
            if self.__handle_flag_key(action, picture_bytes, step):
                continue
            if action == "accept":
                return proposal, "replay_exact"
            if action == "discard":
                return "discard", "human"


    def __label_motion(
            self,
            state: Dict,
            joints_next: List[float],
            tip_now_xy: Optional[List[float]] = None,
            tip_next_xy: Optional[List[float]] = None
    ) -> Tuple[str, Dict]:
        """
        The nearest fixed move for the motion from ``state`` to ``joints_next``, and the motion itself: the magnet
        tip's move in the picture's frame (forward towards the picture's top, left towards its left) and the
        forearm's change. A forearm change of "exact_turn_label_deg" or more is labelled as a turn; a tip move under
        "exact_small_motion_m" as "small_motion".
        """

        tip_now = tip_now_xy if tip_now_xy is not None else magnet_tip_and_lean(list(state["joints"]))[0][:2]
        tip_next = tip_next_xy if tip_next_xy is not None else magnet_tip_and_lean(list(joints_next))[0][:2]
        moved = numpy.array(tip_next) - numpy.array(tip_now)
        heading = picture_heading_rad(state, self.recorder_parameters)
        sign = self.recorder_parameters["picture_left_sign"]
        forward_m = float(moved @ numpy.array([math.cos(heading), math.sin(heading)]))
        left_m = float(moved @ numpy.array([math.cos(heading + sign * math.pi / 2), math.sin(heading + sign * math.pi / 2)]))
        forearm_change_deg = math.degrees(joints_next[3] - state["joints"][3])
        motion = {"tip_forward_m": round(forward_m, 5), "tip_left_m": round(left_m, 5), "forearm_change_deg": round(forearm_change_deg, 2)}
        motion.update(lean_and_hand_labels(list(state["joints"]), list(joints_next), heading, self.recorder_parameters))

        if abs(forearm_change_deg) >= self.recorder_parameters["exact_turn_label_deg"] - 0.01:
            return ("turn_left" if forearm_change_deg > 0 else "turn_right"), motion
        if math.hypot(forward_m, left_m) < self.recorder_parameters["exact_small_motion_m"]:
            return "small_motion", motion
        picture_angle = math.degrees(math.atan2(left_m, forward_m))
        moving = {action: angle for action, angle in self.recorder_parameters["actions"].items() if angle is not None}
        nearest = min(moving, key = lambda action: abs((picture_angle - moving[action] + 180.0) % 360.0 - 180.0))
        return nearest, motion


    def __say(self, message: str) -> None:
        """Print ``message`` with the number of the step it belongs to (the next step to be logged), for debugging."""

        print(f"[step {getattr(self, 'current_step', 0)}] {message}")


    def __track_puck(
            self,
            action: str,
            tip_xy: List[float],
            step_x: float,
            step_y: float
    ) -> None:
        """
        Move the believed puck position after a step that happened, for the landing after a forearm turn.

        When the tip ends closer to the believed puck centre than the contact distance (puck radius plus tip radius),
        the step has reached the puck and pushed it: the puck is put at contact distance ahead of the tip along the
        step. Otherwise the puck stays. A step-over passes above the puck and leaves it where it is.
        """

        step = numpy.array([step_x, step_y])
        length = float(numpy.linalg.norm(step))
        if length < 1e-12 or self.__is_step_over(action):
            return
        self.last_step_direction = step / length
        contact = self.recorder_parameters["puck_centre_ahead_of_tip_m"]
        tip = numpy.array(tip_xy, dtype = float)
        if float(numpy.linalg.norm(self.puck_xy_estimate - tip)) < contact - 1e-9:
            self.puck_xy_estimate = tip + contact * step / length


    def __turn_hint(self, target_xy: List[float]) -> str:
        """
        After a turned step was refused for the rod's lean: the largest forearm turn, in key presses back towards
        straight, at which the same step would go through (straight always does).
        """

        turn_step_deg = abs(self.recorder_parameters["turn_actions"]["turn_left"])
        back_action = "turn_right" if self.forearm_turn_deg > 0 else "turn_left"
        back_key = next((key for key, action in self.recorder_parameters["keys"].items() if action == back_action), "?")
        candidate_deg = self.forearm_turn_deg
        presses = 0
        while abs(candidate_deg) > 1e-9:
            candidate_deg -= math.copysign(min(turn_step_deg, abs(candidate_deg)), candidate_deg)
            presses += 1
            try:
                self.__turned_joints(target_xy, candidate_deg, 0.0)
                break
            except Exception:
                continue
        return (f"hint: that spot can be reached with the forearm at {candidate_deg:+.0f} degrees: press {back_key} {presses} "
                f"time{'s' if presses > 1 else ''} (the picture turns, so the keys point differently afterwards)")


    def __leaned_pose(
            self,
            pose: List[float],
            lean_deg: float
    ) -> Optional[List[float]]:
        """
        The pose after leaning rod and camera together by ``lean_deg`` about the camera's forward axis, the rod's end
        staying in place; None when the lean would exceed ``max_lean_deg``.

        The orientation is rotated as a matrix (pyniryo's roll, pitch, yaw are extrinsic x, y, z), because at the
        vertical rod's pitch of 90 degrees roll and yaw both spin the rod about its own axis and cannot express a lean.
        """

        x, y, z = pose[:3]
        heading = camera_heading_rad(pose)
        forward_axis = numpy.array([math.cos(heading), math.sin(heading), 0.0])
        current = rotation_from_roll_pitch_yaw(*pose[3:])
        leaned = rotation_about_axis(forward_axis, math.radians(lean_deg)) @ current

        # Niryo's tool frame points along its x axis (pitch 90 degrees = rod straight down), so the rod's own axis is
        # the first column; its angle from straight down is the total lean.
        rod_axis = leaned[:, 0]
        total_lean_deg = math.degrees(math.acos(max(-1.0, min(1.0, -rod_axis[2]))))
        if total_lean_deg > self.recorder_parameters["max_lean_deg"] + 0.5:
            return None

        return [x, y, z] + list(roll_pitch_yaw_from_rotation(leaned))


    def __turned_orientation(self, turn_deg: float) -> List[float]:
        """The vertical rod's orientation with the hand turned by ``turn_deg``; see hand_orientation."""

        return hand_orientation(turn_deg, self.recorder_parameters["picture_left_sign"])


    def __turned_joints(
            self,
            tip_xy: List[float],
            turn_deg: float,
            lift: float,
            is_lean_limited: bool = True,
            near_wrist_rad: Optional[float] = None
    ) -> List[float]:
        """
        The joints that put the magnet tip at ``tip_xy``, ``lift`` metres above pushing height, with the forearm
        (joint 4) turned by ``turn_deg`` and the rod leaning least (Marshal's solver, 4.11.26).

        ``near_wrist_rad`` narrows the wrist search to 8 degrees either side of a neighbouring solution (a quarter of
        the time; the full search is the fallback). With the forearm untwisted, joint 6 is set equal to joint 1, the
        hand's usual orientation for a vertical rod, so the first step after turning back does not spin the block
        against the puck. Raises when no joint angles within the limits reach the point, or when the rod would lean
        more than the limit for that height ("max_turn_lean_deg" down, "max_turn_lifted_lean_deg" lifted). The wrist
        (joint 5) is kept below the bracket's limit.
        """

        height = pushing_height(tip_xy[0], tip_xy[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"] + lift
        wrist_max_deg = min(60.0, math.degrees(self.recorder_parameters["wrist_pitch_max_rad"]))
        target = numpy.array([tip_xy[0], tip_xy[1], height])
        joints = None
        if near_wrist_rad is not None:
            near_deg = math.degrees(near_wrist_rad)
            joints = solve_least_lean(target, math.radians(turn_deg), wrist_range_deg = (max(-60.0, near_deg - 8.0), min(wrist_max_deg, near_deg + 8.0)))
        if joints is None:
            joints = solve_least_lean(target, math.radians(turn_deg), wrist_range_deg = (-60.0, wrist_max_deg))
        if joints is None:
            raise RuntimeError(f"no arm position reaches ({tip_xy[0]:.3f}, {tip_xy[1]:.3f}) with the forearm turned {turn_deg:.0f} degrees")
        if abs(turn_deg) < 1e-9:
            joints[5] = joints[0]
        lean_deg = magnet_tip_and_lean(joints)[1]
        limit = self.recorder_parameters["max_turn_lifted_lean_deg" if lift > 0.0 else "max_turn_lean_deg"]
        if is_lean_limited and lean_deg > limit:
            raise RuntimeError(f"the rod would lean {lean_deg:.1f} degrees at ({tip_xy[0]:.3f}, {tip_xy[1]:.3f}) with the forearm "
                               f"turned {turn_deg:.0f} degrees, above the limit of {limit}")
        return joints


    def __turn_plan(
            self,
            from_turn_deg: float,
            to_turn_deg: float,
            tip_xy: List[float],
            landing_xy: List[float],
            current_joints: List[float]
    ) -> List[List[float]]:
        """
        The joint positions of a forearm turn: up over ``tip_xy`` at the current turn (through the near-paper heights,
        so the tip rises nearly straight), lifted at every "turn_increment_deg" on the way, lifted over ``landing_xy``,
        then down there at the new turn through the same heights. All are solved before anything moves, so a refusal
        moves nothing.
        """

        lift = self.recorder_parameters["turn_lift_m"]
        increment = self.recorder_parameters["turn_increment_deg"]
        near_paper = sorted(height for height in self.recorder_parameters["turn_near_paper_heights_m"] if height < lift)
        stops = [(from_turn_deg, height, tip_xy) for height in near_paper] + [(from_turn_deg, lift, tip_xy)]
        turn_deg = from_turn_deg
        while abs(to_turn_deg - turn_deg) > 1e-9:
            turn_deg += math.copysign(min(increment, abs(to_turn_deg - turn_deg)), to_turn_deg - turn_deg)
            stops.append((turn_deg, lift, tip_xy))
        stops.append((to_turn_deg, lift, landing_xy))
        stops += [(to_turn_deg, height, landing_xy) for height in reversed(near_paper)]
        stops.append((to_turn_deg, 0.0, landing_xy))

        plan = []
        near_wrist_rad = current_joints[4]
        for turn_deg, height, point_xy in stops:
            joints = self.__turned_joints(point_xy, turn_deg, height, near_wrist_rad = near_wrist_rad)
            plan.append(joints)
            near_wrist_rad = joints[4]
        return plan


    def __check_arm_model(self, joints: List[float]) -> None:
        """
        Once per episode, before the first forearm turn: the arm model and the robot's own kinematics must agree on
        where ``joints`` put the rod's end, within "turn_model_tolerance_mm"; nothing moves for the check.
        """

        if self.is_arm_model_checked:
            return
        model = numpy.array(ned2_fk.fk(list(joints))["tool"][0])
        robot = numpy.array(self.robot_link.predicted_position(list(joints)))
        error_mm = 1000.0 * float(numpy.linalg.norm(model - robot))
        if error_mm > self.recorder_parameters["turn_model_tolerance_mm"]:
            raise RuntimeError(f"the arm model and the robot disagree by {error_mm:.1f} mm on the turned position")
        self.is_arm_model_checked = True


    def __turn_back_lifted(self) -> None:
        """
        Lift at the current forearm turn and turn back to straight in increments; the arm stays lifted.

        The lean limit does not apply here: lifted, the rod touches nothing, and on 4.11.26 the limit refused the
        retreat at 38 cm (lifted lean 34.6 degrees) and the arm went to the parking pose from low down instead.
        """

        lift = self.recorder_parameters["turn_lift_m"]
        increment = self.recorder_parameters["turn_increment_deg"]
        turn_deg = self.forearm_turn_deg
        stops = [(turn_deg, height) for height in sorted(self.recorder_parameters["turn_near_paper_heights_m"]) if height < lift] + [(turn_deg, lift)]
        while abs(turn_deg) > 1e-9:
            turn_deg -= math.copysign(min(increment, abs(turn_deg)), turn_deg)
            stops.append((turn_deg, lift))
        plan = []
        near_wrist_rad = self.robot_link.read_state()["joints"][4]
        for turn_deg, height in stops:
            joints = self.__turned_joints(self.tip_xy, turn_deg, height, is_lean_limited = False, near_wrist_rad = near_wrist_rad)
            plan.append(joints)
            near_wrist_rad = joints[4]
        for joints in plan:
            self.robot_link.move_joints(joints)
        self.forearm_turn_deg = 0.0
        self.tip_xy = None


    def __is_step_over(self, action: str) -> bool:

        return action in self.recorder_parameters["step_over_actions"]


    def __step_over(self, target_pose: List[float]) -> None:
        """Lift, cross above the puck and come down on its other side, so the puck is not touched (Marshal, 4.11.26)."""

        height = self.recorder_parameters["approach_height_m"]
        self.__lift()
        self.robot_link.move_to([target_pose[0], target_pose[1], target_pose[2] + height] + target_pose[3:])
        self.robot_link.move_to(target_pose)


    def __handle_flag_key(
            self,
            action: Optional[str],
            picture_bytes: bytes,
            step: int
    ) -> bool:
        """
        Apply the "mistake" and "lost" keys, which flag a step instead of choosing one; True when ``action`` was one.
        """

        if action == "mistake":
            if self.current_rows:
                self.current_rows[-1]["is_mistake"] = 1
                self.__show_picture(picture_bytes, f"step {step}: step {step - 1} flagged as a mistake. Now choose the step.")
            return True
        if action == "lost":
            self.is_lost_pending = True
            self.__show_picture(picture_bytes, f"step {step}: flagged as line lost. Now choose the step.")
            return True
        return False


    def __see_puck(
            self,
            picture_bytes: bytes,
            state: Dict
    ) -> Optional[Dict]:
        """
        The red puck as the camera sees it now (puck_tracking.see_puck), remembered with the hash of the picture it
        belongs to, so that the picture window marks it on that picture only. For display only (Emranul, 5.11.26):
        nothing the robot does depends on it. On the simulated robot, whose joint angles are only approximate, the
        table position is left out; its cartoon puck, about three times the real puck's size in the picture, is not
        found by the detector anyway.
        """

        if not self.recorder_parameters["is_window_shown"]:
            return None
        try:
            picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
            sighting = self.puck_tracker.see_puck(picture, [float(value) for value in state["joints"]])
        except Exception as error:
            # The tracker only draws; a fault in it must never stop an episode, least of all with the motors off.
            self.__report_tracking_error(error)
            self.is_tracking_failing = True
            return None
        self.is_tracking_failing = False
        if sighting is None:
            return None
        sighting = dict(sighting, picture_hash = hash(picture_bytes))
        if self.recorder_parameters["is_simulated"]:
            sighting["table_xy"] = None
        return sighting


    def __picture_outline(self, state: Dict) -> Optional[numpy.ndarray]:
        """
        The picture's outline on the table from the calibrated camera (puck_tracking), or None when the camera is not
        calibrated for the mount in use, the robot is simulated (its joint angles are only approximate) or the tracker
        fails (display only, like the puck).
        """

        if self.recorder_parameters["is_simulated"]:
            return None
        try:
            return self.puck_tracker.picture_outline_on_table([float(value) for value in state["joints"]])
        except Exception as error:
            self.__report_tracking_error(error)
            return None


    def __report_tracking_error(self, error: Exception):
        """Say once per session that the puck tracking failed; the recording goes on without it."""

        if not self.is_tracking_error_reported:
            print(f"puck tracking failed ({type(error).__name__}: {error}); recording goes on without it")
            self.is_tracking_error_reported = True


    def __show_orientation(
            self,
            state: Dict,
            view: str = "keyboard",
            tip_xy: Optional[List[float]] = None
    ) -> None:
        """
        A second window, "orientation" (Emranul, 4.11.26: after a forearm turn the new orientation was not obvious
        from the picture alone), drawn by __orientation_canvas for ``view`` (a key of ORIENTATION_LEGENDS). It stays
        on top, and the first time it opens it is placed beside the picture window on the same monitor so that it
        does not hide behind it (Emranul, 5.11.26); after that it stays where the person puts it.
        """

        if not self.recorder_parameters["is_window_shown"]:
            return

        try:
            canvas = self.__orientation_canvas(state, view, tip_xy)
        except Exception as error:
            # Drawing the tracked puck and the picture's outline is display only, like the tracking itself (review of
            # 6.11.26: this code ran unguarded at every step): after a fault the map is drawn again without them.
            self.__report_tracking_error(error)
            canvas = self.__orientation_canvas(state, view, tip_xy, is_tracking_drawn = False)
        cv2.imshow(ORIENTATION_WINDOW_NAME, canvas)
        if not getattr(self, "is_orientation_window_placed", False):
            self.is_orientation_window_placed = True
            try:
                picture_x, picture_y, picture_width, picture_height = cv2.getWindowImageRect(WINDOW_NAME)
                canvas_height, canvas_width = canvas.shape[:2]
                window_x, window_y = picture_x + picture_width + 20, picture_y - 30
                area = monitor_work_area(picture_x + picture_width // 2, picture_y + picture_height // 2)
                if area is not None:
                    left, top, right, bottom = area
                    if window_x + canvas_width > right:
                        window_x = picture_x - canvas_width - 20
                    if window_x < left:
                        window_x = right - canvas_width
                    window_x = max(left, window_x)
                    window_y = max(top, min(window_y, bottom - canvas_height - 40))
                cv2.moveWindow(ORIENTATION_WINDOW_NAME, int(window_x), int(window_y))
            except (cv2.error, OSError, ValueError):
                pass
        try:
            cv2.setWindowProperty(ORIENTATION_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass


    def __orientation_canvas(
            self,
            state: Dict,
            view: str = "keyboard",
            tip_xy: Optional[List[float]] = None,
            is_tracking_drawn: bool = True
    ) -> numpy.ndarray:
        """
        The orientation window's picture, made to be read at a glance (Emranul, 5.11.26).

        A title names the mode. The map shows the table from above, zoomed on the sheet: the paper, the robot's base
        at the bottom, the arm as a thick line from the base to the tip, the puck, the table-edge guard as a dashed
        line, and in green the part of the table the picture shows, with "TOP" at its top edge. With the camera
        calibrated for the mount in use (puck_tracking, since 6.11.26) the green area is the picture's real outline
        and the puck is drawn solid where the camera sees it (paler when only a sliver shows, whose place is rough);
        otherwise the green frame has a rough size. The tool's own belief of where the puck is (puck_xy_estimate),
        drawn hollow as "puck?", is shown where it is used: in the keyboard and 1 cm replay views, where a forearm
        turn lands back from it, whenever the camera cannot place the puck and also beside the seen puck once the two
        lie further apart than "puck_guess_shown_beyond_m"; in the gamepad view only when the camera cannot place the
        puck; never in the exact replay (which does not move it) or while guiding. Arrows from the tip show where each
        key or the left stick moves the tip now (none in the exact replay and while guiding, where the hand decides).
        Below the map the camera's turn and what the map shows of the puck are written in words, and the view's
        legend (ORIENTATION_LEGENDS) lists every key or button.

        Parameters
        ----------
        state : dict
            The robot's state now.
        view : str
            "keyboard", "gamepad", "replay", "exact_replay" or "guiding".
        tip_xy : list of float, optional
            The tip's position when known better than the recorder's own belief (while guiding by hand).
        is_tracking_drawn : bool
            False to leave out the tracked puck and the calibrated outline (after drawing them failed).

        Returns
        -------
        numpy.ndarray
            The picture, 480 pixels wide.
        """

        settings = self.recorder_parameters
        legend = ORIENTATION_LEGENDS[view]
        is_simulated = settings["is_simulated"]
        forearm_deg = self.forearm_turn_deg if is_simulated else math.degrees(float(state["joints"][3]))
        outline = self.__picture_outline(state) if is_tracking_drawn else None
        sighting = getattr(self, "puck_sighting", None) if is_tracking_drawn else None
        seen_xy = sighting.get("table_xy") if sighting is not None else None

        # The tool's belief of where the puck is, where it is used (see the docstring; review of 6.11.26: the seen puck
        # had hidden the belief that forearm turns still land from).
        guess_xy = getattr(self, "puck_xy_estimate", None)
        guess_gap_m = None
        if guess_xy is not None and seen_xy is not None:
            guess_gap_m = float(numpy.linalg.norm(numpy.asarray(guess_xy, dtype = float) - numpy.asarray(seen_xy, dtype = float)))
        if guess_xy is None or view in ("exact_replay", "guiding"):
            is_guess_drawn = False
        elif seen_xy is None:
            is_guess_drawn = True
        else:
            is_guess_drawn = view in ("keyboard", "replay") and guess_gap_m > settings["puck_guess_shown_beyond_m"]

        # What the map shows of the puck, from what actually happened (review of 6.11.26: the line had given wrong
        # reasons in several cases). The detector misses a puck held in a hand, hence "not found", not "not there".
        # After the camera check found the camera moved (and the person chose to record anyway) the calibrated map is
        # only rough: about 2 mm off at 4 px for a small creep, a centimetre or more for a re-taped camera (review of
        # 6.11.26; the fit holds while the check stays under "camera_shift_warning_px").
        is_camera_moved = outline is not None and getattr(self, "camera_shift_px", 0.0) > settings["camera_shift_warning_px"]
        is_rough = seen_xy is not None and bool(sighting.get("is_position_rough"))
        if seen_xy is not None and is_guess_drawn:
            puck_text = f"{'partly hidden' if is_rough else 'seen'} (solid); v/b turns land by the hollow guess, {100 * guess_gap_m:.1f} cm off"
        elif seen_xy is not None and is_camera_moved:
            puck_text = f"{'partly hidden' if is_rough else 'seen'}, drawn roughly (the camera moved)"
        elif seen_xy is not None:
            puck_text = "partly hidden, drawn roughly where it is" if is_rough else "seen, drawn where it really is"
        elif sighting is not None:
            puck_text = "seen, not placed (simulated arm)" if is_simulated else "seen, not placed (camera not calibrated)"
        elif not is_tracking_drawn or getattr(self, "is_tracking_failing", False):
            puck_text = "tracking failed (see the terminal)"
        elif is_simulated:
            puck_text = "not tracked on the simulated robot"
        else:
            puck_text = "not found in the picture"
        if is_guess_drawn and seen_xy is None:
            puck_text += "; hollow = the tool's guess"

        # The turn is named by its key, not as left or right: a v turn swings the picture's top to the right on the
        # table, which "left" would contradict.
        info_lines = ["Forearm straight: the picture faces away from the robot" if abs(forearm_deg) < 1.0 else
                      f"Forearm turned {abs(forearm_deg):.0f} deg, the {'v' if forearm_deg > 0 else 'b'} way",
                      ("Green = what the picture shows on the table, TOP = its top" if not is_camera_moved else
                       f"Green = the picture on the table, TOP = its top; rough: camera moved {self.camera_shift_px:.0f} px")
                      if outline is not None else "Green frame = the picture on the table (rough size), TOP = its top",
                      f"Puck: {puck_text}"]
        if view == "gamepad":
            height_line = f"Tip lifted {100 * getattr(self, 'pad_height_m', 0.0):.1f} cm"
            if not is_simulated:
                height_line += f", rod leans {magnet_tip_and_lean([float(value) for value in state['joints']])[1]:.0f} deg"
            info_lines.append(height_line)

        width, header_height, map_height = 480, 34, 440
        info_top = header_height + map_height
        legend_top = info_top + 12 + 22 * len(info_lines)
        canvas = numpy.full((legend_top + 14 + 24 * len(legend), width, 3), 250, dtype = numpy.uint8)

        # The map: 1050 pixels per metre, the tip of the table at the top, x = 4 cm from the base's centre at the
        # bottom edge (the base shows as a cap), centred sideways on the workspace.
        scale, map_centre_y, map_bottom_x = 1050.0, -0.015, 0.04
        table = numpy.full((map_height, width, 3), 214, dtype = numpy.uint8)

        def pixel(point_xy) -> tuple:
            return (int(round(width / 2 - (point_xy[1] - map_centre_y) * scale)), int(round(map_height - (point_xy[0] - map_bottom_x) * scale)))

        def put_centred(image: numpy.ndarray, label: str, at: tuple, colour: tuple, font_scale: float, thickness: int) -> None:
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
            cv2.putText(image, label, (at[0] - text_width // 2, at[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness)

        def pulled_onto_map(at: tuple, start: tuple, margin_px: int = 16) -> tuple:
            # A point beyond the map is pulled back along the line from ``start`` until it lies on the map, so a picture
            # whose top looks far across the table still shows which way its top is.
            start_point, end_point = numpy.array(start, dtype = float), numpy.array(at, dtype = float)
            low, high = numpy.array([margin_px, margin_px], dtype = float), numpy.array([width - margin_px, map_height - margin_px], dtype = float)
            fraction = 1.0
            for axis in range(2):
                change = end_point[axis] - start_point[axis]
                if change > 0 and end_point[axis] > high[axis]:
                    fraction = min(fraction, (high[axis] - start_point[axis]) / change)
                elif change < 0 and end_point[axis] < low[axis]:
                    fraction = min(fraction, (low[axis] - start_point[axis]) / change)
            return tuple(int(round(value)) for value in start_point + max(fraction, 0.0) * (end_point - start_point))

        def middle_on_map(points: numpy.ndarray, minimum_length_px: float = 40.0) -> Optional[numpy.ndarray]:
            # The middle, by length, of the part of the line through ``points`` that lies on the map; None when less
            # than ``minimum_length_px`` of it does (a sliver in a corner of the map would carry the label there).
            pieces = []
            for start_point, end_point in zip(points[:-1], points[1:]):
                is_on_map, clipped_start, clipped_end = cv2.clipLine((0, 0, width, map_height), tuple(int(value) for value in start_point),
                                                                     tuple(int(value) for value in end_point))
                if is_on_map:
                    pieces.append((numpy.array(clipped_start, dtype = float), numpy.array(clipped_end, dtype = float)))
            lengths = [float(numpy.linalg.norm(end_point - start_point)) for start_point, end_point in pieces]
            if sum(lengths) < minimum_length_px:
                return None
            remaining = sum(lengths) / 2.0
            for (start_point, end_point), length in zip(pieces, lengths):
                if remaining <= length:
                    return start_point + (end_point - start_point) * (remaining / length)
                remaining -= length
            return pieces[-1][1]

        sheet = numpy.array([pixel(corner) for corner in WORKSPACE_PARAMETERS["sheet_corners_xy"]], dtype = numpy.int32)
        cv2.fillPoly(table, [sheet], (255, 255, 255))
        cv2.polylines(table, [sheet], isClosed = True, color = (150, 150, 150), thickness = 1)
        box_x, box_y = WORKSPACE_PARAMETERS["box_x"], WORKSPACE_PARAMETERS["box_y"]
        guard = [pixel([box_x[1], box_y[1]]), pixel([box_x[1], box_y[0]]), pixel([box_x[0], box_y[0]]), pixel([box_x[0], box_y[1]])]
        for corner_index in range(4):
            start_pixel, end_pixel = numpy.array(guard[corner_index]), numpy.array(guard[(corner_index + 1) % 4])
            dashes = max(1, int(numpy.linalg.norm(end_pixel - start_pixel) // 8))
            for dash in range(0, dashes, 2):
                cv2.line(table, tuple(int(value) for value in start_pixel + (end_pixel - start_pixel) * dash / dashes),
                         tuple(int(value) for value in start_pixel + (end_pixel - start_pixel) * (dash + 1) / dashes), (60, 140, 230), 1)
        cv2.putText(table, "table-edge guard", (guard[0][0] + 4, guard[0][1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 140, 230), 1)

        if tip_xy is None:
            is_tip_known = self.tip_xy is not None and (self.forearm_turn_deg or view == "gamepad")
            tip_xy = self.tip_xy if is_tip_known else state["pose"][:2]
        tip = numpy.array(tip_xy, dtype = float)[:2]
        cv2.line(table, pixel([0.0, 0.0]), pixel(tip), (205, 185, 160), 12)
        cv2.circle(table, pixel([0.0, 0.0]), int(0.07 * scale), (170, 130, 80), -1)
        put_centred(table, "ROBOT BASE", (pixel([0.0, 0.0])[0], map_height - 14), (255, 255, 255), 0.5, 1)

        up_xy, left_xy = picture_axes_on_table(state, settings)
        if outline is not None:
            # The calibrated camera's outline, walked clockwise from the picture's top-left corner: its first
            # points_per_edge + 1 points run along the picture's top edge to the top-right corner.
            outline_xy = numpy.asarray(outline, dtype = float)
            points_per_edge = max(1, len(outline_xy) // 4)
        else:
            frame_half_width_m, frame_depth_m = 0.05, 0.075
            outline_xy = numpy.array([tip + left_xy * frame_half_width_m + up_xy * frame_depth_m, tip - left_xy * frame_half_width_m + up_xy * frame_depth_m,
                                      tip - left_xy * frame_half_width_m, tip + left_xy * frame_half_width_m])
            points_per_edge = 1
        frame = numpy.clip(numpy.array([pixel(corner) for corner in outline_xy]), -10000, 10000).astype(numpy.int32)
        shaded = table.copy()
        cv2.fillPoly(shaded, [frame], (190, 235, 190))
        table = cv2.addWeighted(shaded, 0.55, table, 0.45, 0)
        cv2.polylines(table, [frame], isClosed = True, color = (90, 170, 90), thickness = 1)
        top_edge = frame[:points_per_edge + 1]
        cv2.polylines(table, [top_edge], isClosed = False, color = (30, 140, 30), thickness = 3)
        # "TOP" just beyond the middle of the part of the top edge that lies on the map (review of 6.11.26: with the
        # forearm turned, the whole edge's middle can lie far off the map); when no part of it does, the label is
        # pulled back along the line from the tip, so it still shows which way the top is.
        top_middle = middle_on_map(top_edge)
        if top_middle is not None:
            # Far enough out along the picture's up that the label's box clears the line whichever way up points.
            up_on_map = numpy.array(pixel(tip + up_xy * 0.01), dtype = float) - numpy.array(pixel(tip), dtype = float)
            up_on_map /= max(float(numpy.linalg.norm(up_on_map)), 1e-9)
            label_width, label_height = cv2.getTextSize("TOP", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            clearance_px = 6 + label_width / 2 * abs(up_on_map[0]) + label_height / 2 * abs(up_on_map[1])
            top_label_at = pulled_onto_map(tuple(top_middle + clearance_px * up_on_map), tuple(top_middle))
        else:
            top_label_at = pulled_onto_map(pixel((outline_xy[0] + outline_xy[points_per_edge]) / 2 + up_xy * 0.012), pixel(tip))
        put_centred(table, "TOP", top_label_at, (30, 140, 30), 0.55, 2)

        puck_radius_px = int(PUCK_PARAMETERS["diameter_m"] / 2 * scale)
        if is_guess_drawn:
            # The tool's own guess from the steps taken: drawn hollow, because it drifts.
            cv2.circle(table, pixel(guess_xy), puck_radius_px, (60, 60, 210), 2)
            cv2.putText(table, "puck?", (pixel(guess_xy)[0] + 14, pixel(guess_xy)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 210), 1)
        if seen_xy is not None:
            # Paler when only a sliver shows and no circle fitted its outline: its place is rough.
            fill_colour = (130, 130, 235) if sighting.get("is_position_rough") else (40, 40, 225)
            cv2.circle(table, pixel(seen_xy), puck_radius_px, fill_colour, -1)
            cv2.circle(table, pixel(seen_xy), puck_radius_px, (20, 20, 120), 1)
            cv2.putText(table, "puck", (pixel(seen_xy)[0] + 15, pixel(seen_xy)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 160), 1)

        arrows = []
        if view == "gamepad":
            arrows = [("UP", up_xy), ("DOWN", -up_xy), ("LEFT", left_xy), ("RIGHT", -left_xy)]
        elif view in ("keyboard", "replay"):
            key_of = {action: key for key, action in settings["keys"].items()}
            heading_rad = picture_heading_rad(state, settings)
            for action, angle in settings["actions"].items():
                if angle is None or action not in key_of:
                    continue
                step_x, step_y = action_step_to_table(action, state["pose"], settings["hand_turn_deg"], heading_rad = heading_rad)
                arrows.append((key_of[action], numpy.array([step_x, step_y]) / math.hypot(step_x, step_y)))
        for label, direction in arrows:
            is_forward = label in ("UP", "w")
            colour = (30, 140, 30) if is_forward else (70, 70, 70)
            cv2.arrowedLine(table, pixel(tip), pixel(tip + 0.05 * direction), colour, 2, tipLength = 0.22)
            put_centred(table, label, pixel(tip + (0.066 if len(label) > 1 else 0.06) * direction), colour, 0.5 if len(label) > 1 else 0.6, 2)
        cv2.circle(table, pixel(tip), 6, (20, 20, 20), -1)
        canvas[header_height:info_top] = table

        cv2.putText(canvas, f"{ORIENTATION_TITLES[view]}: the table from above", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        for index, line in enumerate(info_lines):
            cv2.putText(canvas, line, (8, info_top + 24 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if index == 0 else 0.48,
                        (20, 20, 20), 2 if index == 0 else 1)
        cv2.line(canvas, (0, legend_top), (width, legend_top), (180, 180, 180), 1)
        for index, (control, meaning, colour) in enumerate(legend):
            line_y = legend_top + 26 + 24 * index
            cv2.putText(canvas, control, (8, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2 if colour != LEGEND_TEXT_COLOUR else 1)
            cv2.putText(canvas, meaning, (178, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        return canvas


    def __keyboard_decider(
            self,
            picture_bytes: bytes,
            state: Dict,
            step: int
    ) -> Tuple[str, str]:
        """The person chooses the step from the picture on screen."""

        turned = f" (forearm turned {self.forearm_turn_deg:+.0f})" if self.forearm_turn_deg else ""
        self.__show_orientation(state, view = "keyboard")
        self.__show_picture(picture_bytes, f"step {step}{turned}: q w e / a s d / y c = steps, v b = forearm turn, Space = stop, u l x Esc")
        self.__drain_keys(step)
        while True:
            key = self.key_reader(0)
            if key == "escape":
                return "quit", "human"
            action = self.recorder_parameters["keys"].get(key)
            if self.__handle_flag_key(action, picture_bytes, step):
                continue
            if self.__is_executable(action):
                return action, "human"


    def __make_replay_decider(self, guided_path: numpy.ndarray) -> Callable[[bytes, Dict, int], Tuple[str, str]]:
        """
        A decider that follows the hand-guided path, snapped back onto it at every step, and lets the person override.

        The proposed step runs only when Enter is pressed (decision of 31.10.26: a 400 ms window was shorter than a
        considered seven-way choice, so overrides landed a step late); a direction key overrides, there is no timeout.

        Parameters
        ----------
        guided_path : numpy.ndarray
            The logged tip positions and forearm angles, shape (points, 3): x and y in metres, joint 4 in radians.
            Where the guided forearm angle, rounded to the nearest turn step, differs from the current turn, the
            proposal is a turn towards it (4.11.26, Emranul: the replay did not copy the turns made by hand).
        """

        path_xy = guided_path[:, :2]
        path_forearm_rad = guided_path[:, 2] if guided_path.shape[1] > 2 else numpy.zeros(len(guided_path))
        segment_lengths = numpy.linalg.norm(numpy.diff(path_xy, axis = 0), axis = 1)
        arc_lengths = numpy.concatenate([[0.0], numpy.cumsum(segment_lengths)])
        progress = {"segment": 0}
        step_m = self.recorder_parameters["step_m"]
        moving_actions = [action for action, angle in self.recorder_parameters["actions"].items() if angle is not None]
        turn_step_deg = abs(self.recorder_parameters["turn_actions"]["turn_left"])

        def replay_decider(
                picture_bytes: bytes,
                state: Dict,
                step: int
        ) -> Tuple[str, str]:

            position = numpy.array(self.tip_xy if self.forearm_turn_deg and self.tip_xy is not None else state["pose"][:2])
            arc_now = self.__project_onto_path(position, path_xy, arc_lengths, progress)
            # The turn follows the furthest point reached along the guided path: the landing after a turn steps the
            # tip back, and the projection alone would then ask to turn back again.
            progress["furthest_arc"] = max(progress.get("furthest_arc", 0.0), arc_now)
            guided_forearm_deg = math.degrees(float(numpy.interp(progress["furthest_arc"], arc_lengths, path_forearm_rad)))
            wanted_turn_deg = max(self.recorder_parameters["min_turn_deg"], min(self.recorder_parameters["max_turn_deg"],
                                  turn_step_deg * round(guided_forearm_deg / turn_step_deg)))

            if abs(wanted_turn_deg - self.forearm_turn_deg) > 1e-9:
                proposal = "turn_left" if wanted_turn_deg > self.forearm_turn_deg else "turn_right"
            elif arc_lengths[-1] - arc_now < step_m / 2:
                proposal = "stop"
            else:
                target = self.__point_at_arc_length(path_xy, arc_lengths, arc_now + step_m)
                proposal = min(moving_actions, key = lambda action: numpy.linalg.norm(
                    position + numpy.array(action_step_to_table(action, state["pose"], self.recorder_parameters["hand_turn_deg"],
                                                                heading_rad = picture_heading_rad(state, self.recorder_parameters))) - target))

            self.__show_orientation(state, view = "replay")
            print(f"[step {step}] replay proposes {proposal} (guided forearm {guided_forearm_deg:+.0f} deg, now {self.forearm_turn_deg:+.0f} deg)")
            self.__show_picture(picture_bytes, f"replay step {step}: {proposal}. Enter = do it, direction key = override, u = mistake, l = lost, x = discard")
            self.__drain_keys(step)
            while True:
                key = self.key_reader(0)
                if key == "escape":
                    return "quit", "human"
                action = self.recorder_parameters["keys"].get(key)
                if self.__handle_flag_key(action, picture_bytes, step):
                    continue
                if action == "accept":
                    return proposal, "replay"
                if self.__is_executable(action):
                    return action, "human"

        return replay_decider


    def __is_executable(self, action: Optional[str]) -> bool:
        """True for a key that ends the wait for a decision: a class, a step-over, a lean or turn key, or discard."""

        return (action in self.recorder_parameters["actions"]
                or action in self.recorder_parameters["flick_actions"]
                or action in self.recorder_parameters["step_over_actions"]
                or action in self.recorder_parameters["lean_actions"]
                or action in self.recorder_parameters["turn_actions"]
                or action == "discard")


    @staticmethod
    def __project_onto_path(
            position: numpy.ndarray,
            path: numpy.ndarray,
            arc_lengths: numpy.ndarray,
            progress: Dict
    ) -> float:
        """
        The arc length of the path point closest to ``position``, searching forward from the last match only, so
        that a path crossing itself (a loop) is not jumped along.
        """

        best_distance, best_arc = numpy.inf, arc_lengths[progress["segment"]]
        last_segment = min(len(path) - 1, progress["segment"] + 50)
        for segment in range(progress["segment"], last_segment):
            start, end = path[segment], path[segment + 1]
            length_squared = float((end - start) @ (end - start))
            fraction = 0.0 if length_squared == 0 else float(numpy.clip((position - start) @ (end - start) / length_squared, 0, 1))
            distance = numpy.linalg.norm(start + fraction * (end - start) - position)
            if distance < best_distance:
                best_distance = distance
                best_arc = arc_lengths[segment] + fraction * (arc_lengths[segment + 1] - arc_lengths[segment])
                progress["segment"] = segment

        return best_arc


    @staticmethod
    def __point_at_arc_length(
            path: numpy.ndarray,
            arc_lengths: numpy.ndarray,
            arc_length: float
    ) -> numpy.ndarray:
        """The point ``arc_length`` metres along the path, clipped to its end."""

        arc_length = min(arc_length, arc_lengths[-1])
        return numpy.array([numpy.interp(arc_length, arc_lengths, path[:, 0]), numpy.interp(arc_length, arc_lengths, path[:, 1])])


    def __record_guided_path(
            self,
            episode_folder: Path,
            file_name: str = "guided_path.csv"
    ) -> Optional[numpy.ndarray]:
        """
        Let the person push the puck to the goal by hand while the tip's path is logged; None when discarded.
        """

        self.__show_message("Hold the arm at the elbow. Enter = motors OFF (the arm will sag), x = discard.")
        if self.__wait_for(["enter", "x"]) == "x":
            return None
        self.__set_free_motion(True)

        rows = []
        is_discarded = False
        try:
            while True:
                state = self.robot_link.read_state()
                # The magnet tip from the arm model on the robot (the rod may lean when the forearm is twisted by
                # hand); the rod's end on the simulated robot, whose joint angles are only approximate.
                if self.recorder_parameters["is_simulated"]:
                    position = list(state["pose"][:2])
                else:
                    position = [float(value) for value in magnet_tip_and_lean(state["joints"])[0][:2]]
                joints = [float(value) for value in state["joints"]]
                # A point when the tip has moved, or when any joint has turned by 5 degrees in place (4.11.26, Emranul:
                # the replay did not copy the forearm turns made by hand; the exact replay copies every joint).
                if (not rows or numpy.hypot(position[0] - rows[-1]["x"], position[1] - rows[-1]["y"]) >= self.recorder_parameters["guided_path_sample_m"]
                        or max(abs(joint - rows[-1][f"joint_{index + 1}"]) for index, joint in enumerate(joints)) >= math.radians(5.0)):
                    row = {"unix_time": round(time.time(), 3), "x": position[0], "y": position[1], "z": state["pose"][2],
                           "roll": state["pose"][3], "pitch": state["pose"][4], "yaw": state["pose"][5]}
                    row.update({f"joint_{index + 1}": joint for index, joint in enumerate(joints)})
                    rows.append(row)

                picture_bytes = self.robot_link.read_picture_bytes()
                self.puck_sighting = self.__see_puck(picture_bytes, state)
                self.__show_orientation(state, view = "guiding", tip_xy = position)
                self.__show_picture(picture_bytes, f"MOTORS OFF, guiding: {len(rows)} points. Space = puck at goal, x = discard")
                key = self.key_reader(int(self.recorder_parameters["guided_path_poll_s"] * 1000))
                if key in (" ", "x", "escape"):
                    is_discarded = key != " "
                    break
        finally:
            # Whatever ended the guiding (Space, x, Ctrl-C, an error), the motors come back on only after the person
            # has let go and said so; the arm must never be powered against a holding hand.
            self.__release_arm_then_motors_on()

        with open(episode_folder / file_name, "w", newline = "", encoding = "utf-8") as file:
            writer = csv.DictWriter(file, fieldnames = ["unix_time", "x", "y", "z", "roll", "pitch", "yaw"] + [f"joint_{index}" for index in range(1, 7)])
            writer.writeheader()
            writer.writerows(rows)

        if is_discarded or len(rows) < 2:
            return None
        return numpy.array([[row["x"], row["y"]] + [row[f"joint_{index}"] for index in range(1, 7)] for row in rows])


    def __fresh_picture(self) -> bytes:
        """
        Wait for ``steps_until_fresh_picture`` new pictures after the move returned and take the last one, so no
        picture taken during the move is logged.
        """

        previous = self.robot_link.read_picture_bytes()
        new_pictures = 0
        deadline = time.monotonic() + self.recorder_parameters["fresh_picture_timeout_s"]
        while new_pictures < self.recorder_parameters["steps_until_fresh_picture"]:
            if time.monotonic() > deadline:
                raise RuntimeError("no new camera picture arrived; is the camera plugged in and streaming?")
            time.sleep(0.01)
            picture_bytes = self.robot_link.read_picture_bytes()
            if picture_bytes != previous:
                new_pictures += 1
                previous = picture_bytes

        return previous


    def __log_step(
            self,
            episode_folder: Path,
            step: int,
            picture_bytes: bytes,
            state: Dict,
            action: str,
            source: str,
            times: Dict[str, float]
    ) -> Dict:
        """Write the picture exactly as the robot sent it (no re-compression) and return the step's row."""

        frame_name = f"{step:03d}.jpg"
        (episode_folder / "frames" / frame_name).write_bytes(picture_bytes)
        picture_md5 = hashlib.md5(picture_bytes).hexdigest()
        if picture_md5 == self.last_picture_md5:
            print(f"warning: step {step} has the same picture as the step before")
        self.last_picture_md5 = picture_md5

        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        row = {"step": step, "unix_time": round(time.time(), 3), "frame": f"frames/{frame_name}", "frame_md5": picture_md5,
               "sharpness": round(sharpness_of(picture), 1),
               "action": action, "source": source, "is_mistake": 0, "is_lost": int(self.is_lost_pending),
               "forearm_turn_deg": float(self.forearm_turn_deg),
               # The table direction the picture's top pointed to (degrees from the table's x axis, counter-clockwise):
               # the key's table direction is this plus the action's angle. Not a model input (the camera is the only
               # input); for checking and relabelling the data, and the controller reads it live from the robot.
               "picture_heading_deg": round(math.degrees(picture_heading_rad(state, self.recorder_parameters)), 2)}
        row.update({name: round(value, 3) for name, value in times.items()})
        row.update({name: round(value, 5) for name, value in zip(["x", "y", "z", "roll", "pitch", "yaw"], state["pose"])})
        row.update({f"joint_{index + 1}": round(value, 5) for index, value in enumerate(state["joints"])})
        self.is_lost_pending = False
        # One terminal line per logged step, with the same number as the frame and the row in steps.csv.
        print(f"[step {step}] {action} ({source}), forearm {self.forearm_turn_deg:+.0f} deg, rod at x {100 * state['pose'][0]:.1f} "
              f"y {100 * state['pose'][1]:.1f} cm, picture top {row['picture_heading_deg']:+.0f} deg")
        return row


    def __drain_keys(self, step: int) -> None:
        """Throw away keys pressed during the move and say so, so a double press never labels an unseen picture."""

        drained = self.key_drainer()
        if drained:
            print(f"[step {step}] {drained} key press(es) made during the move were ignored")


    def __set_free_motion(self, is_enabled: bool) -> None:
        """Switch the motors off or on and remember the state, so the shutdown path knows whether a hand may be on the arm."""

        self.robot_link.set_free_motion(is_enabled)
        self.is_free_motion = is_enabled


    def __release_arm_then_motors_on(self) -> None:
        """Keep the motors off until the person confirms the arm is released, then switch them back on."""

        if not self.is_free_motion:
            return
        self.__show_message("MOTORS OFF. Let go of the arm completely, then press Enter = motors back ON.")
        self.__wait_for(["enter"])
        self.__set_free_motion(False)
        print("motors ON")


    @staticmethod
    def __write_steps(
            episode_folder: Path,
            rows: List[Dict],
            file_name: str
    ) -> None:
        """Write the step rows to ``file_name`` in the episode folder."""

        with open(episode_folder / file_name, "w", newline = "", encoding = "utf-8") as file:
            writer = csv.DictWriter(file, fieldnames = STEP_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)


    def __finish_episode(
            self,
            episode_folder: Path,
            summary: Dict,
            rows: List[Dict]
    ) -> Dict:
        """Write steps.csv and episode.json and a line in the session notes."""

        goal_labels(rows, self.recorder_parameters)
        self.__write_steps(episode_folder, rows, "steps.csv")
        (episode_folder / "steps_in_progress.csv").unlink(missing_ok = True)

        human_steps = sum(1 for row in rows if row["source"] == "human")
        summary.update({
            "finished": datetime.datetime.now().isoformat(timespec = "seconds"),
            "steps": len(rows),
            "human_steps": human_steps,
            "mistake_steps": sum(row["is_mistake"] for row in rows),
            "lost_steps": sum(row["is_lost"] for row in rows)
        })
        (episode_folder / "episode.json").write_text(json.dumps(summary, indent = 2), encoding = "utf-8")

        overrides = f", {human_steps} overridden by hand" if summary["mode"] == "hand_guided" else ""
        ending = summary["outcome"] or summary["discard_reason"]
        ending = f", {ending}" if ending else ""
        self.__append_session_notes(f"- {episode_folder.name}: {summary['status']}{ending}, {len(rows)} steps{overrides}\n")
        print(f"{episode_folder.name}: {summary['status']}{ending}, {len(rows)} steps")
        return summary


    def __simulate_puck_reset(self) -> None:
        """On the simulated robot, do what the person does by hand on the real one: put the puck at the start."""

        if hasattr(self.robot_link, "place_puck_at_start"):
            self.robot_link.place_puck_at_start()


    def __start_pose(self) -> List[float]:
        """
        The rod's pose behind the puck at the start point: opposite to the first push ("start_push_direction_deg"),
        the hand turned by "hand_turn_deg" so the camera sits beside the rod (4.11.26; until then the rod stood on
        the camera's side of the puck, which was the same thing while the camera looked along the push).
        """

        puck_x, puck_y = self.recorder_parameters["start_point_xy"]
        push = math.radians(self.recorder_parameters["start_push_direction_deg"])
        offset = self.recorder_parameters["puck_centre_ahead_of_tip_m"]
        x = puck_x - offset * math.cos(push)
        y = puck_y - offset * math.sin(push)
        return [x, y, pushing_height(x, y)] + self.__turned_orientation(self.recorder_parameters["hand_turn_deg"])


    def __lower_to_start(self) -> None:
        """
        Come down onto the start pose from above, so the pusher does not sweep across the sheet.

        Close to the base the robot can refuse a straight-line descent ("cannot be reached with a linear trajectory",
        seen on 31.10.26 at x = 0.12 m); the joint-wise descent is then used, a slight arc over 5 cm before any contact.
        """

        start = self.__start_pose()
        above = list(start)
        above[2] += self.recorder_parameters["approach_height_m"]
        self.robot_link.move_to(above, is_linear = False)
        try:
            self.robot_link.move_to(start)
        except Exception as error:
            print(f"straight descent refused ({error}); descending joint-wise instead")
            self.robot_link.move_to(start, is_linear = False)


    def __lift(self) -> None:

        if self.forearm_turn_deg:
            # Lift first so the tip cannot sweep the puck, then turn the forearm straight again (Marshal's retreat).
            try:
                self.__turn_back_lifted()
                return
            except Exception as error:
                print(f"turning the forearm back failed ({str(error).splitlines()[0]}); going to the parking pose instead")
                self.forearm_turn_deg = 0.0
                self.tip_xy = None
                self.robot_link.move_home(self.recorder_parameters["home_joints"])
                return

        state = self.robot_link.read_state()
        pose = list(state["pose"])
        pose[2] += self.recorder_parameters["approach_height_m"]
        try:
            self.robot_link.move_to(pose)
        except Exception as error:
            # 6.11.26: refused after every hand correction (the arm leans where the hand left it); lifting in joint
            # space first saves the trip to the parking pose and back.
            try:
                self.robot_link.move_joints(self.__joints_lifted(list(state["joints"])))
                print(f"straight lift refused ({str(error).splitlines()[0]}); lifted with joints 2 and 3 instead")
                return
            except Exception:
                pass
            # 3.11.26: after a refused start the arm was still at the parking pose, 5 cm above which is out of reach;
            # the tool crashed here. Going to the parking pose is the safe alternative to lifting.
            print(f"lift refused ({str(error).splitlines()[0]}); going to the parking pose instead")
            self.robot_link.move_home(self.recorder_parameters["home_joints"])


    def __joints_lifted(self, joints: List[float]) -> List[float]:
        """``joints`` with joints 2 and 3 changed so the tip rises about 4 cm nearly straight up ("joint_lift_change_rad")."""

        lifted = [float(value) for value in joints]
        for name, change in self.recorder_parameters["joint_lift_change_rad"].items():
            lifted[int(name.split("_")[1]) - 1] += change
        return lifted


    @staticmethod
    def __is_inside_workspace(
            x: float,
            y: float
    ) -> bool:

        box_x, box_y = WORKSPACE_PARAMETERS["box_x"], WORKSPACE_PARAMETERS["box_y"]
        return box_x[0] <= x <= box_x[1] and box_y[0] <= y <= box_y[1]


    def __is_step_allowed(
            self,
            target_x: float,
            target_y: float,
            step_x: float,
            step_y: float
    ) -> bool:
        """
        True when the rod's target and the puck's predicted centre both stay inside the workspace box and the puck
        stays within the sheet plus its margin.

        The puck's centre is taken as "puck_centre_ahead_of_tip_m" beyond the rod along the step; the box's far edge
        is the table edge, so checking the rod alone would let a forward key push the puck off the table.
        """

        if not self.__is_inside_workspace(target_x, target_y):
            return False

        step_length = numpy.hypot(step_x, step_y)
        if step_length < 1e-9:
            return True
        ahead = self.recorder_parameters["puck_centre_ahead_of_tip_m"]
        puck_x = target_x + ahead * step_x / step_length
        puck_y = target_y + ahead * step_y / step_length
        if not self.__is_inside_workspace(puck_x, puck_y):
            return False

        corners = numpy.array(WORKSPACE_PARAMETERS["sheet_corners_xy"])
        margin = self.recorder_parameters["sheet_margin_m"]
        return (corners[:, 0].min() - margin <= puck_x <= corners[:, 0].max() + margin
                and corners[:, 1].min() - margin <= puck_y <= corners[:, 1].max() + margin)


    def __open_session_folder(self) -> Path:
        """
        Today's latest session folder of this operator, or a new one when there is none yet or a new one was asked for.

        The operator is part of the name, so two laptops recording on the same day never write into folders of the same
        name when the recordings are merged.
        """

        raw_folder = PROJECT_FOLDER / self.recorder_parameters["raw_folder"]
        today = datetime.date.today().isoformat()
        operator = re.sub(r"[^a-z0-9]+", "_", self.operator.lower()).strip("_")
        numbers = [int(match.group(1)) for folder in raw_folder.glob(f"{today}_session_*_{operator}")
                   if (match := re.fullmatch(rf"{today}_session_(\d+)_{re.escape(operator)}", folder.name))]

        number = max(numbers, default = 0)
        if number == 0 or self.recorder_parameters["is_new_session"]:
            number += 1
        folder = raw_folder / f"{today}_session_{number:02d}_{operator}"
        folder.mkdir(parents = True, exist_ok = True)
        return folder


    def __next_episode_folder(self) -> Path:
        """
        Episode numbers count up per path_id across every session on this laptop (README rule 4). With a plan number
        the name also says which plan episode this is and the how-many-th try of it (Emranul, 6.11.26), for example
        path_01_straight_line_episode_004_plan_01-01_try_2.
        """

        raw_folder = PROJECT_FOLDER / self.recorder_parameters["raw_folder"]
        name_pattern = rf"{re.escape(self.path_id)}_episode_(\d+)(?:_plan_(\d\d-\d\d)_try_\d+)?(?:_score_\d\d)?"
        folders = [(folder, match) for folder in raw_folder.glob(f"*/{self.path_id}_episode_*")
                   if (match := re.fullmatch(name_pattern, folder.name))]
        name = f"{self.path_id}_episode_{max((int(match.group(1)) for _, match in folders), default = 0) + 1:03d}"
        plan_number = self.recorder_parameters["plan_number"]
        if plan_number:
            # Folders named before 6.11.26 evening carry the plan number only in their note ("plan 01-01, end A, ...").
            tries = sum(match.group(2) == plan_number or (match.group(2) is None and self.__note_of(folder).startswith(f"plan {plan_number},"))
                        for folder, match in folders)
            name += f"_plan_{plan_number}_try_{tries + 1}"
        return self.session_folder / name


    @staticmethod
    def __note_of(episode_folder: Path) -> str:
        """The sheet note written into an episode's episode.json, or "" when there is none."""

        try:
            return json.loads((episode_folder / "episode.json").read_text(encoding = "utf-8")).get("sheet_orientation") or ""
        except (OSError, ValueError):
            return ""


    def __append_session_notes(self, text: str) -> None:

        notes = self.session_folder / "session_notes.md"
        if not notes.exists():
            notes.write_text(f"# Session {self.session_folder.name}\n", encoding = "utf-8")
        with open(notes, "a", encoding = "utf-8") as file:
            file.write(text)


    @staticmethod
    def __code_commit() -> str:

        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd = PROJECT_FOLDER, capture_output = True,
                                  text = True, check = True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"


    def __wait_for(self, accepted_keys: List[str]) -> str:

        while True:
            key = self.key_reader(0)
            if key in accepted_keys:
                return key


    def __show_message(self, message: str) -> None:

        print(message)
        self.__show_picture(self.robot_link.read_picture_bytes(), message)


    def __show_picture(
            self,
            picture_bytes: bytes,
            status: str
    ) -> None:
        """Show the picture with a status bar BELOW it, so the bar never covers what the camera sees."""

        if not self.recorder_parameters["is_window_shown"]:
            return

        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        status_bar = status_bar_image(f"sharpness {sharpness_of(picture):.0f} | {status}", picture.shape[1])
        sighting = getattr(self, "puck_sighting", None)
        shown = picture.copy()
        if sighting is not None and sighting["picture_hash"] == hash(picture_bytes):
            # The puck the tracker sees, marked on the shown copy only: the saved frame stays as the camera took it.
            centre = tuple(int(round(value)) for value in sighting["pixel"])
            cv2.circle(shown, centre, int(round(sighting["radius_px"])) + 5, (0, 230, 255), 1)
        show_in_window(numpy.vstack([shown, status_bar]))



def main() -> None:

    parser = argparse.ArgumentParser(description = "Record demonstration episodes, one fresh picture and one 1 cm step at a time.")
    parser.add_argument("--operator", required = True, help = "who records, e.g. marshal or emranul")
    parser.add_argument("--mode", choices = ["keyboard", "hand_guided", "hand_guided_exact", "gamepad"], default = "keyboard",
                        help = "hand_guided replays the guided path in 1 cm steps of the fixed moves; hand_guided_exact replays every joint; "
                               "gamepad steers with a gamepad (sticks: move, twist, wrist; L1 R1: hand; L2 R2: height)")
    parser.add_argument("--path-id", default = SHEET_PARAMETERS["path_id"], help = "the sheet, e.g. path_01_straight_line")
    parser.add_argument("--start-x", type = float, help = "the puck's start x in metres (default: the taught line start)")
    parser.add_argument("--start-y", type = float, help = "the puck's start y in metres")
    parser.add_argument("--new-session", action = "store_true", help = "open a new session folder even if one exists today")
    parser.add_argument("--simulated", action = "store_true", help = "use the simulated robot instead of the real one")
    parser.add_argument("--trial", action = "store_true",
                        help = "a trial, not training data: saved in the tool-test folder (config tool_test_folder), not the main data")
    parser.add_argument("--single-episode", action = "store_true", help = "end the session after one episode, so each episode can carry its own plan number")
    parser.add_argument("--camera-mount", default = RECORDING_PARAMETERS["camera_mount"],
                        help = "which mount the camera is on, e.g. stock_wrist_holder or bracket_F; never mix two in one training set")
    parser.add_argument("--orientation", default = "", help = "how the sheet lies, e.g. 'line end away from base' or 'rotated 60 deg'")
    parser.add_argument("--plan-number", default = "", help = "the episode plan's number, e.g. 01-07; it goes into the folder name with the try")
    parser.add_argument("--start-index", type = int, default = 0, help = "which marked start point along the line, 0 = the line start")
    parser.add_argument("--offset-cm", type = float, default = 0.0, help = "deliberate sideways offset of the puck from the line, in cm")
    parser.add_argument("--offset-deg", type = float, default = 0.0, help = "deliberate angle of the rod off the line's tangent, in degrees")
    parser.add_argument("--lighting", default = "", help = "a short note on the room light, e.g. 'ceiling lamps, blinds closed'")
    parser.add_argument("--max-turn-lean-deg", type = float, default = RECORDING_PARAMETERS["max_turn_lean_deg"],
                        help = "for a measurement run only: the rod lean allowed with the forearm turned, tip down (config: the measured limit)")
    arguments = parser.parse_args()

    recorder_parameters = {
        "is_new_session": arguments.new_session,
        "is_simulated": arguments.simulated,
        "is_single_episode": arguments.single_episode,
        "camera_mount": arguments.camera_mount,
        "sheet_orientation": arguments.orientation,
        "plan_number": arguments.plan_number,
        "start_index": arguments.start_index,
        "deliberate_offset": {"lateral_cm": arguments.offset_cm, "angle_deg": arguments.offset_deg},
        "lighting_note": arguments.lighting,
        "max_turn_lean_deg": arguments.max_turn_lean_deg
    }
    if arguments.max_turn_lean_deg != RECORDING_PARAMETERS["max_turn_lean_deg"]:
        print(f"MEASUREMENT RUN: turned steps allowed up to a {arguments.max_turn_lean_deg} degree lean (config: {RECORDING_PARAMETERS['max_turn_lean_deg']})")
    if arguments.start_x is not None and arguments.start_y is not None:
        recorder_parameters["start_point_xy"] = [arguments.start_x, arguments.start_y]

    if arguments.simulated:
        from robot_link import SimulatedRobotLink, simulated_hand_path
        robot_link = SimulatedRobotLink({"move_seconds": 0.5, "hand_path_xy": simulated_hand_path()})
        recorder_parameters["raw_folder"] = "03_data/raw_simulated"
    else:
        if arguments.trial:
            recorder_parameters["raw_folder"] = RECORDING_PARAMETERS["tool_test_folder"]
        from robot_link import RobotLink
        robot_link = RobotLink(ROBOT_PARAMETERS["ip_address"])

    recorder = EpisodeRecorder(robot_link, operator = arguments.operator, path_id = arguments.path_id,
                               recorder_parameters = recorder_parameters)
    recorder.record_session(arguments.mode)


if __name__ == "__main__":
    main()
