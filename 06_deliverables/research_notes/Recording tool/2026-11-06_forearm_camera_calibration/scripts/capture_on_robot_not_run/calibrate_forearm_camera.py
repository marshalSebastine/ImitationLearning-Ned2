"""
Capture pictures of the red puck from many arm poses, with the puck at known places, to fit the forearm camera model.

The camera is taped to the forearm cover and turns with joint 4, so where it looks follows from the arm's joints
through the arm model (ned2_fk) once its pose on the forearm and its lens are known. This routine collects what such a
fit needs: the puck's top-face centre in the robot's base frame, and pictures of the puck from many arm poses, each with
the joints read back from the robot when the picture was taken. It only collects; the fit is a separate step, and
nothing it saves is training data.

One spot after the other (the robot never moves down towards a puck that is already there):
    1. The tip comes down over an empty spot and hovers with the magnet block's bottom 1 cm above where the puck's top
       will be.
    2. The person slides the puck under the block, centred by eye, takes the hand away and presses Enter. The puck's
       top-face centre is then known: x and y are the tip's, from the joints read back through the arm model; z is the
       table there (config's table plane) plus the puck's height.
    3. The tip rises straight up to the travel height (the first 1.5 cm as a move of its own) and visits the survey
       poses: the tip behind the puck along the picture's up direction (where the camera sees the puck past the hand),
       to either side, at several heights, the forearm twisted from -60 to +60 degrees, the rod leaning as little as the
       arm allows (side_view_test's solver). It rises to the travel height before every sideways move and comes
       straight down to the next pose. At each pose: wait for the move to end, a fresh picture, the joints read back,
       one row in captures.csv.

Safety, checked in the arm model for the whole plan before anything moves, and again before every single move from the
joints read back at that moment (the robot is assumed to move along the straight line in joint space):
    - Wherever the magnet block comes within "near_puck_horizontal_m" (4 cm) sideways of a place where the puck lies or
      lay, its lowest point (the tip, lowered by the block's half-diagonal when the rod leans) stays at least
      "near_puck_minimum_clearance_m" (2 cm) above the puck's top, and so does the rod's end; at every survey pose the
      block is at least 3 cm above it.
    - The only exceptions are the hover (1 cm, by design, before the puck is there) and the rise from it, which must go
      straight up (within 2 mm sideways, never down) until the block is 2 cm clear.
    - The block stays at least 2 cm above the table everywhere.
    - Esc, Ctrl-C or any error: rise straight up, turn the forearm back at height, go to the parking pose. A second
      Ctrl-C, or a step the check refuses, leaves the arm where it is (motors on).
Only one program may be connected to the robot at a time: close every viewer and the recording tool first.

Run (checklist: operator_checklist.md):
    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/calibrate_forearm_camera.py --operator emranul
The plan and its safety numbers, no robot:
    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/calibrate_forearm_camera.py --operator emranul --plan-only
Practice without the robot (simulated sheet, puck and camera):
    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/calibrate_forearm_camera.py --operator emranul --simulated

Output, a new folder per run, never overwritten (simulated and plan-only runs go to 03_data/raw_simulated):
    03_data/raw/camera_calibration/<yyyy-mm-dd_hhmm>/
        captures.csv          one row per survey picture
        spots.csv             one row per spot: where the puck's top-face centre was
        moves.csv             every move with its sampled clearances, as planned and as checked before moving
        planned_moves.csv, planned_poses.csv, plan_summary.json   the plan, written before anything moves
        calibration_run.json  settings, versions and the summary
        log.txt               everything printed in the terminal
        pictures/             spot_SS_pose_PP.jpg exactly as the robot sent them, hover_spot_SS.jpg
        camera_check_start.jpg, camera_check_end.jpg   the same parking pose before and after
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import re
import time
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional

import cv2
import numpy

from config import (PUCK_PARAMETERS, PUSHER_PARAMETERS, RECORDING_PARAMETERS, ROBOT_PARAMETERS, WORKSPACE_PARAMETERS,
                    table_height)
from record_episodes import (camera_shift_px, forearm_camera_heading_rad, read_window_key, sharpness_of,
                             status_bar_image)
from side_view_test import JOINT_LIMITS_RAD, magnet_tip_and_lean, ned2_fk, solve_least_lean

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
WINDOW_NAME = "calibrate_forearm_camera"

# Settings of this routine; they move to config.py when the routine is adopted.
CALIBRATION_CAPTURE_PARAMETERS = {
    # Where the puck goes, one spot after the other (x, y in metres, robot base frame): the middle of the area the sheets
    # cover, ordered from the base outwards. Every survey pose around them was solved and checked with the arm model on
    # 6.11.26 with config's workspace of that day (calc); beyond about 38 cm from the base the twisted poses would lean
    # more than "survey_max_lean_deg". Re-check with --plan-only whenever the workspace in config is taught again.
    "spots_xy": [[0.22, 0.04], [0.22, -0.06], [0.30, -0.01], [0.37, -0.06], [0.37, 0.04]],
    # The survey around each spot, in this order (by twist, so the forearm turns little from one pose to the next; the
    # order is reversed at every second spot). twist_deg: joint 4 in degrees. ahead_m, left_m: where the puck's centre
    # lies from the tip along the picture's up and left directions on the table (the recording tool's heading model),
    # so the camera sees the puck past the hand whatever the twist. clearance_m: the block's lowest point above the
    # puck's top, at least "minimum_survey_clearance_m".
    "survey_poses": [
        {"twist_deg": -60.0, "ahead_m": 0.050, "left_m": 0.000, "clearance_m": 0.030},
        {"twist_deg": -60.0, "ahead_m": 0.070, "left_m": 0.030, "clearance_m": 0.050},
        {"twist_deg": -30.0, "ahead_m": 0.040, "left_m": -0.030, "clearance_m": 0.040},
        {"twist_deg": -30.0, "ahead_m": 0.080, "left_m": 0.000, "clearance_m": 0.070},
        {"twist_deg": 0.0, "ahead_m": 0.030, "left_m": 0.000, "clearance_m": 0.030},
        {"twist_deg": 0.0, "ahead_m": 0.050, "left_m": 0.040, "clearance_m": 0.050},
        {"twist_deg": 0.0, "ahead_m": 0.060, "left_m": -0.040, "clearance_m": 0.030},
        {"twist_deg": 0.0, "ahead_m": 0.080, "left_m": 0.000, "clearance_m": 0.050},
        {"twist_deg": 30.0, "ahead_m": 0.040, "left_m": 0.030, "clearance_m": 0.040},
        {"twist_deg": 30.0, "ahead_m": 0.080, "left_m": 0.000, "clearance_m": 0.070},
        {"twist_deg": 60.0, "ahead_m": 0.050, "left_m": 0.000, "clearance_m": 0.030},
        {"twist_deg": 60.0, "ahead_m": 0.070, "left_m": -0.030, "clearance_m": 0.050}
    ],
    "hover_clearance_m": 0.010,
    "minimum_survey_clearance_m": 0.030,
    "travel_clearance_m": 0.080,
    # Extra height of the detour over the tip when the straight joint line to the parking pose fails a check.
    "home_detour_extra_m": 0.060,
    "near_puck_horizontal_m": 0.040,
    "near_puck_minimum_clearance_m": 0.020,
    "table_minimum_clearance_m": 0.020,
    # The magnet block is 37 mm wide and 13 mm deep (config's comments of 4.11.26); half its diagonal, rounded up, is the
    # radius of the disc taken as its footprint, so the clearance holds whichever way joint 6 turns it. The rod (21 mm)
    # is checked at the tool point with its own radius.
    "block_footprint_radius_m": 0.020,
    "rod_radius_m": 0.0105,
    # The rise from the hover: at most this far sideways and this far down while the block is less than 2 cm clear (or
    # less than "table_minimum_clearance_m" above the table). It stops once at "rise_step_clearance_m", so the first
    # 1.5 cm over the puck are a move of their own: a move that short cannot swing far even if the robot's planner does
    # not follow the straight joint line the checks assume.
    "hover_rise_sideways_tolerance_m": 0.002,
    "hover_rise_drop_tolerance_m": 0.0002,
    "rise_step_clearance_m": 0.025,
    # The lean changes nothing for the camera (it sits before the wrist) and is part of the clearance; 35 keeps all
    # twelve poses at 37 cm from the base, where poses 2 and 12 need 30.8 to 31.1 degrees (calc 6.11.26; 30 would drop
    # those four). The exact replay of episode 039 ended at 47.6 degrees (arm model from its steps.csv).
    "survey_max_lean_deg": 35.0,
    "travel_max_lean_deg": 40.0,
    # Above this lean on a sampled path the hand's housing could come lower than the block, which the clearance model
    # does not cover; such a sample is accepted only with the rod's end "hand_reach_m" plus the near clearance above the
    # table and every puck within reach. 10 cm bounds the hand from the rod's end (arm model: joint 5's axis about 5 cm
    # from it, plus the wrist housing's radius, guessed under 5 cm; not measured).
    "path_max_lean_deg": 45.0,
    "hand_reach_m": 0.10,
    "wrist_range_deg": [-60.0, 60.0],
    "wrist_search_near_deg": 8.0,
    # Sideways moves at the travel height are split so the forearm turns at most this much and the tip moves at most
    # this far per move; every split point is solved, so the straight joint line between them stays near the height.
    "travel_twist_increment_deg": 15.0,
    "travel_step_m": 0.04,
    "path_sample_deg": 0.5,
    "heading_iterations": 8,
    "heading_tolerance_deg": 0.2,
    "velocity_percent": 15,
    "settle_s": 0.4,
    "steps_until_fresh_picture": 2,
    "fresh_picture_timeout_s": 3.0,
    "joint_warning_deg": 1.0,
    "joint_stop_deg": 5.0,
    "model_tolerance_mm": RECORDING_PARAMETERS["turn_model_tolerance_mm"],
    "home_joints": RECORDING_PARAMETERS["home_joints"],
    "camera_check_region": RECORDING_PARAMETERS["camera_check_region"],
    "camera_shift_warning_px": RECORDING_PARAMETERS["camera_shift_warning_px"],
    "raw_folder": "03_data/raw/camera_calibration",
    "simulated_folder": "03_data/raw_simulated/camera_calibration",
    # The live red-puck finder, for the operator's feedback and the log only; the fit finds the puck again itself.
    # Thresholds read from frames of episodes 037 and 042 (puck, wooden table, a mouse's red wheel) on 6.11.26.
    "detector": {
        "minimum_red_excess": 45,
        "minimum_red": 80,
        "minimum_saturation": 90,
        "hue_low_max": 5,
        "hue_high_min": 160,
        "minimum_area_px": 150,
        "minimum_fill": 0.45,
        "aspect_range": [0.4, 2.5]
    },
    # Simulation only: how far from the tip the simulated person puts the puck (by eye), and the seed for it.
    "simulated_placement_error_m": 0.002,
    "seed": 61_126
}

CAPTURE_COLUMNS = ["spot", "pose", "picture", "picture_md5", "capture_time", "sharpness",
                   "puck_top_x", "puck_top_y", "puck_top_z", "table_z",
                   "planned_twist_deg", "planned_ahead_m", "planned_left_m", "planned_clearance_m",
                   "twist_deg", "lean_deg", "clearance_m", "tip_x", "tip_y", "tip_z", "puck_ahead_m", "puck_left_m",
                   "picture_heading_deg", "joint_error_max_deg",
                   "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                   "commanded_joint_1", "commanded_joint_2", "commanded_joint_3", "commanded_joint_4",
                   "commanded_joint_5", "commanded_joint_6",
                   "x", "y", "z", "roll", "pitch", "yaw",
                   "is_puck_seen", "puck_u", "puck_v", "puck_area_px", "puck_touches_edge"]
SPOT_COLUMNS = ["spot", "status", "planned_x", "planned_y", "puck_top_x", "puck_top_y", "puck_top_z", "table_z",
                "hover_clearance_m", "hover_lean_deg", "model_error_mm", "placed_time", "pictures", "pictures_with_puck",
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "x", "y", "z", "roll", "pitch", "yaw"]
MOVE_COLUMNS = ["checked_at", "spot", "move", "kind", "label", "samples", "is_safe", "reasons",
                "min_clearance_above_spot_puck_m", "min_near_puck_clearance_m", "min_tool_point_near_clearance_m",
                "min_table_clearance_m", "max_lean_deg", "max_rise_sideways_m", "largest_joint_change_deg"]
POSE_COLUMNS = ["spot", "pose", "twist_deg", "ahead_m", "left_m", "planned_clearance_m", "clearance_m", "lean_deg",
                "tip_x", "tip_y", "tip_z", "tip_to_puck_m", "heading_deg", "wrist_deg",
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]



class UnsafeMove(Exception):
    """A planned or requested move failed a clearance check, or the robot did not arrive."""



class StopRequested(Exception):
    """The operator pressed Esc."""



def find_red_puck(
        picture_bgr: numpy.ndarray,
        detector_parameters: Dict
) -> Optional[Dict]:
    """
    Find the red puck as the largest compact, strongly red blob of a picture.

    Strongly red means a red channel well above green and blue, a red or magenta hue (the camera's white balance tints
    the puck towards magenta in some pictures) and enough saturation; small blobs (a mouse's red wheel), thin ones and
    loose ones are ignored.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The picture as OpenCV decodes it.
    detector_parameters : dict
        The thresholds, as in ``CALIBRATION_CAPTURE_PARAMETERS["detector"]``.

    Returns
    -------
    dict or None
        "u", "v" (the blob's centroid in pixels), "area_px" and "touches_edge"; None when nothing qualifies.
    """

    picture = picture_bgr.astype(numpy.int32)
    blue, green, red = picture[:, :, 0], picture[:, :, 1], picture[:, :, 2]
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    is_red = (((red - numpy.maximum(green, blue)) >= detector_parameters["minimum_red_excess"])
              & (red >= detector_parameters["minimum_red"])
              & (saturation >= detector_parameters["minimum_saturation"])
              & ((hue >= detector_parameters["hue_high_min"]) | (hue <= detector_parameters["hue_low_max"])))
    mask = cv2.morphologyEx(is_red.astype(numpy.uint8) * 255, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    count, _, statistics, centroids = cv2.connectedComponentsWithStats(mask)

    best = None
    height_px, width_px = picture_bgr.shape[:2]
    for label in range(1, count):
        left, top, width, height, area = (int(value) for value in statistics[label])
        aspect = width / max(height, 1)
        is_compact = (area >= detector_parameters["minimum_area_px"]
                      and area >= detector_parameters["minimum_fill"] * width * height
                      and detector_parameters["aspect_range"][0] <= aspect <= detector_parameters["aspect_range"][1])
        if not is_compact or (best is not None and area <= best["area_px"]):
            continue
        best = {"u": float(centroids[label][0]), "v": float(centroids[label][1]), "area_px": area,
                "touches_edge": left <= 1 or top <= 1 or left + width >= width_px - 1 or top + height >= height_px - 1}
    return best


def block_geometry(
        joints: List[float],
        footprint_radius_m: float
) -> Dict:
    """
    Where the magnet block is for ``joints``, from the arm model.

    Parameters
    ----------
    joints : list of float
        The six joint angles in radians.
    footprint_radius_m : float
        The radius of the disc taken as the block's footprint.

    Returns
    -------
    dict
        "tip" (the magnet tip [x, y, z], on the rod's axis), "tool" (the rod's end, the robot's reported point),
        "lean_deg" (the rod's angle from straight down) and "lowest_z" (the tip lowered by the footprint disc's drop
        when the rod leans: the block's lowest point).
    """

    tip, lean_deg = magnet_tip_and_lean(list(joints))
    tool = numpy.array(ned2_fk.fk(list(joints))["tool"][0])
    lowest_z = float(tip[2]) - footprint_radius_m * math.sin(math.radians(lean_deg))
    return {"tip": numpy.array(tip, dtype = float), "tool": tool, "lean_deg": float(lean_deg), "lowest_z": lowest_z}


def puck_top_centre(point_xy: List[float]) -> numpy.ndarray:
    """The puck's top-face centre [x, y, z] when the puck stands at ``point_xy``: the table plane there plus its height."""

    return numpy.array([float(point_xy[0]), float(point_xy[1]),
                        table_height(float(point_xy[0]), float(point_xy[1])) + PUCK_PARAMETERS["height_m"]])


def picture_axes_xy(joints: List[float]) -> tuple:
    """The picture's up and left directions on the table for ``joints`` (the recording tool's heading model)."""

    heading = forearm_camera_heading_rad(list(joints))
    left = heading + RECORDING_PARAMETERS["picture_left_sign"] * math.pi / 2
    return numpy.array([math.cos(heading), math.sin(heading)]), numpy.array([math.cos(left), math.sin(left)])


def check_joint_path(
        start_joints: List[float],
        end_joints: List[float],
        puck_tops: List[numpy.ndarray],
        capture_parameters: Dict,
        spot_puck_top: Optional[numpy.ndarray] = None,
        is_rise: bool = False
) -> Dict:
    """
    Sample the straight line in joint space from ``start_joints`` to ``end_joints`` and check every clearance on it.

    Parameters
    ----------
    start_joints, end_joints : list of float
        The move's ends, in radians.
    puck_tops : list of numpy.ndarray
        The top-face centres of every place where the puck lies or may still lie.
    capture_parameters : dict
        The routine's settings (clearances, footprint, sampling).
    spot_puck_top : numpy.ndarray, optional
        The current spot's puck top, for the report's "lowest point above the spot's puck top" (also before the puck
        is there).
    is_rise : bool, optional
        True for a rise from a pose that may start closer than the near clearance to a puck (the hover) or than the
        table clearance to the table: while still that close, the block must not go down by more than
        "hover_rise_drop_tolerance_m" and must stay within "hover_rise_sideways_tolerance_m" sideways of where it
        started.

    Returns
    -------
    dict
        "is_safe", "reasons" and the extreme values found: "min_clearance_above_spot_puck_m",
        "min_near_puck_clearance_m", "min_tool_point_near_clearance_m", "min_table_clearance_m", "max_lean_deg",
        "max_rise_sideways_m", "samples" and "largest_joint_change_deg".
    """

    largest_change_deg = max(abs(math.degrees(end - start)) for start, end in zip(start_joints, end_joints))
    count = max(2, int(math.ceil(largest_change_deg / capture_parameters["path_sample_deg"])) + 1)
    footprint = capture_parameters["block_footprint_radius_m"]
    puck_radius = PUCK_PARAMETERS["diameter_m"] / 2
    near_horizontal = capture_parameters["near_puck_horizontal_m"]
    near_minimum = capture_parameters["near_puck_minimum_clearance_m"]

    start_geometry = block_geometry(start_joints, footprint)
    start_tip_xy = start_geometry["tip"][:2]
    rising_pucks = set()
    is_rising_from_table = False
    if is_rise:
        for index, puck_top in enumerate(puck_tops):
            gap = float(numpy.linalg.norm(start_tip_xy - puck_top[:2])) - footprint - puck_radius
            if gap < near_horizontal and start_geometry["lowest_z"] - float(puck_top[2]) < near_minimum:
                rising_pucks.add(index)
        start_table_clearance = start_geometry["lowest_z"] - table_height(float(start_tip_xy[0]), float(start_tip_xy[1]))
        is_rising_from_table = start_table_clearance < capture_parameters["table_minimum_clearance_m"]

    result = {"samples": count, "largest_joint_change_deg": largest_change_deg, "min_clearance_above_spot_puck_m": math.inf,
              "min_near_puck_clearance_m": math.inf, "min_tool_point_near_clearance_m": math.inf,
              "min_table_clearance_m": math.inf, "max_lean_deg": 0.0, "max_rise_sideways_m": 0.0, "reasons": []}
    for step in range(count):
        fraction = step / (count - 1)
        joints = [start + (end - start) * fraction for start, end in zip(start_joints, end_joints)]
        geometry = block_geometry(joints, footprint)
        tip_xy = geometry["tip"][:2]
        sideways = float(numpy.linalg.norm(tip_xy - start_tip_xy))
        is_going_down = geometry["lowest_z"] < start_geometry["lowest_z"] - capture_parameters["hover_rise_drop_tolerance_m"]
        result["max_lean_deg"] = max(result["max_lean_deg"], geometry["lean_deg"])
        table_clearance = geometry["lowest_z"] - table_height(float(tip_xy[0]), float(tip_xy[1]))
        if is_rising_from_table and table_clearance < capture_parameters["table_minimum_clearance_m"]:
            # Rising from close to the table (an arm left low): only straight up is allowed until it is clear.
            result["max_rise_sideways_m"] = max(result["max_rise_sideways_m"], sideways)
            if is_going_down or sideways > capture_parameters["hover_rise_sideways_tolerance_m"]:
                result["reasons"].append(f"the block does not rise straight from {100 * table_clearance:.1f} cm above the table")
        else:
            result["min_table_clearance_m"] = min(result["min_table_clearance_m"], table_clearance)
        if geometry["lean_deg"] > capture_parameters["path_max_lean_deg"]:
            # Steeply leaning: only safe high up, where no part of the hand can reach the table or a puck.
            needed = capture_parameters["hand_reach_m"] + near_minimum
            tool_xy = geometry["tool"][:2]
            surfaces = [table_height(float(tool_xy[0]), float(tool_xy[1]))]
            surfaces += [float(puck_top[2]) for puck_top in puck_tops
                         if float(numpy.linalg.norm(tool_xy - puck_top[:2])) < capture_parameters["hand_reach_m"] + near_horizontal + puck_radius]
            if float(geometry["tool"][2]) - max(surfaces) < needed:
                result["reasons"].append(f"the rod leans {geometry['lean_deg']:.0f} degrees with the rod's end less than "
                                         f"{100 * needed:.0f} cm above the table or a puck")
        if spot_puck_top is not None:
            result["min_clearance_above_spot_puck_m"] = min(result["min_clearance_above_spot_puck_m"],
                                                            geometry["lowest_z"] - float(spot_puck_top[2]))

        for index, puck_top in enumerate(puck_tops):
            clearance = geometry["lowest_z"] - float(puck_top[2])
            gap = float(numpy.linalg.norm(tip_xy - puck_top[:2])) - footprint - puck_radius
            tool_gap = float(numpy.linalg.norm(geometry["tool"][:2] - puck_top[:2])) - capture_parameters["rod_radius_m"] - puck_radius
            if tool_gap < near_horizontal:
                result["min_tool_point_near_clearance_m"] = min(result["min_tool_point_near_clearance_m"],
                                                                float(geometry["tool"][2]) - float(puck_top[2]))
            if gap >= near_horizontal:
                continue
            if index in rising_pucks and clearance < near_minimum:
                # Still in the hover's 1 cm gap: the block may only go straight up.
                result["max_rise_sideways_m"] = max(result["max_rise_sideways_m"], sideways)
                if is_going_down:
                    result["reasons"].append(f"the block goes down by {1000 * (start_geometry['lowest_z'] - geometry['lowest_z']):.1f} mm "
                                             f"while rising from the puck")
                if sideways > capture_parameters["hover_rise_sideways_tolerance_m"]:
                    result["reasons"].append(f"the block moves {1000 * sideways:.1f} mm sideways while less than "
                                             f"{100 * near_minimum:.0f} cm above the puck")
                continue
            result["min_near_puck_clearance_m"] = min(result["min_near_puck_clearance_m"], clearance)
            if clearance < near_minimum:
                result["reasons"].append(f"the block comes {100 * clearance:.1f} cm above a puck's top within "
                                         f"{100 * near_horizontal:.0f} cm sideways of it")

    if result["min_tool_point_near_clearance_m"] < near_minimum:
        result["reasons"].append(f"the rod's end comes {100 * result['min_tool_point_near_clearance_m']:.1f} cm above a puck's top")
    if result["min_table_clearance_m"] < capture_parameters["table_minimum_clearance_m"]:
        result["reasons"].append(f"the block comes {100 * result['min_table_clearance_m']:.1f} cm above the table")
    # The first reason of each kind (the same words with other numbers) is enough for the log.
    kept = {}
    for reason in result["reasons"]:
        kept.setdefault(re.sub(r"-?\d+(\.\d+)?", "#", reason), reason)
    result["reasons"] = list(kept.values())
    result["is_safe"] = not result["reasons"]
    return result


def move_row(
        checked_at: str,
        spot_index: int,
        move: str,
        kind: str,
        label: str,
        check: Dict
) -> Dict:
    """One row of moves.csv from a path check; infinite values (nothing near) are written empty."""

    def metres(value: float) -> object:
        return "" if math.isinf(value) else round(value, 4)

    return {"checked_at": checked_at, "spot": spot_index, "move": move, "kind": kind, "label": label,
            "samples": check["samples"], "is_safe": int(check["is_safe"]), "reasons": "; ".join(check["reasons"]),
            "min_clearance_above_spot_puck_m": metres(check["min_clearance_above_spot_puck_m"]),
            "min_near_puck_clearance_m": metres(check["min_near_puck_clearance_m"]),
            "min_tool_point_near_clearance_m": metres(check["min_tool_point_near_clearance_m"]),
            "min_table_clearance_m": metres(check["min_table_clearance_m"]),
            "max_lean_deg": round(check["max_lean_deg"], 2), "max_rise_sideways_m": round(check["max_rise_sideways_m"], 5),
            "largest_joint_change_deg": round(check["largest_joint_change_deg"], 2)}



class CalibrationPlanner:

    def __init__(self, capture_parameters: Optional[dict] = None):
        """
        Solve and check every arm position of the capture in the arm model; nothing here talks to the robot.

        Parameters
        ----------
        capture_parameters : dict, optional
            Overrides merged over ``CALIBRATION_CAPTURE_PARAMETERS``.
        """

        self.capture_parameters = dict(CALIBRATION_CAPTURE_PARAMETERS)
        if capture_parameters:
            self.capture_parameters.update(capture_parameters)


    def solve_tip(
            self,
            target_tip: numpy.ndarray,
            twist_deg: float,
            maximum_lean_deg: float,
            near_wrist_rad: Optional[float] = None
    ) -> Optional[List[float]]:
        """
        The joints that put the magnet tip on ``target_tip`` with joint 4 at ``twist_deg`` and the rod leaning least.

        Uses side_view_test.solve_least_lean, first with the wrist searched near ``near_wrist_rad`` (quicker, and no
        jump to another wrist solution between neighbouring positions), then over the whole range. Joint 6 is set
        equal to joint 1, the hand's usual orientation in the recording tool, so the block looks the same as in the
        recorded pictures; joint 6 turns the block about the rod's axis only, which carries the tip.

        Returns
        -------
        list of float or None
            The six joint angles in radians; None when out of reach, out of the joint limits or leaning more than
            ``maximum_lean_deg``.
        """

        wrist_low, wrist_high = self.capture_parameters["wrist_range_deg"]
        joints = None
        if near_wrist_rad is not None:
            near_deg = math.degrees(near_wrist_rad)
            width = self.capture_parameters["wrist_search_near_deg"]
            window = (max(wrist_low, near_deg - width), min(wrist_high, near_deg + width))
            joints = solve_least_lean(numpy.array(target_tip, dtype = float), math.radians(twist_deg), wrist_range_deg = window)
            # A best wrist at the window's edge means the least lean lies outside it: search the whole range instead.
            if joints is not None and not window[0] + 1.0 < math.degrees(joints[4]) < window[1] - 1.0:
                joints = None
        if joints is None:
            joints = solve_least_lean(numpy.array(target_tip, dtype = float), math.radians(twist_deg),
                                      wrist_range_deg = (wrist_low, wrist_high))
        if joints is None:
            return None

        joints = [float(value) for value in joints]
        joint_6_low, joint_6_high = JOINT_LIMITS_RAD[5]
        if joint_6_low <= joints[0] <= joint_6_high:
            joints[5] = joints[0]
        if magnet_tip_and_lean(joints)[1] > maximum_lean_deg:
            return None
        return joints


    def solve_at_clearance(
            self,
            tip_xy: numpy.ndarray,
            twist_deg: float,
            reference_top_z: float,
            clearance_m: float,
            maximum_lean_deg: float,
            near_wrist_rad: Optional[float] = None,
            lean_guess_deg: float = 0.0
    ) -> Optional[List[float]]:
        """
        The joints that put the tip over ``tip_xy`` with the block's lowest point ``clearance_m`` above
        ``reference_top_z``: the tip is raised by the footprint's drop at the rod's lean, found in up to four passes
        starting from ``lean_guess_deg`` (a neighbouring position's lean saves passes).
        """

        footprint = self.capture_parameters["block_footprint_radius_m"]
        lean_deg = lean_guess_deg
        joints = None
        for _ in range(4):
            tip_z = reference_top_z + clearance_m + footprint * math.sin(math.radians(lean_deg))
            joints = self.solve_tip(numpy.array([tip_xy[0], tip_xy[1], tip_z]), twist_deg, maximum_lean_deg, near_wrist_rad)
            if joints is None:
                return None
            new_lean_deg = magnet_tip_and_lean(joints)[1]
            if abs(new_lean_deg - lean_deg) < 0.05:
                break
            lean_deg = new_lean_deg
            near_wrist_rad = joints[4]
        return joints


    def survey_joints(
            self,
            puck_top: numpy.ndarray,
            survey_pose: Dict,
            near_wrist_rad: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Solve one survey pose: the tip behind the puck along the picture's up direction (and to the side) at the
        pose's clearance and twist.

        The picture's direction and the rod's lean depend on the joints being solved, so the tip's place and the joints
        are found alternately until the heading ("heading_tolerance_deg") and the lean (0.05 degrees) settle.

        Returns
        -------
        dict or None
            "joints", the "tip_xy" aimed at, and the "heading_deg" it settled on; None when the pose cannot be reached
            within the joint limits and the lean limit, when it does not settle, or when its tip or the puck would
            leave the workspace box.
        """

        twist_deg = survey_pose["twist_deg"]
        footprint = self.capture_parameters["block_footprint_radius_m"]
        # Turning the forearm by +t turns the picture's top by about -t from the direction base to rod (arm model,
        # 6.11.26: -57 to -59 degrees for t = 60), a good first guess for the loop.
        heading = math.atan2(puck_top[1], puck_top[0]) - math.radians(twist_deg)
        lean_deg = 0.0
        is_settled = False
        joints, tip_xy = None, None
        for _ in range(self.capture_parameters["heading_iterations"]):
            up = numpy.array([math.cos(heading), math.sin(heading)])
            left_rad = heading + RECORDING_PARAMETERS["picture_left_sign"] * math.pi / 2
            left = numpy.array([math.cos(left_rad), math.sin(left_rad)])
            tip_xy = puck_top[:2] - survey_pose["ahead_m"] * up - survey_pose["left_m"] * left
            tip_z = float(puck_top[2]) + survey_pose["clearance_m"] + footprint * math.sin(math.radians(lean_deg))
            joints = self.solve_tip(numpy.array([tip_xy[0], tip_xy[1], tip_z]), twist_deg,
                                    self.capture_parameters["survey_max_lean_deg"], near_wrist_rad)
            if joints is None:
                return None
            near_wrist_rad = joints[4]
            new_heading = forearm_camera_heading_rad(joints)
            new_lean_deg = magnet_tip_and_lean(joints)[1]
            heading_change = math.degrees(math.atan2(math.sin(new_heading - heading), math.cos(new_heading - heading)))
            is_settled = abs(heading_change) < self.capture_parameters["heading_tolerance_deg"] and abs(new_lean_deg - lean_deg) < 0.05
            heading, lean_deg = new_heading, new_lean_deg
            if is_settled:
                break

        if not is_settled or not (self.is_inside_box(tip_xy) and self.is_inside_box(puck_top[:2])):
            return None
        return {"joints": joints, "tip_xy": tip_xy, "heading_deg": math.degrees(heading), "lean_deg": lean_deg}


    @staticmethod
    def is_inside_box(point_xy) -> bool:
        """True when ``point_xy`` lies inside the workspace box (whose far edge is the table's edge)."""

        box_x, box_y = WORKSPACE_PARAMETERS["box_x"], WORKSPACE_PARAMETERS["box_y"]
        return box_x[0] <= point_xy[0] <= box_x[1] and box_y[0] <= point_xy[1] <= box_y[1]


    def travel_joints(
            self,
            start_joints: List[float],
            end_tip_xy: numpy.ndarray,
            end_twist_deg: float,
            reference_top_z: float
    ) -> List[List[float]]:
        """
        Solved positions at the travel height from the tip over ``start_joints`` to the tip over ``end_tip_xy`` with the
        forearm at ``end_twist_deg``, split so that each move turns the forearm by at most "travel_twist_increment_deg"
        and carries the tip at most "travel_step_m"; the start is not included, the last position is the end.

        Raises
        ------
        UnsafeMove
            When a split point cannot be solved.
        """

        start_tip = magnet_tip_and_lean(start_joints)[0][:2]
        start_twist_deg = math.degrees(start_joints[3])
        distance = float(numpy.linalg.norm(numpy.array(end_tip_xy) - start_tip))
        pieces = max(1, int(math.ceil(abs(end_twist_deg - start_twist_deg) / self.capture_parameters["travel_twist_increment_deg"] - 1e-9)),
                     int(math.ceil(distance / self.capture_parameters["travel_step_m"] - 1e-9)))
        positions = []
        near_wrist_rad = start_joints[4]
        lean_guess_deg = magnet_tip_and_lean(start_joints)[1]
        for piece in range(1, pieces + 1):
            fraction = piece / pieces
            tip_xy = start_tip + (numpy.array(end_tip_xy) - start_tip) * fraction
            twist_deg = start_twist_deg + (end_twist_deg - start_twist_deg) * fraction
            joints = self.solve_at_clearance(tip_xy, twist_deg, reference_top_z, self.capture_parameters["travel_clearance_m"],
                                             self.capture_parameters["travel_max_lean_deg"], near_wrist_rad, lean_guess_deg)
            if joints is None:
                raise UnsafeMove(f"no arm position holds the travel height over ({tip_xy[0]:.3f}, {tip_xy[1]:.3f}) with the "
                                 f"forearm at {twist_deg:.0f} degrees")
            positions.append(joints)
            near_wrist_rad = joints[4]
            lean_guess_deg = magnet_tip_and_lean(joints)[1]
        return positions


    def home_route(
            self,
            start_joints: List[float],
            puck_tops: List[numpy.ndarray],
            reference_top_z: float
    ) -> List[List[float]]:
        """
        The positions from ``start_joints`` to the parking pose: straight there when that joint line passes every check,
        otherwise first "home_detour_extra_m" higher over the tip with the forearm straight.
        """

        home = [float(value) for value in self.capture_parameters["home_joints"]]
        if check_joint_path(start_joints, home, puck_tops, self.capture_parameters)["is_safe"]:
            return [home]
        tip_xy = magnet_tip_and_lean(start_joints)[0][:2]
        higher = self.solve_at_clearance(tip_xy, 0.0, reference_top_z,
                                         self.capture_parameters["travel_clearance_m"] + self.capture_parameters["home_detour_extra_m"],
                                         self.capture_parameters["path_max_lean_deg"], start_joints[4])
        return [home] if higher is None else [higher, home]


    def plan_capture(
            self,
            spots_xy: List[List[float]],
            say: Callable[[str], None] = print
    ) -> Dict:
        """
        Plan every move of the capture from the parking pose back to the parking pose, as the robot will run it, and
        check each move's clearances; nothing moves.

        Returns
        -------
        dict
            "spots": one dict per spot with its "puck_top", "hover", "above_hover", "travel_in", "survey" (list of dict
            with "pose", "spec", "joints", "above", "travel_in", "heading_deg") and "travel_out"; "home_route"; "moves"
            (the checked moves, rows for moves.csv); "skipped_poses"; and "is_safe".
        """

        home = [float(value) for value in self.capture_parameters["home_joints"]]
        plan = {"spots": [], "moves": [], "skipped_poses": [], "is_safe": True}
        current = home
        placed_tops = []
        survey_poses = self.capture_parameters["survey_poses"]
        minimum_clearance = self.capture_parameters["minimum_survey_clearance_m"]
        for pose_index, survey_pose in enumerate(survey_poses, start = 1):
            if survey_pose["clearance_m"] < minimum_clearance - 1e-9:
                raise UnsafeMove(f"survey pose {pose_index} asks for {100 * survey_pose['clearance_m']:.1f} cm over the puck, "
                                 f"below the minimum of {100 * minimum_clearance:.1f} cm")
        for spot_index, spot_xy in enumerate(spots_xy, start = 1):
            started = time.monotonic()
            puck_top = puck_top_centre(spot_xy)
            spot = {"spot": spot_index, "spot_xy": [float(value) for value in spot_xy], "puck_top": puck_top, "survey": []}
            hover = self.solve_at_clearance(numpy.array(spot_xy, dtype = float), 0.0, float(puck_top[2]),
                                            self.capture_parameters["hover_clearance_m"], self.capture_parameters["survey_max_lean_deg"])
            if hover is None or not self.is_inside_box(spot_xy):
                raise UnsafeMove(f"spot {spot_index} at ({spot_xy[0]:.3f}, {spot_xy[1]:.3f}) cannot be reached with the rod "
                                 f"upright, or lies outside the workspace box")
            spot["hover"] = hover
            spot["hover_step"] = self.solve_at_clearance(numpy.array(spot_xy, dtype = float), 0.0, float(puck_top[2]),
                                                         self.capture_parameters["rise_step_clearance_m"],
                                                         self.capture_parameters["survey_max_lean_deg"], hover[4])
            spot["travel_in"] = self.travel_joints(current, numpy.array(spot_xy, dtype = float), 0.0, float(puck_top[2]))
            spot["above_hover"] = spot["travel_in"][-1]
            if spot["hover_step"] is None:
                raise UnsafeMove(f"spot {spot_index}: no upright position {100 * self.capture_parameters['rise_step_clearance_m']:.1f} cm over the puck")

            # To the hover: from the last position through the travel points, then down; the puck is not there yet.
            self.__check_chain(plan, spot_index, "in", "travel", "to the spot at travel height", [current] + spot["travel_in"],
                               placed_tops, puck_top)
            self.__record_check(plan, spot_index, "hover_down", "hover_descend", "down to the hover, no puck yet",
                                spot["above_hover"], hover, placed_tops, puck_top)
            placed_tops = placed_tops + [puck_top]
            self.__record_check(plan, spot_index, "hover_up_1", "hover_rise", "straight up from the puck, first 1.5 cm", hover,
                                spot["hover_step"], placed_tops, puck_top, is_rise = True)
            self.__record_check(plan, spot_index, "hover_up_2", "rise", "up to the travel height", spot["hover_step"],
                                spot["above_hover"], placed_tops, puck_top, is_rise = True)

            ordered = list(enumerate(survey_poses, start = 1))
            if spot_index % 2 == 0:
                ordered.reverse()
            current = spot["above_hover"]
            for pose_index, survey_pose in ordered:
                solution = self.survey_joints(puck_top, survey_pose, near_wrist_rad = current[4])
                above = None
                if solution is not None:
                    achieved = block_geometry(solution["joints"], self.capture_parameters["block_footprint_radius_m"])["lowest_z"] - float(puck_top[2])
                    if achieved < minimum_clearance - 0.0005:
                        solution = None
                if solution is not None:
                    above = self.solve_at_clearance(solution["tip_xy"], survey_pose["twist_deg"], float(puck_top[2]),
                                                    self.capture_parameters["travel_clearance_m"],
                                                    self.capture_parameters["travel_max_lean_deg"], solution["joints"][4],
                                                    solution["lean_deg"])
                if solution is None or above is None:
                    reason = ("out of reach, over the lean limit, under the minimum clearance or outside the box" if solution is None
                              else "no travel height above it")
                    plan["skipped_poses"].append({"spot": spot_index, "pose": pose_index, "reason": reason})
                    say(f"spot {spot_index} pose {pose_index}: skipped ({reason})")
                    continue
                travel_in = self.travel_joints(current, solution["tip_xy"], survey_pose["twist_deg"], float(puck_top[2]))
                # The last travel point aims at the same tip and twist as "above"; keep the one solved from the pose.
                travel_in[-1] = above
                self.__check_chain(plan, spot_index, f"p{pose_index:02d}_travel", "travel", "sideways at travel height",
                                   [current] + travel_in, placed_tops, puck_top)
                self.__record_check(plan, spot_index, f"p{pose_index:02d}_down", "descend", "down to the survey pose", above,
                                    solution["joints"], placed_tops, puck_top)
                self.__record_check(plan, spot_index, f"p{pose_index:02d}_up", "rise", "up from the survey pose",
                                    solution["joints"], above, placed_tops, puck_top, is_rise = True)
                spot["survey"].append({"pose": pose_index, "spec": dict(survey_pose), "joints": solution["joints"], "above": above,
                                       "travel_in": travel_in, "heading_deg": solution["heading_deg"]})
                current = above

            # The forearm back to straight at the travel height, over the last pose's tip.
            last_tip_xy = magnet_tip_and_lean(current)[0][:2]
            spot["travel_out"] = self.travel_joints(current, last_tip_xy, 0.0, float(puck_top[2])) if abs(current[3]) > 1e-6 else []
            self.__check_chain(plan, spot_index, "out", "travel", "forearm back at travel height", [current] + spot["travel_out"],
                               placed_tops, puck_top)
            current = ([current] + spot["travel_out"])[-1]
            plan["spots"].append(spot)
            say(f"spot {spot_index} planned: {len(spot['survey'])} poses ({time.monotonic() - started:.0f} s of arm-model solving)")

        reference_top_z = max(float(puck_top[2]) for puck_top in placed_tops)
        plan["home_route"] = self.home_route(current, placed_tops, reference_top_z)
        self.__check_chain(plan, len(spots_xy) + 1, "home", "travel", "to the parking pose", [current] + plan["home_route"],
                           placed_tops, None)
        plan["is_safe"] = all(move["is_safe"] for move in plan["moves"])
        return plan


    def __check_chain(
            self,
            plan: Dict,
            spot_index: int,
            name: str,
            kind: str,
            label: str,
            positions: List[List[float]],
            puck_tops: List[numpy.ndarray],
            spot_puck_top: Optional[numpy.ndarray]
    ) -> None:
        """Check every move between consecutive ``positions``, numbered ``name``_1, ``name``_2 and so on."""

        for move_index, (start, end) in enumerate(zip(positions[:-1], positions[1:]), start = 1):
            self.__record_check(plan, spot_index, f"{name}_{move_index}", kind, label, start, end, puck_tops, spot_puck_top)


    def __record_check(
            self,
            plan: Dict,
            spot_index: int,
            move: str,
            kind: str,
            label: str,
            start_joints: List[float],
            end_joints: List[float],
            puck_tops: List[numpy.ndarray],
            spot_puck_top: Optional[numpy.ndarray],
            is_rise: bool = False
    ) -> None:
        """Check one planned move and add its row to ``plan["moves"]``."""

        check = check_joint_path(start_joints, end_joints, puck_tops, self.capture_parameters, spot_puck_top, is_rise)
        plan["moves"].append(move_row("plan", spot_index, move, kind, label, check))



class ForearmCameraCalibration:

    def __init__(
            self,
            robot_link,
            operator: str,
            plan: Dict,
            output_folder: Path,
            capture_parameters: Optional[dict] = None,
            key_reader: Callable[[int], Optional[str]] = read_window_key,
            is_window_shown: bool = True,
            is_simulated: bool = False
    ):
        """
        Run a planned capture on an open robot link.

        Parameters
        ----------
        robot_link : RobotLink or SimulatedRobotLink
            The robot; it needs move_joints, predicted_position, read_state, read_picture_bytes and close.
        operator : str
            Who runs the capture, written into calibration_run.json.
        plan : dict
            The checked plan from CalibrationPlanner.plan_capture, made with the same settings.
        output_folder : pathlib.Path
            The run's new folder.
        capture_parameters : dict, optional
            Overrides merged over ``CALIBRATION_CAPTURE_PARAMETERS``.
        key_reader : callable, optional
            Returns the next key ("enter", "escape" or a character); replaced in simulation by a script.
        is_window_shown : bool, optional
            Show the camera picture with the instructions in a window.
        is_simulated : bool, optional
            The robot is SimulatedRobotLink: the simulated person places the simulated puck.
        """

        self.robot_link = robot_link
        self.operator = operator
        self.plan = plan
        self.output_folder = output_folder
        self.key_reader = key_reader
        self.is_window_shown = is_window_shown
        self.is_simulated = is_simulated

        self.__initialize_variables()
        self.update_capture_parameters(capture_parameters)


    def __initialize_variables(self):

        self.capture_parameters = dict(CALIBRATION_CAPTURE_PARAMETERS)
        self.placed_puck_tops = []
        self.spot_puck_top = None
        self.is_person_near_arm = False
        self.is_retreating = False
        self.has_moved = False
        self.last_saved_md5 = None
        self.simulated_puck_placed = None
        self.capture_rows = []
        self.spot_rows = []
        self.move_rows = []
        self.summary = {}
        (self.output_folder / "pictures").mkdir(parents = True, exist_ok = True)
        self.log_file = open(self.output_folder / "log.txt", "a", encoding = "utf-8")


    def update_capture_parameters(self, capture_parameters: Optional[dict]):
        """Merge ``capture_parameters`` over the current settings and rebuild what depends on them."""

        if capture_parameters:
            self.capture_parameters.update(capture_parameters)
        self.planner = CalibrationPlanner(self.capture_parameters)
        self.randomness_generator = Random(self.capture_parameters["seed"])


    def run(self) -> Dict:
        """
        The whole capture: parking pose, every spot, parking pose. Whatever happens, the arm rises, turns back and goes
        to the parking pose, the files are written and the connection is closed.

        Returns
        -------
        dict
            The run's summary, also in calibration_run.json.
        """

        started = datetime.datetime.now()
        self.summary = {"operator": self.operator, "started": started.isoformat(timespec = "seconds"), "is_simulated": self.is_simulated,
                        "status": "running", "laptop": platform.node(), "script_md5": script_md5(),
                        "camera_mount": RECORDING_PARAMETERS["camera_mount"], "mapping_version": RECORDING_PARAMETERS["mapping_version"],
                        "table_plane": WORKSPACE_PARAMETERS["table_plane"], "puck_height_m": PUCK_PARAMETERS["height_m"],
                        "pusher": PUSHER_PARAMETERS, "capture_parameters": self.capture_parameters}
        try:
            self.__check_model_against_robot()
            if self.__ask("Plan checked. Puck in your HAND, not on the table. Enter = start (the robot goes to its parking pose), "
                          "Esc = quit", ["enter", "escape"]) == "escape":
                raise StopRequested()
            self.__go_home()
            self.__save_camera_check("camera_check_start.jpg")
            for spot in self.plan["spots"]:
                self.__capture_spot(spot)
            self.__go_home()
            self.__save_camera_check("camera_check_end.jpg")
            self.summary["status"] = "completed"
        except StopRequested:
            self.summary["status"] = "stopped_by_operator"
            self.__say("stopped by the operator (Esc)")
            self.__retreat("Esc")
        except KeyboardInterrupt:
            self.summary["status"] = "stopped_by_ctrl_c"
            self.__say("Ctrl-C")
            self.__retreat("Ctrl-C")
        except Exception as error:
            self.summary["status"] = f"error: {type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"
            self.__say(f"ERROR: {type(error).__name__}: {error}")
            self.__retreat("an error")
        finally:
            self.__finish(started)
        return self.summary


    def __check_model_against_robot(self) -> None:
        """
        Before anything moves: the arm model and the robot's own kinematics must agree on the rod's end at every hover
        and survey pose, within "model_tolerance_mm", because the clearances are worked out in the arm model.
        """

        worst_mm = 0.0
        for spot in self.plan["spots"]:
            for joints in [spot["hover"]] + [survey["joints"] for survey in spot["survey"]]:
                model = numpy.array(ned2_fk.fk(list(joints))["tool"][0])
                robot = numpy.array(self.robot_link.predicted_position(list(joints)))
                worst_mm = max(worst_mm, 1000.0 * float(numpy.linalg.norm(model - robot)))
        self.summary["model_check_worst_mm"] = round(worst_mm, 2)
        if worst_mm > self.capture_parameters["model_tolerance_mm"]:
            raise UnsafeMove(f"the arm model and the robot disagree by {worst_mm:.1f} mm; nothing was moved")
        self.__say(f"arm model checked against the robot's own kinematics: at most {worst_mm:.2f} mm apart at the hovers and survey poses")


    def __capture_spot(self, spot: Dict) -> None:
        """Hover over the spot, let the person place the puck, measure it, survey it, turn the forearm back lifted."""

        spot_index = spot["spot"]
        self.spot_puck_top = spot["puck_top"]
        self.__say(f"spot {spot_index} of {len(self.plan['spots'])}: going to x {100 * spot['spot_xy'][0]:.1f}, y {100 * spot['spot_xy'][1]:.1f} cm")
        # The way in is solved again from where the arm really is (a skipped spot leaves it elsewhere than planned).
        current = [float(value) for value in self.robot_link.read_state()["joints"]]
        travel = self.planner.travel_joints(current, numpy.array(spot["spot_xy"]), 0.0, float(spot["puck_top"][2]))
        travel[-1] = spot["above_hover"]
        for move_index, joints in enumerate(travel, start = 1):
            self.__move(joints, "travel", "to the spot at travel height", spot_index, f"in_{move_index}")
        self.__move(spot["hover"], "hover_descend", "down to the hover, no puck yet", spot_index, "hover_down")

        # From the question on, the puck may be under the block (also after s or Esc): every later move keeps clear of it.
        self.placed_puck_tops.append(spot["puck_top"])
        self.is_person_near_arm = True
        answer = self.__ask(f"SPOT {spot_index}: slide the puck under the block, centred by eye (gap about 1 cm). Hand away, "
                            f"then Enter. s = skip this spot, Esc = finish", ["enter", "s", "escape"])
        if answer == "escape":
            raise StopRequested()
        spot_row = {"spot": spot_index, "planned_x": spot["spot_xy"][0], "planned_y": spot["spot_xy"][1]}
        if answer == "s":
            self.is_person_near_arm = False
            spot_row["status"] = "skipped"
            self.spot_rows.append(spot_row)
            self.__say(f"spot {spot_index} skipped")
            self.__move(spot["hover_step"], "hover_rise", "straight up, first 1.5 cm", spot_index, "hover_up_1", is_rise = True)
            self.__move(spot["above_hover"], "rise", "up to the travel height", spot_index, "hover_up_2", is_rise = True)
            return

        if self.is_simulated:
            self.__simulate_placing_the_puck()
        state = self.robot_link.read_state()
        self.is_person_near_arm = False
        puck_top = self.__measure_puck(state, spot_row)
        self.placed_puck_tops[-1] = puck_top
        self.spot_puck_top = puck_top
        (self.output_folder / "pictures" / f"hover_spot_{spot_index:02d}.jpg").write_bytes(self.__fresh_picture())

        self.__move(spot["hover_step"], "hover_rise", "straight up from the puck, first 1.5 cm", spot_index, "hover_up_1", is_rise = True)
        self.__move(spot["above_hover"], "rise", "up to the travel height", spot_index, "hover_up_2", is_rise = True)
        seen = 0
        for survey in spot["survey"]:
            for move_index, joints in enumerate(survey["travel_in"], start = 1):
                self.__move(joints, "travel", "sideways at travel height", spot_index, f"p{survey['pose']:02d}_travel_{move_index}")
            self.__move(survey["joints"], "descend", "down to the survey pose", spot_index, f"p{survey['pose']:02d}_down")
            row = self.__capture(spot_index, survey, puck_top)
            seen += row["is_puck_seen"]
            self.__move(survey["above"], "rise", "up from the survey pose", spot_index, f"p{survey['pose']:02d}_up", is_rise = True)

        self.__turn_back_lifted(spot_index)
        spot_row.update({"status": "captured", "pictures": len(spot["survey"]), "pictures_with_puck": seen})
        self.spot_rows.append(spot_row)
        self.__write_tables()
        self.__say(f"spot {spot_index} done: the live finder saw the puck in {seen} of {len(spot['survey'])} pictures")


    def __turn_back_lifted(self, spot_index: int) -> None:
        """Turn the forearm back to straight at the travel height over where the tip is now."""

        joints = [float(value) for value in self.robot_link.read_state()["joints"]]
        if abs(joints[3]) < math.radians(0.5):
            return
        tip_xy = magnet_tip_and_lean(joints)[0][:2]
        for move_index, target in enumerate(self.planner.travel_joints(joints, tip_xy, 0.0, self.__reference_top_z(tip_xy)), start = 1):
            self.__move(target, "travel", "forearm back at travel height", spot_index, f"out_{move_index}")


    def __go_home(self) -> None:
        """
        To the parking pose from wherever the arm is: first straight up to the travel height over the tip when it is
        lower (stopping once at "rise_step_clearance_m" when it starts closer than that, as from the hover), then the
        turn back at height, then the checked route home.
        """

        joints = [float(value) for value in self.robot_link.read_state()["joints"]]
        geometry = block_geometry(joints, self.capture_parameters["block_footprint_radius_m"])
        tip_xy = geometry["tip"][:2]
        reference_top_z = self.__reference_top_z(tip_xy)
        clearance_now = geometry["lowest_z"] - reference_top_z
        if clearance_now < self.capture_parameters["travel_clearance_m"] - 0.005:
            clearances = [self.capture_parameters["travel_clearance_m"]]
            if clearance_now < self.capture_parameters["rise_step_clearance_m"] - 0.002:
                clearances.insert(0, self.capture_parameters["rise_step_clearance_m"])
            for step_index, clearance in enumerate(clearances, start = 1):
                lifted = self.planner.solve_at_clearance(tip_xy, math.degrees(joints[3]), reference_top_z, clearance,
                                                         self.capture_parameters["path_max_lean_deg"], joints[4])
                if lifted is None:
                    raise UnsafeMove("no arm position straight above the tip at the travel height")
                self.__move(lifted, "rise", "straight up before going home", 0, f"home_up_{step_index}", is_rise = True)
                joints = lifted
        self.__turn_back_lifted(0)
        joints = [float(value) for value in self.robot_link.read_state()["joints"]]
        for move_index, target in enumerate(self.planner.home_route(joints, self.placed_puck_tops, reference_top_z), start = 1):
            self.__move(target, "travel", "to the parking pose", 0, f"home_{move_index}")


    def __measure_puck(
            self,
            state: Dict,
            spot_row: Dict
    ) -> numpy.ndarray:
        """
        The puck's top-face centre from the hover the person placed it under: x and y of the magnet tip from the joints
        read back (arm model), z from the table plane there plus the puck's height.
        """

        joints = [float(value) for value in state["joints"]]
        geometry = block_geometry(joints, self.capture_parameters["block_footprint_radius_m"])
        puck_top = puck_top_centre([float(geometry["tip"][0]), float(geometry["tip"][1])])
        model = numpy.array(ned2_fk.fk(joints)["tool"][0])
        robot = numpy.array(state["pose"][:3], dtype = float)
        spot_row.update({"puck_top_x": round(float(puck_top[0]), 5), "puck_top_y": round(float(puck_top[1]), 5),
                         "puck_top_z": round(float(puck_top[2]), 5),
                         "table_z": round(float(puck_top[2]) - PUCK_PARAMETERS["height_m"], 5),
                         "hover_clearance_m": round(geometry["lowest_z"] - float(puck_top[2]), 5),
                         "hover_lean_deg": round(geometry["lean_deg"], 2),
                         "model_error_mm": round(1000.0 * float(numpy.linalg.norm(model - robot)), 2),
                         "placed_time": round(time.time(), 3)})
        spot_row.update({f"joint_{index + 1}": round(value, 5) for index, value in enumerate(joints)})
        spot_row.update({pose_name: round(float(value), 5) for pose_name, value in zip(["x", "y", "z", "roll", "pitch", "yaw"], state["pose"])})
        self.__say(f"puck's top centre taken as x {100 * puck_top[0]:.2f}, y {100 * puck_top[1]:.2f}, z {100 * puck_top[2]:.2f} cm "
                   f"(block {1000 * (geometry['lowest_z'] - puck_top[2]):.1f} mm above it; arm model and robot "
                   f"{spot_row['model_error_mm']:.1f} mm apart at the rod's end)")
        return puck_top


    def __capture(
            self,
            spot_index: int,
            survey: Dict,
            puck_top: numpy.ndarray
    ) -> Dict:
        """Take a fresh picture at a survey pose, read the joints back and log one row; the picture is saved untouched."""

        picture_bytes = self.__fresh_picture()
        capture_time = time.time()
        state = self.robot_link.read_state()
        joints = [float(value) for value in state["joints"]]
        picture_name = f"spot_{spot_index:02d}_pose_{survey['pose']:02d}.jpg"
        (self.output_folder / "pictures" / picture_name).write_bytes(picture_bytes)
        picture_md5 = hashlib.md5(picture_bytes).hexdigest()
        if picture_md5 == self.last_saved_md5:
            self.__say(f"warning: {picture_name} is the same picture as the one saved before it")
        self.last_saved_md5 = picture_md5

        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        sighting = find_red_puck(picture, self.capture_parameters["detector"])
        geometry = block_geometry(joints, self.capture_parameters["block_footprint_radius_m"])
        up, left = picture_axes_xy(joints)
        offset = puck_top[:2] - geometry["tip"][:2]
        joint_error_deg = max(abs(math.degrees(read - commanded)) for read, commanded in zip(joints, survey["joints"]))
        spec = survey["spec"]
        row = {"spot": spot_index, "pose": survey["pose"], "picture": f"pictures/{picture_name}", "picture_md5": picture_md5,
               "capture_time": round(capture_time, 3), "sharpness": round(sharpness_of(picture), 1),
               "puck_top_x": round(float(puck_top[0]), 5), "puck_top_y": round(float(puck_top[1]), 5),
               "puck_top_z": round(float(puck_top[2]), 5), "table_z": round(float(puck_top[2]) - PUCK_PARAMETERS["height_m"], 5),
               "planned_twist_deg": spec["twist_deg"], "planned_ahead_m": spec["ahead_m"], "planned_left_m": spec["left_m"],
               "planned_clearance_m": spec["clearance_m"],
               "twist_deg": round(math.degrees(joints[3]), 2), "lean_deg": round(geometry["lean_deg"], 2),
               "clearance_m": round(geometry["lowest_z"] - float(puck_top[2]), 5),
               "tip_x": round(float(geometry["tip"][0]), 5), "tip_y": round(float(geometry["tip"][1]), 5),
               "tip_z": round(float(geometry["tip"][2]), 5),
               "puck_ahead_m": round(float(offset @ up), 5), "puck_left_m": round(float(offset @ left), 5),
               "picture_heading_deg": round(math.degrees(forearm_camera_heading_rad(joints)), 2),
               "joint_error_max_deg": round(joint_error_deg, 3),
               "is_puck_seen": int(sighting is not None),
               "puck_u": round(sighting["u"], 1) if sighting else "", "puck_v": round(sighting["v"], 1) if sighting else "",
               "puck_area_px": sighting["area_px"] if sighting else "",
               "puck_touches_edge": int(sighting["touches_edge"]) if sighting else ""}
        row.update({f"joint_{index + 1}": round(value, 5) for index, value in enumerate(joints)})
        row.update({f"commanded_joint_{index + 1}": round(value, 5) for index, value in enumerate(survey["joints"])})
        row.update({pose_name: round(float(value), 5) for pose_name, value in zip(["x", "y", "z", "roll", "pitch", "yaw"], state["pose"])})
        self.capture_rows.append(row)

        where = f"puck seen at ({row['puck_u']:.0f}, {row['puck_v']:.0f}) px" if sighting else "PUCK NOT SEEN by the live finder"
        self.__say(f"[spot {spot_index} pose {survey['pose']:02d}] twist {row['twist_deg']:+.0f} deg, lean {row['lean_deg']:.1f} deg, "
                   f"block {100 * row['clearance_m']:.1f} cm above the puck, puck {100 * row['puck_ahead_m']:.1f} cm ahead "
                   f"and {100 * row['puck_left_m']:+.1f} cm left in the picture; {where}")
        self.__show(picture, f"spot {spot_index} pose {survey['pose']}: {where}. Esc = stop and go home", sighting)
        return row


    def __move(
            self,
            target_joints: List[float],
            kind: str,
            label: str,
            spot_index: int = 0,
            move: str = "",
            is_rise: bool = False
    ) -> None:
        """
        Check the straight joint-space line from the joints read back now to ``target_joints`` against every puck place
        and the table, then move and check that the robot arrived.

        Raises
        ------
        UnsafeMove
            When the check fails (nothing moves) or the robot stopped more than "joint_stop_deg" short.
        StopRequested
            When Esc was pressed in the picture window (not during the retreat).
        """

        self.__poll_stop_key()
        current = [float(value) for value in self.robot_link.read_state()["joints"]]
        target = [float(value) for value in target_joints]
        check = check_joint_path(current, target, self.placed_puck_tops, self.capture_parameters, self.spot_puck_top, is_rise)
        self.move_rows.append(move_row("run", spot_index, move or f"move_{len(self.move_rows) + 1}", kind, label, check))
        if not check["is_safe"]:
            raise UnsafeMove(f"refused before moving ({label}): {'; '.join(check['reasons'])}")

        # Set before the call: a Ctrl-C during the very first move must still lead to the retreat.
        self.has_moved = True
        self.robot_link.move_joints(target)
        arrived = [float(value) for value in self.robot_link.read_state()["joints"]]
        error_deg = max(abs(math.degrees(read - wanted)) for read, wanted in zip(arrived, target))
        if error_deg > self.capture_parameters["joint_stop_deg"]:
            raise UnsafeMove(f"the robot stopped {error_deg:.1f} degrees short of the target ({label}); collision stop?")
        if error_deg > self.capture_parameters["joint_warning_deg"]:
            self.__say(f"warning: the robot arrived {error_deg:.2f} degrees off the target ({label})")
        self.__poll_stop_key()


    def __retreat(self, reason: str) -> None:
        """
        Rise, turn the forearm back at height and go to the parking pose; never raises. A second Ctrl-C, or a step the
        check refuses, leaves the arm where it is with the motors on.
        """

        self.is_retreating = True
        try:
            if not self.has_moved:
                self.summary["retreat"] = "not needed: the arm had not moved"
                self.__say("the arm has not moved; it stays where it is")
                return
            if self.is_person_near_arm and not self.is_simulated:
                self.__ask("Hands away from the robot? Enter = it rises and goes to its parking pose", ["enter"])
            self.is_person_near_arm = False
            self.__say(f"after {reason}: rising, turning the forearm back at height, going to the parking pose")
            self.__go_home()
            self.summary["retreat"] = "completed: the arm is at its parking pose"
            self.__say("the arm is at its parking pose")
        except KeyboardInterrupt:
            self.summary["retreat"] = "stopped by a second Ctrl-C"
            self.__say("second Ctrl-C: the arm stays where it is, motors on. Bring it home with the robot's own controls.")
        except Exception as error:
            self.summary["retreat"] = f"stopped: {type(error).__name__}: {error}"
            self.__say(f"THE SAFE RETREAT STOPPED: {error}. The arm stays where it is, motors on. Do not push it; bring it "
                       f"up and home with the robot's own controls (or FreeMotion with a hand on the arm).")


    def __reference_top_z(self, tip_xy: numpy.ndarray) -> float:
        """The highest top of any puck place in use, or of a puck standing under ``tip_xy`` when none is in use yet."""

        if self.placed_puck_tops:
            return max(float(puck_top[2]) for puck_top in self.placed_puck_tops)
        return float(puck_top_centre([float(tip_xy[0]), float(tip_xy[1])])[2])


    def __fresh_picture(self) -> bytes:
        """
        Let the arm settle, then wait for "steps_until_fresh_picture" new pictures and take the last one, so no
        picture taken during the move is saved (the recording tool's stale-frame guard); raises when none arrives.
        """

        time.sleep(self.capture_parameters["settle_s"])
        previous = self.robot_link.read_picture_bytes()
        new_pictures = 0
        deadline = time.monotonic() + self.capture_parameters["fresh_picture_timeout_s"]
        while new_pictures < self.capture_parameters["steps_until_fresh_picture"]:
            if time.monotonic() > deadline:
                raise RuntimeError("no new camera picture arrived; is the camera plugged in and streaming?")
            time.sleep(0.01)
            picture_bytes = self.robot_link.read_picture_bytes()
            if picture_bytes != previous:
                new_pictures += 1
                previous = picture_bytes
        return previous


    def __save_camera_check(self, file_name: str) -> None:
        """Save a picture at the parking pose; at the end, compare the hand's strip with the one taken at the start."""

        picture_bytes = self.__fresh_picture()
        (self.output_folder / file_name).write_bytes(picture_bytes)
        start_path = self.output_folder / "camera_check_start.jpg"
        if file_name == "camera_check_start.jpg" or not start_path.exists():
            return
        shift_x, shift_y = camera_shift_px(start_path.read_bytes(), picture_bytes, self.capture_parameters["camera_check_region"])
        shift = math.hypot(shift_x, shift_y)
        self.summary["camera_shift_start_to_end_px"] = round(shift, 2)
        is_warning = shift > self.capture_parameters["camera_shift_warning_px"]
        self.__say(f"camera check: the hand's strip moved {shift:.1f} px between the start and the end at the parking pose"
                   f"{'; THE CAMERA MAY HAVE MOVED ON ITS MOUNT, check it before using these pictures' if is_warning else ''}")


    def __simulate_placing_the_puck(self) -> None:
        """Simulation only: the simulated person puts the puck under the tip, a little off by eye."""

        joints = self.robot_link.read_state()["joints"]
        tip_xy = magnet_tip_and_lean(joints)[0][:2]
        error = self.capture_parameters["simulated_placement_error_m"]
        angle = self.randomness_generator.uniform(0.0, 2 * math.pi)
        self.robot_link.puck_xy = numpy.array([tip_xy[0] + error * math.cos(angle), tip_xy[1] + error * math.sin(angle)])
        self.simulated_puck_placed = numpy.array(self.robot_link.puck_xy)


    def __ask(
            self,
            message: str,
            accepted_keys: List[str]
    ) -> str:
        """Show ``message`` under the live picture and wait for one of ``accepted_keys``."""

        self.__say(message)
        if self.is_window_shown:
            picture = cv2.imdecode(numpy.frombuffer(self.robot_link.read_picture_bytes(), numpy.uint8), cv2.IMREAD_COLOR)
            self.__show(picture, message, None)
        while True:
            key = self.key_reader(0)
            if key in accepted_keys:
                return key


    def __poll_stop_key(self) -> None:
        """Raise StopRequested when Esc was pressed in the picture window; other keys pressed meanwhile are dropped."""

        if not self.is_window_shown or self.is_retreating:
            return
        while True:
            code = cv2.waitKey(1)
            if code < 0:
                return
            if code & 0xFF == 27:
                raise StopRequested()


    def __show(
            self,
            picture: numpy.ndarray,
            status: str,
            sighting: Optional[Dict]
    ) -> None:
        """Show the picture with a status bar below it; the found puck is ringed on the shown copy only."""

        if not self.is_window_shown:
            return
        shown = picture.copy()
        if sighting is not None:
            radius = int(round(math.sqrt(sighting["area_px"] / math.pi))) + 6
            cv2.circle(shown, (int(round(sighting["u"])), int(round(sighting["v"]))), radius, (0, 230, 255), 2)
        cv2.imshow(WINDOW_NAME, numpy.vstack([shown, status_bar_image(status, shown.shape[1])]))
        cv2.waitKey(1)


    def __say(self, message: str) -> None:
        """Print ``message`` and keep it in log.txt."""

        print(message)
        self.log_file.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} {message}\n")
        self.log_file.flush()


    def __write_tables(self) -> None:

        write_rows(self.output_folder / "captures.csv", CAPTURE_COLUMNS, self.capture_rows)
        write_rows(self.output_folder / "spots.csv", SPOT_COLUMNS, self.spot_rows)
        write_rows(self.output_folder / "moves.csv", MOVE_COLUMNS, self.plan["moves"] + self.move_rows)


    def __finish(self, started: datetime.datetime) -> None:
        """Write every table and calibration_run.json, close the window and the robot connection."""

        try:
            self.__write_tables()
            near = [row["min_near_puck_clearance_m"] for row in self.move_rows if row["min_near_puck_clearance_m"] != ""]
            above_spot = [row["min_clearance_above_spot_puck_m"] for row in self.move_rows if row["min_clearance_above_spot_puck_m"] != ""]
            self.summary.update({
                "finished": datetime.datetime.now().isoformat(timespec = "seconds"),
                "minutes": round((datetime.datetime.now() - started).total_seconds() / 60.0, 1),
                "spots_captured": sum(1 for row in self.spot_rows if row.get("status") == "captured"),
                "pictures": len(self.capture_rows),
                "pictures_with_puck_seen": sum(row["is_puck_seen"] for row in self.capture_rows),
                "moves_run": sum(1 for row in self.move_rows if row["is_safe"]),
                "moves_refused": sum(1 for row in self.move_rows if not row["is_safe"]),
                "lowest_block_above_a_puck_within_4_cm_m": min(near) if near else None,
                "lowest_block_above_spot_puck_any_move_m": min(above_spot) if above_spot else None
            })
            if self.is_simulated and self.simulated_puck_placed is not None:
                moved = float(numpy.linalg.norm(numpy.array(self.robot_link.puck_xy) - self.simulated_puck_placed))
                self.summary["simulated_puck_moved_since_last_placing_m"] = round(moved, 6)
            (self.output_folder / "calibration_run.json").write_text(json.dumps(self.summary, indent = 2, default = str), encoding = "utf-8")
            self.__say(f"run {self.summary['status']}: {self.summary['pictures']} pictures, puck seen live in "
                       f"{self.summary['pictures_with_puck_seen']}; files in {self.output_folder}")
        finally:
            self.log_file.close()
            self.robot_link.close()
            if self.is_window_shown:
                cv2.destroyAllWindows()



def write_rows(
        path: Path,
        columns: List[str],
        rows: List[Dict]
) -> None:
    """Write ``rows`` to the CSV file ``path`` with ``columns`` as its header (missing values empty)."""

    with open(path, "w", newline = "", encoding = "utf-8") as file:
        writer = csv.DictWriter(file, fieldnames = columns, extrasaction = "ignore")
        writer.writeheader()
        writer.writerows(rows)


def script_md5() -> str:
    """The md5 of this file, so a run records exactly which version of the routine took it."""

    return hashlib.md5(Path(__file__).read_bytes()).hexdigest()


def new_run_folder(root: Path) -> Path:
    """A new folder named after the minute under ``root``; a suffix is added rather than reusing a folder."""

    name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    folder = root / name
    suffix = 2
    while folder.exists():
        folder = root / f"{name}_{suffix}"
        suffix += 1
    folder.mkdir(parents = True)
    return folder


def summarise_plan(
        plan: Dict,
        capture_parameters: Dict,
        say: Callable[[str], None] = print
) -> Dict:
    """
    Print and return the plan's safety numbers per kind of move (the lowest the block comes above the spot's puck top,
    within 4 cm of any puck place, above the table; the largest lean) and the survey poses as planned.
    """

    def lowest(rows: List[Dict], column: str) -> Optional[float]:
        values = [row[column] for row in rows if row[column] != ""]
        return min(values) if values else None

    def centimetres(value: Optional[float]) -> str:
        return "-" if value is None else f"{100 * value:.1f}"

    summary = {"moves": len(plan["moves"]), "unsafe_moves": sum(1 for row in plan["moves"] if not row["is_safe"]),
               "skipped_poses": plan["skipped_poses"], "by_kind": {}}
    for kind in ("travel", "hover_descend", "hover_rise", "descend", "rise"):
        rows = [row for row in plan["moves"] if row["kind"] == kind]
        if rows:
            summary["by_kind"][kind] = {"moves": len(rows), "lowest_above_spot_puck_m": lowest(rows, "min_clearance_above_spot_puck_m"),
                                        "lowest_within_4_cm_m": lowest(rows, "min_near_puck_clearance_m"),
                                        "lowest_rod_end_within_4_cm_m": lowest(rows, "min_tool_point_near_clearance_m"),
                                        "lowest_above_table_m": lowest(rows, "min_table_clearance_m"),
                                        "largest_lean_deg": max(row["max_lean_deg"] for row in rows),
                                        "largest_rise_sideways_m": max(row["max_rise_sideways_m"] for row in rows)}
    poses = []
    for spot in plan["spots"]:
        for survey in spot["survey"]:
            joints = survey["joints"]
            geometry = block_geometry(joints, capture_parameters["block_footprint_radius_m"])
            tip_xy = geometry["tip"][:2]
            row = {"spot": spot["spot"], "pose": survey["pose"], "twist_deg": round(math.degrees(joints[3]), 1),
                   "ahead_m": survey["spec"]["ahead_m"], "left_m": survey["spec"]["left_m"],
                   "planned_clearance_m": survey["spec"]["clearance_m"],
                   "clearance_m": round(geometry["lowest_z"] - float(spot["puck_top"][2]), 4), "lean_deg": round(geometry["lean_deg"], 1),
                   "tip_x": round(float(tip_xy[0]), 4), "tip_y": round(float(tip_xy[1]), 4), "tip_z": round(float(geometry["tip"][2]), 4),
                   "tip_to_puck_m": round(float(numpy.linalg.norm(spot["puck_top"][:2] - tip_xy)), 4),
                   "heading_deg": round(survey["heading_deg"], 1), "wrist_deg": round(math.degrees(joints[4]), 1)}
            row.update({f"joint_{index + 1}": round(value, 5) for index, value in enumerate(joints)})
            poses.append(row)
    summary["survey_poses"] = poses
    summary["lowest_survey_clearance_m"] = min(pose["clearance_m"] for pose in poses) if poses else None
    summary["largest_survey_lean_deg"] = max(pose["lean_deg"] for pose in poses) if poses else None
    say(f"plan: {summary['moves']} moves, {summary['unsafe_moves']} failing a check, {len(poses)} survey poses, "
        f"{len(plan['skipped_poses'])} skipped")
    for kind, numbers in summary["by_kind"].items():
        say(f"  {kind:13s} {numbers['moves']:3d} moves: block at least {centimetres(numbers['lowest_above_spot_puck_m'])} cm above the "
            f"spot's puck top, {centimetres(numbers['lowest_within_4_cm_m'])} cm within 4 cm of a puck place (rod end "
            f"{centimetres(numbers['lowest_rod_end_within_4_cm_m'])}), {centimetres(numbers['lowest_above_table_m'])} cm above the table; "
            f"lean up to {numbers['largest_lean_deg']:.1f} deg")
    if poses:
        say(f"  survey poses: block {100 * summary['lowest_survey_clearance_m']:.1f} cm or more above the puck's top, "
            f"lean up to {summary['largest_survey_lean_deg']:.1f} deg")
    return summary


def main() -> None:

    parser = argparse.ArgumentParser(description = "Capture pictures of the puck at known places from many arm poses, for fitting the forearm camera.")
    parser.add_argument("--operator", required = True, help = "who runs it, e.g. emranul")
    parser.add_argument("--simulated", action = "store_true", help = "use the simulated robot instead of the real one")
    parser.add_argument("--plan-only", action = "store_true", help = "plan and check every move in the arm model, write the plan, move nothing")
    parser.add_argument("--no-window", action = "store_true", help = "with --simulated only: no picture window, Enter at every question")
    parser.add_argument("--output-root", help = "where the run's new folder goes (default 03_data/raw/camera_calibration; "
                                                "simulated and plan-only: 03_data/raw_simulated/camera_calibration)")
    parser.add_argument("--spots", type = int, help = "use only the first N spots (a short trial)")
    arguments = parser.parse_args()
    if arguments.no_window and not (arguments.simulated or arguments.plan_only):
        parser.error("--no-window answers every question with Enter, so it is only allowed with --simulated or --plan-only")

    overrides = {}
    if arguments.spots:
        overrides["spots_xy"] = CALIBRATION_CAPTURE_PARAMETERS["spots_xy"][:arguments.spots]
    planner = CalibrationPlanner(overrides)
    capture_parameters = planner.capture_parameters
    is_simulated_folder = arguments.simulated or arguments.plan_only
    default_root = capture_parameters["simulated_folder"] if is_simulated_folder else capture_parameters["raw_folder"]
    output_folder = new_run_folder(Path(arguments.output_root) if arguments.output_root else PROJECT_FOLDER / default_root)

    print("Only one program may be connected to the robot at a time: close every viewer and the recording tool first.")
    print("Keep the robot's STOP button in reach. Keep the puck in your hand until the window asks for it.")
    print(f"planning {len(capture_parameters['spots_xy'])} spots x {len(capture_parameters['survey_poses'])} poses in the arm model "
          f"(about 40 s per spot on Emranul's laptop; nothing is connected yet) ...")
    try:
        plan = planner.plan_capture(capture_parameters["spots_xy"])
    except UnsafeMove as error:
        print(f"THE PLAN CANNOT BE BUILT: {error}. Nothing moved.")
        return
    plan_summary = summarise_plan(plan, capture_parameters)
    write_rows(output_folder / "planned_moves.csv", MOVE_COLUMNS, plan["moves"])
    write_rows(output_folder / "planned_poses.csv", POSE_COLUMNS, plan_summary["survey_poses"])
    (output_folder / "plan_summary.json").write_text(json.dumps(plan_summary, indent = 2), encoding = "utf-8")
    if not plan["is_safe"]:
        print(f"THE PLAN FAILS A CLEARANCE CHECK (see {output_folder / 'planned_moves.csv'}); nothing will move.")
        return
    if arguments.plan_only:
        print(f"plan only: nothing moved; plan in {output_folder}")
        return

    if arguments.simulated:
        from robot_link import SimulatedRobotLink
        # The person holds the puck at the start: the simulated puck waits far off the sheet.
        robot_link = SimulatedRobotLink({"puck_xy": [1.0, 1.0], "move_seconds": 0.0})
    else:
        from robot_link import RobotLink
        try:
            robot_link = RobotLink(ROBOT_PARAMETERS["ip_address"], velocity_percent = capture_parameters["velocity_percent"])
        except Exception as error:
            print(f"cannot use the robot: {error}. Nothing moved.")
            return

    is_window_shown = not arguments.no_window
    key_reader = read_window_key if is_window_shown else (lambda wait_ms: "enter")
    capture = ForearmCameraCalibration(robot_link, arguments.operator, plan, output_folder, overrides,
                                       key_reader = key_reader, is_window_shown = is_window_shown, is_simulated = arguments.simulated)
    capture.run()


if __name__ == "__main__":
    main()
