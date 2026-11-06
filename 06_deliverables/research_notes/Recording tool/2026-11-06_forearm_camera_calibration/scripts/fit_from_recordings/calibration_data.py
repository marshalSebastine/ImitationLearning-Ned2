"""
Per-picture records of the recorded episodes 037 to 045 for fitting the forearm camera, read without the robot.

Every picture of the episodes becomes one record: the arm's joints when the picture was taken (from steps.csv, or for
episode 041 the joints rebuilt from guided_path.csv by the detector step), the step row (action, source, gamepad
command, waypoint), the red-puck detection of the detector step, and quantities of the arm model: the magnet tip
(``side_view_test.magnet_tip_and_lean``), the rod's lean and the forearm's turn (joint 4). Nothing here talks to the
robot; the arm model is pure Python.
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy

CALIBRATION_DATA_PARAMETERS = {
    "project_code_folder": "D:/AI_Project/02_code",
    "camera_model_folder": "C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/3bc216cd-ac14-4d82-9daf-a61bf72eef65/scratchpad/wf/camera_model",
    "detections_path": "C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/3bc216cd-ac14-4d82-9daf-a61bf72eef65/scratchpad/wf/detector/detections_037_045.json",
    "raw_folder": "D:/AI_Project/03_data/raw",
    # episode -> session folder; 040 has no pictures and is left out
    "episode_folders": {
        "037": "2026-11-04_session_01_emranul/path_04_meander_episode_037",
        "038": "2026-11-04_session_01_emranul/path_04_meander_episode_038",
        "039": "2026-11-04_session_01_emranul/path_04_meander_episode_039",
        "041": "2026-11-04_session_01_emranul/path_04_meander_episode_041",
        "042": "2026-11-05_session_01_emranul/path_04_meander_episode_042",
        "043": "2026-11-05_session_01_emranul/path_04_meander_episode_043",
        "044": "2026-11-05_session_01_emranul/path_04_meander_episode_044",
        "045": "2026-11-05_session_01_emranul/path_04_meander_episode_045"
    },
    # episode 041 crashed before writing episode.json; its mode is in the session notes (test 8, exact replay)
    "mode_without_episode_file": "hand_guided_exact"
}

for module_folder in (CALIBRATION_DATA_PARAMETERS["project_code_folder"], CALIBRATION_DATA_PARAMETERS["camera_model_folder"]):
    if module_folder not in sys.path:
        sys.path.insert(0, module_folder)

from config import WORKSPACE_PARAMETERS, table_height
from side_view_test import magnet_tip_and_lean, ned2_fk

# the table plane z = a x + b y + c taught on the old desk on 25.10.26 (config), where episodes 037 to 045 were recorded
CALIBRATION_TABLE_PLANE = dict(WORKSPACE_PARAMETERS["table_plane"])


NUMERIC_ROW_COLUMNS = ["forearm_turn_deg", "picture_heading_deg", "waypoint", "tip_forward_m", "tip_left_m", "forearm_change_deg",
                       "pad_move_x", "pad_move_y", "pad_twist", "pad_lean", "pad_hand", "pad_up", "pad_down",
                       "tip_height_above_push_m", "x", "y", "z", "roll", "pitch", "yaw", "unix_time", "capture_time"]



def merged_data_parameters(data_parameters: Optional[Dict] = None) -> Dict:
    """
    Return ``CALIBRATION_DATA_PARAMETERS`` with ``data_parameters`` merged over them.

    Parameters
    ----------
    data_parameters : dict, optional
        Overrides.

    Returns
    -------
    dict
        A new dictionary.
    """

    parameters = json.loads(json.dumps(CALIBRATION_DATA_PARAMETERS))
    if data_parameters:
        parameters.update(data_parameters)
    return parameters



def read_step_rows(
        episode: str,
        data_parameters: Dict
) -> Dict[int, Dict]:
    """
    Return the step rows of one episode by frame index, numbers converted to float where present.

    Parameters
    ----------
    episode : str
        The episode number, e.g. "037".
    data_parameters : dict
        The data settings.

    Returns
    -------
    dict of int to dict
        Frame index -> row. Episode 041 has only steps_from_terminal_log.csv (action, source, forearm turn, rod x
        and y in centimetres); its rows are converted to the same names where they exist.
    """

    episode_folder = Path(data_parameters["raw_folder"]) / data_parameters["episode_folders"][episode]
    steps_path = episode_folder / "steps.csv"
    is_terminal_log = not steps_path.exists()
    if is_terminal_log:
        steps_path = episode_folder / "steps_from_terminal_log.csv"

    rows = {}
    with open(steps_path, newline = "", encoding = "utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            for column in NUMERIC_ROW_COLUMNS:
                value = row.get(column)
                converted[column] = float(value) if value not in (None, "") else None
            if is_terminal_log:
                converted["x"] = float(row["x_cm"]) / 100.0
                converted["y"] = float(row["y_cm"]) / 100.0
            frame_index = int(Path(row["frame"]).stem)
            rows[frame_index] = converted
    return rows



def read_episode_mode(
        episode: str,
        data_parameters: Dict
) -> str:
    """
    Return the recording mode of an episode from its episode.json.

    Parameters
    ----------
    episode : str
        The episode number.
    data_parameters : dict
        The data settings.

    Returns
    -------
    str
        "keyboard", "hand_guided", "hand_guided_exact" or "gamepad".
    """

    episode_path = Path(data_parameters["raw_folder"]) / data_parameters["episode_folders"][episode] / "episode.json"
    if not episode_path.exists():
        return data_parameters["mode_without_episode_file"]
    with open(episode_path, encoding = "utf-8") as handle:
        return json.load(handle)["mode"]



def arm_quantities(joints: List[float]) -> Dict:
    """
    Return the arm model's quantities for one set of joints.

    Parameters
    ----------
    joints : list of float
        The six joint angles in radians.

    Returns
    -------
    dict
        "tip_xyz" (the magnet tip in metres), "lean_deg" (the rod's angle from straight down), "rod_end_xyz" (the
        robot's tool point), "forearm_deg" (joint 4 in degrees) and "tip_above_table_m" (the tip's height above the
        table plane in config).
    """

    tip, lean_deg = magnet_tip_and_lean(list(joints))
    rod_end = numpy.array(ned2_fk.fk(list(joints))["tool"][0], dtype = float)
    return {
        "tip_xyz": numpy.asarray(tip, dtype = float),
        "lean_deg": float(lean_deg),
        "rod_end_xyz": rod_end,
        "forearm_deg": math.degrees(float(joints[3])),
        "tip_above_table_m": float(tip[2] - table_height(float(tip[0]), float(tip[1])))
    }



def load_frame_records(data_parameters: Optional[Dict] = None) -> List[Dict]:
    """
    Return one record per picture of the episodes, in episode and frame order.

    Parameters
    ----------
    data_parameters : dict, optional
        Overrides for ``CALIBRATION_DATA_PARAMETERS``.

    Returns
    -------
    list of dict
        Keys: "episode", "frame_index", "picture_path", "day", "mode", "joints", "joints_source", "row" (the step
        row, or None), "detection" (the detector's dictionary, or None), "no_detection_review", and the arm model's
        quantities of ``arm_quantities``.
    """

    parameters = merged_data_parameters(data_parameters)
    with open(parameters["detections_path"], encoding = "utf-8") as handle:
        detections = json.load(handle)

    records = []
    for episode, folder in parameters["episode_folders"].items():
        step_rows = read_step_rows(episode, parameters)
        mode = read_episode_mode(episode, parameters)
        day = folder[:10]
        frame_keys = sorted(key for key, frame in detections["frames"].items() if frame["episode"] == episode)
        for key in frame_keys:
            frame = detections["frames"][key]
            if frame.get("joints") is None:
                continue
            record = {
                "episode": episode,
                "frame_index": int(frame["frame_index"]),
                "picture_path": key,
                "day": day,
                "mode": mode,
                "joints": [float(value) for value in frame["joints"]],
                "joints_source": frame.get("joints_source"),
                "row": step_rows.get(int(frame["frame_index"])),
                "detection": frame.get("detection"),
                "no_detection_review": frame.get("no_detection_review"),
                "start_picture_review": frame.get("start_picture_review")
            }
            record.update(arm_quantities(record["joints"]))
            records.append(record)

    records.sort(key = lambda record: (record["episode"], record["frame_index"]))
    return records



def records_by_episode(records: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group records by episode, each list in frame order.

    Parameters
    ----------
    records : list of dict
        Records from ``load_frame_records``.

    Returns
    -------
    dict of str to list of dict
        Episode -> its records.
    """

    grouped = {}
    for record in records:
        grouped.setdefault(record["episode"], []).append(record)
    for episode_records in grouped.values():
        episode_records.sort(key = lambda record: record["frame_index"])
    return grouped
