"""
Shared readers for the step-length analysis of the robot test of 11.11.26 (analyst key "step_length").

Nothing here connects to the robot: the scripts only read the logs of the ten runs (03_data/model_runs), the training
index (03_data/processed/v01_training_index) and the saved fold models (04_runs/gpu_ki014). Every script of this folder
runs from this folder and writes only into it.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas

EVIDENCE_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = EVIDENCE_FOLDER.parents[5]
CODE_FOLDER = ROOT_FOLDER / "02_code"
if str(CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(CODE_FOLDER))

STEP_LENGTH_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "index_folder": Path("D:/AI_Project/03_data/processed/v01_training_index"),
    "gpu_runs_folder": ROOT_FOLDER / "04_runs" / "gpu_ki014",
    # the ten robot runs of 11.11.26 in the order of the brief: number, folder, model, sheet, and how the brief says
    # the run went ("held" = stayed on the line to the end, "failed" = left the line)
    "robot_runs": [
        {"number": 1, "folder": "2026-11-11_172021_check_no_puck", "model": "A", "sheet": "path_01", "outcome": "held"},
        {"number": 2, "folder": "2026-11-11_172421_E_sheet01_run1", "model": "E", "sheet": "path_01", "outcome": "held"},
        {"number": 3, "folder": "2026-11-11_172743_E_sheet02_run1", "model": "E", "sheet": "path_02", "outcome": "failed"},
        {"number": 4, "folder": "2026-11-11_173036_E_sheet02_run2", "model": "E", "sheet": "path_02", "outcome": "held"},
        {"number": 5, "folder": "2026-11-11_173549_E_sheet03_run1", "model": "E", "sheet": "path_03", "outcome": "failed"},
        {"number": 6, "folder": "2026-11-11_174108_E_sheet04_run1", "model": "E", "sheet": "path_04", "outcome": "failed"},
        {"number": 7, "folder": "2026-11-11_174448_E_sheet05_run1", "model": "E", "sheet": "path_05", "outcome": "held"},
        {"number": 8, "folder": "2026-11-11_174936_E_sheet06_run1", "model": "E", "sheet": "path_06", "outcome": "failed"},
        {"number": 9, "folder": "2026-11-11_175203_E_sheet03_run2_no_hoodie", "model": "E", "sheet": "path_03", "outcome": "failed"},
        {"number": 10, "folder": "2026-11-11_175544_B_team5_sheet03", "model": "B", "sheet": "path_03", "outcome": "failed"}
    ],
    # the demonstrations each robot run is compared with: the standard start on the start mark
    "reference_start_kind": "STD",
    # a twist in place, as the task defines it: the tip moved less than this while the forearm twisted at least that
    "twist_in_place_tip_cm": 0.3,
    "twist_in_place_twist_deg": 3.0
}



def merged_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``STEP_LENGTH_PARAMETERS`` with any overrides merged over it."""

    merged = dict(STEP_LENGTH_PARAMETERS)
    merged.update(parameters or {})
    return merged



def load_robot_run(
        run: Dict,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    Read the steps.csv of one robot run and add what the model asked for and what the driver did.

    Parameters
    ----------
    run : dict
        One entry of "robot_runs".
    parameters : dict, optional
        Overrides of ``STEP_LENGTH_PARAMETERS``.

    Returns
    -------
    pandas.DataFrame
        One row per picture with the extra columns "asked_cm" (the length of the exact step the model gave), "done_cm"
        (how far the tip of the arm model moved to the next picture, NaN on the last row), "planned_cm" (the planned
        step of the driver, 1.0 by construction), "twist_done_deg" (the joint-4 change to the next picture),
        "joint_4_deg" and "travelled_cm" (the path length of the tip up to this picture).
    """

    parameters = merged_parameters(parameters)
    steps = pandas.read_csv(Path(parameters["model_runs_folder"]) / run["folder"] / "steps.csv")
    steps["asked_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
    moved_x = steps["tip_x"].shift(-1) - steps["tip_x"]
    moved_y = steps["tip_y"].shift(-1) - steps["tip_y"]
    steps["done_cm"] = 100.0 * numpy.hypot(moved_x, moved_y)
    steps["planned_cm"] = 100.0 * numpy.hypot(steps["tip_next_x"] - steps["tip_x"], steps["tip_next_y"] - steps["tip_y"])
    steps["joint_4_deg"] = numpy.degrees(steps["joint_4"])
    steps["twist_done_deg"] = steps["joint_4_deg"].shift(-1) - steps["joint_4_deg"]
    steps["travelled_cm"] = numpy.r_[0.0, numpy.nancumsum(steps["done_cm"].to_numpy()[:-1])]
    return steps



def load_demonstration_frames(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    Read frames.csv as recorded (no cleaning), sorted by episode and row, with the step to the next row added.

    Returns
    -------
    pandas.DataFrame
        The rows of frames.csv with "step_cm" (the labelled tip move of the row, from tip_forward_m and tip_left_m),
        "twist_deg" (forearm_change_deg), "joint_4_deg" and "travelled_cm" (the path length of the tip up to the row,
        from tip_x_m and tip_y_m).
    """

    parameters = merged_parameters(parameters)
    columns = ["episode_id", "episode_folder", "row_in_episode", "sheet", "plan_number", "start_kind", "start_end", "plan_end",
               "action", "tip_forward_m", "tip_left_m", "forearm_change_deg", "joint_4", "tip_x_m", "tip_y_m",
               "picture_heading_deg", "move_length_m", "waypoint", "score", "is_scored", "is_mistake_row", "frame_index",
               "frame_path", "started", "hand_corrections", "steps_to_end", "goal_distance_cm"]
    frames = pandas.read_csv(Path(parameters["index_folder"]) / "frames.csv", usecols = columns, low_memory = False)
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    frames["step_cm"] = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    frames["twist_deg"] = frames["forearm_change_deg"].fillna(0.0)
    frames["joint_4_deg"] = numpy.degrees(frames["joint_4"])
    grouped = frames.groupby("episode_id", sort = False)
    moved_cm = 100.0 * numpy.hypot(frames["tip_x_m"] - grouped["tip_x_m"].shift(1), frames["tip_y_m"] - grouped["tip_y_m"].shift(1)).fillna(0.0)
    frames["travelled_cm"] = moved_cm.groupby(frames["episode_id"], sort = False).cumsum()
    return frames



def corrected_start_ends(frames: pandas.DataFrame) -> pandas.DataFrame:
    """
    Replace "start_end" by the corrected end of the data audit where the audit lists one (the same table the training
    cleaning uses), so that the demonstrations of one end are compared with a robot run from that end.
    """

    from config import TRAINING_TARGET_PARAMETERS

    corrected = pandas.read_csv(TRAINING_TARGET_PARAMETERS["audit_cleaning"]["corrected_ends_path"]).set_index("episode_id")["corrected_end"]
    frames = frames.copy()
    frames["start_end"] = frames["episode_id"].map(corrected).fillna(frames["start_end"])
    return frames



def polyline_distance_cm(
        points_xy_m: numpy.ndarray,
        line_xy_m: numpy.ndarray
) -> numpy.ndarray:
    """
    Distance of each point from a polyline, and where along the polyline its nearest point lies.

    Parameters
    ----------
    points_xy_m : numpy.ndarray
        (points, 2) in metres.
    line_xy_m : numpy.ndarray
        (vertices, 2) in metres, in path order.

    Returns
    -------
    numpy.ndarray
        (points, 2): the distance in cm and the arc length of the nearest point in cm.
    """

    starts = line_xy_m[:-1]
    segments = line_xy_m[1:] - starts
    lengths = numpy.linalg.norm(segments, axis = 1)
    arc_starts = numpy.r_[0.0, numpy.cumsum(lengths)[:-1]]
    results = numpy.zeros((len(points_xy_m), 2))
    for index, point in enumerate(points_xy_m):
        along = numpy.clip(((point - starts) * segments).sum(axis = 1) / numpy.maximum(lengths ** 2, 1e-18), 0.0, 1.0)
        nearest = starts + along[:, None] * segments
        distances = numpy.linalg.norm(nearest - point, axis = 1)
        best = int(numpy.argmin(distances))
        results[index] = (100.0 * distances[best], 100.0 * (arc_starts[best] + along[best] * lengths[best]))
    return results



def print_and_keep(
        lines: List[str],
        text: str = ""
) -> None:
    """Print ``text`` and keep it in ``lines``, so that the printed output of a script can be saved as a text file."""

    print(text)
    lines.append(text)
