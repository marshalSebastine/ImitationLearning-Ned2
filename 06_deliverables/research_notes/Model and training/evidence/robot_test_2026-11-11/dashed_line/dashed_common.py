"""
Shared helpers of the dashed-line analysis (robot test of 11.11.26, key "dashed_line").

Nothing here connects to the robot: the helpers read the logged runs, the training index and the saved models, and
turn camera pixels into table points with the fitted forearm camera model. The scripts in this folder import this
module; run them with this folder as the working directory, for example

    D:/AI_Project/.venv/Scripts/python.exe 01_shapes_and_overlay.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy
import pandas

EVIDENCE_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = EVIDENCE_FOLDER.parents[5]
CODE_FOLDER = ROOT_FOLDER / "02_code"
sys.path.insert(0, str(CODE_FOLDER))

from config import TRAINING_DATA_PARAMETERS, WORKSPACE_PARAMETERS  # noqa: E402
from forearm_camera_model import project_points, viewing_rays_in_base  # noqa: E402

ANALYSIS_PARAMETERS = {
    "model_runs_folder": Path(TRAINING_DATA_PARAMETERS["processed_folder"]).parents[1] / "model_runs",
    "processed_folder": Path(TRAINING_DATA_PARAMETERS["processed_folder"]),
    "raw_folder": Path(TRAINING_DATA_PARAMETERS["raw_folder"]),
    "gpu_runs_folder": ROOT_FOLDER / "04_runs" / "gpu_ki014",
    "run_7_folder": "2026-11-11_174448_E_sheet05_run1",
    "run_8_folder": "2026-11-11_174936_E_sheet06_run1",
    # the camera shows the black ink as dark blue (hue about 100 to 106 of 180, saturation 150 to 195, value 120 to
    # 165) and the paper as light blue (saturation 55 to 120, value above 200), measured on run 8's pictures 000 to 015;
    # the black pusher block is darker and more purple (hue 110 to 118, value 58 to 84), so it is not ink
    "ink_hue_range": [90, 112],
    "ink_saturation_least": 125,
    "ink_value_least": 95,
    "ink_value_most": 190,
    "paper_value_least": 195,
    "paper_saturation_most": 124,
    # where the robot's own parts always are in the picture (left, top, right, bottom); the red puck is never ink
    "robot_area_px": [185, 335, 640, 480],
    # the top view drawn from the pictures: x and y ranges in metres and the cell size
    "map_x_m": [0.08, 0.52],
    "map_y_m": [-0.30, 0.30],
    "map_cell_m": 0.001
}
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]



def table_points_from_pixels(
        pixels: numpy.ndarray,
        joints: Sequence[float]
) -> numpy.ndarray:
    """
    Return where the rays through pixels meet the taught table plane (tilted, WORKSPACE_PARAMETERS["table_plane"]).

    Parameters
    ----------
    pixels : numpy.ndarray
        Pixels (u, v) of the 640 x 480 picture, shape (N, 2).
    joints : sequence of float
        The six joint angles in radians when the picture was taken.

    Returns
    -------
    numpy.ndarray
        Table points (x, y) in metres in the robot's base frame, shape (N, 2); NaN where the ray misses the table.
    """

    plane = WORKSPACE_PARAMETERS["table_plane"]
    camera_position, rays, is_valid = viewing_rays_in_base(numpy.asarray(pixels, dtype = float), joints)
    denominator = rays[:, 2] - plane["a"] * rays[:, 0] - plane["b"] * rays[:, 1]
    numerator = plane["a"] * camera_position[0] + plane["b"] * camera_position[1] + plane["c"] - camera_position[2]
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        distance = numerator / denominator
    points = camera_position[None, :2] + distance[:, None] * rays[:, :2]
    points[~is_valid | ~numpy.isfinite(distance) | (distance <= 0.0)] = numpy.nan
    return points



def pixels_from_table_points(
        points_xy: numpy.ndarray,
        joints: Sequence[float]
) -> numpy.ndarray:
    """Return the pixels (u, v) where table points (x, y) in metres appear in the picture taken at ``joints``."""

    plane = WORKSPACE_PARAMETERS["table_plane"]
    points_xy = numpy.atleast_2d(numpy.asarray(points_xy, dtype = float))
    height = plane["a"] * points_xy[:, 0] + plane["b"] * points_xy[:, 1] + plane["c"]
    return project_points(numpy.column_stack([points_xy, height]), joints)



def ink_and_paper_masks(
        picture_bgr: numpy.ndarray,
        parameters: Optional[Dict] = None
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Return which pixels show the line's ink, which show white paper and which show the red puck.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The 640 x 480 camera picture as OpenCV reads it.
    parameters : dict, optional
        Overrides of ANALYSIS_PARAMETERS (the colour limits and the robot's area).

    Returns
    -------
    tuple of numpy.ndarray
        Three boolean pictures: ink (outside the robot's area), paper and puck.
    """

    settings = dict(ANALYSIS_PARAMETERS)
    settings.update(parameters or {})
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    is_ink = ((hue >= settings["ink_hue_range"][0]) & (hue <= settings["ink_hue_range"][1])
              & (saturation >= settings["ink_saturation_least"]) & (value >= settings["ink_value_least"])
              & (value <= settings["ink_value_most"]))
    is_paper = (value >= settings["paper_value_least"]) & (saturation <= settings["paper_saturation_most"])
    is_puck = ((hue <= 12) | (hue >= 160)) & (saturation >= 110) & (value >= 60)
    left, top, right, bottom = settings["robot_area_px"]
    is_ink[top:bottom, left:right] = False
    return is_ink, is_paper, is_puck



def read_run(run_folder_name: str) -> pandas.DataFrame:
    """Return steps.csv of a logged model run with the tip in cm, joint 4 in degrees and the asked length added."""

    folder = Path(ANALYSIS_PARAMETERS["model_runs_folder"]) / run_folder_name
    steps = pandas.read_csv(folder / "steps.csv")
    steps["tip_x_cm"] = 100.0 * steps["tip_x"]
    steps["tip_y_cm"] = 100.0 * steps["tip_y"]
    steps["joint_4_deg"] = numpy.degrees(steps["joint_4"])
    steps["asked_length_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
    steps["frame_file"] = [str(folder / name) for name in steps["frame"]]
    return steps



def top_view_of_line(
        picture_paths: List[str],
        joint_rows: numpy.ndarray,
        parameters: Optional[Dict] = None
) -> Dict[str, numpy.ndarray]:
    """
    Draw the line as seen from above by putting many pictures' ink pixels onto the table.

    Every cell of a table grid is projected into every picture; a cell counts as seen when it lands inside the
    picture, outside the robot's area, on ink or paper, and as ink when the pixel there is ink.

    Parameters
    ----------
    picture_paths : list of str
        The pictures (640 x 480 JPEG).
    joint_rows : numpy.ndarray
        The six joint angles in radians of every picture, shape (pictures, 6).
    parameters : dict, optional
        Overrides of ANALYSIS_PARAMETERS (the grid and the colour limits).

    Returns
    -------
    dict
        "ink_votes" and "seen_votes" (2-D arrays indexed [x cell, y cell]), "x_m" and "y_m" (the cell centres).
    """

    settings = dict(ANALYSIS_PARAMETERS)
    settings.update(parameters or {})
    x_m = numpy.arange(settings["map_x_m"][0], settings["map_x_m"][1], settings["map_cell_m"])
    y_m = numpy.arange(settings["map_y_m"][0], settings["map_y_m"][1], settings["map_cell_m"])
    grid_x, grid_y = numpy.meshgrid(x_m, y_m, indexing = "ij")
    points_xy = numpy.column_stack([grid_x.ravel(), grid_y.ravel()])
    ink_votes = numpy.zeros(len(points_xy))
    seen_votes = numpy.zeros(len(points_xy))
    left, top, right, bottom = settings["robot_area_px"]
    for picture_path, joints in zip(picture_paths, joint_rows):
        picture = cv2.imread(str(picture_path))
        is_ink, is_paper, _ = ink_and_paper_masks(picture, settings)
        pixels = pixels_from_table_points(points_xy, list(joints))
        with numpy.errstate(invalid = "ignore"):
            is_inside = (numpy.isfinite(pixels).all(axis = 1) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= 639)
                         & (pixels[:, 1] >= 0) & (pixels[:, 1] <= 479))
        columns = numpy.clip(numpy.round(numpy.nan_to_num(pixels[:, 0])).astype(int), 0, 639)
        rows = numpy.clip(numpy.round(numpy.nan_to_num(pixels[:, 1])).astype(int), 0, 479)
        is_in_robot_area = (columns >= left) & (columns < right) & (rows >= top) & (rows < bottom)
        is_seen = is_inside & ~is_in_robot_area & (is_ink[rows, columns] | is_paper[rows, columns])
        seen_votes += is_seen
        ink_votes += is_seen & is_ink[rows, columns]
    shape = grid_x.shape
    return {"ink_votes": ink_votes.reshape(shape), "seen_votes": seen_votes.reshape(shape), "x_m": x_m, "y_m": y_m}
