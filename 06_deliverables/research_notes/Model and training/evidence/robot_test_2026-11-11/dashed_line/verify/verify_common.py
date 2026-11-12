"""
Shared helpers of the verifier (my own code, written without the analyst's modules).

Only the project's fitted camera model (02_code/forearm_camera_model.py) and config.py are imported; the ink and puck
detection, the top view, the placing of the sheet's centre line and the distances are written here from scratch. The
sheet's shape comes from the Recording session's course-photo centre lines
(06_deliverables/research_notes/Recording tool/2026-11-06_sheet_measurements.json), not from the analyst's
reference_line files. Nothing here connects to the robot.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy
from scipy import ndimage, optimize, spatial

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
CODE_FOLDER = ROOT_FOLDER / "02_code"
sys.path.insert(0, str(CODE_FOLDER))

from config import PUCK_TRACKING_PARAMETERS, WORKSPACE_PARAMETERS  # noqa: E402
from forearm_camera_model import viewing_rays_in_base  # noqa: E402

VERIFY_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "index_folder": Path("D:/AI_Project/03_data/processed/v01_training_index"),
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "sheet_measurements_path": Path("D:/AI_Project/06_deliverables/research_notes/Recording tool/2026-11-06_sheet_measurements.json"),
    "run_7_folder": "2026-11-11_174448_E_sheet05_run1",
    "run_8_folder": "2026-11-11_174936_E_sheet06_run1",
    # my own colour limits, chosen from v03_colour_samples.py (ink: hue 96 to 108, saturation about 170, value 100 to
    # 195, blue minus red 43 to 98; paper: value above 200, saturation 57 to 103)
    "ink_hue": [92, 113],
    "ink_saturation_least": 120,
    "ink_value": [85, 196],
    "ink_blue_minus_red_least": 40,
    "paper_value_least": 197,
    "paper_saturation_most": 125,
    "puck_saturation_least": 90,
    "puck_value_least": 35,
    # the robot's own parts fill the bottom of every picture: rows below this are never used for the top view
    "usable_rows_most": 285,
    "pixel_stride": 2,
    "map_x_m": [0.04, 0.56],
    "map_y_m": [-0.36, 0.36],
    "map_cell_m": 0.002,
    "puck_point_height_m": float(PUCK_TRACKING_PARAMETERS["puck_point_height_above_table_m"]),
    "chamfer_cap_m": 0.015
}



def pixels_on_table(
        pixels: numpy.ndarray,
        joints: Sequence[float],
        height_above_table_m: float = 0.0
) -> numpy.ndarray:
    """Return where the rays through pixels (u, v) meet the taught, tilted table plane raised by a height (metres)."""

    plane = WORKSPACE_PARAMETERS["table_plane"]
    camera_position, rays, is_valid = viewing_rays_in_base(numpy.asarray(pixels, dtype = float), list(joints))
    denominator = rays[:, 2] - plane["a"] * rays[:, 0] - plane["b"] * rays[:, 1]
    numerator = (plane["a"] * camera_position[0] + plane["b"] * camera_position[1] + plane["c"] + height_above_table_m
                 - camera_position[2])
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        distance = numerator / denominator
    points = camera_position[None, :2] + distance[:, None] * rays[:, :2]
    points[~is_valid | ~numpy.isfinite(distance) | (distance <= 0.0)] = numpy.nan
    return points



def colour_masks(picture_bgr: numpy.ndarray) -> Dict[str, numpy.ndarray]:
    """Return boolean pictures "ink", "paper" and "puck" (my own colour rules)."""

    settings = VERIFY_PARAMETERS
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    blue_minus_red = picture_bgr[:, :, 0].astype(int) - picture_bgr[:, :, 2].astype(int)
    is_ink = ((hue >= settings["ink_hue"][0]) & (hue <= settings["ink_hue"][1])
              & (saturation >= settings["ink_saturation_least"])
              & (value >= settings["ink_value"][0]) & (value <= settings["ink_value"][1])
              & (blue_minus_red >= settings["ink_blue_minus_red_least"]))
    is_paper = (value >= settings["paper_value_least"]) & (saturation <= settings["paper_saturation_most"])
    # the puck in the block's shadow is dark and purple (hue about 140, value about 48, red 31 above green; measured on
    # demonstration 06-03 row 35), so the rule is wider than "red and bright" and asks for red well above green instead
    red_minus_green = picture_bgr[:, :, 2].astype(int) - picture_bgr[:, :, 1].astype(int)
    is_puck = (((hue <= 8) | (hue >= 135)) & (saturation >= settings["puck_saturation_least"])
               & (value >= settings["puck_value_least"]) & (red_minus_green >= 20))
    # ink counts only inside the sheet: the paper mask closed over 41 px, so that dark blue things beside the table
    # (seen in run 7 at the picture's edge) are not taken for the line
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    is_sheet = cv2.morphologyEx(is_paper.astype(numpy.uint8), cv2.MORPH_CLOSE, kernel) > 0
    is_ink = is_ink & is_sheet
    return {"ink": is_ink, "paper": is_paper, "puck": is_puck}



def puck_pixel(is_puck: numpy.ndarray) -> Optional[Tuple[float, float, int]]:
    """Return the centre (u, v) and the pixel count of the largest red blob, or None when it is smaller than 150 px."""

    cleaned = cv2.morphologyEx(is_puck.astype(numpy.uint8), cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity = 8)
    if count <= 1:
        return None
    largest = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    if statistics[largest, cv2.CC_STAT_AREA] < 150:
        return None
    return float(centroids[largest, 0]), float(centroids[largest, 1]), int(statistics[largest, cv2.CC_STAT_AREA])



def top_view(
        picture_paths: Sequence[str],
        joint_rows: numpy.ndarray
) -> Dict[str, numpy.ndarray]:
    """
    Draw the ink from above: every usable pixel of every picture is put on the table and counted in a 2 mm grid.

    Returns "ink" and "seen" (counts of pictures per cell, indexed [x cell, y cell]) and the grid's "x_m" and "y_m" edges.
    """

    settings = VERIFY_PARAMETERS
    stride = settings["pixel_stride"]
    columns, rows = numpy.meshgrid(numpy.arange(0, 640, stride), numpy.arange(0, settings["usable_rows_most"], stride))
    pixels = numpy.column_stack([columns.ravel(), rows.ravel()]).astype(float)
    x_edges = numpy.arange(settings["map_x_m"][0], settings["map_x_m"][1] + 1e-9, settings["map_cell_m"])
    y_edges = numpy.arange(settings["map_y_m"][0], settings["map_y_m"][1] + 1e-9, settings["map_cell_m"])
    ink = numpy.zeros((len(x_edges) - 1, len(y_edges) - 1))
    seen = numpy.zeros_like(ink)
    for picture_path, joints in zip(picture_paths, joint_rows):
        picture = cv2.imread(str(picture_path))
        masks = colour_masks(picture)
        wide_puck = cv2.dilate(masks["puck"].astype(numpy.uint8), numpy.ones((9, 9), numpy.uint8)) > 0
        is_ink = masks["ink"][rows.ravel(), columns.ravel()]
        is_usable = (masks["ink"] | masks["paper"])[rows.ravel(), columns.ravel()] & ~wide_puck[rows.ravel(), columns.ravel()]
        points = pixels_on_table(pixels, joints)
        is_usable &= numpy.isfinite(points).all(axis = 1)
        seen_here, _, _ = numpy.histogram2d(points[is_usable, 0], points[is_usable, 1], bins = [x_edges, y_edges])
        ink_here, _, _ = numpy.histogram2d(points[is_usable & is_ink, 0], points[is_usable & is_ink, 1], bins = [x_edges, y_edges])
        seen += seen_here > 0
        ink += (ink_here > 0) & (ink_here >= 0.5 * seen_here)
    return {"ink": ink, "seen": seen, "x_m": x_edges, "y_m": y_edges}



def ink_cells(view: Dict[str, numpy.ndarray], share_least: float = 0.5, seen_least: int = 2) -> numpy.ndarray:
    """Return the boolean grid of cells that count as ink (seen often enough and ink in most of those pictures)."""

    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        share = numpy.where(view["seen"] > 0, view["ink"] / numpy.maximum(view["seen"], 1), 0.0)
    is_ink = (view["seen"] >= seen_least) & (share >= share_least)
    return ndimage.binary_opening(is_ink, structure = numpy.ones((2, 2)))



def sheet_line(sheet: str, spacing_cm: float = 0.1) -> numpy.ndarray:
    """Return the course-photo centre line of a sheet ("05" or "06") in metres on the sheet, resampled evenly."""

    measurements = json.loads(Path(VERIFY_PARAMETERS["sheet_measurements_path"]).read_text())
    line = numpy.asarray(measurements[sheet]["line_xy_cm"], dtype = float)
    lengths = numpy.concatenate([[0.0], numpy.cumsum(numpy.hypot(*numpy.diff(line, axis = 0).T))])
    wanted = numpy.arange(0.0, lengths[-1], spacing_cm)
    resampled = numpy.column_stack([numpy.interp(wanted, lengths, line[:, 0]), numpy.interp(wanted, lengths, line[:, 1])])
    return resampled / 100.0



def placed_line(line_sheet_m: numpy.ndarray, placement: Sequence[float], is_mirrored: bool) -> numpy.ndarray:
    """Return the sheet's line on the table for a placement (angle in radians, shift x, shift y in metres)."""

    centred = line_sheet_m - line_sheet_m.mean(axis = 0)
    if is_mirrored:
        centred = centred * numpy.array([1.0, -1.0])
    angle = placement[0]
    rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
    return centred @ rotation.T + numpy.asarray(placement[1:3])



def register_line(
        line_sheet_m: numpy.ndarray,
        view: Dict[str, numpy.ndarray],
        first_guess: Optional[Tuple[float, float, float, bool]] = None
) -> Dict:
    """
    Lay the sheet's centre line onto the top view (two-sided capped chamfer distance, coarse angles then Nelder-Mead).

    Returns "placement" (angle rad, x, y), "is_mirrored", "line_m", "cost", "ink_to_line_mm" (median distance of ink
    cells to the line), "line_on_ink_share" (share of seen line points with ink within 3 mm).
    """

    settings = VERIFY_PARAMETERS
    cell = settings["map_cell_m"]
    is_ink = ink_cells(view)
    x_centres = 0.5 * (view["x_m"][:-1] + view["x_m"][1:])
    y_centres = 0.5 * (view["y_m"][:-1] + view["y_m"][1:])
    ink_x, ink_y = numpy.nonzero(is_ink)
    ink_points = numpy.column_stack([x_centres[ink_x], y_centres[ink_y]])
    distance_to_ink = ndimage.distance_transform_edt(~is_ink) * cell
    is_seen = view["seen"] >= 2
    cap = settings["chamfer_cap_m"]

    def lookup(points: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        x_index = numpy.floor((points[:, 0] - view["x_m"][0]) / cell).astype(int)
        y_index = numpy.floor((points[:, 1] - view["y_m"][0]) / cell).astype(int)
        is_inside = (x_index >= 0) & (x_index < is_ink.shape[0]) & (y_index >= 0) & (y_index < is_ink.shape[1])
        x_index = numpy.clip(x_index, 0, is_ink.shape[0] - 1)
        y_index = numpy.clip(y_index, 0, is_ink.shape[1] - 1)
        return distance_to_ink[x_index, y_index], is_inside & is_seen[x_index, y_index]

    def cost(parameters: numpy.ndarray, is_mirrored: bool) -> float:
        line = placed_line(line_sheet_m, parameters, is_mirrored)
        tree = spatial.cKDTree(line)
        ink_distance, _ = tree.query(ink_points)
        line_distance, is_line_seen = lookup(line)
        ink_term = numpy.mean(numpy.minimum(ink_distance, cap) ** 2)
        # line points that were never seen cost a little, so the line is not pushed out of the seen area
        line_term = numpy.mean(numpy.where(is_line_seen, numpy.minimum(line_distance, cap) ** 2, (0.4 * cap) ** 2))
        return float(ink_term + line_term)

    candidates = []
    if first_guess is not None:
        starts = [(first_guess[0], first_guess[1], first_guess[2], first_guess[3])]
    else:
        centre = ink_points.mean(axis = 0)
        starts = [(numpy.radians(angle), centre[0], centre[1], is_mirrored)
                  for is_mirrored in (False, True) for angle in range(0, 360, 10)]
    for angle, shift_x, shift_y, is_mirrored in starts:
        candidates.append((cost(numpy.array([angle, shift_x, shift_y]), is_mirrored), angle, shift_x, shift_y, is_mirrored))
    candidates.sort(key = lambda candidate: candidate[0])
    best = None
    for _, angle, shift_x, shift_y, is_mirrored in candidates[:8]:
        simplex = numpy.array([[angle, shift_x, shift_y], [angle + 0.1, shift_x, shift_y],
                               [angle, shift_x + 0.02, shift_y], [angle, shift_x, shift_y + 0.02]])
        result = optimize.minimize(cost, numpy.array([angle, shift_x, shift_y]), args = (is_mirrored,), method = "Nelder-Mead",
                                   options = {"xatol": 1e-4, "fatol": 1e-10, "maxiter": 600, "initial_simplex": simplex})
        if best is None or result.fun < best[0]:
            best = (float(result.fun), result.x, is_mirrored)
    line = placed_line(line_sheet_m, best[1], best[2])
    ink_distance, _ = spatial.cKDTree(line).query(ink_points)
    line_distance, is_line_seen = lookup(line)
    return {
        "placement": [float(value) for value in best[1]], "is_mirrored": bool(best[2]), "line_m": line, "cost": best[0],
        "ink_to_line_mm": float(1000.0 * numpy.median(ink_distance)),
        "ink_to_line_90_mm": float(1000.0 * numpy.percentile(ink_distance, 90)),
        "line_seen_share": float(is_line_seen.mean()),
        "line_on_ink_share": float((line_distance[is_line_seen] <= 0.003).mean()) if is_line_seen.any() else float("nan"),
        "ink_cells": int(len(ink_points))
    }



def crossing_indices(line: numpy.ndarray, least_apart: int = 30) -> Optional[Tuple[int, int]]:
    """Return the two indices where an evenly sampled line passes its own crossing (closest pair far apart along it)."""

    tree = spatial.cKDTree(line)
    best = None
    for first, neighbours in enumerate(tree.query_ball_point(line, r = 0.004)):
        for second in neighbours:
            if second - first >= least_apart:
                distance = float(numpy.hypot(*(line[first] - line[second])))
                if best is None or distance < best[0]:
                    best = (distance, first, second)
    return None if best is None else (best[1], best[2])



def along_and_beside(
        line: numpy.ndarray,
        points: numpy.ndarray,
        back_most: int = 30,
        ahead_most: int = 60,
        first_index: Optional[int] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Place points one after another on a line sampled every millimetre: the nearest line point within a window around
    the last place (so the two passes of a crossing are kept apart). Returns the index along the line and the signed
    distance beside it in metres (+ = left of the direction of growing index).
    """

    tangent = numpy.gradient(line, axis = 0)
    tangent /= numpy.linalg.norm(tangent, axis = 1)[:, None]
    index = numpy.full(len(points), -1)
    beside = numpy.full(len(points), numpy.nan)
    last = None
    for number, point in enumerate(points):
        if not numpy.all(numpy.isfinite(point)):
            continue
        if last is None and first_index is not None:
            low, high = max(0, first_index - back_most), min(len(line), first_index + ahead_most + 1)
        elif last is None:
            low, high = 0, max(1, len(line) // 3)
        else:
            low, high = max(0, last - back_most), min(len(line), last + ahead_most + 1)
        distances = numpy.hypot(*(line[low:high] - point).T)
        here = low + int(numpy.argmin(distances))
        offset = point - line[here]
        side = tangent[here, 0] * offset[1] - tangent[here, 1] * offset[0]
        index[number] = here
        beside[number] = float(numpy.hypot(*offset) * (1.0 if side >= 0.0 else -1.0))
        last = here
    return index, beside
