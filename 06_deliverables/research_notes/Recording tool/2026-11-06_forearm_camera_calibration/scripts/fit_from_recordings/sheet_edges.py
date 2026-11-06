"""
Find the paper sheet's outer edge in the pictures: the pixels where the white sheet meets the table.

The sheet is taped to the table and does not move during an episode, so its four straight edges are fixed lines on
the table plane, seen from many arm positions. Their pixels are found in four steps: the table is recognised by its
warm colour (orange wood on 5.11, yellow-grey under the blue cast of 4.11), the robot by saturated blue or by being
dark, and the paper as the bright remainder; the largest paper region with its holes filled is taken as the sheet;
its border is kept only where the pixel just outside is table (not the black line, the robot, a shadow, or the
picture's edge); and each kept point is moved, at full size and to a fraction of a pixel, to the steepest fall of
brightness along its outward direction. The points are thinned to one per grid cell. Which side of the sheet a point
belongs to is decided later, on the table, with a camera model.
"""

from typing import Dict, Optional

import cv2
import numpy

SHEET_EDGE_PARAMETERS = {
    # the picture is searched at this scale (the points are returned in full-size pixels)
    "analysis_scale": 0.5,
    "blur_kernel_px": 5,
    # the table is warm-coloured (orange wood on 5.11, yellow-grey under the blue cast of 4.11): OpenCV hue in this
    # range with at least this saturation; the red puck (hue below 8) and everything blue or grey are not table
    "table_hue_range": [8, 45],
    "table_saturation_minimum": 20,
    "table_value_minimum": 45,
    # paper: not table, not robot, and at least this bright relative to the picture's 95th percentile of value
    "paper_relative_value_minimum": 0.62,
    # bright regions smaller than this share of the picture are not the sheet
    "minimum_sheet_share": 0.03,
    "opening_kernel_px": 3,
    # border pixels this close to the picture's edge are dropped (full-size pixels)
    "picture_margin_px": 8,
    # how far outside the border the table is looked for (full-size pixels)
    "outside_step_px": 5,
    # robot parts: saturated blue, or dark; grown by this many pixels (full-size)
    "robot_blue_hue_range": [95, 130],
    "robot_blue_saturation_minimum": 110,
    "robot_dark_value_maximum": 55,
    "robot_growth_px": 6,
    # one point per cell of this size (full-size pixels)
    "thinning_cell_px": 6,
    # sub-pixel refinement at full size along the outward direction: how far either way, in what steps, and how much
    # darker (smallest colour channel, 0 to 255) the table side must be
    "is_refined": True,
    "refine_half_length_px": 6.0,
    "refine_step_px": 0.5,
    "refine_minimum_drop": 25.0
}



def merged_edge_parameters(edge_parameters: Optional[Dict] = None) -> Dict:
    """
    Return ``SHEET_EDGE_PARAMETERS`` with ``edge_parameters`` merged over them.

    Parameters
    ----------
    edge_parameters : dict, optional
        Overrides.

    Returns
    -------
    dict
        A new dictionary.
    """

    parameters = dict(SHEET_EDGE_PARAMETERS)
    if edge_parameters:
        parameters.update(edge_parameters)
    return parameters



def find_sheet_edge_points(
        picture_bgr: numpy.ndarray,
        edge_parameters: Optional[Dict] = None
) -> Dict:
    """
    Return the pixels of the sheet's outer edge in one picture.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The 640 x 480 picture as OpenCV reads it.
    edge_parameters : dict, optional
        Overrides for ``SHEET_EDGE_PARAMETERS``.

    Returns
    -------
    dict
        "points_px" (N, 2) edge pixels (u, v) in the full picture, "outward_px" (N, 2) unit vectors from the paper
        towards the table, "sheet_mask" (the sheet at the analysis scale) and "threshold" (the paper's brightness
        threshold, 0 to 255).
    """

    parameters = merged_edge_parameters(edge_parameters)
    scale = float(parameters["analysis_scale"])
    small = cv2.resize(picture_bgr, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (parameters["blur_kernel_px"], parameters["blur_kernel_px"]), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    low_table_hue, high_table_hue = parameters["table_hue_range"]
    is_table = ((hue >= low_table_hue) & (hue <= high_table_hue) & (saturation >= parameters["table_saturation_minimum"])
                & (value >= parameters["table_value_minimum"]))
    low_hue, high_hue = parameters["robot_blue_hue_range"]
    is_robot = (((hue >= low_hue) & (hue <= high_hue) & (saturation >= parameters["robot_blue_saturation_minimum"]))
                | (value <= parameters["robot_dark_value_maximum"])).astype(numpy.uint8)
    growth = int(round(parameters["robot_growth_px"] * scale))
    if growth > 0:
        is_robot = cv2.dilate(is_robot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * growth + 1, 2 * growth + 1)))
    bright_level = float(numpy.percentile(value, 95))
    is_paper = (~is_table) & (is_robot == 0) & (value >= parameters["paper_relative_value_minimum"] * bright_level)
    bright = is_paper.astype(numpy.uint8) * 255
    threshold = parameters["paper_relative_value_minimum"] * bright_level

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (parameters["opening_kernel_px"], parameters["opening_kernel_px"]))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(bright, connectivity = 8)
    empty = {"points_px": numpy.zeros((0, 2)), "outward_px": numpy.zeros((0, 2)), "sheet_mask": None, "threshold": float(threshold)}
    if count <= 1:
        return empty
    largest = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    if statistics[largest, cv2.CC_STAT_AREA] < parameters["minimum_sheet_share"] * bright.size:
        return empty
    sheet = (labels == largest).astype(numpy.uint8) * 255

    # fill the holes (the black line, the puck) so that only the sheet's outer border remains
    contours, _ = cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = numpy.zeros_like(sheet)
    cv2.drawContours(filled, contours, -1, 255, thickness = cv2.FILLED)

    distance_inside = cv2.distanceTransform((filled > 0).astype(numpy.uint8), cv2.DIST_L2, 3)
    border = (filled > 0) & (distance_inside <= 1.0)
    rows, columns = numpy.nonzero(border)
    if len(rows) == 0:
        return empty

    # the outward direction is the negative gradient of a smoothed sheet mask
    smooth = cv2.GaussianBlur((filled > 0).astype(numpy.float32), (9, 9), 0)
    gradient_u = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize = 3)
    gradient_v = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize = 3)
    outward = -numpy.column_stack([gradient_u[rows, columns], gradient_v[rows, columns]])
    lengths = numpy.linalg.norm(outward, axis = 1)
    is_directed = lengths > 1e-6
    rows, columns, outward = rows[is_directed], columns[is_directed], outward[is_directed] / lengths[is_directed, None]

    step = parameters["outside_step_px"] * scale
    height, width = filled.shape
    outside_u = numpy.clip(numpy.round(columns + step * outward[:, 0]).astype(int), 0, width - 1)
    outside_v = numpy.clip(numpy.round(rows + step * outward[:, 1]).astype(int), 0, height - 1)
    inside_u = numpy.clip(numpy.round(columns - step * outward[:, 0]).astype(int), 0, width - 1)
    inside_v = numpy.clip(numpy.round(rows - step * outward[:, 1]).astype(int), 0, height - 1)
    is_table_outside = (is_table[outside_v, outside_u] & (filled[outside_v, outside_u] == 0)
                        & (is_robot[outside_v, outside_u] == 0) & (is_robot[inside_v, inside_u] == 0))

    full_u = (columns + 0.5) / scale - 0.5
    full_v = (rows + 0.5) / scale - 0.5
    margin = parameters["picture_margin_px"]
    full_height, full_width = picture_bgr.shape[:2]
    is_inside_picture = (full_u >= margin) & (full_u <= full_width - 1 - margin) & (full_v >= margin) & (full_v <= full_height - 1 - margin)
    keep = is_table_outside & is_inside_picture
    points = numpy.column_stack([full_u[keep], full_v[keep]])
    directions = outward[keep]

    # one point per grid cell, so long edges near the camera do not outweigh the far ones
    cell = float(parameters["thinning_cell_px"])
    cells = numpy.floor(points / cell).astype(int)
    _, first_indices = numpy.unique(cells[:, 0] * 10_000 + cells[:, 1], return_index = True)
    first_indices = numpy.sort(first_indices)
    points, directions = points[first_indices], directions[first_indices]
    if parameters["is_refined"]:
        points, directions = refine_edge_points(picture_bgr, points, directions, parameters)
    return {"points_px": points, "outward_px": directions, "sheet_mask": filled, "threshold": float(threshold)}



def refine_edge_points(
        picture_bgr: numpy.ndarray,
        points: numpy.ndarray,
        directions: numpy.ndarray,
        edge_parameters: Dict
) -> tuple:
    """
    Move each edge point to the steepest fall of brightness along its outward direction, to a fraction of a pixel.

    The coarse points come from a mask at half size whose threshold depends on the picture, so they sit up to a
    couple of pixels inside the paper. Along the outward line the smallest colour channel (bright on white paper,
    darker on the table) is sampled at full size; the edge is where it falls fastest, refined with a parabola through
    the three samples around the steepest one. Points without a clear fall are dropped.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The full-size picture.
    points, directions : numpy.ndarray
        Coarse edge points (N, 2) and unit outward directions (N, 2).
    edge_parameters : dict
        The edge settings ("refine_half_length_px", "refine_step_px", "refine_minimum_drop").

    Returns
    -------
    tuple of numpy.ndarray
        The refined points and their directions.
    """

    if len(points) == 0:
        return points, directions
    smallest_channel = cv2.GaussianBlur(picture_bgr.min(axis = 2).astype(numpy.float32), (3, 3), 0)
    half_length = float(edge_parameters["refine_half_length_px"])
    step = float(edge_parameters["refine_step_px"])
    offsets = numpy.arange(-half_length, half_length + 0.5 * step, step)
    sample_u = (points[:, 0:1] + offsets[None, :] * directions[:, 0:1]).astype(numpy.float32)
    sample_v = (points[:, 1:2] + offsets[None, :] * directions[:, 1:2]).astype(numpy.float32)
    profiles = cv2.remap(smallest_channel, sample_u, sample_v, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)

    slopes = numpy.diff(profiles, axis = 1) / step
    steepest = numpy.argmin(slopes, axis = 1)
    rows = numpy.arange(len(points))
    drop = profiles[:, :3].mean(axis = 1) - profiles[:, -3:].mean(axis = 1)
    inner = numpy.clip(steepest, 1, slopes.shape[1] - 2)
    before, centre, after = slopes[rows, inner - 1], slopes[rows, inner], slopes[rows, inner + 1]
    curvature = before - 2.0 * centre + after
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        shift = numpy.where(numpy.abs(curvature) > 1e-9, 0.5 * (before - after) / curvature, 0.0)
    shift = numpy.clip(shift, -0.5, 0.5)
    # the slope between samples i and i + 1 belongs to the middle of the two
    position = offsets[0] + (inner + 0.5 + shift) * step
    is_clear = (drop >= edge_parameters["refine_minimum_drop"]) & (steepest > 0) & (steepest < slopes.shape[1] - 1)
    refined = points + position[:, None] * directions
    return refined[is_clear], directions[is_clear]
