"""
Geometry of a sheet's line on the table (dashed-line analysis, robot test of 11.11.26).

A sheet's centre line is a polyline with one point per millimetre. It is made once per sheet from the cleanest top
view (``refined_centre_line``), then laid onto every other recording's own top view by turning and shifting it
(``registered_line``: the paper is rigid, so only its place on the table differs). With the line in a recording's
frame, any table point (the puck, the magnet tip) gets its place along the line and its distance beside it
(``place_along_line``).
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.signal import savgol_filter

GEOMETRY_PARAMETERS = {
    # the centre line is resampled to this spacing
    "spacing_m": 0.001,
    # snapping onto the ink: the ink's middle is looked for this far to each side of the line, a point counts as on
    # ink when at least "snap_least_ink_cells" of the cells across it are ink, and each round moves the line by
    # "snap_gain" of the smoothed offset
    "snap_half_width_m": 0.008,
    "snap_least_ink_cells": 4,
    "snap_gain": 0.7,
    "snap_rounds": 8,
    "snap_smoothing_m": 0.015,
    "snap_crossing_clear_m": 0.02,
    "final_smoothing_m": 0.015,
    # registration: blur of the ink map for the coarse and the fine search (metres), the coarse grid, and how much ink
    # found where the line should have a gap counts against a placement
    "coarse_blur_m": 0.006,
    "fine_blur_m": 0.0025,
    "coarse_angle_step_deg": 4.0,
    "coarse_shift_most_m": 0.08,
    "coarse_shift_step_m": 0.01,
    "gap_penalty": 0.5,
    # curvature: the tangent is compared this far apart
    "curvature_window_m": 0.02
}



def resampled(
        points_m: numpy.ndarray,
        spacing_m: float
) -> numpy.ndarray:
    """Return a polyline resampled to equal spacing along its length."""

    steps = numpy.hypot(*numpy.diff(points_m, axis = 0).T)
    distance = numpy.r_[0.0, numpy.cumsum(steps)]
    wanted = numpy.arange(0.0, distance[-1], spacing_m)
    return numpy.column_stack([numpy.interp(wanted, distance, points_m[:, 0]), numpy.interp(wanted, distance, points_m[:, 1])])



def spline_through(
        control_points_m: numpy.ndarray,
        spacing_m: float
) -> numpy.ndarray:
    """Return a smooth polyline through control points (a cubic spline over the chord length), resampled."""

    control_points_m = numpy.asarray(control_points_m, dtype = float)
    chord = numpy.r_[0.0, numpy.cumsum(numpy.hypot(*numpy.diff(control_points_m, axis = 0).T))]
    spline = CubicSpline(chord, control_points_m, axis = 0)
    dense = spline(numpy.arange(0.0, chord[-1], spacing_m / 4.0))
    return resampled(dense, spacing_m)



def tangents_and_normals(points_m: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Return the unit tangent (along the line) and the unit normal (to its left) at every polyline point."""

    difference = numpy.gradient(points_m, axis = 0)
    tangent = difference / numpy.maximum(numpy.linalg.norm(difference, axis = 1), 1e-12)[:, None]
    normal = numpy.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal



def map_values(
        picture: numpy.ndarray,
        view: Dict[str, numpy.ndarray],
        points_m: numpy.ndarray
) -> numpy.ndarray:
    """Return a top-view map's values at table points (nearest cell; 0 outside the map)."""

    cell = float(view["x_m"][1] - view["x_m"][0])
    column = numpy.round((points_m[:, 0] - view["x_m"][0]) / cell).astype(int)
    row = numpy.round((points_m[:, 1] - view["y_m"][0]) / cell).astype(int)
    is_inside = (column >= 0) & (column < picture.shape[0]) & (row >= 0) & (row < picture.shape[1])
    values = numpy.zeros(len(points_m))
    values[is_inside] = picture[column[is_inside], row[is_inside]]
    return values



def ink_share_of(
        view: Dict[str, numpy.ndarray],
        least_seen: int = 2
) -> numpy.ndarray:
    """Return the share of pictures that saw ink in every cell of a top view (0 where seen too rarely)."""

    return numpy.where(view["seen_votes"] >= least_seen, view["ink_votes"] / numpy.maximum(view["seen_votes"], 1), 0.0)



def refined_centre_line(
        first_guess_m: numpy.ndarray,
        view: Dict[str, numpy.ndarray],
        parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Move a first guess of the centre line onto the middle of the ink of a clean top view.

    At every point the ink's middle is looked for across the line; where the point lies on ink the offset is known,
    in the gaps of a dashed line it is filled in from the neighbours; the offsets are smoothed along the line and
    applied, several rounds, and the result is smoothed lightly.

    Parameters
    ----------
    first_guess_m : numpy.ndarray
        The polyline to start from, shape (N, 2), in metres in the top view's frame.
    view : dict
        From ``dashed_common.top_view_of_line``.
    parameters : dict, optional
        Overrides of GEOMETRY_PARAMETERS.

    Returns
    -------
    numpy.ndarray
        The refined polyline, one point per "spacing_m".
    """

    settings = dict(GEOMETRY_PARAMETERS)
    settings.update(parameters or {})
    share = ink_share_of(view)
    points = resampled(numpy.asarray(first_guess_m, dtype = float), settings["spacing_m"])
    offsets = numpy.arange(-settings["snap_half_width_m"], settings["snap_half_width_m"] + 1e-9, 0.001)
    window = max(5, int(round(settings["snap_smoothing_m"] / settings["spacing_m"])) | 1)
    for _ in range(settings["snap_rounds"]):
        _, normal = tangents_and_normals(points)
        across = numpy.stack([map_values(share, view, points + offset * normal) for offset in offsets], axis = 1)
        is_ink = across > 0.5
        is_on_ink = is_ink.sum(axis = 1) >= settings["snap_least_ink_cells"]
        # near the crossing the other branch's ink lies beside the line and would pull it sideways: no snapping there
        crossing = crossing_of(points)
        if crossing is not None:
            near = int(round(settings["snap_crossing_clear_m"] / settings["spacing_m"]))
            for position in crossing:
                is_on_ink[max(0, position - near):position + near + 1] = False
        middle = numpy.where(is_on_ink, (is_ink * offsets[None, :]).sum(axis = 1) / numpy.maximum(is_ink.sum(axis = 1), 1), numpy.nan)
        positions = numpy.arange(len(points))
        filled = numpy.interp(positions, positions[is_on_ink], middle[is_on_ink])
        smoothed = savgol_filter(filled, window, 1)
        points = resampled(points + settings["snap_gain"] * smoothed[:, None] * normal, settings["spacing_m"])
    final_window = max(5, int(round(settings["final_smoothing_m"] / settings["spacing_m"])) | 1)
    points = numpy.column_stack([savgol_filter(points[:, 0], final_window, 3), savgol_filter(points[:, 1], final_window, 3)])
    return resampled(points, settings["spacing_m"])



def line_description(
        points_m: numpy.ndarray,
        parameters: Optional[Dict] = None
) -> Dict[str, numpy.ndarray]:
    """
    Return a centre line's distance along the line, direction and bending at every point.

    Returns
    -------
    dict
        "distance_m", "direction_deg" (unwrapped, 0 = +x, positive = turning left), "curvature_per_m" (positive =
        bending left) and "radius_m" (its inverse size, capped at 1 m).
    """

    settings = dict(GEOMETRY_PARAMETERS)
    settings.update(parameters or {})
    steps = numpy.hypot(*numpy.diff(points_m, axis = 0).T)
    distance = numpy.r_[0.0, numpy.cumsum(steps)]
    tangent, _ = tangents_and_normals(points_m)
    direction = numpy.unwrap(numpy.arctan2(tangent[:, 1], tangent[:, 0]))
    half = max(1, int(round(0.5 * settings["curvature_window_m"] / settings["spacing_m"])))
    padded = numpy.r_[numpy.full(half, direction[0]), direction, numpy.full(half, direction[-1])]
    curvature = (padded[2 * half:] - padded[:-2 * half]) / (2 * half * settings["spacing_m"])
    radius = numpy.minimum(1.0, 1.0 / numpy.maximum(numpy.abs(curvature), 1e-9))
    return {"distance_m": distance, "direction_deg": numpy.degrees(direction), "curvature_per_m": curvature, "radius_m": radius}



def crossing_of(points_m: numpy.ndarray) -> Optional[Tuple[int, int]]:
    """Return the two polyline positions where the line crosses itself (None when it does not)."""

    count = len(points_m)
    stride = 5
    coarse = points_m[::stride]
    best = None
    for first in range(len(coarse)):
        later = numpy.arange(first + 60 // stride * 1, len(coarse))
        if len(later) == 0:
            break
        distances = numpy.hypot(*(coarse[later] - coarse[first]).T)
        nearest = int(distances.argmin())
        if best is None or distances[nearest] < best[0]:
            best = (float(distances[nearest]), first * stride, int(later[nearest]) * stride)
    if best is None or best[0] > 0.004:
        return None
    first_range = numpy.arange(max(0, best[1] - stride), min(count, best[1] + stride + 1))
    second_range = numpy.arange(max(0, best[2] - stride), min(count, best[2] + stride + 1))
    distances = numpy.hypot(points_m[first_range, None, 0] - points_m[None, second_range, 0], points_m[first_range, None, 1] - points_m[None, second_range, 1])
    position = numpy.unravel_index(distances.argmin(), distances.shape)
    return int(first_range[position[0]]), int(second_range[position[1]])



def moved_line(
        points_m: numpy.ndarray,
        angle_deg: float,
        shift_m: Tuple[float, float]
) -> numpy.ndarray:
    """Return a polyline turned by ``angle_deg`` about the origin and then shifted."""

    angle = math.radians(angle_deg)
    rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return numpy.asarray(points_m, dtype = float) @ rotation.T + numpy.asarray(shift_m, dtype = float)



def registered_line(
        reference_m: numpy.ndarray,
        is_ink_point: numpy.ndarray,
        view: Dict[str, numpy.ndarray],
        first_guess: Optional[Tuple[float, float, float]] = None,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    Lay a sheet's reference centre line onto a recording's top view by turning and shifting it.

    The score of a placement is the mean ink share under the reference's ink points minus "gap_penalty" times the
    mean ink share under its gap points (a dashed line), on a blurred ink map. Without a first guess every direction
    is tried in coarse steps with shifts around the ink's centre of mass; the best is refined with a simplex search.

    Parameters
    ----------
    reference_m : numpy.ndarray
        The reference polyline, shape (N, 2), metres, centred near its own mean.
    is_ink_point : numpy.ndarray
        Whether each reference point lies on ink (all True for a solid line).
    view : dict
        The recording's top view.
    first_guess : tuple, optional
        (angle in degrees, shift x, shift y in metres) to refine instead of the coarse search.
    parameters : dict, optional
        Overrides of GEOMETRY_PARAMETERS.

    Returns
    -------
    dict
        "angle_deg", "shift_m", "line_m" (the placed polyline), "score" (fine), "ink_share_on_ink_points" and
        "ink_share_on_gap_points" on the unblurred map, "seen_share_of_line" (how much of the line the recording's
        pictures showed) and the two ink shares counted only where the line was seen.
    """

    settings = dict(GEOMETRY_PARAMETERS)
    settings.update(parameters or {})
    share = ink_share_of(view).astype(numpy.float32)
    cell = float(view["x_m"][1] - view["x_m"][0])
    coarse_map = cv2.GaussianBlur(share, (0, 0), settings["coarse_blur_m"] / cell)
    fine_map = cv2.GaussianBlur(share, (0, 0), settings["fine_blur_m"] / cell)
    centred = reference_m - reference_m.mean(axis = 0)
    thinned = slice(None, None, 3)

    def score(values, picture, points_slice):
        placed = moved_line(centred[points_slice], values[0], values[1:3])
        found = map_values(picture, view, placed)
        on_ink = is_ink_point[points_slice]
        gap_part = settings["gap_penalty"] * found[~on_ink].mean() if (~on_ink).any() else 0.0
        return float(found[on_ink].mean() - gap_part)

    if first_guess is None:
        ink_cells = numpy.argwhere(share > 0.5)
        centre = numpy.array([view["x_m"][0] + cell * ink_cells[:, 0].mean(), view["y_m"][0] + cell * ink_cells[:, 1].mean()])
        shifts = numpy.arange(-settings["coarse_shift_most_m"], settings["coarse_shift_most_m"] + 1e-9, settings["coarse_shift_step_m"])
        best = (-math.inf, None)
        for angle in numpy.arange(0.0, 360.0, settings["coarse_angle_step_deg"]):
            for shift_x in shifts:
                for shift_y in shifts:
                    values = [angle, centre[0] + shift_x, centre[1] + shift_y]
                    value = score(values, coarse_map, slice(None, None, 10))
                    if value > best[0]:
                        best = (value, values)
        start = best[1]
    else:
        start = [first_guess[0], first_guess[1], first_guess[2]]
    for picture in (coarse_map, fine_map):
        result = minimize(lambda values: -score(values, picture, thinned), start, method = "Nelder-Mead",
                          options = {"xatol": 1e-4, "fatol": 1e-5, "maxiter": 3000, "initial_simplex": numpy.array(start) + numpy.array([[0, 0, 0], [3.0, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])})
        start = list(result.x)
    placed = moved_line(centred, start[0], start[1:3])
    found = map_values(share, view, placed)
    is_seen = map_values((view["seen_votes"] >= 2).astype(float), view, placed) > 0
    return {"angle_deg": float(start[0]), "shift_m": (float(start[1]), float(start[2])), "line_m": placed, "score": float(-result.fun),
            "ink_share_on_ink_points": float(found[is_ink_point].mean()),
            "ink_share_on_gap_points": float(found[~is_ink_point].mean()) if (~is_ink_point).any() else float("nan"),
            "seen_share_of_line": float(is_seen.mean()),
            "ink_share_on_seen_ink_points": float(found[is_ink_point & is_seen].mean()) if (is_ink_point & is_seen).any() else 0.0,
            "ink_share_on_seen_gap_points": float(found[~is_ink_point & is_seen].mean()) if (~is_ink_point & is_seen).any() else float("nan")}



def place_along_line(
        line_m: numpy.ndarray,
        points_m: numpy.ndarray,
        most_jump_m: float = 0.04,
        start_position: Optional[int] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return, for a sequence of table points in time order, each point's place along the line and its distance beside it.

    Because the line crosses itself, the nearest line point is looked for only within ``most_jump_m`` (forwards or
    backwards along the line) of the previous point's place; the first point is looked for along the whole line, or
    near ``start_position`` when given.

    Parameters
    ----------
    line_m : numpy.ndarray
        The placed polyline, one point per millimetre, shape (N, 2).
    points_m : numpy.ndarray
        The points in time order, shape (M, 2); NaN rows give NaN.
    most_jump_m : float, optional
        How far along the line the place may move between two points.
    start_position : int, optional
        The polyline position near which the first point is looked for.

    Returns
    -------
    tuple of numpy.ndarray
        The polyline position (index, NaN for missing points) and the signed distance beside the line in metres
        (positive = to the left of the line's direction).
    """

    _, normal = tangents_and_normals(line_m)
    spacing = float(numpy.hypot(*(line_m[1] - line_m[0])))
    jump = int(round(most_jump_m / spacing))
    positions = numpy.full(len(points_m), numpy.nan)
    beside = numpy.full(len(points_m), numpy.nan)
    previous = start_position
    for index, point in enumerate(numpy.asarray(points_m, dtype = float)):
        if not numpy.all(numpy.isfinite(point)):
            continue
        if previous is None:
            low, high = 0, len(line_m)
        else:
            low, high = max(0, previous - jump), min(len(line_m), previous + jump + 1)
        distances = numpy.hypot(*(line_m[low:high] - point).T)
        nearest = low + int(distances.argmin())
        positions[index] = nearest
        beside[index] = float((point - line_m[nearest]) @ normal[nearest])
        previous = nearest
    return positions, beside
