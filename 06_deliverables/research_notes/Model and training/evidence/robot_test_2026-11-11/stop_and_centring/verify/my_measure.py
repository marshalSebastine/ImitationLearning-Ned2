"""
The verifier's own picture tools (11.11.26), written apart from the analyst's picture_measures.py so that the two can
disagree. Shared with the analyst: the project's fitted forearm camera (forearm_camera_model.py), the taught table
plane (config.WORKSPACE_PARAMETERS) and the arm model. Different from the analyst's: how the puck is located (the
outline's extent in 16 directions instead of an area overlap), how the line is found (a dark-on-paper threshold in the
picture itself instead of a voted table map) and that one picture at a time is used.

Nothing here connects to the robot; pictures and logged joints are only read.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy

ROOT_FOLDER = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

import forearm_camera_model
from config import PUCK_PARAMETERS, WORKSPACE_PARAMETERS
from side_view_test import magnet_tip_and_lean

PUCK_RADIUS_M = PUCK_PARAMETERS["diameter_m"] / 2
PUCK_HEIGHT_M = PUCK_PARAMETERS["height_m"]
PLANE = WORKSPACE_PARAMETERS["table_plane"]


def shifted_camera(shift_px: Sequence[float] = (0.0, 0.0), roll_deg: float = 0.0) -> Dict:
    """Camera overrides: the fitted camera with its principal point moved and rolled about its viewing direction."""

    fitted = forearm_camera_model.merged_camera_parameters()
    angle = math.radians(roll_deg)
    roll = numpy.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    rotation = forearm_camera_model.rotation_matrix_from_vector(fitted["rotation_vector_in_forearm_rad"]) @ roll
    return {"principal_point_px": [fitted["principal_point_px"][0] + shift_px[0], fitted["principal_point_px"][1] + shift_px[1]],
            "rotation_vector_in_forearm_rad": forearm_camera_model.rotation_vector_from_matrix(rotation).tolist()}


def table_height(x, y, lift_m: float = 0.0):
    return PLANE["a"] * x + PLANE["b"] * y + PLANE["c"] + lift_m


def to_table(pixels: numpy.ndarray, joints: Sequence[float], camera: Optional[Dict] = None, height_m: float = 0.0, lift_m: float = 0.0) -> numpy.ndarray:
    """Where the rays through pixels (N, 2) meet the plane ``height_m`` above the taught table: (N, 2) in metres."""

    position, rays, is_valid = forearm_camera_model.viewing_rays_in_base(numpy.asarray(pixels, dtype = float), joints, camera)
    numerator = PLANE["a"] * position[0] + PLANE["b"] * position[1] + PLANE["c"] + lift_m + height_m - position[2]
    denominator = rays[:, 2] - PLANE["a"] * rays[:, 0] - PLANE["b"] * rays[:, 1]
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        distance = numerator / denominator
    points = position[None, :2] + distance[:, None] * rays[:, :2]
    points[~(is_valid & numpy.isfinite(distance) & (distance > 0))] = numpy.nan
    return points


def to_picture(points_xy: numpy.ndarray, joints: Sequence[float], camera: Optional[Dict] = None, height_m: float = 0.0, lift_m: float = 0.0) -> numpy.ndarray:
    """Where table points (N, 2), lifted ``height_m`` above the table, appear in the picture: (N, 2) pixels."""

    points_xy = numpy.atleast_2d(numpy.asarray(points_xy, dtype = float))
    heights = table_height(points_xy[:, 0], points_xy[:, 1], lift_m) + height_m
    return forearm_camera_model.project_points(numpy.column_stack([points_xy, heights]), joints, camera)


def red_mask(picture_bgr: numpy.ndarray) -> numpy.ndarray:
    """The puck's red pixels: the largest red blob, boolean."""

    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    is_red = ((hsv[:, :, 0] >= 163) | (hsv[:, :, 0] <= 6)) & (hsv[:, :, 1] >= 95) & (hsv[:, :, 2] >= 60)
    is_red = cv2.morphologyEx(is_red.astype(numpy.uint8), cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(is_red, connectivity = 8)
    if count < 2:
        return numpy.zeros(is_red.shape, dtype = bool)
    largest = 1 + int(numpy.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 300:
        return numpy.zeros(is_red.shape, dtype = bool)
    return labels == largest


def tip_pixel(joints: Sequence[float], camera: Optional[Dict] = None) -> numpy.ndarray:
    tip = numpy.asarray(magnet_tip_and_lean(list(joints))[0], dtype = float)
    return forearm_camera_model.project_points(tip[None, :], joints, camera)[0]


def puck_foot(picture_bgr: numpy.ndarray, joints: Sequence[float], camera: Optional[Dict] = None, lift_m: float = 0.0) -> Optional[Dict]:
    """
    The centre of the puck's foot on the table, from the extent of its red TOP FACE in the picture.

    The puck is a cylinder (radius 12 mm, height 29 mm) standing on the table. The red blob is taken as the top face
    (in most pictures the puck's side is in shadow and no longer red; where a strip of the side is red it lies towards
    the foot). For a trial place the top rim is projected and its extent is compared with the blob's in the directions
    that point away from the foot (within 100 degrees of the picture direction from the foot to the top), so that
    neither the side nor the block that hides the puck's near edge counts.

    Returns
    -------
    dict or None
        "xy" (metres), "residual_px" (root mean square of the used directions), "directions" (how many were used),
        "top_px", "foot_px" (the centres of the top face and of the foot in the picture), "mask" (the red blob).
    """

    mask = red_mask(picture_bgr)
    if not mask.any():
        return None
    rows, columns = numpy.nonzero(mask)
    blob = numpy.column_stack([columns, rows]).astype(float)
    centre = blob.mean(axis = 0)
    start = to_table(centre[None, :], joints, camera, PUCK_HEIGHT_M, lift_m)[0]
    if not numpy.all(numpy.isfinite(start)):
        return None
    down = to_picture(start[None, :], joints, camera, 0.0, lift_m)[0] - to_picture(start[None, :], joints, camera, PUCK_HEIGHT_M, lift_m)[0]
    down = down / max(float(numpy.linalg.norm(down)), 1e-9)
    angles = numpy.linspace(0.0, 2.0 * math.pi, 16, endpoint = False)
    units = numpy.column_stack([numpy.cos(angles), numpy.sin(angles)])
    blob_extent = (blob @ units.T).max(axis = 0)
    is_used = units @ down <= math.cos(math.radians(80.0))
    # directions in which the blob touches the picture's edge are not used either
    height, width = mask.shape
    for index, unit in enumerate(units):
        extreme = blob[numpy.argmax(blob @ unit)]
        if extreme[0] <= 2 or extreme[0] >= width - 3 or extreme[1] <= 2 or extreme[1] >= height - 3:
            is_used[index] = False
    if is_used.sum() < 5:
        return None
    ring_angles = numpy.linspace(0.0, 2.0 * math.pi, 72, endpoint = False)
    ring = PUCK_RADIUS_M * numpy.column_stack([numpy.cos(ring_angles), numpy.sin(ring_angles)])

    def residuals(xy: numpy.ndarray) -> numpy.ndarray:
        rim = to_picture(xy[None, :] + ring, joints, camera, PUCK_HEIGHT_M, lift_m)
        return ((rim @ units.T).max(axis = 0) - blob_extent)[is_used]

    xy = start.copy()
    for _ in range(12):
        base = residuals(xy)
        jacobian = numpy.column_stack([(residuals(xy + numpy.array([1e-4, 0.0])) - base) / 1e-4, (residuals(xy + numpy.array([0.0, 1e-4])) - base) / 1e-4])
        step = numpy.linalg.lstsq(jacobian, -base, rcond = None)[0]
        xy = xy + step
        if float(numpy.linalg.norm(step)) < 1e-5:
            break
    final = residuals(xy)
    return {"xy": xy, "residual_px": float(numpy.sqrt(numpy.mean(final ** 2))), "directions": int(is_used.sum()),
            "top_px": to_picture(xy[None, :], joints, camera, PUCK_HEIGHT_M, lift_m)[0], "foot_px": to_picture(xy[None, :], joints, camera, 0.0, lift_m)[0],
            "mask": mask}


def hand_mask(shape, joints: Sequence[float], camera: Optional[Dict] = None, radius_px: float = 62.0) -> numpy.ndarray:
    """Where the robot's hand hides the table: a disc at the magnet tip joined to the bottom of the picture (boolean)."""

    hidden = numpy.zeros(shape, dtype = numpy.uint8)
    tip = tip_pixel(joints, camera)
    if numpy.all(numpy.isfinite(tip)):
        angles = numpy.linspace(0.0, 2.0 * math.pi, 24, endpoint = False)
        disc = tip[None, :] + radius_px * numpy.column_stack([numpy.cos(angles), numpy.sin(angles)])
        right = min(shape[1] - 1.0, tip[0] + 250.0)
        corners = numpy.vstack([disc, [[tip[0] - 150.0, shape[0] - 1.0], [right, shape[0] - 1.0], [right, shape[0] - 35.0]]])
        cv2.fillConvexPoly(hidden, cv2.convexHull(corners.astype(numpy.float32)).astype(numpy.int32), 1)
    return hidden.astype(bool)


def line_mask(picture_bgr: numpy.ndarray, joints: Sequence[float], camera: Optional[Dict] = None, puck: Optional[numpy.ndarray] = None) -> numpy.ndarray:
    """
    The pixels that show the black line: clearly darker than the paper around them (a closing wider than the line gives
    the paper's level), on bright paper, not the robot's hand and not the puck (its red blob grown by 6 px).
    """

    grey = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    closed = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
    paper_level = float(numpy.percentile(grey[::4, ::4], 90))
    is_on_paper = closed >= 0.72 * paper_level
    is_line = is_on_paper & (grey < 0.72 * closed)
    is_line &= ~hand_mask(grey.shape, joints, camera)
    if puck is not None and puck.any():
        is_line &= ~cv2.dilate(puck.astype(numpy.uint8), numpy.ones((13, 13), numpy.uint8)).astype(bool)
    return cv2.morphologyEx(is_line.astype(numpy.uint8), cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8)).astype(bool)


def foot_circle_px(xy: numpy.ndarray, joints: Sequence[float], camera: Optional[Dict] = None, radius_m: float = PUCK_RADIUS_M, lift_m: float = 0.0) -> numpy.ndarray:
    """The puck's foot circle on the table as picture pixels, for drawing."""

    angles = numpy.linspace(0.0, 2.0 * math.pi, 72, endpoint = False)
    ring = numpy.asarray(xy)[None, :] + radius_m * numpy.column_stack([numpy.cos(angles), numpy.sin(angles)])
    return to_picture(ring, joints, camera, 0.0, lift_m)
