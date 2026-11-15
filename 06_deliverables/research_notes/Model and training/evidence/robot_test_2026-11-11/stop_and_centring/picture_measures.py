"""
Picture measures for the two quality requirements of the robot test of 11.11.26 (Emranul: the puck must cover the line's
end completely at the stop, and stay in the middle of the line all the way).

Everything here reads pictures and logged joint angles; nothing connects to the robot. Three building blocks:

    TableCamera         the forearm camera at one set of joints together with the taught table plane: where a table
                        point appears in the picture and where a pixel's ray meets the table (the same numbers as
                        forearm_camera_model.py, written for whole grids at once);
    fit_puck_on_table   where the red puck stands on the table: the puck is a cylinder of known size, its outline in the
                        picture is predicted through the camera for a trial place on the table, and the place whose
                        outline covers the red blob best is kept. The blob's centre alone would be a point somewhere
                        between the puck's top and its foot, depending on how much of its side the block hides;
    LineMap             a map of the table (0.5 mm per cell) in which every picture of a run votes where it saw the
                        black line and where it saw bare paper; the robot's hand, the puck and their shadows move from
                        picture to picture and are voted out. One picture alone gives the one-picture measure.

From the puck's place and the line map: the sideways offset between the puck's centre and the line's middle
(``offset_from_line``) and how the puck stands on the line's round end (``end_cover``).

The detector's private helper for the blob at full size and its occluders is used as it is (through its mangled name)
so that the blob is exactly the one puck_detector.py reports.
"""

import collections
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy

ROOT_FOLDER = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

import forearm_camera_model
from config import PUCK_PARAMETERS, SHEET_PARAMETERS, WORKSPACE_PARAMETERS
from puck_detector import PuckDetector
from side_view_test import magnet_tip_and_lean


MEASURE_PARAMETERS = {
    # the puck, measured by the team (config.PUCK_PARAMETERS): 2.4 cm across and 2.9 cm tall
    "puck_radius_m": PUCK_PARAMETERS["diameter_m"] / 2,
    "puck_height_m": PUCK_PARAMETERS["height_m"],
    "line_width_m": SHEET_PARAMETERS["line_width_m"],
    # the puck's outline: points on each rim, the margin kept around the detector's blob, the weights of red pixels
    # outside the predicted outline and of outline pixels that are neither red nor hidden, and the search (step sizes
    # in metres, each tried "fit_search_half_steps" to either side)
    "rim_points": 48,
    "blob_margin_px": 45,
    "fit_missing_weight": 2.0,
    "fit_extra_weight": 1.0,
    "fit_search_steps_m": [0.002, 0.0005, 0.000125],
    "fit_search_half_steps": 4,
    "occluder_dilation_px": 5,
    # a fit whose outline and blob share less than this (shared area over joined area, hidden pixels left out) is
    # not used
    "fit_quality_minimum": 0.80,
    # the line map: cell size, the margin around the tip's path, the closing kernel (wider than the line), how much
    # darker than its closed surroundings a line pixel is, how bright and how free of the table's warm colour the
    # surroundings must be (a share of the paper's grey level; red minus blue above the paper's), what hides the table
    # (a disc at the magnet block joined to the bottom of the picture where the hand comes in; the red blob grown by
    # some pixels), and the votes a cell needs
    "map_mm_per_px": 0.5,
    "map_margin_m": 0.07,
    "line_closing_m": 0.017,
    "line_darkness_ratio": 0.75,
    "paper_darkness_ratio": 0.88,
    "paper_brightness_share": 0.62,
    "paper_level_percentile": 90.0,
    "paper_warmth_margin": 28.0,
    "hand_radius_px": 58.0,
    "hand_bottom_px": [[150, 479], [639, 479], [639, 430]],
    "puck_exclusion_px": 9,
    "picture_margin_px": 4,
    "unseen_margin_m": 0.0055,
    "minimum_line_votes": 2,
    "line_vote_share": 0.5,
    # the line's middle: cells at least this share of half the line's width from the line's edge; the offset of a
    # point is measured to a parabola fitted to the middle cells within "fit_radius_m" of the point's nearest one
    "core_share_of_half_width": 0.55,
    "fit_radius_m": 0.022,
    "fit_minimum_cells": 25,
    "search_radius_m": 0.06,
    # where the line comes near itself: how many stretches are tried, how far a stretch may run across the direction
    # of travel, and how far its middle cells may spread around one curve (one clean stretch: about 1.3 mm)
    "branch_tries": 3,
    "branch_angle_most_deg": 55.0,
    "fit_spread_most_m": 0.002,
    # one picture alone: only the line beside the puck or ahead of it within this cone of the direction of travel
    "ahead_cone_deg": 60.0,
    "beside_radius_m": 0.015,
    # the line's end: how far before and beyond the puck's last place the line is looked at, how far to the side of
    # the line's middle a cell may lie and still belong to it (a narrow band to find the tip, a wider one that holds
    # the line's full width to find its sideways middle and its width), the gap that separates two dashes, the shortest run of
    # cells that counts as line, how far ahead of the puck the last run may begin, and how far beyond the end bare
    # paper must have been seen for the end to count as seen
    "end_look_back_m": 0.05,
    "end_band_m": 0.007,
    "end_wide_band_m": 0.009,
    "end_gap_m": 0.0015,
    "end_run_least_m": 0.004,
    "end_ahead_most_m": 0.015,
    "end_paper_check_m": 0.004,
    "end_direction_span_least_m": 0.012
}



def merged_measure_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``MEASURE_PARAMETERS`` with any overrides merged over it."""

    merged = dict(MEASURE_PARAMETERS)
    merged.update(parameters or {})
    return merged



class TableCamera:

    def __init__(
            self,
            joints: Sequence[float],
            camera_parameters: Optional[Dict] = None
    ):
        """
        The forearm camera at ``joints`` and the taught table plane.

        Parameters
        ----------
        joints : sequence of float
            The six joint angles in radians when the picture was taken.
        camera_parameters : dict, optional
            Overrides for ``FOREARM_CAMERA_PARAMETERS``.
        """

        self.joints = [float(value) for value in joints]
        self.camera_parameters = forearm_camera_model.merged_camera_parameters(camera_parameters)
        self.__initialize_variables()


    def __initialize_variables(self):

        self.position, self.rotation = forearm_camera_model.camera_pose_in_base(self.joints, self.camera_parameters)
        self.focal_length = float(self.camera_parameters["focal_length_px"])
        self.principal_point = numpy.asarray(self.camera_parameters["principal_point_px"], dtype = float)
        self.distortion = float(self.camera_parameters["radial_distortion_k1"])
        self.picture_width, self.picture_height = self.camera_parameters["image_size_px"]
        plane = WORKSPACE_PARAMETERS["table_plane"]
        # "table_height_offset_m" (not a key of the fitted camera) lifts or lowers the taught table plane
        self.plane = (float(plane["a"]), float(plane["b"]), float(plane["c"]) + float(self.camera_parameters.get("table_height_offset_m", 0.0)))


    def table_height(
            self,
            x: numpy.ndarray,
            y: numpy.ndarray
    ) -> numpy.ndarray:
        """The table's height at (x, y), from the taught plane."""

        return self.plane[0] * x + self.plane[1] * y + self.plane[2]


    def project(self, points_base: numpy.ndarray) -> numpy.ndarray:
        """
        Where points of the base frame appear in the picture.

        Parameters
        ----------
        points_base : numpy.ndarray
            Points in metres, shape (N, 3).

        Returns
        -------
        numpy.ndarray
            Pixels (u, v), shape (N, 2); NaN behind the camera.
        """

        points_camera = (numpy.atleast_2d(numpy.asarray(points_base, dtype = float)) - self.position[None, :]) @ self.rotation
        depth = points_camera[:, 2]
        with numpy.errstate(divide = "ignore", invalid = "ignore"):
            normalised = points_camera[:, :2] / depth[:, None]
        factor = 1.0 + self.distortion * (normalised ** 2).sum(axis = 1)
        pixels = self.focal_length * normalised * factor[:, None] + self.principal_point[None, :]
        pixels[depth <= 1e-9] = numpy.nan
        return pixels


    def project_table(
            self,
            points_xy: numpy.ndarray,
            height_above_table_m: float = 0.0
    ) -> numpy.ndarray:
        """Where points ``height_above_table_m`` above the table at (x, y) appear in the picture, shape (N, 2)."""

        points_xy = numpy.atleast_2d(numpy.asarray(points_xy, dtype = float))
        heights = self.table_height(points_xy[:, 0], points_xy[:, 1]) + height_above_table_m
        return self.project(numpy.column_stack([points_xy, heights]))


    def table_from_pixels(
            self,
            pixels: numpy.ndarray,
            height_above_table_m: float = 0.0
    ) -> numpy.ndarray:
        """
        Where the rays through pixels meet the plane ``height_above_table_m`` above the table.

        Parameters
        ----------
        pixels : numpy.ndarray
            Pixels (u, v), shape (N, 2).
        height_above_table_m : float, optional
            The height above the taught table plane.

        Returns
        -------
        numpy.ndarray
            Points (x, y) in metres in the base frame, shape (N, 2); NaN where the ray misses the plane.
        """

        pixels = numpy.atleast_2d(numpy.asarray(pixels, dtype = float))
        distorted = (pixels - self.principal_point[None, :]) / self.focal_length
        normalised, is_valid = forearm_camera_model.undistort_normalised(distorted, self.distortion)
        rays = numpy.column_stack([normalised, numpy.ones(len(normalised))]) @ self.rotation.T
        slope_x, slope_y, offset = self.plane
        # the plane is z = a x + b y + c + height: solve for the distance along the ray
        numerator = -(slope_x * self.position[0] + slope_y * self.position[1] - self.position[2] + offset + height_above_table_m)
        denominator = slope_x * rays[:, 0] + slope_y * rays[:, 1] - rays[:, 2]
        with numpy.errstate(divide = "ignore", invalid = "ignore"):
            distance = numerator / denominator
        points = self.position[None, :2] + distance[:, None] * rays[:, :2]
        points[~(is_valid & numpy.isfinite(distance) & (distance > 0.0))] = numpy.nan
        return points


    def table_grid_maps(
            self,
            x_values: numpy.ndarray,
            y_values: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        The picture pixel of every cell of a grid on the table, for ``cv2.remap``.

        Parameters
        ----------
        x_values : numpy.ndarray
            The x of every grid row, in metres.
        y_values : numpy.ndarray
            The y of every grid column, in metres.

        Returns
        -------
        tuple of numpy.ndarray
            The u and the v of every cell, float32, shape (rows, columns); -1 behind the camera.
        """

        slope_x, slope_y, offset = self.plane
        x_column = numpy.asarray(x_values, dtype = numpy.float64)[:, None]
        y_row = numpy.asarray(y_values, dtype = numpy.float64)[None, :]
        components = []
        for axis in range(3):
            # the table point is (x, y, a x + b y + c), so each camera coordinate is linear in x and y
            along_x = self.rotation[0, axis] + slope_x * self.rotation[2, axis]
            along_y = self.rotation[1, axis] + slope_y * self.rotation[2, axis]
            constant = -float(self.rotation[:, axis] @ self.position) + offset * self.rotation[2, axis]
            components.append((along_x * x_column + along_y * y_row + constant).astype(numpy.float32))
        depth = components[2]
        is_behind = depth <= 1e-6
        depth = numpy.where(is_behind, numpy.float32(1.0), depth)
        normalised_x = components[0] / depth
        normalised_y = components[1] / depth
        factor = 1.0 + numpy.float32(self.distortion) * (normalised_x * normalised_x + normalised_y * normalised_y)
        map_u = numpy.float32(self.focal_length) * normalised_x * factor + numpy.float32(self.principal_point[0])
        map_v = numpy.float32(self.focal_length) * normalised_y * factor + numpy.float32(self.principal_point[1])
        map_u[is_behind] = -1.0
        map_v[is_behind] = -1.0
        return map_u.astype(numpy.float32), map_v.astype(numpy.float32)


    def tip_pixel(self) -> numpy.ndarray:
        """Where the magnet tip (the arm model on the joints) appears in the picture, (u, v)."""

        return self.project(numpy.asarray(magnet_tip_and_lean(list(self.joints))[0], dtype = float)[None, :])[0]



def puck_blob(
        picture_bgr: numpy.ndarray,
        detector: PuckDetector,
        parameters: Optional[Dict] = None
) -> Optional[Dict]:
    """
    The red puck's blob at full size, as puck_detector.py finds it, in a crop with room for the puck's hidden part.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The camera picture in OpenCV's BGR order.
    detector : PuckDetector
        A kept detector.
    parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS`` ("blob_margin_px").

    Returns
    -------
    dict or None
        None when the detector accepts no blob. Otherwise "mask" (boolean, the crop), "left" and "top" (the crop's
        corner in the picture), "occluder" (boolean, the dark block and the blue brackets in the crop), "confidence",
        "is_partial" and "centre_px" (the blob's centre in the picture).
    """

    parameters = merged_measure_parameters(parameters)
    candidates, _, cast_share = detector.find_candidates(picture_bgr)
    accepted = [candidate for candidate in candidates if candidate["rejection_reason"] is None]
    if not accepted:
        return None
    best = accepted[0]
    full_size = detector._PuckDetector__full_size_blob(
        picture_bgr = picture_bgr,
        bounding_box = best["bounding_box"],
        margin = parameters["blob_margin_px"],
        cast_share = cast_share
    )
    if full_size is None:
        return None
    mask = full_size["blob"].astype(bool)
    rows, columns = numpy.nonzero(mask)
    return {
        "mask": mask,
        "left": int(full_size["crop_left"]),
        "top": int(full_size["crop_top"]),
        "occluder": full_size["occluder"].astype(bool),
        "confidence": float(best["score"]),
        "is_partial": bool(best["is_partial"]),
        "centre_px": [float(columns.mean()) + full_size["crop_left"], float(rows.mean()) + full_size["crop_top"]]
    }



def puck_outline(
        base_xy: Sequence[float],
        camera: TableCamera,
        parameters: Dict,
        radius_m: Optional[float] = None
) -> numpy.ndarray:
    """
    The outline in the picture of a puck standing at ``base_xy`` on the table: the hull of its top and bottom rims.

    Parameters
    ----------
    base_xy : sequence of float
        The centre of the puck's foot on the table, metres in the base frame.
    camera : TableCamera
        The camera of the picture.
    parameters : dict
        ``MEASURE_PARAMETERS`` ("puck_radius_m", "puck_height_m", "rim_points").
    radius_m : float, optional
        Another radius than the parameters' (used to check the puck's size).

    Returns
    -------
    numpy.ndarray
        The hull's corners (u, v), float32, shape (N, 2).
    """

    radius = parameters["puck_radius_m"] if radius_m is None else radius_m
    angles = numpy.linspace(0.0, 2.0 * math.pi, parameters["rim_points"], endpoint = False)
    ring = numpy.asarray(base_xy, dtype = float)[None, :] + radius * numpy.column_stack([numpy.cos(angles), numpy.sin(angles)])
    rims = numpy.vstack([camera.project_table(ring, parameters["puck_height_m"]), camera.project_table(ring, 0.0)])
    return cv2.convexHull(rims.astype(numpy.float32)).reshape(-1, 2)



def fit_puck_on_table(
        blob: Dict,
        camera: TableCamera,
        parameters: Optional[Dict] = None,
        radius_m: Optional[float] = None
) -> Dict:
    """
    Find where the puck stands on the table from its red blob.

    The predicted outline of a cylinder of the puck's size is moved over the table until it covers the blob best: red
    pixels outside the outline and outline pixels that are neither red nor hidden (by the block, a bracket or the
    picture's edge) both count against a place.

    Parameters
    ----------
    blob : dict
        From ``puck_blob``.
    camera : TableCamera
        The camera of the picture.
    parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS``.
    radius_m : float, optional
        Another puck radius than the parameters' (used to check the puck's size).

    Returns
    -------
    dict
        "base_xy" (the centre of the puck's foot on the table, metres), "cost", "quality" (shared area over joined
        area of blob and outline, hidden pixels left out; 1 is perfect), "missing_px", "extra_px", "outline_px" (the
        fitted outline in the picture), "base_px" and "top_px" (the centres of the foot and of the top face in the
        picture).
    """

    parameters = merged_measure_parameters(parameters)
    is_red = blob["mask"]
    height, width = is_red.shape
    kernel_size = 2 * parameters["occluder_dilation_px"] + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    is_hidden = cv2.dilate(blob["occluder"].astype(numpy.uint8), kernel).astype(bool)
    is_free = ~is_red & ~is_hidden
    corner = numpy.array([blob["left"], blob["top"]], dtype = float)

    def outline_mask(base_xy: numpy.ndarray) -> numpy.ndarray:
        outline = puck_outline(base_xy, camera, parameters, radius_m) - corner[None, :]
        mask = numpy.zeros((height, width), dtype = numpy.uint8)
        # four fractional bits keep the outline's corners at a sixteenth of a pixel
        cv2.fillConvexPoly(mask, numpy.round(outline * 16.0).astype(numpy.int32), 1, lineType = cv2.LINE_8, shift = 4)
        return mask.astype(bool)

    def cost_of(base_xy: numpy.ndarray) -> Tuple[float, int, int]:
        is_outline = outline_mask(base_xy)
        missing = int(numpy.count_nonzero(is_red & ~is_outline))
        extra = int(numpy.count_nonzero(is_outline & is_free))
        return parameters["fit_missing_weight"] * missing + parameters["fit_extra_weight"] * extra, missing, extra

    best_xy = camera.table_from_pixels(numpy.array([blob["centre_px"]]), parameters["puck_height_m"] / 2.0)[0]
    best = cost_of(best_xy)
    half_steps = parameters["fit_search_half_steps"]
    for step in parameters["fit_search_steps_m"]:
        centre_xy = best_xy.copy()
        for index_x in range(-half_steps, half_steps + 1):
            for index_y in range(-half_steps, half_steps + 1):
                trial_xy = centre_xy + step * numpy.array([index_x, index_y], dtype = float)
                trial = cost_of(trial_xy)
                if trial[0] < best[0]:
                    best, best_xy = trial, trial_xy

    is_outline = outline_mask(best_xy)
    shared = int(numpy.count_nonzero(is_red & is_outline))
    joined = int(numpy.count_nonzero(is_red | (is_outline & ~is_hidden)))
    return {
        "base_xy": best_xy,
        "cost": best[0],
        "missing_px": best[1],
        "extra_px": best[2],
        "quality": shared / max(joined, 1),
        "outline_px": puck_outline(best_xy, camera, parameters, radius_m),
        "base_px": camera.project_table(best_xy[None, :], 0.0)[0],
        "top_px": camera.project_table(best_xy[None, :], parameters["puck_height_m"])[0]
    }



def hidden_table_mask(
        picture_shape: Tuple[int, int],
        camera: TableCamera,
        blob: Optional[Dict],
        parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Where the picture does not show the table: the robot's hand and the puck.

    The hand is taken as a disc at the magnet block joined to the bottom of the picture, where the rod, the brackets
    and the black box come in; the puck as its red blob grown by a few pixels (its outline and its shadow).

    Parameters
    ----------
    picture_shape : tuple of int
        (height, width) of the picture.
    camera : TableCamera
        The camera of the picture.
    blob : dict or None
        From ``puck_blob``; None when no puck was found.
    parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS``.

    Returns
    -------
    numpy.ndarray
        ``uint8`` mask of the picture, 255 where the table is hidden.
    """

    parameters = merged_measure_parameters(parameters)
    hidden = numpy.zeros(picture_shape, dtype = numpy.uint8)
    tip = camera.tip_pixel()
    if numpy.all(numpy.isfinite(tip)):
        angles = numpy.linspace(0.0, 2.0 * math.pi, 24, endpoint = False)
        disc = tip[None, :] + parameters["hand_radius_px"] * numpy.column_stack([numpy.cos(angles), numpy.sin(angles)])
        corners = numpy.vstack([disc, numpy.asarray(parameters["hand_bottom_px"], dtype = float)])
        cv2.fillConvexPoly(hidden, cv2.convexHull(corners.astype(numpy.float32)).astype(numpy.int32), 255)
    if blob is not None:
        kernel_size = 2 * parameters["puck_exclusion_px"] + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        grown = cv2.dilate(blob["mask"].astype(numpy.uint8), kernel)
        region = hidden[blob["top"]:blob["top"] + grown.shape[0], blob["left"]:blob["left"] + grown.shape[1]]
        region[grown[:region.shape[0], :region.shape[1]] > 0] = 255
    return hidden



class LineMap:

    def __init__(
            self,
            x_range: Sequence[float],
            y_range: Sequence[float],
            parameters: Optional[Dict] = None
    ):
        """
        A grid on the table that collects where the pictures of a run saw the line and where bare paper.

        Row 0 is the largest x (away from the robot's base, as on a map with the base at the bottom) and column 0 the
        largest y (the base's left), so the map looks like the table seen from above.

        Parameters
        ----------
        x_range, y_range : sequence of float
            The smallest and largest x and y covered, metres in the base frame.
        parameters : dict, optional
            Overrides for ``MEASURE_PARAMETERS``.
        """

        self.parameters = merged_measure_parameters(parameters)
        self.x_range = [float(value) for value in x_range]
        self.y_range = [float(value) for value in y_range]
        self.__initialize_variables()


    def __initialize_variables(self):

        self.cell_m = self.parameters["map_mm_per_px"] / 1000.0
        row_count = int(round((self.x_range[1] - self.x_range[0]) / self.cell_m)) + 1
        column_count = int(round((self.y_range[1] - self.y_range[0]) / self.cell_m)) + 1
        self.x_values = self.x_range[1] - self.cell_m * numpy.arange(row_count)
        self.y_values = self.y_range[1] - self.cell_m * numpy.arange(column_count)
        self.line_votes = numpy.zeros((row_count, column_count), dtype = numpy.uint16)
        self.paper_votes = numpy.zeros((row_count, column_count), dtype = numpy.uint16)
        closing_px = int(round(self.parameters["line_closing_m"] / self.cell_m)) | 1
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_px, closing_px))
        border_px = int(round(self.parameters["unseen_margin_m"] / self.cell_m)) | 1
        self.border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_px, border_px))
        self.pictures = 0
        self.history = collections.deque()


    def cell_of(self, points_xy: numpy.ndarray) -> numpy.ndarray:
        """The (row, column) of table points, as floats, shape (N, 2)."""

        points_xy = numpy.atleast_2d(numpy.asarray(points_xy, dtype = float))
        return numpy.column_stack([(self.x_range[1] - points_xy[:, 0]) / self.cell_m,
                                   (self.y_range[1] - points_xy[:, 1]) / self.cell_m])


    def point_of(self, cells: numpy.ndarray) -> numpy.ndarray:
        """The table point (x, y) of cells given as (row, column), shape (N, 2)."""

        cells = numpy.atleast_2d(numpy.asarray(cells, dtype = float))
        return numpy.column_stack([self.x_range[1] - self.cell_m * cells[:, 0], self.y_range[1] - self.cell_m * cells[:, 1]])


    def view_of(
            self,
            picture_bgr: numpy.ndarray,
            camera: TableCamera,
            hidden: numpy.ndarray
    ) -> Dict:
        """
        One picture laid onto the map: the picture as seen from above, and where it shows line and where bare paper.

        Parameters
        ----------
        picture_bgr : numpy.ndarray
            The camera picture.
        camera : TableCamera
            The camera of the picture.
        hidden : numpy.ndarray
            From ``hidden_table_mask``.

        Returns
        -------
        dict
            "view" (the picture on the map's grid, BGR), "is_line" and "is_paper" (boolean maps of the cells that
            vote) and "is_seen" (cells the picture shows and nothing hides).
        """

        parameters = self.parameters
        map_u, map_v = camera.table_grid_maps(self.x_values, self.y_values)
        view = cv2.remap(picture_bgr, map_u, map_v, cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))
        margin = parameters["picture_margin_px"]
        is_inside = ((map_u >= margin) & (map_u <= picture_bgr.shape[1] - 1 - margin)
                     & (map_v >= margin) & (map_v <= picture_bgr.shape[0] - 1 - margin))
        is_hidden = cv2.remap(hidden, map_u, map_v, cv2.INTER_NEAREST, borderMode = cv2.BORDER_CONSTANT, borderValue = 255) > 0
        # a cell votes only when its close surroundings ("unseen_margin_m") are in the picture and unhidden; nearer
        # to the puck, the hand or the picture's edge the closing cannot tell line from paper reliably
        is_seen = cv2.erode((is_inside & ~is_hidden).astype(numpy.uint8), self.border_kernel).astype(bool)

        grey_picture = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2GRAY)[::4, ::4]
        paper_level = float(numpy.percentile(grey_picture, parameters["paper_level_percentile"]))
        sample = picture_bgr[::4, ::4].astype(numpy.float32)
        paper_warmth = float(numpy.median((sample[:, :, 2] - sample[:, :, 0])[grey_picture >= paper_level - 15.0]))

        closed = cv2.morphologyEx(view, cv2.MORPH_CLOSE, self.closing_kernel)
        grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
        closed_grey = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
        closed_warmth = closed[:, :, 2].astype(numpy.float32) - closed[:, :, 0].astype(numpy.float32)
        is_on_paper = ((closed_grey >= parameters["paper_brightness_share"] * paper_level)
                       & (closed_warmth <= paper_warmth + parameters["paper_warmth_margin"]))
        is_line = is_seen & is_on_paper & (grey < parameters["line_darkness_ratio"] * closed_grey)
        is_paper = is_seen & is_on_paper & (grey >= parameters["paper_darkness_ratio"] * closed_grey)
        return {"view": view, "is_line": is_line, "is_paper": is_paper, "is_seen": is_seen}


    def add_picture(
            self,
            picture_bgr: numpy.ndarray,
            camera: TableCamera,
            hidden: numpy.ndarray,
            window_pictures: Optional[int] = None
    ) -> Dict:
        """
        Let one picture vote (see ``view_of``, whose result is returned).

        With ``window_pictures`` only the last so many pictures keep their votes: views taken close together in time
        differ little in the camera's place and heading, so what is left of the camera model's error moves the line
        least between them.
        """

        seen = self.view_of(picture_bgr, camera, hidden)
        self.line_votes += seen["is_line"].astype(numpy.uint16)
        self.paper_votes += seen["is_paper"].astype(numpy.uint16)
        self.pictures += 1
        if window_pictures is not None:
            self.history.append((seen["is_line"], seen["is_paper"]))
            while len(self.history) > window_pictures:
                was_line, was_paper = self.history.popleft()
                self.line_votes -= was_line.astype(numpy.uint16)
                self.paper_votes -= was_paper.astype(numpy.uint16)
        return seen


    def line_cells(self) -> numpy.ndarray:
        """The cells voted line: at least "minimum_line_votes" line votes and "line_vote_share" of all votes."""

        votes = self.line_votes.astype(numpy.float32) + self.paper_votes.astype(numpy.float32)
        return ((self.line_votes >= self.parameters["minimum_line_votes"])
                & (self.line_votes >= self.parameters["line_vote_share"] * votes))



def middle_of_line(
        is_line: numpy.ndarray,
        cell_m: float,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    The cells along the middle of the line: those far from the line's edge.

    Parameters
    ----------
    is_line : numpy.ndarray
        Boolean map of line cells.
    cell_m : float
        The size of a cell in metres.
    parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS`` ("line_width_m", "core_share_of_half_width").

    Returns
    -------
    dict
        "cells" (row, column of the middle cells, shape (N, 2)) and "edge_distance" (every cell's distance to the
        line's edge, in cells).
    """

    parameters = merged_measure_parameters(parameters)
    edge_distance = cv2.distanceTransform(is_line.astype(numpy.uint8), cv2.DIST_L2, 5)
    least = parameters["core_share_of_half_width"] * parameters["line_width_m"] / 2.0 / cell_m
    rows, columns = numpy.nonzero(edge_distance >= least)
    return {"cells": numpy.column_stack([rows, columns]).astype(float), "edge_distance": edge_distance}



def fitted_offset(
        point: numpy.ndarray,
        cells: numpy.ndarray,
        travel: Optional[numpy.ndarray],
        fit_radius_m: float
) -> Dict:
    """
    The distance of a point to a parabola fitted to middle cells of the line, in the cells' own main direction.

    Parameters
    ----------
    point : numpy.ndarray
        The point on the table, (x, y) in metres.
    cells : numpy.ndarray
        Table points of middle cells of one stretch of line, shape (N, 2).
    travel : numpy.ndarray or None
        The unit direction of travel; the line's direction is turned towards it.
    fit_radius_m : float
        The radius the cells were taken from (a stretch much shorter than it is fitted by a straight line).

    Returns
    -------
    dict
        "offset_m" (positive to the left of the line's direction), "foot_xy", "direction_xy", "extrapolated_m" (how
        far beyond the cells the curve was followed to reach the point), "spread_m" (root mean square distance of the
        cells from the curve: about 1.3 mm for one clean stretch, more where two stretches cross), "span_m" (the
        length of the fitted stretch) and "alignment" (the cosine between the line's direction and the travel, 1
        without a travel).
    """

    centre = cells.mean(axis = 0)
    _, _, axes = numpy.linalg.svd(cells - centre[None, :], full_matrices = False)
    along_axis = axes[0]
    if travel is not None and float(along_axis @ travel) < 0.0:
        along_axis = -along_axis
    # the left of the line's direction, seen from above (from x towards y is counter-clockwise)
    left_axis = numpy.array([-along_axis[1], along_axis[0]])
    along = (cells - centre[None, :]) @ along_axis
    across = (cells - centre[None, :]) @ left_axis
    degree = 2 if float(along.max() - along.min()) > 0.6 * fit_radius_m else 1
    curve = numpy.poly1d(numpy.polyfit(along, across, degree))
    slope = curve.deriv()
    bend = slope.deriv()
    spread = float(numpy.sqrt(numpy.mean((across - curve(along)) ** 2)))
    span = float(along.max() - along.min())

    point_along = float((point - centre) @ along_axis)
    point_across = float((point - centre) @ left_axis)
    # the nearest point of the curve: a few Newton steps on the squared distance, started beside the point
    foot_along = point_along
    for _ in range(8):
        gap = float(curve(foot_along)) - point_across
        gradient = (foot_along - point_along) + gap * float(slope(foot_along))
        curvature = 1.0 + float(slope(foot_along)) ** 2 + gap * float(bend(foot_along))
        if abs(curvature) < 1e-9:
            break
        foot_along -= gradient / curvature
    foot_across = float(curve(foot_along))
    tangent = numpy.array([1.0, float(slope(foot_along))])
    tangent /= numpy.linalg.norm(tangent)
    direction = tangent[0] * along_axis + tangent[1] * left_axis
    return {
        "offset_m": float(tangent[0] * (point_across - foot_across) - tangent[1] * (point_along - foot_along)),
        "foot_xy": centre + foot_along * along_axis + foot_across * left_axis,
        "direction_xy": direction,
        "extrapolated_m": float(max(0.0, along.min() - foot_along, foot_along - along.max())),
        "spread_m": spread,
        "span_m": span,
        "alignment": 1.0 if travel is None else float(direction @ travel)
    }



def offset_from_line(
        point_xy: Sequence[float],
        middle_xy: numpy.ndarray,
        travel_xy: Optional[Sequence[float]] = None,
        parameters: Optional[Dict] = None,
        is_ahead_only: bool = False
) -> Optional[Dict]:
    """
    The sideways offset between a point on the table and the line's middle.

    The middle cells within "fit_radius_m" of the point's nearest middle cell are fitted by a parabola in their own
    main direction (a straight line would cut the corner on the loops), and the point's distance to that curve is
    the offset. Where the line comes near itself (the loops, the crossing) the nearest stretch may be the wrong one:
    a stretch that runs across the direction of travel, or whose cells do not lie on one curve, is passed over for
    the next nearest, up to "branch_tries" stretches; when none fits, the nearest is reported and flagged.

    The line under the puck is hidden in the picture it is measured in, so with one picture the curve is followed
    back from the line seen ahead; with the votes of several pictures the line was seen before the puck reached it.

    Parameters
    ----------
    point_xy : sequence of float
        The point (the centre of the puck's foot), metres in the base frame.
    middle_xy : numpy.ndarray
        The table points of the line's middle cells, shape (N, 2).
    travel_xy : sequence of float, optional
        The direction of travel on the table; with it the offset is signed, positive to the left of the line when
        looking along the travel.
    parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS`` ("search_radius_m", "fit_radius_m", "fit_minimum_cells", "branch_tries",
        "branch_angle_most_deg", "fit_spread_most_m", "ahead_cone_deg", "beside_radius_m").
    is_ahead_only : bool, optional
        Use only middle cells beside the point (within "beside_radius_m") or ahead of it (within "ahead_cone_deg" of
        the direction of travel): for one picture alone, in which the line behind the hand is another stretch.

    Returns
    -------
    dict or None
        None when no middle cell lies within "search_radius_m" or too few lie near it. Otherwise the entries of
        ``fitted_offset`` ("offset_m" positive when ``travel_xy`` is not given), "nearest_m" (the distance to the
        stretch's nearest middle cell), "cells" (how many cells were fitted), "is_clean" (the cells lie on one curve)
        and "is_aligned" (the stretch runs along the direction of travel).
    """

    parameters = merged_measure_parameters(parameters)
    point = numpy.asarray(point_xy, dtype = float)
    travel = None
    if travel_xy is not None:
        travel = numpy.asarray(travel_xy, dtype = float)
        travel = travel / max(float(numpy.linalg.norm(travel)), 1e-12)
    if len(middle_xy) == 0:
        return None
    relative = middle_xy - point[None, :]
    distances = numpy.hypot(relative[:, 0], relative[:, 1])
    is_candidate = distances <= parameters["search_radius_m"]
    if is_ahead_only and travel is not None:
        cosine = (relative @ travel) / numpy.maximum(distances, 1e-9)
        is_candidate &= (distances <= parameters["beside_radius_m"]) | (cosine >= math.cos(math.radians(parameters["ahead_cone_deg"])))
    least_alignment = math.cos(math.radians(parameters["branch_angle_most_deg"]))
    fallback = None
    for _ in range(parameters["branch_tries"]):
        if not is_candidate.any():
            break
        nearest = int(numpy.argmin(numpy.where(is_candidate, distances, numpy.inf)))
        anchor = middle_xy[nearest]
        is_near = numpy.hypot(middle_xy[:, 0] - anchor[0], middle_xy[:, 1] - anchor[1]) <= parameters["fit_radius_m"]
        # the next try looks beyond this stretch
        is_candidate &= ~is_near
        if numpy.count_nonzero(is_near) < parameters["fit_minimum_cells"]:
            continue
        result = fitted_offset(point, middle_xy[is_near], travel, parameters["fit_radius_m"])
        result.update({"nearest_m": float(distances[nearest]), "cells": int(numpy.count_nonzero(is_near)),
                       "is_clean": bool(result["spread_m"] <= parameters["fit_spread_most_m"]),
                       "is_aligned": bool(abs(result["alignment"]) >= least_alignment)})
        if travel is None:
            result["offset_m"] = abs(result["offset_m"])
        if result["is_clean"] and result["is_aligned"]:
            return result
        if fallback is None:
            fallback = result
    return fallback



def end_cover(
        puck_xy: Sequence[float],
        line_map: LineMap,
        travel_xy: Sequence[float],
        parameters: Optional[Dict] = None
) -> Optional[Dict]:
    """
    How the puck stands on the line's end at the stop.

    The line's end is looked for in the run's line map around the puck's last place: the line's direction there comes
    from its middle cells, the end's tip is where the line cells stop along that direction within a narrow band, and
    the centre of the round end lies half a line width before the tip. (Sheet 06's dashes end square; its last dash is
    treated the same way, so its "centre" is the point half a line width before the square end.)

    Parameters
    ----------
    puck_xy : sequence of float
        The centre of the puck's foot at the stop, metres in the base frame.
    line_map : LineMap
        The run's line map.
    travel_xy : sequence of float
        The direction of travel over the last steps, on the table.
    parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS``.

    Returns
    -------
    dict or None
        None when no line is found near the puck. Otherwise "end_centre_xy" (the centre of the round end),
        "direction_xy" (the line's direction at its end), "short_m" (how far the puck's centre is short of the end's
        centre along the line; negative when it is beyond), "aside_m" (the puck's centre to the left of the line's
        middle, looking along it; negative to the right), "centre_gap_m" (the distance between the two centres),
        "uncovered_m" (how far the end's rim sticks out from under the puck at its worst point; 0 or less means the
        end is covered completely), "ink_beyond_mm2" (line cells ahead of the puck's centre and outside its foot, the
        ink a person sees beyond the puck from above), "is_end_seen" (bare paper was seen just beyond the tip, so the
        tip is the line's real end and not the edge of what the pictures showed), "line_width_m" (the line's width
        in the map just before the round end, a check of the map) and "is_direction_from_line" (False when only a stub
        of the line was seen and the direction of travel stands in for the line's direction).
    """

    parameters = merged_measure_parameters(parameters)
    puck = numpy.asarray(puck_xy, dtype = float)
    travel = numpy.asarray(travel_xy, dtype = float)
    travel = travel / max(float(numpy.linalg.norm(travel)), 1e-12)
    is_line = line_map.line_cells()
    middle = middle_of_line(is_line, line_map.cell_m, parameters)
    middle_xy = line_map.point_of(middle["cells"])
    if len(middle_xy) == 0:
        return None
    rows, columns = numpy.nonzero(is_line)
    line_xy = line_map.point_of(numpy.column_stack([rows, columns]))
    # the stretch of line the puck came along: its nearest stretch that runs along the travel. When only a stub of
    # the line was seen (an episode that starts near the end), the stub is too short to give a direction, and the
    # direction of travel stands in, through the middle of the line cells near the puck.
    nearest = offset_from_line(puck, middle_xy, travel, dict(parameters, fit_radius_m = parameters["end_look_back_m"] / 2.0))
    is_direction_from_line = bool(nearest is not None and nearest["is_clean"] and nearest["is_aligned"]
                                  and nearest["span_m"] >= parameters["end_direction_span_least_m"])
    if is_direction_from_line:
        direction = nearest["direction_xy"] / numpy.linalg.norm(nearest["direction_xy"])
        foot = nearest["foot_xy"]
    else:
        is_close = numpy.hypot(line_xy[:, 0] - puck[0], line_xy[:, 1] - puck[1]) <= parameters["end_look_back_m"] / 2.0
        if numpy.count_nonzero(is_close) < parameters["fit_minimum_cells"]:
            return None
        direction = travel
        sideways = numpy.array([-travel[1], travel[0]])
        foot = puck + float(numpy.median((line_xy[is_close] - puck[None, :]) @ sideways)) * sideways
    left = numpy.array([-direction[1], direction[0]])
    line_along = (line_xy - foot[None, :]) @ direction
    line_across = (line_xy - foot[None, :]) @ left
    is_band = (numpy.abs(line_across) <= parameters["end_band_m"]) & (line_along >= -parameters["end_look_back_m"]) & (line_along <= parameters["end_look_back_m"])
    band_along = numpy.sort(line_along[is_band])
    if len(band_along) < parameters["fit_minimum_cells"]:
        return None
    # the band's cells in runs along the line (a dashed line has several); the end is the far end of the last run
    # that begins before or close ahead of the puck, so that a later dash or a speck further on is not taken
    breaks = numpy.flatnonzero(numpy.diff(band_along) > parameters["end_gap_m"])
    starts = numpy.concatenate([[0], breaks + 1])
    stops = numpy.concatenate([breaks, [len(band_along) - 1]])
    tip_along = None
    for start, stop in zip(starts, stops):
        is_long = band_along[stop] - band_along[start] >= parameters["end_run_least_m"]
        if is_long and band_along[start] <= parameters["end_ahead_most_m"]:
            tip_along = float(band_along[stop])
    if tip_along is None:
        return None
    half_width = parameters["line_width_m"] / 2.0
    # the line's sideways middle at its end: the line cells of the last centimetre (the round end and as much line
    # before it), taken from a band wide enough to hold the line's full width where it has curved away from the foot
    is_last = (numpy.abs(line_across) <= parameters["end_wide_band_m"]) & (line_along <= tip_along + line_map.cell_m) & (line_along >= tip_along - 0.01)
    middle_across = float(numpy.mean(line_across[is_last])) if numpy.count_nonzero(is_last) >= 20 else 0.0
    end_centre = foot + (tip_along - half_width) * direction + middle_across * left

    # the width of the line in the map over the half centimetre before the round end
    is_slice = (numpy.abs(line_across - middle_across) <= parameters["end_wide_band_m"]) & (line_along <= tip_along - half_width) & (line_along >= tip_along - 2.0 * half_width)
    width = float(numpy.count_nonzero(is_slice) * line_map.cell_m ** 2 / half_width) if numpy.count_nonzero(is_slice) else float("nan")

    # bare paper just beyond the tip
    beyond = end_centre + (half_width + parameters["end_paper_check_m"]) * direction
    cell = numpy.round(line_map.cell_of(beyond[None, :])[0]).astype(int)
    is_end_seen = False
    if 0 <= cell[0] < is_line.shape[0] and 0 <= cell[1] < is_line.shape[1]:
        is_end_seen = bool(line_map.paper_votes[cell[0], cell[1]] >= 2 and not is_line[cell[0], cell[1]])

    gap = end_centre - puck
    # the rim of the round end (a half circle beyond its centre): its furthest point from the puck's centre
    angles = numpy.linspace(-math.pi / 2.0, math.pi / 2.0, 181)
    rim = end_centre[None, :] + half_width * (numpy.cos(angles)[:, None] * direction[None, :] + numpy.sin(angles)[:, None] * left[None, :])
    uncovered = float(numpy.hypot(rim[:, 0] - puck[0], rim[:, 1] - puck[1]).max() - parameters["puck_radius_m"])
    is_ahead = ((line_xy - puck[None, :]) @ direction > 0.0) & (numpy.abs((line_xy - end_centre[None, :]) @ left) <= parameters["end_band_m"])
    is_outside = numpy.hypot(line_xy[:, 0] - puck[0], line_xy[:, 1] - puck[1]) > parameters["puck_radius_m"]
    is_before_tip = line_along <= tip_along + line_map.cell_m
    ink_beyond = float(numpy.count_nonzero(is_ahead & is_outside & is_before_tip) * (1000.0 * line_map.cell_m) ** 2)
    return {
        "end_centre_xy": end_centre,
        "direction_xy": direction,
        "short_m": float(gap @ direction),
        "aside_m": float(-(gap @ left)),
        "centre_gap_m": float(numpy.linalg.norm(gap)),
        "uncovered_m": uncovered,
        "ink_beyond_mm2": ink_beyond,
        "is_end_seen": is_end_seen,
        "line_width_m": width,
        "is_direction_from_line": is_direction_from_line
    }
