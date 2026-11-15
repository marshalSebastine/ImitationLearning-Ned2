"""
Where the printed line and the puck were on the table in a robot run or a demonstration, read from the logged pictures.

The sheets of the robot test of 11.11.26 were laid by hand, so their place differs from the demonstrations by a few
centimetres. To compare a robot run with the demonstrations without guessing the sheet's place, two things are read
from the pictures with the fitted forearm camera model (02_code/forearm_camera_model.py) and the puck detector
(02_code/puck_tracking.py):
    the line map    every picture's ink pixels are put onto the table; a table cell is line when most pictures that
                    saw it saw ink there;
    the puck track  the puck's centre on the table in every picture.
A demonstration's puck track lies on the line of its own sheet placement, so turning and shifting it until it lies on
a robot run's line map gives the move between the two sheet placements (``register_track_to_line``); the same move
then carries the demonstration's tip path into the robot run's frame.

Run from this folder (no robot, reads pictures only):
    python line_and_puck.py          builds tracks/run_NN.npz for the ten robot runs and tracks/demonstration_<plan>.npz
                                     for the standard-start demonstrations the runs are compared with
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy
import pandas

import step_length_common as common

from config import WORKSPACE_PARAMETERS
from forearm_camera_model import viewing_rays_in_base
from puck_tracking import PuckTracker

LINE_AND_PUCK_PARAMETERS = {
    # the camera shows the black ink as dark blue and the paper as light blue; limits checked on pictures of runs 5, 6
    # and 9 and of demonstration 03-01 (ink: hue 99 to 118 of 180, saturation 130 to 255, value 62 to 187; paper: hue 94
    # to 104, saturation 41 to 114, value 200 to 241)
    "ink_hue_range": [90, 125],
    "ink_saturation_least": 125,
    "ink_value_most": 190,
    "paper_value_least": 195,
    "paper_saturation_most": 124,
    "paper_patch_least_px": 3000,
    # the parts of the picture the robot's own hand, pusher and jaws always cover, and the corner where its blue base
    # shows at the start of a run (each: left, top, right, bottom)
    "robot_areas_px": [[180, 330, 640, 480], [0, 400, 180, 480]],
    # every this many pixels in both directions is put onto the table
    "pixel_stride": 2,
    # the table grid of the line map, metres
    "map_x_m": [0.05, 0.56],
    "map_y_m": [-0.36, 0.36],
    "map_cell_m": 0.002,
    # a cell is line when at least this share of the pictures that saw it saw ink, and at least this many saw it (1:
    # the first centimetres of the line are free of the puck and the pusher only in the first picture or two of a run);
    # a patch of line cells smaller than "line_patch_least_cells" is a speck (a shadow), not line
    "line_share_least": 0.5,
    "seen_votes_least": 1,
    "line_patch_least_cells": 25,
    # the printed line is 1 cm wide (config.SHEET_PARAMETERS["line_width_m"])
    "line_width_m": 0.010,
    # the demonstrations' puck tracks are only needed to lay their sheets onto a robot run's line map (sheets 03 and
    # 04, the two overlays asked for), so every "demonstration_picture_stride"-th picture is enough
    "demonstration_sheets": ["path_03", "path_04"],
    "demonstration_picture_stride": 2,
    # the registration search: turn in degrees and shift in metres around the first guess, coarse then fine
    "registration_turn_deg": [-30.0, 30.0, 2.5],
    "registration_shift_m": [0.06, 0.005],
    "registration_fine_turn_deg": [2.5, 0.5],
    "registration_fine_shift_m": [0.006, 0.001],
    "registration_distance_cap_m": 0.02,
    # which demonstrations each robot run is compared with: the sheet's standard starts from the end the run began at
    # (the corrected ends of the data audit: plan 02-01 really started from end B, plan 05-01 from end B)
    "reference_end": {4: "B", 5: "A", 6: "A", 7: "B", 8: "A", 9: "A", 10: "A"},
    "tracks_folder": common.EVIDENCE_FOLDER / "tracks",
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data")
}
JOINT_COLUMNS = [f"joint_{number}" for number in range(1, 7)]



def merged_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``LINE_AND_PUCK_PARAMETERS`` with any overrides merged over it."""

    merged = dict(LINE_AND_PUCK_PARAMETERS)
    merged.update(parameters or {})
    return merged



def sheet_masks(
        picture_bgr: numpy.ndarray,
        parameters: Optional[Dict] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return which pixels show the ink of the line and which show white paper.

    Ink counts only inside the outline of the sheet (the convex outline of the largest bright patch), so that dark blue
    things beside the sheet (a hoodie, the robot's shadow on the table) are not taken for ink, and never in the
    robot's own areas.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The 640 x 480 camera picture as OpenCV reads it.
    parameters : dict, optional
        Overrides of ``LINE_AND_PUCK_PARAMETERS``.

    Returns
    -------
    tuple of numpy.ndarray
        Two boolean pictures: ink and paper.
    """

    settings = merged_parameters(parameters)
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    is_ink = ((hue >= settings["ink_hue_range"][0]) & (hue <= settings["ink_hue_range"][1])
              & (saturation >= settings["ink_saturation_least"]) & (value <= settings["ink_value_most"]))
    is_paper = (value >= settings["paper_value_least"]) & (saturation <= settings["paper_saturation_most"])
    closed = cv2.morphologyEx(is_paper.astype(numpy.uint8), cv2.MORPH_CLOSE, numpy.ones((5, 5), numpy.uint8))
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(closed)
    is_on_sheet = numpy.zeros(is_paper.shape, dtype = numpy.uint8)
    # every large bright patch belongs to the sheet: the line itself can cut the paper into two patches (sheet 01)
    large_patches = [label for label in range(1, count) if statistics[label, cv2.CC_STAT_AREA] >= settings["paper_patch_least_px"]]
    if large_patches:
        contours, _ = cv2.findContours(numpy.isin(labels, large_patches).astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(is_on_sheet, [cv2.convexHull(numpy.vstack(contours))], -1, 1, thickness = -1)
    is_ink = is_ink & (is_on_sheet > 0)
    is_paper = is_paper & (is_on_sheet > 0)
    for left, top, right, bottom in settings["robot_areas_px"]:
        is_ink[top:bottom, left:right] = False
        is_paper[top:bottom, left:right] = False
    return is_ink, is_paper



def table_points_from_pixels(
        pixels: numpy.ndarray,
        joints: Sequence[float]
) -> numpy.ndarray:
    """
    Return where the rays through pixels meet the taught, slightly tilted table plane.

    Parameters
    ----------
    pixels : numpy.ndarray
        Pixels (u, v) of the 640 x 480 picture, shape (N, 2).
    joints : sequence of float
        The six joint angles in radians when the picture was taken.

    Returns
    -------
    numpy.ndarray
        Table points (x, y) in metres in the base frame of the robot, shape (N, 2); NaN where the ray misses the table.
    """

    plane = WORKSPACE_PARAMETERS["table_plane"]
    camera_position, rays, is_valid = viewing_rays_in_base(numpy.asarray(pixels, dtype = float), list(joints))
    denominator = rays[:, 2] - plane["a"] * rays[:, 0] - plane["b"] * rays[:, 1]
    numerator = plane["a"] * camera_position[0] + plane["b"] * camera_position[1] + plane["c"] - camera_position[2]
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        distance = numerator / denominator
    points = camera_position[None, :2] + distance[:, None] * rays[:, :2]
    points[~is_valid | ~numpy.isfinite(distance) | (distance <= 0.0)] = numpy.nan
    return points



def line_map_and_puck_track(
        picture_paths: List[Path],
        joint_rows: numpy.ndarray,
        parameters: Optional[Dict] = None,
        is_line_mapped: bool = True
) -> Dict[str, numpy.ndarray]:
    """
    Put the ink of many pictures onto the table and find the puck in every picture.

    Parameters
    ----------
    picture_paths : list of pathlib.Path
        The pictures (640 x 480 JPEG), in step order.
    joint_rows : numpy.ndarray
        The six joint angles in radians of every picture, shape (pictures, 6).
    parameters : dict, optional
        Overrides of ``LINE_AND_PUCK_PARAMETERS``.
    is_line_mapped : bool, optional
        False skips the line map (only the puck track is needed for a demonstration).

    Returns
    -------
    dict of str to numpy.ndarray
        "ink_votes" and "seen_votes" (indexed [x cell, y cell]), "x_m" and "y_m" (the lower edges of the cells), and
        "puck_xy_m" (pictures, 2; NaN where no puck was found) with "puck_confidence".
    """

    settings = merged_parameters(parameters)
    cell = settings["map_cell_m"]
    x_edges = numpy.arange(settings["map_x_m"][0], settings["map_x_m"][1] + cell / 2, cell)
    y_edges = numpy.arange(settings["map_y_m"][0], settings["map_y_m"][1] + cell / 2, cell)
    ink_votes = numpy.zeros((len(x_edges) - 1, len(y_edges) - 1))
    seen_votes = numpy.zeros_like(ink_votes)
    stride = settings["pixel_stride"]
    grid_v, grid_u = numpy.mgrid[0:480:stride, 0:640:stride]
    tracker = PuckTracker()
    puck_xy = numpy.full((len(picture_paths), 2), numpy.nan)
    puck_confidence = numpy.zeros(len(picture_paths))
    for index, (picture_path, joints) in enumerate(zip(picture_paths, joint_rows)):
        picture = cv2.imread(str(picture_path))
        if picture is None:
            continue
        seen = tracker.see_puck(picture, list(joints))
        if seen is not None and seen["table_xy"] is not None:
            puck_xy[index] = seen["table_xy"]
            puck_confidence[index] = seen["confidence"]
        if not is_line_mapped:
            continue
        is_ink, is_paper = sheet_masks(picture, settings)
        is_ink_sampled = is_ink[::stride, ::stride]
        is_used = is_ink_sampled | is_paper[::stride, ::stride]
        pixels = numpy.column_stack([grid_u[is_used], grid_v[is_used]])
        points = table_points_from_pixels(pixels, joints)
        is_on_table = numpy.isfinite(points).all(axis = 1)
        is_ink_point = is_ink_sampled[is_used][is_on_table]
        points = points[is_on_table]
        # one vote per picture and cell, however many pixels of the picture fall into the cell
        seen_cells = numpy.histogram2d(points[:, 0], points[:, 1], bins = [x_edges, y_edges])[0]
        ink_cells = numpy.histogram2d(points[is_ink_point, 0], points[is_ink_point, 1], bins = [x_edges, y_edges])[0]
        seen_votes += seen_cells > 0
        ink_votes += ink_cells > 0.5 * numpy.maximum(seen_cells, 1.0)
    return {"ink_votes": ink_votes, "seen_votes": seen_votes, "x_m": x_edges[:-1], "y_m": y_edges[:-1],
            "puck_xy_m": puck_xy, "puck_confidence": puck_confidence}



def line_mask(
        track: Dict[str, numpy.ndarray],
        parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """Return the cells of a line map that are line: seen often enough, and ink in most of the pictures that saw them."""

    settings = merged_parameters(parameters)
    share = track["ink_votes"] / numpy.maximum(track["seen_votes"], 1.0)
    is_line = (track["seen_votes"] >= settings["seen_votes_least"]) & (share >= settings["line_share_least"])
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(is_line.astype(numpy.uint8), connectivity = 8)
    for label in range(1, count):
        if statistics[label, cv2.CC_STAT_AREA] < settings["line_patch_least_cells"]:
            is_line[labels == label] = False
    return is_line



def distance_to_line_m(
        track: Dict[str, numpy.ndarray],
        parameters: Optional[Dict] = None,
        is_start_added: bool = False
) -> numpy.ndarray:
    """
    Return, for every cell of a line map, the distance in metres to the nearest line cell (0 on the line).

    With ``is_start_added`` the line under the puck's first place is drawn in first: that piece and the centimetre or
    two in front of it are hidden by the puck and the pusher in every picture that could see them, so a strip as wide
    as the line is drawn from the puck's first place (the puck is put on the line by hand) to the nearest mapped line
    cell.
    """

    settings = merged_parameters(parameters)
    is_line = line_mask(track, settings).astype(numpy.uint8)
    puck_xy = track["puck_xy_m"]
    seen = numpy.flatnonzero(numpy.isfinite(puck_xy[:, 0]))
    if is_start_added and len(seen) and is_line.any():
        cell = settings["map_cell_m"]
        start = puck_xy[seen[0]]
        start_cell = (int((start[0] - track["x_m"][0]) / cell), int((start[1] - track["y_m"][0]) / cell))
        line_cells = numpy.argwhere(is_line > 0)
        nearest = line_cells[numpy.argmin(((line_cells - numpy.array(start_cell)) ** 2).sum(axis = 1))]
        # OpenCV points are (column, row) = (y cell, x cell) for a grid indexed [x cell, y cell]
        cv2.line(is_line, (start_cell[1], start_cell[0]), (int(nearest[1]), int(nearest[0])), 1, thickness = int(round(settings["line_width_m"] / cell)))
    return cv2.distanceTransform((1 - is_line).astype(numpy.uint8), cv2.DIST_L2, 5) * settings["map_cell_m"]



def values_at_points(
        grid: numpy.ndarray,
        track: Dict[str, numpy.ndarray],
        points_xy_m: numpy.ndarray,
        outside_value: float,
        parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """Read a map grid at table points (nearest cell); ``outside_value`` for points off the grid or NaN."""

    settings = merged_parameters(parameters)
    cell = settings["map_cell_m"]
    points = numpy.atleast_2d(numpy.asarray(points_xy_m, dtype = float))
    with numpy.errstate(invalid = "ignore"):
        columns = numpy.floor((points[:, 0] - track["x_m"][0]) / cell)
        rows = numpy.floor((points[:, 1] - track["y_m"][0]) / cell)
        is_inside = numpy.isfinite(columns) & numpy.isfinite(rows) & (columns >= 0) & (columns < grid.shape[0]) & (rows >= 0) & (rows < grid.shape[1])
    values = numpy.full(len(points), outside_value, dtype = float)
    values[is_inside] = grid[columns[is_inside].astype(int), rows[is_inside].astype(int)]
    return values



def moved_points(
        points_xy_m: numpy.ndarray,
        turn_deg: float,
        shift_m: Sequence[float],
        pivot_xy_m: Sequence[float]
) -> numpy.ndarray:
    """Turn points by ``turn_deg`` about ``pivot_xy_m`` (counter-clockwise seen from above), then shift them."""

    angle = math.radians(turn_deg)
    rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    pivot = numpy.asarray(pivot_xy_m, dtype = float)
    return (numpy.asarray(points_xy_m, dtype = float) - pivot) @ rotation.T + pivot + numpy.asarray(shift_m, dtype = float)



def register_track_to_line(
        points_xy_m: numpy.ndarray,
        track: Dict[str, numpy.ndarray],
        first_shift_m: Sequence[float],
        parameters: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Find the turn and shift that lay a puck track of a demonstration onto the line map of a robot run.

    The puck of a demonstration stayed on the line, so its track is the line of its own sheet placement. The turn is
    about the track's first point; the cost is the mean distance of the moved points from the line of the map, each
    distance capped so that points the map never saw do not decide. Searched on a coarse grid around the first guess,
    then on a fine one.

    Parameters
    ----------
    points_xy_m : numpy.ndarray
        The puck track of the demonstration, (points, 2), without NaN rows.
    track : dict
        From ``line_map_and_puck_track`` for the robot run.
    first_shift_m : sequence of float
        The first guess of the shift (the difference of the pucks' first positions).
    parameters : dict, optional
        Overrides of ``LINE_AND_PUCK_PARAMETERS``.

    Returns
    -------
    dict
        "turn_deg", "shift_x_m", "shift_y_m", "pivot_x_m", "pivot_y_m", "mean_distance_m" (after the fit, capped) and
        "share_within_5_mm".
    """

    settings = merged_parameters(parameters)
    cap = settings["registration_distance_cap_m"]
    distance = numpy.minimum(distance_to_line_m(track, settings, is_start_added = True), cap)
    pivot = points_xy_m[0]

    def best_on_grid(turns: numpy.ndarray, shifts_x: numpy.ndarray, shifts_y: numpy.ndarray) -> Tuple[float, float, float, float]:
        best = (math.inf, 0.0, 0.0, 0.0)
        for turn_deg in turns:
            turned = moved_points(points_xy_m, turn_deg, (0.0, 0.0), pivot)
            for shift_x in shifts_x:
                for shift_y in shifts_y:
                    value = float(values_at_points(distance, track, turned + numpy.array([shift_x, shift_y]), cap, settings).mean())
                    if value < best[0]:
                        best = (value, float(turn_deg), float(shift_x), float(shift_y))
        return best

    low, high, step = settings["registration_turn_deg"]
    reach, shift_step = settings["registration_shift_m"]
    offsets = numpy.arange(-reach, reach + shift_step / 2, shift_step)
    _, turn_deg, shift_x, shift_y = best_on_grid(numpy.arange(low, high + step / 2, step), first_shift_m[0] + offsets, first_shift_m[1] + offsets)
    fine_reach, fine_step = settings["registration_fine_turn_deg"]
    fine_shift_reach, fine_shift_step = settings["registration_fine_shift_m"]
    fine_offsets = numpy.arange(-fine_shift_reach, fine_shift_reach + fine_shift_step / 2, fine_shift_step)
    value, turn_deg, shift_x, shift_y = best_on_grid(numpy.arange(turn_deg - fine_reach, turn_deg + fine_reach + fine_step / 2, fine_step),
                                                     shift_x + fine_offsets, shift_y + fine_offsets)
    final = values_at_points(distance, track, moved_points(points_xy_m, turn_deg, (shift_x, shift_y), pivot), cap, settings)
    return {"turn_deg": turn_deg, "shift_x_m": shift_x, "shift_y_m": shift_y, "pivot_x_m": float(pivot[0]), "pivot_y_m": float(pivot[1]),
            "mean_distance_m": value, "share_within_5_mm": float((final <= 0.005).mean())}



def run_track_path(run_number: int, parameters: Optional[Dict] = None) -> Path:
    """Return the file that holds the line map and puck track of a robot run."""

    return Path(merged_parameters(parameters)["tracks_folder"]) / f"run_{run_number:02d}.npz"



def demonstration_track_path(plan_number: str, parameters: Optional[Dict] = None) -> Path:
    """Return the file that holds the puck track of a demonstration."""

    return Path(merged_parameters(parameters)["tracks_folder"]) / f"demonstration_{plan_number}.npz"



def reference_demonstrations(
        frames: pandas.DataFrame,
        run: Dict,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    Return the rows of the demonstrations a robot run is compared with: the same sheet, the standard start, and the
    end the run began at; empty for a run without such demonstrations (sheet 01, whose end is not in the brief, and
    run 3, whose sheet lay as in no plan).
    """

    settings = merged_parameters(parameters)
    end = settings["reference_end"].get(run["number"])
    if end is None:
        return frames.iloc[0:0]
    return frames[(frames["sheet"] == run["sheet"]) & (frames["start_kind"] == common.STEP_LENGTH_PARAMETERS["reference_start_kind"])
                  & (frames["start_end"] == end)]



def main() -> None:

    settings = merged_parameters()
    Path(settings["tracks_folder"]).mkdir(parents = True, exist_ok = True)
    # several analyses share this laptop's six cores
    cv2.setNumThreads(2)
    lines = []
    frames = common.corrected_start_ends(common.load_demonstration_frames())
    joints_of_frames = pandas.read_csv(Path(common.STEP_LENGTH_PARAMETERS["index_folder"]) / "frames.csv", usecols = ["frame_index"] + JOINT_COLUMNS,
                                       low_memory = False).set_index("frame_index")
    plans_done = set()
    for run in common.STEP_LENGTH_PARAMETERS["robot_runs"]:
        started = time.perf_counter()
        steps = common.load_robot_run(run)
        folder = Path(common.STEP_LENGTH_PARAMETERS["model_runs_folder"]) / run["folder"]
        track = line_map_and_puck_track([folder / name for name in steps["frame"]], steps[JOINT_COLUMNS].to_numpy(dtype = float), settings)
        numpy.savez_compressed(run_track_path(run["number"], settings), **track)
        common.print_and_keep(lines, f"run {run['number']:2d}: {len(steps)} pictures, puck found in {int(numpy.isfinite(track['puck_xy_m'][:, 0]).sum())}, "
                                     f"line cells {int(line_mask(track, settings).sum())} ({time.perf_counter() - started:.0f} s)")
        if run["sheet"] not in settings["demonstration_sheets"]:
            continue
        for episode_id, group in reference_demonstrations(frames, run, settings).groupby("episode_id", sort = False):
            plan_number = group["plan_number"].iloc[0]
            if plan_number in plans_done:
                continue
            plans_done.add(plan_number)
            started = time.perf_counter()
            group = group.iloc[::settings["demonstration_picture_stride"]]
            joint_rows = joints_of_frames.loc[group["frame_index"].to_numpy(), JOINT_COLUMNS].to_numpy(dtype = float)
            track = line_map_and_puck_track([Path(settings["raw_folder"]) / name for name in group["frame_path"]], joint_rows, settings, is_line_mapped = False)
            numpy.savez_compressed(demonstration_track_path(plan_number, settings), puck_xy_m = track["puck_xy_m"], puck_confidence = track["puck_confidence"],
                                   row_in_episode = group["row_in_episode"].to_numpy(), tip_xy_m = group[["tip_x_m", "tip_y_m"]].to_numpy())
            common.print_and_keep(lines, f"   demonstration {plan_number}: {len(group)} pictures, puck found in "
                                         f"{int(numpy.isfinite(track['puck_xy_m'][:, 0]).sum())} ({time.perf_counter() - started:.0f} s)")
    (common.EVIDENCE_FOLDER / "line_and_puck.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")



if __name__ == "__main__":
    main()
