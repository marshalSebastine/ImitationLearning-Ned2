"""
The picture measures of picture_measures.py applied to whole runs: the robot's model runs of 11.11.26 and the recorded
demonstrations. Reads pictures and logged joints only; nothing connects to the robot, nothing is written into 03_data.

    robot_run_steps, demonstration_steps    the pictures, joints and tip positions of one run, in step order;
    camera_parameters_for                   the forearm camera with the picture shift found for the run's period
                                            (camera_shift.py writes camera_shift.json);
    measure_steps                           per step: where the puck stands, and its sideways offset from the line,
                                            once against a map voted by the last pictures and once from the step's
                                            own picture alone; at the last step: how the puck stands on the line's end;
    map_sharpness, view_registration        two checks of the camera model: how well the pictures of a run agree on
                                            where the line lies.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy
import pandas

import picture_measures
from picture_measures import LineMap, TableCamera

import forearm_camera_model
from config import TRAINING_DATA_PARAMETERS
from puck_detector import PuckDetector


EVIDENCE_FOLDER = Path(__file__).resolve().parent
RUN_MEASURE_PARAMETERS = {
    # the pictures whose votes make the line map at a step (the step's own and the ones before it)
    "window_pictures": 12,
    # the end of a run is measured against the votes of this many last pictures
    "end_window_pictures": 16,
    # the direction of travel at a step: the tip's move between the nearest steps before and after that lie at
    # least this far apart
    "travel_baseline_m": 0.015,
    # one picture alone: its line is fitted over a longer stretch, because the curve must be followed back to the puck
    "single_fit_radius_m": 0.035,
    # the camera's view moved on 8.11.26 between 18:07 and 19:33 (the data audit); episodes from then on and the robot
    # test of 11.11.26 are the later period
    "camera_moved_at": "2026-11-08T19:00:00",
    "camera_shift_file": "camera_shift.json",
    "opencv_threads": 2
}
JOINT_COLUMNS = [f"joint_{number}" for number in range(1, 7)]



def merged_run_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``RUN_MEASURE_PARAMETERS`` with any overrides merged over it."""

    merged = dict(RUN_MEASURE_PARAMETERS)
    merged.update(parameters or {})
    return merged



def robot_run_steps(run_folder: Path) -> pandas.DataFrame:
    """
    The steps of one model run: "picture_path", the six joints and the magnet tip ("tip_x", "tip_y", metres).

    Parameters
    ----------
    run_folder : Path
        A folder of 03_data/model_runs with steps.csv and frames/.

    Returns
    -------
    pandas.DataFrame
        One row per picture, in step order, with every column of steps.csv.
    """

    run_folder = Path(run_folder)
    steps = pandas.read_csv(run_folder / "steps.csv")
    steps["picture_path"] = [str(run_folder / frame) for frame in steps["frame"]]
    return steps



def demonstration_steps(
        frames: pandas.DataFrame,
        episode_id: str
) -> pandas.DataFrame:
    """
    The rows of one recorded demonstration as steps: "picture_path", the six joints and the magnet tip.

    Parameters
    ----------
    frames : pandas.DataFrame
        frames.csv of the training index.
    episode_id : str
        The episode.

    Returns
    -------
    pandas.DataFrame
        One row per picture, in step order; "tip_x" and "tip_y" are the index's "tip_x_m" and "tip_y_m".
    """

    raw_folder = Path(TRAINING_DATA_PARAMETERS["raw_folder"])
    steps = frames[frames["episode_id"] == episode_id].sort_values("row_in_episode").reset_index(drop = True).copy()
    steps["picture_path"] = [str(raw_folder / frame_path) for frame_path in steps["frame_path"]]
    steps["tip_x"] = steps["tip_x_m"]
    steps["tip_y"] = steps["tip_y_m"]
    return steps



def camera_period(started: str, parameters: Optional[Dict] = None) -> str:
    """The camera period of a run from its start time (ISO text): "before_move" or "after_move"."""

    parameters = merged_run_parameters(parameters)
    return "after_move" if str(started) >= parameters["camera_moved_at"] else "before_move"



def corrected_camera_parameters(
        shift_px: Sequence[float],
        roll_deg: float = 0.0
) -> Dict:
    """
    The overrides of ``FOREARM_CAMERA_PARAMETERS`` for a camera whose picture is shifted and turned against the fitted
    one: the principal point moved by ``shift_px`` (right, down) and the camera turned by ``roll_deg`` about its own
    viewing direction.

    Returns
    -------
    dict
        "principal_point_px" and "rotation_vector_in_forearm_rad", to pass on as camera parameters.
    """

    fitted = forearm_camera_model.merged_camera_parameters()
    angle = math.radians(roll_deg)
    roll = numpy.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    rotation = forearm_camera_model.rotation_matrix_from_vector(fitted["rotation_vector_in_forearm_rad"]) @ roll
    return {"principal_point_px": [fitted["principal_point_px"][0] + shift_px[0], fitted["principal_point_px"][1] + shift_px[1]],
            "rotation_vector_in_forearm_rad": forearm_camera_model.rotation_vector_from_matrix(rotation).tolist()}



def camera_parameters_for(
        period: str,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    The overrides of ``FOREARM_CAMERA_PARAMETERS`` for a camera period: the picture shift and the roll that
    camera_shift.py found for the period (the fitted camera when its result file is missing).

    Parameters
    ----------
    period : str
        "before_move", "after_move" or "robot_test".
    parameters : dict, optional
        Overrides for ``RUN_MEASURE_PARAMETERS`` ("camera_shift_file").

    Returns
    -------
    dict
        See ``corrected_camera_parameters``.
    """

    parameters = merged_run_parameters(parameters)
    shift_path = EVIDENCE_FOLDER / parameters["camera_shift_file"]
    shift, roll = [0.0, 0.0], 0.0
    if shift_path.exists():
        found = json.loads(shift_path.read_text(encoding = "utf-8"))[period]
        shift, roll = found["shift_px"], found.get("roll_deg", 0.0)
    return corrected_camera_parameters(shift, roll)



def map_ranges(
        steps: pandas.DataFrame,
        parameters: Optional[Dict] = None
) -> tuple:
    """The x and y ranges of a line map around the tip's path of a run, with the map margin."""

    margin = picture_measures.merged_measure_parameters(parameters)["map_margin_m"]
    return ([float(steps["tip_x"].min()) - margin, float(steps["tip_x"].max()) + margin],
            [float(steps["tip_y"].min()) - margin, float(steps["tip_y"].max()) + margin])



def travel_direction(
        tips_xy: numpy.ndarray,
        index: int,
        baseline_m: float
) -> numpy.ndarray:
    """
    The direction of travel at a step: from the nearest earlier to the nearest later tip that lie at least
    ``baseline_m`` apart (a twist in place has no direction of its own).

    Returns
    -------
    numpy.ndarray
        A unit vector on the table; (1, 0) when the tip never moves that far.
    """

    first, last = index, index
    while True:
        move = tips_xy[last] - tips_xy[first]
        if float(numpy.linalg.norm(move)) >= baseline_m:
            return move / numpy.linalg.norm(move)
        if first == 0 and last == len(tips_xy) - 1:
            length = float(numpy.linalg.norm(move))
            return move / length if length > 1e-6 else numpy.array([1.0, 0.0])
        first = max(first - 1, 0)
        last = min(last + 1, len(tips_xy) - 1)



def measure_steps(
        steps: pandas.DataFrame,
        camera_parameters: Optional[Dict] = None,
        parameters: Optional[Dict] = None,
        measure_parameters: Optional[Dict] = None,
        first_step: int = 0,
        is_offset_measured: bool = True,
        detector: Optional[PuckDetector] = None,
        used_steps: Optional[Sequence[int]] = None,
        fitted_last_steps: int = 1
) -> Dict:
    """
    Measure a run step by step from its pictures.

    Parameters
    ----------
    steps : pandas.DataFrame
        From ``robot_run_steps`` or ``demonstration_steps``.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS`` (``camera_parameters_for``).
    parameters : dict, optional
        Overrides for ``RUN_MEASURE_PARAMETERS``.
    measure_parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS``.
    first_step : int, optional
        The first row used (to measure only the end of a run).
    is_offset_measured : bool, optional
        False measures only the end (the puck is then fitted in the last pictures only), which is much faster.
    detector : PuckDetector, optional
        A kept detector.
    used_steps : sequence of int, optional
        The rows used, in rising order and ending with the run's last row, instead of every row from ``first_step``
        (to measure the end from the pictures that showed the end's place, however long ago).
    fitted_last_steps : int, optional
        With ``is_offset_measured`` False: in how many of the last used pictures the puck is fitted.

    Returns
    -------
    dict
        "rows" (a pandas.DataFrame, one row per step: "step", "puck_x", "puck_y" (metres), "fit_quality",
        "offset_mm" (the puck's centre to the left of the line's middle, against the map voted by the last pictures;
        negative to the right), "nearest_mm", "extrapolated_mm", "line_cells", "is_clean" and "is_aligned" (see
        ``offset_from_line``), "whole_run_offset_mm" (the same against the votes of every picture of the run), and
        the offset from the step's own picture alone as "single_offset_mm", "single_nearest_mm",
        "single_extrapolated_mm", "is_single_clean" and "is_single_aligned"), "end" (``end_cover`` at
        the last step, or None), "line_map" (the map with the last pictures' votes), "last" (the last picture, its
        camera, blob and puck fit, and the line cells and seen cells of that picture alone) and "all_line_votes" and
        "all_paper_votes" (the votes of every picture of the run, for drawing the whole line).
    """

    parameters = merged_run_parameters(parameters)
    settings = picture_measures.merged_measure_parameters(measure_parameters)
    cv2.setNumThreads(parameters["opencv_threads"])
    detector = detector or PuckDetector()
    steps = steps.reset_index(drop = True)
    tips_xy = steps[["tip_x", "tip_y"]].to_numpy(dtype = float)
    used_steps = list(range(first_step, len(steps))) if used_steps is None else [int(index) for index in used_steps]
    x_range, y_range = map_ranges(steps.iloc[used_steps], settings)
    line_map = LineMap(x_range, y_range, settings)
    single_settings = dict(settings, fit_radius_m = parameters["single_fit_radius_m"])
    window = max(parameters["window_pictures"], parameters["end_window_pictures"])

    rows = []
    last = None
    all_line_votes = numpy.zeros(line_map.line_votes.shape, dtype = numpy.uint16)
    all_paper_votes = numpy.zeros(line_map.line_votes.shape, dtype = numpy.uint16)
    for position, index in enumerate(used_steps):
        row = steps.iloc[index]
        is_last = index == len(steps) - 1
        is_fitted = is_offset_measured or position >= len(used_steps) - fitted_last_steps
        picture = cv2.imread(row["picture_path"])
        if picture is None:
            continue
        camera = TableCamera([row[name] for name in JOINT_COLUMNS], camera_parameters)
        blob = picture_measures.puck_blob(picture, detector, settings)
        hidden = picture_measures.hidden_table_mask(picture.shape[:2], camera, blob, settings)
        seen = line_map.add_picture(picture, camera, hidden, window_pictures = window)
        all_line_votes += seen["is_line"].astype(numpy.uint16)
        all_paper_votes += seen["is_paper"].astype(numpy.uint16)
        measured = {"step": index, "puck_x": numpy.nan, "puck_y": numpy.nan, "fit_quality": numpy.nan,
                    "offset_mm": numpy.nan, "nearest_mm": numpy.nan, "extrapolated_mm": numpy.nan, "line_cells": 0,
                    "is_clean": False, "is_aligned": False, "whole_run_offset_mm": numpy.nan,
                    "single_offset_mm": numpy.nan, "single_nearest_mm": numpy.nan, "single_extrapolated_mm": numpy.nan,
                    "is_single_clean": False, "is_single_aligned": False}
        fit = None
        if blob is not None and is_fitted:
            fit = picture_measures.fit_puck_on_table(blob, camera, settings)
            measured.update({"puck_x": float(fit["base_xy"][0]), "puck_y": float(fit["base_xy"][1]), "fit_quality": fit["quality"]})
        if fit is not None and is_offset_measured and fit["quality"] >= settings["fit_quality_minimum"]:
            travel = travel_direction(tips_xy, index, parameters["travel_baseline_m"])
            # the map of the last "window_pictures": the votes of the pictures that left that window are taken out
            # by counting the kept ones again (the map itself keeps the longer end window)
            recent = list(line_map.history)[-parameters["window_pictures"]:]
            line_votes = numpy.sum([entry[0] for entry in recent], axis = 0, dtype = numpy.uint16)
            paper_votes = numpy.sum([entry[1] for entry in recent], axis = 0, dtype = numpy.uint16)
            is_line = (line_votes >= settings["minimum_line_votes"]) & (line_votes >= settings["line_vote_share"] * (line_votes + paper_votes))
            middle = picture_measures.middle_of_line(is_line, line_map.cell_m, settings)
            offset = picture_measures.offset_from_line(fit["base_xy"], line_map.point_of(middle["cells"]), travel, settings)
            if offset is not None:
                measured.update({"offset_mm": 1000.0 * offset["offset_m"], "nearest_mm": 1000.0 * offset["nearest_m"],
                                 "extrapolated_mm": 1000.0 * offset["extrapolated_m"], "line_cells": offset["cells"],
                                 "is_clean": offset["is_clean"], "is_aligned": offset["is_aligned"]})
            single_middle = picture_measures.middle_of_line(seen["is_line"], line_map.cell_m, settings)
            single = picture_measures.offset_from_line(fit["base_xy"], line_map.point_of(single_middle["cells"]), travel, single_settings,
                                                       is_ahead_only = True)
            if single is not None:
                measured.update({"single_offset_mm": 1000.0 * single["offset_m"], "single_nearest_mm": 1000.0 * single["nearest_m"],
                                 "single_extrapolated_mm": 1000.0 * single["extrapolated_m"],
                                 "is_single_clean": single["is_clean"], "is_single_aligned": single["is_aligned"]})
        rows.append(measured)
        if is_last:
            last = {"camera": camera, "blob": blob, "fit": fit, "picture": picture, "is_line": seen["is_line"], "is_seen": seen["is_seen"]}

    if is_offset_measured:
        # the same offsets against the votes of every picture of the run: a check of the camera model, because views
        # taken far apart in time must then agree on where the line lies
        is_line = (all_line_votes >= settings["minimum_line_votes"]) & (all_line_votes >= settings["line_vote_share"] * (all_line_votes.astype(numpy.float32) + all_paper_votes))
        middle_xy = line_map.point_of(picture_measures.middle_of_line(is_line, line_map.cell_m, settings)["cells"])
        for measured in rows:
            if not numpy.isfinite(measured["offset_mm"]):
                continue
            travel = travel_direction(tips_xy, measured["step"], parameters["travel_baseline_m"])
            offset = picture_measures.offset_from_line([measured["puck_x"], measured["puck_y"]], middle_xy, travel, settings)
            if offset is not None:
                measured["whole_run_offset_mm"] = 1000.0 * offset["offset_m"]

    end = None
    if last is not None and last["fit"] is not None:
        # the end is measured against the votes of the pictures before the last one's puck hid the line's end
        while len(line_map.history) > parameters["end_window_pictures"]:
            was_line, was_paper = line_map.history.popleft()
            line_map.line_votes -= was_line.astype(numpy.uint16)
            line_map.paper_votes -= was_paper.astype(numpy.uint16)
        travel = travel_direction(tips_xy, len(steps) - 1, 2.0 * parameters["travel_baseline_m"])
        end = picture_measures.end_cover(last["fit"]["base_xy"], line_map, travel, settings)
    return {"rows": pandas.DataFrame(rows), "end": end, "line_map": line_map, "last": last,
            "all_line_votes": all_line_votes, "all_paper_votes": all_paper_votes}



def loaded_views(
        steps: pandas.DataFrame,
        every: int = 1,
        detector: Optional[PuckDetector] = None
) -> List[Dict]:
    """
    The pictures of a run with their joints and puck blobs, kept in memory for the checks of the camera model.

    Parameters
    ----------
    steps : pandas.DataFrame
        From ``robot_run_steps`` or ``demonstration_steps``.
    every : int, optional
        Keep every so many pictures.
    detector : PuckDetector, optional
        A kept detector.

    Returns
    -------
    list of dict
        "joints", "picture" and "blob" per kept picture.
    """

    detector = detector or PuckDetector()
    views = []
    for index in range(0, len(steps), every):
        row = steps.iloc[index]
        picture = cv2.imread(row["picture_path"])
        if picture is None:
            continue
        views.append({"joints": [float(row[name]) for name in JOINT_COLUMNS], "picture": picture,
                      "blob": picture_measures.puck_blob(picture, detector)})
    return views



def map_sharpness(
        views: List[Dict],
        x_range: Sequence[float],
        y_range: Sequence[float],
        camera_parameters: Optional[Dict] = None,
        measure_parameters: Optional[Dict] = None
) -> float:
    """
    How well the pictures of a run agree on where the line lies: the mean number of pictures that vote "line" on a
    cell, weighted by the votes (the sum of squared votes over the sum of votes). A camera model that puts the same
    piece of line in different places from different views spreads the votes and lowers it.

    Parameters
    ----------
    views : list of dict
        From ``loaded_views``.
    x_range, y_range : sequence of float
        The map's ranges on the table.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.
    measure_parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS`` (a coarser "map_mm_per_px" makes it faster).

    Returns
    -------
    float
        The sharpness; larger is better.
    """

    line_map = LineMap(x_range, y_range, measure_parameters)
    for view in views:
        camera = TableCamera(view["joints"], camera_parameters)
        hidden = picture_measures.hidden_table_mask(view["picture"].shape[:2], camera, view["blob"], line_map.parameters)
        line_map.add_picture(view["picture"], camera, hidden)
    votes = line_map.line_votes.astype(float)
    return float((votes ** 2).sum() / max(float(votes.sum()), 1.0))



def view_registration(
        views: List[Dict],
        x_range: Sequence[float],
        y_range: Sequence[float],
        steps_apart: int,
        camera_parameters: Optional[Dict] = None,
        measure_parameters: Optional[Dict] = None,
        smallest_shared_cells: int = 400,
        nearly_best_share: float = 0.985
) -> pandas.DataFrame:
    """
    How far apart two views of a run put the same piece of line on the table.

    For every pair of pictures ``steps_apart`` apart, the line cells of both (only cells both pictures show) are
    slid over each other, and the smallest slide with (nearly) the largest overlap is the disagreement of the two views.

    Parameters
    ----------
    views : list of dict
        From ``loaded_views`` (every picture).
    x_range, y_range : sequence of float
        The map's ranges on the table.
    steps_apart : int
        The distance between the two pictures of a pair.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.
    measure_parameters : dict, optional
        Overrides for ``MEASURE_PARAMETERS``.
    smallest_shared_cells : int, optional
        Pairs that share fewer line cells are left out.
    nearly_best_share : float, optional
        A slide counts as fitting when its overlap reaches this share of the largest overlap.

    Returns
    -------
    pandas.DataFrame
        One row per pair: "first", "second", "shift_mm" (the size of the disagreement), "shift_x_mm", "shift_y_mm",
        "shared_cells" and "heading_change_deg" (the change of joint 4 between the two pictures).
    """

    line_map = LineMap(x_range, y_range, measure_parameters)
    seen = []
    for view in views:
        camera = TableCamera(view["joints"], camera_parameters)
        hidden = picture_measures.hidden_table_mask(view["picture"].shape[:2], camera, view["blob"], line_map.parameters)
        result = line_map.view_of(view["picture"], camera, hidden)
        seen.append((result["is_line"], result["is_seen"]))
    reach = 12
    rows = []
    for second in range(steps_apart, len(views)):
        first = second - steps_apart
        is_both = seen[first][1] & seen[second][1]
        line_first = (seen[first][0] & is_both).astype(numpy.float32)
        line_second = (seen[second][0] & is_both).astype(numpy.float32)
        if min(line_first.sum(), line_second.sum()) < smallest_shared_cells:
            continue
        # slide the second view's line over the first's: the overlap for every slide up to "reach" cells
        padded = cv2.copyMakeBorder(line_first, reach, reach, reach, reach, cv2.BORDER_CONSTANT, value = 0)
        overlap = cv2.matchTemplate(padded, line_second, cv2.TM_CCORR)
        # along a straight piece of line every slide overlaps almost equally, so the smallest slide that comes
        # within a little of the largest overlap is taken, not the largest overlap itself
        slide_rows, slide_columns = numpy.nonzero(overlap >= nearly_best_share * float(overlap.max()))
        nearest = int(numpy.argmin(numpy.hypot(slide_rows - reach, slide_columns - reach)))
        shift_columns, shift_rows = int(slide_columns[nearest]) - reach, int(slide_rows[nearest]) - reach
        # rows count down the x axis and columns down the y axis
        shift_x = -shift_rows * line_map.cell_m
        shift_y = -shift_columns * line_map.cell_m
        rows.append({"first": first, "second": second, "shift_mm": 1000.0 * math.hypot(shift_x, shift_y),
                     "shift_x_mm": 1000.0 * shift_x, "shift_y_mm": 1000.0 * shift_y,
                     "shared_cells": int(min(line_first.sum(), line_second.sum())),
                     "heading_change_deg": math.degrees(views[second]["joints"][3] - views[first]["joints"][3])})
    return pandas.DataFrame(rows)
