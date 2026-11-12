"""
Question 5 of the orientation-coverage analysis: which placements of the six course sheets and the six practice tracks
are physically possible, which add the most new coverage, and in which order should they be recorded?

Every candidate placement (a sheet, the end the push starts from, a turn and a shift) is laid on the table by
calculation; the line's traced geometry gives, centimetre by centimetre, the tip's reach and the line's direction from
radial. A placement is possible when the tip stays within the driver's reach and box and the paper stays off the base.
Its line is then sorted into the physics classes of 04_physics.py.

Two families of placements that are easy to lay:
    mark     the start cap where the start caps of the demonstrations lay (the taped start mark); the sheet turned about
             it so that the line leaves at a chosen angle
    centre   the paper's centre on one taped cross in the middle of the reach; the sheet turned in steps of 15 degrees
             (the start is then taught once per lay-out with teach_sheet_start.py)

Coverage is counted in three ways, the same for the demonstrations and for a candidate:
    travel cells   direction of travel from radial (30-degree bins) against reach (5 bins): pictures of all sheets; a
                   cell is full at "full_cell_pictures"
    start cells    the line's direction from radial at the start against the start's reach: episodes; a cell is full
                   at "full_cell_starts"
    new for the sheet   centimetres of the line in travel cells where this sheet has fewer than
                   "new_for_sheet_pictures" pictures (the same line seen in a direction or at a reach it was never
                   pushed in)
A placement's score is how much fuller the travel and start cells become when it is recorded "episodes_per_placement"
times, plus "new_for_sheet_weight" per new-for-the-sheet centimetre; placements are picked one at a time by score per
hour of robot time, each pick added to the tables before the next. Only placements that are easy to lay are picked
(the paper's edges square to the robot, or the demonstrations' own lay-out rule), and only "comfortable" ones: the line
leaves the start within "comfortable_start_deg" of radial and at most "comfortable_back_share" of it lies in the back
sector. The square placements that are not comfortable are listed as test placements, not as recordings.

Printed and saved: 05_recording_plan.out.txt, 05_candidates.csv (every candidate), 05_plan.csv (the picked placements
in order), 05_plan_course_sheets.png and 05_plan_practice_tracks.png (top views of the picked placements).

Run from this folder (after 01_placements.py, 02_coverage.py and 04_physics.py):
    D:/AI_Project/.venv/Scripts/python.exe 05_recording_plan.py
"""

import math
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas
from matplotlib import pyplot

import orientation_common
from orientation_common import (EVIDENCE_FOLDER, arc_lengths, course_sheet_lines, demonstration_frames, practice_track_lines, rotated,
                                wrapped_deg)

from config import MODEL_MODE_PARAMETERS, SETUP_VISUALS_PARAMETERS  # noqa: E402

PLAN_PARAMETERS = {
    "paper_size_cm": SETUP_VISUALS_PARAMETERS["paper_size_cm"],
    "base_half_width_cm": SETUP_VISUALS_PARAMETERS["base_half_width_cm"],
    # the tip must stay inside the driver's reach and box (MODEL_MODE_PARAMETERS), with this margin; the far limit is
    # where the rod's lean passes the driver's limit with the forearm straight (04_physics.py, part A: 48 cm)
    "tip_reach_cm": [100.0 * MODEL_MODE_PARAMETERS["tip_reach_m"][0], 47.5],
    "tip_box_x_cm": [100.0 * value for value in MODEL_MODE_PARAMETERS["workspace_x_m"]],
    "tip_box_y_cm": [100.0 * value for value in MODEL_MODE_PARAMETERS["workspace_y_m"]],
    "reach_margin_cm": 0.5,
    # the paper must not lie on the base's square footprint, with this margin (it may lie beside the base, as some
    # recorded lay-outs did)
    "paper_clear_of_base_cm": 0.5,
    # the mark family: where the start cap lies, and the angles at which the line leaves it (0 = straight away from
    # the robot). The start caps of the 148 start-mark demonstrations lay at a median (19.2, -1.2) cm (calculated from
    # the fits of 01_placements.py; quartiles of x 17.7 to 20.3), about 2.7 cm further out than the set-up pictures
    # draw them (16.4 cm); the tip starts at (14.0, -1.5) and first travels to the puck
    "mark_puck_xy_cm": [19.2, -1.2],
    "mark_start_directions_deg": list(range(-180, 180, 15)),
    # the centre family: the taped cross and the sheet turns
    "centre_xy_cm": [31.0, 0.0],
    "centre_turns_deg": list(range(-180, 180, 15)),
    "direction_bin_deg": 30.0,
    "reach_edges_cm": [12.0, 20.0, 27.5, 35.0, 42.5, 49.5],
    # when a cell counts as full (choices: 150 pictures is about the fewest any direction holds at reach 20 to 42.5 cm;
    # 4 starts is two placements of two episodes)
    "full_cell_pictures": 150.0,
    "full_cell_starts": 4.0,
    # a centimetre is new for its sheet when the sheet's own demonstrations hold fewer pictures than this in its cell;
    # 100 such centimetres weigh as much as one start or travel cell filled (a choice)
    "new_for_sheet_pictures": 30.0,
    "new_for_sheet_weight": 0.01,
    # the square turns of the paper, and what makes a placement comfortable for the arm (04_physics.py)
    "square_turns_deg": [-180.0, -90.0, 0.0, 90.0],
    "comfortable_start_deg": 120.0,
    "comfortable_back_share": 0.35,
    # picking stops when the best remaining placement scores less than this (neither a new start cell nor 30 new
    # centimetres for its sheet)
    "least_score": 0.3,
    # physics classes of a line direction from radial (either side), from 04_physics.py: the camera can point straight
    # along the line up to the largest turn of the picture's top (106 to 123 degrees by reach; 110 used here), and the
    # line stays within 30 degrees of the picture's top up to 140
    "look_along_most_deg": 110.0,
    "back_sector_from_deg": 140.0,
    # episodes recorded per picked placement, and how many pictures one centimetre of line gives (one waypoint per cm)
    "episodes_per_placement": 2,
    "pictures_per_cm": 1.0,
    "course_placements_to_pick": 12,
    "course_placements_per_sheet": 2,
    "practice_placements_per_track": 3,
    # the first placement of every practice track is the lay-out rule of the demonstrations (mark family, the line
    # leaving as near to straight away as is possible), because an unseen exam sheet would be laid by that rule
    "practice_first_family": "mark",
    # the bend the forearm can follow while the driver moves 1 cm and turns at most 10 degrees per step
    "twist_most_deg_per_cm": MODEL_MODE_PARAMETERS["twist_step_most_deg"] / (100.0 * MODEL_MODE_PARAMETERS["step_m"]),
    # the test ladder (no recording): the straight sheet and the arc on the centre cross, every this many degrees
    "ladder_shapes": ["sheet_01", "sheet_02"],
    "ladder_end": "A",
    "ladder_turn_step_deg": 45.0,
    # robot time: minutes per demonstration = a fixed part plus a part per centimetre of line (fitted by eye to the
    # measured median start-to-start times per sheet, 4.6 to 8.0 minutes, episodes.csv), and minutes per new lay-out
    # (taping the sheet and teaching the start: a guess)
    "minutes_per_episode_fixed": 3.5,
    "minutes_per_episode_per_cm": 0.06,
    "minutes_per_layout": 10.0
}



def direction_bins(angle_deg, bin_deg: float) -> numpy.ndarray:
    """Return the centre of the direction bin of each angle, a multiple of ``bin_deg`` in -180 + bin to 180."""

    centres = wrapped_deg(numpy.round(numpy.atleast_1d(numpy.asarray(angle_deg, dtype = float)) / bin_deg) * bin_deg)
    centres[centres == -180.0] = 180.0
    return centres



def placed_candidate(
        line_cm: numpy.ndarray,
        turn_deg: float,
        shift_cm: numpy.ndarray,
        parameters: Dict
) -> Dict:
    """
    Lay a line on the table and describe it centimetre by centimetre.

    Parameters
    ----------
    line_cm : numpy.ndarray
        The line in the sheet's frame, ordered from the start end, every 0.1 cm, shape (N, 2).
    turn_deg : float
        The sheet frame's turn on the table, counter-clockwise.
    shift_cm : numpy.ndarray
        Where the sheet's bottom-left corner lies on the table.
    parameters : dict
        ``PLAN_PARAMETERS`` merged over ``ORIENTATION_PARAMETERS``.

    Returns
    -------
    dict
        "line" (the placed line), "paper" (its four corners), and per centimetre of line: "puck" (the puck's centre),
        "tip" (where the tip is then), "reach_cm", "from_radial_deg" and "curve_deg_per_cm"; plus "is_possible" and
        "why_not".
    """

    width, height = parameters["paper_size_cm"]
    paper = rotated(numpy.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]), turn_deg) + shift_cm
    line = rotated(line_cm, turn_deg) + shift_cm
    lengths = arc_lengths(line)
    samples = numpy.arange(0.5, lengths[-1], 1.0)

    def at(arc: numpy.ndarray) -> numpy.ndarray:
        return numpy.column_stack([numpy.interp(arc, lengths, line[:, 0]), numpy.interp(arc, lengths, line[:, 1])])

    puck = at(samples)
    tangent = at(numpy.minimum(samples + 0.5, lengths[-1])) - at(numpy.maximum(samples - 0.5, 0.0))
    tangent = tangent / numpy.linalg.norm(tangent, axis = 1)[:, None]
    tip = puck - parameters["tip_behind_puck_cm"] * tangent
    travel_deg = numpy.degrees(numpy.arctan2(tangent[:, 1], tangent[:, 0]))
    reach = numpy.hypot(tip[:, 0], tip[:, 1])
    from_radial = wrapped_deg(travel_deg - numpy.degrees(numpy.arctan2(tip[:, 1], tip[:, 0])))
    curve = numpy.abs(wrapped_deg(numpy.diff(travel_deg, append = travel_deg[-1])))

    margin = parameters["reach_margin_cm"]
    reasons = []
    if reach.min() < parameters["tip_reach_cm"][0] + margin:
        reasons.append("tip too near the base")
    if reach.max() > parameters["tip_reach_cm"][1] - margin:
        reasons.append("tip too far out")
    if tip[:, 0].min() < parameters["tip_box_x_cm"][0] or tip[:, 0].max() > parameters["tip_box_x_cm"][1]:
        reasons.append("tip outside the box in x")
    if tip[:, 1].min() < parameters["tip_box_y_cm"][0] or tip[:, 1].max() > parameters["tip_box_y_cm"][1]:
        reasons.append("tip outside the box in y")
    # the paper lies on the base when one of its edge points is inside the base's square or the square's corner is
    # inside the paper
    half = parameters["base_half_width_cm"] + parameters["paper_clear_of_base_cm"]
    edge_points = numpy.concatenate([numpy.linspace(paper[index], paper[(index + 1) % 4], 60) for index in range(4)])
    base_corners = numpy.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    in_paper = rotated(base_corners - shift_cm, -turn_deg)
    is_corner_in_paper = numpy.any((in_paper[:, 0] > 0) & (in_paper[:, 0] < width) & (in_paper[:, 1] > 0) & (in_paper[:, 1] < height))
    if numpy.any(numpy.all(numpy.abs(edge_points) < half, axis = 1)) or is_corner_in_paper:
        reasons.append("paper on the base")
    return {"line": line, "paper": paper, "puck": puck, "tip": tip, "reach_cm": reach, "from_radial_deg": from_radial, "curve_deg_per_cm": curve,
            "start_direction_deg": float(travel_deg[0]), "is_possible": not reasons, "why_not": "; ".join(reasons)}



def candidate_row(
        shape: str,
        end: str,
        family: str,
        turn_deg: float,
        placed: Dict,
        parameters: Dict
) -> Dict:
    """Return the table row of one candidate: its place, whether it is possible, its physics classes and its time."""

    absolute = numpy.abs(placed["from_radial_deg"])
    crossings = int(numpy.sum((numpy.abs(numpy.diff(placed["from_radial_deg"])) > 180.0)))
    centre = placed["paper"].mean(axis = 0)
    minutes = parameters["minutes_per_episode_fixed"] + parameters["minutes_per_episode_per_cm"] * len(absolute)
    return {
        "shape": shape, "end": end, "family": family, "turn_deg": round(float(turn_deg), 1),
        "paper_centre_x_cm": round(float(centre[0]), 1), "paper_centre_y_cm": round(float(centre[1]), 1),
        "puck_start_x_cm": round(float(placed["line"][0, 0]), 1), "puck_start_y_cm": round(float(placed["line"][0, 1]), 1),
        "start_direction_deg": round(placed["start_direction_deg"], 0),
        "start_from_radial_deg": round(float(placed["from_radial_deg"][0]), 0), "start_reach_cm": round(float(placed["reach_cm"][0]), 1),
        "line_cm": len(absolute), "reach_least_cm": round(float(placed["reach_cm"].min()), 1), "reach_most_cm": round(float(placed["reach_cm"].max()), 1),
        "is_possible": placed["is_possible"], "why_not": placed["why_not"],
        "outward_cm": int((absolute <= 45).sum()),
        "sideways_cm": int(((absolute > 45) & (absolute <= parameters["look_along_most_deg"])).sum()),
        "back_slanted_cm": int(((absolute > parameters["look_along_most_deg"]) & (absolute <= parameters["back_sector_from_deg"])).sum()),
        "back_sector_cm": int((absolute > parameters["back_sector_from_deg"]).sum()),
        "flips": crossings,
        "tight_bend_cm": int((placed["curve_deg_per_cm"] > parameters["twist_most_deg_per_cm"]).sum()),
        "minutes_per_episode": round(minutes, 1),
        "is_square": bool(min(abs(float(wrapped_deg(turn_deg - square))) for square in parameters["square_turns_deg"]) < 0.5),
        "is_comfortable": bool(abs(float(placed["from_radial_deg"][0])) <= parameters["comfortable_start_deg"]
                               and (absolute > parameters["back_sector_from_deg"]).mean() <= parameters["comfortable_back_share"])
    }



def all_candidates(parameters: Dict) -> List[Dict]:
    """
    Build every candidate placement of the two families for the six course sheets and the six practice tracks.

    Returns
    -------
    list of dict
        One entry per candidate: "row" (the table row) and "placed" (the laid line and its per-centimetre values).
    """

    shapes = {f"sheet_{number}": sheet["line"] for number, sheet in course_sheet_lines().items()}
    shapes.update({name: track["line"] for name, track in practice_track_lines(parameters).items()})
    width, height = parameters["paper_size_cm"]
    paper_centre = numpy.array([width / 2, height / 2])
    candidates = []
    for shape, line_from_a in shapes.items():
        for end in ("A", "B"):
            line = line_from_a if end == "A" else line_from_a[::-1]
            own_start_deg = math.degrees(math.atan2(*(line[10] - line[0])[::-1]))
            mark_turns = [float(wrapped_deg(start_direction - own_start_deg)) for start_direction in parameters["mark_start_directions_deg"]]
            for turn in mark_turns + list(parameters["square_turns_deg"]):
                shift = numpy.array(parameters["mark_puck_xy_cm"]) - rotated(line[:1], turn)[0]
                placed = placed_candidate(line, turn, shift, parameters)
                candidates.append({"row": candidate_row(shape, end, "mark", turn, placed, parameters), "placed": placed})
            for turn in parameters["centre_turns_deg"]:
                shift = numpy.array(parameters["centre_xy_cm"]) - rotated(paper_centre[None, :], turn)[0]
                placed = placed_candidate(line, float(turn), shift, parameters)
                candidates.append({"row": candidate_row(shape, end, "centre", float(turn), placed, parameters), "placed": placed})
    return candidates



def cell_indices(
        from_radial_deg,
        reach_cm,
        parameters: Dict
) -> tuple:
    """Return the (reach bin, direction bin) indices of directions from radial and reaches."""

    bin_deg = parameters["direction_bin_deg"]
    centres = numpy.arange(-180.0 + bin_deg, 180.0 + 1e-9, bin_deg)
    edges = numpy.array(parameters["reach_edges_cm"])
    direction_index = numpy.searchsorted(centres, direction_bins(from_radial_deg, bin_deg))
    reach_index = numpy.clip(numpy.searchsorted(edges, numpy.atleast_1d(numpy.asarray(reach_cm, dtype = float)), side = "right") - 1, 0, len(edges) - 2)
    return reach_index, direction_index



def demonstration_tables(
        frames: pandas.DataFrame,
        parameters: Dict
) -> Dict:
    """
    Count what the demonstrations hold: pictures per travel cell (overall and per sheet) and episodes per start cell.

    The travel direction is the three-step path direction, which follows the line and not the hand's jitter. A
    start-mark episode's start is its sheet's fitted line start (01_placements_per_episode.csv); a taught start's is
    the net direction of its first five moving steps (02_first_five_steps.csv).

    Returns
    -------
    dict
        "travel" (reach bins x direction bins), "travel_per_shape" {shape: array}, "starts" (same shape, episodes).
    """

    shape = (len(parameters["reach_edges_cm"]) - 1, int(round(360.0 / parameters["direction_bin_deg"])))
    rows = frames[frames["path_table_deg"].notna()]
    reach_index, direction_index = cell_indices(rows["path_from_radial_deg"], rows["reach_cm"], parameters)
    travel = numpy.zeros(shape)
    per_shape = {}
    for sheet_number in sorted(rows["sheet_number"].unique()):
        is_sheet = (rows["sheet_number"] == sheet_number).to_numpy()
        counts = numpy.zeros(shape)
        numpy.add.at(counts, (reach_index[is_sheet], direction_index[is_sheet]), 1.0)
        per_shape[f"sheet_{sheet_number}"] = counts
        travel += counts

    layouts = pandas.read_csv(EVIDENCE_FOLDER / "01_placements_per_episode.csv", dtype = {"sheet": str})
    first_steps = pandas.read_csv(EVIDENCE_FOLDER / "02_first_five_steps.csv").set_index("episode_id")
    sheets = course_sheet_lines()
    starts = numpy.zeros(shape)
    for _, layout in layouts.iterrows():
        if layout["start_kind"] in ("NE", "MID", "X1", "X2"):
            if layout["episode_id"] not in first_steps.index:
                continue
            first = first_steps.loc[layout["episode_id"]]
            from_radial, reach = first["net_direction_from_radial_deg"], first["start_reach_cm"]
        else:
            line = sheets[layout["sheet"]]["line"] if layout["end"] == "A" else sheets[layout["sheet"]]["line"][::-1]
            cap = rotated(line[:1], layout["turn_deg"])[0] + numpy.array([layout["shift_x_cm"], layout["shift_y_cm"]])
            direction = math.radians(layout["line_start_direction_deg"])
            tip = cap - parameters["tip_behind_puck_cm"] * numpy.array([math.cos(direction), math.sin(direction)])
            from_radial = float(wrapped_deg(layout["line_start_direction_deg"] - math.degrees(math.atan2(tip[1], tip[0]))))
            reach = float(numpy.hypot(*tip))
        reach_index, direction_index = cell_indices(from_radial, reach, parameters)
        starts[reach_index[0], direction_index[0]] += 1.0
    return {"travel": travel, "travel_per_shape": per_shape, "starts": starts}



def candidate_tables(
        placed: Dict,
        parameters: Dict
) -> Dict:
    """Return one episode of a placed line in the two tables: centimetres per travel cell and its one start cell."""

    shape = (len(parameters["reach_edges_cm"]) - 1, int(round(360.0 / parameters["direction_bin_deg"])))
    travel = numpy.zeros(shape)
    reach_index, direction_index = cell_indices(placed["from_radial_deg"], placed["reach_cm"], parameters)
    numpy.add.at(travel, (reach_index, direction_index), parameters["pictures_per_cm"])
    starts = numpy.zeros(shape)
    starts[reach_index[0], direction_index[0]] = 1.0
    return {"travel": travel, "starts": starts}



def fullness(
        tables: Dict,
        parameters: Dict
) -> Dict:
    """Return how full the two tables are: the sum over cells of min(1, count / full count), and the cells' count."""

    travel = numpy.minimum(1.0, tables["travel"] / parameters["full_cell_pictures"])
    starts = numpy.minimum(1.0, tables["starts"] / parameters["full_cell_starts"])
    return {"travel": float(travel.sum()), "starts": float(starts.sum()), "cells": int(travel.size)}



def picked_placements(
        candidates: List[Dict],
        tables: Dict,
        pool_indices: List[int],
        how_many: int,
        parameters: Dict,
        most_per_shape: Optional[int] = None,
        forced_first: Optional[List[int]] = None
) -> pandas.DataFrame:
    """
    Pick placements one at a time, each time the one whose recording adds the most coverage per hour.

    Parameters
    ----------
    candidates : list of dict
        From ``all_candidates``.
    tables : dict
        The coverage tables to start from ("travel", "travel_per_shape" and "starts"); they are not changed.
    pool_indices : list of int
        The indices into ``candidates`` that may be picked.
    how_many : int
        The number of placements to pick, the forced ones included.
    parameters : dict
        ``PLAN_PARAMETERS`` merged over ``ORIENTATION_PARAMETERS``.
    most_per_shape : int, optional
        The most placements of one shape.
    forced_first : list of int, optional
        Indices into ``candidates`` that are picked first, in this order.

    Returns
    -------
    pandas.DataFrame
        The picked candidates' rows in order, with "travel_gain" and "start_gain" (in full cells), "new_for_sheet_cm",
        "score", "hours" and the fullness of the tables after the pick; ``attrs["tables"]`` holds the tables after
        all picks.
    """

    empty = numpy.zeros_like(tables["travel"])
    current = {"travel": tables["travel"].copy(), "starts": tables["starts"].copy(),
               "travel_per_shape": {shape: counts.copy() for shape, counts in tables["travel_per_shape"].items()}}
    episodes = parameters["episodes_per_placement"]
    additions = {index: candidate_tables(candidates[index]["placed"], parameters) for index in list(pool_indices) + list(forced_first or [])}
    picked, picks_per_shape, taken = [], {}, set()
    queue = list(forced_first or [])
    while len(picked) < how_many:
        before = fullness(current, parameters)
        best_index, best_rate, best_parts = None, -1.0, None
        for index in ([queue[0]] if queue else pool_indices):
            row = candidates[index]["row"]
            if index in taken or (not queue and most_per_shape is not None and picks_per_shape.get(row["shape"], 0) >= most_per_shape):
                continue
            after = fullness({"travel": current["travel"] + episodes * additions[index]["travel"],
                              "starts": current["starts"] + episodes * additions[index]["starts"]}, parameters)
            own = current["travel_per_shape"].get(row["shape"], empty)
            new_for_sheet = float(additions[index]["travel"][own < parameters["new_for_sheet_pictures"]].sum())
            hours = (episodes * row["minutes_per_episode"] + parameters["minutes_per_layout"]) / 60.0
            parts = (after["travel"] - before["travel"], after["starts"] - before["starts"], new_for_sheet)
            score = parts[0] + parts[1] + parameters["new_for_sheet_weight"] * parts[2]
            if score / hours > best_rate:
                best_index, best_rate, best_parts = index, score / hours, parts
        if best_index is None:
            break
        best_score = best_parts[0] + best_parts[1] + parameters["new_for_sheet_weight"] * best_parts[2]
        if not queue and best_score < parameters["least_score"]:
            break
        if queue:
            queue.pop(0)
        taken.add(best_index)
        row = dict(candidates[best_index]["row"])
        current["travel"] += episodes * additions[best_index]["travel"]
        current["starts"] += episodes * additions[best_index]["starts"]
        current["travel_per_shape"][row["shape"]] = current["travel_per_shape"].get(row["shape"], empty) + episodes * additions[best_index]["travel"]
        picks_per_shape[row["shape"]] = picks_per_shape.get(row["shape"], 0) + 1
        after = fullness(current, parameters)
        hours = (episodes * row["minutes_per_episode"] + parameters["minutes_per_layout"]) / 60.0
        row.update({"travel_gain": round(best_parts[0], 2), "start_gain": round(best_parts[1], 2), "new_for_sheet_cm": int(best_parts[2]),
                    "score": round(best_parts[0] + best_parts[1] + parameters["new_for_sheet_weight"] * best_parts[2], 2), "hours": round(hours, 2),
                    "travel_full_after": round(after["travel"], 1), "starts_full_after": round(after["starts"], 1), "candidate_index": best_index})
        picked.append(row)
    table = pandas.DataFrame(picked)
    table.attrs["tables"] = current
    return table



def draw_plan(
        plan: pandas.DataFrame,
        candidates: List[Dict],
        title: str,
        figure_path,
        parameters: Dict
) -> None:
    """Draw a top view of every picked placement: the paper, the line by physics class, and its start."""

    columns = 4
    rows = int(math.ceil(len(plan) / columns))
    figure, axes = pyplot.subplots(rows, columns, figsize = (4.7 * columns, 4.7 * rows), squeeze = False)
    half = parameters["base_half_width_cm"]
    for axis in axes.ravel():
        axis.set_visible(False)
    for axis, (order, row) in zip(axes.ravel(), enumerate(plan.to_dict("records"), start = 1)):
        axis.set_visible(True)
        placed = candidates[row["candidate_index"]]["placed"]
        paper = numpy.vstack([placed["paper"], placed["paper"][:1]])
        axis.plot(paper[:, 0], paper[:, 1], "-", color = "0.5", linewidth = 1.0)
        absolute = numpy.abs(placed["from_radial_deg"])
        colours = numpy.where(absolute <= parameters["look_along_most_deg"], "tab:green",
                              numpy.where(absolute <= parameters["back_sector_from_deg"], "gold", "tab:red"))
        axis.scatter(placed["puck"][:, 0], placed["puck"][:, 1], c = colours, s = 14)
        axis.plot(placed["line"][0, 0], placed["line"][0, 1], "o", color = "black", markersize = 8)
        axis.add_patch(pyplot.Rectangle((-half, -half), 2 * half, 2 * half, color = "0.75"))
        for reach in parameters["tip_reach_cm"]:
            axis.add_patch(pyplot.Circle((0.0, 0.0), reach, fill = False, linestyle = "--", color = "0.4", linewidth = 0.8))
        axis.plot(*parameters["centre_xy_cm"], "+", color = "blue", markersize = 10)
        axis.plot(*parameters["mark_puck_xy_cm"], "x", color = "blue", markersize = 8)
        axis.set_aspect("equal")
        axis.set_xlim(-12, 56)
        axis.set_ylim(-32, 32)
        axis.grid(alpha = 0.3)
        axis.set_title(f"{order}. {row['shape']} end {row['end']}, {row['family']}, turn {row['turn_deg']:+.0f}\n"
                       f"line leaves {row['start_direction_deg']:+.0f} deg at ({row['puck_start_x_cm']:.0f}, {row['puck_start_y_cm']:.0f}) cm; "
                       f"back sector {row['back_sector_cm']} of {row['line_cm']} cm, flips {row['flips']}", fontsize = 8)
    figure.suptitle(title + "\n(black dot: start; green: the camera can look along the line; yellow: slanted back; red: back sector; "
                    "blue x: start mark; blue +: centre cross; dashed: tip reach 12 and 47.5 cm)", fontsize = 10)
    figure.tight_layout(rect = (0.0, 0.0, 1.0, 0.95))
    figure.savefig(figure_path, dpi = 75)
    pyplot.close(figure)



def main(parameters: Optional[Dict] = None) -> None:
    """Print the feasibility of the placements, the coverage they add and the picked plans; save the tables and figures."""

    settings = orientation_common.merged_parameters(PLAN_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 260)
    pandas.set_option("display.max_rows", 600)
    pandas.set_option("display.max_columns", 40)
    frames = demonstration_frames(settings)
    tables = demonstration_tables(frames, settings)
    candidates = all_candidates(settings)
    table = pandas.DataFrame([candidate["row"] for candidate in candidates])
    table.to_csv(EVIDENCE_FOLDER / "05_candidates.csv", index = False)
    course_shapes = sorted(shape for shape in table["shape"].unique() if shape.startswith("sheet_"))
    practice_shapes = sorted(shape for shape in table["shape"].unique() if shape.startswith("practice_"))
    bin_deg = settings["direction_bin_deg"]
    centres = numpy.arange(-180.0 + bin_deg, 180.0 + 1e-9, bin_deg).astype(int)
    labels = [f"{low:g} to {high:g}" for low, high in zip(settings["reach_edges_cm"][:-1], settings["reach_edges_cm"][1:])]

    print("=== 1. Which placements are physically possible (calculated from the traced lines and the driver's limits) ===")
    print(f"Tip reach {settings['tip_reach_cm'][0]:g} to {settings['tip_reach_cm'][1]:g} cm with {settings['reach_margin_cm']} cm margin; tip box x "
          f"{settings['tip_box_x_cm']}, y {settings['tip_box_y_cm']} cm; paper off the base's square (half width {settings['base_half_width_cm']} cm, "
          f"margin {settings['paper_clear_of_base_cm']} cm). Table size not checked (unknown here). Mark family: start cap at {settings['mark_puck_xy_cm']} cm; "
          f"centre family: paper centre at {settings['centre_xy_cm']} cm.")
    in_steps = table[(table["family"] == "centre") | ~table["is_square"]]
    possible = in_steps.groupby(["shape", "end", "family"])["is_possible"].agg(["sum", "size"]).reset_index()
    possible["possible"] = possible["sum"].astype(int).astype(str) + " of " + possible["size"].astype(str)
    print("\nPossible placements per sheet, end and family (24 turns each):")
    print(possible.pivot(index = ["shape", "end"], columns = "family", values = "possible").to_string())
    mark = table[table["family"] == "mark"]
    print("\nMark family: the line's start directions (degrees from straight away, + = to the robot's left) that are possible:")
    for (shape, end), part in mark[~mark["is_square"]].groupby(["shape", "end"]):
        directions = sorted(int(value) for value in part.loc[part["is_possible"], "start_direction_deg"])
        print(f"  {shape} end {end}: {directions if directions else 'none'}")
    blocked = table[~table["is_possible"]]
    print("\nWhy the others are not possible (candidates; a candidate may have several reasons):")
    reasons = pandas.Series([reason for text in blocked["why_not"] for reason in text.split("; ")], name = "reason")
    families = pandas.Series([family for family, text in zip(blocked["family"], blocked["why_not"]) for _ in text.split("; ")], name = "family")
    print(pandas.crosstab(families, reasons).to_string())

    print("\n=== 2. The physics classes of a line, by placement (centimetres of line; calculated) ===")
    print(f"outward: within 45 degrees of radial; sideways: 45 to {settings['look_along_most_deg']:.0f} (the camera can still point along the line); "
          f"back-slanted: to {settings['back_sector_from_deg']:.0f}; back sector: beyond (sideways in the picture even at full twist); flips: the line "
          "passes straight back to the base, so the forearm must swing from one side to the other; tight bend: the line turns more than "
          f"{settings['twist_most_deg_per_cm']:.0f} degrees per cm, more than the driver can twist while it moves 1 cm per step.")
    centre = table[(table["family"] == "centre") & table["is_possible"]]
    summary = centre.groupby("shape").agg(
        turns_possible = ("turn_deg", "nunique"), line_cm = ("line_cm", "median"),
        back_sector_cm_least = ("back_sector_cm", "min"), back_sector_cm_median = ("back_sector_cm", "median"), back_sector_cm_most = ("back_sector_cm", "max"),
        flips_least = ("flips", "min"), flips_most = ("flips", "max"), tight_bend_cm = ("tight_bend_cm", "median"))
    print("\nCentre family over all 24 turns and both ends:")
    print(summary.to_string())
    print("\nThe lay-out rule of the demonstrations (mark family, the line leaving straight away), both ends:")
    shown = ["shape", "end", "turn_deg", "line_cm", "reach_least_cm", "reach_most_cm", "outward_cm", "sideways_cm", "back_slanted_cm", "back_sector_cm", "flips",
             "tight_bend_cm", "is_possible", "why_not"]
    standard = mark[mark["start_direction_deg"] == 0].drop_duplicates(["shape", "end"])
    print(standard[shown].to_string(index = False))
    feasible = standard[standard["is_possible"]]
    print(f"  possible for {len(feasible)} of {len(standard)} shape ends; over these the tip stays between {feasible['reach_least_cm'].min():.1f} and "
          f"{feasible['reach_most_cm'].max():.1f} cm from the base axis, and {int(feasible['back_sector_cm'].sum())} of {int(feasible['line_cm'].sum())} cm of line "
          f"({100.0 * feasible['back_sector_cm'].sum() / feasible['line_cm'].sum():.0f} %) lie in the back sector")
    print("\nThe gentlest possible placement of each shape and end (fewest back-sector centimetres, then fewest flips), either family:")
    gentle = table[table["is_possible"]].sort_values(["back_sector_cm", "flips", "back_slanted_cm"]).groupby(["shape", "end"]).head(1).sort_values(["shape", "end"])
    print(gentle[["shape", "end", "family", "turn_deg", "start_direction_deg", "puck_start_x_cm", "puck_start_y_cm", "outward_cm", "sideways_cm",
                  "back_slanted_cm", "back_sector_cm", "flips"]].to_string(index = False))

    print("\n=== 3. What the demonstrations hold (the two coverage tables) ===")
    print("Pictures per travel cell, all sheets (three-step paths; rows: reach in cm, columns: direction from radial):")
    print(pandas.DataFrame(tables["travel"].astype(int), index = labels, columns = centres).to_string())
    print("\nEpisodes per start cell (rows: start reach, columns: the line's direction from radial at the start):")
    print(pandas.DataFrame(tables["starts"].astype(int), index = labels, columns = centres).to_string())
    now = fullness(tables, settings)
    print(f"\nFullness now: travel {now['travel']:.1f} of {now['cells']} cells (a cell is full at {settings['full_cell_pictures']:.0f} pictures); "
          f"starts {now['starts']:.1f} of {now['cells']} cells (full at {settings['full_cell_starts']:.0f} episodes).")
    print("Per sheet: cells holding under 30 pictures of that sheet (of 60): " + ", ".join(
        f"{shape[-2:]}: {int((tables['travel_per_shape'][shape] < 30).sum())}" for shape in course_shapes))
    attainable = numpy.zeros_like(tables["starts"])
    for candidate in candidates:
        if candidate["row"]["is_possible"]:
            attainable += candidate_tables(candidate["placed"], settings)["starts"]
    print(f"Start cells that some possible candidate reaches: {int((attainable > 0).sum())} of {attainable.size}; of these, the demonstrations hold "
          f"{int(((attainable > 0) & (tables['starts'] > 0)).sum())} with at least one start and {int(((attainable > 0) & (tables['starts'] >= settings['full_cell_starts'])).sum())} full.")

    plan_columns = ["shape", "end", "family", "turn_deg", "start_direction_deg", "puck_start_x_cm", "puck_start_y_cm", "start_from_radial_deg", "start_reach_cm",
                    "line_cm", "travel_gain", "start_gain", "new_for_sheet_cm", "score", "hours", "sideways_cm", "back_slanted_cm", "back_sector_cm", "flips",
                    "tight_bend_cm", "starts_full_after"]
    is_rule = (table["family"] == "mark") & (table["start_direction_deg"] == 0) & ~table.duplicated(["shape", "end", "family", "start_direction_deg"])
    is_easy = table["is_possible"] & (table["is_square"] | is_rule)
    print("\n=== 4. The placements that are easy to lay (paper square to the robot in either family, or the demonstrations' lay-out rule) ===")
    easy = table[is_easy]
    print(f"{len(easy)} possible; comfortable (line leaves within {settings['comfortable_start_deg']:.0f} degrees of radial and at most "
          f"{100 * settings['comfortable_back_share']:.0f} % of it in the back sector): {int(easy['is_comfortable'].sum())}")
    print(easy.groupby(["shape", "family"]).agg(possible = ("turn_deg", "size"), comfortable = ("is_comfortable", "sum")).unstack("family").fillna(0).astype(int).to_string())
    print("\nSquare placements that are possible but not comfortable (test placements, hardest first):")
    hard = easy[~easy["is_comfortable"]].sort_values("back_sector_cm", ascending = False)
    print(hard[["shape", "end", "family", "turn_deg", "start_direction_deg", "start_from_radial_deg", "start_reach_cm", "line_cm", "back_sector_cm", "flips"]].head(24).to_string(index = False))

    comfortable = [int(index) for index in table.index[is_easy & table["is_comfortable"]]]
    course_pool = [index for index in comfortable if table.loc[index, "shape"] in course_shapes]
    practice_pool = [index for index in comfortable if table.loc[index, "shape"] in practice_shapes]

    print(f"\nPicking rule: {settings['episodes_per_placement']} episodes per placement; score = travel cells filled + start cells filled + "
          f"{settings['new_for_sheet_weight']} x new-for-the-sheet centimetres; best score per hour first; stop below a score of {settings['least_score']}.")
    print("\n=== 5. Plan C: the six course sheets only (at most two new placements per sheet) ===")
    course_plan = picked_placements(candidates, tables, course_pool, settings["course_placements_to_pick"], settings,
                                    most_per_shape = settings["course_placements_per_sheet"])
    print(course_plan[plan_columns].to_string())

    print("\n=== 6. Plan P: the six practice tracks (first the demonstrations' lay-out rule, then two more placements per track) ===")
    forced = []
    for shape in practice_shapes:
        part = table[(table["shape"] == shape) & (table["family"] == settings["practice_first_family"]) & table["is_possible"]
                     & (table["end"] == "A") & ~table["is_square"]]
        if len(part):
            forced.append(int(part["start_direction_deg"].abs().idxmin()))
    practice_plan = picked_placements(candidates, tables, practice_pool, settings["practice_placements_per_track"] * len(practice_shapes), settings,
                                      most_per_shape = settings["practice_placements_per_track"], forced_first = forced)
    print(practice_plan[plan_columns].to_string())

    print("\n=== 7. Both together: the course sheets picked after plan P is in the tables ===")
    both_plan = picked_placements(candidates, practice_plan.attrs["tables"], course_pool, settings["course_placements_to_pick"], settings,
                                  most_per_shape = settings["course_placements_per_sheet"])
    print(both_plan[plan_columns].to_string())

    print("\n=== 8. A test ladder that needs no recording: the straight sheet and the arc on the centre cross, every "
          f"{settings['ladder_turn_step_deg']:.0f} degrees, easiest first ===")
    is_ladder = ((table["family"] == "centre") & table["shape"].isin(settings["ladder_shapes"]) & (table["end"] == settings["ladder_end"])
                 & (numpy.abs(wrapped_deg(table["turn_deg"] / settings["ladder_turn_step_deg"] * 360.0)) < 1e-6) & table["is_possible"])
    ladder = table[is_ladder].sort_values(["shape", "back_sector_cm", "back_slanted_cm", "sideways_cm"])
    print(ladder[["shape", "end", "turn_deg", "start_direction_deg", "puck_start_x_cm", "puck_start_y_cm", "start_from_radial_deg", "start_reach_cm", "line_cm",
                  "outward_cm", "sideways_cm", "back_slanted_cm", "back_sector_cm", "flips", "is_comfortable"]].to_string(index = False))
    print(f"{len(ladder)} placements; with one run each at about 2.5 to 3 minutes a run (report 8.4) and a taught start per placement, about "
          f"{len(ladder) * (3.0 + 4.0) / 60.0:.1f} hours (a guess: 4 minutes to tape and teach each placement).")

    episodes = settings["episodes_per_placement"]
    print()
    for name, part in [("Plan C (course sheets alone)", course_plan), ("Plan P (practice tracks alone)", practice_plan),
                       ("Course sheets after plan P", both_plan)]:
        end_state = fullness(part.attrs["tables"], settings)
        print(f"{name}: {len(part)} placements x {episodes} episodes = {episodes * len(part)} episodes, about {int(part['line_cm'].sum()) * episodes} pictures, "
              f"{part['hours'].sum():.1f} robot hours; travel cells full {end_state['travel']:.1f} of {end_state['cells']} (now {now['travel']:.1f}), "
              f"start cells full {end_state['starts']:.1f} (now {now['starts']:.1f})")
    plans = []
    for name, part in [("P", practice_plan), ("C after P", both_plan), ("C alone", course_plan)]:
        copy = pandas.DataFrame(part.drop(columns = ["candidate_index"]).to_dict("records"))
        copy.insert(0, "plan", name)
        copy.insert(1, "order", range(1, len(copy) + 1))
        plans.append(copy)
    pandas.concat(plans, ignore_index = True).to_csv(EVIDENCE_FOLDER / "05_plan.csv", index = False)

    draw_plan(course_plan, candidates, "Plan C: picked placements of the six course sheets, in order", EVIDENCE_FOLDER / "05_plan_course_sheets.png", settings)
    draw_plan(practice_plan, candidates, "Plan P: picked placements of the six practice tracks, in order", EVIDENCE_FOLDER / "05_plan_practice_tracks.png", settings)
    print("\nSaved 05_candidates.csv, 05_plan.csv, 05_plan_course_sheets.png, 05_plan_practice_tracks.png")



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding = "utf-8")
    main()
