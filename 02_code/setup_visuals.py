"""
Set-up pictures, one per plan number: where the sheet lies, where the puck goes, as a top view in robot coordinates.

Read-only: nothing here talks to the robot or changes the episode plan. The pictures and a page that lists them by sheet
(index.html) are written to SETUP_VISUALS_PARAMETERS["output_folder"]; open the page in a browser beside the recording
page. What is measured and what is drawn from the plan's words is said in config.py above SETUP_VISUALS_PARAMETERS.

Run from anywhere:
    python setup_visuals.py                 (every plan number)
    python setup_visuals.py --plan 02-07    (one picture, for a quick look)
"""

import argparse
import html
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
from matplotlib import patches, pyplot
from matplotlib.path import Path as PolygonPath

from config import (NEW_TRACK_RECORDING_PARAMETERS, RECORDING_PAGE_PARAMETERS, SETUP_VISUALS_PARAMETERS, SHAPE_OF_SHEET, WORKSPACE_PARAMETERS,
                    start_pose)
from recording_page import expand_episode_plan


CODE_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = CODE_FOLDER.parent
DISTANCE_IN_NOTE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(?: to (\d+(?:\.\d+)?))? cm before the end cap")
RESAMPLE_STEP_CM = 0.1
TURN_CODES = ("TL10", "TR10", "TL15", "TR15")
SQUARE_CODES = ("SQP", "SQL")
TAUGHT_CODES = ("NE", "MID", "X1", "X2")



def load_sheet_lines(parameters: Dict = SETUP_VISUALS_PARAMETERS) -> Dict:
    """
    The traced line and its two ends of every sheet, in the sheet's own frame (cm, x along the bottom edge, y up).

    Returns
    -------
    dict
        Sheet number ("02") -> {"line": (N, 2) array resampled every RESAMPLE_STEP_CM, "ends": the JSON's ends}.
    """

    with open(PROJECT_FOLDER / parameters["sheet_measurements"], encoding = "utf-8") as file:
        measurements = json.load(file)

    sheets = {}
    for number, sheet in measurements.items():
        traced = numpy.array(sheet["line_xy_cm"], dtype = float)
        sheets[number] = {"line": resample_line(traced, step_cm = RESAMPLE_STEP_CM), "ends": sheet["ends"]}
    return sheets



def resample_line(
        line: numpy.ndarray,
        step_cm: float
) -> numpy.ndarray:
    """The polyline ``line`` with points every ``step_cm`` of arc length, both ends kept."""

    arc_length = numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(line, axis = 0), axis = 1))])
    samples = numpy.arange(0.0, arc_length[-1], step_cm).tolist() + [arc_length[-1]]
    return numpy.column_stack([numpy.interp(samples, arc_length, line[:, 0]), numpy.interp(samples, arc_length, line[:, 1])])



def arc_lengths(line: numpy.ndarray) -> numpy.ndarray:
    """Arc length in cm from the first point to every point of ``line``."""

    return numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(line, axis = 0), axis = 1))])



def point_at(
        line: numpy.ndarray,
        distance_cm: float
) -> numpy.ndarray:
    """The point of ``line`` that lies ``distance_cm`` of arc length from its first point."""

    lengths = arc_lengths(line)
    distance_cm = min(max(distance_cm, 0.0), lengths[-1])
    return numpy.array([numpy.interp(distance_cm, lengths, line[:, 0]), numpy.interp(distance_cm, lengths, line[:, 1])])



def curvature_peaks(
        line: numpy.ndarray,
        smoothing_cm: float,
        separation_cm: float
) -> List[Dict]:
    """
    The bends of ``line``, tightest first: arc length and curvature (1/cm) of every local peak of the absolute curvature,
    at least ``separation_cm`` apart.

    The heading is smoothed over ``smoothing_cm`` first, because the traced points carry about a millimetre of noise and
    a raw second derivative would find bends in it.
    """

    steps = numpy.diff(line, axis = 0)
    heading = numpy.unwrap(numpy.arctan2(steps[:, 1], steps[:, 0]))
    window = max(1, int(round(smoothing_cm / RESAMPLE_STEP_CM)))
    kernel = numpy.ones(window) / window
    smoothed = numpy.convolve(numpy.pad(heading, window, mode = "edge"), kernel, mode = "same")[window:-window]
    curvature = numpy.abs(numpy.gradient(smoothed, RESAMPLE_STEP_CM))
    lengths = arc_lengths(line)[:-1] + RESAMPLE_STEP_CM / 2

    peaks = []
    for index in numpy.argsort(-curvature):
        # The first and last centimetre are the round caps' neighbourhood, not bends of the path.
        if lengths[index] < 1.0 or lengths[index] > lengths[-1] - 1.0:
            continue
        if all(abs(lengths[index] - peak["arc_cm"]) >= separation_cm for peak in peaks):
            peaks.append({"arc_cm": float(lengths[index]), "curvature_per_cm": float(curvature[index])})
    return peaks



def self_crossings(line: numpy.ndarray) -> List[List[float]]:
    """
    Where ``line`` crosses itself: for each crossing the arc lengths of the first and the second pass, in cm.
    """

    lengths = arc_lengths(line)
    starts, ends = line[:-1], line[1:]
    crossings = []
    for index in range(len(starts)):
        # Segments close along the line touch their neighbours; only passes far apart along the line can cross.
        others = numpy.arange(index + 20, len(starts))
        if len(others) == 0:
            continue
        direction = ends[index] - starts[index]
        other_direction = ends[others] - starts[others]
        offset = starts[others] - starts[index]
        denominator = direction[0] * other_direction[:, 1] - direction[1] * other_direction[:, 0]
        with numpy.errstate(divide = "ignore", invalid = "ignore"):
            along_first = (offset[:, 0] * other_direction[:, 1] - offset[:, 1] * other_direction[:, 0]) / denominator
            along_other = (offset[:, 0] * direction[1] - offset[:, 1] * direction[0]) / denominator
        hits = numpy.nonzero((along_first >= 0) & (along_first < 1) & (along_other >= 0) & (along_other < 1))[0]
        for hit in hits:
            other = others[hit]
            first = lengths[index] + along_first[hit] * RESAMPLE_STEP_CM
            second = lengths[other] + along_other[hit] * RESAMPLE_STEP_CM
            if not any(abs(first - known[0]) < 1.0 and abs(second - known[1]) < 1.0 for known in crossings):
                crossings.append([float(first), float(second)])
    return sorted(crossings)



def rotate(
        points: numpy.ndarray,
        angle_rad: float,
        centre: numpy.ndarray
) -> numpy.ndarray:
    """``points`` turned counter-clockwise (seen from above) by ``angle_rad`` about ``centre``."""

    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    matrix = numpy.array([[cosine, -sine], [sine, cosine]])
    return (numpy.asarray(points, dtype = float) - centre) @ matrix.T + centre



def near_end_distance_cm(
        note: str,
        parameters: Dict = SETUP_VISUALS_PARAMETERS
) -> float:
    """How far before the end cap a near-end start lies: the plan note's distance (the middle of a range), else the default."""

    match = DISTANCE_IN_NOTE_PATTERN.search(note or "")
    if match is None:
        return parameters["near_end_cm"]
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) else low
    return (low + high) / 2



def square_turn_rad(
        turn_to_push_rad: float,
        lay_out: str
) -> float:
    """
    The smallest extra turn that makes the sheet's edges square to the robot after ``turn_to_push_rad``: "portrait"
    puts the long edges along the push direction, "landscape" across it.
    """

    # The paper's long edges run along its own y axis; turned by an angle, they point that angle plus 90 degrees.
    wanted = math.pi / 2 if lay_out == "portrait" else 0.0
    extra = (wanted - turn_to_push_rad) % math.pi
    return extra - math.pi if extra > math.pi / 2 else extra


def square_turn_deg(
        path_id: str,
        line_end: str,
        lay_out: str,
        parameters: Dict = SETUP_VISUALS_PARAMETERS
) -> float:
    """The extra turn in degrees (positive to the LEFT) from the STD lay-out to the square one, for one sheet and end."""

    settings = dict(SETUP_VISUALS_PARAMETERS)
    settings.update(parameters)
    sheet = load_sheet_lines(settings)[RECORDING_PAGE_PARAMETERS["sheets"][path_id]["number"]]
    end = sheet["ends"][settings["end_keys"][line_end]]
    push_angle = math.atan2(WORKSPACE_PARAMETERS["push_direction_xy"][1], WORKSPACE_PARAMETERS["push_direction_xy"][0])
    return round(math.degrees(square_turn_rad(push_angle - math.radians(end["direction_into_line_deg"]), lay_out)), 1)


def compute_setup(
        path_id: str,
        plan_number: str,
        entry: Dict,
        sheet_lines: Dict,
        parameters: Dict = SETUP_VISUALS_PARAMETERS
) -> Dict:
    """
    Where everything lies for one plan entry, in robot coordinates (cm, x away from the robot, y to its left).

    Parameters
    ----------
    path_id : str
        The sheet, as in RECORDING_PAGE_PARAMETERS["sheets"].
    plan_number : str
        For example "02-07".
    entry : dict
        The plan entry from recording_page.expand_episode_plan(): "start_code", "line_end", "note".
    sheet_lines : dict
        From load_sheet_lines().
    parameters : dict, optional
        SETUP_VISUALS_PARAMETERS or a copy with changes.

    Returns
    -------
    dict
        The placed paper and line, the start mark, the start cap, the puck, the point to teach (taught starts only),
        the travel direction at the puck, the turn and offset used, and warnings from the reach check.
    """

    settings = dict(SETUP_VISUALS_PARAMETERS)
    settings.update(parameters)
    start_code = entry["start_code"]
    start = RECORDING_PAGE_PARAMETERS["start_codes"][start_code]
    sheet_number = RECORDING_PAGE_PARAMETERS["sheets"][path_id]["number"]
    sheet = sheet_lines[sheet_number]
    end = sheet["ends"][settings["end_keys"][entry["line_end"]]]
    other_end = sheet["ends"][settings["end_keys"]["B" if entry["line_end"] == "A" else "A"]]

    # The line ordered from the start end; the traced points run from the "first" end to the "last".
    line = sheet["line"] if settings["end_keys"][entry["line_end"]] == "first" else sheet["line"][::-1]

    # Rule A: the ink's beginning on the start mark, the line's first centimetres along the push direction.
    mark = numpy.array(WORKSPACE_PARAMETERS["line_start_xy"]) * 100
    push_angle = math.atan2(WORKSPACE_PARAMETERS["push_direction_xy"][1], WORKSPACE_PARAMETERS["push_direction_xy"][0])
    ink_tip = numpy.array(end["ink_tip_xy_cm"])
    turn_to_push = push_angle - math.radians(end["direction_into_line_deg"])

    def place(points: numpy.ndarray) -> numpy.ndarray:
        return rotate(points, turn_to_push, ink_tip) - ink_tip + mark

    width, height = settings["paper_size_cm"]
    paper = numpy.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    standard_paper = place(paper)
    placed_line = place(line)
    start_cap = place(numpy.array(end["cap_centre_xy_cm"]))
    end_cap = place(numpy.array(other_end["cap_centre_xy_cm"]))

    # A turned start: the sheet turned about the start cap, which stays where it was; a square start turns it until
    # its edges are square to the robot.
    offset_deg = start["offset_deg"]
    if start.get("lay_out"):
        offset_deg = round(math.degrees(square_turn_rad(turn_to_push, start["lay_out"])), 1)
    turned_paper = rotate(standard_paper, math.radians(offset_deg), start_cap)
    placed_line = rotate(placed_line, math.radians(offset_deg), start_cap)
    end_cap = rotate(end_cap, math.radians(offset_deg), start_cap)

    left = numpy.array([-math.sin(push_angle), math.cos(push_angle)])
    forward = numpy.array([math.cos(push_angle), math.sin(push_angle)])
    puck = start_cap + start["offset_cm"] * left
    puck_arc_cm = 0.0
    total_cm = float(arc_lengths(placed_line)[-1])
    spot = None
    if start_code == "NE":
        puck_arc_cm = total_cm - near_end_distance_cm(entry["note"], settings)
        spot = f"{near_end_distance_cm(entry['note'], settings):g} cm before the end cap"
    elif start_code == "MID":
        peaks = curvature_peaks(placed_line, settings["curvature_smoothing_cm"], settings["hairpin_separation_cm"])
        if settings["mid_spots"][path_id] == "tightest":
            puck_arc_cm = peaks[0]["arc_cm"] - settings["sharp_turn_lead_cm"]
            spot = f"{settings['sharp_turn_lead_cm']:g} cm before the sharp turn"
        else:
            hairpins = sorted(peaks[:settings["hairpin_count"]], key = lambda peak: peak["arc_cm"])
            puck_arc_cm = hairpins[len(hairpins) // 2]["arc_cm"]
            spot = "at the middle hairpin"
    elif start_code in ("X1", "X2"):
        passes = sorted(self_crossings(placed_line)[0])
        crossing_arc_cm = passes[0] if start_code == "X1" else passes[1]
        puck_arc_cm = crossing_arc_cm - settings["crossing_lead_cm"]
        spot = f"{settings['crossing_lead_cm']:g} cm before the crossing, {'first' if start_code == 'X1' else 'second'} pass"
    if start_code in TAUGHT_CODES:
        puck = point_at(placed_line, puck_arc_cm)

    ahead = point_at(placed_line, puck_arc_cm + 1.0) - point_at(placed_line, max(puck_arc_cm - 1.0, 0.0))
    travel = ahead / numpy.linalg.norm(ahead)

    # The plan: touch the point, then put the puck's centre half a centimetre further from the robot than the touch.
    teach_point = None
    if start_code in TAUGHT_CODES:
        radial = puck / numpy.linalg.norm(puck)
        teach_point = puck - 0.5 * radial

    return {
        "path_id": path_id, "plan_number": plan_number, "start_code": start_code, "line_end": entry["line_end"],
        "note": entry["note"], "offset_cm": start["offset_cm"], "offset_deg": offset_deg,
        "mark": mark, "forward": forward, "left": left, "standard_paper": standard_paper, "paper": turned_paper,
        "line": placed_line, "start_cap": start_cap, "end_cap": end_cap, "puck": puck, "puck_arc_cm": puck_arc_cm,
        "line_length_cm": total_cm, "travel": travel, "teach_point": teach_point, "spot": spot,
        "warnings": reach_warnings(turned_paper, placed_line, settings)
    }



def is_new_track(path_id: str) -> bool:
    """True for a print of one of the new tracks of 12.11.26 (NEW_TRACK_RECORDING_PARAMETERS), False for a course sheet."""

    return SHAPE_OF_SHEET.get(path_id, "").startswith("T")



def load_new_track_lines(parameters: Dict = NEW_TRACK_RECORDING_PARAMETERS) -> Dict:
    """
    The drawn centre line and the lay-out of every new track, from tracks.json (cm, from the paper's left and bottom edges).

    Returns
    -------
    dict
        Track name ("T03") -> {"line": (N, 2) array resampled every RESAMPLE_STEP_CM, end A first, "lay_out": the
        chosen lay-out (start end, turn), "mark_cm": the lay-out rule's mark in robot coordinates}.
    """

    with open(parameters["tracks_file"], encoding = "utf-8") as file:
        listing = json.load(file)
    mark_cm = numpy.array(listing["lay_out_rule"]["mark_m"]) * 100
    return {track["name"]: {"line": resample_line(numpy.array(track["centre_line_mm"], dtype = float) / 10.0, step_cm = RESAMPLE_STEP_CM),
                            "lay_out": track["lay_out"], "mark_cm": mark_cm}
            for track in listing["tracks"] if track["role"] == "training"}



def compute_new_track_setup(
        path_id: str,
        plan_number: str,
        entry: Dict,
        track_lines: Dict,
        parameters: Dict = SETUP_VISUALS_PARAMETERS
) -> Dict:
    """
    Where everything lies for one plan entry on a new track, in robot coordinates (cm), by the lay-out rule of tracks.json.

    The centre of the start end's cap (the line's first point) lies on the mark and the line's first centimetre points
    along +x, turned by the lay-out's turn about the mark: the same placing as make_test_tracks.layout_check, so the
    picture shows the lay-out that was checked. The returned dictionary has the keys of compute_setup(), plus
    "is_new_track", "mark_label", "corner_distances_cm" (each paper corner's distance from the mark) and "start_tip"
    (where the arm's magnet tip comes down at the start).

    Parameters
    ----------
    path_id : str
        The print, as in RECORDING_PAGE_PARAMETERS["sheets"].
    plan_number : str
        For example "15-01".
    entry : dict
        The plan entry from recording_page.expand_episode_plan(): "start_code", "line_end", "note".
    track_lines : dict
        From load_new_track_lines().
    parameters : dict, optional
        SETUP_VISUALS_PARAMETERS or a copy with changes.

    Returns
    -------
    dict
        As compute_setup(), for the new track.
    """

    settings = dict(SETUP_VISUALS_PARAMETERS)
    settings.update(parameters)
    start_code = entry["start_code"]
    start = RECORDING_PAGE_PARAMETERS["start_codes"][start_code]
    if start_code in ("MID", "X1", "X2") or start.get("lay_out"):
        raise ValueError(f"{plan_number}: the new tracks' pictures know standard, offset, turned and near-end starts only, not {start_code}")
    track = track_lines[SHAPE_OF_SHEET[path_id]]
    if entry["line_end"] != track["lay_out"]["start_end"]:
        raise ValueError(f"{plan_number}: the plan starts from end {entry['line_end']}, the lay-out table from end {track['lay_out']['start_end']}")
    line = track["line"] if entry["line_end"] == "A" else track["line"][::-1]

    # The lay-out rule: the start cap's centre on the mark, the first centimetre of line along +x, turned about the mark.
    mark = track["mark_cm"]
    direction = point_at(line, 1.0) - line[0]
    direction = direction / numpy.linalg.norm(direction)
    lay_out_turn = math.radians(track["lay_out"]["turn_deg"])

    def place(points: numpy.ndarray) -> numpy.ndarray:
        offsets = numpy.atleast_2d(points) - line[0]
        forward = offsets @ direction
        left = direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]
        return numpy.column_stack([mark[0] + forward * math.cos(lay_out_turn) - left * math.sin(lay_out_turn),
                                   mark[1] + forward * math.sin(lay_out_turn) + left * math.cos(lay_out_turn)])

    width, height = settings["paper_size_cm"]
    standard_paper = place(numpy.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]))
    placed_line = place(line)
    start_cap = placed_line[0].copy()
    offset_deg = start["offset_deg"]
    turned_paper = rotate(standard_paper, math.radians(offset_deg), start_cap)
    placed_line = rotate(placed_line, math.radians(offset_deg), start_cap)
    end_cap = placed_line[-1].copy()

    forward_angle = lay_out_turn
    left_axis = numpy.array([-math.sin(forward_angle), math.cos(forward_angle)])
    forward_axis = numpy.array([math.cos(forward_angle), math.sin(forward_angle)])
    puck = start_cap + start["offset_cm"] * left_axis
    puck_arc_cm = 0.0
    total_cm = float(arc_lengths(placed_line)[-1])
    spot = None
    teach_point = None
    if start_code == "NE":
        puck_arc_cm = total_cm - near_end_distance_cm(entry["note"], settings)
        spot = f"{near_end_distance_cm(entry['note'], settings):g} cm before the end cap"
        puck = point_at(placed_line, puck_arc_cm)
        teach_point = puck - 0.5 * puck / numpy.linalg.norm(puck)
    ahead = point_at(placed_line, puck_arc_cm + 1.0) - point_at(placed_line, max(puck_arc_cm - 1.0, 0.0))

    return {
        "path_id": path_id, "plan_number": plan_number, "start_code": start_code, "line_end": entry["line_end"],
        "note": entry["note"], "offset_cm": start["offset_cm"], "offset_deg": offset_deg,
        "mark": mark, "forward": forward_axis, "left": left_axis, "standard_paper": standard_paper, "paper": turned_paper,
        "line": placed_line, "start_cap": start_cap, "end_cap": end_cap, "puck": puck, "puck_arc_cm": puck_arc_cm,
        "line_length_cm": total_cm, "travel": ahead / numpy.linalg.norm(ahead), "teach_point": teach_point, "spot": spot,
        "warnings": reach_warnings(turned_paper, placed_line, settings),
        "is_new_track": True, "mark_label": "NEW-TRACK\nMARK",
        "corner_distances_cm": [float(numpy.linalg.norm(corner - mark)) for corner in turned_paper],
        "start_tip": numpy.array(start_pose()[:2]) * 100
    }



def reach_warnings(
        paper: numpy.ndarray,
        line: numpy.ndarray,
        parameters: Dict = SETUP_VISUALS_PARAMETERS
) -> List[str]:
    """What the drawn layout runs into (calc): paper over the base square, line beyond the reach ring."""

    warnings = []
    half = parameters["base_half_width_cm"]
    edges = numpy.concatenate([numpy.linspace(paper[index], paper[(index + 1) % 4], 60) for index in range(4)])
    base_corners = numpy.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    is_on_base = numpy.any(numpy.all(numpy.abs(edges) < half, axis = 1)) or numpy.any(PolygonPath(paper).contains_points(base_corners))
    if is_on_base:
        warnings.append("calc: a paper corner reaches the drawn base square; check it clears the base")
    farthest = float(numpy.max(numpy.linalg.norm(line, axis = 1)))
    if farthest > parameters["reach_cm"][1]:
        warnings.append(f"calc: the line reaches {farthest:.1f} cm from the base, beyond the {parameters['reach_cm'][1]:g} cm ring")
    return warnings



def short_instruction(setup: Dict) -> str:
    """The one thing that differs from a standard start, short enough for the picture's subtitle."""

    code = setup["start_code"]
    if code == "STD":
        return "Puck centred on the start cap"
    if code in TURN_CODES:
        side = "LEFT" if setup["offset_deg"] > 0 else "RIGHT"
        return f"Sheet turned {abs(setup['offset_deg']):g} deg {side} about the start cap; puck on the cap"
    if code in SQUARE_CODES:
        lay_out = RECORDING_PAGE_PARAMETERS["start_codes"][code]["lay_out"].upper()
        return f"Sheet square to the robot, {lay_out}; puck centred on the start cap"
    if code in TAUGHT_CODES:
        return f"Puck {setup['spot']} (from the plan, not taught)"
    side = "LEFT" if setup["offset_cm"] > 0 else "RIGHT"
    return f"Puck {abs(setup['offset_cm']):g} cm {side} of the start cap"



def corner_name(
        setup: Dict,
        index: int
) -> str:
    """Where paper corner ``index`` lies, seen from behind the robot: "near left", "far right" and so on."""

    corner = setup["paper"][index]
    centre = setup["paper"].mean(axis = 0)
    return f"{'near' if corner[0] < centre[0] else 'far'} {'left' if corner[1] > centre[1] else 'right'}"



def caption_for(setup: Dict) -> str:
    """One or two plain sentences that say what to do for this plan number."""

    code = setup["start_code"]
    sheet_label = RECORDING_PAGE_PARAMETERS["sheets"][setup["path_id"]]["label"]
    lay = f"Sheet {sheet_label}, end {setup['line_end']} as the start: its ink begins on the START MARK, line pointing away from the robot."
    if setup.get("is_new_track"):
        corners = "; ".join(f"{corner_name(setup, index)} {distance:.1f} cm" for index, distance in enumerate(setup["corner_distances_cm"]))
        lay = (f"Sheet {sheet_label}, end {setup['line_end']} as the start: the centre of its round cap on the NEW-TRACK MARK, the first "
               f"straight piece along the direction line (table_marks.pdf), not turned. Paper corners from the mark: {corners}.")
    if code == "STD":
        what = "Puck centred on the start cap."
    elif code in ("OL1", "OR1", "OL1.5", "OR1.5"):
        side = "LEFT" if setup["offset_cm"] > 0 else "RIGHT"
        what = f"Puck {abs(setup['offset_cm']):g} cm {side} of the start cap (seen from behind the robot); back onto the line in the first 5 to 10 cm."
    elif code in TURN_CODES:
        side = "LEFT" if setup["offset_deg"] > 0 else "RIGHT"
        what = f"Then turn the sheet {abs(setup['offset_deg']):g} degrees {side} about the start cap (cap stays on the mark). Puck on the start cap."
    elif code in SQUARE_CODES:
        lay_out = RECORDING_PAGE_PARAMETERS["start_codes"][code]["lay_out"].upper()
        what = (f"Then turn the sheet about the start cap until it lies {lay_out} with its edges square to the robot, as drawn "
                f"(cap stays on the mark). Puck centred on the start cap.")
    else:
        what = (f"Puck on the line {setup['spot']}, about {setup['puck_arc_cm']:.0f} cm along the line from the start (drawn from the plan, "
                f"not taught yet): touch the cross with the magnet tip, puck centre 0.5 cm further out.")
    parts = [lay, what]
    if setup["note"]:
        parts.append(setup["note"])
    parts += [warning[0].upper() + warning[1:] + "." for warning in setup["warnings"]]
    return " ".join(parts)



def draw_setup(
        setup: Dict,
        output_path: Path,
        parameters: Dict = SETUP_VISUALS_PARAMETERS
) -> None:
    """
    Draw one plan entry as a top view in robot coordinates and save it to ``output_path``.

    Parameters
    ----------
    setup : dict
        From compute_setup().
    output_path : Path
        The PNG to write.
    parameters : dict, optional
        SETUP_VISUALS_PARAMETERS or a copy with changes.
    """

    settings = dict(SETUP_VISUALS_PARAMETERS)
    settings.update(parameters)
    figure, axis = pyplot.subplots(figsize = settings["figure_size_inches"])
    code = setup["start_code"]

    # The robot: base square, reach rings, the workspace box the tool guards.
    half = settings["base_half_width_cm"]
    axis.add_patch(patches.Rectangle((-half, -half), 2 * half, 2 * half, facecolor = "#b8b8b8", edgecolor = "#555555"))
    axis.text(0, 0, "ROBOT\nbase", ha = "center", va = "center", fontsize = 9, color = "#333333")
    for radius, style in zip(settings["reach_cm"], (":", "--")):
        axis.add_patch(patches.Circle((0, 0), radius, fill = False, linestyle = style, edgecolor = "#666666", linewidth = 0.8))
    box_x = numpy.array(WORKSPACE_PARAMETERS["box_x"]) * 100
    box_y = numpy.array(WORKSPACE_PARAMETERS["box_y"]) * 100
    axis.add_patch(patches.Rectangle((box_x[0], box_y[0]), box_x[1] - box_x[0], box_y[1] - box_y[0], fill = False,
                                     linestyle = "-.", edgecolor = "tab:green", linewidth = 0.8))

    # The paper: where it lies for a standard start, and turned for TL / TR.
    if code in TURN_CODES:
        axis.add_patch(patches.Polygon(setup["standard_paper"], closed = True, fill = False, linestyle = "--",
                                       edgecolor = "#999999", linewidth = 1.0))
    axis.add_patch(patches.Polygon(setup["paper"], closed = True, facecolor = "#f4f4f4", edgecolor = "#777777", linewidth = 1.2))
    # Sheet 06's line and the new tracks' dashed prints are dashed on the paper; the dash pattern here only says so, it is
    # not the printed one.
    line_style = (0, (2.0, 0.8)) if setup["path_id"].endswith("_dashed") else "-"
    axis.plot(setup["line"][:, 0], setup["line"][:, 1], color = "#222222", linewidth = 4.5, linestyle = line_style, zorder = 3)

    # The line's ends, named as on the sheets figure.
    other = "B" if setup["line_end"] == "A" else "A"
    axis.plot(*setup["start_cap"], marker = "o", markersize = 5, color = "tab:blue", zorder = 4)
    axis.annotate(f"end {setup['line_end']} (start)", setup["start_cap"], xytext = (-10, -30), textcoords = "offset points",
                  fontsize = 8, color = "tab:blue", zorder = 6,
                  bbox = {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    axis.plot(*setup["end_cap"], marker = "o", markersize = 6, color = "tab:purple", zorder = 4)
    axis.annotate(f"end {other} (goal)", setup["end_cap"], xytext = (8, 8), textcoords = "offset points", fontsize = 8,
                  color = "tab:purple", fontweight = "bold", zorder = 6)

    # The start mark taped on the table.
    mark = setup["mark"]
    axis.plot(*mark, marker = "+", markersize = 16, markeredgewidth = 2.2, color = "black", zorder = 5)
    axis.annotate(setup.get("mark_label", "START\nMARK"), mark, xytext = (-46, 10), textcoords = "offset points", fontsize = 8,
                  fontweight = "bold", zorder = 6)
    if setup.get("corner_distances_cm"):
        # A new track: the taped cross of table_marks.pdf (the direction line along +x and the cross line along y through
        # the mark), where the arm's tip comes down, and every paper corner's distance from the mark.
        axis.plot([settings["base_half_width_cm"], settings["view_x_cm"][1]], [mark[1], mark[1]], color = "black", linewidth = 0.9,
                  linestyle = "-.", zorder = 2)
        axis.plot([mark[0], mark[0]], settings["view_y_cm"], color = "black", linewidth = 0.9, linestyle = "-.", zorder = 2)
        axis.plot(*setup["start_tip"], marker = "o", markersize = 4, markerfacecolor = "white", markeredgecolor = "black", zorder = 5)
        axis.annotate("arm's tip\nat the start", setup["start_tip"], xytext = (-44, -26), textcoords = "offset points", fontsize = 7, zorder = 6)
        for corner, distance in zip(setup["paper"], setup["corner_distances_cm"]):
            outward = corner - setup["paper"].mean(axis = 0)
            outward = outward / numpy.linalg.norm(outward)
            axis.annotate(f"{distance:.1f} cm", corner, xytext = tuple(14 * outward), textcoords = "offset points", fontsize = 8,
                          color = "#444444", ha = "center", va = "center", zorder = 8,
                          bbox = {"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})

    # The puck, and for an offset start the cap it is moved away from.
    radius = settings["puck_diameter_cm"] / 2
    axis.add_patch(patches.Circle(setup["puck"], radius, facecolor = "#d62728", edgecolor = "#7a0000", linewidth = 1.2, zorder = 7))
    if setup["offset_cm"]:
        axis.add_patch(patches.Circle(setup["start_cap"], radius, fill = False, linestyle = ":", edgecolor = "#d62728", zorder = 7))
        side = "left" if setup["offset_cm"] > 0 else "right"
        axis.annotate(f"{abs(setup['offset_cm']):g} cm {side}", setup["puck"], xytext = (-30, 22 if setup["offset_cm"] > 0 else -30),
                      textcoords = "offset points", fontsize = 9, color = "#d62728", fontweight = "bold",
                      arrowprops = {"arrowstyle": "-", "color": "#d62728"}, zorder = 8)
    if setup["offset_deg"]:
        # Two rays from the start cap: the line's first direction before the turn (grey) and after it (orange).
        side = "left" if setup["offset_deg"] > 0 else "right"
        ray_cm, arc_cm = 12.0, 8.0
        start_angle = math.degrees(math.atan2(setup["forward"][1], setup["forward"][0]))
        turned_angle = start_angle + setup["offset_deg"]
        for angle, colour, style in ((start_angle, "#777777", "--"), (turned_angle, "tab:orange", "-")):
            tip = setup["start_cap"] + ray_cm * numpy.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            axis.plot(*numpy.column_stack([setup["start_cap"], tip]), color = colour, linestyle = style, linewidth = 1.8, zorder = 6)
        low, high = sorted([start_angle, turned_angle])
        axis.add_patch(patches.Arc(setup["start_cap"], 2 * arc_cm, 2 * arc_cm, theta1 = low, theta2 = high, color = "tab:orange",
                                   linewidth = 2, zorder = 6))
        axis.annotate(f"turned {abs(setup['offset_deg']):g} deg {side}", tip, xytext = (4, 6 if setup["offset_deg"] > 0 else -14),
                      textcoords = "offset points", fontsize = 9, color = "tab:orange", fontweight = "bold", zorder = 8,
                      bbox = {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    if setup["teach_point"] is not None:
        axis.plot(*setup["teach_point"], marker = "x", markersize = 8, markeredgewidth = 2, color = "tab:blue", zorder = 9)

    # Which way the puck goes first.
    arrow_start = setup["puck"] + setup["travel"] * (radius + 0.3)
    axis.annotate("", arrow_start + setup["travel"] * 4.0, arrow_start,
                  arrowprops = {"arrowstyle": "-|>", "color": "#d62728", "linewidth": 1.8}, zorder = 8)

    # Directions as seen from behind the robot.
    view_x, view_y = settings["view_x_cm"], settings["view_y_cm"]
    axis.annotate("", (view_x[0] + 2, view_y[1] - 3), (view_x[0] + 2, view_y[1] - 10),
                  arrowprops = {"arrowstyle": "-|>", "color": "#444444"})
    axis.text(view_x[0] + 3, view_y[1] - 6, "LEFT (seen from\nbehind the robot)", fontsize = 7.5, color = "#444444", va = "center")
    axis.annotate("", (view_x[0] + 10, view_y[0] + 2.5), (view_x[0] + 2, view_y[0] + 2.5),
                  arrowprops = {"arrowstyle": "-|>", "color": "#444444"})
    axis.text(view_x[0] + 11, view_y[0] + 2.5, "away from the robot", fontsize = 7.5, color = "#444444", va = "center")

    sheet_label = RECORDING_PAGE_PARAMETERS["sheets"][setup["path_id"]]["label"]
    title = f"Plan {setup['plan_number']}: sheet {sheet_label}, end {setup['line_end']}, {code}"
    figure.suptitle(title, fontsize = 11, fontweight = "bold")
    axis.set_title(short_instruction(setup), fontsize = 9.5, color = "#d62728" if code != "STD" else "#222222")
    if setup["warnings"]:
        axis.text(0.99, 0.01, "\n".join(setup["warnings"]), transform = axis.transAxes, ha = "right", va = "bottom",
                  fontsize = 7.5, color = "darkred")
    axis.set_xlim(view_x)
    axis.set_ylim(view_y)
    axis.set_aspect("equal")
    axis.grid(alpha = 0.3)
    axis.set_xlabel("x from the base axis (cm)")
    axis.set_ylabel("y (cm), left is up")
    figure.tight_layout()
    figure.savefig(output_path, dpi = settings["dots_per_inch"])
    pyplot.close(figure)



def write_index_page(
        setups: List[Dict],
        output_folder: Path
) -> Path:
    """
    Write index.html beside the pictures: every plan number by sheet, with its picture and caption, and a box to jump to one.

    Returns
    -------
    Path
        The page written.
    """

    sections = []
    for path_id, sheet in RECORDING_PAGE_PARAMETERS["sheets"].items():
        cards = []
        for setup in (setup for setup in setups if setup["path_id"] == path_id):
            number = setup["plan_number"]
            warning_class = " warn" if setup["warnings"] else ""
            cards.append(
                f'<figure id="{number}" class="card{warning_class}"><a href="plan_{number}.png" target="_blank">'
                f'<img loading="lazy" src="plan_{number}.png" alt="Set-up for plan {number}"></a>'
                f'<figcaption><b>{number}</b> &middot; end {setup["line_end"]} &middot; {setup["start_code"]}<br>'
                f'{html.escape(caption_for(setup))}</figcaption></figure>')
        links = " ".join(f'<a href="#{setup["plan_number"]}">{setup["plan_number"][3:]}</a>'
                         for setup in setups if setup["path_id"] == path_id)
        sections.append(f'<section id="sheet-{sheet["number"]}"><h2>Sheet {html.escape(sheet["label"])} '
                        f'<small>({html.escape(sheet["plan"])})</small></h2>'
                        f'<p class="note">{html.escape(RECORDING_PAGE_PARAMETERS["sheet_setups"][path_id])}</p>'
                        f'<p class="jump">{links}</p><div class="grid">{"".join(cards)}</div></section>')

    sheet_links = " ".join(f'<a href="#sheet-{sheet["number"]}">{html.escape(sheet["label"])}</a>'
                           for sheet in RECORDING_PAGE_PARAMETERS["sheets"].values())
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Set-up per episode</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0 16px 40px; background: #fff; color: #222; }}
header {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ddd; z-index: 1; }}
header a {{ margin-right: 10px; }}
input {{ font-size: 16px; width: 7em; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; }}
.card {{ margin: 0; border: 1px solid #ddd; border-radius: 6px; padding: 6px; scroll-margin-top: 70px; }}
.card.warn {{ border-color: #c33; }}
.card:target {{ outline: 3px solid #1f77b4; }}
img {{ width: 100%; height: auto; }}
figcaption {{ font-size: 14px; line-height: 1.35; }}
.note, .key {{ font-size: 14px; color: #444; }}
.jump a {{ margin-right: 6px; font-size: 14px; }}
h2 {{ scroll-margin-top: 70px; }}
</style></head><body>
<header><b>Set-up per episode</b> &nbsp; Go to plan number <input id="go" placeholder="02-07"
onkeydown="if(event.key==='Enter'){{location.hash=this.value.trim();}}"> &nbsp; {sheet_links}</header>
<p class="key">Top view, robot coordinates in cm, seen from above with the robot on the left. Black + = START MARK
(taught {WORKSPACE_PARAMETERS["line_start_xy"][0] * 100:.1f}, {WORKSPACE_PARAMETERS["line_start_xy"][1] * 100:.1f} cm).
Red disc = the puck; red arrow = the way it goes first. Blue dot = the start end's cap, purple = the goal end. Dashed grey
paper = where the sheet lies before a turn. Blue x = the point to touch with the magnet tip (NE, MID, X only). Dotted and
dashed rings: 13 and 44 cm reach (calc); green dash-dot: the tool's workspace box. The lines come from the photo-traced
sheets (about 0.3 cm); NE, MID and X positions are drawn from the plan's words, not taught; a red frame = a reach warning
(calc). Click a picture to see it full size.</p>
<p class="key">New tracks T01 to T16 (sheets 11 to 26 solid, 31 to 46 dashed): black + = the NEW-TRACK MARK, where the centre of
end A's round cap goes; black dash-dot = the taped direction and cross lines (<a href="table_marks.pdf">table_marks.pdf</a>);
white dot = the arm's magnet tip at the start; grey numbers = each paper corner's distance from the mark in cm. The lines
come from the drawn tracks (tracks.json), laid by its lay-out rule.</p>
{"".join(sections)}
</body></html>
"""
    page_path = output_folder / "index.html"
    page_path.write_text(page, encoding = "utf-8")
    return page_path



def all_setups(parameters: Dict = SETUP_VISUALS_PARAMETERS) -> List[Dict]:
    """compute_setup() for every plan number of a course sheet and compute_new_track_setup() for a new track, in plan order."""

    sheet_lines = load_sheet_lines(parameters)
    track_lines = load_new_track_lines()
    setups = []
    for path_id, entries in expand_episode_plan().items():
        for plan_number, entry in entries.items():
            if is_new_track(path_id):
                setups.append(compute_new_track_setup(path_id, plan_number, entry, track_lines, parameters))
            else:
                setups.append(compute_setup(path_id, plan_number, entry, sheet_lines, parameters))
    return setups



def write_table_marks(
        output_path: Path,
        parameters: Dict = SETUP_VISUALS_PARAMETERS
) -> Path:
    """
    Write table_marks.pdf: one A4 page on taping the new-track mark and its two lines onto the table once.

    The mark is the lay-out rule's (tracks.json); the base is the pictures' square ("base_half_width_cm", calc, not
    measured); the arm's tip at the start is the recorder's start pose (config start_pose()); the paper's reach comes
    from the chosen lay-outs in tracks.json (calc).

    Parameters
    ----------
    output_path : Path
        The PDF to write.
    parameters : dict, optional
        SETUP_VISUALS_PARAMETERS or a copy with changes.

    Returns
    -------
    Path
        The page written.
    """

    settings = dict(SETUP_VISUALS_PARAMETERS)
    settings.update(parameters)
    with open(NEW_TRACK_RECORDING_PARAMETERS["tracks_file"], encoding = "utf-8") as file:
        lay_outs = [track["lay_out"] for track in json.load(file)["tracks"] if track["role"] == "training"]
    track_lines = load_new_track_lines()
    mark = next(iter(track_lines.values()))["mark_cm"]
    half = settings["base_half_width_cm"]
    tip = numpy.array(start_pose()[:2]) * 100
    nearest_cm = min(lay_out["paper_near_cm"] for lay_out in lay_outs)
    farthest_cm = max(lay_out["paper_front_cm"] for lay_out in lay_outs)
    most_left_cm = max(lay_out["paper_left_cm"] for lay_out in lay_outs)
    most_right_cm = max(lay_out["paper_right_cm"] for lay_out in lay_outs)

    figure = pyplot.figure(figsize = (8.27, 11.69))
    axis = figure.add_axes([0.08, 0.47, 0.84, 0.45])
    axis.add_patch(patches.Rectangle((-half, -half), 2 * half, 2 * half, facecolor = "#b8b8b8", edgecolor = "#555555"))
    axis.text(0, 0, "ROBOT\nbase", ha = "center", va = "center", fontsize = 10)
    axis.plot([-half - 3, farthest_cm + 4], [0, 0], color = "#999999", linewidth = 0.8, linestyle = ":")
    axis.text(farthest_cm + 4, 0.6, "base axis line", fontsize = 8, color = "#777777", ha = "right")
    axis.plot([half, farthest_cm + 5], [mark[1], mark[1]], color = "black", linewidth = 2.2)
    axis.text(farthest_cm + 5, mark[1] - 1.6, "DIRECTION LINE", fontsize = 9, fontweight = "bold", ha = "right", va = "top")
    axis.plot([mark[0], mark[0]], [-most_right_cm - 3, most_left_cm + 3], color = "black", linewidth = 2.2)
    axis.text(mark[0] + 0.6, most_left_cm + 3, "CROSS LINE", fontsize = 9, fontweight = "bold", va = "top")
    axis.plot(*mark, marker = "+", markersize = 22, markeredgewidth = 2.5, color = "tab:red", zorder = 5)
    axis.annotate("NEW-TRACK MARK\n(centre of end A's cap)", mark, xytext = (14, 18), textcoords = "offset points", fontsize = 9,
                  color = "tab:red", fontweight = "bold")
    axis.plot(*tip, marker = "o", markersize = 6, markerfacecolor = "white", markeredgecolor = "black", zorder = 5)
    axis.annotate("arm's magnet tip\nat the start", tip, xytext = (-30, -36), textcoords = "offset points", fontsize = 8)
    example = compute_new_track_setup(*next((path_id, number, entry) for path_id, entries in expand_episode_plan().items()
                                            if is_new_track(path_id) for number, entry in entries.items()), track_lines)
    axis.add_patch(patches.Polygon(example["paper"], closed = True, fill = False, linestyle = "--", edgecolor = "#888888"))
    axis.plot(example["line"][:, 0], example["line"][:, 1], color = "#555555", linewidth = 3, alpha = 0.5)
    axis.text(example["paper"][:, 0].max() - 1, example["paper"][:, 1].min() + 1, "example: T01 laid by the rule", fontsize = 8,
              color = "#666666", ha = "right")
    axis.annotate("", (mark[0], -most_right_cm - 1), (half, -most_right_cm - 1), arrowprops = {"arrowstyle": "<->", "color": "tab:blue"})
    axis.text((mark[0] + half) / 2, -most_right_cm - 0.5, f"{mark[0] - half:.1f} cm", color = "tab:blue", ha = "center", fontsize = 9)
    axis.annotate("", (mark[0], mark[1] - 9), (tip[0], mark[1] - 9), arrowprops = {"arrowstyle": "<->", "color": "tab:blue"})
    axis.text((mark[0] + tip[0]) / 2, mark[1] - 11, f"about {round(2 * (mark[0] - tip[0])) / 2:g} cm", color = "tab:blue", ha = "center", fontsize = 9)
    axis.set_xlim(-half - 4, farthest_cm + 8)
    axis.set_ylim(-most_right_cm - 5, most_left_cm + 5)
    axis.set_aspect("equal")
    axis.grid(alpha = 0.25)
    axis.set_xlabel("cm in front of the base axis (away from the robot)")
    axis.set_ylabel("cm, left is up (seen from behind the robot)")
    figure.suptitle("Table marks for the new tracks T01 to T16 (tape once, keep all day)", fontsize = 13, fontweight = "bold", y = 0.965)

    steps = [
        f"1. DIRECTION LINE: a straight tape line on the table, pointing straight away from the robot, {abs(mark[1]):.1f} cm to the RIGHT of "
        f"the middle of the base's front edge (seen from behind the robot), from the base's front edge to about {farthest_cm + 5:.0f} cm out.",
        f"2. CROSS LINE: a straight tape line square to the direction line, crossing it {mark[0]:.1f} cm in front of the base axis, which is "
        f"{mark[0] - half:.1f} cm in front of the base's front edge (calc: the base drawn as a {2 * half:.1f} cm square around its axis; "
        f"Niryo's sheet gives a footprint of 18.2 x 18.8 cm, so measure your base's front edge once). At least {max(most_left_cm, most_right_cm) + 3:.0f} cm to each side.",
        "3. Where the two lines cross is the NEW-TRACK MARK. Use dark tape or a pen line on light tape, so it shows beside the paper. "
        "It is not the course sheets' START MARK: lay the course sheets as before.",
        f"4. Check, only with the robot running and Emranul's go: when the arm comes down at the start of an episode, its magnet tip "
        f"stands on the direction line about {round(2 * (mark[0] - tip[0])) / 2:g} cm before the mark (calc from config.py: "
        f"{tip[0]:.2f}, {tip[1]:.2f} cm; measured in the 148 start-mark recordings: 14.0, -1.5 cm). Do not touch the arm while its "
        f"motors are on.",
        "5. Every new sheet: the centre of end A's round cap (A is written on the back) over the mark, the first straight piece of the "
        "line along the direction line (the tape shows at the paper's near and far edges, the cross line at both sides), the sheet not "
        "turned. Check one or two corner distances of the set-up picture with a ruler, then tape the corners down.",
        f"Sizes, calc from the lay-out check (tracks.json): the paper's near edge {nearest_cm:.1f} cm or more in front of the base axis, "
        f"its far edge at most {farthest_cm:.1f} cm; it reaches at most {most_left_cm:.1f} cm to the left and {most_right_cm:.1f} cm to "
        f"the right of the base axis line. Keep the table clear that far."
    ]
    text_axis = figure.add_axes([0.08, 0.03, 0.84, 0.40])
    text_axis.axis("off")
    wrapped = "\n\n".join(textwrap.fill(step, 105) for step in steps)
    text_axis.text(0.0, 1.0, wrapped, va = "top", ha = "left", fontsize = 9.5, family = "sans-serif")
    figure.savefig(output_path)
    pyplot.close(figure)
    return output_path



def main() -> None:

    parser = argparse.ArgumentParser(description = "Draw one set-up picture per plan number and a page that lists them.")
    parser.add_argument("--plan", help = "only this plan number, for example 02-07 (the page is not rewritten)")
    parser.add_argument("--new-tracks", action = "store_true",
                        help = "only the new tracks' pictures and table_marks.pdf; the course sheets' pictures stay as they are")
    arguments = parser.parse_args()

    output_folder = PROJECT_FOLDER / SETUP_VISUALS_PARAMETERS["output_folder"]
    output_folder.mkdir(parents = True, exist_ok = True)
    setups = all_setups()
    chosen = [setup for setup in setups if arguments.plan in (None, setup["plan_number"])
              and (not arguments.new_tracks or setup.get("is_new_track"))]
    if arguments.new_tracks:
        print(f"Table marks: {write_table_marks(output_folder / 'table_marks.pdf')}")
    if not chosen:
        raise SystemExit(f"No plan number {arguments.plan} in the episode plan.")
    for setup in chosen:
        draw_setup(setup, output_folder / f"plan_{setup['plan_number']}.png")
    print(f"{len(chosen)} pictures in {output_folder}")
    if arguments.plan is None:
        print(f"Page: {write_index_page(setups, output_folder)}")
    for setup in chosen:
        if setup["warnings"]:
            print(f"  {setup['plan_number']}: {'; '.join(setup['warnings'])}")


if __name__ == "__main__":
    main()
