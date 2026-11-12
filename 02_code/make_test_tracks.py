"""
Make printable tracks for the robot: lines the models have never seen, drawn by a fixed random rule, as true-size A4 pages.

Every page is an A4 PDF at true size (print at 100 %, "actual size", no fitting to the page) with its SVG beside it, the
line 10 mm wide like the course sheets: solid with round ends, or dashed with square-ended dashes like sheet 06. The rule
is matched to the six course sheets, measured from their traced centre lines: straight pieces, gentle, medium and sharp
bends, corners down to about 6 mm radius, hairpins and loops that cross their own line once. Each track is built from a
plan of pieces with its own fixed seed (TEST_TRACK_PARAMETERS), and tracks.json is written from the same objects as the
drawings. Every track is checked for the lay-out at the recording (end A's cap on the mark 19.5 cm in front of the base
axis, the line leaving straight away from the robot) with the arm model and the driver's guards (lay_outs.csv).

The face of a track page carries the line only (the two circle pages also their grey circles): the forearm camera sees
the paper round the puck, and the course sheets carry nothing else. Names, the start end and the print order are on
the list pages of the two print files, print_set_saturday.pdf and print_set_tests.pdf (order in *_order.json).

Run from 02_code:
    python make_test_tracks.py                the course sheets' numbers, every set and both print files
    python make_test_tracks.py pilot          T01 to T04 only, and pilot_ready.txt
    python make_test_tracks.py training       T01 to T16 and print_set_saturday.pdf
    python make_test_tracks.py development    D01 to D10
    python make_test_tracks.py diagnostic     the course-shape pages (print_set_tests.pdf comes with "all")
    python make_test_tracks.py old            the set of 11.11.26 (into test_tracks/)
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz
import matplotlib
matplotlib.use("Agg")
import numpy
from matplotlib import pyplot
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from scipy.spatial import cKDTree

from config import (MODEL_MODE_PARAMETERS, PUCK_PARAMETERS, PUSHER_PARAMETERS, RECORDING_PARAMETERS, SETUP_VISUALS_PARAMETERS,
                    TEST_TRACK_PARAMETERS, pushing_height)
from record_episodes import forearm_camera_heading_rad
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed


POINTS_PER_MM = 72.0 / 25.4
# the keys of TEST_TRACK_PARAMETERS added on 12.11.26; the set of 11.11.26 lists only the others, as it did
NEW_TRACK_KEYS = ("new_output_folder", "drawing_rule", "lay_out", "page_marks", "training", "development", "diagnostic")


def smallest_radius_mm(curve: numpy.ndarray) -> float:
    """The tightest bend of ``curve``: the smallest radius of the circle through three points 5 mm apart."""

    step = 5
    first, middle, last = curve[:-2 * step], curve[step:-step], curve[2 * step:]
    sides = [numpy.linalg.norm(middle - first, axis = 1), numpy.linalg.norm(last - middle, axis = 1), numpy.linalg.norm(last - first, axis = 1)]
    twice_area = numpy.abs(numpy.cross(middle - first, last - first))
    radii = sides[0] * sides[1] * sides[2] / numpy.maximum(2 * twice_area, 1e-12)
    return float(radii.min()) if len(radii) else math.inf



def crossings(curve: numpy.ndarray) -> List[Dict]:
    """
    Where ``curve`` crosses itself.

    Returns
    -------
    list of dict
        One per crossing: "positions" (the two sample indices) and "angle_deg" (0 to 90, 90 = at right angles).
    """

    found = []
    starts, ends = curve[:-1], curve[1:]
    directions = ends - starts
    for index in range(0, len(starts) - 20):
        later = numpy.arange(index + 20, len(starts))
        offset = starts[later] - starts[index]
        denominator = numpy.cross(directions[index], directions[later])
        with numpy.errstate(divide = "ignore", invalid = "ignore"):
            along_first = numpy.cross(offset, directions[later]) / denominator
            along_second = numpy.cross(offset, directions[index]) / denominator
        hits = later[(numpy.abs(denominator) > 1e-12) & (along_first >= 0) & (along_first <= 1) & (along_second >= 0) & (along_second <= 1)]
        for other in hits:
            cosine = abs(directions[index] @ directions[other]) / (numpy.linalg.norm(directions[index]) * numpy.linalg.norm(directions[other]))
            found.append({"positions": (int(index), int(other)), "angle_deg": math.degrees(math.acos(min(1.0, cosine)))})
    return found



def close_approaches(
        curve: numpy.ndarray,
        smallest_gap_mm: float,
        crossing_positions: List[tuple]
) -> int:
    """
    How many sample pairs of ``curve`` come closer than ``smallest_gap_mm`` although they lie more than twice that apart
    along the line, leaving out the stretches around the crossings.
    """

    near_crossing = numpy.zeros(len(curve), dtype = bool)
    reach = int(2.5 * smallest_gap_mm)
    for first, second in crossing_positions:
        for position in (first, second):
            near_crossing[max(0, position - reach):position + reach] = True
    sampled = numpy.arange(0, len(curve), 3)
    points = curve[sampled]
    distances = numpy.linalg.norm(points[:, None, :] - points[None, :, :], axis = 2)
    apart_along = numpy.abs(sampled[:, None] - sampled[None, :]) > 2 * smallest_gap_mm
    is_free = ~near_crossing[sampled][:, None] & ~near_crossing[sampled][None, :]
    return int(numpy.sum((distances < smallest_gap_mm) & apart_along & is_free) // 2)



def drawn_track(
        plan: Dict,
        randomness_generator: numpy.random.Generator,
        parameters: Dict = TEST_TRACK_PARAMETERS
) -> Optional[Dict]:
    """
    Draw random curves until one keeps every rule and matches ``plan`` ("is_dashed", "has_crossing", "is_end_near").

    A curve is driven like a car: from a start near the bottom edge, heading roughly up the page, it runs through pieces
    of constant bend (straight, or a circle no tighter than "smallest_radius_mm"), each "piece_mm" long, until it
    reaches a length drawn within "length_mm".

    Returns
    -------
    dict or None
        "curve" (samples in mm, x right, y down, the start first), "length_mm", "smallest_radius_mm", "crossing_deg"
        and "end_from_edge_mm"; None when no attempt succeeded.
    """

    width, height = parameters["page_mm"]
    margin = parameters["edge_margin_mm"]
    tightest = 1.0 / parameters["smallest_radius_mm"]
    for _ in range(parameters["attempts_per_track"]):
        position = numpy.array([randomness_generator.uniform(margin + 30, width - margin - 30),
                                randomness_generator.uniform(height - parameters["start_from_bottom_mm"], height - margin - 5)])
        heading = math.radians(-90.0 + randomness_generator.uniform(-35.0, 35.0))
        target_length = randomness_generator.uniform(*parameters["length_mm"])
        points = [position.copy()]
        is_inside = True
        last_side = randomness_generator.choice([-1.0, 1.0])
        while len(points) - 1 < target_length and is_inside:
            piece = randomness_generator.uniform(*parameters["piece_mm"])
            # bends that change side make S-curves and meanders, bends to the same side make loops
            side = -last_side if randomness_generator.random() < parameters["side_change_share"] else last_side
            bend = 0.0 if randomness_generator.random() < parameters["straight_share"] else side * randomness_generator.uniform(0.25 * tightest, tightest)
            last_side = side if bend else last_side
            for _ in range(int(piece)):
                heading += bend
                position = position + numpy.array([math.cos(heading), math.sin(heading)])
                points.append(position.copy())
                if not (margin <= position[0] <= width - margin and margin <= position[1] <= height - margin):
                    is_inside = False
                    break
        if not is_inside:
            continue
        curve = numpy.array(points)
        length = len(curve) - 1.0
        found = crossings(curve)
        if len(found) != int(plan["has_crossing"]):
            continue
        if found and found[0]["angle_deg"] < parameters["smallest_crossing_deg"]:
            continue
        if close_approaches(curve, parameters["smallest_gap_mm"], [crossing["positions"] for crossing in found]):
            continue
        end = curve[-1]
        end_from_edge = float(min(end[0], width - end[0], end[1], height - end[1]))
        is_near = end_from_edge < parameters["near_end_mm"]
        is_far = end_from_edge > parameters["far_end_mm"]
        if (plan["is_end_near"] and not is_near) or (not plan["is_end_near"] and not is_far):
            continue
        return {"curve": curve, "length_mm": round(length, 1), "smallest_radius_mm": round(smallest_radius_mm(curve), 1),
                "crossing_deg": round(found[0]["angle_deg"], 1) if found else None, "end_from_edge_mm": round(end_from_edge, 1)}
    return None



def track_svg(
        curve: numpy.ndarray,
        is_dashed: bool,
        name: str,
        parameters: Dict = TEST_TRACK_PARAMETERS
) -> str:
    """An A4 SVG at true size: the line, and the track's name small and light grey in the top corner."""

    width, height = parameters["page_mm"]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in curve[::2]) + f" L {curve[-1][0]:.1f} {curve[-1][1]:.1f}"
    stroke = (f'stroke-dasharray="{parameters["dash_mm"]} {parameters["gap_mm"]}" stroke-linecap="butt"' if is_dashed
              else 'stroke-linecap="round"')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">\n'
            f'  <rect width="{width}" height="{height}" fill="white"/>\n'
            f'  <path d="{path}" fill="none" stroke="black" stroke-width="{parameters["line_width_mm"]}" stroke-linejoin="round" {stroke}/>\n'
            f'  <text x="4" y="6" font-family="Arial" font-size="3" fill="#bbbbbb">{name}: print at 100 %, the line must measure 10 mm</text>\n'
            f'</svg>\n')



def feature_plans(count: int) -> List[Dict]:
    """``count`` feature plans that cover dashed or solid, a crossing or none, and a near or far end evenly."""

    combinations = [{"is_dashed": dashed, "has_crossing": crossing, "is_end_near": near}
                    for crossing in (False, True) for near in (False, True) for dashed in (False, True)]
    return [dict(combinations[index % len(combinations)]) for index in range(count)]



def main_old_sets() -> None:
    """Draw the set of 11.11.26 (practice, validation, test) into "output_folder", as on 11.11.26."""

    parameters = TEST_TRACK_PARAMETERS
    randomness_generator = numpy.random.default_rng(parameters["seed"])
    folder = Path(parameters["output_folder"])
    listing = []
    for set_name, count in parameters["sets"].items():
        (folder / set_name).mkdir(parents = True, exist_ok = True)
        plans = feature_plans(count)
        randomness_generator.shuffle(plans)
        for number, plan in enumerate(plans, start = 1):
            name = f"{set_name}_{number:02d}"
            track = drawn_track(plan, randomness_generator, parameters)
            if track is None:
                print(f"{name}: no track kept the rules for {plan}; skipped")
                continue
            (folder / set_name / f"{name}.svg").write_text(track_svg(track["curve"], plan["is_dashed"], name, parameters), encoding = "utf-8")
            entry = {"name": name, "set": set_name, **plan, **{key: value for key, value in track.items() if key != "curve"},
                     "start_mm": [round(float(value), 1) for value in track["curve"][0]], "end_mm": [round(float(value), 1) for value in track["curve"][-1]]}
            listing.append(entry)
            print(f"{name}: {entry['length_mm'] / 10:.1f} cm, {'dashed' if plan['is_dashed'] else 'solid'}, "
                  f"{'one crossing' if plan['has_crossing'] else 'no crossing'}, end {entry['end_from_edge_mm'] / 10:.1f} cm from the edge, "
                  f"tightest bend {entry['smallest_radius_mm'] / 10:.1f} cm")
    (folder / "tracks.json").write_text(json.dumps({"rule": {key: (str(value) if isinstance(value, Path) else value) for key, value in parameters.items() if key not in NEW_TRACK_KEYS},
                                                    "tracks": listing}, indent = 1), encoding = "utf-8")



# ----------------------------------------------------------------------------------------------------------------------
# The tracks of 12.11.26: measuring a centre line
# ----------------------------------------------------------------------------------------------------------------------

def arc_lengths_mm(curve: numpy.ndarray) -> numpy.ndarray:
    """The distance along ``curve`` from its first point to every point, in millimetres."""

    return numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(curve, axis = 0), axis = 1))]



def resampled_curve(
        curve: numpy.ndarray,
        spacing_mm: float = 1.0
) -> numpy.ndarray:
    """``curve`` with one point every ``spacing_mm`` along it, its last point kept."""

    along = arc_lengths_mm(curve)
    wanted = numpy.arange(0.0, along[-1], spacing_mm)
    if along[-1] - wanted[-1] > 1e-6:
        wanted = numpy.r_[wanted, along[-1]]
    return numpy.column_stack([numpy.interp(wanted, along, curve[:, 0]), numpy.interp(wanted, along, curve[:, 1])])



def three_point_radii_mm(
        line: numpy.ndarray,
        step: int = 5
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    The radius of the circle through the points ``step`` samples before and after every sample of ``line``.

    With one sample per millimetre the circle spans 1 cm of line, the measure of smallest_radius_mm. The first and last
    ``step`` samples get an infinite radius.

    Returns
    -------
    tuple of numpy.ndarray
        The radii in millimetres and the turning side (+1 left, -1 right, 0 straight) per sample.
    """

    radii = numpy.full(len(line), numpy.inf)
    sides = numpy.zeros(len(line))
    if len(line) <= 2 * step:
        return radii, sides
    first, middle, last = line[:-2 * step], line[step:-step], line[2 * step:]
    to_middle, to_last = middle - first, last - middle
    turning = to_middle[:, 0] * to_last[:, 1] - to_middle[:, 1] * to_last[:, 0]
    chord = last - first
    twice_area = numpy.abs(to_middle[:, 0] * chord[:, 1] - to_middle[:, 1] * chord[:, 0])
    product = numpy.linalg.norm(to_middle, axis = 1) * numpy.linalg.norm(to_last, axis = 1) * numpy.linalg.norm(chord, axis = 1)
    with numpy.errstate(divide = "ignore"):
        radii[step:-step] = numpy.where(twice_area > 1e-9, product / numpy.maximum(2.0 * twice_area, 1e-12), numpy.inf)
    sides[step:-step] = numpy.sign(turning)
    return radii, sides



def segment_headings_rad(line: numpy.ndarray) -> numpy.ndarray:
    """The direction of every segment of ``line``, unwrapped so that a turn adds up."""

    steps = numpy.diff(line, axis = 0)
    return numpy.unwrap(numpy.arctan2(steps[:, 1], steps[:, 0]))



def measured_bends(
        curve: numpy.ndarray,
        rule: Optional[Dict] = None
) -> List[Dict]:
    """
    The bends of ``curve`` as measured from the line itself.

    A bend is a stretch that bends one way with a radius below "bend_radius_below_mm" (three points 5 mm apart). Its
    class follows its smallest radius ("radius_classes_mm"), or it is a hairpin when that radius is at most
    "hairpin_radius_most_mm" and the line turns at least "hairpin_turn_deg" within "hairpin_window_mm" around the
    tightest point. Bends turning less than 10 degrees are left out (tracing jitter on the course sheets).

    Returns
    -------
    list of dict
        Per bend: "class", "side" ("left" or "right"), "radius_mm", "turn_deg", "at_mm" (the tightest point, along the
        line from end A), "from_mm" and "to_mm".
    """

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    line = resampled_curve(curve, 1.0)
    radii, sides = three_point_radii_mm(line)
    headings = segment_headings_rad(line)
    last_segment = len(headings) - 1
    is_bent = radii < settings["bend_radius_below_mm"]
    bends = []
    index = 0
    while index < len(line):
        if not is_bent[index]:
            index += 1
            continue
        first = index
        while index + 1 < len(line) and is_bent[index + 1] and sides[index + 1] == sides[first]:
            index += 1
        last = index
        index += 1
        turn = math.degrees(headings[min(last, last_segment)] - headings[max(first - 1, 0)])
        if abs(turn) < 10.0:
            continue
        smallest = float(radii[first:last + 1].min())
        tightest_samples = first + numpy.nonzero(radii[first:last + 1] <= smallest * 1.02 + 1e-9)[0]
        tightest = int(round(float(tightest_samples.mean())))
        half_window = int(round(settings["hairpin_window_mm"] / 2))
        window_turn = abs(math.degrees(headings[min(tightest + half_window, last_segment)] - headings[max(tightest - half_window, 0)]))
        classes = settings["radius_classes_mm"]
        if smallest <= settings["hairpin_radius_most_mm"] and abs(turn) >= settings["hairpin_turn_deg"]:
            name = "hairpin"
        elif smallest < classes["corner"]:
            name = "corner"
        elif smallest < classes["sharp"]:
            name = "sharp"
        elif smallest < classes["medium"]:
            name = "medium"
        else:
            name = "gentle"
        bends.append({"class": name, "side": "left" if sides[first] > 0 else "right", "radius_mm": round(smallest, 1),
                      "turn_deg": round(abs(turn), 1), "tight_turn_deg": round(window_turn, 1), "at_mm": float(tightest),
                      "from_mm": float(first), "to_mm": float(last)})
    return bends



def measured_crossings(curve: numpy.ndarray) -> List[Dict]:
    """
    Where ``curve`` crosses itself, with the loop each crossing closes.

    Returns
    -------
    list of dict
        Per crossing: "angle_deg" (0 to 90), "point_mm", "first_mm" and "second_mm" (along the line), "loop_length_mm",
        "loop_size_mm" (the diameter of a circle of the loop's area), "loop_extent_mm" (its largest width) and
        "loop_side" ("left" when the loop is run anticlockwise).
    """

    line = resampled_curve(curve, 1.0)
    found = []
    for crossing in crossings(line):
        first, second = crossing["positions"]
        loop = line[first:second + 2]
        twice_area = float(numpy.sum(loop[:-1, 0] * loop[1:, 1] - loop[1:, 0] * loop[:-1, 1]) + loop[-1, 0] * loop[0, 1] - loop[0, 0] * loop[-1, 1])
        distances = numpy.linalg.norm(loop[::2, None, :] - loop[None, ::2, :], axis = 2)
        found.append({"angle_deg": round(crossing["angle_deg"], 1), "point_mm": [round(float(value), 1) for value in (line[first] + line[second]) / 2],
                      "first_mm": float(first), "second_mm": float(second), "loop_length_mm": float(second - first),
                      "loop_size_mm": round(math.sqrt(4.0 * abs(twice_area) / 2.0 / math.pi), 1),
                      "loop_extent_mm": round(float(distances.max()), 1), "loop_side": "left" if twice_area > 0 else "right"})
    return found



def smallest_self_distance_mm(
        curve: numpy.ndarray,
        found_crossings: List[Dict],
        rule: Optional[Dict] = None
) -> float:
    """
    The smallest distance between two parts of ``curve`` that lie more than "gap_apart_along_mm" apart along it.

    At a crossing the pairs of one point on each of its two passes, each within "smallest_gap_mm" / sin(crossing angle) +
    "crossing_clear_mm" along the line of where that pass crosses, are left out, because there the two passes must come
    together; every other part of the line, its ends included, keeps the distance from both passes.
    """

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    line = resampled_curve(curve, 2.0)
    along = arc_lengths_mm(line)
    distances = numpy.linalg.norm(line[:, None, :] - line[None, :, :], axis = 2)
    is_counted = numpy.abs(along[:, None] - along[None, :]) > settings["gap_apart_along_mm"]
    for crossing in found_crossings:
        zone = settings["smallest_gap_mm"] / math.sin(math.radians(max(crossing["angle_deg"], 1.0))) + settings["crossing_clear_mm"]
        is_first_pass = numpy.abs(along - crossing["first_mm"]) < zone
        is_second_pass = numpy.abs(along - crossing["second_mm"]) < zone
        is_counted &= ~((is_first_pass[:, None] & is_second_pass[None, :]) | (is_second_pass[:, None] & is_first_pass[None, :]))
    return float(distances[is_counted].min()) if is_counted.any() else math.inf



def end_distances_mm(
        point: numpy.ndarray,
        page_mm: List[float]
) -> Dict[str, float]:
    """How far ``point`` lies from each paper edge (x from the left edge, y from the bottom edge)."""

    width, height = page_mm
    return {"left": round(float(point[0]), 1), "right": round(float(width - point[0]), 1),
            "bottom": round(float(point[1]), 1), "top": round(float(height - point[1]), 1)}



def line_summary(
        curve: numpy.ndarray,
        rule: Optional[Dict] = None
) -> Dict:
    """Everything measured from a centre line: length, tightest radius, bends, crossings, self-distance, the ends."""

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    line = resampled_curve(curve, 1.0)
    found = measured_crossings(curve)
    bends = measured_bends(curve, settings)
    return {"length_mm": round(float(arc_lengths_mm(curve)[-1]), 1),
            "tightest_radius_mm": round(smallest_radius_mm(line), 1),
            "bends": bends,
            "hairpins": sum(bend["class"] == "hairpin" for bend in bends),
            "crossings": found,
            "smallest_self_distance_mm": round(smallest_self_distance_mm(curve, found, settings), 1),
            "end_a_from_edges_mm": end_distances_mm(curve[0], settings["page_mm"]),
            "end_b_from_edges_mm": end_distances_mm(curve[-1], settings["page_mm"]),
            "nearest_edge_to_line_mm": round(float(min(line[:, 0].min(), settings["page_mm"][0] - line[:, 0].max(),
                                                       line[:, 1].min(), settings["page_mm"][1] - line[:, 1].max())), 1)}



def feature_names(summary: Dict) -> List[str]:
    """Short words for the features of a measured line, for the list and the report."""

    names = []
    for bend in summary["bends"]:
        if bend["class"] == "hairpin":
            names.append(f"hairpin {bend['radius_mm']:.0f} mm {bend['turn_deg']:.0f} deg {bend['side']}")
        elif bend["class"] in ("corner", "sharp", "medium"):
            names.append(f"{bend['class']} {bend['radius_mm']:.0f} mm {bend['turn_deg']:.0f} deg {bend['side']}")
    gentle = [bend for bend in summary["bends"] if bend["class"] == "gentle"]
    if gentle:
        names.append(f"{len(gentle)} gentle bend{'s' if len(gentle) > 1 else ''}")
    for crossing in summary["crossings"]:
        names.append(f"loop {crossing['loop_side']} {crossing['loop_size_mm']:.0f} mm, crossing {crossing['angle_deg']:.0f} deg")
    return names



def compact_features(summary: Dict) -> str:
    """The features of a measured line in a few words: how many bends of each class, the tightest, the loops."""

    parts = []
    for name in ("hairpin", "corner", "sharp", "medium", "gentle"):
        bends = [bend for bend in summary["bends"] if bend["class"] == name]
        if bends:
            tightest = min(bend["radius_mm"] for bend in bends)
            parts.append(f"{len(bends)} {name} ({tightest:.0f} mm)" if name != "gentle" else f"{len(bends)} gentle")
    for crossing in summary["crossings"]:
        parts.append(f"loop {crossing['loop_side']}, crossing {crossing['angle_deg']:.0f} deg")
    return ", ".join(parts) or "straight"



# ----------------------------------------------------------------------------------------------------------------------
# The six course sheets
# ----------------------------------------------------------------------------------------------------------------------

def course_sheet_lines(path: Optional[Path] = None) -> Dict[str, numpy.ndarray]:
    """
    The six course sheets' traced centre lines in millimetres on the paper, end A first.

    The traces (2026-11-06_sheet_measurements.json) have end A near their top edge; they are turned half a turn, so
    that end A lies near the bottom edge as on the new tracks: x' = 210 - 10 x and y' = 297 - 10 y.
    """

    path = Path(path or TEST_TRACK_PARAMETERS["diagnostic"]["sheet_measurements"])
    sheets = json.loads(path.read_text(encoding = "utf-8"))
    width, height = TEST_TRACK_PARAMETERS["drawing_rule"]["page_mm"]
    lines = {}
    for sheet, entry in sheets.items():
        points = 10.0 * numpy.array(entry["line_xy_cm"], dtype = float)
        lines[sheet] = numpy.column_stack([width - points[:, 0], height - points[:, 1]])
    return lines



def course_sheet_measurements(path: Optional[Path] = None) -> List[Dict]:
    """The numbers of the six course sheets, measured from their traced centre lines with the same functions as the new tracks."""

    rows = []
    for sheet, line in course_sheet_lines(path).items():
        summary = line_summary(line)
        smoothed = line_summary(running_mean(resampled_curve(line, 1.0), 7))
        rows.append({"sheet": sheet, **summary, "tightest_radius_smoothed_mm": smoothed["tightest_radius_mm"],
                     "bends_smoothed": smoothed["bends"], "features": feature_names(summary)})
    return rows



def running_mean(
        line: numpy.ndarray,
        window: int
) -> numpy.ndarray:
    """A running mean over ``window`` samples, the ends held: the photo traces jitter at the 0.1 mm level."""

    kernel = numpy.ones(window) / window
    padded = numpy.pad(line, ((window // 2, window // 2), (0, 0)), mode = "edge")
    return numpy.column_stack([numpy.convolve(padded[:, 0], kernel, mode = "valid"), numpy.convolve(padded[:, 1], kernel, mode = "valid")])



# ----------------------------------------------------------------------------------------------------------------------
# Building a track from pieces
# ----------------------------------------------------------------------------------------------------------------------

def piece_points(
        start_xy: numpy.ndarray,
        start_heading_rad: float,
        length_mm: float,
        curvature_per_mm: float,
        step_mm: float
) -> Tuple[numpy.ndarray, float]:
    """The points of one piece of constant bend (exact arc or straight), the start left out, and the heading at its end."""

    count = max(1, int(math.ceil(length_mm / step_mm)))
    along = numpy.linspace(0.0, length_mm, count + 1)[1:]
    if abs(curvature_per_mm) < 1e-12:
        points = start_xy + numpy.column_stack([along * math.cos(start_heading_rad), along * math.sin(start_heading_rad)])
        return points, start_heading_rad
    headings = start_heading_rad + curvature_per_mm * along
    points = start_xy + numpy.column_stack([(numpy.sin(headings) - math.sin(start_heading_rad)) / curvature_per_mm,
                                            -(numpy.cos(headings) - math.cos(start_heading_rad)) / curvature_per_mm])
    return points, start_heading_rad + curvature_per_mm * length_mm



def planned_item(item) -> Dict:
    """A plan entry as a dictionary: a bare word becomes {"kind": word}."""

    return {"kind": item} if isinstance(item, str) else dict(item)



def feature_pieces(
        item: Dict,
        side: float,
        randomness_generator: numpy.random.Generator,
        rule: Dict
) -> List[Dict]:
    """
    The pieces of constant bend for one planned feature, its numbers drawn within the rule.

    Parameters
    ----------
    item : dict
        "kind" (a key of the rule's "pieces") and optional overrides such as "radius_mm".
    side : float
        +1 to bend left (anticlockwise), -1 to bend right.

    Returns
    -------
    list of dict
        Per piece: "kind", "length_mm" and "curvature_per_mm" (signed, 0 for a straight piece).
    """

    settings = dict(rule["pieces"][item["kind"]])
    settings.update({key: value for key, value in item.items() if key not in ("kind", "side")})
    uniform = randomness_generator.uniform
    if item["kind"] == "straight":
        return [{"kind": "straight", "length_mm": uniform(*settings["length_mm"]), "curvature_per_mm": 0.0}]
    radius = uniform(*settings["radius_mm"])
    turn = math.radians(uniform(*settings["turn_deg"]))
    if item["kind"] == "hairpin":
        shoulder_turns = [math.radians(uniform(*settings["shoulder_turn_deg"])) for _ in range(2)]
        shoulder_radii = [uniform(*settings["shoulder_radius_mm"]) for _ in range(2)]
        tight_turn = turn - sum(shoulder_turns)
        return [{"kind": "hairpin shoulder", "length_mm": shoulder_radii[0] * shoulder_turns[0], "curvature_per_mm": side / shoulder_radii[0]},
                {"kind": "hairpin", "length_mm": radius * tight_turn, "curvature_per_mm": side / radius},
                {"kind": "hairpin shoulder", "length_mm": shoulder_radii[1] * shoulder_turns[1], "curvature_per_mm": side / shoulder_radii[1]}]
    if item["kind"] == "loop":
        lead = radius * math.tan((2.0 * math.pi - turn) / 2.0)
        return [{"kind": "loop lead", "length_mm": lead + uniform(*settings["lead_extra_mm"]), "curvature_per_mm": 0.0},
                {"kind": "loop", "length_mm": radius * turn, "curvature_per_mm": side / radius},
                {"kind": "loop lead", "length_mm": lead + uniform(*settings["lead_extra_mm"]), "curvature_per_mm": 0.0}]
    return [{"kind": item["kind"], "length_mm": radius * turn, "curvature_per_mm": side / radius}]



def robot_frame_mm(
        points: numpy.ndarray,
        start_xy: numpy.ndarray,
        start_direction: numpy.ndarray,
        turn_deg: float = 0.0,
        lay_out: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Paper points in the robot's frame (millimetres, x away from the base, y to its left) for the lay-out rule.

    The start point lands on the mark and ``start_direction`` (a unit vector on the paper) points along +x, turned by
    ``turn_deg`` (anticlockwise seen from above) about the mark.
    """

    settings = dict(TEST_TRACK_PARAMETERS["lay_out"])
    settings.update(lay_out or {})
    offsets = numpy.atleast_2d(points) - start_xy
    forward = offsets @ start_direction
    left = start_direction[0] * offsets[:, 1] - start_direction[1] * offsets[:, 0]
    angle = math.radians(turn_deg)
    mark = 1000.0 * numpy.array(settings["mark_m"])
    return numpy.column_stack([mark[0] + forward * math.cos(angle) - left * math.sin(angle),
                               mark[1] + forward * math.sin(angle) + left * math.cos(angle)])



def start_direction_of(
        curve: numpy.ndarray,
        length_mm: float = 10.0
) -> numpy.ndarray:
    """The unit direction of the first ``length_mm`` of ``curve`` (the first straight piece on a new track)."""

    along = arc_lengths_mm(curve)
    ahead = curve[min(int(numpy.searchsorted(along, length_mm)), len(curve) - 1)] - curve[0]
    return ahead / numpy.linalg.norm(ahead)



def __steering_side(
        position: numpy.ndarray,
        heading: float,
        last_side: float,
        randomness_generator: numpy.random.Generator,
        is_free
) -> float:
    """Left (+1) or right (-1) for the next bend: towards the middle of the free area when straight on would leave it."""

    rule = TEST_TRACK_PARAMETERS["drawing_rule"]
    look = position + rule["steer_look_ahead_mm"] * numpy.array([math.cos(heading), math.sin(heading)])
    if not is_free(look[None, :]):
        towards = numpy.array(rule["steer_towards_mm"]) - position
        turning = math.cos(heading) * towards[1] - math.sin(heading) * towards[0]
        return 1.0 if turning >= 0 else -1.0
    return -last_side if randomness_generator.random() < rule["side_change_share"] else last_side



def track_attempt(
        specification: Dict,
        randomness_generator: numpy.random.Generator,
        rule: Optional[Dict] = None,
        lay_out: Optional[Dict] = None
) -> Optional[Dict]:
    """
    One try at a track for ``specification``: the pieces are laid one after the other, the planned features at drawn
    places along the length and fillers between, every bend steered back towards the middle when the line would leave
    the paper's margin or the reach aim. None as soon as a point leaves them.

    Returns
    -------
    dict or None
        "curve" (points every "step_mm" or less, millimetres from the paper's left and bottom edges, end A first) and
        "pieces" (as laid).
    """

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    arrangement = dict(TEST_TRACK_PARAMETERS["lay_out"])
    arrangement.update(lay_out or {})
    uniform = randomness_generator.uniform
    width, height = settings["page_mm"]
    margin = settings["edge_margin_mm"]
    reach_aim_mm = 1000.0 * arrangement["reach_aim_m"]

    start = numpy.array([uniform(settings["start_from_side_mm"], width - settings["start_from_side_mm"]), uniform(*settings["start_from_bottom_mm"])])
    heading = math.radians(90.0 + uniform(-settings["start_tilt_deg"], settings["start_tilt_deg"]))
    start_direction = numpy.array([math.cos(heading), math.sin(heading)])

    def is_free(points: numpy.ndarray) -> bool:
        is_on_paper = numpy.all((points[:, 0] >= margin) & (points[:, 0] <= width - margin) & (points[:, 1] >= margin) & (points[:, 1] <= height - margin))
        robot_points = robot_frame_mm(points, start, start_direction, 0.0, arrangement)
        return bool(is_on_paper and numpy.all(numpy.hypot(robot_points[:, 0], robot_points[:, 1]) <= reach_aim_mm))

    target_length = uniform(*specification["length_mm"])
    plan = [planned_item(item) for item in specification["plan"]]
    slots = numpy.sort(uniform(0.12, 0.8, size = len(plan)))
    laid = [{"kind": "start straight", "length_mm": uniform(*settings["start_straight_mm"]), "curvature_per_mm": 0.0}]
    curve_parts = [start[None, :]]
    position, length = start, 0.0
    plan_index, last_side = 0, float(randomness_generator.choice([-1.0, 1.0]))
    pending = list(laid)
    while True:
        for piece in pending:
            points, heading = piece_points(position, heading, piece["length_mm"], piece["curvature_per_mm"], settings["step_mm"])
            if not is_free(points):
                return None
            curve_parts.append(points)
            position = points[-1]
            length += piece["length_mm"]
        if length >= target_length - 1.0:
            break
        if plan_index < len(plan) and length >= slots[plan_index] * target_length:
            item = plan[plan_index]
            plan_index += 1
            if "side" in item:
                side = 1.0 if item["side"] == "left" else -1.0
            else:
                side = __steering_side(position, heading, last_side, randomness_generator, is_free)
            pending = feature_pieces(item, side, randomness_generator, settings)
        else:
            kind = str(randomness_generator.choice(specification["fillers"]))
            side = __steering_side(position, heading, last_side, randomness_generator, is_free)
            pending = feature_pieces({"kind": kind}, side, randomness_generator, settings)
            remaining = target_length - length
            if plan_index >= len(plan) and pending[0]["length_mm"] > remaining:
                pending[0]["length_mm"] = remaining
        if any(abs(piece["curvature_per_mm"]) > 0 for piece in pending):
            last_side = side
        laid.extend(pending)
    if plan_index < len(plan):
        return None
    return {"curve": numpy.vstack(curve_parts), "pieces": laid}



def planned_features_present(
        specification: Dict,
        summary: Dict,
        rule: Dict
) -> bool:
    """Whether every planned feature is found in the measured line, one measured bend per planned bend, and nothing tighter than planned."""

    unused = list(summary["bends"])
    plan = [planned_item(item) for item in specification["plan"]]
    tightest_allowed = min([rule["pieces"][kind]["radius_mm"][0] for kind in specification["fillers"] if kind != "straight"] or [math.inf])
    for item in plan:
        kind = item["kind"]
        low, high = item.get("radius_mm", rule["pieces"][kind]["radius_mm"])
        tightest_allowed = min(tightest_allowed, low)
        if kind == "loop":
            continue
        for bend in unused:
            is_hairpin = bend["class"] == "hairpin"
            if kind == "hairpin":
                # a hairpin that merged with a neighbouring bend of the same side into more than a half turn is not kept
                is_match = is_hairpin and bend["turn_deg"] <= rule["pieces"]["hairpin"]["turn_deg"][1] + 15.0
            else:
                is_match = (not is_hairpin) and low - 0.6 <= bend["radius_mm"] <= high + 0.6
            if is_match:
                unused.remove(bend)
                break
        else:
            return False
    loops = [item for item in plan if item["kind"] == "loop"]
    if len(summary["crossings"]) != len(loops):
        return False
    for crossing, item in zip(summary["crossings"], loops):
        if crossing["angle_deg"] < rule["smallest_crossing_deg"] or crossing["loop_side"] != item.get("side", crossing["loop_side"]):
            return False
    return summary["tightest_radius_mm"] >= tightest_allowed - 0.6



def end_b_kept(
        specification: Dict,
        summary: Dict,
        rule: Dict
) -> bool:
    """Whether end B lies near an edge or far from every edge, as planned."""

    nearest = min(summary["end_b_from_edges_mm"].values())
    if specification["end_b"] == "near":
        return nearest <= rule["near_end_mm"]
    if specification["end_b"] == "far":
        return nearest >= rule["far_end_mm"]
    return True



def drawn_new_track(
        specification: Dict,
        role: str,
        rule: Optional[Dict] = None,
        lay_out: Optional[Dict] = None
) -> Dict:
    """
    Draw the track of ``specification`` with its own seed: tries until one keeps every rule.

    The rules: the margins and the reach aim (while laying the pieces), the planned features as measured from the line,
    the crossings (as many as planned loops, each at "smallest_crossing_deg" or more), the self-distance, end B near or
    far as planned, at most "running_back_share_most" of the length running back, and the full lay-out check from end A.

    Returns
    -------
    dict
        The track's entry: name, role, seed, plan, the centre line, the measured summary, dash and gap, the lay-out.

    Raises
    ------
    RuntimeError
        When no try within "attempts_per_track" keeps every rule.
    """

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    randomness_generator = numpy.random.default_rng(specification["seed"])
    for attempt in range(1, settings["attempts_per_track"] + 1):
        built = track_attempt(specification, randomness_generator, settings, lay_out)
        if built is None:
            continue
        curve = numpy.round(built["curve"], 2)
        length = float(arc_lengths_mm(curve)[-1])
        if not specification["length_mm"][0] <= length <= specification["length_mm"][1]:
            continue
        summary = line_summary(curve, settings)
        if summary["smallest_self_distance_mm"] < settings["smallest_gap_mm"] or not end_b_kept(specification, summary, settings):
            continue
        if not planned_features_present(specification, summary, settings):
            continue
        check = layout_check(curve, "A", 0.0, lay_out)
        if not check["passes"] or check["running_back_cm"] * 10.0 > settings["running_back_share_most"] * length:
            continue
        dash_generator = numpy.random.default_rng([specification["seed"], 1])
        dash = round(float(dash_generator.uniform(*settings["dash_mm"])), 1)
        gap = round(float(dash_generator.uniform(*settings["gap_mm"])), 1)
        return {"name": specification["name"], "role": role, "label": specification["label"], "seed": specification["seed"],
                "plan": specification["plan"], "fillers": specification["fillers"], "end_b_planned": specification["end_b"],
                "attempts": attempt, "curve": curve, "pieces": built["pieces"], "summary": summary,
                "features": feature_names(summary), "dash_mm": dash, "gap_mm": gap, "lay_out": check}
    raise RuntimeError(f"{specification['name']}: no try within {settings['attempts_per_track']} kept every rule")



# ----------------------------------------------------------------------------------------------------------------------
# The lay-out check with the arm model and the driver's guards
# ----------------------------------------------------------------------------------------------------------------------

def held_arm_at(
        tip_m: numpy.ndarray,
        heading_rad: float,
        start_joints: List[float],
        settings: Dict
) -> Dict:
    """
    The arm holding the tip at ``tip_m`` (on the table, pushing height) with the wrist and hand as the driver holds
    them, the forearm turned so that the camera's picture points along ``heading_rad`` (within the forearm's limit).

    Returns
    -------
    dict
        "joints" (None when no arm position reaches the point), "lean_deg", "forearm_needed_deg" (the turn the heading
        asks for) and "forearm_deg" (the turn used, within "joint_4_most_deg").
    """

    target = numpy.array([tip_m[0], tip_m[1], pushing_height(tip_m[0], tip_m[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    most = math.radians(settings["joint_4_most_deg"])
    wrist = math.radians(settings["wrist_held_deg"])
    forearm = 0.0
    joints = solve_with_wrist_fixed(target, forearm, wrist, start_joints)
    needed = forearm
    for _ in range(4):
        if joints is None:
            break
        error = math.atan2(math.sin(heading_rad - forearm_camera_heading_rad(joints)), math.cos(heading_rad - forearm_camera_heading_rad(joints)))
        needed = math.atan2(math.sin(needed - error), math.cos(needed - error))
        forearm = max(-most, min(most, needed))
        joints = solve_with_wrist_fixed(target, forearm, wrist, joints[:3])
        if abs(error) < math.radians(0.5):
            break
    if joints is None:
        joints = solve_with_wrist_fixed(target, forearm, wrist, [math.atan2(tip_m[1], tip_m[0])] + list(settings["solver_start_joints_2_and_3_rad"]))
    if joints is None:
        return {"joints": None, "lean_deg": None, "forearm_needed_deg": math.degrees(needed), "forearm_deg": math.degrees(forearm)}
    joints[5] = math.radians(settings["hand_held_deg"])
    return {"joints": joints, "lean_deg": magnet_tip_and_lean(joints)[1], "forearm_needed_deg": math.degrees(needed), "forearm_deg": math.degrees(forearm)}



def layout_check(
        curve: numpy.ndarray,
        start_end: str = "A",
        turn_deg: float = 0.0,
        lay_out: Optional[Dict] = None,
        is_detail_kept: bool = False
) -> Dict:
    """
    Check one lay-out of a printed line with the arm model and the guards of model_mode.model_step.

    The sheet is laid with the start end's cap centre on the mark and the line's first centimetre along +x, turned by
    ``turn_deg`` about the mark. Every "check_every_mm" along the line the tip stands "tip_behind_puck_m" behind the
    puck's centre along the line, and the guards are applied as the driver applies them: the tip's reach
    ("tip_reach_m"), the workspace box for the tip and, widened by "puck_margin_m", for the puck, an arm position with
    the wrist and hand held, the rod's lean ("rod_lean_most_deg").

    Parameters
    ----------
    curve : numpy.ndarray
        The centre line on the paper in millimetres, end A first.
    start_end : str
        "A" or "B": the end laid on the mark.
    turn_deg : float
        The turn of the sheet about the mark, anticlockwise seen from above.
    lay_out : dict, optional
        Overrides of TEST_TRACK_PARAMETERS["lay_out"].
    is_detail_kept : bool
        Keep the numbers of every checked point under "points".

    Returns
    -------
    dict
        The counts of refused points, the largest reach of line and tip, the largest lean and forearm turn, the
        centimetres running back, whether the paper reaches over the base, the paper's extent, "passes" (no refused
        point, no paper over the base, the whole line within the reach aim) and "is_within_aim".
    """

    arrangement = dict(TEST_TRACK_PARAMETERS["lay_out"])
    arrangement.update(lay_out or {})
    guards = dict(MODEL_MODE_PARAMETERS)
    ordered = curve if start_end == "A" else curve[::-1]
    line = resampled_curve(ordered, 1.0)
    direction = start_direction_of(line)
    robot_line = robot_frame_mm(line, line[0], direction, turn_deg, arrangement) / 1000.0
    tangents = numpy.gradient(robot_line, axis = 0)
    tangents /= numpy.maximum(numpy.linalg.norm(tangents, axis = 1, keepdims = True), 1e-12)

    radial = robot_line / numpy.linalg.norm(robot_line, axis = 1, keepdims = True)
    cosine = numpy.clip(numpy.sum(tangents * radial, axis = 1), -1.0, 1.0)
    is_back = numpy.degrees(numpy.arccos(cosine)) > arrangement["running_back_deg"]
    segment = numpy.r_[numpy.linalg.norm(numpy.diff(robot_line, axis = 0), axis = 1), 0.0]
    running_back_cm = 100.0 * float(numpy.sum(segment[is_back]))

    width, height = TEST_TRACK_PARAMETERS["drawing_rule"]["page_mm"]
    paper = numpy.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [0.0, 0.0]])
    edge_points = resampled_curve(paper, 1.0)
    robot_edges = robot_frame_mm(edge_points, line[0], direction, turn_deg, arrangement)
    half = 10.0 * SETUP_VISUALS_PARAMETERS["base_half_width_cm"]
    is_edge_on_base = bool(numpy.any((numpy.abs(robot_edges[:, 0]) < half) & (numpy.abs(robot_edges[:, 1]) < half)))
    base_corners = numpy.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    forward = numpy.array([math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))])
    mark = 1000.0 * numpy.array(arrangement["mark_m"])
    relative = base_corners - mark
    along_start = relative @ forward
    left_of_start = forward[0] * relative[:, 1] - forward[1] * relative[:, 0]
    paper_left = numpy.array([-direction[1], direction[0]])
    corners_on_paper = line[0] + along_start[:, None] * direction + left_of_start[:, None] * paper_left
    is_corner_under_paper = bool(numpy.any((corners_on_paper[:, 0] >= 0) & (corners_on_paper[:, 0] <= width) & (corners_on_paper[:, 1] >= 0) & (corners_on_paper[:, 1] <= height)))

    counts = {"out_of_reach": 0, "outside_workspace": 0, "puck_outside_workspace": 0, "no_arm_position": 0, "beyond_lean": 0}
    lean_most, forearm_most, tip_reach = 0.0, 0.0, []
    beyond_forearm_cm = 0.0
    details = []
    step = max(1, int(round(arrangement["check_every_mm"])))
    joints_before = [math.atan2(robot_line[0, 1], robot_line[0, 0])] + list(guards["solver_start_joints_2_and_3_rad"])
    area_x, area_y = guards["workspace_x_m"], guards["workspace_y_m"]
    margin = guards["puck_margin_m"]
    for index in list(range(0, len(line), step)) + ([len(line) - 1] if (len(line) - 1) % step else []):
        puck = robot_line[index]
        tip = puck - arrangement["tip_behind_puck_m"] * tangents[index]
        reach = float(numpy.hypot(*tip))
        tip_reach.append(reach)
        detail = {"along_mm": float(index), "puck_m": [round(float(value), 4) for value in puck], "tip_reach_m": round(reach, 4),
                  "line_reach_m": round(float(numpy.hypot(*puck)), 4), "refusal": None}
        if not guards["tip_reach_m"][0] <= reach <= guards["tip_reach_m"][1]:
            counts["out_of_reach"] += 1
            detail["refusal"] = "out of reach"
        elif not (area_x[0] <= tip[0] <= area_x[1] and area_y[0] <= tip[1] <= area_y[1]):
            counts["outside_workspace"] += 1
            detail["refusal"] = "tip outside the workspace"
        elif not (area_x[0] - margin <= puck[0] <= area_x[1] + margin and area_y[0] - margin <= puck[1] <= area_y[1] + margin):
            counts["puck_outside_workspace"] += 1
            detail["refusal"] = "puck outside the workspace"
        else:
            arm = held_arm_at(tip, math.atan2(tangents[index, 1], tangents[index, 0]), joints_before, guards)
            detail.update({"lean_deg": None if arm["lean_deg"] is None else round(arm["lean_deg"], 1),
                           "forearm_needed_deg": round(arm["forearm_needed_deg"], 1), "forearm_deg": round(arm["forearm_deg"], 1)})
            forearm_most = max(forearm_most, abs(arm["forearm_needed_deg"]))
            if abs(arm["forearm_needed_deg"]) > guards["joint_4_most_deg"]:
                beyond_forearm_cm += 0.1 * step
            if arm["joints"] is None:
                counts["no_arm_position"] += 1
                detail["refusal"] = "no arm position with the wrist held"
            else:
                joints_before = arm["joints"][:3]
                lean_most = max(lean_most, arm["lean_deg"])
                if arm["lean_deg"] > guards["rod_lean_most_deg"]:
                    counts["beyond_lean"] += 1
                    detail["refusal"] = "rod leans beyond the limit"
        details.append(detail)

    line_reach = numpy.hypot(robot_line[:, 0], robot_line[:, 1])
    is_within_aim = bool(line_reach.max() <= arrangement["reach_aim_m"] + 1e-9)
    is_paper_on_base = is_edge_on_base or is_corner_under_paper
    result = {"start_end": start_end, "turn_deg": turn_deg, **counts,
              "largest_line_reach_cm": round(100.0 * float(line_reach.max()), 1),
              "largest_tip_reach_cm": round(100.0 * max(tip_reach), 1), "smallest_tip_reach_cm": round(100.0 * min(tip_reach), 1),
              "largest_lean_deg": round(lean_most, 1), "largest_forearm_turn_needed_deg": round(forearm_most, 1),
              "cm_beyond_forearm_limit": round(beyond_forearm_cm, 1), "running_back_cm": round(running_back_cm, 1),
              "paper_over_base": is_paper_on_base,
              "paper_left_cm": round(float(robot_edges[:, 1].max()) / 10.0, 1), "paper_right_cm": round(float(-robot_edges[:, 1].min()) / 10.0, 1),
              "paper_front_cm": round(float(robot_edges[:, 0].max()) / 10.0, 1), "paper_near_cm": round(float(robot_edges[:, 0].min()) / 10.0, 1),
              "is_within_aim": is_within_aim,
              "passes": (sum(counts.values()) == 0) and not is_paper_on_base and is_within_aim}
    if is_detail_kept:
        result["points"] = details
    return result



def chosen_layout(
        curve: numpy.ndarray,
        lay_out: Optional[Dict] = None
) -> Tuple[Dict, List[Dict]]:
    """
    The lay-out a student uses: from end A unturned when it passes, else from end B, else turned by up to 15 degrees.

    Returns
    -------
    tuple
        The chosen check (or the end-A check when none passes) and every check that was made.
    """

    arrangement = dict(TEST_TRACK_PARAMETERS["lay_out"])
    arrangement.update(lay_out or {})
    tried = [layout_check(curve, "A", 0.0, arrangement)]
    if tried[0]["passes"]:
        return tried[0], tried
    tried.append(layout_check(curve, "B", 0.0, arrangement))
    if tried[-1]["passes"]:
        return tried[-1], tried
    for turn in arrangement["turns_tried_deg"]:
        for end in ("A", "B"):
            tried.append(layout_check(curve, end, turn, arrangement))
            if tried[-1]["passes"]:
                return tried[-1], tried
    return tried[0], tried



# ----------------------------------------------------------------------------------------------------------------------
# Drawing the pages
# ----------------------------------------------------------------------------------------------------------------------

def lengthened_curve(
        curve: numpy.ndarray,
        extension_mm: float,
        step_mm: float = 0.5
) -> numpy.ndarray:
    """``curve`` lengthened at both ends by ``extension_mm`` along its end directions (where the round caps reach)."""

    if extension_mm <= 0:
        return curve
    start_direction = start_direction_of(curve, 2.0)
    end_direction = start_direction_of(curve[::-1], 2.0)
    count = max(1, int(math.ceil(extension_mm / step_mm)))
    along = numpy.linspace(extension_mm, 0.0, count + 1)[:-1]
    before = curve[0] - along[:, None] * start_direction
    after = curve[-1] - along[::-1][:, None] * end_direction
    return numpy.vstack([before, curve, after])



def dash_intervals(
        total_mm: float,
        dash_mm: float,
        gap_mm: float
) -> List[Tuple[float, float]]:
    """
    Where the dashes lie along a line of ``total_mm``: inner dashes and gaps exactly as given, the first and the last
    dash lengthened or shortened by the same amount so that the line begins and ends with ink.
    """

    period = dash_mm + gap_mm
    count = int(math.floor((total_mm + gap_mm) / period))
    leftover = total_mm - (count * dash_mm + (count - 1) * gap_mm)
    options = [(count, leftover / 2.0), (count + 1, (leftover - period) / 2.0)]
    count, change = min(options, key = lambda option: abs(option[1]) if option[0] >= 2 else math.inf)
    intervals = []
    position = 0.0
    for index in range(count):
        length = dash_mm + (change if index in (0, count - 1) else 0.0)
        if count == 1:
            length = total_mm
        intervals.append((position, position + length))
        position += length + gap_mm
    intervals[-1] = (intervals[-1][0], total_mm)
    return intervals



def cut_curve(
        curve: numpy.ndarray,
        along: numpy.ndarray,
        first_mm: float,
        last_mm: float
) -> numpy.ndarray:
    """The part of ``curve`` between ``first_mm`` and ``last_mm`` along it, its ends interpolated."""

    inside = (along > first_mm) & (along < last_mm)
    start = numpy.array([numpy.interp(first_mm, along, curve[:, 0]), numpy.interp(first_mm, along, curve[:, 1])])
    end = numpy.array([numpy.interp(last_mm, along, curve[:, 0]), numpy.interp(last_mm, along, curve[:, 1])])
    return numpy.vstack([start, curve[inside], end])



def ink_paths(
        curve: numpy.ndarray,
        is_dashed: bool,
        dash_mm: Optional[float] = None,
        gap_mm: Optional[float] = None,
        rule: Optional[Dict] = None
) -> List[numpy.ndarray]:
    """The polylines that are inked: the whole centre line (solid), or one polyline per dash along the lengthened line."""

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    if not is_dashed:
        return [curve]
    lengthened = lengthened_curve(curve, settings["line_width_mm"] / 2.0)
    along = arc_lengths_mm(lengthened)
    return [cut_curve(lengthened, along, first, last) for first, last in dash_intervals(float(along[-1]), dash_mm, gap_mm)]



def __is_clear(
        samples: numpy.ndarray,
        ink_tree: cKDTree,
        taken: List[numpy.ndarray],
        settings: Dict,
        marks: Dict
) -> bool:
    """Whether every sample point keeps the clearance from the ink and from the marks already placed, inside the printable area."""

    width, height = settings["page_mm"]
    inset = marks["printable_inset_mm"]
    if numpy.any(samples[:, 0] < inset) or numpy.any(samples[:, 0] > width - inset) or numpy.any(samples[:, 1] < inset) or numpy.any(samples[:, 1] > height - inset):
        return False
    distances, _ = ink_tree.query(samples)
    if distances.min() < marks["clearance_mm"]:
        return False
    for other in taken:
        if numpy.min(numpy.linalg.norm(samples[:, None, :] - other[None, :, :], axis = 2)) < 2.0:
            return False
    return True



def ink_area_points(
        curve: numpy.ndarray,
        rule: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Points over the whole inked area of both prints of ``curve``: the band of the line's width along the line lengthened
    by half a width at each end, which holds the solid print's round caps and the dashed print's square ends.
    Spacing 0.5 mm along and 1 mm across.
    """

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    half = settings["line_width_mm"] / 2.0
    band = resampled_curve(lengthened_curve(curve, half), 0.5)
    tangents = numpy.gradient(band, axis = 0)
    tangents /= numpy.maximum(numpy.linalg.norm(tangents, axis = 1, keepdims = True), 1e-12)
    normals = numpy.column_stack([-tangents[:, 1], tangents[:, 0]])
    return numpy.vstack([band + offset * normals for offset in numpy.arange(-half, half + 1e-9, 1.0)])



def box_samples(
        centre: numpy.ndarray,
        half_width: float,
        half_height: float
) -> numpy.ndarray:
    """Points every millimetre over a box around ``centre``."""

    xs = numpy.arange(-half_width, half_width + 1e-9, 1.0) + centre[0]
    ys = numpy.arange(-half_height, half_height + 1e-9, 1.0) + centre[1]
    grid_x, grid_y = numpy.meshgrid(xs, ys)
    return numpy.column_stack([grid_x.ravel(), grid_y.ravel()])



def placed_marks(
        curve: numpy.ndarray,
        extra_ink: Optional[List[numpy.ndarray]] = None,
        rule: Optional[Dict] = None,
        marks: Optional[Dict] = None
) -> Dict:
    """
    Where the small marks go, clear of the line: the name and print note along the top edge, the 50 mm check bar at the
    top right, "A" beside end A, the arrow beside the first straight piece, "B" beside end B.

    Returns
    -------
    dict
        "header_mm" (the text's left end), "bar_mm" (left and right end), "letter_a_mm", "letter_b_mm" (centres),
        "arrow_mm" (tail and head) and "is_clear" (every mark found a clear place).
    """

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    page_marks = dict(TEST_TRACK_PARAMETERS["page_marks"])
    page_marks.update(marks or {})
    width, height = settings["page_mm"]
    ink_tree = cKDTree(numpy.vstack([ink_area_points(curve, settings)] + list(extra_ink or [])))
    taken = []
    is_clear = True

    header = numpy.array([page_marks["printable_inset_mm"] + 1.0, height - 12.4])
    bar_right = width - page_marks["printable_inset_mm"] - 3.0
    bar = numpy.array([[bar_right - page_marks["check_bar_mm"], height - 10.0], [bar_right, height - 10.0]])
    header_room = bar[0, 0] - 14.0 - header[0]
    header_box = box_samples(header + numpy.array([header_room / 2.0, 2.5]), header_room / 2.0, 2.9)
    is_clear &= __is_clear(header_box, ink_tree, taken, settings, page_marks)
    taken.append(header_box)
    # the bar at the top right, or, where the line or a circle comes too close there, along the bottom edge
    bar_left_ends = [(bar[0, 0], height - 10.0), (bar[0, 0], 10.0), (page_marks["printable_inset_mm"] + 16.0, 10.0)]
    for left, level in bar_left_ends:
        candidate = numpy.array([[left, level], [left + page_marks["check_bar_mm"], level]])
        bar_box = box_samples(candidate.mean(axis = 0) - numpy.array([6.0, 0.0]), page_marks["check_bar_mm"] / 2.0 + 6.0, 2.0)
        if __is_clear(bar_box, ink_tree, taken, settings, page_marks):
            bar = candidate
            break
    else:
        is_clear = False
    taken.append(bar_box)

    start_direction = start_direction_of(curve, 10.0)
    start_left = numpy.array([-start_direction[1], start_direction[0]])
    end_direction = -start_direction_of(curve[::-1], 2.0)
    found = {}
    for key, anchor, forward in (("letter_a_mm", curve[0], -start_direction), ("letter_b_mm", curve[-1], end_direction)):
        found[key] = None
        for distance in (13.0, 16.0, 20.0):
            for angle in (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
                turn = math.radians(angle)
                pointing = numpy.array([forward[0] * math.cos(turn) - forward[1] * math.sin(turn), forward[0] * math.sin(turn) + forward[1] * math.cos(turn)])
                centre = anchor + distance * pointing
                samples = box_samples(centre, 2.5, 2.5)
                if __is_clear(samples, ink_tree, taken, settings, page_marks):
                    found[key] = centre
                    taken.append(samples)
                    break
            if found[key] is not None:
                break
        is_clear &= found[key] is not None

    found["arrow_mm"] = None
    along = arc_lengths_mm(curve)
    for offset in (12.0, -12.0, 15.0, -15.0):
        tail = curve[0] + 6.0 * start_direction + offset * start_left
        head = curve[0] + 22.0 * start_direction + offset * start_left
        samples = numpy.vstack([tail + fraction * (head - tail) for fraction in numpy.linspace(0.0, 1.0, 17)])
        samples = numpy.vstack([samples + shift * start_left for shift in (-1.5, 0.0, 1.5)])
        if __is_clear(samples, ink_tree, taken, settings, page_marks):
            found["arrow_mm"] = numpy.array([tail, head])
            taken.append(samples)
            break
    is_clear &= found["arrow_mm"] is not None
    return {"header_mm": header, "header_room_mm": header_room, "bar_mm": bar, **found, "is_clear": bool(is_clear), "line_length_mm": float(along[-1])}



def write_page_pdf(
        path: Path,
        ink: List[numpy.ndarray],
        is_dashed: bool,
        header_lines: List[str],
        page_marks_placed: Dict,
        circles: Optional[List[Dict]] = None,
        rule: Optional[Dict] = None,
        marks: Optional[Dict] = None
) -> None:
    """
    Write one true-size A4 page: the inked line (round caps when solid, square-ended dashes when dashed), the grey
    circles when given, and the small marks only when "are_marks_printed" is on (off by default: the face carries
    nothing the camera could learn instead of the line). The header lines go into the PDF's title, not onto the face.

    Coordinates are millimetres from the paper's left and bottom edges, as PDF's own (points from the bottom left), so
    nothing is turned or scaled.
    """

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    page_marks = dict(TEST_TRACK_PARAMETERS["page_marks"])
    page_marks.update(marks or {})
    width, height = settings["page_mm"]
    scale = POINTS_PER_MM
    page = canvas.Canvas(str(path), pagesize = (width * scale, height * scale), pageCompression = 0, invariant = 1)
    page.setTitle(header_lines[0])
    page.setAuthor("make_test_tracks.py")

    page.setStrokeColorRGB(0, 0, 0)
    page.setLineWidth(settings["line_width_mm"] * scale)
    page.setLineJoin(1)
    page.setLineCap(0 if is_dashed else 1)
    for polyline in ink:
        drawn = page.beginPath()
        drawn.moveTo(polyline[0, 0] * scale, polyline[0, 1] * scale)
        for x, y in polyline[1:]:
            drawn.lineTo(x * scale, y * scale)
        page.drawPath(drawn, stroke = 1, fill = 0)

    grey = page_marks["grey"]
    thin = page_marks["mark_line_mm"] * scale
    page.setLineCap(0)
    page.setLineWidth(thin)
    page.setStrokeGray(grey)
    page.setFillGray(grey)
    for circle in circles or []:
        page.setStrokeGray(circle["grey"])
        page.setLineWidth(circle["line_mm"] * scale)
        page.circle(circle["centre_mm"][0] * scale, circle["centre_mm"][1] * scale, circle["diameter_mm"] / 2.0 * scale, stroke = 1, fill = 0)
        if page_marks["are_marks_printed"]:
            page.setFillGray(circle["grey"] - 0.15)
            page.setFont("Helvetica", 7)
            page.drawCentredString(circle["number_mm"][0] * scale, circle["number_mm"][1] * scale - 2.5, str(circle["number"]))
    if not page_marks["are_marks_printed"]:
        page.showPage()
        page.save()
        return
    page.setStrokeGray(grey)
    page.setFillGray(grey)
    page.setLineWidth(thin)

    header = page_marks_placed["header_mm"]
    bar = page_marks_placed["bar_mm"]
    # the header ends at least 14 mm left of the bar's place at the top right; a long line is set smaller
    room_mm = page_marks_placed["header_room_mm"]
    for number, text in enumerate(header_lines):
        font = "Helvetica-Bold" if number == 0 else "Helvetica"
        size = page_marks["text_size_pt"] + (1.0 if number == 0 else 0.0)
        size = min(size, size * room_mm * scale / max(pdfmetrics.stringWidth(text, font, size), 1e-9))
        page.setFont(font, size)
        page.drawString(header[0] * scale, (header[1] + 3.1 - 3.1 * number) * scale, text)

    page.line(bar[0, 0] * scale, bar[0, 1] * scale, bar[1, 0] * scale, bar[1, 1] * scale)
    for end in bar:
        page.line(end[0] * scale, (end[1] - 2.0) * scale, end[0] * scale, (end[1] + 2.0) * scale)
    page.drawRightString((bar[0, 0] - 2.5) * scale, (bar[0, 1] - 1.0) * scale, f"{page_marks['check_bar_mm']:.0f} mm")

    page.setFont("Helvetica-Bold", page_marks["letter_size_pt"])
    for key, letter in (("letter_a_mm", "A"), ("letter_b_mm", "B")):
        centre = page_marks_placed.get(key)
        if centre is not None:
            page.drawCentredString(centre[0] * scale, centre[1] * scale - page_marks["letter_size_pt"] * 0.35, letter)

    arrow = page_marks_placed.get("arrow_mm")
    if arrow is not None:
        tail, head = arrow
        page.setLineWidth(thin * 1.5)
        page.line(tail[0] * scale, tail[1] * scale, head[0] * scale, head[1] * scale)
        pointing = (head - tail) / numpy.linalg.norm(head - tail)
        side = numpy.array([-pointing[1], pointing[0]])
        barbs = page.beginPath()
        barbs.moveTo(*(scale * (head - 3.0 * pointing + 1.6 * side)))
        barbs.lineTo(*(scale * head))
        barbs.lineTo(*(scale * (head - 3.0 * pointing - 1.6 * side)))
        page.drawPath(barbs, stroke = 1, fill = 0)
    page.showPage()
    page.save()



def write_page_svg(
        path: Path,
        ink: List[numpy.ndarray],
        is_dashed: bool,
        header_lines: List[str],
        page_marks_placed: Dict,
        circles: Optional[List[Dict]] = None,
        rule: Optional[Dict] = None,
        marks: Optional[Dict] = None
) -> None:
    """The same page as an SVG in millimetres (SVG's y runs down the page, so y is written as the height minus y); marks as in write_page_pdf."""

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    page_marks = dict(TEST_TRACK_PARAMETERS["page_marks"])
    page_marks.update(marks or {})
    width, height = settings["page_mm"]
    grey = f"#{int(255 * page_marks['grey']):02x}" * 3

    def flipped(x: float, y: float) -> str:
        return f"{x:.2f} {height - y:.2f}"

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}mm" height="{height:g}mm" viewBox="0 0 {width:g} {height:g}">',
             f'  <rect width="{width:g}" height="{height:g}" fill="white"/>']
    cap = "butt" if is_dashed else "round"
    for polyline in ink:
        path_text = "M " + " L ".join(flipped(x, y) for x, y in polyline)
        parts.append(f'  <path d="{path_text}" fill="none" stroke="black" stroke-width="{settings["line_width_mm"]:g}" stroke-linecap="{cap}" stroke-linejoin="round"/>')
    for circle in circles or []:
        shade = f"#{int(255 * circle['grey']):02x}" * 3
        x, y = circle["centre_mm"]
        parts.append(f'  <circle cx="{x:.2f}" cy="{height - y:.2f}" r="{circle["diameter_mm"] / 2:g}" fill="none" stroke="{shade}" stroke-width="{circle["line_mm"]:g}"/>')
        if page_marks["are_marks_printed"]:
            parts.append(f'  <text x="{circle["number_mm"][0]:.2f}" y="{height - circle["number_mm"][1] + 0.9:.2f}" font-family="Arial" font-size="2.5" text-anchor="middle" fill="{shade}">{circle["number"]}</text>')
    if not page_marks["are_marks_printed"]:
        parts.append("</svg>\n")
        path.write_text("\n".join(parts), encoding = "utf-8")
        return
    header = page_marks_placed["header_mm"]
    bar = page_marks_placed["bar_mm"]
    room_mm = page_marks_placed["header_room_mm"]
    for number, text in enumerate(header_lines):
        font = "Helvetica-Bold" if number == 0 else "Helvetica"
        size_pt = page_marks["text_size_pt"] + (1.0 if number == 0 else 0.0)
        size_pt = min(size_pt, size_pt * room_mm * POINTS_PER_MM / max(pdfmetrics.stringWidth(text, font, size_pt), 1e-9))
        weight = ' font-weight="bold"' if number == 0 else ""
        parts.append(f'  <text x="{header[0]:.2f}" y="{height - header[1] - 3.1 + 3.1 * number:.2f}" font-family="Arial"{weight} '
                     f'font-size="{size_pt / POINTS_PER_MM:.2f}" fill="{grey}">{text}</text>')
    stroke = f'stroke="{grey}" stroke-width="{page_marks["mark_line_mm"]:g}"'
    parts.append(f'  <path d="M {flipped(*bar[0])} L {flipped(*bar[1])} M {flipped(bar[0, 0], bar[0, 1] - 2)} L {flipped(bar[0, 0], bar[0, 1] + 2)} '
                 f'M {flipped(bar[1, 0], bar[1, 1] - 2)} L {flipped(bar[1, 0], bar[1, 1] + 2)}" fill="none" {stroke}/>')
    parts.append(f'  <text x="{bar[0, 0] - 2.5:.2f}" y="{height - bar[0, 1] + 1.0:.2f}" font-family="Arial" font-size="2.6" text-anchor="end" fill="{grey}">{page_marks["check_bar_mm"]:.0f} mm</text>')
    for key, letter in (("letter_a_mm", "A"), ("letter_b_mm", "B")):
        centre = page_marks_placed.get(key)
        if centre is not None:
            parts.append(f'  <text x="{centre[0]:.2f}" y="{height - centre[1] + 1.6:.2f}" font-family="Arial" font-weight="bold" font-size="4.6" text-anchor="middle" fill="{grey}">{letter}</text>')
    arrow = page_marks_placed.get("arrow_mm")
    if arrow is not None:
        tail, head = arrow
        pointing = (head - tail) / numpy.linalg.norm(head - tail)
        side = numpy.array([-pointing[1], pointing[0]])
        parts.append(f'  <path d="M {flipped(*tail)} L {flipped(*head)} M {flipped(*(head - 3 * pointing + 1.6 * side))} L {flipped(*head)} '
                     f'L {flipped(*(head - 3 * pointing - 1.6 * side))}" fill="none" {stroke}/>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding = "utf-8")



def write_printer_check_pdf(
        path: Path,
        rule: Optional[Dict] = None
) -> None:
    """The page "print me first": a 100 mm bar across and one down, a 10 mm wide line and a 24 mm circle."""

    settings = dict(TEST_TRACK_PARAMETERS["drawing_rule"])
    settings.update(rule or {})
    width, height = settings["page_mm"]
    scale = POINTS_PER_MM
    page = canvas.Canvas(str(path), pagesize = (width * scale, height * scale), pageCompression = 0, invariant = 1)
    page.setTitle("Print me first: 100 mm check")
    page.setFont("Helvetica-Bold", 13)
    page.drawString(22 * scale, (height - 24) * scale, "Print me first: printer check")
    page.setFont("Helvetica", 10)
    for number, text in enumerate(["Print on A4 at 100 % (\"Actual size\"). Do NOT use \"Fit to page\" or \"Shrink\".",
                                   "Measure with a ruler. If a bar is not 100 mm, change the print setting and print again.",
                                   "Only then print the tracks, with the same setting."]):
        page.drawString(22 * scale, (height - 31 - 5 * number) * scale, text)
    page.setLineWidth(0.25 * scale)
    across = (height - 70) * scale
    page.line(55 * scale, across, 155 * scale, across)
    for x in (55, 155):
        page.line(x * scale, across - 4 * scale, x * scale, across + 4 * scale)
    page.drawString(80 * scale, across + 3 * scale, "A: 100 mm across, tick to tick")
    page.line(35 * scale, (height - 80) * scale, 35 * scale, (height - 180) * scale)
    for y in (height - 80, height - 180):
        page.line(31 * scale, y * scale, 39 * scale, y * scale)
    page.saveState()
    page.translate(31 * scale, (height - 160) * scale)
    page.rotate(90)
    page.drawString(0, 0, "B: 100 mm down, tick to tick")
    page.restoreState()
    page.setLineWidth(settings["line_width_mm"] * scale)
    page.setLineCap(0)
    page.line(60 * scale, (height - 110) * scale, 160 * scale, (height - 110) * scale)
    page.setLineWidth(0.25 * scale)
    page.drawString(60 * scale, (height - 122) * scale, "C: this line must be 10 mm wide (and 100 mm long)")
    page.circle(90 * scale, (height - 150) * scale, 12 * scale, stroke = 1, fill = 0)
    page.drawString(108 * scale, (height - 149) * scale, "D: circle of 24 mm; the red puck just covers it")
    page.setFont("Helvetica", 11)
    for number, text in enumerate(["A across: ............ mm", "B down: ............ mm", "C line width: ............ mm",
                                   "D puck covers the circle exactly:   yes  /  no", "Print setting used: ......................................"]):
        page.drawString(55 * scale, (height - 190 - 11 * number) * scale, text)
    page.showPage()
    page.save()



def draw_track_pages(
        entry: Dict,
        folder: Path,
        rule: Optional[Dict] = None
) -> List[Path]:
    """Write a track's solid and dashed pages (PDF and SVG) into ``folder``; returns the PDF files."""

    folder.mkdir(parents = True, exist_ok = True)
    curve = entry["curve"]
    page_marks_placed = placed_marks(curve, None, rule)
    entry["marks_clear"] = page_marks_placed["is_clear"]
    written = []
    for is_dashed in (False, True):
        style = "dashed" if is_dashed else "solid"
        ink = ink_paths(curve, is_dashed, entry["dash_mm"], entry["gap_mm"], rule)
        layout = entry["lay_out"]
        start_note = "start at A" if layout["start_end"] == "A" else "start at B (from A the arm cannot follow)"
        header = [f"{entry['name']} {entry['label']}, {style} ({entry['role']}, {entry['summary']['length_mm'] / 10:.1f} cm"
                  + (f", dash {entry['dash_mm']:g} gap {entry['gap_mm']:g} mm" if is_dashed else "") + f"), {start_note}"
                  + (f", sheet turned {layout['turn_deg']:+g} deg" if layout["turn_deg"] else ""),
                  "Print at 100 % (actual size), no fit to page."]
        pdf_path = folder / f"{entry['name']}_{style}.pdf"
        write_page_pdf(pdf_path, ink, is_dashed, header, page_marks_placed, None, rule)
        write_page_svg(folder / f"{entry['name']}_{style}.svg", ink, is_dashed, header, page_marks_placed, None, rule)
        written.append(pdf_path)
    return written



# ----------------------------------------------------------------------------------------------------------------------
# The diagnostic pages
# ----------------------------------------------------------------------------------------------------------------------

def course_entry(
        name: str,
        sheet: str,
        dash_mm: Optional[float],
        gap_mm: Optional[float],
        lines: Dict[str, numpy.ndarray]
) -> Dict:
    """A list entry for a course sheet's shape drawn as a diagnostic page."""

    curve = numpy.round(lines[sheet], 2)
    summary = line_summary(curve)
    return {"name": name, "role": "diagnostic", "label": f"course sheet {sheet} shape", "seed": None, "plan": None, "fillers": None,
            "end_b_planned": None, "attempts": None, "curve": curve, "pieces": None, "summary": summary, "features": feature_names(summary),
            "dash_mm": dash_mm, "gap_mm": gap_mm, "lay_out": chosen_layout(curve)[0], "course_sheet": sheet}



def first_bend_apex_mm(curve: numpy.ndarray) -> float:
    """Where the first bend of ``curve`` has turned half its turn, along the line from end A (its apex)."""

    line = resampled_curve(curve, 1.0)
    headings = segment_headings_rad(line)
    bend = measured_bends(curve)[0]
    first, last = int(bend["from_mm"]), int(min(bend["to_mm"], len(headings) - 1))
    turned = numpy.abs(headings[first:last + 1] - headings[max(first - 1, 0)])
    return float(first + int(numpy.searchsorted(turned, turned[-1] / 2.0)))



def circle_places(
        curve: numpy.ndarray,
        diagnostic: Optional[Dict] = None
) -> List[Dict]:
    """
    The twenty places of sitting A2 on sheet 03's shape, ten per page (proofreading facts, item 15).

    The places: on the line; 7, 12 and 17 mm beside it, on both sides; at the first bend's apex (on it, and 7 mm to its
    outside); at the end (centred, 5 mm short and 7 mm aside); two where the forearm must turn "forearm_turned_deg" or
    more, for the lay-out from end A. Laid by that rule, only the last 2 cm of sheet 03 lie beyond "far_reach_m"
    (40 cm), so the places beyond it are the end places themselves (the reach is listed per place). Each place is moved
    along the line by up to 40 mm until it keeps 26 mm from every circle of its page; only the two end circles of
    page 2 (5 mm short and 7 mm aside) are allowed to overlap.

    Returns
    -------
    list of dict
        Per place: "page" (1 or 2), "number" (1 to 10 on its page), "what", "along_mm", "beside_mm" (+ left of the
        direction of travel), "centre_mm", "line_reach_cm" and "forearm_needed_deg" there.
    """

    settings = dict(TEST_TRACK_PARAMETERS["diagnostic"])
    settings.update(diagnostic or {})
    line = resampled_curve(curve, 1.0)
    along_total = len(line) - 1
    tangents = numpy.gradient(line, axis = 0)
    tangents /= numpy.linalg.norm(tangents, axis = 1, keepdims = True)
    normals = numpy.column_stack([-tangents[:, 1], tangents[:, 0]])
    points = layout_check(curve, "A", 0.0, None, is_detail_kept = True)["points"]
    apex = first_bend_apex_mm(curve)
    outside = 7.0 if measured_bends(curve)[0]["side"] == "right" else -7.0
    turned = [point["along_mm"] for point in points if abs(point.get("forearm_needed_deg") or 0.0) >= settings["forearm_turned_deg"]]
    beside = settings["beside_mm"]
    end = float(along_total)
    wanted = [
        (1, "first bend apex, on the line", apex, 0.0), (1, f"{beside[0]:g} mm left of the line", 100.0, beside[0]),
        (1, f"{beside[1]:g} mm right of the line", 165.0, -beside[1]), (1, f"{beside[2]:g} mm left of the line", 230.0, beside[2]),
        (1, "on the line", 300.0, 0.0), (1, "on the line", 370.0, 0.0), (1, "forearm turned 60 deg or more, on the line", turned, 0.0),
        (1, "on the line", 430.0, 0.0), (1, "end, centred", end, 0.0), (1, "on the line", 470.0, 0.0),
        (2, "first bend apex, 7 mm outside", apex, outside), (2, f"{beside[0]:g} mm right of the line", 75.0, -beside[0]),
        (2, f"{beside[1]:g} mm left of the line", 135.0, beside[1]), (2, f"{beside[2]:g} mm right of the line", 200.0, -beside[2]),
        (2, "on the line", 265.0, 0.0), (2, "on the line", 335.0, 0.0), (2, "forearm turned 60 deg or more, on the line", turned, 0.0),
        (2, "on the line", 400.0, 0.0), (2, f"end, {settings['end_short_mm']:g} mm short", end - settings["end_short_mm"], 0.0),
        (2, f"end, {settings['end_aside_mm']:g} mm aside", end, settings["end_aside_mm"])]
    radius = settings["circle_diameter_mm"] / 2.0
    places = []
    for page, what, where, offset in wanted:
        is_end = what.startswith("end")
        if isinstance(where, list):
            used = [place["along_mm"] for place in places if place["what"] == what]
            candidates = [value for value in where if all(abs(value - other) >= 40.0 for other in used)] or where
        else:
            shifts = [0.0] if is_end else [0.0] + [sign * step for step in range(5, 45, 5) for sign in (1.0, -1.0)]
            candidates = [where + shift for shift in shifts]
        chosen = None
        for along in candidates:
            index = int(min(max(round(along), 0), along_total))
            centre = line[index] + offset * normals[index]
            others = [place for place in places if place["page"] == page and not (is_end and place["what"].startswith("end") and page == 2)]
            is_apart = all(numpy.linalg.norm(numpy.array(place["centre_mm"]) - centre) >= 2.0 * radius + 2.0 for place in others)
            is_on_paper = radius + 8.0 <= centre[0] <= 210.0 - radius - 8.0 and radius + 8.0 <= centre[1] <= 297.0 - radius - 9.0
            if is_apart and is_on_paper:
                chosen = (index, centre)
                break
        if chosen is None:
            raise RuntimeError(f"no free place for '{what}' on page {page}")
        index, centre = chosen
        nearest = min(points, key = lambda point: abs(point["along_mm"] - index))
        places.append({"page": page, "number": sum(place["page"] == page for place in places) + 1, "what": what, "along_mm": float(index),
                       "beside_mm": offset, "centre_mm": [round(float(value), 1) for value in centre],
                       "line_reach_cm": round(100.0 * float(numpy.hypot(*nearest["puck_m"])), 1),
                       "forearm_needed_deg": nearest.get("forearm_needed_deg")})
    return places



def draw_circle_pages(
        entry: Dict,
        folder: Path,
        diagnostic: Optional[Dict] = None
) -> Tuple[List[Path], List[Dict]]:
    """Write the two pages of sheet 03's shape with the grey circles; returns the PDF files, the places and the circles per page."""

    settings = dict(TEST_TRACK_PARAMETERS["diagnostic"])
    settings.update(diagnostic or {})
    places = circle_places(entry["curve"], settings)
    radius = settings["circle_diameter_mm"] / 2.0
    line_tree = cKDTree(ink_area_points(entry["curve"]))
    angles = numpy.linspace(0.0, 2.0 * math.pi, 73)
    written = []
    circles_per_page = {}
    for page_number in (1, 2):
        on_page = [place for place in places if place["page"] == page_number]
        outlines = [numpy.array(place["centre_mm"]) + radius * numpy.column_stack([numpy.cos(angles), numpy.sin(angles)]) for place in on_page]
        page_marks_placed = placed_marks(entry["curve"], outlines)
        circles = []
        numbers_placed = []
        for place, outline in zip(on_page, outlines):
            centre = numpy.array(place["centre_mm"])
            other_outlines = numpy.vstack([other for other in outlines if other is not outline] + numbers_placed or [numpy.full((1, 2), -1e6)])
            other_tree = cKDTree(other_outlines)
            number_at = centre + numpy.array([0.0, radius + 3.0])
            for angle in numpy.radians([90.0, 45.0, 135.0, 0.0, 180.0, -45.0, -135.0, -90.0, 67.5, 112.5, 22.5, 157.5]):
                candidate = centre + (radius + 3.5) * numpy.array([math.cos(angle), math.sin(angle)])
                is_inside = 9.0 <= candidate[0] <= 201.0 and 9.0 <= candidate[1] <= 280.0
                if is_inside and line_tree.query(candidate)[0] > 2.5 and other_tree.query(candidate)[0] > 3.0:
                    number_at = candidate
                    break
            numbers_placed.append(number_at[None, :])
            circles.append({"centre_mm": centre, "diameter_mm": settings["circle_diameter_mm"], "grey": settings["circle_grey"],
                            "line_mm": settings["circle_line_mm"], "number": place["number"], "number_mm": number_at})
        name = f"C03_circles_page_{page_number}"
        header = [f"{name}: course sheet 03 shape, solid, start at A; 24 mm circles {page_number} of 2 (diagnostic_places.csv)",
                  "Print at 100 % (actual size), no fit to page."]
        ink = ink_paths(entry["curve"], False)
        pdf_path = folder / f"{name}.pdf"
        write_page_pdf(pdf_path, ink, False, header, page_marks_placed, circles)
        write_page_svg(folder / f"{name}.svg", ink, False, header, page_marks_placed, circles)
        written.append(pdf_path)
        circles_per_page[page_number] = circles
    return written, places, circles_per_page



# ----------------------------------------------------------------------------------------------------------------------
# Lists, overview, print files
# ----------------------------------------------------------------------------------------------------------------------

def listed_entry(entry: Dict) -> Dict:
    """A track's entry for tracks.json: everything but the numpy arrays, the centre line as a list of points."""

    curve = entry["curve"]
    start_direction = start_direction_of(curve, 10.0)
    end_direction = -start_direction_of(curve[::-1], 2.0)
    is_dashed_listed = entry["dash_mm"] is not None
    listed = {key: value for key, value in entry.items() if key not in ("curve",)}
    listed.update({"length_mm": entry["summary"]["length_mm"], "tightest_radius_mm": entry["summary"]["tightest_radius_mm"],
                   "caps": {"A": {"centre_mm": [round(float(value), 2) for value in curve[0]],
                                  "leaving_direction_deg": round(math.degrees(math.atan2(start_direction[1], start_direction[0])), 1)},
                            "B": {"centre_mm": [round(float(value), 2) for value in curve[-1]],
                                  "arriving_direction_deg": round(math.degrees(math.atan2(end_direction[1], end_direction[0])), 1)},
                            "solid": "round: the ink reaches 5 mm beyond each end point",
                            "dashed": "square dash ends; the dashes run along the line lengthened by 5 mm at each end, so the ink reaches as far as the solid print's" if is_dashed_listed else None},
                   "centre_line_mm": [[float(x), float(y)] for x, y in curve]})
    return listed



def write_listing(
        entries: List[Dict],
        folder: Path,
        extra: Optional[Dict] = None
) -> None:
    """Write tracks.json (from the same entries the pages were drawn from) and lay_outs.csv."""

    rule = {key: (str(value) if isinstance(value, Path) else value) for key, value in TEST_TRACK_PARAMETERS["drawing_rule"].items()}
    lay_out = dict(TEST_TRACK_PARAMETERS["lay_out"])
    document = {"made_by": "02_code/make_test_tracks.py", "coordinates": "millimetres from the paper's left and bottom edges; end A first",
                "rule": rule, "lay_out_rule": lay_out, "tracks": [listed_entry(entry) for entry in entries]}
    document.update(extra or {})
    (folder / "tracks.json").write_text(json.dumps(document, indent = 1), encoding = "utf-8")

    columns = ["name", "role", "start_end", "turn_deg", "passes", "is_chosen", "largest_line_reach_cm", "largest_tip_reach_cm",
               "smallest_tip_reach_cm", "largest_lean_deg", "largest_forearm_turn_needed_deg", "cm_beyond_forearm_limit",
               "running_back_cm", "length_cm", "out_of_reach", "outside_workspace", "puck_outside_workspace", "no_arm_position",
               "beyond_lean", "paper_over_base", "is_within_aim", "paper_left_cm", "paper_right_cm", "paper_front_cm", "paper_near_cm"]
    with open(folder / "lay_outs.csv", "w", newline = "", encoding = "utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames = columns, extrasaction = "ignore")
        writer.writeheader()
        for entry in entries:
            for check in entry["lay_outs_tried"]:
                writer.writerow({**check, "name": entry["name"], "role": entry["role"], "length_cm": round(entry["summary"]["length_mm"] / 10.0, 1),
                                 "is_chosen": check is entry["lay_out"] or (check["start_end"] == entry["lay_out"]["start_end"] and check["turn_deg"] == entry["lay_out"]["turn_deg"])})



def write_overview(
        entries: List[Dict],
        path: Path
) -> None:
    """One picture with every track as a thumbnail: the paper, the line, end A (green) and end B (red), name and length."""

    columns = 8
    rows = int(math.ceil(len(entries) / columns))
    width, height = TEST_TRACK_PARAMETERS["drawing_rule"]["page_mm"]
    figure, axes = pyplot.subplots(rows, columns, figsize = (2.0 * columns, 2.9 * rows))
    for axis in numpy.atleast_1d(axes).ravel():
        axis.axis("off")
    for axis, entry in zip(numpy.atleast_1d(axes).ravel(), entries):
        axis.add_patch(pyplot.Rectangle((0, 0), width, height, fill = False, edgecolor = "#999999", linewidth = 0.8))
        curve = entry["curve"]
        axis.plot(curve[:, 0], curve[:, 1], color = "black", linewidth = 2.6, solid_capstyle = "round")
        axis.plot(*curve[0], "o", color = "#2e9d4a", markersize = 4)
        axis.plot(*curve[-1], "s", color = "#c23b3b", markersize = 4)
        axis.set_xlim(-5, width + 5)
        axis.set_ylim(-5, height + 5)
        axis.set_aspect("equal")
        axis.set_title(f"{entry['name']} {entry['summary']['length_mm'] / 10:.0f} cm\n{entry['label']}", fontsize = 7.5)
    figure.suptitle("Tracks of 12.11.26 (end A green, end B red; the bottom edge lies on the robot's side)", fontsize = 10)
    figure.tight_layout()
    figure.savefig(path, dpi = 110)
    pyplot.close(figure)



def thumbnail(
        page: canvas.Canvas,
        curve: numpy.ndarray,
        left_mm: float,
        bottom_mm: float,
        width_mm: float,
        title: str,
        circles: Optional[List[Dict]] = None
) -> None:
    """
    Draw a small picture of one track page on a list page: the paper's outline, the line at scale, end A with a dot, the
    letter and an arrow along the leaving direction, end B with its letter, and (circle pages) the numbered circles.
    """

    paper_width, paper_height = TEST_TRACK_PARAMETERS["drawing_rule"]["page_mm"]
    factor = width_mm / paper_width
    scale = POINTS_PER_MM

    def placed(point: numpy.ndarray) -> Tuple[float, float]:
        return (left_mm + factor * point[0]) * scale, (bottom_mm + factor * point[1]) * scale

    page.setStrokeGray(0.6)
    page.setLineWidth(0.3)
    page.rect(left_mm * scale, bottom_mm * scale, width_mm * scale, factor * paper_height * scale, stroke = 1, fill = 0)
    page.setFillGray(0.0)
    page.setFont("Helvetica-Bold", 7.5)
    page.drawString(left_mm * scale, (bottom_mm + factor * paper_height + 1.5) * scale, title)
    page.setStrokeGray(0.0)
    page.setLineWidth(TEST_TRACK_PARAMETERS["drawing_rule"]["line_width_mm"] * factor * scale)
    page.setLineCap(1)
    page.setLineJoin(1)
    drawn = page.beginPath()
    drawn.moveTo(*placed(curve[0]))
    for point in curve[1::4]:
        drawn.lineTo(*placed(point))
    drawn.lineTo(*placed(curve[-1]))
    page.drawPath(drawn, stroke = 1, fill = 0)
    for circle in circles or []:
        page.setStrokeGray(0.45)
        page.setLineWidth(0.4)
        centre = placed(numpy.array(circle["centre_mm"]))
        page.circle(centre[0], centre[1], circle["diameter_mm"] / 2.0 * factor * scale, stroke = 1, fill = 0)
        page.setFont("Helvetica", 5.5)
        label = placed(numpy.array(circle["number_mm"]))
        page.drawCentredString(label[0], label[1] - 2.0, str(circle["number"]))
    direction = start_direction_of(curve, 10.0)
    side = numpy.array([-direction[1], direction[0]])
    page.setFillColorRGB(0.85, 0.1, 0.1)
    page.setStrokeColorRGB(0.85, 0.1, 0.1)
    start = placed(curve[0])
    page.circle(start[0], start[1], 1.3 * scale, stroke = 0, fill = 1)
    tail = curve[0] + 14.0 * side + 4.0 * direction
    head = curve[0] + 14.0 * side + 40.0 * direction
    page.setLineWidth(0.8)
    page.line(*placed(tail), *placed(head))
    barbs = page.beginPath()
    barbs.moveTo(*placed(head - 9.0 * direction + 5.0 * side))
    barbs.lineTo(*placed(head))
    barbs.lineTo(*placed(head - 9.0 * direction - 5.0 * side))
    page.drawPath(barbs, stroke = 1, fill = 0)
    page.setFont("Helvetica-Bold", 9)
    letter_a = placed(curve[0] - 16.0 * direction)
    page.drawCentredString(letter_a[0], letter_a[1] - 3.0, "A")
    end_direction = -start_direction_of(curve[::-1], 2.0)
    page.setFillGray(0.0)
    letter_b = placed(curve[-1] + 16.0 * end_direction)
    page.drawCentredString(letter_b[0], letter_b[1] - 3.0, "B")



def write_list_pages(
        rows: List[Dict],
        shapes: List[Dict],
        path: Path,
        title: str,
        notes: List[str]
) -> int:
    """
    The list pages of a print file: what to do, a numbered table in print order, then thumbnails of every shape.

    Parameters
    ----------
    rows : list of dict
        Per printed page, in print order: "number" (in the pile), "name", "style", "entry" (or None for the printer
        page and the list pages themselves) and "note".
    shapes : list of dict
        Per thumbnail: "title", "curve" and optional "circles".

    Returns
    -------
    int
        The number of list pages written.
    """

    width, height = TEST_TRACK_PARAMETERS["drawing_rule"]["page_mm"]
    page_marks = TEST_TRACK_PARAMETERS["page_marks"]
    scale = POINTS_PER_MM
    page = canvas.Canvas(str(path), pagesize = (width * scale, height * scale), pageCompression = 0, invariant = 1)
    page.setTitle(title)
    count = 0
    per_page = page_marks["table_rows_per_page"]
    for first in range(0, len(rows), per_page):
        y = height - 20
        page.setFillGray(0.0)
        page.setFont("Helvetica-Bold", 13)
        page.drawString(18 * scale, y * scale, title)
        y -= 8
        if first == 0:
            page.setFont("Helvetica", 10)
            for note in notes:
                page.drawString(18 * scale, y * scale, note)
                y -= 5.0
            y -= 3
        page.setFont("Helvetica-Bold", 8)
        headings = [("no.", 18), ("page", 28), ("print", 52), ("length", 66), ("dash / gap", 82), ("features", 104)]
        for text, x in headings:
            page.drawString(x * scale, y * scale, text)
        y -= 4.6
        page.setFont("Helvetica", 7.5)
        for row in rows[first:first + per_page]:
            entry = row["entry"]
            values = [(str(row["number"]), 18), (row["name"], 28), (row["style"], 52)]
            if entry is not None:
                values.append((f"{entry['summary']['length_mm'] / 10:.1f} cm", 66))
                if row["style"] == "dashed":
                    values.append((f"{entry['dash_mm']:g} / {entry['gap_mm']:g} mm", 82))
            features = row["note"] or (compact_features(entry["summary"]) if entry is not None else "")
            while features and pdfmetrics.stringWidth(features, "Helvetica", 7.5) > (width - 104 - 10) * scale:
                features = features[:-1]
            values.append((features, 104))
            for text, x in values:
                page.drawString(x * scale, y * scale, text)
            y -= 4.3
        page.showPage()
        count += 1
    per_page = page_marks["thumbnails_per_page"]
    columns = 4
    box_width = 40.0
    box_height = box_width * height / width
    for first in range(0, len(shapes), per_page):
        page.setFillGray(0.0)
        page.setFont("Helvetica-Bold", 11)
        page.drawString(18 * scale, (height - 16) * scale, title + ": the shapes")
        page.setFont("Helvetica", 8.5)
        page.drawString(18 * scale, (height - 22) * scale,
                        "Red dot and A: the start end; the arrow shows where the line leaves. B: the other end. Bottom edge = robot side.")
        for index, shape in enumerate(shapes[first:first + per_page]):
            row, column = divmod(index, columns)
            left = 18.0 + column * (box_width + 5.0)
            bottom = height - 30.0 - (row + 1) * (box_height + 8.0)
            thumbnail(page, shape["curve"], left, bottom, box_width, shape["title"], shape.get("circles"))
        page.showPage()
        count += 1
    page.save()
    return count



def merged_pdf(
        path: Path,
        files: List[Path]
) -> int:
    """One PDF of ``files``' pages in the given order (pages copied as they are, nothing scaled); returns the page count."""

    merged = fitz.open()
    for file in files:
        with fitz.open(str(file)) as source:
            merged.insert_pdf(source)
    merged.save(str(path), garbage = 3, deflate = True, no_new_id = True)
    pages = merged.page_count
    merged.close()
    return pages



def write_print_set(
        folder: Path,
        name: str,
        title: str,
        pages: List[Dict],
        shapes: List[Dict]
) -> List[Dict]:
    """
    Write the print file ``name``: the printer check page, the list pages, then ``pages`` in their order.

    Parameters
    ----------
    pages : list of dict
        Per track page: "file" (its single-page PDF), "name", "style", "entry" and "note".

    Returns
    -------
    list of dict
        The print order: per page of the file its number, name and the single-page source file.
    """

    notes = ["Print single-sided at 100 % (actual size).", "Never \"fit to page\" or \"shrink\".",
             "Page 1 checks the printer: measure it first.", "Keep the pile in order.",
             "Write each page's name on its BACK,", "and an A on the back at the start end.",
             "The start end is the red A on the small pictures (last list page).",
             "Lay a sheet: its A end on the table mark,", "the first straight piece pointing straight away from the robot."]
    list_path = folder / f"{Path(name).stem}_list.pdf"
    list_page_count = 0
    for _ in range(2):
        rows = [{"number": 1, "name": "printer check", "style": "-", "entry": None, "note": "measure: bars 100 mm, line 10 mm"}]
        rows += [{"number": 2 + index, "name": "list", "style": "-", "entry": None, "note": "this list"} for index in range(list_page_count)]
        offset = len(rows) + 1
        rows += [{"number": offset + index, "name": page["name"], "style": page["style"], "entry": page["entry"], "note": page["note"]}
                 for index, page in enumerate(pages)]
        written = write_list_pages(rows, shapes, list_path, title, notes)
        if written == list_page_count:
            break
        list_page_count = written
    files = [folder / "print_me_first_100mm_bar.pdf", list_path] + [page["file"] for page in pages]
    merged_pdf(folder / name, files)
    order = [{"page": 1, "name": "printer check", "source": "print_me_first_100mm_bar.pdf"}]
    order += [{"page": 2 + index, "name": "list", "source": f"{list_path.name} page {index + 1}"} for index in range(list_page_count)]
    order += [{"page": row["number"], "name": row["name"], "style": row["style"],
               "source": str(page["file"].relative_to(folder)).replace("\\", "/")} for row, page in zip(rows[-len(pages):], pages)]
    return order



def write_course_measurements(
        rows: List[Dict],
        folder: Path
) -> None:
    """Write the six course sheets' numbers as course_sheets_measured.json and .csv."""

    (folder / "course_sheets_measured.json").write_text(json.dumps({
        "source": "06_deliverables/research_notes/Recording tool/2026-11-06_sheet_measurements.json, key line_xy_cm, traced from the course's "
                  "photos (about 3 mm; the long-edge position of sheets 04 and 05 uncertain by 7 and 3 mm), turned half a turn so that end A "
                  "lies at the bottom; measured with make_test_tracks.line_summary, the functions the new tracks are checked with",
        "sheet_06_dashes": {"dash_mm": 20.0, "gap_mm": 10.0, "ends": "square",
                            "how": "photo path_06_alpha_loop_dashed.jpeg: 14 single dashes, length over width of each dash's smallest box "
                                   "times 10 mm gives 19.5 to 21.4 mm (median 20.2); centre-to-centre spacing of neighbouring dashes on "
                                   "straight stretches over the local dash width gives 29 to 30 mm, so gaps of 9 to 10 mm; the dash ends "
                                   "are square in the photo"},
        "sheets": rows}, indent = 1), encoding = "utf-8")
    columns = ["sheet", "length_mm", "tightest_radius_mm", "tightest_radius_smoothed_mm", "bends", "hairpins", "crossings",
               "smallest_self_distance_mm", "end_a_from_edges_mm", "end_b_from_edges_mm", "nearest_edge_to_line_mm", "features"]
    with open(folder / "course_sheets_measured.csv", "w", newline = "", encoding = "utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            bends = "; ".join(f"{bend['class']} r{bend['radius_mm']:g} turn {bend['turn_deg']:g} {bend['side']}" for bend in row["bends"])
            crossings_text = "; ".join(f"angle {crossing['angle_deg']:g}, loop {crossing['loop_side']} size {crossing['loop_size_mm']:g} "
                                       f"extent {crossing['loop_extent_mm']:g} length {crossing['loop_length_mm']:g}" for crossing in row["crossings"])
            writer.writerow([row["sheet"], row["length_mm"], row["tightest_radius_mm"], row["tightest_radius_smoothed_mm"], bends, row["hairpins"],
                             crossings_text, row["smallest_self_distance_mm"], json.dumps(row["end_a_from_edges_mm"]),
                             json.dumps(row["end_b_from_edges_mm"]), row["nearest_edge_to_line_mm"], "; ".join(row["features"])])



def drawn_set(
        role: str,
        names: Optional[List[str]] = None
) -> List[Dict]:
    """Draw the tracks of the set ``role`` ("training" or "development"), or only those in ``names``, with their lay-outs."""

    entries = []
    for specification in TEST_TRACK_PARAMETERS[role]:
        if names is not None and specification["name"] not in names:
            continue
        entry = drawn_new_track(specification, role)
        entry["lay_out"], entry["lay_outs_tried"] = entry["lay_out"], [entry["lay_out"], layout_check(entry["curve"], "B", 0.0)]
        entries.append(entry)
        summary = entry["summary"]
        print(f"{entry['name']}: {summary['length_mm'] / 10:.1f} cm, tightest {summary['tightest_radius_mm']:.1f} mm, "
              f"reach {entry['lay_out']['largest_line_reach_cm']:.1f} cm, back {entry['lay_out']['running_back_cm']:.1f} cm, "
              f"dash {entry['dash_mm']:g}/{entry['gap_mm']:g}, {entry['attempts']} tries; {', '.join(entry['features'])}")
    return entries



def pilot_note(
        entries: List[Dict],
        folder: Path,
        order: Optional[List[Dict]] = None
) -> None:
    """Write pilot_ready.txt: the eight pilot pages, where they are in the print file, and per track the numbers a student needs."""

    lines = ["Pilot tracks T01 to T04 (12.11.26, second version: the pages carry the black line only).",
             "Pages printed earlier today with grey letters, an arrow, a bar or a title on them: do not record on them; print again.",
             "Print single-sided at 100 % (\"actual size\", no fit to page). First print print_me_first_100mm_bar.pdf and measure:",
             "both bars 100 mm, the line 10 mm. Then print these eight pages with the same setting:", ""]
    pile = {(row["name"], row.get("style")): row["page"] for row in order or []}
    for entry in entries:
        for style in ("solid", "dashed"):
            where = f"   (page {pile[(entry['name'], style)]} of print_set_saturday.pdf)" if (entry["name"], style) in pile else ""
            lines.append(f"  training/{entry['name']}_{style}.pdf{where}")
    lines.append("")
    lines.append("Nothing is printed on the face, so: write each page's name on its BACK, and an A on the back at the start end.")
    lines.append("Which end is A: see the small pictures on the list pages of print_set_saturday.pdf, or overview.png.")
    lines.append("")
    for entry in entries:
        layout = entry["lay_out"]
        lines.append(f"{entry['name']} {entry['label']}: length {entry['summary']['length_mm'] / 10:.1f} cm, tightest radius "
                     f"{entry['summary']['tightest_radius_mm']:.1f} mm, dash {entry['dash_mm']:g} mm and gap {entry['gap_mm']:g} mm (dashed page), "
                     f"largest reach {layout['largest_line_reach_cm']:.1f} cm from the base axis, start at end {layout['start_end']}"
                     + (f" with the sheet turned {layout['turn_deg']:+g} deg" if layout["turn_deg"] else "") + ".")
        start = entry["curve"][0]
        lines.append(f"    end A: {start[0]:.0f} mm from the left edge and {start[1]:.0f} mm from the bottom edge (bottom = robot side)")
        lines.append(f"    features: {', '.join(entry['features']) or 'straight pieces only'}")
    lines.append("")
    lines.append("Lay every sheet the same way: the centre of end A's round cap on the mark 19.5 cm in front of the robot's base axis and")
    lines.append("1.5 cm to its right, the first straight piece pointing straight away from the robot.")
    lines.append("Numbers: lengths and radii are measured from the drawn centre line; the reach is calculated with the arm model (not measured at the robot).")
    (folder / "pilot_ready.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")



def main() -> None:
    """Draw the stage named on the command line (see the module's docstring)."""

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "old":
        main_old_sets()
        return
    folder = Path(TEST_TRACK_PARAMETERS["new_output_folder"])
    folder.mkdir(parents = True, exist_ok = True)
    write_printer_check_pdf(folder / "print_me_first_100mm_bar.pdf")
    course_rows = course_sheet_measurements()
    write_course_measurements(course_rows, folder)

    entries = []
    print_orders = {}
    if stage in ("pilot", "training", "all"):
        names = ["T01", "T02", "T03", "T04"] if stage == "pilot" else None
        training = drawn_set("training", names)
        for entry in training:
            draw_track_pages(entry, folder / "training")
        entries += training
        if stage == "pilot":
            pilot_note(training, folder)
        else:
            pages = [{"file": folder / "training" / f"{entry['name']}_{style}.pdf", "name": entry["name"], "style": style, "entry": entry, "note": None}
                     for entry in training for style in ("solid", "dashed")]
            shapes = [{"title": f"{entry['name']} {entry['label']}", "curve": entry["curve"]} for entry in training]
            print_orders["print_set_saturday.pdf"] = write_print_set(folder, "print_set_saturday.pdf", "Print set for 14.11.26: training tracks T01 to T16",
                                                                     pages, shapes)
            pilot_note([entry for entry in training if entry["name"] in ("T01", "T02", "T03", "T04")], folder, print_orders["print_set_saturday.pdf"])
    if stage in ("development", "all"):
        development = drawn_set("development")
        for entry in development:
            draw_track_pages(entry, folder / "development")
        entries += development
    places = None
    if stage in ("diagnostic", "all"):
        diagnostic = TEST_TRACK_PARAMETERS["diagnostic"]
        lines = course_sheet_lines()
        diagnostic_entries = []
        for page in diagnostic["pages"]:
            entry = course_entry(page["name"], page["sheet"], page.get("dash_mm"), page.get("gap_mm"), lines)
            entry["lay_outs_tried"] = [entry["lay_out"]]
            folder_diagnostic = folder / "diagnostic"
            folder_diagnostic.mkdir(parents = True, exist_ok = True)
            style = "dashed" if page["is_dashed"] else "solid"
            page_marks_placed = placed_marks(entry["curve"])
            ink = ink_paths(entry["curve"], page["is_dashed"], page.get("dash_mm"), page.get("gap_mm"))
            header = [f"{page['name']}: course sheet {page['sheet']} shape, {style} (diagnostic, {entry['summary']['length_mm'] / 10:.1f} cm"
                      + (f", dash {page['dash_mm']:g} gap {page['gap_mm']:g} mm" if page["is_dashed"] else "") + "), start at A",
                      "Print at 100 % (actual size), no fit to page."]
            write_page_pdf(folder_diagnostic / f"{page['name']}.pdf", ink, page["is_dashed"], header, page_marks_placed)
            write_page_svg(folder_diagnostic / f"{page['name']}.svg", ink, page["is_dashed"], header, page_marks_placed)
            diagnostic_entries.append(entry)
        circle_entry = course_entry("C03_circles", diagnostic["circle_sheet"], None, None, lines)
        circle_entry["lay_outs_tried"] = [circle_entry["lay_out"]]
        _, places, circle_pages = draw_circle_pages(circle_entry, folder / "diagnostic")
        diagnostic_entries.append(circle_entry)
        with open(folder / "diagnostic_places.csv", "w", newline = "", encoding = "utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames = list(places[0].keys()))
            writer.writeheader()
            writer.writerows(places)
        entries += diagnostic_entries
        if stage == "all":
            development = [entry for entry in entries if entry["role"] == "development"]
            pages = [{"file": folder / "development" / f"{entry['name']}_{style}.pdf", "name": entry["name"], "style": style, "entry": entry, "note": None}
                     for entry in development for style in ("solid", "dashed")]
            for entry in diagnostic_entries:
                if entry["name"] == "C03_circles":
                    for number in (1, 2):
                        pages.append({"file": folder / "diagnostic" / f"C03_circles_page_{number}.pdf", "name": f"C03_circles_page_{number}",
                                      "style": "solid", "entry": entry, "note": f"sheet 03 shape, circles 1 to 10 of page {number} (diagnostic_places.csv)"})
                else:
                    style = "dashed" if entry["dash_mm"] is not None else "solid"
                    pages.append({"file": folder / "diagnostic" / f"{entry['name']}.pdf", "name": entry["name"], "style": style, "entry": entry, "note": None})
            shapes = [{"title": f"{entry['name']} {entry['label']}", "curve": entry["curve"]} for entry in development]
            shapes += [{"title": f"{entry['name'].split('_')[0]} {entry['label']}", "curve": entry["curve"]}
                       for entry in diagnostic_entries if entry["name"] in ("C05_solid", "C06_solid", "C01_solid")]
            shapes += [{"title": f"C03 circles page {number}", "curve": circle_entry["curve"], "circles": circle_pages[number]} for number in (1, 2)]
            print_orders["print_set_tests.pdf"] = write_print_set(folder, "print_set_tests.pdf", "Print set for tests: development and diagnostic pages",
                                                                  pages, shapes)
    write_listing(entries, folder, {"diagnostic_places": places} if places else None)
    for name, order in print_orders.items():
        (folder / f"{Path(name).stem}_order.json").write_text(json.dumps(order, indent = 1), encoding = "utf-8")
    write_overview(entries, folder / ("overview_pilot.png" if stage == "pilot" else "overview.png"))
    print(f"written to {folder}")



if __name__ == "__main__":
    main()
