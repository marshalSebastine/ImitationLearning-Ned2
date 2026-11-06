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
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
from matplotlib import patches, pyplot
from matplotlib.path import Path as PolygonPath

from config import RECORDING_PAGE_PARAMETERS, SETUP_VISUALS_PARAMETERS, WORKSPACE_PARAMETERS
from recording_page import expand_episode_plan


CODE_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = CODE_FOLDER.parent
DISTANCE_IN_NOTE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(?: to (\d+(?:\.\d+)?))? cm before the end cap")
RESAMPLE_STEP_CM = 0.1
TURN_CODES = ("TL10", "TR10", "TL15", "TR15")
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

    # A turned start: the sheet turned about the start cap, which stays where it was.
    offset_deg = start["offset_deg"]
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
    if code in TAUGHT_CODES:
        return f"Puck {setup['spot']} (from the plan, not taught)"
    side = "LEFT" if setup["offset_cm"] > 0 else "RIGHT"
    return f"Puck {abs(setup['offset_cm']):g} cm {side} of the start cap"



def caption_for(setup: Dict) -> str:
    """One or two plain sentences that say what to do for this plan number."""

    code = setup["start_code"]
    sheet_label = RECORDING_PAGE_PARAMETERS["sheets"][setup["path_id"]]["label"]
    lay = f"Sheet {sheet_label}, end {setup['line_end']} as the start: its ink begins on the START MARK, line pointing away from the robot."
    if code == "STD":
        what = "Puck centred on the start cap."
    elif code in ("OL1", "OR1", "OL1.5", "OR1.5"):
        side = "LEFT" if setup["offset_cm"] > 0 else "RIGHT"
        what = f"Puck {abs(setup['offset_cm']):g} cm {side} of the start cap (seen from behind the robot); back onto the line in the first 5 to 10 cm."
    elif code in TURN_CODES:
        side = "LEFT" if setup["offset_deg"] > 0 else "RIGHT"
        what = f"Then turn the sheet {abs(setup['offset_deg']):g} degrees {side} about the start cap (cap stays on the mark). Puck on the start cap."
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
    # Sheet 06's line is dashed on the paper; the dash pattern here only says so, it is not the printed one.
    line_style = (0, (2.0, 0.8)) if setup["path_id"] == "path_06_alpha_loop_dashed" else "-"
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
    axis.annotate("START\nMARK", mark, xytext = (-46, 10), textcoords = "offset points", fontsize = 8, fontweight = "bold", zorder = 6)

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
{"".join(sections)}
</body></html>
"""
    page_path = output_folder / "index.html"
    page_path.write_text(page, encoding = "utf-8")
    return page_path



def all_setups(parameters: Dict = SETUP_VISUALS_PARAMETERS) -> List[Dict]:
    """compute_setup() for every plan number, in plan order."""

    sheet_lines = load_sheet_lines(parameters)
    setups = []
    for path_id, entries in expand_episode_plan().items():
        for plan_number, entry in entries.items():
            setups.append(compute_setup(path_id, plan_number, entry, sheet_lines, parameters))
    return setups



def main() -> None:

    parser = argparse.ArgumentParser(description = "Draw one set-up picture per plan number and a page that lists them.")
    parser.add_argument("--plan", help = "only this plan number, for example 02-07 (the page is not rewritten)")
    arguments = parser.parse_args()

    output_folder = PROJECT_FOLDER / SETUP_VISUALS_PARAMETERS["output_folder"]
    output_folder.mkdir(parents = True, exist_ok = True)
    setups = all_setups()
    chosen = [setup for setup in setups if arguments.plan in (None, setup["plan_number"])]
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
