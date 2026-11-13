"""
Shared pieces for scoring the test visit of 12.11.26 from its logged pictures.

Nothing here connects to the robot and nothing is written under 03_data. The picture measures of the robot test of
11.11.26 (``evidence/robot_test_2026-11-11/stop_and_centring``: picture_measures.py and run_measures.py, with the
camera shift of camera_shift.json) are imported as they are; this module adds what the visit needs on top of them:

    run_folders             the 36 run folders of the visit by label;
    drawing_of_sheet        the centre line of a course sheet as printed (the traced redrawings of course_sheets/,
                            or the original drawings of course_sheets_original/), densely sampled, from end A's cap
                            centre;
    DrawingOnTable          the drawing laid onto the table by a turn and a shift (and, for a check, a scale): where
                            each millimetre of the line lies in the robot's frame;
    register_drawing        the turn and shift that lay the drawing best onto the line a run's pictures voted for;
    progress_and_offset     where a point stands against the laid drawing: how far along the line, and how far to its
                            left or right;
    stretch_verdict         the rule of the visit: is the puck's centre more than 12 mm from the centre line over
                            10 mm or more of tip travel before the mark?
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy
import scipy.optimize
import scipy.spatial

EVIDENCE_FOLDER = Path(__file__).resolve().parent
MODEL_AND_TRAINING_FOLDER = EVIDENCE_FOLDER.parents[1]
OLD_MEASURE_FOLDER = EVIDENCE_FOLDER.parent / "robot_test_2026-11-11" / "stop_and_centring"
sys.path.insert(0, str(OLD_MEASURE_FOLDER))

import picture_measures
import run_measures


VISIT_PARAMETERS = {
    "run_folder": "D:/AI_Project/03_data/model_runs",
    "run_prefix": "2026-11-12_",
    "scratch_folder": "C:/Users/e7dc/AppData/Local/Temp/claude/1808d3456aab7b9a/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/scoring_agent",
    "traced_sheet_folder": str(MODEL_AND_TRAINING_FOLDER / "tracks_2026-11-12" / "course_sheets"),
    "original_sheet_folder": str(MODEL_AND_TRAINING_FOLDER / "tracks_2026-11-12" / "course_sheets_original"),
    # the camera period of camera_shift.json whose shift and roll are used (found on the robot test of 11.11.26; the
    # camera was not touched between the two days, and s01_camera_check.py tests it on the visit's own pictures)
    "camera_period": "robot_test",
    "sample_spacing_mm": 0.5,
    "sheet_of_label": {"R00": "01", "R01": "01", "R02": "01", "R03": "01", "R04": "01", "X01": "01", "X02": "01", "X03": "01",
                       "R29": "04", "R30": "04", "R31": "04", "demo": "05"},
    # the registration: observed middle cells further from the drawing than this count only this much; drawing points
    # on cells voted bare paper cost this much each (they keep a straight line from sliding along itself)
    "registration_truncation_mm": 8.0,
    "registration_paper_cost_mm": 8.0,
    "registration_turn_grid_deg": 4.0,
    # the rule of the visit
    "offset_limit_mm": 12.0,
    "travel_limit_mm": 10.0,
    "mark_mm": {"03": 119.0, "04": 160.0},
    # where the nearest point of the line is looked for: within this window around the last step's place, so that the
    # puck is not handed to a neighbouring stretch where the line comes near itself
    "progress_back_mm": 25.0,
    "progress_ahead_mm": 60.0
}



def merged_visit_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``VISIT_PARAMETERS`` with any overrides merged over it."""

    merged = dict(VISIT_PARAMETERS)
    merged.update(parameters or {})
    return merged



def run_folders(parameters: Optional[Dict] = None) -> Dict[str, Path]:
    """
    The run folders of the visit by label ("R00" to "R31", "X01" to "X03", "demo").

    Returns
    -------
    dict
        Label to folder, in the order the runs were driven.
    """

    parameters = merged_visit_parameters(parameters)
    folders = {}
    for folder in sorted(Path(parameters["run_folder"]).glob(parameters["run_prefix"] + "*")):
        label = folder.name.split("_", 2)[2]
        folders["demo" if label.startswith("demo") else label] = folder
    return folders



def sheet_of(label: str, parameters: Optional[Dict] = None) -> str:
    """The sheet of a run: "01", "03", "04" or "05"."""

    parameters = merged_visit_parameters(parameters)
    return parameters["sheet_of_label"].get(label, "03")



def run_summary(folder: Path) -> Dict:
    """The run's summary.json."""

    return json.loads((Path(folder) / "summary.json").read_text(encoding = "utf-8"))



def resampled(points_mm: numpy.ndarray, spacing_mm: float) -> numpy.ndarray:
    """A polyline resampled at equal spacing along its length, shape (N, 2)."""

    lengths = numpy.concatenate([[0.0], numpy.cumsum(numpy.hypot(*numpy.diff(points_mm, axis = 0).T))])
    targets = numpy.arange(0.0, lengths[-1], spacing_mm)
    targets = numpy.append(targets, lengths[-1])
    return numpy.column_stack([numpy.interp(targets, lengths, points_mm[:, 0]), numpy.interp(targets, lengths, points_mm[:, 1])])



def svg_path_points(path_text: str, curve_samples: int = 400) -> numpy.ndarray:
    """
    The points of one SVG path in its own units: M, L, V, H and cubic C (absolute and relative) are read, cubic
    curves sampled finely.

    Parameters
    ----------
    path_text : str
        The path's "d" attribute.
    curve_samples : int, optional
        Points per cubic curve.

    Returns
    -------
    numpy.ndarray
        Shape (N, 2), x right and y down as in the drawing.
    """

    tokens = re.findall(r"[MmLlHhVvCc]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", path_text)
    points = []
    current = numpy.zeros(2)
    command = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if re.fullmatch(r"[A-Za-z]", token):
            command = token
            index += 1
            continue
        if command in ("M", "m", "L", "l"):
            value = numpy.array([float(tokens[index]), float(tokens[index + 1])])
            current = value if command in ("M", "L") else current + value
            points.append(current.copy())
            index += 2
            if command == "M":
                command = "L"
            elif command == "m":
                command = "l"
        elif command in ("V", "v"):
            value = float(tokens[index])
            current = numpy.array([current[0], value if command == "V" else current[1] + value])
            points.append(current.copy())
            index += 1
        elif command in ("H", "h"):
            value = float(tokens[index])
            current = numpy.array([value if command == "H" else current[0] + value, current[1]])
            points.append(current.copy())
            index += 1
        elif command in ("C", "c"):
            values = numpy.array([float(tokens[index + offset]) for offset in range(6)]).reshape(3, 2)
            if command == "c":
                values = values + current[None, :]
            start = current.copy()
            parameter = numpy.linspace(0.0, 1.0, curve_samples)[1:, None]
            curve = ((1 - parameter) ** 3 * start + 3 * (1 - parameter) ** 2 * parameter * values[0]
                     + 3 * (1 - parameter) * parameter ** 2 * values[1] + parameter ** 3 * values[2])
            points.extend(curve)
            current = values[2].copy()
            index += 6
        else:
            raise ValueError(f"path command {command} is not read")
    return numpy.asarray(points, dtype = float)



def drawing_of_sheet(
        sheet: str,
        source: str = "traced",
        parameters: Optional[Dict] = None
) -> Dict:
    """
    The centre line of a course sheet as drawn, from end A's cap centre to end B's, in millimetres of the drawing
    (x right, y down, the A4 page).

    The traced redrawings (course_sheets/C0N_course.svg) begin at end A. The original drawings (Niryo N.svg) may run the
    other way or lie half a turn round on the page; of the four ways (as drawn or half a turn round, either direction)
    the one that lies closest to the traced line is taken, and how close is returned.

    Parameters
    ----------
    sheet : str
        "01" to "06".
    source : str, optional
        "traced" or "original".
    parameters : dict, optional
        Overrides for ``VISIT_PARAMETERS``.

    Returns
    -------
    dict
        "points_mm" (shape (N, 2), equally spaced), "length_mm", "source", and for the originals "distance_to_traced_mm"
        (the median and the largest distance of the original's points to the traced line) and "orientation".
    """

    parameters = merged_visit_parameters(parameters)
    spacing = parameters["sample_spacing_mm"]
    traced_text = (Path(parameters["traced_sheet_folder"]) / f"C{sheet}_course.svg").read_text(encoding = "utf-8")
    traced = resampled(svg_path_points(re.search(r' d="([^"]+)"', traced_text).group(1)), spacing)
    if source == "traced":
        return {"points_mm": traced, "length_mm": float(spacing * (len(traced) - 1)), "source": "traced"}

    number = int(sheet)
    original_text = (Path(parameters["original_sheet_folder"]) / f"Niryo{number}.svg").read_text(encoding = "utf-8")
    original = svg_path_points(re.search(r'\sd="([^"]+)"', original_text).group(1))
    page_centre = numpy.array([105.0, 148.5])
    turned = 2.0 * page_centre[None, :] - original
    tree = scipy.spatial.cKDTree(traced)
    best = None
    for name, candidate in [("as drawn", original), ("as drawn, reversed", original[::-1]),
                            ("half a turn", turned), ("half a turn, reversed", turned[::-1])]:
        candidate = resampled(candidate, spacing)
        distances, _ = tree.query(candidate)
        start_gap = float(numpy.hypot(*(candidate[0] - traced[0])))
        score = float(numpy.median(distances)) + start_gap
        if best is None or score < best[0]:
            best = (score, name, candidate, distances)
    _, name, candidate, distances = best
    return {"points_mm": candidate, "length_mm": float(spacing * (len(candidate) - 1)), "source": "original",
            "orientation": name, "distance_to_traced_mm": [float(numpy.median(distances)), float(distances.max())]}



class DrawingOnTable:

    def __init__(
            self,
            drawing_mm: numpy.ndarray,
            turn_rad: float,
            shift_m: Sequence[float],
            scale: float = 1.0
    ):
        """
        A drawing laid on the table: the drawing's (x, -y) in metres, scaled, turned by ``turn_rad`` and shifted by
        ``shift_m`` into the robot's frame (x away from the base, y to the base's left; seen from above both frames
        turn the same way, so no mirror is needed).

        Parameters
        ----------
        drawing_mm : numpy.ndarray
            Equally spaced points of the drawing, shape (N, 2), millimetres, x right and y down.
        turn_rad : float
            The turn.
        shift_m : sequence of float
            Where the drawing's origin (the page's top-left corner) lies, metres in the robot's frame.
        scale : float, optional
            1 for a true-size print; another value only for the check of the camera's scale.
        """

        self.drawing_mm = numpy.asarray(drawing_mm, dtype = float)
        self.turn_rad = float(turn_rad)
        self.shift_m = numpy.asarray(shift_m, dtype = float)
        self.scale = float(scale)
        self.__initialize_variables()


    def __initialize_variables(self):

        rotation = numpy.array([[math.cos(self.turn_rad), -math.sin(self.turn_rad)], [math.sin(self.turn_rad), math.cos(self.turn_rad)]])
        flipped = numpy.column_stack([self.drawing_mm[:, 0], -self.drawing_mm[:, 1]]) / 1000.0
        self.points_m = self.scale * flipped @ rotation.T + self.shift_m[None, :]
        steps = numpy.hypot(*numpy.diff(self.drawing_mm, axis = 0).T)
        self.arc_mm = numpy.concatenate([[0.0], numpy.cumsum(steps)])
        tangents = numpy.gradient(self.points_m, axis = 0)
        self.tangents = tangents / numpy.maximum(numpy.linalg.norm(tangents, axis = 1)[:, None], 1e-12)
        self.tree = scipy.spatial.cKDTree(self.points_m)


    def point_at(self, arc_mm: float) -> numpy.ndarray:
        """The table point (metres) of the line's centre ``arc_mm`` from end A's cap centre."""

        return numpy.array([numpy.interp(arc_mm, self.arc_mm, self.points_m[:, 0]), numpy.interp(arc_mm, self.arc_mm, self.points_m[:, 1])])



def registration_cost(
        drawing_on_table: DrawingOnTable,
        middle_xy: numpy.ndarray,
        paper_lookup,
        parameters: Dict
) -> float:
    """
    How badly a laid drawing fits a run's votes: the mean distance (millimetres, truncated) of the voted middle cells
    to the drawing's centre line, plus a cost for every drawing point that lies on a cell voted bare paper.
    """

    distances, _ = drawing_on_table.tree.query(middle_xy)
    truncation = parameters["registration_truncation_mm"]
    line_cost = float(numpy.mean(numpy.minimum(1000.0 * distances, truncation)))
    paper_share = float(numpy.mean(paper_lookup(drawing_on_table.points_m)))
    return line_cost + parameters["registration_paper_cost_mm"] * paper_share



def paper_lookup_of(
        line_votes: numpy.ndarray,
        paper_votes: numpy.ndarray,
        line_map: "picture_measures.LineMap"
):
    """A function that says for table points whether their cell was voted bare paper (at least 2 votes, most of them paper)."""

    is_paper = (paper_votes >= 2) & (paper_votes > 2 * line_votes)

    def lookup(points_xy: numpy.ndarray) -> numpy.ndarray:
        cells = numpy.round(line_map.cell_of(points_xy)).astype(int)
        is_inside = (cells[:, 0] >= 0) & (cells[:, 0] < is_paper.shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < is_paper.shape[1])
        result = numpy.zeros(len(points_xy), dtype = bool)
        result[is_inside] = is_paper[cells[is_inside, 0], cells[is_inside, 1]]
        return result

    return lookup



def register_drawing(
        drawing_mm: numpy.ndarray,
        middle_xy: numpy.ndarray,
        paper_lookup,
        start_xy: Sequence[float],
        is_scale_free: bool = False,
        parameters: Optional[Dict] = None,
        start_turn_rad: Optional[float] = None
) -> Dict:
    """
    Lay the drawing onto the line the pictures voted for.

    Every turn on a grid is tried with the drawing's end A on ``start_xy`` (the puck's first place: the puck stands on
    end A's cap at the start), then the best few are refined freely (turn and shift, and the scale when
    ``is_scale_free``) by Nelder and Mead's method on ``registration_cost``.

    Returns
    -------
    dict
        "drawing" (the ``DrawingOnTable``), "turn_deg", "shift_m", "scale", "cost", "start_moved_mm" (how far the
        fitted end A lies from ``start_xy``) and "middle_distance_mm" (median and 90th percentile of the voted middle
        cells' distances to the fitted centre line).
    """

    parameters = merged_visit_parameters(parameters)
    start_xy = numpy.asarray(start_xy, dtype = float)
    flipped_start = numpy.array([drawing_mm[0, 0], -drawing_mm[0, 1]]) / 1000.0

    def shift_for(turn: float, scale: float = 1.0) -> numpy.ndarray:
        rotation = numpy.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
        return start_xy - scale * rotation @ flipped_start

    def cost_of(values: numpy.ndarray) -> float:
        scale = float(values[3]) if len(values) > 3 else 1.0
        laid = DrawingOnTable(drawing_mm, values[0], values[1:3], scale)
        return registration_cost(laid, middle_xy, paper_lookup, parameters)

    if start_turn_rad is None:
        turns = numpy.radians(numpy.arange(0.0, 360.0, parameters["registration_turn_grid_deg"]))
    else:
        turns = start_turn_rad + numpy.radians(numpy.arange(-20.0, 20.5, 2.0))
    trials = sorted(((cost_of(numpy.array([turn, *shift_for(turn)])), turn) for turn in turns))[:4]
    best = None
    for _, turn in trials:
        start = [turn, *shift_for(turn)] + ([1.0] if is_scale_free else [])
        steps = [math.radians(2.0), 0.004, 0.004] + ([0.01] if is_scale_free else [])
        simplex = numpy.array([start] + [numpy.array(start) + numpy.eye(len(start))[index] * steps[index] for index in range(len(start))])
        result = scipy.optimize.minimize(cost_of, numpy.array(start), method = "Nelder-Mead",
                                         options = {"initial_simplex": simplex, "xatol": 1e-6, "fatol": 1e-4, "maxiter": 4000})
        if best is None or result.fun < best.fun:
            best = result
    values = best.x
    scale = float(values[3]) if is_scale_free else 1.0
    laid = DrawingOnTable(drawing_mm, values[0], values[1:3], scale)
    distances, _ = laid.tree.query(middle_xy)
    return {"drawing": laid, "turn_deg": math.degrees(values[0]) % 360.0, "shift_m": values[1:3].tolist(), "scale": scale,
            "cost": float(best.fun), "start_moved_mm": 1000.0 * float(numpy.linalg.norm(laid.points_m[0] - start_xy)),
            "middle_distance_mm": [1000.0 * float(numpy.median(distances)), 1000.0 * float(numpy.percentile(distances, 90))]}



def progress_and_offset(
        points_xy: numpy.ndarray,
        drawing_on_table: DrawingOnTable,
        parameters: Optional[Dict] = None
) -> List[Dict]:
    """
    Where a run's puck places stand against the laid drawing, step by step.

    The nearest point of the centre line is looked for within a window around the previous step's place (from
    "progress_back_mm" behind to "progress_ahead_mm" ahead), so that the puck stays on its own stretch where the line
    comes near itself; the first step looks within the first 40 mm.

    Parameters
    ----------
    points_xy : numpy.ndarray
        The puck's places, shape (N, 2), metres; NaN rows (no puck found) are passed over.

    Returns
    -------
    list of dict
        Per point: "arc_mm" (how far along the line from end A's cap centre the nearest centre-line point lies),
        "offset_mm" (positive to the left looking along the line from end A to end B, negative to the right) and
        "distance_mm" (unsigned).
    """

    parameters = merged_visit_parameters(parameters)
    arc = drawing_on_table.arc_mm
    previous = None
    results = []
    for point in numpy.asarray(points_xy, dtype = float):
        if not numpy.all(numpy.isfinite(point)):
            results.append({"arc_mm": numpy.nan, "offset_mm": numpy.nan, "distance_mm": numpy.nan})
            continue
        if previous is None:
            is_window = arc <= 40.0
        else:
            is_window = (arc >= previous - parameters["progress_back_mm"]) & (arc <= previous + parameters["progress_ahead_mm"])
        indices = numpy.flatnonzero(is_window)
        gaps = drawing_on_table.points_m[indices] - point[None, :]
        nearest = int(indices[numpy.argmin(numpy.hypot(gaps[:, 0], gaps[:, 1]))])
        tangent = drawing_on_table.tangents[nearest]
        relative = point - drawing_on_table.points_m[nearest]
        along = float(relative @ tangent)
        across = float(tangent[0] * relative[1] - tangent[1] * relative[0])
        arc_here = float(arc[nearest] + 1000.0 * along) if 0 < nearest < len(arc) - 1 else float(arc[nearest])
        previous = arc[nearest]
        results.append({"arc_mm": arc_here, "offset_mm": 1000.0 * across, "distance_mm": 1000.0 * float(numpy.hypot(*relative))})
    return results



def tip_travel_mm(tips_xy: numpy.ndarray) -> numpy.ndarray:
    """The tip's travel up to each picture, millimetres (0 at the first)."""

    return numpy.concatenate([[0.0], numpy.cumsum(1000.0 * numpy.hypot(*numpy.diff(tips_xy, axis = 0).T))])



def stretch_verdict(
        offsets_mm: Sequence[float],
        travel_mm: Sequence[float],
        last_index: int,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    The rule of the visit on the pictures 0 to ``last_index``: is the puck's centre more than 12 mm from the centre
    line over 10 mm or more of tip travel?

    A picture stands for the tip's travel from halfway since the previous picture to halfway to the next one (the
    middle reading). The shortest reading counts only the travel between the first and the last picture of a stretch
    over the limit, the longest the travel from the picture before the stretch to the picture after it. A picture
    without a measure breaks no stretch and starts none.

    Returns
    -------
    dict
        "is_kept" (by the middle reading), "is_kept_shortest" and "is_kept_longest" (by the other two; they differ from
        the middle one only on a doubtful run), "longest_stretch_mm" (middle reading), "stretch_from" and "stretch_to"
        (its pictures) and "first_over" (the first picture over the limit, or None).
    """

    parameters = merged_visit_parameters(parameters)
    offsets = numpy.abs(numpy.asarray(offsets_mm, dtype = float))[:last_index + 1]
    travel = numpy.asarray(travel_mm, dtype = float)
    count = len(travel)
    is_over = offsets > parameters["offset_limit_mm"]
    is_known = numpy.isfinite(offsets)
    stretches = []
    index = 0
    while index <= last_index:
        if is_known[index] and is_over[index]:
            first = index
            last = index
            probe = index + 1
            while probe <= last_index and (not is_known[probe] or is_over[probe]):
                if is_known[probe]:
                    last = probe
                probe += 1
            before = travel[first - 1] if first > 0 else travel[first]
            after = travel[last + 1] if last + 1 < count else travel[last]
            middle = (travel[last] + 0.5 * (after - travel[last])) - (travel[first] - 0.5 * (travel[first] - before))
            stretches.append({"from": first, "to": last, "shortest": travel[last] - travel[first], "middle": middle, "longest": after - before})
            index = probe
        else:
            index += 1
    limit = parameters["travel_limit_mm"]
    longest = max(stretches, key = lambda stretch: stretch["middle"]) if stretches else None
    failing = [stretch for stretch in stretches if stretch["middle"] >= limit]
    return {
        "first_failing_from": int(failing[0]["from"]) if failing else None,
        "is_kept": not any(stretch["middle"] >= limit for stretch in stretches),
        "is_kept_shortest": not any(stretch["shortest"] >= limit for stretch in stretches),
        "is_kept_longest": not any(stretch["longest"] >= limit for stretch in stretches),
        "longest_stretch_mm": float(longest["middle"]) if longest else 0.0,
        "stretch_from": int(longest["from"]) if longest else None,
        "stretch_to": int(longest["to"]) if longest else None,
        "first_over": int(numpy.flatnonzero(is_known & is_over)[0]) if numpy.any(is_known & is_over) else None
    }
