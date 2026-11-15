"""
Shared readers and geometry for the orientation-coverage analysis of the robot test of 11.11.26 (analyst key
"orientation_coverage").

Nothing here connects to the robot: the scripts read the training index (03_data/processed/v01_training_index) through
training_targets.load_training_frames with the audit's cleaning, the logs of the ten robot runs (03_data/model_runs), the
photo-traced lines of the six course sheets (setup_visuals.load_sheet_lines), the practice tracks' SVG files, the arm
model (ned2_fk through side_view_test) and the fitted forearm camera (forearm_camera_model). Every script of this
folder runs from this folder and writes only into it:
    D:/AI_Project/.venv/Scripts/python.exe 01_placements.py

Angles. All directions are table directions in the robot's base frame, in degrees, counter-clockwise seen from above
(0 = the base's x axis, straight away from the robot; +90 = to the robot's left).
    radial      the direction from the base axis to the magnet tip, atan2(tip_y, tip_x)
    heading     the table direction the top of the picture points to (picture_heading_deg of the index)
    step        the direction of the labelled step (heading plus the step's angle in the picture)
    "from radial" means the direction minus the radial direction: 0 = straight away from the base, +90 = sideways to
    the left on a circle round the base, 180 = straight back to the base.
"""

import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas

EVIDENCE_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = EVIDENCE_FOLDER.parents[5]
CODE_FOLDER = ROOT_FOLDER / "02_code"
if str(CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(CODE_FOLDER))

from training_targets import load_training_frames  # noqa: E402

ORIENTATION_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "index_folder": Path("D:/AI_Project/03_data/processed/v01_training_index"),
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "setup_picture_folder": Path("D:/AI_Project/06_deliverables/research_notes/Recording tool/figures/setup_per_episode"),
    "practice_track_folder": ROOT_FOLDER / "06_deliverables" / "research_notes" / "Model and training" / "test_tracks" / "practice",
    # the ten robot runs of 11.11.26 in the order of the brief
    "robot_runs": [
        {"number": 1, "folder": "2026-11-11_172021_check_no_puck", "model": "A", "sheet": "01", "outcome": "reached the end"},
        {"number": 2, "folder": "2026-11-11_172421_E_sheet01_run1", "model": "E", "sheet": "01", "outcome": "reached the end"},
        {"number": 3, "folder": "2026-11-11_172743_E_sheet02_run1", "model": "E", "sheet": "02", "outcome": "failed (lost the line, Esc)"},
        {"number": 4, "folder": "2026-11-11_173036_E_sheet02_run2", "model": "E", "sheet": "02", "outcome": "reached the end"},
        {"number": 5, "folder": "2026-11-11_173549_E_sheet03_run1", "model": "E", "sheet": "03", "outcome": "failed (overshot the first bend)"},
        {"number": 6, "folder": "2026-11-11_174108_E_sheet04_run1", "model": "E", "sheet": "04", "outcome": "failed (stuck at the hairpin)"},
        {"number": 7, "folder": "2026-11-11_174448_E_sheet05_run1", "model": "E", "sheet": "05", "outcome": "reached the end"},
        {"number": 8, "folder": "2026-11-11_174936_E_sheet06_run1", "model": "E", "sheet": "06", "outcome": "failed (left the dashed loop)"},
        {"number": 9, "folder": "2026-11-11_175203_E_sheet03_run2_no_hoodie", "model": "E", "sheet": "03", "outcome": "failed (overshot the first bend)"},
        {"number": 10, "folder": "2026-11-11_175544_B_team5_sheet03", "model": "B", "sheet": "03", "outcome": "failed (overshot the first bend)"}
    ],
    # a labelled step shorter than this has no trustworthy direction (hand-guided jitter, twists in place)
    "smallest_step_cm": 0.3,
    # the path direction of a row: from its tip to the tip this many rows later (the report's "path within 45" uses 3)
    "path_steps_ahead": 3,
    # two consecutive rows belong to one replay when the tip really went where the label says, within this distance
    "continuity_tolerance_cm": 0.3,
    # the tip trails the centre of the puck by about this much while pushing (measured in 04_physics.py, part C: the
    # puck detector puts the puck a median 2.56 cm from the tip on forward pushes): used to turn a line on the table
    # into the tip path that follows it, and to place the puck in the camera's view
    "tip_behind_puck_cm": 2.5
}



def merged_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``ORIENTATION_PARAMETERS`` with any overrides merged over it."""

    merged = dict(ORIENTATION_PARAMETERS)
    merged.update(parameters or {})
    return merged



def wrapped_deg(angle_deg):
    """Return an angle (or array of angles) in degrees wrapped into -180 to 180."""

    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0



def circular_mean_deg(angles_deg) -> float:
    """Return the mean direction of angles in degrees (NaN for no angles)."""

    angles = numpy.radians(numpy.asarray(angles_deg, dtype = float))
    angles = angles[numpy.isfinite(angles)]
    if len(angles) == 0:
        return float("nan")
    return float(numpy.degrees(numpy.arctan2(numpy.sin(angles).mean(), numpy.cos(angles).mean())))



def demonstration_frames(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    Read the demonstrations as the models were trained on them and add the directions this analysis needs.

    Parameters
    ----------
    parameters : dict, optional
        Overrides of ``ORIENTATION_PARAMETERS``.

    Returns
    -------
    pandas.DataFrame
        The audit-cleaned rows (18,032 pictures of 209 episodes), sorted by episode and row, with the extra columns
        "sheet_number", "joint_4_deg", "reach_cm", "radial_deg", "step_cm", "step_picture_deg" (the labelled step's
        angle in the picture, 0 = towards its top, positive = to its left), "step_table_deg",
        "step_from_radial_deg", "heading_from_radial_deg", "is_next_consecutive" (the next row is the next picture of
        the same replay), "path_table_deg" (from this tip to the tip "path_steps_ahead" rows later within one
        replay, NaN otherwise), "path_from_radial_deg" and "path_picture_deg".
    """

    parameters = merged_parameters(parameters)
    frames = load_training_frames({"is_audit_cleaning": True})
    frames["sheet_number"] = frames["sheet"].str[-2:]
    frames["start_kind"] = frames["start_kind"].fillna("STD")
    frames["joint_4_deg"] = numpy.degrees(frames["joint_4"])
    frames["reach_cm"] = 100.0 * numpy.hypot(frames["tip_x_m"], frames["tip_y_m"])
    frames["radial_deg"] = numpy.degrees(numpy.arctan2(frames["tip_y_m"], frames["tip_x_m"]))
    forward_cm = 100.0 * frames["tip_forward_m"].fillna(0.0)
    left_cm = 100.0 * frames["tip_left_m"].fillna(0.0)
    frames["step_cm"] = numpy.hypot(forward_cm, left_cm)
    frames["step_picture_deg"] = numpy.degrees(numpy.arctan2(left_cm, forward_cm))
    frames["step_table_deg"] = wrapped_deg(frames["picture_heading_deg"] + frames["step_picture_deg"])
    frames["step_from_radial_deg"] = wrapped_deg(frames["step_table_deg"] - frames["radial_deg"])
    frames["heading_from_radial_deg"] = wrapped_deg(frames["picture_heading_deg"] - frames["radial_deg"])

    # the next row continues this replay when the tip really arrived where the label says
    heading = numpy.radians(frames["picture_heading_deg"].to_numpy())
    labelled_x = forward_cm.to_numpy() * numpy.cos(heading) - left_cm.to_numpy() * numpy.sin(heading)
    labelled_y = forward_cm.to_numpy() * numpy.sin(heading) + left_cm.to_numpy() * numpy.cos(heading)
    grouped = frames.groupby("episode_id", sort = False)
    moved_x = 100.0 * (grouped["tip_x_m"].shift(-1) - frames["tip_x_m"]).to_numpy()
    moved_y = 100.0 * (grouped["tip_y_m"].shift(-1) - frames["tip_y_m"]).to_numpy()
    is_next_row = (grouped["row_in_episode"].shift(-1) == frames["row_in_episode"] + 1).to_numpy()
    with numpy.errstate(invalid = "ignore"):
        is_arrived = numpy.hypot(moved_x - labelled_x, moved_y - labelled_y) <= parameters["continuity_tolerance_cm"]
    frames["is_next_consecutive"] = is_next_row & is_arrived

    steps_ahead = parameters["path_steps_ahead"]
    is_chain = frames["is_next_consecutive"].to_numpy().copy()
    for ahead in range(1, steps_ahead):
        is_chain &= numpy.r_[frames["is_next_consecutive"].to_numpy()[ahead:], numpy.zeros(ahead, dtype = bool)]
    path_x = 100.0 * (grouped["tip_x_m"].shift(-steps_ahead) - frames["tip_x_m"]).to_numpy()
    path_y = 100.0 * (grouped["tip_y_m"].shift(-steps_ahead) - frames["tip_y_m"]).to_numpy()
    with numpy.errstate(invalid = "ignore"):
        is_path = is_chain & (numpy.hypot(path_x, path_y) >= steps_ahead * parameters["smallest_step_cm"])
    path_table = numpy.where(is_path, numpy.degrees(numpy.arctan2(path_y, path_x)), numpy.nan)
    frames["path_table_deg"] = path_table
    frames["path_from_radial_deg"] = wrapped_deg(path_table - frames["radial_deg"].to_numpy())
    frames["path_picture_deg"] = wrapped_deg(path_table - frames["picture_heading_deg"].to_numpy())
    return frames



def robot_run_steps(
        run: Dict,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    Read the steps.csv of one robot run of 11.11.26 and add the same directions as ``demonstration_frames``.

    Parameters
    ----------
    run : dict
        One entry of "robot_runs".
    parameters : dict, optional
        Overrides of ``ORIENTATION_PARAMETERS``.

    Returns
    -------
    pandas.DataFrame
        One row per picture with "joint_4_deg", "reach_cm", "radial_deg", "step_table_deg" (the direction the driver
        planned, from the tip to "tip_next"), "step_from_radial_deg", "heading_from_radial_deg" and "step_picture_deg".
    """

    parameters = merged_parameters(parameters)
    steps = pandas.read_csv(Path(parameters["model_runs_folder"]) / run["folder"] / "steps.csv")
    steps["joint_4_deg"] = numpy.degrees(steps["joint_4"])
    steps["reach_cm"] = 100.0 * numpy.hypot(steps["tip_x"], steps["tip_y"])
    steps["radial_deg"] = numpy.degrees(numpy.arctan2(steps["tip_y"], steps["tip_x"]))
    steps["step_table_deg"] = numpy.degrees(numpy.arctan2(steps["tip_next_y"] - steps["tip_y"], steps["tip_next_x"] - steps["tip_x"]))
    steps["step_from_radial_deg"] = wrapped_deg(steps["step_table_deg"] - steps["radial_deg"])
    steps["heading_from_radial_deg"] = wrapped_deg(steps["picture_heading_deg"] - steps["radial_deg"])
    steps["step_picture_deg"] = wrapped_deg(steps["step_table_deg"] - steps["picture_heading_deg"])
    return steps



def rotated(
        points: numpy.ndarray,
        angle_deg: float
) -> numpy.ndarray:
    """Return ``points`` (N, 2) turned counter-clockwise by ``angle_deg`` about the origin."""

    cosine, sine = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return numpy.asarray(points, dtype = float) @ numpy.array([[cosine, sine], [-sine, cosine]])



def arc_lengths(line: numpy.ndarray) -> numpy.ndarray:
    """Return the arc length from the first point to every point of the polyline ``line`` (N, 2)."""

    return numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(line, axis = 0), axis = 1))])



def resampled(
        line: numpy.ndarray,
        step: float
) -> numpy.ndarray:
    """Return the polyline ``line`` with points every ``step`` of arc length, both ends kept."""

    lengths = arc_lengths(line)
    samples = numpy.append(numpy.arange(0.0, lengths[-1], step), lengths[-1])
    return numpy.column_stack([numpy.interp(samples, lengths, line[:, 0]), numpy.interp(samples, lengths, line[:, 1])])



def course_sheet_lines() -> Dict[str, Dict]:
    """
    Return the photo-traced centre line of each of the six course sheets in the sheet's own frame.

    Returns
    -------
    dict
        Sheet number ("02") -> {"line": (N, 2) array in cm every 0.1 cm, x along the bottom edge and y up the long
        edge, running from end A to end B; "is_dashed": bool}.
    """

    from setup_visuals import load_sheet_lines

    return {number: {"line": sheet["line"], "is_dashed": number == "06"} for number, sheet in load_sheet_lines().items()}



def practice_track_lines(parameters: Optional[Dict] = None) -> Dict[str, Dict]:
    """
    Return the centre line of each printable practice track, read from its SVG file, in the sheet's own frame.

    The SVG's y axis points down the page; it is flipped so that, as for the course sheets, x runs along the bottom
    edge and y up the long edge (cm). The line runs from the track's start (near the bottom edge) to its end.

    Returns
    -------
    dict
        Track name ("practice_03") -> {"line": (N, 2) array in cm every 0.1 cm, "is_dashed": bool}.
    """

    parameters = merged_parameters(parameters)
    tracks = {}
    for path in sorted(Path(parameters["practice_track_folder"]).glob("practice_*.svg")):
        text = path.read_text(encoding = "utf-8")
        height_mm = float(re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', text).group(2))
        numbers = re.findall(r"[ML] ([0-9.]+) ([0-9.]+)", re.search(r'<path d="([^"]+)"', text).group(1))
        points_mm = numpy.array([[float(x), height_mm - float(y)] for x, y in numbers])
        is_kept = numpy.r_[True, numpy.linalg.norm(numpy.diff(points_mm, axis = 0), axis = 1) > 1e-6]
        tracks[path.stem] = {"line": resampled(points_mm[is_kept] / 10.0, 0.1), "is_dashed": "stroke-dasharray" in text}
    return tracks
