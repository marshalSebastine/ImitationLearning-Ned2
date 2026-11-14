"""
Where a new track's line should appear in the forearm camera's picture when its sheet lies right (14.11.26, Emranul:
"the faint start helped us position the paper; implement the same thing in the recording tool").

The new tracks (T01 to T16) were never recorded, so there is no old picture to lay over the live one. Instead the line
is drawn: its centre line from tracks.json (millimetres from the paper's left and bottom edges), laid on the table by
the lay-out rule (the centre of end A's round cap on the new-track mark, the first straight piece straight away from
the robot), and projected into the picture with the fitted forearm camera model at the arm's joints. The operator then
moves the sheet until the printed line fills the band between the two drawn edges.

How far to trust it: the drawing is as good as the camera model (fitted to the recordings for this taping of the
camera) and the table plane; checked against round 1 pictures in test_track_lay_out.py and on 14.11.26 by
06_deliverables/research_notes/Recording tool/2026-11-14_lay_out_line_check.md.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy

from config import NEW_TRACK_RECORDING_PARAMETERS, SHAPE_OF_SHEET, table_height
from forearm_camera_model import project_points

TRACK_LAY_OUT_PARAMETERS = {
    "tracks_file": NEW_TRACK_RECORDING_PARAMETERS["tracks_file"],
    "line_width_m": 0.010,
    # the drawn part of the line: from end A this far along it (the camera at the start sees about the first 15 cm)
    "drawn_length_m": 0.30,
    "edge_colour_bgr": (0, 220, 0),
    "centre_colour_bgr": (0, 220, 0),
    "edge_thickness_px": 2,
    "centre_thickness_px": 1
}



def track_of_sheet(path_id: str) -> Optional[str]:
    """The new track a sheet carries ("T03" for "path_13_T03_solid"), or None for a course sheet."""

    shape = SHAPE_OF_SHEET.get(path_id)
    return shape if shape is not None and str(shape).startswith("T") else None



def load_tracks(parameters: Optional[Dict] = None) -> Dict:
    """Read tracks.json: the tracks by name and the lay-out rule."""

    settings = dict(TRACK_LAY_OUT_PARAMETERS)
    settings.update(parameters or {})
    content = json.loads(Path(settings["tracks_file"]).read_text(encoding = "utf-8"))
    tracks = content["tracks"]
    by_name = tracks if isinstance(tracks, dict) else {track["name"]: track for track in tracks}
    return {"tracks": by_name, "lay_out_rule": content["lay_out_rule"]}



def track_centre_line_on_table(
        track: Dict,
        lay_out_rule: Dict
) -> numpy.ndarray:
    """
    Lay a track's drawn centre line on the table by the lay-out rule, in metres in the robot's base frame.

    The paper's frame (x to the right, y up the page, seen from above) and the base frame (x away from the robot, y to
    its left, z up) are both right-handed with z up, so laying the sheet face up is a turn, never a mirror: the turn
    that points end A's leaving direction along +x, then a shift that puts end A's cap centre on the mark.

    Parameters
    ----------
    track : dict
        One track of tracks.json, with "centre_line_mm" (end A first) and "caps".
    lay_out_rule : dict
        tracks.json "lay_out_rule", with "mark_m".

    Returns
    -------
    numpy.ndarray
        (N, 2) points, end A first.
    """

    points_mm = numpy.asarray(track["centre_line_mm"], dtype = float)
    cap_mm = numpy.asarray(track["caps"]["A"]["centre_mm"], dtype = float)
    turn = -math.radians(track["caps"]["A"]["leaving_direction_deg"])
    rotation = numpy.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
    return numpy.asarray(lay_out_rule["mark_m"], dtype = float) + (points_mm - cap_mm) @ rotation.T / 1000.0



def band_on_table(
        centre_line_xy: numpy.ndarray,
        line_width_m: float
) -> Dict[str, numpy.ndarray]:
    """The centre line and its two edges, half a line width to each side, as (N, 3) points on the table."""

    direction = numpy.gradient(centre_line_xy, axis = 0)
    direction /= numpy.maximum(numpy.linalg.norm(direction, axis = 1, keepdims = True), 1e-12)
    normal = numpy.column_stack([-direction[:, 1], direction[:, 0]])
    band = {}
    for name, offset in (("centre", 0.0), ("left_edge", line_width_m / 2), ("right_edge", -line_width_m / 2)):
        xy = centre_line_xy + offset * normal
        heights = numpy.array([table_height(x, y) for x, y in xy])
        band[name] = numpy.column_stack([xy, heights])
    return band



def expected_track_in_picture(
        path_id: str,
        joints: Sequence[float],
        camera_parameters: Optional[Dict] = None,
        parameters: Optional[Dict] = None
) -> Optional[Dict[str, List[numpy.ndarray]]]:
    """
    Where the line of a new track's sheet should appear in the picture taken at ``joints``.

    Parameters
    ----------
    path_id : str
        The sheet, for example "path_11_T01_solid".
    joints : sequence of float
        The six joint angles in radians when the picture is taken.
    camera_parameters : dict, optional
        The fitted forearm camera (the recorder's ``PuckTracker.camera_parameters``).
    parameters : dict, optional
        Overrides for ``TRACK_LAY_OUT_PARAMETERS``.

    Returns
    -------
    dict or None
        "centre", "left_edge" and "right_edge": lists of int32 pixel polylines (N, 1, 2), split where the line leaves
        the camera model's field; None for a course sheet or when no part of the line is in the picture.
    """

    track_name = track_of_sheet(path_id)
    if track_name is None:
        return None
    settings = dict(TRACK_LAY_OUT_PARAMETERS)
    settings.update(parameters or {})
    content = load_tracks(settings)
    centre_line = track_centre_line_on_table(content["tracks"][track_name], content["lay_out_rule"])
    along = numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(centre_line, axis = 0), axis = 1))])
    centre_line = centre_line[along <= settings["drawn_length_m"]]
    drawn = {}
    for name, points in band_on_table(centre_line, settings["line_width_m"]).items():
        pixels = project_points(points, joints, camera_parameters)
        is_usable = ~numpy.isnan(pixels).any(axis = 1) & (numpy.abs(pixels) < 5_000).all(axis = 1)
        pieces = []
        start = None
        for index, usable in enumerate(list(is_usable) + [False]):
            if usable and start is None:
                start = index
            elif not usable and start is not None:
                if index - start >= 2:
                    pieces.append(numpy.round(pixels[start:index]).astype(numpy.int32).reshape(-1, 1, 2))
                start = None
        drawn[name] = pieces
    if not any(drawn.values()):
        return None
    return drawn



def with_expected_track(
        picture: numpy.ndarray,
        drawn: Dict[str, List[numpy.ndarray]],
        parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """A copy of ``picture`` with the expected line's two edges (and its centre, thinner) drawn on it."""

    settings = dict(TRACK_LAY_OUT_PARAMETERS)
    settings.update(parameters or {})
    shown = picture.copy()
    for name in ("left_edge", "right_edge"):
        cv2.polylines(shown, drawn[name], False, settings["edge_colour_bgr"], settings["edge_thickness_px"], cv2.LINE_AA)
    cv2.polylines(shown, drawn["centre"], False, settings["centre_colour_bgr"], settings["centre_thickness_px"], cv2.LINE_AA)
    return shown
