"""
Two edits of real camera pictures along a sheet's known centre line (dashed-line analysis, robot test of 11.11.26).

    filled_gaps   paints ink over the gaps of the dashed line, so sheet 06 looks like a solid line;
    cut_dashes    paints paper over stretches of a solid line, so sheet 05 looks dashed.

Both need the line's place on the table in the recording (line_geometry.registered_line) and the joints of the
picture; the band to paint is drawn on the table, projected into the picture with the fitted camera model, shifted
by a few pixels so that the line's known ink lies on the ink the picture shows, and painted with soft edges. The
puck and the robot's own parts are never painted over. They are test tools for the question "do the dashes matter?",
and a first proof that a dash-cutting augmentation is feasible on real pictures.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy

from dashed_common import ink_and_paper_masks, pixels_from_table_points
from line_geometry import tangents_and_normals

EDIT_PARAMETERS = {
    # the printed line is 1.0 cm wide; ink is painted 1 mm wider and 3 mm into the dashes at both ends so that no light
    # seam is left, and paper is painted wider still so that no dark rim is left
    "line_width_m": 0.010,
    "ink_band_width_m": 0.011,
    "ink_overlap_points": 3,
    # filled_gaps paints the whole line in one ink colour (True) or only the gaps (False, which leaves faint seams)
    "is_whole_line_painted": True,
    "paper_band_width_m": 0.016,
    # the paper colour of a cut is the median of the paper pixels in a strip this wide on both sides of the band
    "paper_sample_width_m": 0.008,
    # the painted band's edge is blurred by this many pixels (the camera's own edge blur is about 1.5 px)
    "edge_blur_px": 1.5,
    # the puck is kept clear by this many pixels around its red pixels
    "puck_margin_px": 5,
    # the outline of the robot's own parts in the picture (the block, the wrist and the blue covers), read from the
    # pictures of 9 and 11.11.26 with the wrist between 6 and 26 degrees; nothing inside it is painted
    "robot_outline_px": [[190, 480], [205, 405], [270, 390], [275, 315], [390, 315], [395, 390], [455, 405], [575, 440], [575, 480]],
    # the band is shifted by up to this many pixels (in steps of 2) so that the line's known ink, where the robot and the
    # puck do not hide it, covers the picture's ink (the projection was within about 10 px in the checked pictures)
    "align_most_px": 12,
    "align_least_ink_px": 400,
    # used when a picture shows too little ink or paper to measure the colour (B, G, R)
    "default_ink_bgr": [130, 92, 42],
    "default_paper_bgr": [212, 196, 155]
}



def runs_where(is_wanted: numpy.ndarray) -> List[Tuple[int, int]]:
    """Return (first position, last position + 1) of every run of True values."""

    padded = numpy.r_[False, numpy.asarray(is_wanted, dtype = bool), False]
    edges = numpy.flatnonzero(padded[1:] != padded[:-1])
    return [(int(first), int(last)) for first, last in zip(edges[::2], edges[1::2])]



def band_labels(
        line_m: numpy.ndarray,
        runs: List[Tuple[int, int]],
        joints: Sequence[float],
        width_m: float,
        shape: Tuple[int, int] = (480, 640)
) -> numpy.ndarray:
    """
    Return a picture in which the band along every listed stretch of the centre line carries that stretch's number.

    Parameters
    ----------
    line_m : numpy.ndarray
        The centre line in the recording's base frame, one point per millimetre.
    runs : list of tuple
        (first position, last position + 1) of every stretch to draw; stretch k is drawn with the value k + 1.
    joints : sequence of float
        The six joint angles of the picture.
    width_m : float
        The band's width on the table.
    shape : tuple, optional
        The picture's (height, width).

    Returns
    -------
    numpy.ndarray
        int32 picture, 0 outside every band.
    """

    labels = numpy.zeros(shape, numpy.int32)
    _, normal = tangents_and_normals(line_m)
    left = pixels_from_table_points(line_m + 0.5 * width_m * normal, joints)
    right = pixels_from_table_points(line_m - 0.5 * width_m * normal, joints)
    for number, (first, last) in enumerate(runs):
        first, last = max(0, first), min(len(line_m), last)
        if last - first < 2:
            continue
        positions = numpy.arange(first, last)
        outline = numpy.vstack([left[positions], right[positions][::-1]])
        if not numpy.all(numpy.isfinite(outline)):
            continue
        if outline[:, 0].max() < -200 or outline[:, 0].min() > shape[1] + 200 or outline[:, 1].max() < -200 or outline[:, 1].min() > shape[0] + 200:
            continue
        cv2.fillPoly(labels, [numpy.round(numpy.clip(outline, -2000, 3000)).astype(numpy.int32)], number + 1)
    return labels



def picture_shift(
        picture_ink: numpy.ndarray,
        known_ink: numpy.ndarray,
        parameters: Dict
) -> Tuple[int, int]:
    """
    Return the pixel shift (columns, rows) that lays the line's known ink best onto the ink the picture shows.

    Both masks are compared at half size for every shift up to "align_most_px"; (0, 0) when the picture shows too
    little ink to tell.
    """

    if int(picture_ink.sum()) < parameters["align_least_ink_px"] or int(known_ink.sum()) < parameters["align_least_ink_px"]:
        return 0, 0
    small_picture = cv2.resize(picture_ink.astype(numpy.float32), (320, 240), interpolation = cv2.INTER_AREA)
    small_known = cv2.resize(known_ink.astype(numpy.float32), (320, 240), interpolation = cv2.INTER_AREA)
    reach = parameters["align_most_px"] // 2
    padded = cv2.copyMakeBorder(small_picture, reach, reach, reach, reach, cv2.BORDER_CONSTANT, value = 0.0)
    scores = cv2.matchTemplate(padded, small_known, cv2.TM_CCORR)
    row, column = numpy.unravel_index(int(scores.argmax()), scores.shape)
    if row in (0, scores.shape[0] - 1) or column in (0, scores.shape[1] - 1):
        # the best shift lies at the edge of the search: not trusted
        return 0, 0
    return 2 * (int(column) - reach), 2 * (int(row) - reach)



def shifted(
        labels: numpy.ndarray,
        shift: Tuple[int, int]
) -> numpy.ndarray:
    """Return a label picture moved by (columns, rows) pixels, 0 where nothing moves in."""

    if shift == (0, 0):
        return labels
    matrix = numpy.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(labels.astype(numpy.float32), matrix, (labels.shape[1], labels.shape[0]), flags = cv2.INTER_NEAREST, borderValue = 0).astype(numpy.int32)



def painted(
        picture_bgr: numpy.ndarray,
        mask: numpy.ndarray,
        colour_bgr: numpy.ndarray,
        is_protected: numpy.ndarray,
        parameters: Dict
) -> numpy.ndarray:
    """Return the picture with ``mask`` painted in ``colour_bgr`` (one colour, or one per pixel) with soft edges."""

    alpha = cv2.GaussianBlur(mask.astype(numpy.float32), (0, 0), parameters["edge_blur_px"])
    alpha[is_protected] = 0.0
    colour = numpy.broadcast_to(numpy.asarray(colour_bgr, dtype = numpy.float32), picture_bgr.shape)
    result = picture_bgr.astype(numpy.float32) * (1.0 - alpha[:, :, None]) + colour * alpha[:, :, None]
    return numpy.clip(numpy.round(result), 0, 255).astype(numpy.uint8)



def protected_pixels(
        picture_bgr: numpy.ndarray,
        parameters: Dict
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Return the pixels never painted (the puck with a margin, the robot's parts, the table) and the ink and paper masks."""

    is_ink, is_paper, is_puck = ink_and_paper_masks(picture_bgr)
    margin = 2 * parameters["puck_margin_px"] + 1
    puck_wide = cv2.dilate(is_puck.astype(numpy.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))) > 0
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    is_bluish = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 118) & (hsv[:, :, 2] >= 95)
    is_protected = puck_wide | ~is_bluish
    robot = numpy.zeros(picture_bgr.shape[:2], numpy.uint8)
    cv2.fillPoly(robot, [numpy.array(parameters["robot_outline_px"], dtype = numpy.int32)], 255)
    is_protected |= robot > 0
    return is_protected, is_ink, is_paper



def filled_gaps(
        picture_bgr: numpy.ndarray,
        line_m: numpy.ndarray,
        is_ink_point: numpy.ndarray,
        joints: Sequence[float],
        parameters: Optional[Dict] = None
) -> Tuple[numpy.ndarray, Dict]:
    """
    Paint ink over the gaps of a dashed line.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The camera picture.
    line_m : numpy.ndarray
        The sheet's centre line placed in the recording's frame.
    is_ink_point : numpy.ndarray
        Whether each centre-line point lies on a dash.
    joints : sequence of float
        The six joint angles of the picture.
    parameters : dict, optional
        Overrides of EDIT_PARAMETERS.

    Returns
    -------
    tuple of numpy.ndarray and dict
        The edited picture, and "shift_px" (the alignment found) and "painted_px" (the size of the painted bands).
    """

    settings = dict(EDIT_PARAMETERS)
    settings.update(parameters or {})
    is_protected, is_ink, _ = protected_pixels(picture_bgr, settings)
    known_ink = (band_labels(line_m, runs_where(is_ink_point), joints, settings["line_width_m"]) > 0) & ~is_protected
    shift = picture_shift(is_ink, known_ink, settings)
    if settings["is_whole_line_painted"]:
        # the dashes are painted too, in the same colour, so that no seam is left between a dash and a filled gap
        runs = [(0, len(line_m))]
    else:
        overlap = 2 * settings["ink_overlap_points"] + 1
        runs = runs_where(cv2.dilate(numpy.asarray(~is_ink_point, dtype = numpy.uint8), numpy.ones((overlap, 1), numpy.uint8)).ravel() > 0)
    gaps = shifted(band_labels(line_m, runs, joints, settings["ink_band_width_m"]), shift) > 0
    ink_colour = numpy.median(picture_bgr[is_ink], axis = 0) if int(is_ink.sum()) >= 200 else numpy.array(settings["default_ink_bgr"], dtype = float)
    return painted(picture_bgr, gaps, ink_colour, is_protected, settings), {"shift_px": shift, "painted_px": int(gaps.sum())}



def cut_dashes(
        picture_bgr: numpy.ndarray,
        line_m: numpy.ndarray,
        is_gap_point: numpy.ndarray,
        joints: Sequence[float],
        parameters: Optional[Dict] = None
) -> Tuple[numpy.ndarray, Dict]:
    """
    Paint paper over stretches of a solid line, so that it looks dashed.

    Every gap gets the median colour of the paper pixels right beside it, so the paper's shading is kept.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The camera picture.
    line_m : numpy.ndarray
        The sheet's centre line placed in the recording's frame.
    is_gap_point : numpy.ndarray
        Whether each centre-line point is to become a gap.
    joints : sequence of float
        The six joint angles of the picture.
    parameters : dict, optional
        Overrides of EDIT_PARAMETERS.

    Returns
    -------
    tuple of numpy.ndarray and dict
        The edited picture, and "shift_px" and "painted_px".
    """

    settings = dict(EDIT_PARAMETERS)
    settings.update(parameters or {})
    is_protected, is_ink, is_paper = protected_pixels(picture_bgr, settings)
    known_ink = (band_labels(line_m, [(0, len(line_m))], joints, settings["line_width_m"]) > 0) & ~is_protected
    shift = picture_shift(is_ink, known_ink, settings)
    runs = runs_where(is_gap_point)
    bands = shifted(band_labels(line_m, runs, joints, settings["paper_band_width_m"]), shift)
    wide = shifted(band_labels(line_m, runs, joints, settings["paper_band_width_m"] + 2 * settings["paper_sample_width_m"]), shift)
    paper_default = numpy.median(picture_bgr[is_paper], axis = 0) if int(is_paper.sum()) >= 500 else numpy.array(settings["default_paper_bgr"], dtype = float)
    colours = numpy.zeros(picture_bgr.shape, numpy.float32)
    colours[:] = paper_default
    is_beside = (wide > 0) & (bands == 0) & is_paper
    for number in numpy.unique(bands[bands > 0]):
        beside = is_beside & (wide == number)
        colours[bands == number] = numpy.median(picture_bgr[beside], axis = 0) if int(beside.sum()) >= 30 else paper_default
    return painted(picture_bgr, bands > 0, colours, is_protected, settings), {"shift_px": shift, "painted_px": int((bands > 0).sum())}



def dash_pattern(
        point_count: int,
        dash_m: float,
        gap_m: float,
        phase_m: float,
        spacing_m: float = 0.001
) -> numpy.ndarray:
    """Return which centre-line points become gaps for dashes of ``dash_m`` and gaps of ``gap_m``, starting ``phase_m`` into a dash."""

    distance = numpy.arange(point_count) * spacing_m + phase_m
    return (distance % (dash_m + gap_m)) >= dash_m
