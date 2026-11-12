"""
An objective reading of WHEN the puck left the line in every robot run of 11.11.26, to check the failure steps the
analyst read by eye (common.FAILURE_STEPS) and to redo the before/after split of the view-novelty question.

Per logged picture (640 x 480):
    puck    the largest red-violet region (Lab a* and b* thresholds; the puck is violet in shadow);
    paper   the large bright bluish-white regions together, filled to their convex hull;
    line    pixels at least 55 lightness steps darker than the paper, inside the paper's hull, outside the robot's
            own body (the pixels that hardly change over the day's pictures, widened, plus fixed boxes for the block
            and the arm) and at least 10 pixels away from the puck's red top (its dark side wall is not line);
    gap     the smallest distance between the puck's region and any line pixel, in pixels and in centimetres (the
            puck's own width in the picture is the 4 cm scale).
A puck that sits on the line has the line running out from under it, so the gap is 0 to a few pixels; on the dashed
sheet 06 the gap can reach about half a dash gap. "Off the line" below means a gap of 0.5 cm or more.

The measure is blind to how far the puck's centre is from the line's centre while the two still touch (up to about
2.5 cm), and to line hidden under the robot's own block. It is a check of the analyst's "puck no longer touches the
line", not a centring measure.

Run from this folder (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe puck_on_line.py
Output: puck_on_line.out.txt, puck_on_line_per_picture.csv, figures/own_puck_on_line_check.png
"""

import cv2
import numpy
import pandas

from verify_common import RUN_FOLDERS, VERIFY_FOLDER, Report, labelled_montage, read_picture, run_log, run_picture_path

PUCK_LINE_PARAMETERS = {
    "puck_a_least": 148,
    "puck_b_least": 106,
    "puck_area_least_px": 600,
    "paper_lightness_least": 170,
    "paper_b_most": 128,
    "line_darker_than_paper": 55,
    "line_lightness_most": 160,
    "line_area_least_px": 80,
    "puck_side_wall_px": 10,
    "static_spread_most": 22.0,
    "static_widen_px": 13,
    "block_box": [258, 338, 402, 480],
    "arm_box": [190, 410, 590, 480],
    "puck_diameter_cm": 4.0,
    "off_line_gap_cm": 0.5,
    "off_line_pictures_in_a_row": 3,
    "check_pictures": [(3, 2), (3, 5), (5, 5), (5, 11), (5, 17), (6, 9), (6, 11), (6, 15), (8, 10), (8, 16), (8, 21), (9, 10), (9, 17), (10, 20), (10, 27), (4, 20)]
}



def own_body_mask() -> numpy.ndarray:
    """
    Return the pixels the robot's own body covers: those that hardly change over a third of the day's pictures, widened.

    Returns
    -------
    numpy.ndarray
        (480, 640) bool.
    """

    stack = []
    for run_number in RUN_FOLDERS:
        for step in range(0, len(run_log(run_number)), 3):
            stack.append(cv2.cvtColor(read_picture(run_picture_path(run_number, step)), cv2.COLOR_BGR2GRAY))
    spread = numpy.stack(stack).astype(numpy.float32).std(axis = 0)
    static = (spread < PUCK_LINE_PARAMETERS["static_spread_most"]).astype(numpy.uint8)
    size = PUCK_LINE_PARAMETERS["static_widen_px"]
    body = cv2.dilate(static, numpy.ones((size, size), numpy.uint8)) > 0
    # the block's upper part is often hidden by the puck's shadow or lit differently, so it changes too much to be
    # caught by the spread; the block and the arm are added as fixed boxes (left, top, right, bottom)
    for left, top, right, bottom in (PUCK_LINE_PARAMETERS["block_box"], PUCK_LINE_PARAMETERS["arm_box"]):
        body[top:bottom, left:right] = True
    return body



def largest_region(mask: numpy.ndarray, area_least: int = 1):
    """Return the largest connected region of ``mask`` as a bool picture (None when smaller than ``area_least``)."""

    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask.astype(numpy.uint8), connectivity = 8)
    if count < 2:
        return None
    best = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    if statistics[best, cv2.CC_STAT_AREA] < area_least:
        return None
    return labels == best



def measure_picture(picture_bgr: numpy.ndarray, body: numpy.ndarray) -> dict:
    """
    Measure the gap between the puck and the nearest visible line pixel in one picture.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The logged 640 x 480 picture.
    body : numpy.ndarray
        From ``own_body_mask``.

    Returns
    -------
    dict
        "puck_area_px", "puck_x", "puck_y", "puck_width_px", "gap_px", "gap_cm", "line_px" and the masks "puck",
        "line" (for the check figure); the gap is NaN when the puck or the line is not found.
    """

    settings = PUCK_LINE_PARAMETERS
    lab = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = lab[..., 0], lab[..., 1], lab[..., 2]
    result = {"puck_area_px": 0, "puck_x": numpy.nan, "puck_y": numpy.nan, "puck_width_px": numpy.nan, "gap_px": numpy.nan,
              "gap_cm": numpy.nan, "line_px": 0, "puck": None, "line": None}

    red = (a_channel >= settings["puck_a_least"]) & (b_channel >= settings["puck_b_least"])
    red = cv2.morphologyEx(red.astype(numpy.uint8), cv2.MORPH_CLOSE, numpy.ones((5, 5), numpy.uint8)) > 0
    puck = largest_region(red, settings["puck_area_least_px"])
    if puck is None:
        return result
    rows, columns = numpy.nonzero(puck)
    result.update({"puck_area_px": int(puck.sum()), "puck_x": float(columns.mean()), "puck_y": float(rows.mean()),
                   "puck_width_px": float(columns.max() - columns.min() + 1), "puck": puck})

    bright = (lightness >= settings["paper_lightness_least"]) & (b_channel <= settings["paper_b_most"])
    # the line can cut the paper into several bright regions, so every large one belongs to the paper
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(bright.astype(numpy.uint8), connectivity = 8)
    large = [label for label in range(1, count) if statistics[label, cv2.CC_STAT_AREA] >= 5000]
    if not large:
        return result
    paper = numpy.isin(labels, large)
    hull = cv2.convexHull(numpy.argwhere(paper)[:, ::-1].astype(numpy.int32))
    paper_hull = numpy.zeros(paper.shape, numpy.uint8)
    cv2.fillConvexPoly(paper_hull, hull, 1)

    paper_lightness = float(numpy.median(lightness[paper]))
    dark = lightness <= min(settings["line_lightness_most"], paper_lightness - settings["line_darker_than_paper"])
    # the puck's side wall is dark and not red, so it would count as line: everything within "puck_side_wall_px" of
    # the red top face is left out, and the same number of pixels is taken off the measured distance below
    wall = settings["puck_side_wall_px"]
    puck_wide = cv2.dilate(puck.astype(numpy.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wall + 1, 2 * wall + 1))) > 0
    line = dark & (paper_hull > 0) & ~body & ~puck_wide
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(line.astype(numpy.uint8), connectivity = 8)
    kept = numpy.zeros(line.shape, bool)
    for label in range(1, count):
        if statistics[label, cv2.CC_STAT_AREA] >= settings["line_area_least_px"]:
            kept |= labels == label
    result["line"] = kept
    result["line_px"] = int(kept.sum())
    if not kept.any():
        return result
    # distance of every pixel to the puck's red top face; the smallest over the line pixels, less the side-wall margin
    distance = cv2.distanceTransform((~puck).astype(numpy.uint8), cv2.DIST_L2, 5)
    gap_px = max(0.0, float(distance[kept].min()) - wall - 1.0)
    result["gap_px"] = gap_px
    result["gap_cm"] = gap_px * settings["puck_diameter_cm"] / result["puck_width_px"]
    return result



def first_off_line(gaps_cm: numpy.ndarray) -> float:
    """Return the first step from which the gap stays at or above the limit for the set number of pictures in a row."""

    limit, needed = PUCK_LINE_PARAMETERS["off_line_gap_cm"], PUCK_LINE_PARAMETERS["off_line_pictures_in_a_row"]
    is_off = numpy.nan_to_num(gaps_cm, nan = 0.0) >= limit
    for step in range(len(is_off) - needed + 1):
        if is_off[step:step + needed].all():
            return step
    return numpy.nan



def main() -> None:

    report = Report(VERIFY_FOLDER / "puck_on_line.out.txt")
    body = own_body_mask()
    report.say(f"robot's own body: {int(body.sum())} of {body.size} pixels masked")
    rows = []
    for run_number in RUN_FOLDERS:
        for step in range(len(run_log(run_number))):
            measured = measure_picture(read_picture(run_picture_path(run_number, step)), body)
            rows.append({"run": run_number, "step": step, **{key: value for key, value in measured.items() if key not in ("puck", "line")}})
    table = pandas.DataFrame(rows)
    table.round(3).to_csv(VERIFY_FOLDER / "puck_on_line_per_picture.csv", index = False)

    report.say("")
    report.say("GAP BETWEEN THE PUCK AND THE NEAREST VISIBLE LINE PIXEL, in cm, per step (x = puck or line not found)")
    for run_number in RUN_FOLDERS:
        run_rows = table[table["run"] == run_number]
        cells = " ".join("  x " if numpy.isnan(gap) else f"{gap:4.1f}" for gap in run_rows["gap_cm"])
        report.say(f"run {run_number:2d} ({len(run_rows)} pictures, puck found in {int((run_rows['puck_area_px'] > 0).sum())}, "
                   f"puck width median {run_rows['puck_width_px'].median():.0f} px)")
        for start in range(0, len(run_rows), 20):
            report.say(f"   steps {start:3d}+: " + " ".join("  x " if numpy.isnan(gap) else f"{gap:4.1f}" for gap in run_rows["gap_cm"].to_numpy()[start:start + 20]))

    report.say("")
    report.say(f"FIRST STEP FROM WHICH THE PUCK IS OFF THE LINE (gap of {PUCK_LINE_PARAMETERS['off_line_gap_cm']} cm or more on "
               f"{PUCK_LINE_PARAMETERS['off_line_pictures_in_a_row']} pictures in a row)")
    report.say("run   first step off the line   share of pictures off the line   largest gap cm")
    for run_number in RUN_FOLDERS:
        gaps = table.loc[table["run"] == run_number, "gap_cm"].to_numpy()
        first = first_off_line(gaps)
        report.say(f"{run_number:3d}   {'never' if numpy.isnan(first) else int(first):>23}   {100 * numpy.mean(numpy.nan_to_num(gaps) >= PUCK_LINE_PARAMETERS['off_line_gap_cm']):28.0f} %   {numpy.nanmax(gaps):14.1f}")

    pictures, labels = [], []
    for run_number, step in PUCK_LINE_PARAMETERS["check_pictures"]:
        picture = read_picture(run_picture_path(run_number, step))
        measured = measure_picture(picture, body)
        shown = picture.copy()
        if measured["line"] is not None:
            shown[measured["line"]] = (0, 255, 255)
        if measured["puck"] is not None:
            outline = cv2.morphologyEx(measured["puck"].astype(numpy.uint8), cv2.MORPH_GRADIENT, numpy.ones((3, 3), numpy.uint8)) > 0
            shown[outline] = (0, 255, 0)
        shown[body & (numpy.indices(body.shape).sum(axis = 0) % 4 == 0)] = (255, 0, 255)
        pictures.append(shown)
        labels.append(f"run {run_number} step {step}: gap {measured['gap_cm']:.1f} cm")
    montage = labelled_montage(pictures, labels, columns = 4, cell_width = 480)
    cv2.imencode(".png", montage)[1].tofile(str(VERIFY_FOLDER / "figures" / "own_puck_on_line_check.png"))
    report.say("")
    report.say("written: puck_on_line_per_picture.csv, figures/own_puck_on_line_check.png (yellow = line pixels, green = puck outline, magenta dots = own body)")
    report.save()



if __name__ == "__main__":
    main()
