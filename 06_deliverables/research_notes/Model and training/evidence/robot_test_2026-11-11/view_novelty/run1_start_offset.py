"""
Run 1 was said to start with the tip about 1.5 cm left of the line. How far was the puck beside the line in the
pictures, and did the demonstrations show such a start?

The straight sheet 01 makes the measurement simple: in a picture the line runs up from the puck, so the sideways
distance between the puck's centre and the line just beyond the puck can be read in pixels and turned into centimetres
with the puck's own width (4 cm). The measure is checked on the demonstrations of sheet 01, whose offset starts are
known: OL1 and OL1.5 (puck 1 and 1.5 cm left of the line), OR1 and OR1.5 (right), STD (centred).

Measured for the first pictures of every sheet-01 demonstration, and for every picture of the robot runs 1 and 2
(sheet 01). Positive = the puck sits LEFT of the line (the line is to the right of the puck's centre).

Nothing connects to the robot. Run from this folder (a few seconds):
    D:/AI_Project/.venv/Scripts/python.exe run1_start_offset.py
Output: run1_start_offset.out.txt, run1_start_offset_per_picture.csv
"""

from typing import Dict, Optional

import cv2
import numpy
import pandas

from common import EVIDENCE_FOLDER, VIEW_NOVELTY_PARAMETERS, OutputLog, run_picture_paths, run_steps, training_frames
from config import PUCK_PARAMETERS

OFFSET_PARAMETERS = {
    # the line in the camera's colours (OpenCV hue 0 to 180), read off sample pixels: a dull blue, darker than the paper
    # (saturation 130 to 185); the puck's shadow on the paper is the same hue but paler (saturation 95 to 115), so the
    # saturation limit sits between the two
    "line_hue": [92, 114],
    "line_saturation_at_least": 125,
    "line_value": [50, 180],
    # the band beyond the puck in which the line is looked for, in puck widths above the puck's top edge and to each side
    "band_above_widths": [0.10, 0.70],
    "band_half_width_widths": 1.2,
    "least_line_pixels": 40,
    "demonstration_rows": [1, 3]
}



def puck_beside_line_cm(
        picture_bgr: numpy.ndarray,
        parameters: Dict
) -> Optional[float]:
    """
    Measure how far the puck's centre sits beside the line that runs up from it, in centimetres.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The camera picture.
    parameters : dict
        ``OFFSET_PARAMETERS``.

    Returns
    -------
    float or None
        Positive when the puck is left of the line; None when no puck or no line is found beyond it.
    """

    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    is_red = ((hue < 10) | (hue > 165)) & (saturation >= 90) & (value >= 50)
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(is_red.astype(numpy.uint8))
    if count < 2:
        return None
    largest = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    if statistics[largest, cv2.CC_STAT_AREA] < 400:
        return None
    left, top, width = statistics[largest, cv2.CC_STAT_LEFT], statistics[largest, cv2.CC_STAT_TOP], statistics[largest, cv2.CC_STAT_WIDTH]
    centre_u = left + width / 2.0
    low, high = parameters["band_above_widths"]
    first_row, last_row = int(max(0, top - high * width)), int(max(0, top - low * width))
    first_column = int(max(0, centre_u - parameters["band_half_width_widths"] * width))
    last_column = int(min(hue.shape[1], centre_u + parameters["band_half_width_widths"] * width))
    is_line = ((hue >= parameters["line_hue"][0]) & (hue <= parameters["line_hue"][1]) & (saturation >= parameters["line_saturation_at_least"])
               & (value >= parameters["line_value"][0]) & (value <= parameters["line_value"][1]))
    band = is_line[first_row:last_row, first_column:last_column]
    if band.sum() < parameters["least_line_pixels"]:
        return None
    line_u = first_column + float(numpy.nonzero(band)[1].mean())
    pixels_per_cm = width / (100 * PUCK_PARAMETERS["diameter_m"])
    return float((line_u - centre_u) / pixels_per_cm)



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(OFFSET_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "run1_start_offset.out.txt")
    frames = training_frames()
    first_row, last_row = settings["demonstration_rows"]
    demonstrations = frames[(frames["sheet"] == "path_01") & frames["row_in_episode"].between(first_row, last_row)].copy()
    demonstrations["offset_cm"] = [puck_beside_line_cm(cv2.imread(str(VIEW_NOVELTY_PARAMETERS["raw_folder"] / path)), settings) for path in demonstrations["frame_path"]]

    log.say("THE PUCK BESIDE THE LINE on sheet 01 (cm, measured in the picture; positive = the puck sits LEFT of the line)")
    log.say("")
    log.say(f"Check of the measure on the demonstrations' first pictures (rows {first_row} to {last_row}); the plan's offset is what the demonstrator was asked to set up")
    log.say("kind of start   plan's offset cm   episodes   measured: median   [smallest, largest]")
    planned = {"STD": 0.0, "OL1": 1.0, "OL1.5": 1.5, "OR1": -1.0, "OR1.5": -1.5, "TL15": 0.0, "TR15": 0.0, "NE": 0.0}
    for kind, group in demonstrations.groupby(demonstrations["start_kind"].fillna("not in the plan")):
        values = group["offset_cm"].dropna()
        if values.empty:
            continue
        log.say(f"{kind:15s} {planned.get(kind, float('nan')):+16.1f}   {group['episode_id'].nunique():8d}   {values.median():+16.2f}   [{values.min():+.2f}, {values.max():+.2f}]")

    rows = []
    for run_number in (1, 2):
        steps = run_steps(run_number)
        for step, path in enumerate(run_picture_paths(run_number)):
            rows.append({"run": run_number, "step": step, "offset_cm": puck_beside_line_cm(cv2.imread(str(path)), settings),
                         "tip_y_cm": 100 * steps.iloc[step]["tip_y"], "model_angle_deg": steps.iloc[step]["model_angle_deg"]})
    robot = pandas.DataFrame(rows)
    log.say("")
    log.say("Robot runs on sheet 01 (run 1: model A, said to start 1.5 cm left; run 2: model E, puck centred)")
    log.say("run  step 0   step 1   step 2 | steps 3 to 9: median | steps 10 to 19: median | largest to the left, to the right (step)")
    for run_number in (1, 2):
        part = robot[robot["run"] == run_number]
        seen = part.dropna(subset = ["offset_cm"])
        first = [part.loc[part["step"] == step, "offset_cm"].iloc[0] for step in (0, 1, 2)]
        log.say(f"{run_number:3d}  " + "  ".join(f"{value:+6.2f}" if value is not None and numpy.isfinite(value) else "  none" for value in first)
                + f" | {seen[seen['step'].between(3, 9)]['offset_cm'].median():+20.2f} | {seen[seen['step'].between(10, 19)]['offset_cm'].median():+22.2f} | "
                f"{seen['offset_cm'].max():+.2f} (step {int(seen.loc[seen['offset_cm'].idxmax(), 'step'])}), {seen['offset_cm'].min():+.2f} (step {int(seen.loc[seen['offset_cm'].idxmin(), 'step'])})")
    log.say("")
    log.say("Every step of runs 1 and 2 (offset cm; the direction the model asked, degrees, positive = to the left)")
    for run_number in (1, 2):
        part = robot[robot["run"] == run_number]
        log.say(f"run {run_number}: " + "  ".join(f"{int(row['step'])}:{row['offset_cm']:+.1f}({row['model_angle_deg']:+.0f})" if pandas.notna(row["offset_cm"]) else f"{int(row['step'])}:none"
                                                 for _, row in part.iterrows()))

    robot.round(3).to_csv(EVIDENCE_FOLDER / "run1_start_offset_per_picture.csv", index = False)
    log.say("")
    log.say("written: run1_start_offset_per_picture.csv")
    log.save()



if __name__ == "__main__":
    main()
