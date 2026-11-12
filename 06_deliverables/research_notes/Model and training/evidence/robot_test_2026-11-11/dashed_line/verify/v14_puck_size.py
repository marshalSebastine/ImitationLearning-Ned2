"""
Verifier check 14: how wide is the puck? The brief says 4 cm, config.PUCK_PARAMETERS says 2.4 cm (and 2.9 cm tall).

The puck's red silhouette is measured in pixels in the first eight pictures of runs 7 and 8 (widest row of the red
blob) and turned into centimetres with the camera model at half the puck's height and at its top. The line's width is measured the same way on the table plane as a check of the method
(it is printed 1.0 cm wide).

Run:  D:/AI_Project/.venv/Scripts/python.exe v14_puck_size.py > v14_puck_size.out.txt
"""

import cv2
import numpy
import pandas

from verify_common import VERIFY_PARAMETERS, colour_masks, pixels_on_table

from config import PUCK_PARAMETERS  # noqa: E402

JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]
widths, line_widths = [], []
for key in ("run_8_folder", "run_7_folder"):
    folder = VERIFY_PARAMETERS["model_runs_folder"] / VERIFY_PARAMETERS[key]
    steps = pandas.read_csv(folder / "steps.csv")
    for position in range(0, 8):
        picture = cv2.imread(str(folder / steps["frame"].iloc[position]))
        joints = steps[JOINT_NAMES].iloc[position].to_numpy()
        # the puck's silhouette (top face and side): a standing cylinder is as wide as its diameter in every row of the
        # picture where its full width shows, so the widest row of the red blob gives the diameter; the scale is taken
        # at half the puck's height and at its top, which brackets the true distance from the camera
        is_puck = colour_masks(picture)["puck"].astype(numpy.uint8)
        is_puck = cv2.morphologyEx(is_puck, cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))
        count, labels, statistics, _ = cv2.connectedComponentsWithStats(is_puck, connectivity = 8)
        if count <= 1:
            continue
        largest = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
        blob = labels == largest
        row_widths = blob.sum(axis = 1)
        row = int(numpy.argmax(row_widths))
        columns = numpy.flatnonzero(blob[row])
        ends = numpy.array([[columns.min() - 0.5, row], [columns.max() + 0.5, row]], dtype = float)
        on_half = pixels_on_table(ends, joints, 0.5 * PUCK_PARAMETERS["height_m"])
        on_top = pixels_on_table(ends, joints, PUCK_PARAMETERS["height_m"])
        widths.append((100.0 * float(numpy.hypot(*(on_half[1] - on_half[0]))), 100.0 * float(numpy.hypot(*(on_top[1] - on_top[0]))), int(row_widths[row])))
        # the line: the widest ink run in a row 60 px above the puck (the line runs up the picture at the start)
        ink = colour_masks(picture)["ink"]
        check_row = max(0, row - 90)
        runs = numpy.flatnonzero(numpy.diff(numpy.r_[0, ink[check_row].astype(int), 0]))
        if len(runs) >= 2:
            lengths = runs[1::2] - runs[::2]
            best = int(numpy.argmax(lengths))
            ends = numpy.array([[runs[2 * best] - 0.5, check_row], [runs[2 * best + 1] - 0.5, check_row]], dtype = float)
            on_table = pixels_on_table(ends, joints, 0.0)
            line_widths.append(100.0 * float(numpy.hypot(*(on_table[1] - on_table[0]))))
widths = numpy.array(widths)
print(f"puck's widest row, {len(widths)} pictures: {numpy.median(widths[:, 2]):.0f} px (median); that is {numpy.median(widths[:, 0]):.2f} cm at half the puck's height"
      f" (range {widths[:, 0].min():.2f} to {widths[:, 0].max():.2f}) and {numpy.median(widths[:, 1]):.2f} cm at its top; config says {100 * PUCK_PARAMETERS['diameter_m']:.1f} cm wide, {100 * PUCK_PARAMETERS['height_m']:.1f} cm tall; the brief says 4 cm")
print(f"line's width across a picture row 90 px above the puck, {len(line_widths)} pictures: median {numpy.median(line_widths):.2f} cm (range {min(line_widths):.2f} to {max(line_widths):.2f}); the row cuts the line at an angle in some pictures, which makes it wider than 1.0")
