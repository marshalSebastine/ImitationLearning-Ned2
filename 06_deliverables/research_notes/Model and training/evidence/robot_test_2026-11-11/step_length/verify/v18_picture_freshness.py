"""
Verifier: could a STALE picture (the decision made on the view before the last move) explain the overshoot at the
bend instead of the step length? Test: the turn of the view between consecutive logged pictures, measured from the
pictures alone (a rigid alignment of picture i - 1 onto picture i, upper part of the picture only, where the sheet
is and the robot's own parts are not), against the change of the logged picture heading
    from decision i - 1 to decision i      (what a fresh picture must show), and
    from decision i - 2 to decision i - 1  (what a picture that is one move late would show).
Reads the logged pictures and steps.csv of the robot runs only.

Run from this folder:  python v18_picture_freshness.py > v18_picture_freshness.out.txt
"""

import hashlib
import math
import os

import cv2
import numpy
import pandas

cv2.setNumThreads(2)
RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = {5: "2026-11-11_173549_E_sheet03_run1", 9: "2026-11-11_175203_E_sheet03_run2_no_hoodie", 10: "2026-11-11_175544_B_team5_sheet03",
               7: "2026-11-11_174448_E_sheet05_run1", 4: "2026-11-11_173036_E_sheet02_run2"}


def prepared(path):
    picture = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(picture, (320, 240), interpolation = cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), 2.0).astype(numpy.float32) / 255.0


def view_turn_deg(previous, current):
    """The rotation (degrees) of the rigid motion that lays ``previous`` onto ``current``, upper two thirds only."""
    mask = numpy.zeros(previous.shape, numpy.uint8)
    mask[:150, :] = 255
    warp = numpy.eye(2, 3, dtype = numpy.float32)
    try:
        _, warp = cv2.findTransformECC(current, previous, warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 5)
    except cv2.error:
        return math.nan
    return math.degrees(math.atan2(warp[1, 0], warp[0, 0]))


for number, folder in RUN_FOLDERS.items():
    steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
    heading = steps["picture_heading_deg"].ffill().to_numpy()
    pictures = [prepared(os.path.join(RUNS, folder, name)) for name in steps["frame"]]
    digests = [hashlib.md5(open(os.path.join(RUNS, folder, name), "rb").read()).hexdigest() for name in steps["frame"]]
    same = sum(1 for index in range(1, len(digests)) if digests[index] == digests[index - 1])
    turns = numpy.array([math.nan] + [view_turn_deg(pictures[index - 1], pictures[index]) for index in range(1, len(pictures))])
    heading_change = numpy.r_[math.nan, numpy.diff(heading)]
    late_change = numpy.r_[math.nan, heading_change[:-1]]
    valid = numpy.isfinite(turns) & numpy.isfinite(heading_change) & numpy.isfinite(late_change)
    # the sign of the view's turn against the heading's is fixed by the camera; use the size of the correlation
    fresh = numpy.corrcoef(turns[valid], heading_change[valid])[0, 1]
    late = numpy.corrcoef(turns[valid], late_change[valid])[0, 1]
    early = numpy.corrcoef(turns[valid][:-1], heading_change[valid][1:])[0, 1]
    slope = numpy.polyfit(heading_change[valid], turns[valid], 1)[0]
    residual_fresh = numpy.abs(turns[valid] - slope * heading_change[valid])
    residual_late = numpy.abs(turns[valid] - slope * late_change[valid])
    print(f"run {number}: {len(steps)} pictures, identical to the one before: {same}; aligned pairs {int(valid.sum())}")
    print(f"   correlation of the view's turn with the heading change of the SAME move {fresh:+.3f}; of the move BEFORE (a picture one move late) {late:+.3f}; of the move AFTER {early:+.3f}")
    print(f"   view turn per degree of heading change {slope:+.2f}; median misfit: same move {numpy.median(residual_fresh):.2f} degrees, one move late {numpy.median(residual_late):.2f} degrees")
    shown = min(12, len(steps))
    print("   first decisions: view turn " + ", ".join(f"{value:+.1f}" for value in turns[1:shown]) + " | heading change " + ", ".join(f"{value:+.1f}" for value in heading_change[1:shown]))
