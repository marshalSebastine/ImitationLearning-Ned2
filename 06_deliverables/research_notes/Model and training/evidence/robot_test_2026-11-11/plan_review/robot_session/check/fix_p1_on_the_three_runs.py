"""
Check of the critic's FIX for P1: "the distance is taken to sheet 03's traced centre line, laid onto the taped sheet;
a run keeps the line when the puck's centre is never more than 12 mm from the centre line on two pictures in a row,
from the start to 8 cm of line past the bend's apex".

The critic tested its rule with the bend and the 8 cm taken from the puck's own path and the offsets of
stop_and_centring/robot_runs_steps.csv, not with the traced line. Here the fix is tried as it is worded, on the three
sheet-03 runs of 11.11.26 (runs 5, 9, 10), with what the check left behind in step_length/tracks/run_NN.npz: the ink
map of each run (2 mm cells) and the puck's place in every picture (both made with the same camera model).
    1. the traced line (Recording tool/2026-11-06_sheet_measurements.json) is laid onto each run's ink map by the best
       turn, shift and mirror (the middle cells of the ink band are matched to the nearest point of the line);
    2. every picture's puck is given its distance from that line and its place along it (mm from the start cap);
    3. the rule is applied with several windows and limits.
Reads files only. Nothing connects to the robot.

Run:  D:/AI_Project/.venv/Scripts/python.exe fix_p1_on_the_three_runs.py > fix_p1_on_the_three_runs.out.txt
"""

import json
from pathlib import Path

import cv2
import numpy

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
NOTES = ROOT / "06_deliverables" / "research_notes"
EVIDENCE = NOTES / "Model and training" / "evidence" / "robot_test_2026-11-11"
sheet = json.loads((NOTES / "Recording tool" / "2026-11-06_sheet_measurements.json").read_text(encoding = "utf-8"))["03"]
APEX_MM = 39.0          # the first bend's apex along the traced line from the start cap (sheet_03_bends.out.txt)
SECOND_BEND_MM = 104.0  # where the second bend begins


def resampled(points_mm, step_mm = 1.0):
    along = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(points_mm, axis = 0), axis = 1))]
    wanted = numpy.arange(0.0, along[-1], step_mm)
    return numpy.column_stack([numpy.interp(wanted, along, points_mm[:, 0]), numpy.interp(wanted, along, points_mm[:, 1])])


def best_turn_and_shift(source, target):
    source_centre, target_centre = source.mean(axis = 0), target.mean(axis = 0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = numpy.linalg.svd(covariance)
    sign = numpy.sign(numpy.linalg.det(vt.T @ u.T))
    rotation = vt.T @ numpy.diag([1.0, sign]) @ u.T
    return rotation, target_centre - rotation @ source_centre


traced = resampled(10.0 * numpy.array(sheet["line_xy_cm"], dtype = float))   # mm on the paper, the start cap (end A) first

for run in (5, 9, 10):
    data = numpy.load(EVIDENCE / "step_length" / "tracks" / f"run_{run:02d}.npz")
    is_ink = (data["seen_votes"] >= 1) & (data["ink_votes"] >= 0.5 * numpy.maximum(data["seen_votes"], 1))
    # the middle of the ink band: cells at least 3 mm from the band's edge (cells are 2 mm)
    distance_to_edge = cv2.distanceTransform(is_ink.astype(numpy.uint8), cv2.DIST_L2, 5) * 2.0
    rows, columns = numpy.nonzero(distance_to_edge >= 3.0)
    middle = 1000.0 * numpy.column_stack([data["x_m"][rows] + 0.001, data["y_m"][columns] + 0.001])   # mm in the base frame
    puck = 1000.0 * data["puck_xy_m"]
    best = None
    for is_mirrored in (False, True):
        line = traced * numpy.array([1.0, -1.0]) if is_mirrored else traced
        for turn_deg in range(0, 360, 5):
            angle = numpy.radians(turn_deg)
            rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
            # first guess: the line's start cap lies under the puck's first place
            shift = puck[0] - rotation @ line[0]
            for _ in range(25):
                moved = line @ rotation.T + shift
                nearest_index = numpy.linalg.norm(middle[:, None, :] - moved[None, :, :], axis = 2).argmin(axis = 1)
                rotation, shift = best_turn_and_shift(line[nearest_index], middle)
            moved = line @ rotation.T + shift
            gaps = numpy.linalg.norm(middle[:, None, :] - moved[None, :, :], axis = 2).min(axis = 1)
            cost = float(numpy.mean(numpy.minimum(gaps, 20.0)))
            if best is None or cost < best[0]:
                best = (cost, is_mirrored, turn_deg, moved, gaps)
    cost, is_mirrored, turn_deg, moved, gaps = best
    seen_to = int(numpy.linalg.norm(middle[:, None, :] - moved[None, :, :], axis = 2).argmin(axis = 1).max())
    print(f"run {run}: the traced line laid onto the run's ink map (mirrored {is_mirrored}); middle cells of the ink band lie a median "
          f"{numpy.median(gaps):.1f} mm from it (90th percentile {numpy.percentile(gaps, 90):.1f} mm, largest {gaps.max():.1f} mm); the pictures saw the line up to {seen_to} mm from the start cap")
    distances = numpy.linalg.norm(puck[:, None, :] - moved[None, :, :], axis = 2)
    # the puck's place along the line: the nearest line point that is not far ahead of the last one (the line comes near itself later)
    along, offset = [], []
    last = 0
    for index in range(len(puck)):
        if not numpy.all(numpy.isfinite(puck[index])):
            along.append(numpy.nan)
            offset.append(numpy.nan)
            continue
        window = slice(max(0, last - 30), min(len(moved), last + 60))
        local = distances[index, window]
        place = window.start + int(local.argmin())
        along.append(float(place))
        offset.append(float(local.min()))
        last = place
    along, offset = numpy.array(along), numpy.array(offset)
    print("   picture: mm along the line / mm from the centre line")
    print("   " + "  ".join(f"{index}: {along[index]:.0f}/{offset[index]:.0f}" for index in range(min(len(puck), 34))))
    for past_cm in (6.0, 6.5, 8.0, 10.0):
        end_mm = APEX_MM + 10.0 * past_cm
        in_window = [index for index in range(len(puck)) if numpy.isfinite(along[index]) and along[index] <= end_mm and (index == 0 or numpy.nanmax(along[:index + 1]) <= end_mm)]
        last_picture = in_window[-1] if in_window else -1
        results = []
        for limit in (12.0, 17.0):
            over = offset[:last_picture + 1] > limit
            fails = [index for index in range(1, len(over)) if over[index] and over[index - 1]]
            results.append(f"{limit:.0f} mm: {'PASS' if not fails else 'FAIL at picture ' + str(fails[0])}")
        print(f"   window to {past_cm:.1f} cm past the apex (to {end_mm:.0f} mm of line, pictures 0 to {last_picture}; the second bend begins at {SECOND_BEND_MM:.0f} mm): "
              f"{'; '.join(results)}; largest distance in the window {numpy.nanmax(offset[:last_picture + 1]):.1f} mm")
    print()
