"""
Check of the critic's FIX for P1, second part: block A's runs are SHORT (ended a little after the first bend). Can the
traced centre line of sheet 03 still be laid onto such a run from its own pictures? The three sheet-03 runs of
11.11.26 are cut after the picture at which a short run would have ended (the puck 8 cm past the bend's apex by the
whole-run measure: run 5 picture 26, run 9 picture 24, run 10 picture 17), the ink map is built from those pictures
only (the step-length analyst's own function), the traced line is laid on, and the puck's distances are compared with
those from the whole run (fix_p1_on_the_three_runs.out.txt). Reads pictures and logs only. Nothing connects to the
robot.

Run:  D:/AI_Project/.venv/Scripts/python.exe fix_p1_on_short_runs.py > fix_p1_on_short_runs.out.txt
"""

import json
import sys
from pathlib import Path

import cv2
import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
NOTES = ROOT / "06_deliverables" / "research_notes"
EVIDENCE = NOTES / "Model and training" / "evidence" / "robot_test_2026-11-11"
sys.path.insert(0, str(EVIDENCE / "step_length"))
import line_and_puck  # noqa: E402

RUNS = {5: ("2026-11-11_173549_E_sheet03_run1", 26), 9: ("2026-11-11_175203_E_sheet03_run2_no_hoodie", 24), 10: ("2026-11-11_175544_B_team5_sheet03", 17)}
sheet = json.loads((NOTES / "Recording tool" / "2026-11-06_sheet_measurements.json").read_text(encoding = "utf-8"))["03"]
APEX_MM = 39.0


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


def laid_on(traced, middle, first_puck):
    best = None
    for turn_deg in range(0, 360, 5):
        angle = numpy.radians(turn_deg)
        rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
        shift = first_puck - rotation @ traced[0]
        for _ in range(25):
            moved = traced @ rotation.T + shift
            nearest_index = numpy.linalg.norm(middle[:, None, :] - moved[None, :, :], axis = 2).argmin(axis = 1)
            rotation, shift = best_turn_and_shift(traced[nearest_index], middle)
        moved = traced @ rotation.T + shift
        gaps = numpy.linalg.norm(middle[:, None, :] - moved[None, :, :], axis = 2).min(axis = 1)
        cost = float(numpy.mean(numpy.minimum(gaps, 20.0)))
        if best is None or cost < best[0]:
            best = (cost, moved, gaps)
    return best[1], best[2]


def along_and_offset(puck, moved):
    distances = numpy.linalg.norm(puck[:, None, :] - moved[None, :, :], axis = 2)
    along, offset, last = [], [], 0
    for index in range(len(puck)):
        window = slice(max(0, last - 30), min(len(moved), last + 60))
        place = window.start + int(distances[index, window].argmin())
        along.append(float(place))
        offset.append(float(distances[index, window].min()))
        last = place
    return numpy.array(along), numpy.array(offset)


traced = resampled(10.0 * numpy.array(sheet["line_xy_cm"], dtype = float))
for run, (folder_name, last_picture) in RUNS.items():
    folder = Path("D:/AI_Project/03_data/model_runs") / folder_name
    steps = pandas.read_csv(folder / "steps.csv")
    joints = steps[[f"joint_{number}" for number in range(1, 7)]].to_numpy(dtype = float)
    paths = [folder / frame for frame in steps["frame"]]
    results = {}
    for name, count in (("the whole run", len(paths)), (f"pictures 0 to {last_picture} only", last_picture + 1)):
        track = line_and_puck.line_map_and_puck_track(paths[:count], joints[:count])
        is_ink = (track["seen_votes"] >= 1) & (track["ink_votes"] >= 0.5 * numpy.maximum(track["seen_votes"], 1))
        distance_to_edge = cv2.distanceTransform(is_ink.astype(numpy.uint8), cv2.DIST_L2, 5) * 2.0
        rows, columns = numpy.nonzero(distance_to_edge >= 3.0)
        middle = 1000.0 * numpy.column_stack([track["x_m"][rows] + 0.001, track["y_m"][columns] + 0.001])
        puck = 1000.0 * track["puck_xy_m"][:last_picture + 1]
        moved, gaps = laid_on(traced, middle, puck[0])
        along, offset = along_and_offset(puck, moved)
        seen_to = int(numpy.linalg.norm(middle[:, None, :] - moved[None, :, :], axis = 2).argmin(axis = 1).max())
        results[name] = (along, offset)
        over = offset > 12.0
        fails = [index for index in range(1, len(over)) if over[index] and over[index - 1] and along[index] <= APEX_MM + 80.0]
        print(f"run {run}, line map from {name} ({len(middle)} middle cells, line seen up to {seen_to} mm from the start cap): middle cells a median {numpy.median(gaps):.1f} mm "
              f"from the laid-on line (90th percentile {numpy.percentile(gaps, 90):.1f}); rule (12 mm twice in a row, to 8 cm past the apex): "
              f"{'PASS' if not fails else 'FAIL at picture ' + str(fails[0])}; largest distance {offset[along <= APEX_MM + 80.0].max():.1f} mm")
    whole, short = results["the whole run"], results[f"pictures 0 to {last_picture} only"]
    difference = numpy.abs(whole[1] - short[1])
    print(f"   the puck's distance from the line, short map against whole-run map: median difference {numpy.median(difference):.1f} mm, largest {difference.max():.1f} mm; "
          f"place along the line: largest difference {numpy.abs(whole[0] - short[0]).max():.0f} mm\n")
