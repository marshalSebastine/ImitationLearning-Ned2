"""
Check of the critic's P1 and P10: how well does the centre line traced from the course's photos
(Recording tool/2026-11-06_sheet_measurements.json, line_xy_cm) agree in shape with the centre line the dashed-line
analysis measured from the robot's own pictures (dashed_line/reference_line_05.csv and reference_line_06.csv)?
The two are laid over each other by the best turn and shift (and, as a second try, a turn, shift and mirror), points
matched to the nearest point of the other line, a few rounds. Reads three files. Nothing connects to the robot.

Run:  D:/AI_Project/.venv/Scripts/python.exe traced_against_camera_line.py > traced_against_camera_line.out.txt
"""

import json
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
NOTES = ROOT / "06_deliverables" / "research_notes"
EVIDENCE = NOTES / "Model and training" / "evidence" / "robot_test_2026-11-11"
sheets = json.loads((NOTES / "Recording tool" / "2026-11-06_sheet_measurements.json").read_text(encoding = "utf-8"))


def resampled(points_mm, step_mm = 1.0):
    along = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(points_mm, axis = 0), axis = 1))]
    wanted = numpy.arange(0.0, along[-1], step_mm)
    return numpy.column_stack([numpy.interp(wanted, along, points_mm[:, 0]), numpy.interp(wanted, along, points_mm[:, 1])])


def best_turn_and_shift(source, target):
    """Kabsch: the turn and shift that bring matched ``source`` points onto ``target`` points."""

    source_centre, target_centre = source.mean(axis = 0), target.mean(axis = 0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = numpy.linalg.svd(covariance)
    sign = numpy.sign(numpy.linalg.det(vt.T @ u.T))
    rotation = vt.T @ numpy.diag([1.0, sign]) @ u.T
    return rotation, target_centre - rotation @ source_centre


def laid_over(camera_line, traced_line):
    """Lay the camera's line (shorter: its first centimetres are missing) onto the traced line, by position along the line first."""

    best = None
    for is_mirrored in (False, True):
        for is_reversed in (False, True):
            traced = traced_line[::-1] if is_reversed else traced_line
            traced = traced * numpy.array([1.0, -1.0]) if is_mirrored else traced
            for start in range(0, max(1, len(traced) - len(camera_line) + 1), 2):
                target = traced[start:start + len(camera_line)]
                rotation, shift = best_turn_and_shift(camera_line[:len(target)], target)
                moved = camera_line[:len(target)] @ rotation.T + shift
                error = numpy.linalg.norm(moved - target, axis = 1)
                if best is None or numpy.median(error) < best[0]:
                    best = (float(numpy.median(error)), is_mirrored, is_reversed, start, rotation, shift, traced)
    _, is_mirrored, is_reversed, start, rotation, shift, traced = best
    # refine: match every camera point to its nearest traced point, a few rounds
    for _ in range(10):
        moved = camera_line @ rotation.T + shift
        distances = numpy.linalg.norm(moved[:, None, :] - traced[None, :, :], axis = 2)
        nearest = traced[distances.argmin(axis = 1)]
        rotation, shift = best_turn_and_shift(camera_line, nearest)
    moved = camera_line @ rotation.T + shift
    distances = numpy.linalg.norm(moved[:, None, :] - traced[None, :, :], axis = 2).min(axis = 1)
    return distances, is_mirrored, is_reversed, start


for name in ("05", "06"):
    traced_line = resampled(10.0 * numpy.array(sheets[name]["line_xy_cm"], dtype = float))
    reference = pandas.read_csv(EVIDENCE / "dashed_line" / f"reference_line_{name}.csv")
    camera_line = resampled(10.0 * reference[["x_cm", "y_cm"]].to_numpy(dtype = float))
    distances, is_mirrored, is_reversed, start = laid_over(camera_line, traced_line)
    print(f"sheet {name}: traced from the course photo {len(traced_line)} mm long, measured from the robot's pictures {len(camera_line)} mm long "
          f"(laid on from {start} mm of the traced line, reversed {is_reversed}, mirrored {is_mirrored})")
    print(f"   distance between the two centre lines after the best turn and shift: median {numpy.median(distances):.1f} mm, 90th percentile {numpy.percentile(distances, 90):.1f} mm, "
          f"largest {distances.max():.1f} mm; share of the line within 2 mm {100 * numpy.mean(distances <= 2.0):.0f} %, within 3 mm {100 * numpy.mean(distances <= 3.0):.0f} %")
