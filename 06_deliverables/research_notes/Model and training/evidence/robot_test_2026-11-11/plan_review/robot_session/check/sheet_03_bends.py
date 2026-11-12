"""
Check of the critic's P1: where are the bends of sheet 03 along its traced centre line, and how long is the leg after
the first bend? Reads the traced line only (Recording tool/2026-11-06_sheet_measurements.json). Nothing connects to
the robot.

Run:  D:/AI_Project/.venv/Scripts/python.exe sheet_03_bends.py > sheet_03_bends.out.txt
"""

import json
import math
from pathlib import Path

import numpy

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
MEASUREMENTS = ROOT / "06_deliverables" / "research_notes" / "Recording tool" / "2026-11-06_sheet_measurements.json"
sheets = json.loads(MEASUREMENTS.read_text(encoding = "utf-8"))


def resampled_mm(points_cm):
    points = 10.0 * numpy.array(points_cm, dtype = float)
    along = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(points, axis = 0), axis = 1))]
    wanted = numpy.arange(0.0, along[-1], 1.0)
    return numpy.column_stack([numpy.interp(wanted, along, points[:, 0]), numpy.interp(wanted, along, points[:, 1])])


for name in ("03", "04", "05", "06"):
    curve = resampled_mm(sheets[name]["line_xy_cm"])
    for label, line in (("from the first traced end", curve), ("from the last traced end", curve[::-1])):
        # heading over a baseline of 10 mm, and its change over 10 mm of line
        heading = numpy.degrees(numpy.arctan2(line[10:, 1] - line[:-10, 1], line[10:, 0] - line[:-10, 0]))
        heading = numpy.degrees(numpy.unwrap(numpy.radians(heading)))
        turned = heading - heading[0]
        # a bend: the stretch where the heading changes by more than 3 degrees per 5 mm
        rate = numpy.abs(heading[5:] - heading[:-5])
        is_bending = rate > 3.0
        stretches = []
        start = None
        for index, value in enumerate(is_bending):
            if value and start is None:
                start = index
            if not value and start is not None:
                stretches.append((start, index))
                start = None
        if start is not None:
            stretches.append((start, len(is_bending)))
        # join stretches closer than 8 mm
        joined = []
        for stretch in stretches:
            if joined and stretch[0] - joined[-1][1] < 8:
                joined[-1] = (joined[-1][0], stretch[1])
            else:
                joined.append(stretch)
        text = []
        for first, last in joined:
            total = heading[min(last + 5, len(heading) - 1)] - heading[first]
            if abs(total) < 20.0:
                continue
            apex = first + int(numpy.argmax(rate[first:last]))
            text.append(f"bend from {first + 5} to {last + 5} mm of line (apex near {apex + 7} mm), turning {total:+.0f} degrees")
        print(f"sheet {name}, {label}: length {len(line)} mm")
        for entry in text[:6]:
            print("     ", entry)
    print()
