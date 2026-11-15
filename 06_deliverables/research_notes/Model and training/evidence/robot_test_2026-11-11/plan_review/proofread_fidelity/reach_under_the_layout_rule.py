"""
Proofread (fidelity): can decision 2's lay-out rule be met by the six provided sheets?

The rule in version 2 of the plan: every sheet on one taped start mark, the line leaving straight away from the robot,
started from the end that keeps the whole line within about 42 cm, turned by up to 15 degrees if neither end does.
Calculated from the traced centre lines (2026-11-06_sheet_measurements.json, line_xy_cm). Nothing touches the robot.
Two starts are tried: the start cap as the set-up pictures draw it, (16.4, -1.5) cm (config.py: the ink begins at the
start mark, (15.9, -1.5) cm), and the start cap where the check found it in the recordings, (19.2, -1.5) cm (check
results, orientation coverage, verdict 4: "caps at 19.2 cm where the pictures draw 16.4").
"""

import json
import math
from pathlib import Path

import numpy

MEASUREMENTS_PATH = Path(__file__).resolve().parents[5] / "Recording tool" / "2026-11-06_sheet_measurements.json"
START_CAPS_XY_CM = {"as drawn": numpy.array([16.4, -1.5]), "as recorded": numpy.array([19.2, -1.5])}
FIRST_PIECE_CM = 2.5
TURNS_DEG = [0.0, 15.0, -15.0]
SOFT_LIMIT_CM = 42.0
HARD_LIMIT_CM = 47.5


def laid_line(line_xy_cm: numpy.ndarray, turn_deg: float, start_cap_xy_cm: numpy.ndarray) -> numpy.ndarray:
    """The centre line in the base frame: first point on the start cap, first piece pointing along +x, then turned."""

    steps = numpy.linalg.norm(numpy.diff(line_xy_cm, axis = 0), axis = 1)
    along = numpy.concatenate([[0.0], numpy.cumsum(steps)])
    index = int(numpy.searchsorted(along, FIRST_PIECE_CM))
    first_direction = line_xy_cm[index] - line_xy_cm[0]
    angle = -math.atan2(first_direction[1], first_direction[0]) + math.radians(turn_deg)
    rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return (line_xy_cm - line_xy_cm[0]) @ rotation.T + start_cap_xy_cm


def main() -> None:
    measurements = json.loads(MEASUREMENTS_PATH.read_text(encoding = "utf-8"))
    for start_name, start_cap_xy_cm in START_CAPS_XY_CM.items():
        print(f"start cap {start_name} at {start_cap_xy_cm.tolist()} cm; farthest reach of the centre line from the base axis, cm")
        print("sheet end   | turn 0 | turn +15 | turn -15 | cm of line beyond 42 cm (turn 0) | beyond 47.5 cm (turn 0)")
        for sheet, entry in measurements.items():
            line = numpy.array(entry["line_xy_cm"], dtype = float)
            for end_name, points in (("first", line), ("last", line[::-1])):
                reaches = [float(numpy.linalg.norm(laid_line(points, turn_deg, start_cap_xy_cm), axis = 1).max()) for turn_deg in TURNS_DEG]
                laid = laid_line(points, 0.0, start_cap_xy_cm)
                radius = numpy.linalg.norm(laid, axis = 1)
                steps = numpy.linalg.norm(numpy.diff(laid, axis = 0), axis = 1)
                beyond_soft = float(steps[(radius[1:] > SOFT_LIMIT_CM)].sum())
                beyond_hard = float(steps[(radius[1:] > HARD_LIMIT_CM)].sum())
                print(f"{sheet}    {end_name:5s} | {reaches[0]:6.1f} | {reaches[1]:8.1f} | {reaches[2]:8.1f} | {beyond_soft:6.1f} | {beyond_hard:6.1f}")
        print()


if __name__ == "__main__":
    main()
