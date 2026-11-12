"""
Robot-session critic: would the six course sheets themselves pass the rule that make_test_tracks.py draws by? Their
traced centre lines (Recording tool/2026-11-06_sheet_measurements.json, traced from the course's photos, in cm on the
paper) are resampled to 1 mm and measured with the generator's own functions: the tightest bend, the smallest gap
between parts of the line that are not neighbours, the length, the distance from the paper's edge. If the sheets fail
the rule, the rule cannot draw tracks "with bends as tight as the course sheets" by changing one number.
Also: how far one constant-bend piece of the generator turns at a given radius. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe course_sheets_under_the_track_rule.py > course_sheets_under_the_track_rule.out.txt
"""

import json
import math
import sys
from pathlib import Path

import numpy

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))

from config import TEST_TRACK_PARAMETERS  # noqa: E402
from make_test_tracks import crossings, smallest_radius_mm  # noqa: E402

MEASUREMENTS = ROOT / "06_deliverables" / "research_notes" / "Recording tool" / "2026-11-06_sheet_measurements.json"
sheets = json.loads(MEASUREMENTS.read_text(encoding = "utf-8"))


def resampled_mm(points_cm: list) -> numpy.ndarray:
    """The traced line as one point per millimetre of its length."""

    points = 10.0 * numpy.array(points_cm, dtype = float)
    along = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(points, axis = 0), axis = 1))]
    wanted = numpy.arange(0.0, along[-1], 1.0)
    return numpy.column_stack([numpy.interp(wanted, along, points[:, 0]), numpy.interp(wanted, along, points[:, 1])])


def smoothed(curve: numpy.ndarray, window: int = 7) -> numpy.ndarray:
    """A running mean over ``window`` millimetres: the tracing is jittery at the 0.1 mm level."""

    kernel = numpy.ones(window) / window
    padded = numpy.pad(curve, ((window // 2, window // 2), (0, 0)), mode = "edge")
    return numpy.column_stack([numpy.convolve(padded[:, 0], kernel, mode = "valid"), numpy.convolve(padded[:, 1], kernel, mode = "valid")])


def smallest_gap_mm(curve: numpy.ndarray, apart_along_mm: float, crossing_positions: list) -> float:
    """The smallest distance between two points of the line that lie more than ``apart_along_mm`` apart along it, away from crossings."""

    near_crossing = numpy.zeros(len(curve), dtype = bool)
    for first, second in crossing_positions:
        for position in (first, second):
            near_crossing[max(0, position - 40):position + 40] = True
    sampled = numpy.arange(0, len(curve), 2)
    points = curve[sampled]
    distances = numpy.linalg.norm(points[:, None, :] - points[None, :, :], axis = 2)
    is_counted = (numpy.abs(sampled[:, None] - sampled[None, :]) > apart_along_mm) & ~near_crossing[sampled][:, None] & ~near_crossing[sampled][None, :]
    return float(distances[is_counted].min()) if is_counted.any() else math.inf


rule = TEST_TRACK_PARAMETERS
print(f"the generator's rule today: tightest bend {rule['smallest_radius_mm']:.0f} mm, smallest gap between parts that are not neighbours {rule['smallest_gap_mm']:.0f} mm "
      f"(parts more than {2 * rule['smallest_gap_mm']:.0f} mm apart along the line), length {rule['length_mm'][0] / 10:.0f} to {rule['length_mm'][1] / 10:.0f} cm, "
      f"centre at least {rule['edge_margin_mm']:.0f} mm from the paper's edge, pieces of constant bend {rule['piece_mm'][0]:.0f} to {rule['piece_mm'][1]:.0f} mm long, "
      f"dashes {rule['dash_mm']:.0f} mm with {rule['gap_mm']:.0f} mm gaps\n")
print("sheet | length cm | tightest bend, mm (generator's own 3-point measure on the traced line, and on the line smoothed over 7 mm) | crossings | smallest gap mm (parts more than 50 mm apart) | would the rule keep it?")
for name, sheet in sheets.items():
    curve = resampled_mm(sheet["line_xy_cm"])
    found = crossings(curve)
    positions = [crossing["positions"] for crossing in found]
    radius_raw = smallest_radius_mm(curve)
    radius_smooth = smallest_radius_mm(smoothed(curve))
    gap = smallest_gap_mm(curve, 2 * rule["smallest_gap_mm"], positions)
    reasons = []
    if radius_smooth < rule["smallest_radius_mm"]:
        reasons.append(f"bend tighter than {rule['smallest_radius_mm']:.0f} mm")
    if gap < rule["smallest_gap_mm"]:
        reasons.append(f"parts closer than {rule['smallest_gap_mm']:.0f} mm")
    if len(found) > 1:
        reasons.append("more than one crossing")
    print(f"  {name}  | {len(curve) / 10:5.1f} | {radius_raw:5.1f} raw, {radius_smooth:5.1f} smoothed | {len(found)} | {gap:5.1f} | {'kept' if not reasons else 'REFUSED: ' + '; '.join(reasons)}")

print("\nhow far one piece of constant bend turns [calculated: piece length / radius]:")
for radius in (35.0, 22.0, 13.0, 10.0, 8.0):
    low, high = rule["piece_mm"]
    print(f"   radius {radius:4.0f} mm: {math.degrees(low / radius):4.0f} to {math.degrees(high / radius):4.0f} degrees; a half turn (hairpin) at this radius leaves its two legs "
          f"{2 * radius:.0f} mm apart, the rule asks for {rule['smallest_gap_mm']:.0f} mm")
