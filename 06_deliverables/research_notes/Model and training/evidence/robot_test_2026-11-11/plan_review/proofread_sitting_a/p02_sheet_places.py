"""
Proofreader sitting_a, check 2: the places on the traced centre lines that sitting A's rules name, recomputed.

Sheet 03 from end A (the "first" traced end, SETUP_VISUALS_PARAMETERS["end_keys"]): the start cap, the first bend and its
apex, the 119 mm mark, the second bend. Sheet 04 from end A: the first S-bend and the first hairpin (part A5).
Sheet 01: its length. Two independent ways to find a bend's apex are printed side by side (the tightest curvature of
the smoothed line; the middle of the bend's turn).

Reads the traced lines only. Run:
    D:/AI_Project/.venv/Scripts/python.exe -B p02_sheet_places.py > p02_sheet_places.out.txt
"""

import json
from pathlib import Path

import numpy

NOTES = Path("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes")
SHEETS = json.loads((NOTES / "Recording tool" / "2026-11-06_sheet_measurements.json").read_text(encoding = "utf-8"))
STEP_MM = 0.5


def resampled_mm(points_cm, step_mm = STEP_MM):
    points = 10.0 * numpy.array(points_cm, dtype = float)
    along = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(points, axis = 0), axis = 1))]
    wanted = numpy.arange(0.0, along[-1] + 1e-9, step_mm)
    return numpy.column_stack([numpy.interp(wanted, along, points[:, 0]), numpy.interp(wanted, along, points[:, 1])]), float(along[-1])


def heading_deg(line, baseline_mm):
    half = int(round(baseline_mm / 2 / STEP_MM))
    index = numpy.arange(len(line))
    before, after = numpy.clip(index - half, 0, len(line) - 1), numpy.clip(index + half, 0, len(line) - 1)
    return numpy.degrees(numpy.unwrap(numpy.arctan2(line[after, 1] - line[before, 1], line[after, 0] - line[before, 0])))


def bends(line, baseline_mm = 10.0, rate_deg_per_mm = 0.6, join_mm = 8.0, least_turn_deg = 20.0):
    """Stretches where the smoothed heading changes faster than ``rate_deg_per_mm``; apex two ways."""

    heading = heading_deg(line, baseline_mm)
    rate = numpy.gradient(heading, STEP_MM)
    is_bending = numpy.abs(rate) > rate_deg_per_mm
    stretches, start = [], None
    for index, value in enumerate(is_bending):
        if value and start is None:
            start = index
        if not value and start is not None:
            stretches.append([start, index])
            start = None
    if start is not None:
        stretches.append([start, len(is_bending) - 1])
    joined = []
    for stretch in stretches:
        if joined and (stretch[0] - joined[-1][1]) * STEP_MM < join_mm:
            joined[-1][1] = stretch[1]
        else:
            joined.append(stretch)
    found = []
    for first, last in joined:
        turn = heading[last] - heading[first]
        if abs(turn) < least_turn_deg:
            continue
        tightest = first + int(numpy.argmax(numpy.abs(rate[first:last + 1])))
        half_turn = first + int(numpy.argmin(numpy.abs((heading[first:last + 1] - heading[first]) - turn / 2)))
        radius_mm = 1.0 / numpy.radians(numpy.abs(rate[tightest]))
        found.append({"from_mm": first * STEP_MM, "to_mm": last * STEP_MM, "turn_deg": float(turn), "apex_tightest_mm": tightest * STEP_MM,
                      "apex_half_turn_mm": half_turn * STEP_MM, "tightest_radius_mm": float(radius_mm)})
    return found, heading


def at(line, heading, place_mm):
    index = int(round(place_mm / STEP_MM))
    index = min(max(index, 0), len(line) - 1)
    return line[index], heading[index]


for name in ("01", "03", "04"):
    sheet = SHEETS[name]
    line, length = resampled_mm(sheet["line_xy_cm"])
    print(f"sheet {name}: traced centre line {length:.1f} mm long (cap centre to cap centre; the file says {10 * sheet['line_length_cap_to_cap_cm']:.0f} mm); "
          f"paper {10 * sheet['sheet_length_cm_fitted']:.1f} mm long as fitted; x from the paper's left edge, y from its bottom edge, in mm")
    for end_name, ordered in (("A (first traced end)", line), ("B (last traced end)", line[::-1])):
        for baseline in (10.0, 6.0, 16.0):
            found, heading = bends(ordered, baseline_mm = baseline)
            if baseline == 10.0:
                start, start_heading = at(ordered, heading, 0.0)
                finish, _ = at(ordered, heading, length)
                print(f"   from end {end_name}: start cap centre at x {start[0]:.1f}, y {start[1]:.1f} mm; the line leaves at {heading[int(10 / STEP_MM)]:.0f} degrees; other cap at x {finish[0]:.1f}, y {finish[1]:.1f} mm")
            text = "; ".join(f"{bend['from_mm']:.0f} to {bend['to_mm']:.0f} mm, turn {bend['turn_deg']:+.0f} deg, apex {bend['apex_tightest_mm']:.0f} (tightest, radius {bend['tightest_radius_mm']:.0f} mm) / {bend['apex_half_turn_mm']:.0f} (half the turn)"
                             for bend in found[:6])
            print(f"      bends, heading over {baseline:.0f} mm: {text}")
    print()

# the places sitting A names on sheet 03 from end A
line, length = resampled_mm(SHEETS["03"]["line_xy_cm"])
found, heading = bends(line, baseline_mm = 10.0)
print("sheet 03 from end A, places for the rule of part A4 (x from the paper's left edge, y from its bottom edge; end A is the cap near the top edge):")
for label, place in (("start cap centre", 0.0), ("first bend begins", found[0]["from_mm"]), ("first bend's apex, tightest", found[0]["apex_tightest_mm"]),
                     ("first bend's apex, half the turn", found[0]["apex_half_turn_mm"]), ("the checker's apex", 39.0), ("first bend ends", found[0]["to_mm"]),
                     ("second bend begins", found[1]["from_mm"]), ("window's end, 119 mm", 119.0), ("second bend's apex, tightest", found[1]["apex_tightest_mm"]),
                     ("second bend ends", found[1]["to_mm"])):
    point, direction = at(line, heading, place)
    print(f"   {label:34s} {place:6.1f} mm of line   x {point[0]:6.1f}  y {point[1]:6.1f}   line direction {direction:7.1f} deg   turned {direction - heading[int(10 / STEP_MM)]:+6.1f} deg since the start")
straight = (found[1]["from_mm"] - found[0]["to_mm"])
print(f"   straight leg between the first and the second bend: {straight:.0f} mm")
print(f"   8 cm past the apex: {found[0]['apex_tightest_mm'] + 80:.0f} mm (tightest) or {found[0]['apex_half_turn_mm'] + 80:.0f} mm (half the turn); the plan says 119 mm")
