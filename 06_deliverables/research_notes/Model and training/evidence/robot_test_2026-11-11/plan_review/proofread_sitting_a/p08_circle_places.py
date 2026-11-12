"""
Proofreader sitting_a, check 8: about 20 places for the grey circles of part A2 on sheet 03's shape.

Each place is given as millimetres of line from end A's cap centre and millimetres to the side (+ = to the left of
the way from end A to end B), and as x (from the paper's left edge) and y (from its bottom edge; end A is at the top)
in the traced frame of 2026-11-06_sheet_measurements.json. Checked for every place: the circle (24 mm) stays on the
paper; the nearest point of the WHOLE centre line is as far away as the place says (no other leg of the line is
nearer); and on one page no two circles overlap. For the lay-out of demonstration 03-01 (the analyst's fit: turn 112.2
degrees, shift 43.47 and 4.25 cm, misfit 0.9 cm) each place also gets its distance from the base axis and the angle
between the line there and the direction away from the base, which is about how far the forearm must be turned to
look along the line.

Reads the traced line only. Run:
    D:/AI_Project/.venv/Scripts/python.exe -B p08_circle_places.py > p08_circle_places.out.txt
"""

import json
import math
from pathlib import Path

import numpy

NOTES = Path("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes")
SHEET = json.loads((NOTES / "Recording tool" / "2026-11-06_sheet_measurements.json").read_text(encoding = "utf-8"))["03"]
STEP_MM = 0.5
PAPER_MM = (210.0, 10.0 * SHEET["sheet_length_cm_fitted"])
LAYOUT_03_01 = {"turn_deg": 112.2, "shift_cm": (43.47, 4.25)}

points = 10.0 * numpy.array(SHEET["line_xy_cm"], dtype = float)
along = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(points, axis = 0), axis = 1))]
wanted = numpy.arange(0.0, along[-1] + 1e-9, STEP_MM)
line = numpy.column_stack([numpy.interp(wanted, along, points[:, 0]), numpy.interp(wanted, along, points[:, 1])])
LENGTH = float(along[-1])


def direction_at(place_mm, baseline_mm = 16.0):
    index = min(int(round(place_mm / STEP_MM)), len(line) - 1)
    half = int(round(baseline_mm / 2 / STEP_MM))
    before, after = line[max(index - half, 0)], line[min(index + half, len(line) - 1)]
    tangent = (after - before) / numpy.linalg.norm(after - before)
    return tangent, numpy.array([-tangent[1], tangent[0]])   # the tangent and its left-hand normal


def place(along_mm, side_mm):
    index = min(int(round(along_mm / STEP_MM)), len(line) - 1)
    tangent, left = direction_at(along_mm)
    if along_mm > LENGTH:     # beyond end B, straight on
        centre = line[-1] + (along_mm - LENGTH) * tangent + side_mm * left
    else:
        centre = line[index] + side_mm * left
    nearest = float(numpy.linalg.norm(line - centre, axis = 1).min())
    nearest_at = float(numpy.linalg.norm(line - centre, axis = 1).argmin() * STEP_MM)
    return centre, nearest, nearest_at, tangent


def on_table(point_mm, tangent):
    angle = math.radians(LAYOUT_03_01["turn_deg"])
    rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    table = rotation @ (point_mm / 10.0) + numpy.array(LAYOUT_03_01["shift_cm"])
    line_direction = math.degrees(math.atan2(*(rotation @ tangent)[::-1]))
    radial = math.degrees(math.atan2(table[1], table[0]))
    return table, float(numpy.hypot(*table)), (line_direction - radial + 180.0) % 360.0 - 180.0


# page, number, mm of line, mm to the side, what it tests
PLACES = [
    ("1", 1, 0.0, 7.0, "end A's cap, 7 mm aside (an end place)"),
    ("1", 2, 39.0, 12.0, "first bend's apex, inside, at the 12 mm limit"),
    ("1", 3, 39.0, -17.0, "first bend's apex, outside, lost (17 mm)"),
    ("1", 4, 72.0, 0.0, "after the first bend, on the line"),
    ("1", 5, 98.0, 7.0, "straight leg, 7 mm beside"),
    ("1", 6, 119.0, -12.0, "the window's end (119 mm), at the 12 mm limit"),
    ("1", 7, 160.0, 0.0, "second bend, on the line"),
    ("1", 8, 246.0, 0.0, "the hairpin's apex (radius 6 mm), on the line"),
    ("1", 9, 346.0, 12.0, "fourth bend's apex, inside, at the 12 mm limit"),
    ("1", 10, LENGTH, 0.0, "the line's end B, centred on the cap"),
    ("2", 11, 0.0, 0.0, "the start cap of end A, where the puck starts"),
    ("2", 12, 39.0, 0.0, "first bend's apex, on the line"),
    ("2", 13, 70.0, -7.0, "end of the first bend, outside, 7 mm"),
    ("2", 14, 90.0, 17.0, "straight leg, lost (17 mm)"),
    ("2", 15, 119.0, 0.0, "the window's end (119 mm), on the line"),
    ("2", 16, 205.0, -12.0, "leg before the hairpin, at the 12 mm limit"),
    ("2", 17, 246.0, 17.0, "the hairpin's apex, outside, lost (17 mm)"),
    ("2", 18, 300.0, 7.0, "leg after the hairpin, 7 mm beside"),
    ("2", 19, LENGTH - 5.0, 0.0, "the line's end B, 5 mm short"),
    ("2", 20, 440.0, -17.0, "last leg, lost (17 mm)"),
    ("3", 21, LENGTH + 11.0, 0.0, "the line's end B, 11 mm straight beyond (the rim still hides the end)"),
]

print(f"sheet 03: traced line {LENGTH:.1f} mm; paper {PAPER_MM[0]:.0f} x {PAPER_MM[1]:.1f} mm; x from the left edge, y from the bottom edge (end A at the top)")
print("page  no  mm of line  mm aside      x      y   nearest line (mm, at mm of line)  edge mm   on the table for lay-out 03-01: x, y cm, reach cm, line against the radial deg   what")
rows = []
for page, number, along_mm, side_mm, what in PLACES:
    centre, nearest, nearest_at, tangent = place(along_mm, side_mm)
    edge = float(min(centre[0], PAPER_MM[0] - centre[0], centre[1], PAPER_MM[1] - centre[1]))
    table, reach, turned = on_table(centre, tangent)
    wanted_distance = abs(side_mm) if along_mm <= LENGTH else math.hypot(along_mm - LENGTH, side_mm)
    flag = "" if abs(nearest - wanted_distance) < 0.6 else f"  <-- NEAREST LINE IS {nearest:.1f} mm AWAY, NOT {wanted_distance:.1f}"
    flag += "" if edge >= 14.0 else "  <-- CIRCLE NEAR THE PAPER'S EDGE"
    rows.append((page, number, centre))
    print(f"  {page}   {number:2d}     {along_mm:6.1f}    {side_mm:+5.0f}   {centre[0]:6.1f} {centre[1]:6.1f}        {nearest:5.1f} at {nearest_at:5.1f}             {edge:5.1f}        {table[0]:5.1f} {table[1]:6.1f}   {reach:5.1f}   {turned:+7.0f}        {what}{flag}")
for page in ("1", "2", "3"):
    centres = [(number, centre) for row_page, number, centre in rows if row_page == page]
    closest = min(((float(numpy.linalg.norm(a - b)), i, j) for index, (i, a) in enumerate(centres) for j, b in centres[index + 1:]), default = (math.inf, 0, 0))
    print(f"page {page}: {len(centres)} circles; the two nearest are {closest[1]} and {closest[2]}, {closest[0]:.1f} mm apart (24 mm = touching)")
