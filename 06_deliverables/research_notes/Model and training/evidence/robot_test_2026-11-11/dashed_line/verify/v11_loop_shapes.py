"""
Verifier check 11: the two loops' size and turning from the course-photo centre lines (my own numbers).

Run:  D:/AI_Project/.venv/Scripts/python.exe v11_loop_shapes.py > v11_loop_shapes.out.txt
"""

import numpy

from verify_common import crossing_indices, sheet_line

for sheet, start_from in (("05", "last end (B, as run 7)"), ("06", "first end (A, as run 8)")):
    line = 100.0 * sheet_line(sheet)
    if sheet == "05":
        line = line[::-1].copy()
    first, second = crossing_indices(line / 100.0)
    loop = line[first:second + 1]
    area = 0.5 * abs(numpy.sum(loop[:-1, 0] * loop[1:, 1] - loop[1:, 0] * loop[:-1, 1]) + loop[-1, 0] * loop[0, 1] - loop[0, 0] * loop[-1, 1])
    centred = loop - loop.mean(axis = 0)
    _, _, axes = numpy.linalg.svd(centred, full_matrices = False)
    along = centred @ axes[0]
    across = centred @ axes[1]
    tangent = numpy.gradient(line, axis = 0)
    heading = numpy.unwrap(numpy.arctan2(tangent[:, 1], tangent[:, 0]))
    turn = numpy.degrees(heading[second] - heading[first])
    # radius from three points 1 cm apart, after smoothing over 1.5 cm
    window = 15
    kernel = numpy.ones(window) / window
    padded = numpy.pad(line, ((7, 7), (0, 0)), mode = "edge")
    smooth = numpy.column_stack([numpy.convolve(padded[:, 0], kernel, mode = "valid"), numpy.convolve(padded[:, 1], kernel, mode = "valid")])
    a, b, c = smooth[:-20], smooth[10:-10], smooth[20:]
    side = numpy.hypot(*(b - a).T) * numpy.hypot(*(c - b).T) * numpy.hypot(*(c - a).T)
    twice_area = numpy.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    radius = side / numpy.maximum(2.0 * twice_area, 1e-12)
    inside = numpy.arange(len(radius)) + 10
    is_loop = (inside >= first) & (inside <= second)
    tightest = inside[is_loop][numpy.argmin(radius[is_loop])]
    print(f"sheet {sheet} from its {start_from}: line {0.1 * (len(line) - 1):.1f} cm; start to crossing {0.1 * first:.1f} cm; loop {0.1 * (second - first):.1f} cm long;"
          f" crossing to the far end {0.1 * (len(line) - 1 - second):.1f} cm")
    print(f"   loop size along its long axis {along.max() - along.min():.1f} cm, across {across.max() - across.min():.1f} cm; area inside {area:.0f} cm2;"
          f" turning through the loop {turn:+.0f} deg (+ = to the left in the sheet-photo frame; the photo may be mirrored against the table)")
    print(f"   tightest bend in the loop: radius {radius[is_loop].min():.1f} cm, {0.1 * (tightest - first):.1f} cm after the first pass of the crossing;"
          f" {0.1 * numpy.sum(radius[is_loop] < 3.0):.1f} cm of the loop tighter than 3 cm radius")
