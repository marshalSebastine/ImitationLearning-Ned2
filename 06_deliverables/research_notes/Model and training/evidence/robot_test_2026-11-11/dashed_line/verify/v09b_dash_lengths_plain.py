"""
Verifier check 9b: dash and gap lengths of sheet 06 again, without widening the ink (v09 widened the ink map by one
2 mm cell on every side, which makes every dash about 4 mm longer and every gap 4 mm shorter). Here the ink pixels of
run 8's little-twisted pictures are put on the table at 1 mm cells and read along the placed centre line; a line
point counts as ink when the cell under it is ink in most of the pictures that saw it.

Run:  D:/AI_Project/.venv/Scripts/python.exe v09b_dash_lengths_plain.py > v09b_dash_lengths_plain.out.txt
"""

import cv2
import numpy
import pandas

import verify_common
from verify_common import VERIFY_PARAMETERS, crossing_indices, ink_cells, register_line, sheet_line, top_view

cv2.setNumThreads(2)
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]

folder = VERIFY_PARAMETERS["model_runs_folder"] / VERIFY_PARAMETERS["run_8_folder"]
steps = pandas.read_csv(folder / "steps.csv")
pictures = [str(folder / name) for name in steps["frame"]]
joints = steps[JOINT_NAMES].to_numpy()
chosen = numpy.flatnonzero(numpy.abs(numpy.degrees(joints[:, 3])) < 30.0)
coarse = top_view([pictures[position] for position in chosen], joints[chosen])
placed = register_line(sheet_line("06"), coarse)
VERIFY_PARAMETERS["map_cell_m"] = 0.001
VERIFY_PARAMETERS["pixel_stride"] = 1
fine = top_view([pictures[position] for position in chosen], joints[chosen])
with numpy.errstate(divide = "ignore", invalid = "ignore"):
    share = numpy.where(fine["seen"] > 0, fine["ink"] / numpy.maximum(fine["seen"], 1), 0.0)
is_ink = (fine["seen"] >= 2) & (share >= 0.5)
line = placed["line_m"]
x_index = numpy.clip(numpy.floor((line[:, 0] - fine["x_m"][0]) / 0.001).astype(int), 0, is_ink.shape[0] - 1)
y_index = numpy.clip(numpy.floor((line[:, 1] - fine["y_m"][0]) / 0.001).astype(int), 0, is_ink.shape[1] - 1)
# the centre line may lie a few millimetres beside the true middle: ink anywhere within 3 mm across the line counts
tangent = numpy.gradient(line, axis = 0)
tangent /= numpy.linalg.norm(tangent, axis = 1)[:, None]
normal = numpy.column_stack([-tangent[:, 1], tangent[:, 0]])
on_ink = numpy.zeros(len(line), dtype = bool)
is_seen = numpy.zeros(len(line), dtype = bool)
for offset in (-0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003):
    points = line + offset * normal
    xi = numpy.clip(numpy.floor((points[:, 0] - fine["x_m"][0]) / 0.001).astype(int), 0, is_ink.shape[0] - 1)
    yi = numpy.clip(numpy.floor((points[:, 1] - fine["y_m"][0]) / 0.001).astype(int), 0, is_ink.shape[1] - 1)
    on_ink |= is_ink[xi, yi]
    is_seen |= fine["seen"][xi, yi] >= 2
crossing = crossing_indices(line)
edges = numpy.flatnonzero(numpy.diff(on_ink.astype(int)) != 0) + 1
starts = numpy.r_[0, edges]
ends = numpy.r_[edges, len(on_ink)]
dashes, gaps = [], []
for start, end in list(zip(starts, ends))[1:-1]:
    if not is_seen[start:end].all():
        continue
    if any(abs(place - position) < 12 for place in crossing for position in (start, end)):
        continue
    (dashes if on_ink[start] else gaps).append(0.1 * (end - start))
print(f"run 8, {len(chosen)} pictures with the forearm under 30 degrees, 1 mm cells, no widening along the line:")
print(f"  {len(dashes)} dashes: median {numpy.median(dashes):.1f} cm, {[round(value, 1) for value in dashes]}")
print(f"  {len(gaps)} gaps: median {numpy.median(gaps):.1f} cm, {[round(value, 1) for value in gaps]}")
print(f"  dash plus gap: {numpy.median(dashes) + numpy.median(gaps):.1f} cm; ink on {100 * on_ink[is_seen].mean():.0f} % of the seen line")
