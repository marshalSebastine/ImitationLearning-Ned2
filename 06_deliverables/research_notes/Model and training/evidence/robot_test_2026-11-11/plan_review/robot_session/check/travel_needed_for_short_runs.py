"""
Check of the critic's fix for P9 ("short runs end by themselves: a tip-travel budget of 22 cm; until it exists
--step-budget 22, 34 and 45"): how much tip travel did the eight standard-start demonstrations of sheet 03 (end A)
need until the puck was 8 cm of line past the first bend's apex (119 mm from the start cap)? And the three robot runs
of 11.11.26 (which started with the block at the puck, about 3.7 cm less free travel than the recorded start).
The traced line is laid onto each demonstration's own puck track (the puck was pushed along the line); the tip's
travel is summed from the logged joints (frames.csv, tip_x_m and tip_y_m). Reads files only.

Run:  D:/AI_Project/.venv/Scripts/python.exe travel_needed_for_short_runs.py > travel_needed_for_short_runs.out.txt
"""

import json
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
NOTES = ROOT / "06_deliverables" / "research_notes"
EVIDENCE = NOTES / "Model and training" / "evidence" / "robot_test_2026-11-11"
INDEX = Path("D:/AI_Project/03_data/processed/v01_training_index")
sheet = json.loads((NOTES / "Recording tool" / "2026-11-06_sheet_measurements.json").read_text(encoding = "utf-8"))["03"]
WINDOW_END_MM = 39.0 + 80.0


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


def place_along_line(puck_mm, traced):
    """Lay the traced line onto a puck track (first 250 mm of line only: the start of the run) and give each puck its place along it."""

    track = puck_mm[numpy.isfinite(puck_mm).all(axis = 1)]
    best = None
    for turn_deg in range(0, 360, 5):
        angle = numpy.radians(turn_deg)
        rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
        shift = track[0] - rotation @ traced[20]
        for _ in range(25):
            moved = traced @ rotation.T + shift
            nearest_index = numpy.linalg.norm(track[:, None, :] - moved[None, :, :], axis = 2).argmin(axis = 1)
            rotation, shift = best_turn_and_shift(traced[nearest_index], track)
        moved = traced @ rotation.T + shift
        gaps = numpy.linalg.norm(track[:, None, :] - moved[None, :, :], axis = 2).min(axis = 1)
        cost = float(numpy.mean(numpy.minimum(gaps, 20.0)))
        if best is None or cost < best[0]:
            best = (cost, moved, float(numpy.median(gaps)))
    _, moved, misfit = best
    along, last = [], 0
    for point in puck_mm:
        if not numpy.isfinite(point).all():
            along.append(numpy.nan)
            continue
        window = slice(max(0, last - 30), min(len(moved), last + 60))
        place = window.start + int(numpy.linalg.norm(moved[window] - point, axis = 1).argmin())
        along.append(float(place))
        last = place
    return numpy.array(along), misfit


traced = resampled(10.0 * numpy.array(sheet["line_xy_cm"], dtype = float))
frames = pandas.read_csv(INDEX / "frames.csv", usecols = ["episode_id", "row_in_episode", "tip_x_m", "tip_y_m"], low_memory = False)
print("demonstrations of sheet 03 from end A with the standard start (the tip starts about 5.5 cm behind the puck):")
needed = []
for plan in ("03-01", "03-02", "03-03", "03-04", "03-05", "03-25", "03-26", "03-27"):
    data = numpy.load(EVIDENCE / "step_length" / "tracks" / f"demonstration_{plan}.npz")
    puck = 1000.0 * data["puck_xy_m"]          # every second picture of the episode
    along, misfit = place_along_line(puck, traced)
    episode_id = next(name for name in frames["episode_id"].unique() if f"plan_{plan}_" in name)
    rows = frames[frames["episode_id"] == episode_id].sort_values("row_in_episode")
    tips = 100.0 * rows[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    travel = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(tips, axis = 0), axis = 1))]
    reached = numpy.flatnonzero(along >= WINDOW_END_MM)
    if len(reached) == 0:
        print(f"   {plan}: the puck never reached {WINDOW_END_MM:.0f} mm")
        continue
    picture = 2 * int(reached[0])
    needed.append(float(travel[min(picture, len(travel) - 1)]))
    print(f"   {plan}: the puck passes {WINDOW_END_MM:.0f} mm of line at picture {picture}; tip travel by then {needed[-1]:.1f} cm; puck starts at {along[0]:.0f} mm of line; "
          f"line laid on with a median misfit of {misfit:.1f} mm")
print(f"   tip travel needed: median {numpy.median(needed):.1f} cm, largest {max(needed):.1f} cm")

print("\nthe three robot runs of 11.11.26 (1 cm steps; started with the block at the puck):")
for run, picture in ((10, 17),):
    print(f"   run {run} (kept the line): the puck passed {WINDOW_END_MM:.0f} mm at picture {picture} = {picture} cm of tip travel; from the recorded start about 3.7 cm more: {picture + 3.7:.1f} cm")
for budget_cm in (22.0, 26.0, 30.0):
    print(f"   a budget of {budget_cm:.0f} cm covers the window in {sum(value <= budget_cm for value in needed)} of {len(needed)} demonstrations")
