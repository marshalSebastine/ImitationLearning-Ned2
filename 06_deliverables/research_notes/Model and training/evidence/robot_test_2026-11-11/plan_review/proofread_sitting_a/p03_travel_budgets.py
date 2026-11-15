"""
Proofreader sitting_a, check 3: how much tip travel and how many pictures a short run needs.

Part A4 (sheet 03, end A): until the puck has passed 119 mm of line. Part A5 (sheet 04, end A): until the puck has
passed 160 mm (through the first S-bend) and 285 mm (past the first hairpin). Counted on the standard-start
demonstrations whose puck tracks the step-length analyst stored (every second picture), with the checker's method
(the traced line laid onto the puck track); the tip's travel is summed from frames.csv. The sheet-03 numbers must give
back the checker's 16.2 to 21.3 cm. Also printed: where the puck and the tip stood in the first picture.

Reads files only. Run:
    D:/AI_Project/.venv/Scripts/python.exe -B p03_travel_budgets.py > p03_travel_budgets.out.txt
"""

import json
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
NOTES = ROOT / "06_deliverables" / "research_notes"
EVIDENCE = NOTES / "Model and training" / "evidence" / "robot_test_2026-11-11"
INDEX = Path("D:/AI_Project/03_data/processed/v01_training_index")
SHEETS = json.loads((NOTES / "Recording tool" / "2026-11-06_sheet_measurements.json").read_text(encoding = "utf-8"))


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


frames = pandas.read_csv(INDEX / "frames.csv", usecols = ["episode_id", "row_in_episode", "tip_x_m", "tip_y_m", "joint_4"], low_memory = False)
cases = {
    "03": {"plans": ["03-01", "03-02", "03-03", "03-04", "03-05", "03-25", "03-26", "03-27"], "marks_mm": [119.0]},
    "04": {"plans": ["04-01", "04-02", "04-03", "04-04", "04-05", "04-13", "04-14", "04-15", "04-16", "04-17", "04-18"], "marks_mm": [160.0, 285.0]},
}
for sheet_name, case in cases.items():
    traced = resampled(10.0 * numpy.array(SHEETS[sheet_name]["line_xy_cm"], dtype = float))
    print(f"sheet {sheet_name}, end A, standard-start demonstrations:")
    needed = {mark: {"travel": [], "pictures": []} for mark in case["marks_mm"]}
    for plan in case["plans"]:
        data = numpy.load(EVIDENCE / "step_length" / "tracks" / f"demonstration_{plan}.npz")
        puck = 1000.0 * data["puck_xy_m"]
        rows_kept = data["row_in_episode"]
        along, misfit = place_along_line(puck, traced)
        episode_id = next(name for name in frames["episode_id"].unique() if f"plan_{plan}_" in name)
        rows = frames[frames["episode_id"] == episode_id].sort_values("row_in_episode")
        tips = 100.0 * rows[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
        travel = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(tips, axis = 0), axis = 1))]
        gap = puck[0] / 10.0 - tips[0]
        text = [f"   {plan}: tip starts at ({tips[0][0]:.1f}, {tips[0][1]:.1f}) cm, forearm {numpy.degrees(rows['joint_4'].iloc[0]):+.1f} deg; puck at ({puck[0][0] / 10:.1f}, {puck[0][1] / 10:.1f}) cm = "
                f"{numpy.linalg.norm(gap):.1f} cm from the tip ({gap[0]:+.1f} ahead in x, {gap[1]:+.1f} in y); line misfit {misfit:.1f} mm; whole episode {len(tips)} pictures, {travel[-1]:.0f} cm of tip travel"]
        for mark in case["marks_mm"]:
            reached = numpy.flatnonzero(along >= mark)
            if len(reached) == 0:
                text.append(f"      never reached {mark:.0f} mm")
                continue
            picture = int(rows_kept[reached[0]])
            position = int(numpy.searchsorted(rows["row_in_episode"].to_numpy(), picture))
            position = min(position, len(travel) - 1)
            needed[mark]["travel"].append(float(travel[position]))
            needed[mark]["pictures"].append(position)
            text.append(f"      puck past {mark:.0f} mm of line at picture {position}: tip travel {travel[position]:.1f} cm")
        print("\n".join(text))
    for mark in case["marks_mm"]:
        values, pictures = needed[mark]["travel"], needed[mark]["pictures"]
        print(f"   until the puck has passed {mark:.0f} mm: tip travel median {numpy.median(values):.1f} cm, range {min(values):.1f} to {max(values):.1f} cm; "
              f"pictures median {numpy.median(pictures):.0f}, range {min(pictures)} to {max(pictures)} ({len(values)} demonstrations)")
    print()
