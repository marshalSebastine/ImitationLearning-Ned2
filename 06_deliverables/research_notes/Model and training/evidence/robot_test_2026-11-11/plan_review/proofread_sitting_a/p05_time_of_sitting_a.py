"""
Proofreader sitting_a, check 5: the runs and the minutes of test sitting A as version 2 words it.

Measured from the logs of 11.11.26: seconds per picture, the gap from the last picture of one run to the first of the
next, and whether today's stall rule in config.py would have ended one of the ten runs. Calculated from them: the
minutes of every part of sitting A. Pictures per run: sheet 03 short runs end at a tip-travel budget of 27 cm (the
checker's number); sheet 04's come from p03_travel_budgets.out.txt; sheet 01 from the recorded standard starts.
What is a guess is marked.

Reads logs only. Run:
    D:/AI_Project/.venv/Scripts/python.exe -B p05_time_of_sitting_a.py > p05_time_of_sitting_a.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
sys.path.insert(0, str(CODE))
from config import MODEL_MODE_PARAMETERS   # noqa: E402

RUNS = Path("D:/AI_Project/03_data/model_runs")
folders = sorted(RUNS.glob("2026-11-11_17*"))
per_picture, first_times, last_times, lengths = [], [], [], []
print("the ten runs of 11.11.26 [measured from steps.csv]:")
for folder in folders:
    steps = pandas.read_csv(folder / "steps.csv")
    times = steps["unix_time"].to_numpy()
    seconds = float(numpy.median(numpy.diff(times)))
    per_picture.extend(numpy.diff(times).tolist())
    first_times.append(times[0])
    last_times.append(times[-1])
    lengths.append(len(steps))
    # today's stall rule on the run as driven
    count, extent_m, twist_deg = MODEL_MODE_PARAMETERS["stall_steps"], MODEL_MODE_PARAMETERS["stall_extent_m"], MODEL_MODE_PARAMETERS["stall_twist_deg"]
    stalled_at = None
    for last in range(count, len(steps) + 1):
        recent = steps.iloc[last - count:last]
        extent = float(numpy.hypot(recent["tip_x"].max() - recent["tip_x"].min(), recent["tip_y"].max() - recent["tip_y"].min()))
        turned = float(numpy.degrees(recent["joint_4"].max() - recent["joint_4"].min()))
        if extent < extent_m and turned < twist_deg:
            stalled_at = last - 1
            break
    print(f"   {folder.name:42s} {len(steps):3d} pictures, median {seconds:.2f} s per picture, whole run {times[-1] - times[0]:5.1f} s; "
          f"today's stall rule ({count} pictures, {100 * extent_m:.0f} cm, {twist_deg:.0f} deg): {'never fires' if stalled_at is None else f'ENDS THE RUN at picture {stalled_at}'}")
gaps = numpy.array(first_times[1:]) - numpy.array(last_times[:-1])
starts = numpy.diff(first_times)
print(f"   seconds per picture: median {numpy.median(per_picture):.2f}, mean {numpy.mean(per_picture):.2f}, 90th percentile {numpy.percentile(per_picture, 90):.2f}")
print(f"   gap, last picture of a run to first picture of the next: {numpy.round(gaps).astype(int).tolist()} s; median {numpy.median(gaps):.0f}, shortest {gaps.min():.0f}, longest {gaps.max():.0f}")
print(f"   first picture to first picture: median {numpy.median(starts) / 60:.1f} min; ten runs in {(last_times[-1] - first_times[0]) / 60:.1f} min")

SECONDS = float(numpy.median(per_picture))
# pictures per run [calculated]: budget in cm divided by the step; the own length is taken as 0.52 cm (the model asked a
# median 0.49 to 0.55 cm on sheet 03 on 11.11, config.py comment) -- a guess for the new start
parts = [
    # (part, what, runs, pictures per run, extra seconds per run that 11.11 did not have, note)
    ("warm-up", "one run, not counted (sheet 01, model E, 1.0 cm)", 1, 30, 0, ""),
    ("A1", "scripted list without the puck (30 small steps, 10 twists)", 1, 40, 60, "ruler photo before and after"),
    ("A1", "the same with the puck on sheet 01", 1, 40, 120, "ruler photo after each group"),
    ("A3", "model E, sheet 01, 1.0 cm, full runs", 3, 30, 45, "photo with ruler at the stop"),
    ("A4", "model E, sheet 03, fixed 1.0 cm, budget 27 cm", 5, 27, 0, ""),
    ("A4", "model E, sheet 03, fixed 0.65 cm, budget 27 cm", 5, 42, 0, ""),
    ("A4", "model E, sheet 03, own length, budget 27 cm", 4, 52, 0, "pictures a guess"),
    ("A4", "team of five, sheet 03, fixed 1.0 cm", 4, 27, 0, ""),
    ("A4", "model without sheet 03, fixed 1.0 cm", 3, 27, 0, ""),
    ("A4", "model without sheet 03, fixed 0.65 cm", 3, 42, 0, ""),
    ("A5", "model E, sheet 04, 1.0 cm, through the first S-bend (to 160 mm of line, budget 42 cm; to the 170 degree turn at 244 mm it would be 75 cm)", 3, 42, 0, ""),
    ("A5", "either: one full run of sheet 04 (155 to 200 cm of tip travel recorded)", 1, 180, 45, "only if the bends are passed"),
    ("A5", "or: the same three runs with the step A4 chose (0.65 cm)", 3, 65, 0, "only if the bends are not passed"),
]
print()
for label, gap in (("the shortest gap of 11.11", gaps.min()), ("the median gap of 11.11", numpy.median(gaps)), ("the longest gap of 11.11", gaps.max())):
    print(f"sitting A with {label} ({gap:.0f} s between runs) and {SECONDS:.2f} s per picture [calculated]:")
    by_part, runs_total = {}, 0
    for part, what, runs, pictures, extra, note in parts:
        minutes = runs * (pictures * SECONDS + gap + extra) / 60.0
        by_part.setdefault(part, []).append((what, runs, pictures, minutes, note))
    low = high = 0.0
    for part, rows in by_part.items():
        if part == "A5":
            fixed = rows[0][3]
            either, other = rows[1][3], rows[2][3]
            print(f"   A5: {rows[0][1]} runs {fixed:.0f} min, then {either:.0f} min (one full run) or {other:.0f} min (three more runs): {fixed + either:.0f} to {fixed + other:.0f} min")
            low, high = low + fixed + either, high + fixed + other
            runs_low, runs_high = rows[0][1] + rows[1][1], rows[0][1] + rows[2][1]
        else:
            minutes = sum(row[3] for row in rows)
            runs = sum(row[1] for row in rows)
            print(f"   {part}: {runs} runs, {minutes:.0f} min" + "".join(f"\n        {row[1]} x {row[2]} pictures: {row[0]}{' (' + row[4] + ')' if row[4] else ''}" for row in rows))
            low, high, runs_total = low + minutes, high + minutes, runs_total + runs
    print(f"   runs: {runs_total + runs_low} to {runs_total + runs_high} (the warm-up and the two scripted lists included); runs alone {low:.0f} to {high:.0f} min = {low / 60:.1f} to {high / 60:.1f} h")
    a2 = (30.0, 45.0)
    setup = {"switching on, the offline tests, camera check, backup at the end": 20, "tape marks and puck place for four lay-outs (sheets 01, 03, 04 and the circle print), 8 min each": 32,
             "each new movement once without sheet and puck (start move, lift and wait, scripted list, snapshot)": 15,
             "the checklist read and the go for each part": 10, "void runs repeated, 1 run in 10": round(0.1 * (runs_total + runs_high) * (40 * SECONDS + gap) / 60)}
    print(f"   A2 (20 places by hand, motors off and on again at each, three pictures each): {a2[0]:.0f} to {a2[1]:.0f} min [guess: 1.5 to 2.25 min per place]")
    print("   on top [guess, nothing like it was timed]: " + "; ".join(f"{name} {minutes} min" for name, minutes in setup.items()) + f" = {sum(setup.values())} min")
    total_low, total_high = low + a2[0] + sum(setup.values()), high + a2[1] + sum(setup.values())
    print(f"   whole sitting: {total_low:.0f} to {total_high:.0f} min = {total_low / 60:.1f} to {total_high / 60:.1f} h")
    print()
