"""
Plan review, key "evidence": (1) how well do the stop-and-centring analyst's two picture measures (12-picture
"offset_mm" and one-picture "single_offset_mm") agree with each other where the plan wants to use them as a judge,
and (2) what do the plan's own rules say on the ten logged runs: section 3's "lost" (more than 12 mm outside the ink
= more than 17 mm from the line's middle, two pictures in a row) and block A's pass rule (within 12 mm of the middle).
Reads the analyst's CSV files only; sizes are compared without sign (the verifier found that the sign flips).
"""

import glob

import numpy
import pandas

folder = "D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/stop_and_centring/"


def agreement(rows: pandas.DataFrame, label: str) -> None:

    both = rows[rows["offset_mm"].notna() & rows["single_offset_mm"].notna() & rows["is_clean"] & rows["is_single_clean"]]
    if len(both) == 0:
        print(f"{label:60s}: no picture with both measures")
        return
    gap = (both["offset_mm"].abs() - both["single_offset_mm"].abs()).abs()
    side_12 = ((both["offset_mm"].abs() > 12.0) != (both["single_offset_mm"].abs() > 12.0)).mean()
    side_7 = ((both["offset_mm"].abs() > 7.0) != (both["single_offset_mm"].abs() > 7.0)).mean()
    print(f"{label:60s}: {len(both):4d} pictures with both measures clean; sizes differ by a median {gap.median():4.1f} mm, "
          f"more than 5 mm in {100 * (gap > 5).mean():3.0f} %, more than 10 mm in {100 * (gap > 10).mean():3.0f} %; "
          f"they disagree on 'over 7 mm' in {100 * side_7:3.0f} % and on 'over 12 mm' in {100 * side_12:3.0f} %")


print("===== 1. the two measures against each other, demonstrations (36 sampled episodes, six per sheet) =====")
for path in sorted(glob.glob(folder + "demonstration_centring_0*.csv")):
    table = pandas.read_csv(path)
    agreement(table, "sheet " + path[-6:-4] + ", all sampled pictures")
    twisting = table[table["forearm_change_deg"].abs() >= 5.0]
    agreement(twisting, "sheet " + path[-6:-4] + ", pictures whose step twists the forearm 5 deg or more")

print()
print("===== 2. the two measures against each other, the ten robot runs =====")
runs = pandas.read_csv(folder + "robot_runs_steps.csv")
for run_number in range(1, 11):
    agreement(runs[runs["run"] == run_number], f"run {run_number}")

print()
print("===== 3. the plan's rules on the ten logged runs (12-picture measure, as the analyst's tables give it) =====")
for run_number in range(1, 11):
    rows = runs[runs["run"] == run_number].sort_values("step").reset_index(drop = True)
    sizes = rows["offset_mm"].abs().to_numpy()
    steps = rows["step"].to_numpy()

    def first_two_in_a_row(limit: float):
        over = sizes > limit
        return next((int(steps[index]) for index in range(1, len(steps)) if over[index] and over[index - 1]), None)

    jumps = numpy.abs(numpy.diff(rows["offset_mm"].to_numpy()))
    print(f"run {run_number:2d}: {len(rows):3d} pictures; within 7 mm {100 * numpy.nanmean(sizes <= 7):3.0f} %; over 12 mm from the middle two in a row first at picture "
          f"{first_two_in_a_row(12.0)}; 'lost' (over 17 mm two in a row) first at picture {first_two_in_a_row(17.0)}; "
          f"picture-to-picture jumps of the offset over 12 mm (the puck moves at most 10 mm per step): {int(numpy.nansum(jumps > 12.0))}")

for run_number, first, last in ((4, 18, 34), (7, 24, 70), (1, 10, 25)):
    rows = runs[(runs["run"] == run_number) & (runs["step"] >= first) & (runs["step"] <= last)]
    print()
    print(f"run {run_number}, pictures {first} to {last}: offset_mm =", " ".join(f"{value:.0f}" for value in rows["offset_mm"]))
