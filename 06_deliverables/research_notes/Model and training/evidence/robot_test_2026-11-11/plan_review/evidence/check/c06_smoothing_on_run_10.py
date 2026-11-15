"""
Check of the plan review, key "evidence_check" (objection P2, the critic's fix): the critic proposes to smooth the
offset ("median of three pictures, no jump above 10 mm per step, sizes only") and asks that the scoring command then
gives "run 10 passes the first bend, runs 5 and 9 fail it". This script applies the plan's block A rule (the puck
within 12 mm of the line's middle; here: no two pictures in a row over 12 mm) to the stop-and-centring analyst's
12-picture offsets of runs 5, 9 and 10 over the first 22 pictures (to about 10 cm past the first bend), three ways:
raw sizes, a median of three pictures, and a median of three after readings that jump by more than 10 mm from BOTH
neighbours' level are dropped. Reads the analyst's CSV only.
"""

import numpy
import pandas

folder = "D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/stop_and_centring/"
runs = pandas.read_csv(folder + "robot_runs_steps.csv")
LAST = 22


def first_two_in_a_row(sizes: numpy.ndarray, steps: numpy.ndarray, limit: float = 12.0):

    over = sizes > limit
    return next((int(steps[index]) for index in range(1, len(steps)) if over[index] and over[index - 1]), None)


def median_of_three(sizes: numpy.ndarray) -> numpy.ndarray:

    padded = numpy.r_[sizes[0], sizes, sizes[-1]]
    return numpy.array([numpy.nanmedian(padded[index:index + 3]) for index in range(len(sizes))])


def without_jumps(sizes: numpy.ndarray, most_mm: float = 10.0) -> numpy.ndarray:
    """A reading is dropped when it lies more than most_mm from the last kept reading; after two drops in a row the level is taken over."""

    kept = sizes.copy()
    last, dropped_in_a_row = None, 0
    for index, value in enumerate(sizes):
        if numpy.isnan(value):
            continue
        if last is not None and abs(value - last) > most_mm and dropped_in_a_row < 2:
            kept[index] = numpy.nan
            dropped_in_a_row += 1
            continue
        last, dropped_in_a_row = value, 0
    return kept


for run_number in (5, 9, 10):
    rows = runs[(runs["run"] == run_number) & (runs["step"] <= LAST)].sort_values("step")
    steps = rows["step"].to_numpy()
    sizes = rows["offset_mm"].abs().to_numpy()
    print(f"run {run_number}, pictures 0 to {LAST}: sizes (mm) " + " ".join("  nan" if numpy.isnan(value) else f"{value:5.1f}" for value in sizes))
    print(f"   raw sizes: first two in a row over 12 mm at picture {first_two_in_a_row(numpy.nan_to_num(sizes), steps)}")
    smooth = median_of_three(sizes)
    print(f"   median of three: " + " ".join(f"{value:5.1f}" for value in smooth))
    print(f"   median of three: first two in a row over 12 mm at picture {first_two_in_a_row(numpy.nan_to_num(smooth), steps)}")
    cleaned = without_jumps(sizes)
    print(f"   jumps over 10 mm dropped (at most two in a row): first two in a row over 12 mm at picture {first_two_in_a_row(numpy.nan_to_num(cleaned), steps)}; "
          f"readings dropped at pictures {[int(step) for step, before, after in zip(steps, sizes, cleaned) if not numpy.isnan(before) and numpy.isnan(after)]}")
