"""
Verifier's check 12 (11.11.26): the analyst's table 'where the fold models would stop' recounted with my own code from
the analyst's puck places (../demonstration_last_steps_*.csv, reliable ends only) and the saved fold predictions, for
the E recipe and the A recipe: the first stop-like picture of the last eight, the two-in-a-row rule, a last straight
push, higher thresholds. Added: what the same pictures say for a driver that moves the predicted length (then the
confirming picture lies a demonstrated nudge further on, not a full centimetre).
"""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent
RUNS_FOLDER = HERE.parents[6] / "04_runs" / "gpu_ki014"

ends = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_ends_01_03.csv", "demonstration_ends_04_06.csv")], ignore_index = True)
ends["is_reliable"] = (ends["is_end_found"] & (ends["is_end_seen"] == True) & ends["line_width_mm"].between(8.0, 12.5) & (ends["fit_quality"] >= 0.8))
reliable_ids = set(ends.loc[ends["is_reliable"], "episode_id"])
last_steps = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_last_steps_01_03.csv", "demonstration_last_steps_04_06.csv")], ignore_index = True)
last_steps = last_steps[last_steps["episode_id"].isin(reliable_ids)]

for label, folder in {"E recipe": "2026-11-10_resnet18_chunk_stop_weight_25", "A recipe": "2026-11-09_resnet18_final_candidate"}.items():
    chance = {}
    for number in range(1, 7):
        saved = numpy.load(RUNS_FOLDER / folder / f"predictions_without_path_{number:02d}.npz")
        chance.update(dict(zip(saved["frame_index"].tolist(), saved["stop_probability"].tolist())))
    table = last_steps.copy()
    table["stop_probability"] = table["frame_index"].map(chance)
    table = table.dropna(subset = ["stop_probability"]).sort_values(["episode_id", "step"])
    print(f"{label}: {table['episode_id'].nunique()} episodes, {len(table)} pictures scored")
    bins = [-30, -5, 0, 5, 10, 15, 25, 45, 90]
    binned = table.groupby(pandas.cut(table["short_mm"], bins), observed = True)["stop_probability"].agg(pictures = "size", share_at_threshold = lambda values: float((values >= 0.5).mean()))
    print("  share of pictures with a stop chance of 0.5 or more by shortfall (mm):", {str(index): f"{row['share_at_threshold']:.2f} of {int(row['pictures'])}" for index, row in binned.iterrows()})
    for threshold in (0.5, 0.7, 0.9):
        first_short, first_aside, second_short, second_gap, never = [], [], [], [], 0
        for _, rows in table.groupby("episode_id", sort = False):
            is_like = (rows["stop_probability"] >= threshold).to_numpy()
            if not is_like.any():
                never += 1
                continue
            index = int(numpy.flatnonzero(is_like)[0])
            first_short.append(rows["short_mm"].iloc[index])
            first_aside.append(rows["aside_mm"].iloc[index])
            if index + 1 < len(rows) and is_like[index + 1]:
                second_short.append(rows["short_mm"].iloc[index + 1])
                second_gap.append(float(numpy.hypot(rows["short_mm"].iloc[index + 1], rows["aside_mm"].iloc[index + 1])))
        first_short, first_aside, second_short, second_gap = (numpy.array(values) for values in (first_short, first_aside, second_short, second_gap))
        print(f"  threshold {threshold}: first stop-like picture found in {len(first_short)} episodes, never in {never}; the puck is then short by median {numpy.median(first_short):+.1f} mm"
              f" (10th to 90th percentile {numpy.percentile(first_short, 10):+.1f} to {numpy.percentile(first_short, 90):+.1f}); over 7 mm short in {int((first_short > 7).sum())}")
        if threshold == 0.5:
            for push in (0.0, 5.0, 8.0, 10.0):
                along = numpy.abs(first_short - push) <= 7.0
                both = numpy.hypot(first_short - push, first_aside) <= 7.0
                print(f"    a last straight push of {push:4.1f} mm after that picture: within 7 mm along the line in {100 * along.mean():.0f} %, with the sideways offset as it was in {100 * both.mean():.0f} %")
            print(f"    the next demonstrated picture is also stop-like in {len(second_short)} episodes: the puck then stands median {numpy.median(second_short):+.1f} mm short"
                  f" (10th to 90th percentile {numpy.percentile(second_short, 10):+.1f} to {numpy.percentile(second_short, 90):+.1f}); centres at most 7 mm apart in {int((second_gap <= 7).sum())} of {len(second_gap)};"
                  f" still more than 7 mm short in {int((second_short > 7).sum())}")
            print("    (this last line is the nearest thing to a driver that moves the predicted length: the confirming picture lies one demonstrated nudge further on;"
                  f" the present driver instead adds a full centimetre to the first picture's median {numpy.median(first_short):+.1f} mm)")
