"""
Does model E already carry a graded "how far to the end" signal, and would it stop more unseen-sheet episodes?

Model E predicts the next five tip positions; after the episode's last row the target stays where the tip stopped
(training_targets.chunk_targets), so near the end the predicted fifth position shrinks towards the tip. That is a
graded distance to the stop, learnt from every picture of the last five steps instead of one stop picture per
episode. This script reads the saved answers of the fold models on sheets they never saw (no training, no robot) and
scores stop rules with the project's own stop measure (training_metrics.stop_metrics: right = first stop within 2 cm
of where the demonstration stopped, early = more than 3 cm before it).

    1. The present rule (stop chance 0.5, two pictures in a row): must give back 106 right, 32 early, 71 never.
    2. The fifth predicted position: how well it follows the demonstrated one in the last rows.
    3. A rule on the chunk alone, and the two rules joined by "or", for a range of limits (shown as a range because
       a limit must not be tuned on the scored sheets).
    4. Where the chunk rule fires early: at bends (twists in place) or elsewhere?

Run from the folder above:
    python check/c02_end_from_the_chunk.py > check/c02_end_from_the_chunk.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import method_common

CHECK_PARAMETERS = {"chunk_steps": 5, "limits_cm": [0.75, 1.0, 1.25, 1.5, 2.0, 2.5], "in_a_row": [1, 2]}



def episode_counts(is_fired: numpy.ndarray, goal_distance_cm: numpy.ndarray, episode_ids: numpy.ndarray, in_a_row: int) -> dict:
    """Return right / early / between / never counts with the project's own stop measure."""

    from training_metrics import stop_metrics

    result = stop_metrics(numpy.zeros(len(is_fired)), is_fired.astype(float), goal_distance_cm, episode_ids, threshold = 0.5,
                          consecutive_pictures = in_a_row)
    episodes = len(numpy.unique(episode_ids))
    right = round(result["episodes_stopped_right"] * episodes)
    early = round(result["episodes_stopped_early"] * episodes)
    never = round(result["episodes_never_stopped"] * episodes)
    return {"right": right, "early": early, "between": episodes - right - early - never, "never": never}



def report() -> None:
    from training_targets import chunk_targets

    settings = method_common.merged_parameters(CHECK_PARAMETERS)
    frames = method_common.cleaned_frames(settings)
    data = numpy.load(settings["fold_outputs"]["E"])
    answers = pandas.DataFrame({"frame_index": data["frame_index"], "stop_chance": data["stop_probability"],
                                "asked_fifth_cm": numpy.linalg.norm(data["chunk"][:, settings["chunk_steps"] - 1, :], axis = 1),
                                "asked_first_cm": numpy.linalg.norm(data["chunk"][:, 0, :], axis = 1)})
    table = frames.merge(answers, on = "frame_index", how = "left")
    shown = chunk_targets(table, settings["chunk_steps"])
    table["shown_fifth_cm"] = numpy.linalg.norm(shown[:, settings["chunk_steps"] - 1, :], axis = 1)
    episode_ids = table["episode_id"].to_numpy()
    goal = table["goal_distance_cm"].to_numpy()
    sheets = table["sheet"].to_numpy()

    print("1. The present rule on sheets the model never saw (recipe E)")
    for in_a_row in settings["in_a_row"]:
        counts = episode_counts(table["stop_chance"].to_numpy() >= 0.5, goal, episode_ids, in_a_row)
        print(f"   stop chance 0.5 or more, {in_a_row} in a row: right {counts['right']}, early {counts['early']}, 2 to 3 cm before {counts['between']}, never {counts['never']} of {len(numpy.unique(episode_ids))}")

    print()
    print("2. The fifth predicted tip position against the demonstrated one (length in cm)")
    steps_to_end = table.groupby("episode_id", sort = False).cumcount(ascending = False)
    for name, mask in (("all rows", numpy.ones(len(table), dtype = bool)), ("the last 6 rows of each episode", (steps_to_end <= 5).to_numpy()),
                       ("rows more than 10 rows before the end", (steps_to_end > 10).to_numpy())):
        part = table[mask]
        print(f"   {name} ({len(part)}): correlation {numpy.corrcoef(part['asked_fifth_cm'], part['shown_fifth_cm'])[0, 1]:+.2f}; shown median {part['shown_fifth_cm'].median():.2f} cm, asked median {part['asked_fifth_cm'].median():.2f} cm")
    print("   by rows left until the stop row: shown median / asked median / asked 10th to 90th percentile; stop chance median")
    for left in range(0, 9):
        part = table[(steps_to_end == left).to_numpy()]
        print(f"      {left} rows left ({len(part)}): {part['shown_fifth_cm'].median():.2f} / {part['asked_fifth_cm'].median():.2f} / {part['asked_fifth_cm'].quantile(0.1):.2f} to {part['asked_fifth_cm'].quantile(0.9):.2f}; {part['stop_chance'].median():.2f}")
    far = table[(steps_to_end > 10).to_numpy()]
    print(f"   far from the end (more than 10 rows left, {len(far)} rows): asked fifth position under 1.0 cm in {100 * (far['asked_fifth_cm'] < 1.0).mean():.1f} %, under 1.5 cm in {100 * (far['asked_fifth_cm'] < 1.5).mean():.1f} %, under 2.0 cm in {100 * (far['asked_fifth_cm'] < 2.0).mean():.1f} %;"
          f" stop chance 0.5 or more in {100 * (far['stop_chance'] >= 0.5).mean():.1f} %")

    print()
    print("3. Stop rules scored over whole episodes (right / early / 2 to 3 cm before / never, of 209)")
    is_chance = table["stop_chance"].to_numpy() >= 0.5
    for in_a_row in settings["in_a_row"]:
        print(f"   {in_a_row} picture(s) in a row")
        for limit in settings["limits_cm"]:
            is_short = table["asked_fifth_cm"].to_numpy() < limit
            alone = episode_counts(is_short, goal, episode_ids, in_a_row)
            joined = episode_counts(is_short | is_chance, goal, episode_ids, in_a_row)
            both = episode_counts(is_short & is_chance, goal, episode_ids, in_a_row)
            print(f"      fifth position under {limit:4.2f} cm: alone {alone['right']:3d} / {alone['early']:3d} / {alone['between']:3d} / {alone['never']:3d};"
                  f"   or stop chance 0.5: {joined['right']:3d} / {joined['early']:3d} / {joined['between']:3d} / {joined['never']:3d};"
                  f"   and stop chance 0.5: {both['right']:3d} / {both['early']:3d} / {both['between']:3d} / {both['never']:3d}")
    print("   per sheet, 2 in a row: present rule against 'fifth position under 1.5 cm or stop chance 0.5'")
    for sheet in sorted(set(sheets)):
        mask = sheets == sheet
        present = episode_counts(is_chance[mask], goal[mask], episode_ids[mask], 2)
        joined = episode_counts((is_chance | (table["asked_fifth_cm"].to_numpy() < 1.5))[mask], goal[mask], episode_ids[mask], 2)
        print(f"      {sheet[-2:]}: present {present['right']:2d} / {present['early']:2d} / {present['between']:2d} / {present['never']:2d};   joined {joined['right']:2d} / {joined['early']:2d} / {joined['between']:2d} / {joined['never']:2d}")

    print()
    print("4. Where 'fifth position under 1.5 cm' holds far from the end (more than 3 cm of tip distance to the stop place)")
    is_short = table["asked_fifth_cm"] < 1.5
    wrong = table[is_short & (table["goal_distance_cm"] > 3.0) & (table["action"] != "stop")]
    length = 100.0 * numpy.hypot(wrong["tip_forward_m"].fillna(0.0), wrong["tip_left_m"].fillna(0.0))
    print(f"   rows: {len(wrong)}; the demonstrated step there is a twist nearly in place (under 0.3 cm) in {100 * (length < 0.3).mean():.0f} %; shown fifth position under 1.5 cm in {100 * (wrong['shown_fifth_cm'] < 1.5).mean():.0f} %;"
          f" by sheet: " + ", ".join(f"{sheet[-2:]} {count}" for sheet, count in wrong["sheet"].value_counts().sort_index().items()))



if __name__ == "__main__":
    report()
