"""
Verifier's extra checks on the step-length recommendation.

1. Sheet 03, first bend: how far the tip advanced in x before it was 2 cm to the left of its start, for the three
   robot runs and for the demonstrations (per start kind), and how far the tip WOULD have advanced had every move had
   the asked length (calculated along the logged directions; the pictures would then have been different ones, so this
   is only a first-order figure).
2. Model E (all sheets) on the pictures it was trained on: the predicted length per kind of place against the
   demonstrations' local pace (needs the cache written by v06_run6_mechanism.py).
3. Per held-out sheet: how the fold model's asked length compares with the demonstrated pace at sharp bends when the
   model has never seen the sheet (needs the cache written by v05_offline_by_place.py).

Read-only. Prints to the screen and to v09_first_bend_and_pace.out.txt.
"""

from pathlib import Path
from typing import List

import numpy
import pandas

import verify_common

PACE_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "sheet_03_runs": ["2026-11-11_173549_E_sheet03_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie", "2026-11-11_175544_B_team5_sheet03"],
    "left_cm": 2.0
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


def advance_before_left(tip_x_cm: numpy.ndarray, tip_y_cm: numpy.ndarray, left_cm: float) -> float:
    """Return the farthest advance in x before the tip is left_cm to the left of its start (NaN when it never is)."""

    is_left = (tip_y_cm - tip_y_cm[0]) >= left_cm
    if not is_left.any():
        return float("nan")
    first = int(numpy.argmax(is_left))
    return float((tip_x_cm[:first + 1] - tip_x_cm[0]).max())


if __name__ == "__main__":
    settings = dict(PACE_PARAMETERS)
    frames = verify_common.demonstration_rows()

    say("== 1. sheet 03 from end A: farthest advance in x before the tip is 2 cm left of its start ==")
    raw = pandas.read_csv(Path(verify_common.TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", low_memory = False).sort_values(["episode_id", "row_in_episode"])
    from_a = set(frames.loc[(frames["sheet"] == "path_03") & (frames["start_end"] == "A"), "episode_id"])
    rows = []
    for episode_id, episode in raw[raw["episode_id"].isin(from_a)].groupby("episode_id"):
        tip_x, tip_y = 100.0 * episode["tip_x_m"].to_numpy(), 100.0 * episode["tip_y_m"].to_numpy()
        is_on_mark = abs(tip_x[0] - 14.0) < 0.2 and abs(tip_y[0] + 1.5) < 0.2
        rows.append({"episode_id": episode_id, "start_kind": episode["start_kind"].iloc[0], "is_on_mark": is_on_mark,
                     "advance_cm": advance_before_left(tip_x, tip_y, settings["left_cm"])})
    table = pandas.DataFrame(rows)
    on_mark = table[table["is_on_mark"]]
    say(f"demonstrations starting on the start mark: {len(on_mark)}; advance median {on_mark['advance_cm'].median():.1f} cm, from {on_mark['advance_cm'].min():.1f} to "
        f"{on_mark['advance_cm'].max():.1f}; sorted: {sorted(round(float(value), 1) for value in on_mark['advance_cm'])}")
    for kind, group in on_mark.groupby("start_kind"):
        say(f"   {kind:6s}: {len(group)} episodes, advance {sorted(round(float(value), 1) for value in group['advance_cm'])}")
    for name in settings["sheet_03_runs"]:
        steps = pandas.read_csv(settings["model_runs_folder"] / name / "steps.csv")
        moves = steps[steps["action"] == "model_step"]
        tip_x, tip_y = 100.0 * moves["tip_x"].to_numpy(), 100.0 * moves["tip_y"].to_numpy()
        advance = advance_before_left(tip_x, tip_y, settings["left_cm"])
        # the same logged directions with the asked lengths (first-order only: the later pictures would have differed)
        asked_cm = numpy.hypot(moves["model_tip_forward_cm"], moves["model_tip_left_cm"]).to_numpy()
        direction = numpy.radians(moves["picture_heading_deg"].to_numpy() + numpy.degrees(numpy.arctan2(moves["model_tip_left_cm"], moves["model_tip_forward_cm"])).to_numpy())
        asked_x = numpy.concatenate([[0.0], numpy.cumsum(asked_cm * numpy.cos(direction))])
        asked_y = numpy.concatenate([[0.0], numpy.cumsum(asked_cm * numpy.sin(direction))])
        say(f"{name[11:]:32s}: driven: advance {advance:.1f} cm ({int((on_mark['advance_cm'] >= advance).sum())} of {len(on_mark)} demonstrations advanced as far or farther); "
            f"with the asked lengths along the same logged directions: {advance_before_left(asked_x, asked_y, settings['left_cm']):.1f} cm (calculated)")

    cache_folder = verify_common.VERIFY_PARAMETERS["cache_folder"]
    moving = ~frames["is_stop"].to_numpy()
    # the demonstration's local pace: the mean of the five steps centred on the row, on the table
    angle = numpy.radians(frames["step_table_deg"].to_numpy())
    vectors = frames["step_cm"].to_numpy()[:, None] * numpy.stack([numpy.cos(angle), numpy.sin(angle)], axis = 1)
    codes = pandas.factorize(frames["episode_id"])[0]
    total, count = numpy.zeros((len(frames), 2)), numpy.zeros(len(frames))
    for offset in [-2, -1, 0, 1, 2]:
        shifted = numpy.roll(vectors, -offset, axis = 0)
        is_same = (numpy.roll(codes, -offset) == codes) & numpy.roll(moving, -offset)
        if offset > 0:
            is_same[-offset:] = False
        elif offset < 0:
            is_same[:-offset] = False
        total += numpy.where(is_same[:, None], shifted, 0.0)
        count += is_same
    local_pace = numpy.linalg.norm(total / numpy.maximum(count, 1)[:, None], axis = 1)

    say("")
    say("== 2. model E (all sheets) on its own training pictures: predicted length per kind of place ==")
    all_sheets_path = cache_folder / "all_sheets_e_demonstrations.npz"
    if all_sheets_path.exists():
        own = numpy.load(all_sheets_path)
        if numpy.array_equal(own["frame_index"], frames["frame_index"].to_numpy()):
            predicted = numpy.linalg.norm(own["exact"][:, :2], axis = 1)
            for place in verify_common.PLACES:
                chosen = moving & (frames["place"] == place).to_numpy()
                say(f"   {place:12s}: predicted median {numpy.median(predicted[chosen]):.2f} cm, demonstrated median {frames.loc[chosen, 'step_cm'].median():.2f}, "
                    f"local pace median {numpy.median(local_pace[chosen]):.2f}; predicted twist size over demonstrated (sums) "
                    f"{numpy.abs(own['exact'][chosen, 2]).sum() / frames.loc[chosen, 'twist_deg'].abs().sum():.2f}")
    else:
        say("   (cache missing: run v06_run6_mechanism.py first)")

    say("")
    say("== 3. fold models (recipe E) on the sheet they never saw: asked length against the local pace, per sheet and kind of place ==")
    fold_path = cache_folder / "2026-11-10_resnet18_chunk_stop_weight_25.npz"
    if fold_path.exists():
        held_out = numpy.load(fold_path)
        if numpy.array_equal(held_out["frame_index"], frames["frame_index"].to_numpy()):
            predicted = numpy.linalg.norm(held_out["exact"][:, :2], axis = 1)
            for sheet in sorted(frames["sheet"].unique()):
                parts = []
                for place in ["straight", "gentle bend", "sharp bend", "hairpin"]:
                    chosen = moving & (frames["place"] == place).to_numpy() & (frames["sheet"] == sheet).to_numpy()
                    if chosen.sum() >= 50:
                        parts.append(f"{place}: asked {numpy.median(predicted[chosen]):.2f} / pace {numpy.median(local_pace[chosen]):.2f} "
                                     f"(sum asked over sum demonstrated {predicted[chosen].sum() / frames.loc[chosen, 'step_cm'].sum():.2f})")
                say(f"   {sheet}: " + "; ".join(parts))
            chosen = moving & (frames["place"] != "no place").to_numpy()
            say(f"   all held-out pictures: sum of asked lengths over sum of demonstrated lengths {predicted[chosen].sum() / frames.loc[chosen, 'step_cm'].sum():.2f}; "
                f"twist sizes {numpy.abs(held_out['exact'][chosen, 2]).sum() / frames.loc[chosen, 'twist_deg'].abs().sum():.2f}; asked twist per asked cm "
                f"{numpy.abs(held_out['exact'][chosen, 2]).sum() / predicted[chosen].sum():.2f} deg against demonstrated {frames.loc[chosen, 'twist_deg'].abs().sum() / frames.loc[chosen, 'step_cm'].sum():.2f}")
    else:
        say("   (cache missing: run v05_offline_by_place.py first)")

    (Path(__file__).resolve().parent / "v09_first_bend_and_pace.out.txt").write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
