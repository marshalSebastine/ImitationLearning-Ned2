"""
Verifier's own recomputation of how the demonstrations behave per kind of place (straight, gentle bend, sharp bend,
hairpin, line end): pictures per cm of line, step length, twists in place, direction jumps, moves against the line,
passes through the hairpins, and the first 16 steps of the standard starts of sheets 03 and 04.

Read-only; prints to the screen and to v04_demonstrations.out.txt. The kinds of place come from the verifier's own
turning measure (verify_common.own_turning) on the audit's traced lines.
"""

from pathlib import Path
from typing import List

import numpy
import pandas

import verify_common

DEMONSTRATION_PARAMETERS = {
    "short_cm": 0.3,
    "twist_in_place_deg": 3.0,
    "retreat_from_deg": 120.0,
    "progress_clip_cm": [-2.0, 3.0],
    "first_steps": 16,
    "pass_half_window_cm": 3.0,
    "hairpin_middles_cm": {"04 hairpin 1": 24.9, "04 hairpin 2": 43.9, "03 first bend": 3.6}
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


if __name__ == "__main__":
    settings = dict(DEMONSTRATION_PARAMETERS)
    frames = verify_common.demonstration_rows()
    say(f"cleaned demonstration rows {len(frames)}, episodes {frames['episode_id'].nunique()}, with a line position {int(frames['arc_cm'].notna().sum())}, "
        f"stop rows {int(frames['is_stop'].sum())}")
    moving = frames[~frames["is_stop"]].copy()
    say(f"non-stop rows {len(moving)}; with a kind of place {int((moving['place'] != 'no place').sum())}")
    say("pictures per sheet and kind of place (non-stop rows):")
    say(pandas.crosstab(moving["sheet"], moving["place"]).to_string())

    # relations between consecutive rows of one episode
    same_episode = (frames["episode_id"] == frames["episode_id"].shift(-1)).to_numpy()
    next_arc = frames["arc_cm"].shift(-1).to_numpy()
    progress = numpy.where(same_episode, (next_arc - frames["arc_cm"].to_numpy()) * frames["travel_sign"].to_numpy(), numpy.nan)
    frames["progress_cm"] = progress
    frames["progress_clipped_cm"] = numpy.clip(progress, *settings["progress_clip_cm"])
    next_table = frames["step_table_deg"].shift(-1).to_numpy()
    next_length = frames["step_cm"].shift(-1).to_numpy()
    next_stop = frames["is_stop"].shift(-1).fillna(True).to_numpy().astype(bool)
    # a pair is two steps recorded one after the other (no dropped row between them), as the analyst counts them;
    # "loose" pairs also bridge a dropped row
    is_next_row = (frames["row_in_episode"].shift(-1) - frames["row_in_episode"]).to_numpy() == 1
    is_loose_pair = same_episode & ~frames["is_stop"].to_numpy() & ~next_stop
    is_pair = is_loose_pair & is_next_row
    frames["loose_jump_deg"] = numpy.where(is_loose_pair, verify_common.angle_between(frames["step_table_deg"].to_numpy(), next_table), numpy.nan)
    frames["jump_deg"] = numpy.where(is_pair, verify_common.angle_between(frames["step_table_deg"].to_numpy(), next_table), numpy.nan)
    frames["pair_both_real"] = is_pair & (frames["step_cm"].to_numpy() >= settings["short_cm"]) & (next_length >= settings["short_cm"])
    frames["pair_one_short"] = is_pair & ~frames["pair_both_real"].to_numpy()

    say("")
    say("== per kind of place, all sheets (the verifier's numbers) ==")
    rows = []
    for place in verify_common.PLACES:
        group = frames[(frames["place"] == place) & ~frames["is_stop"]]
        line_progress = group["progress_clipped_cm"].sum()
        raw_progress = group["progress_cm"].sum()
        real = group[group["step_cm"] >= settings["short_cm"]]
        rows.append({
            "place": place,
            "pictures": len(group),
            "episodes": group["episode_id"].nunique(),
            "pictures per cm (clipped progress)": round(len(group) / line_progress, 2),
            "pictures per cm (raw progress)": round(len(group) / raw_progress, 2),
            "step median cm": round(group["step_cm"].median(), 2),
            "under 0.3 cm %": round(100 * (group["step_cm"] < settings["short_cm"]).mean(), 1),
            "twists in place %": round(100 * ((group["step_cm"] < settings["short_cm"]) & (group["twist_deg"].abs() >= settings["twist_in_place_deg"])).mean(), 1),
            "twist median deg": round(group["twist_deg"].abs().median(), 2),
            "twist per cm of line": round(group["twist_deg"].abs().sum() / line_progress, 2),
            "jumps over 90 %, all pairs": round(100 * (group["jump_deg"].dropna() > 90).mean(), 1),
            "jumps over 90 %, pairs bridging dropped rows too": round(100 * (group["loose_jump_deg"].dropna() > 90).mean(), 1),
            "both real, over 90 %": round(100 * (group.loc[group["pair_both_real"], "jump_deg"] > 90).mean(), 1),
            "both real, over 120 %": round(100 * (group.loc[group["pair_both_real"], "jump_deg"] > 120).mean(), 1),
            "one short, over 90 %": round(100 * (group.loc[group["pair_one_short"], "jump_deg"] > 90).mean(), 1),
            "one short, over 120 %": round(100 * (group.loc[group["pair_one_short"], "jump_deg"] > 120).mean(), 1)
        })
    table = pandas.DataFrame(rows).set_index("place")
    say(table.T.to_string())

    say("")
    say("== jumps over 90 deg per sheet and kind of place (all pairs, %) ==")
    jitter = frames[~frames["is_stop"] & frames["jump_deg"].notna() & (frames["place"] != "no place")]
    say((100 * jitter.assign(is_jump = jitter["jump_deg"] > 90).pivot_table(index = "sheet", columns = "place", values = "is_jump", aggfunc = "mean")).round(1).to_string())
    say("pairs behind those shares:")
    say(jitter.pivot_table(index = "sheet", columns = "place", values = "jump_deg", aggfunc = "count").to_string())

    say("")
    say("== real moves (0.3 cm or more) against the line's direction of travel (more than 120 deg off it), per sheet, sharp bends ==")
    positions = pandas.read_csv(verify_common.VERIFY_PARAMETERS["positions_path"])[["frame_index", "tip_x_sheet_cm", "tip_y_sheet_cm"]]
    with_sheet = frames.merge(positions, on = "frame_index", how = "left")
    profiles = numpy.load(verify_common.VERIFY_PARAMETERS["profiles_path"])
    same = (with_sheet["episode_id"] == with_sheet["episode_id"].shift(-1)).to_numpy()
    move_x = numpy.where(same, with_sheet["tip_x_sheet_cm"].shift(-1) - with_sheet["tip_x_sheet_cm"], numpy.nan)
    move_y = numpy.where(same, with_sheet["tip_y_sheet_cm"].shift(-1) - with_sheet["tip_y_sheet_cm"], numpy.nan)
    tangent_deg = numpy.full(len(with_sheet), numpy.nan)
    for sheet, group in with_sheet.groupby("sheet"):
        points = profiles[f"points_{sheet[-2:]}"]
        index = numpy.clip(numpy.rint(group["arc_cm"].fillna(0.0).to_numpy() / 0.1).astype(int), 5, len(points) - 6)
        vector = points[index + 5] - points[index - 5]
        tangent_deg[group.index.to_numpy()] = numpy.degrees(numpy.arctan2(vector[:, 1], vector[:, 0]))
    travel_deg = numpy.where(with_sheet["travel_sign"].to_numpy() > 0, tangent_deg, tangent_deg + 180.0)
    off_line = verify_common.angle_between(numpy.degrees(numpy.arctan2(move_y, move_x)), travel_deg)
    with_sheet["off_line_deg"] = off_line
    with_sheet["moved_sheet_cm"] = numpy.hypot(move_x, move_y)
    real_moves = with_sheet[~with_sheet["is_stop"] & (with_sheet["step_cm"] >= settings["short_cm"]) & with_sheet["off_line_deg"].notna()]
    say("check of the sheet coordinates: median of (tip move in the sheet's frame minus labelled step length) "
        f"{float((real_moves['moved_sheet_cm'] - real_moves['step_cm']).median()):+.3f} cm")
    for place in verify_common.PLACES:
        group = real_moves[real_moves["place"] == place]
        say(f"all sheets, {place:12s}: against the line {100 * (group['off_line_deg'] > settings['retreat_from_deg']).mean():.1f} % of {len(group)} real moves")
    sharp = real_moves[real_moves["place"] == "sharp bend"]
    for sheet, group in sharp.groupby("sheet"):
        say(f"{sheet}, sharp bend: against the line {100 * (group['off_line_deg'] > settings['retreat_from_deg']).mean():.1f} % of {len(group)} real moves; "
            f"median twist per step {group['twist_deg'].abs().median():.1f} deg")

    say("")
    say("== complete passes through named bends (3 cm of line before and after the middle) ==")
    for name, middle in settings["hairpin_middles_cm"].items():
        sheet = f"path_{name[:2]}"
        low, high = middle - settings["pass_half_window_cm"], middle + settings["pass_half_window_cm"]
        for sign, label in [(1.0, "from end A"), (-1.0, "from end B")]:
            counts, twists, net_twists, travels, in_place = [], [], [], [], []
            for _, episode in frames[(frames["sheet"] == sheet) & (frames["travel_sign"] == sign)].groupby("episode_id"):
                arc = episode["arc_cm"]
                if not ((arc < low).any() and (arc > high).any()):
                    continue
                inside = episode[(arc >= low) & (arc <= high) & ~episode["is_stop"]]
                if len(inside) == 0:
                    continue
                counts.append(len(inside))
                twists.append(inside["twist_deg"].abs().sum())
                net_twists.append(inside["twist_deg"].sum())
                travels.append(inside["step_cm"].sum())
                in_place.append(int(((inside["step_cm"] < settings["short_cm"]) & (inside["twist_deg"].abs() >= settings["twist_in_place_deg"])).sum()))
            if counts:
                same_sign = max((numpy.array(net_twists) > 0).mean(), (numpy.array(net_twists) < 0).mean())
                say(f"{name} {label}: {len(counts)} passes; steps median {numpy.median(counts):.1f} ({min(counts)} to {max(counts)}); tip travel "
                    f"{numpy.median(travels):.1f} cm; twists in place {numpy.median(in_place):.1f}; twist sum of sizes {numpy.median(twists):.0f} deg, net "
                    f"{numpy.median(net_twists):+.0f} deg, same sign in {100 * same_sign:.0f} % of passes")

    say("")
    say(f"== the first {settings['first_steps']} steps of the standard starts from end A ==")
    for sheet in ["path_03", "path_04"]:
        chosen = frames[(frames["sheet"] == sheet) & (frames["start_kind"] == "STD") & (frames["start_end"] == "A")]
        travel, twist_sum, twist_net = [], [], []
        for _, episode in chosen.groupby("episode_id"):
            first = episode[~episode["is_stop"]].iloc[:settings["first_steps"]]
            travel.append(first["step_cm"].sum())
            twist_sum.append(first["twist_deg"].abs().sum())
            twist_net.append(first["twist_deg"].sum())
        say(f"{sheet}: {len(travel)} episodes; tip travel median {numpy.median(travel):.1f} cm ({min(travel):.1f} to {max(travel):.1f}); "
            f"twist sum of sizes median {numpy.median(twist_sum):.0f} deg ({min(twist_sum):.0f} to {max(twist_sum):.0f}); net median {numpy.median(twist_net):+.0f} deg")

    say("")
    say("== the scout's step lengths ==")
    raw = pandas.read_csv(verify_common.TRAINING_DATA_PARAMETERS["processed_folder"] / "frames.csv", low_memory = False) if isinstance(
        verify_common.TRAINING_DATA_PARAMETERS["processed_folder"], Path) else pandas.read_csv(
        Path(verify_common.TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", low_memory = False)
    raw_moving = raw[raw["action"] != "stop"]
    raw_length = 100.0 * numpy.hypot(raw_moving["tip_forward_m"].fillna(0.0), raw_moving["tip_left_m"].fillna(0.0))
    say(f"all non-stop rows of frames.csv ({len(raw_moving)}): median {raw_length.median():.2f} cm, mean {raw_length.mean():.2f}, under 0.5 cm "
        f"{100 * (raw_length < 0.5).mean():.1f} %, under 0.2 cm {100 * (raw_length < 0.2).mean():.1f} %")
    cleaned = frames[~frames["is_stop"]]
    say(f"cleaned non-stop rows ({len(cleaned)}): median {cleaned['step_cm'].median():.2f} cm, mean {cleaned['step_cm'].mean():.2f}, under 0.5 cm "
        f"{100 * (cleaned['step_cm'] < 0.5).mean():.1f} %, under 0.2 cm {100 * (cleaned['step_cm'] < 0.2).mean():.1f} %; steps with a twist of 6 deg or more: "
        f"median {cleaned.loc[cleaned['twist_deg'].abs() >= 6.0, 'step_cm'].median():.2f} cm")
    per_sheet = cleaned.groupby("sheet").apply(lambda group: group["twist_deg"].abs().sum() / group["step_cm"].sum())
    say("twist per cm of tip travel per sheet (sum of sizes over sum of step lengths): " + ", ".join(f"{sheet[-2:]}: {value:.2f}" for sheet, value in per_sheet.items()))

    (Path(__file__).resolve().parent / "v04_demonstrations.out.txt").write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
