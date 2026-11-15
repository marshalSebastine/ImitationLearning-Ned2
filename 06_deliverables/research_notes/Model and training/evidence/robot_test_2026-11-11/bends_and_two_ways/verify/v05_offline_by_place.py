"""
Verifier's own offline check of the fold models at bends: every model answers only the pictures of the sheet it was
trained without. Recipe E (chunk, stop weight 25; the recipe of the model used on the robot), first seed.

Checked: the predicted step length and the direction error per kind of place, reversals between consecutive
predictions, the predicted twist at hairpins, what a short answer means, and the chunk's point 3 and 5 steps ahead
against the exact step.

Read-only; no training. The forward passes are cached in the temporary folder (about 3 minutes the first time).
Prints to the screen and to v05_offline_by_place.out.txt.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy
import pandas

import verify_common

OFFLINE_PARAMETERS = {
    "run_folder": "2026-11-10_resnet18_chunk_stop_weight_25",
    "run_name": "resnet18_chunk_stop_weight_25",
    "path_steps": 3,
    "path_minimum_cm": 1.5,
    "within_deg": 45.0,
    "short_cm": 0.3
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


def fold_outputs(frames: pandas.DataFrame, parameters: Dict) -> Dict[str, numpy.ndarray]:
    """
    Return the held-out outputs of the six fold models, row for row with ``frames`` (cached after the first run).
    """

    cache_folder = verify_common.VERIFY_PARAMETERS["cache_folder"]
    cache_folder.mkdir(parents = True, exist_ok = True)
    cache_path = cache_folder / f"{parameters['run_folder']}.npz"
    if cache_path.exists():
        cached = numpy.load(cache_path)
        if numpy.array_equal(cached["frame_index"], frames["frame_index"].to_numpy()):
            return {name: cached[name] for name in cached.files}
    exact = numpy.zeros((len(frames), 5), dtype = numpy.float32)
    chunk = numpy.zeros((len(frames), 10), dtype = numpy.float32)
    stop = numpy.zeros(len(frames), dtype = numpy.float32)
    run_folder = verify_common.VERIFY_PARAMETERS["fold_runs_folder"] / parameters["run_folder"]
    for sheet in sorted(frames["sheet"].unique()):
        rows = numpy.flatnonzero((frames["sheet"] == sheet).to_numpy())
        loaded = verify_common.load_saved_model(parameters["run_name"], run_folder, f"without_{sheet}")
        outputs = verify_common.forward_pass(loaded, verify_common.cached_training_pictures(frames["frame_index"].to_numpy()[rows]))
        exact[rows] = outputs["exact"]
        chunk[rows] = outputs["chunk"]
        stop[rows] = outputs["stop"][:, 0]
        # the saved predictions of the same fold: frame_index order may differ, so match by frame index
        saved = numpy.load(run_folder / f"predictions_without_{sheet}.npz")
        order = pandas.Series(numpy.arange(len(saved["frame_index"])), index = saved["frame_index"]).reindex(frames["frame_index"].to_numpy()[rows]).to_numpy()
        is_matched = ~numpy.isnan(order)
        own_first_chunk_deg = numpy.degrees(numpy.arctan2(outputs["chunk"][:, 1], outputs["chunk"][:, 0]))
        difference = verify_common.angle_between(own_first_chunk_deg[is_matched], saved["angle_deg"][order[is_matched].astype(int)])
        twist_difference = numpy.abs(outputs["exact"][is_matched, 2] - saved["twist_deg"][order[is_matched].astype(int)])
        print(f"   fold without_{sheet}: {len(rows)} pictures, {int(is_matched.sum())} found in the saved predictions; largest difference to the saved "
              f"angle_deg (first chunk step) {float(difference.max()):.4f} deg, to the saved twist {float(twist_difference.max()):.4f} deg", flush = True)
    result = {"frame_index": frames["frame_index"].to_numpy(), "exact": exact, "chunk": chunk, "stop": stop}
    numpy.savez(cache_path, **result)
    return result


def share_within(predicted_deg: numpy.ndarray, true_deg: numpy.ndarray, limit_deg: float) -> float:
    """Return the share of rows whose predicted direction lies within the limit of the true one."""

    return float((verify_common.angle_between(predicted_deg, true_deg) <= limit_deg).mean()) if len(predicted_deg) else float("nan")


if __name__ == "__main__":
    settings = dict(OFFLINE_PARAMETERS)
    if len(sys.argv) > 2:
        settings["run_folder"], settings["run_name"] = sys.argv[1], sys.argv[2]
    frames = verify_common.demonstration_rows()
    outputs = fold_outputs(frames, settings)
    ahead = verify_common.tips_ahead(frames, 5)
    frames["exact_forward_cm"] = outputs["exact"][:, 0]
    frames["exact_left_cm"] = outputs["exact"][:, 1]
    frames["exact_cm"] = numpy.hypot(frames["exact_forward_cm"], frames["exact_left_cm"])
    frames["exact_deg"] = numpy.degrees(numpy.arctan2(frames["exact_left_cm"], frames["exact_forward_cm"]))
    frames["exact_twist_deg"] = outputs["exact"][:, 2]
    chunk = outputs["chunk"].reshape(len(frames), 5, 2)
    for step in [1, 3, 5]:
        frames[f"chunk_{step}_deg"] = numpy.degrees(numpy.arctan2(chunk[:, step - 1, 1], chunk[:, step - 1, 0]))
        frames[f"chunk_{step}_cm"] = numpy.linalg.norm(chunk[:, step - 1, :], axis = 1)
    path_vector = ahead[:, settings["path_steps"] - 1, :]
    frames["path_deg"] = numpy.degrees(numpy.arctan2(path_vector[:, 1], path_vector[:, 0]))
    frames["path_cm"] = numpy.linalg.norm(path_vector, axis = 1)
    frames["is_scored"] = ~frames["is_stop"] & (frames["path_cm"] >= settings["path_minimum_cm"])
    say(f"run {settings['run_folder']}: {len(frames)} held-out pictures; scored on the path direction ({settings['path_steps']} steps ahead, at least "
        f"{settings['path_minimum_cm']} cm): {int(frames['is_scored'].sum())}")
    scored = frames[frames["is_scored"]]
    say(f"all sheets: path within 45 deg: exact step {share_within(scored['exact_deg'], scored['path_deg'], 45.0):.3f}, first chunk step "
        f"{share_within(scored['chunk_1_deg'], scored['path_deg'], 45.0):.3f}, chunk 3 ahead {share_within(scored['chunk_3_deg'], scored['path_deg'], 45.0):.3f}, "
        f"chunk 5 ahead {share_within(scored['chunk_5_deg'], scored['path_deg'], 45.0):.3f}, always forward {share_within(numpy.zeros(len(scored)), scored['path_deg'], 45.0):.3f}")

    say("")
    say("== 1. per kind of place (held-out sheets) ==")
    same_episode = (frames["episode_id"] == frames["episode_id"].shift(-1)).to_numpy() & ((frames["row_in_episode"].shift(-1) - frames["row_in_episode"]).to_numpy() == 1)
    next_stop = frames["is_stop"].shift(-1).fillna(True).to_numpy().astype(bool)
    is_pair = same_episode & ~frames["is_stop"].to_numpy() & ~next_stop
    # consecutive predictions compared on the table (picture heading added), as the demonstrated steps are
    predicted_table = frames["picture_heading_deg"].to_numpy() + frames["exact_deg"].to_numpy()
    frames["predicted_jump_deg"] = numpy.where(is_pair, verify_common.angle_between(predicted_table, numpy.roll(predicted_table, -1)), numpy.nan)
    frames["predicted_jump_picture_deg"] = numpy.where(is_pair, verify_common.angle_between(frames["exact_deg"].to_numpy(), numpy.roll(frames["exact_deg"].to_numpy(), -1)), numpy.nan)
    frames["demonstrated_jump_deg"] = numpy.where(is_pair, verify_common.angle_between(frames["step_table_deg"].to_numpy(), numpy.roll(frames["step_table_deg"].to_numpy(), -1)), numpy.nan)
    # the demonstration's local pace: length of the mean of the five steps centred on the row (same episode)
    centred = numpy.zeros((len(frames), 2))
    counts = numpy.zeros(len(frames))
    step_vectors = frames[["step_forward_cm", "step_left_cm"]].to_numpy()
    table_angle = numpy.radians(frames["step_table_deg"].to_numpy())
    table_vectors = frames["step_cm"].to_numpy()[:, None] * numpy.stack([numpy.cos(table_angle), numpy.sin(table_angle)], axis = 1)
    episode_codes = pandas.factorize(frames["episode_id"])[0]
    for offset in [-2, -1, 0, 1, 2]:
        shifted = numpy.roll(table_vectors, -offset, axis = 0)
        is_same = (numpy.roll(episode_codes, -offset) == episode_codes) & ~numpy.roll(frames["is_stop"].to_numpy(), -offset)
        if offset > 0:
            is_same[-offset:] = False
        elif offset < 0:
            is_same[:-offset] = False
        centred += numpy.where(is_same[:, None], shifted, 0.0)
        counts += is_same
    frames["local_pace_cm"] = numpy.linalg.norm(centred / numpy.maximum(counts, 1)[:, None], axis = 1)
    rows = []
    for place in verify_common.PLACES + ["all places"]:
        group = frames[~frames["is_stop"] & ((frames["place"] == place) if place != "all places" else (frames["place"] != "nothing"))]
        scored_group = group[group["is_scored"]]
        rows.append({
            "place": place,
            "pictures": len(group),
            "predicted length median cm": round(group["exact_cm"].median(), 2),
            "predicted under 0.3 cm %": round(100 * (group["exact_cm"] < settings["short_cm"]).mean(), 1),
            "demonstrated length median cm": round(group["step_cm"].median(), 2),
            "local pace (mean of 5 steps) median cm": round(group["local_pace_cm"].median(), 2),
            "scored pictures": len(scored_group),
            "within 45: exact step": round(share_within(scored_group["exact_deg"], scored_group["path_deg"], 45.0), 3),
            "within 45: first chunk step": round(share_within(scored_group["chunk_1_deg"], scored_group["path_deg"], 45.0), 3),
            "within 45: chunk 3 ahead": round(share_within(scored_group["chunk_3_deg"], scored_group["path_deg"], 45.0), 3),
            "within 45: chunk 5 ahead": round(share_within(scored_group["chunk_5_deg"], scored_group["path_deg"], 45.0), 3),
            "within 45: always forward": round(share_within(numpy.zeros(len(scored_group)), scored_group["path_deg"], 45.0), 3),
            "predicted reversals over 90 % (table)": round(100 * (group["predicted_jump_deg"].dropna() > 90).mean(), 2),
            "predicted reversals over 90 % (picture)": round(100 * (group["predicted_jump_picture_deg"].dropna() > 90).mean(), 2),
            "demonstrated reversals over 90 %": round(100 * (group["demonstrated_jump_deg"].dropna() > 90).mean(), 2),
            "twist: sum predicted / sum demonstrated": round(group["exact_twist_deg"].abs().sum() / group["twist_deg"].abs().sum(), 2)
        })
    say(pandas.DataFrame(rows).set_index("place").T.to_string())

    say("")
    say("== 2. hairpins: does the predicted twist go into the bend? ==")
    hairpin = frames[(frames["place"] == "hairpin") & ~frames["is_stop"]]
    # the forearm twist that follows a bend: taken from the demonstrations themselves (sign of the demonstrated net twist per travel direction and bend)
    demonstrated_sign = numpy.sign(hairpin.groupby(["travel_sign", numpy.sign(hairpin["bend_turn_deg"])])["twist_deg"].transform("sum"))
    into_predicted = hairpin["exact_twist_deg"] * demonstrated_sign
    into_demonstrated = hairpin["twist_deg"] * demonstrated_sign
    say(f"hairpin pictures {len(hairpin)}: predicted twist into the bend (over 1 deg) on {100 * (into_predicted > 1.0).mean():.1f} %, out of it (under -1 deg) on "
        f"{100 * (into_predicted < -1.0).mean():.1f} %; demonstrated {100 * (into_demonstrated > 1.0).mean():.1f} % and {100 * (into_demonstrated < -1.0).mean():.1f} %; "
        f"mean size predicted {hairpin['exact_twist_deg'].abs().mean():.2f} deg, demonstrated {hairpin['twist_deg'].abs().mean():.2f} deg")

    say("")
    say("== 3. what a short answer means (held-out, non-stop pictures) ==")
    moving = frames[~frames["is_stop"]]
    short = moving[moving["exact_cm"] < settings["short_cm"]]
    say(f"answers under 0.3 cm: {len(short)} of {len(moving)} ({100 * len(short) / len(moving):.1f} %)")
    say(f"   the demonstration then made a real move of 0.7 cm or more on {100 * (short['step_cm'] >= 0.7).mean():.0f} %, a twist in place (under 0.3 cm, "
        f"3 deg or more) on {100 * ((short['step_cm'] < 0.3) & (short['twist_deg'].abs() >= 3.0)).mean():.0f} %; at hairpins "
        f"{100 * ((short.loc[short['place'] == 'hairpin', 'step_cm'] < 0.3) & (short.loc[short['place'] == 'hairpin', 'twist_deg'].abs() >= 3.0)).mean():.0f} % "
        f"(of {int((short['place'] == 'hairpin').sum())})")
    real_after_short = short[short["step_cm"] >= 0.7]
    say(f"   for those real moves the predicted direction is more than 90 deg off the demonstrated step on "
        f"{100 * (verify_common.angle_between(real_after_short['exact_deg'], real_after_short['step_picture_deg']) > 90).mean():.1f} %")
    scored = frames[frames["is_scored"]]
    for low, high in [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 9.0)]:
        group = scored[(scored["exact_cm"] >= low) & (scored["exact_cm"] < high)]
        say(f"   predicted length {low:.1f} to {high:.1f} cm: {len(group):5d} scored pictures, path within 45 deg {share_within(group['exact_deg'], group['path_deg'], 45.0):.3f}")

    say("")
    say("== 4. per held-out sheet: exact step against the chunk's point 3 steps ahead (path within 45 deg) ==")
    for sheet, group in scored.groupby("sheet"):
        say(f"   {sheet}: {len(group):5d} pictures; exact {share_within(group['exact_deg'], group['path_deg'], 45.0):.3f}, chunk 3 ahead "
            f"{share_within(group['chunk_3_deg'], group['path_deg'], 45.0):.3f}, chunk 5 ahead {share_within(group['chunk_5_deg'], group['path_deg'], 45.0):.3f}; "
            f"predicted length median {frames.loc[(frames['sheet'] == sheet) & ~frames['is_stop'], 'exact_cm'].median():.2f} cm against demonstrated "
            f"{frames.loc[(frames['sheet'] == sheet) & ~frames['is_stop'], 'step_cm'].median():.2f}")

    say("")
    say("== 5. the first bend of sheet 03 from end A, held-out model (puck within the first 8 cm of line, standard starts) ==")
    bend = frames[(frames["sheet"] == "path_03") & (frames["travel_sign"] > 0) & (frames["arc_cm"] <= 8.0) & ~frames["is_stop"] & (frames["start_kind"] == "STD")]
    say(f"   pictures {len(bend)}; predicted length median {bend['exact_cm'].median():.2f} cm (mean {bend['exact_cm'].mean():.2f}), demonstrated median "
        f"{bend['step_cm'].median():.2f} (mean {bend['step_cm'].mean():.2f}); predicted twist mean {bend['exact_twist_deg'].mean():+.2f} deg, demonstrated "
        f"{bend['twist_deg'].mean():+.2f}; sum of predicted lengths over sum of demonstrated {bend['exact_cm'].sum() / bend['step_cm'].sum():.2f}; "
        f"length of the summed predicted vectors over the summed demonstrated lengths is not comparable (different frames per picture)")
    out_name = "v05_offline_by_place.out.txt" if settings["run_folder"] == OFFLINE_PARAMETERS["run_folder"] else f"v05_offline_by_place.{settings['run_folder']}.out.txt"
    (Path(__file__).resolve().parent / out_name).write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
