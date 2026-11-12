"""
Verifier's own scoring of saved predictions per kind of place: the seeds of the exact recipe against the chunk
recipes, the backup model's move names, and the frozen-feature history probes of 9.11.26.

Only saved predictions are read (no model is run). Path within 45 degrees: the answer's direction against where the
demonstration's tip was 3 steps later, on rows that are not a stop and whose tip moved at least 1.5 cm in those steps.
Prints to the screen and to v07_saved_predictions.out.txt.
"""

from pathlib import Path
from typing import Dict, List

import numpy
import pandas

import verify_common

SAVED_PARAMETERS = {
    "gpu_folder": verify_common.ROOT_FOLDER / "04_runs" / "gpu_ki014",
    "probe_folder": verify_common.ROOT_FOLDER / "04_runs" / "2026-11-09_frozen_feature_probes" / "predictions",
    "exact_runs": ["2026-11-09_resnet18_final_candidate", "2026-11-09_resnet18_final_candidate_seed_2", "2026-11-10_resnet18_final_candidate_seed_3",
                   "2026-11-10_resnet18_final_candidate_seed_4", "2026-11-10_resnet18_final_candidate_seed_5"],
    "chunk_runs": ["2026-11-10_resnet18_final_candidate_chunk", "2026-11-10_resnet18_final_candidate_chunk_seed_2",
                   "2026-11-10_resnet18_chunk_stop_weight_25", "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"],
    "names_run": "2026-11-09_resnet18_backup_names_plus_exact",
    "history_probes": ["one_picture", "plus_previous_picture", "plus_change_since_previous", "plus_pictures_2_and_4_back"],
    "path_minimum_cm": 1.5
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


def saved_columns(folder: Path, prefix: str, frames: pandas.DataFrame, keys: List[str]) -> pandas.DataFrame:
    """
    Return the saved held-out predictions of the six folds, matched to the rows of ``frames`` by frame index.

    Parameters
    ----------
    folder : pathlib.Path
        Where the npz files are.
    prefix : str
        File name before "without_path_NN.npz".
    frames : pandas.DataFrame
        The demonstration rows.
    keys : list of str
        The arrays wanted.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``frames``; rows that are missing in the saved files hold NaN.
    """

    parts = []
    for sheet in sorted(frames["sheet"].unique()):
        saved = numpy.load(folder / f"{prefix}without_{sheet}.npz")
        part = pandas.DataFrame({key: saved[key] for key in keys if key in saved.files}, index = saved["frame_index"])
        parts.append(part)
    together = pandas.concat(parts)
    together = together[~together.index.duplicated()]
    return together.reindex(frames["frame_index"].to_numpy()).set_index(frames.index)


def by_place(frames: pandas.DataFrame, angle_deg: numpy.ndarray) -> Dict[str, float]:
    """Return path within 45 degrees per kind of place and overall for one answer."""

    is_usable = frames["is_scored"].to_numpy() & ~numpy.isnan(angle_deg)
    is_hit = verify_common.angle_between(angle_deg, frames["path_deg"].to_numpy()) <= 45.0
    result = {}
    for place in verify_common.PLACES:
        chosen = is_usable & (frames["place"] == place).to_numpy()
        result[place] = round(float(is_hit[chosen].mean()), 3)
    result["all places"] = round(float(is_hit[is_usable & (frames["place"] != "no place").to_numpy()].mean()), 3)
    result["rows"] = int((is_usable & (frames["place"] != "no place").to_numpy()).sum())
    return result


if __name__ == "__main__":
    settings = dict(SAVED_PARAMETERS)
    frames = verify_common.demonstration_rows()
    ahead = verify_common.tips_ahead(frames, 3)
    frames["path_deg"] = numpy.degrees(numpy.arctan2(ahead[:, 2, 1], ahead[:, 2, 0]))
    frames["path_cm"] = numpy.linalg.norm(ahead[:, 2, :], axis = 1)
    frames["is_scored"] = ~frames["is_stop"] & (frames["path_cm"] >= settings["path_minimum_cm"])

    say("== 1. seeds: path within 45 deg per kind of place (saved predictions; chunk runs: the first chunk step and the point 3 steps ahead) ==")
    rows = []
    for run in settings["exact_runs"]:
        saved = saved_columns(settings["gpu_folder"] / run, "predictions_", frames, ["angle_deg"])
        rows.append({"run": run[11:], "answer": "exact step", **by_place(frames, saved["angle_deg"].to_numpy())})
    for run in settings["chunk_runs"]:
        saved = saved_columns(settings["gpu_folder"] / run, "predictions_", frames, ["angle_deg", "angle_vector_deg"])
        rows.append({"run": run[11:], "answer": "chunk, first step", **by_place(frames, saved["angle_deg"].to_numpy())})
        rows.append({"run": run[11:], "answer": "chunk, 3 ahead", **by_place(frames, saved["angle_vector_deg"].to_numpy())})
    table = pandas.DataFrame(rows)
    say(table.to_string(index = False))
    exact_rows = table[table["answer"] == "exact step"]
    first_rows = table[table["answer"] == "chunk, first step"]
    ahead_rows = table[table["answer"] == "chunk, 3 ahead"]
    for place in ["sharp bend", "hairpin", "all places"]:
        say(f"{place}: exact seeds {exact_rows[place].min():.3f} to {exact_rows[place].max():.3f}; chunk-trained, first step {first_rows[place].min():.3f} to "
            f"{first_rows[place].max():.3f}; chunk 3 ahead minus first step, per model: "
            + ", ".join(f"{100 * (later - first):+.1f}" for later, first in zip(ahead_rows[place], first_rows[place])) + " points")

    say("")
    say("== 2. the backup model's move names (saved predictions) ==")
    names = saved_columns(settings["gpu_folder"] / settings["names_run"], "predictions_", frames, ["angle_deg", "angle_expected_deg", "angle_vector_deg"])
    for label, key in [("most likely name", "angle_deg"), ("mean direction of the names", "angle_expected_deg"), ("exact step of the same model", "angle_vector_deg")]:
        say(f"{label:30s}: {by_place(frames, names[key].to_numpy())}")
    same_episode = ((frames["episode_id"] == frames["episode_id"].shift(-1)) & ((frames["row_in_episode"].shift(-1) - frames["row_in_episode"]) == 1)).to_numpy()
    is_pair = same_episode & ~frames["is_stop"].to_numpy() & ~frames["is_stop"].shift(-1).fillna(True).to_numpy().astype(bool) & (frames["place"] != "no place").to_numpy()
    heading = frames["picture_heading_deg"].to_numpy()
    for label, key in [("most likely name", "angle_deg"), ("mean direction of the names", "angle_expected_deg"), ("exact step of the same model", "angle_vector_deg")]:
        on_table = heading + names[key].to_numpy()
        jump = verify_common.angle_between(on_table, numpy.roll(on_table, -1))
        say(f"{label:30s}: reverses by more than 90 deg (on the table) between consecutive pictures on {100 * float((jump[is_pair] > 90).mean()):.1f} % of {int(is_pair.sum())} pairs")

    say("")
    say("== 3. frozen-feature history probes of 9.11.26 (names heads; mean direction of the names) ==")
    previous_table = numpy.where((frames["episode_id"] == frames["episode_id"].shift(1)).to_numpy(), frames["step_table_deg"].shift(1).to_numpy(), numpy.nan)
    previous_picture = (previous_table - heading + 180.0) % 360.0 - 180.0
    is_kept_direction = verify_common.angle_between(frames["path_deg"].to_numpy(), previous_picture) <= 45.0
    has_previous = ~numpy.isnan(previous_picture)
    for probe in settings["history_probes"]:
        saved = saved_columns(settings["probe_folder"], f"{probe}_", frames, ["angle_deg", "angle_expected_deg"])
        angle = saved["angle_expected_deg"].to_numpy()
        is_usable = frames["is_scored"].to_numpy() & ~numpy.isnan(angle) & (frames["place"] != "no place").to_numpy()
        is_hit = verify_common.angle_between(angle, frames["path_deg"].to_numpy()) <= 45.0
        kept = is_usable & has_previous & is_kept_direction
        changed = is_usable & has_previous & ~is_kept_direction
        near_previous = verify_common.angle_between(angle, previous_picture) <= 30.0
        say(f"{probe:28s}: {by_place(frames, angle)} | where the demonstration kept its direction {float(is_hit[kept].mean()):.3f} ({int(kept.sum())} rows), "
            f"where it changed {float(is_hit[changed].mean()):.3f} ({int(changed.sum())} rows) | answer within 30 deg of the previous demonstrated step "
            f"{100 * float(near_previous[is_usable & has_previous].mean()):.1f} %")

    (Path(__file__).resolve().parent / "v07_saved_predictions.out.txt").write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
