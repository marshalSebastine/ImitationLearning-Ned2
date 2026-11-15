"""
Two of the options, checked on held-out demonstration pictures: carrying out several steps of one chunk, and a short
picture history as input.

Part 1, several steps of one chunk (recipe E, the fold models' chunks from s02_model_outputs.py):
    how far the chunk's place 1 to 5 steps ahead is from where the demonstration's tip really was, per kind of place;
    the same place reached by looking again after every step (the first chunk steps of the next pictures, turned into
    the first picture's frame and added up), so "one look, three steps" stands beside "three looks, one step each" on
    the same pictures. Offline both follow the demonstrator's pictures, so this says how good the chunk's later steps
    are, not what happens once the robot's own errors feed back;
    how much forearm twist the demonstrations made over 2 and 3 steps, which a chunk carried out blindly would have to
    make without a new picture (the chunk holds no twist of its own; the model gives one twist per picture).

Part 2, a short picture history (the frozen-feature probes of 9.11.26, 04_runs/2026-11-09_frozen_feature_probes, names
head): the saved held-out predictions of "one_picture", "plus_previous_picture", "plus_change_since_previous" and
"plus_pictures_2_and_4_back", scored per kind of place, and split into pictures where the demonstration kept its
direction and pictures where it changed it: a history model that only copies the last move gains on the first and
loses on the second.

Run from this folder:  python s06_chunk_steps_and_history.py
Output: s06_chunk_steps_and_history.out.txt, s06_chunk_error.csv, s06_history_by_place.csv.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas

import common

CHUNK_PARAMETERS = {
    "probe_folder": common.ROOT_FOLDER / "04_runs" / "2026-11-09_frozen_feature_probes" / "predictions",
    "history_variants": ["one_picture", "plus_previous_picture", "plus_change_since_previous", "plus_pictures_2_and_4_back"],
    "change_from_deg": 45.0,
    "copy_within_deg": 30.0
}



def rotated(
        vector: numpy.ndarray,
        angle_deg: numpy.ndarray
) -> numpy.ndarray:
    """Return (forward, left) vectors turned counter-clockwise by ``angle_deg`` (a vector seen from a frame turned by minus that angle)."""

    radians = numpy.radians(angle_deg)
    cosine, sine = numpy.cos(radians), numpy.sin(radians)
    return numpy.stack([cosine * vector[:, 0] - sine * vector[:, 1], sine * vector[:, 0] + cosine * vector[:, 1]], axis = 1)



def chunk_part(
        frames: pandas.DataFrame,
        path: pandas.DataFrame,
        report,
        parameters: Dict
) -> None:
    """Print how good the later steps of a chunk are, against looking again after every step."""

    saved = numpy.load(Path(parameters["cache_folder"]) / "fold_outputs_recipe_e.npz")
    chunk = saved["chunk"].reshape(len(frames), 5, 2)
    is_scored = (~frames["is_stop"]).to_numpy() & (frames["place"] != "no fit").to_numpy()
    heading = frames["picture_heading_deg"].to_numpy()
    episode = frames["episode_id"].to_numpy()
    row_number = frames["row_in_episode"].to_numpy()
    places = frames["place"].to_numpy()
    twist = frames["twist_deg"].to_numpy()

    # "looking again": the first chunk step of pictures t, t+1, t+2 ..., each turned into the frame of picture t
    looked = numpy.zeros((len(frames), 5, 2))
    is_complete = numpy.ones((len(frames), 5), dtype = bool)
    running = numpy.zeros((len(frames), 2))
    twist_sum = numpy.zeros((len(frames), 5))
    for ahead in range(5):
        positions = numpy.arange(len(frames)) + ahead
        is_inside = positions < len(frames)
        positions = numpy.clip(positions, 0, len(frames) - 1)
        is_same = is_inside & (episode[positions] == episode) & (row_number[positions] - row_number == ahead) & ~frames["is_stop"].to_numpy()[positions]
        step = rotated(chunk[positions, 0, :], heading[positions] - heading)
        running = running + numpy.where(is_same[:, None], step, 0.0)
        looked[:, ahead, :] = running
        is_complete[:, ahead] = is_same if ahead == 0 else (is_complete[:, ahead - 1] & is_same)
        twist_sum[:, ahead] = (twist_sum[:, ahead - 1] if ahead else 0.0) + numpy.where(is_same, twist[positions], 0.0)

    rows = []
    for place in common.PLACE_NAMES + ["all places"]:
        for ahead in range(1, 6):
            mask = is_scored & is_complete[:, ahead - 1] & ((places == place) | (place == "all places"))
            true = numpy.stack([path[f"ahead_{ahead}_forward_cm"].to_numpy(), path[f"ahead_{ahead}_left_cm"].to_numpy()], axis = 1)
            error_chunk = numpy.linalg.norm(chunk[mask, ahead - 1] - true[mask], axis = 1)
            error_looked = numpy.linalg.norm(looked[mask, ahead - 1] - true[mask], axis = 1)
            distance = numpy.linalg.norm(true[mask], axis = 1)
            # the step from place ahead-1 to place ahead, as the chunk draws it and as the demonstration made it
            previous_true = numpy.zeros_like(true) if ahead == 1 else numpy.stack([path[f"ahead_{ahead - 1}_forward_cm"].to_numpy(), path[f"ahead_{ahead - 1}_left_cm"].to_numpy()], axis = 1)
            previous_chunk = numpy.zeros_like(true) if ahead == 1 else chunk[:, ahead - 2]
            true_increment = (true - previous_true)[mask]
            chunk_increment = (chunk[:, ahead - 1] - previous_chunk)[mask]
            is_real = numpy.linalg.norm(true_increment, axis = 1) >= parameters["short_step_cm"]
            increment_error = common.angle_difference_deg(numpy.degrees(numpy.arctan2(chunk_increment[:, 1], chunk_increment[:, 0])),
                                                          numpy.degrees(numpy.arctan2(true_increment[:, 1], true_increment[:, 0])))
            rows.append({"place": place, "steps ahead": ahead, "pictures": int(mask.sum()),
                         "demonstrated distance, median cm": round(float(numpy.median(distance)), 2),
                         "chunk error, median cm": round(float(numpy.median(error_chunk)), 2),
                         "looking again error, median cm": round(float(numpy.median(error_looked)), 2),
                         "chunk error, 90th percentile cm": round(float(numpy.percentile(error_chunk, 90)), 2),
                         "looking again error, 90th percentile cm": round(float(numpy.percentile(error_looked, 90)), 2),
                         "this step's direction within 45 (real moves)": round(float((increment_error[is_real] <= 45).mean()), 3),
                         "this step's length, chunk median cm": round(float(numpy.median(numpy.linalg.norm(chunk_increment, axis = 1))), 2),
                         "this step's length, demonstrated median cm": round(float(numpy.median(numpy.linalg.norm(true_increment, axis = 1))), 2),
                         "demonstrated twist summed to here, median abs deg": round(float(numpy.median(numpy.abs(twist_sum[mask, ahead - 1]))), 1),
                         "demonstrated twist summed to here, 90th percentile abs deg": round(float(numpy.percentile(numpy.abs(twist_sum[mask, ahead - 1]), 90)), 1)})
    table = pandas.DataFrame(rows)
    table.to_csv(common.ANALYSIS_FOLDER / "s06_chunk_error.csv", index = False)
    report("== part 1: the chunk's place K steps ahead against where the demonstration's tip was (recipe E, held-out sheets) ==")
    report("(chunk error: one look, K steps; looking again: the first chunk step of each of the K pictures, added up; both in cm from the true place)")
    report(table.to_string(index = False))



def history_part(
        frames: pandas.DataFrame,
        path: pandas.DataFrame,
        report,
        parameters: Dict
) -> None:
    """Print the frozen history probes per kind of place and per kept or changed direction."""

    frames = frames.copy()
    frames["path_deg"] = path["path_deg"].to_numpy()
    frames["is_path_row"] = path["is_path_row"].to_numpy()
    # the demonstrated step before this picture, in this picture's frame
    previous_table = frames.groupby("episode_id")["step_table_deg"].shift(1)
    previous_length = frames.groupby("episode_id")["step_cm"].shift(1)
    frames["previous_deg"] = previous_table - frames["picture_heading_deg"]
    frames["has_previous"] = previous_table.notna() & (previous_length >= parameters["short_step_cm"])
    frames["is_changed"] = common.angle_difference_deg(frames["previous_deg"].fillna(0.0), frames["path_deg"]) > parameters["change_from_deg"]
    indexed = frames.set_index("frame_index")

    rows = []
    for variant in parameters["history_variants"]:
        parts = []
        for sheet in sorted(frames["sheet"].unique()):
            saved = numpy.load(Path(parameters["probe_folder"]) / f"{variant}_without_{sheet}.npz")
            parts.append(pandas.DataFrame({"frame_index": saved["frame_index"], "top_deg": saved["angle_deg"], "mean_deg": saved["angle_expected_deg"]}))
        predictions = pandas.concat(parts).set_index("frame_index")
        joined = indexed.join(predictions, how = "inner")
        joined = joined[joined["is_path_row"] & (joined["place"] != "no fit")]
        for place in common.PLACE_NAMES + ["all places"]:
            part = joined if place == "all places" else joined[joined["place"] == place]
            with_previous = part[part["has_previous"]]
            kept = with_previous[~with_previous["is_changed"]]
            changed = with_previous[with_previous["is_changed"]]
            rows.append({
                "input": variant, "place": place, "path rows": int(len(part)),
                "path within 45, mean direction": round(float((common.angle_difference_deg(part["mean_deg"], part["path_deg"]) <= 45).mean()), 3),
                "path within 45, most likely name": round(float((common.angle_difference_deg(part["top_deg"], part["path_deg"]) <= 45).mean()), 3),
                "rows where the direction was kept": int(len(kept)),
                "within 45 there": round(float((common.angle_difference_deg(kept["mean_deg"], kept["path_deg"]) <= 45).mean()), 3) if len(kept) else numpy.nan,
                "rows where it changed": int(len(changed)),
                "within 45 there ": round(float((common.angle_difference_deg(changed["mean_deg"], changed["path_deg"]) <= 45).mean()), 3) if len(changed) else numpy.nan,
                "answer within 30 deg of the previous demonstrated step": round(float((common.angle_difference_deg(with_previous["mean_deg"], with_previous["previous_deg"]) <= parameters["copy_within_deg"]).mean()), 3)
            })
    table = pandas.DataFrame(rows)
    table.to_csv(common.ANALYSIS_FOLDER / "s06_history_by_place.csv", index = False)
    report("")
    report("== part 2: a short picture history as input (frozen ResNet-18 features, names head, held-out sheets; saved predictions of 9.11.26) ==")
    report("(kept / changed: the demonstration's path over the next 3 steps is within / beyond 45 degrees of its previous step; a model that copies the last move wins on kept and loses on changed)")
    for place in common.PLACE_NAMES + ["all places"]:
        report(f"-- {place} --")
        report(table[table["place"] == place].drop(columns = "place").to_string(index = False))



def main(parameters: Optional[Dict] = None) -> None:

    settings = common.merged_parameters(dict(CHUNK_PARAMETERS, **(parameters or {})))
    report = common.TeeOutput(common.ANALYSIS_FOLDER / "s06_chunk_steps_and_history.out.txt")
    pandas.set_option("display.width", 320)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 400)
    frames = common.load_demonstration_frames(settings)
    path = common.path_direction(frames, settings)
    chunk_part(frames, path, report, settings)
    history_part(frames, path, report, settings)



if __name__ == "__main__":
    main()
