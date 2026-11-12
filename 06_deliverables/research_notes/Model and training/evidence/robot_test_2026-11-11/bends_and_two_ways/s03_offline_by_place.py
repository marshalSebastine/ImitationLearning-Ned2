"""
What the saved models answer at straight stretches, bends and hairpins, on pictures of sheets they were not trained on.

Three answers to the same held-out pictures are compared, per kind of place (common.py):
    exact step    the tip move the robot carried out on 11.11.26 (its direction; the robot ignored its length);
    chunk         the tip's place 1, 3 and 5 steps ahead from the same model (recipe E), never used on the robot so far;
    move names    the backup model's 8 names: the most likely name, the probability-weighted mean direction, and how the
                  probability splits over opposite names.
Measures: the predicted step length (median, share under 0.3 cm); the direction against where the demonstration's tip
went over the next 3 steps (the report's path direction, scored where that is at least 1.5 cm); how much the direction
changes from one picture to the next (steadiness, on the table so that the forearm twist does not count); the names'
probability more than 90 degrees away from the most likely name; and whether the predicted twist and sideways move go
into the bend as the demonstrations' do. A last table says what the demonstration did where the model's step is short.

Needs the cache of s02_model_outputs.py (jobs recipe_e, recipe_a, backup, all_sheets_e; with recipe_e_seed_2 the second
seed is shown beside the first).
Run from this folder:  python s03_offline_by_place.py
Output: s03_offline_by_place.out.txt and s03_*.csv.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas

import common
from training_targets import DIRECTION_ANGLES_DEG

OFFLINE_PARAMETERS = {
    "length_bins_cm": [0.0, 0.3, 0.6, 0.9, 5.0],
    "long_step_cm": 0.7,
    "bend_side_from_deg": 60.0,
    "twist_counts_from_deg": 2.0,
    "opposite_from_deg": 90.0,
    "split_mass": 0.25
}



def direction_of(vector: numpy.ndarray) -> numpy.ndarray:
    """Return the direction of (forward, left) vectors in the picture: 0 = towards its top, positive = to its left."""

    return numpy.degrees(numpy.arctan2(vector[..., 1], vector[..., 0]))



def load_outputs(
        frames: pandas.DataFrame,
        parameters: Dict
) -> Dict[str, Dict[str, numpy.ndarray]]:
    """Return the cached raw outputs of every job of s02_model_outputs.py, checked to be in the rows' order."""

    outputs = {}
    for job_name, file_name in (("recipe_e", "fold_outputs_recipe_e.npz"), ("recipe_a", "fold_outputs_recipe_a.npz"),
                                ("backup", "fold_outputs_backup.npz"), ("all_sheets_e", "all_sheets_e_on_demonstrations.npz")):
        path = Path(parameters["cache_folder"]) / file_name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run s02_model_outputs.py first")
        saved = numpy.load(path)
        if not numpy.array_equal(saved["frame_index"], frames["frame_index"].to_numpy()):
            raise ValueError(f"{file_name} holds other pictures than the cleaned table")
        outputs[job_name] = {name: saved[name] for name in saved.files if name not in ("frame_index", "hidden")}
    return outputs



def names_statistics(
        direction_logits: numpy.ndarray,
        class_names: list,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Return how the backup's probability is spread over the 8 move names of every picture.

    Parameters
    ----------
    direction_logits : numpy.ndarray
        (pictures, classes), the raw outputs of the names head.
    class_names : list of str
        The classes in output order (the stop class is left out; this run has its own stop output).
    parameters : dict
        Supplies "opposite_from_deg".

    Returns
    -------
    pandas.DataFrame
        "top_probability", "top_deg" (the most likely name's direction), "expected_deg" (the probability-weighted mean
        direction), "resultant" (length of the probability-weighted mean of the unit vectors: 1 when one name has
        everything, near 0 when opposite names share it), "opposite_mass" (probability on names more than
        "opposite_from_deg" away from the most likely one) and "entropy_bits".
    """

    moving_columns = [index for index, name in enumerate(class_names) if name in DIRECTION_ANGLES_DEG]
    nominal = numpy.array([DIRECTION_ANGLES_DEG[class_names[index]] for index in moving_columns], dtype = float)
    logits = direction_logits[:, moving_columns].astype(numpy.float64)
    logits = logits - logits.max(axis = 1, keepdims = True)
    probabilities = numpy.exp(logits)
    probabilities = probabilities / probabilities.sum(axis = 1, keepdims = True)
    top = numpy.argmax(probabilities, axis = 1)
    radians = numpy.radians(nominal)
    mean_forward = probabilities @ numpy.cos(radians)
    mean_left = probabilities @ numpy.sin(radians)
    apart = common.angle_difference_deg(nominal[None, :], nominal[top][:, None])
    return pandas.DataFrame({
        "top_probability": probabilities[numpy.arange(len(top)), top],
        "top_deg": nominal[top],
        "expected_deg": numpy.degrees(numpy.arctan2(mean_left, mean_forward)),
        "resultant": numpy.hypot(mean_forward, mean_left),
        "opposite_mass": (probabilities * (apart > parameters["opposite_from_deg"])).sum(axis = 1),
        "entropy_bits": -(probabilities * numpy.log2(numpy.clip(probabilities, 1e-12, None))).sum(axis = 1)
    })



def direction_scores(
        predicted_deg: numpy.ndarray,
        true_deg: numpy.ndarray
) -> Dict:
    """Return the median error, the share within 45 degrees and the share beyond 90 degrees of predicted directions."""

    error = common.angle_difference_deg(predicted_deg, true_deg)
    return {"median_error_deg": round(float(numpy.median(error)), 1), "within_45": round(float((error <= 45).mean()), 3),
            "beyond_90": round(float((error > 90).mean()), 3)}



def main(parameters: Optional[Dict] = None) -> None:

    settings = common.merged_parameters(dict(OFFLINE_PARAMETERS, **(parameters or {})))
    report = common.TeeOutput(common.ANALYSIS_FOLDER / "s03_offline_by_place.out.txt")
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 400)
    frames = common.load_demonstration_frames(settings)
    path = common.path_direction(frames, settings)
    outputs = load_outputs(frames, settings)
    class_names = ["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right", "stop"]
    names = names_statistics(outputs["backup"]["direction"], class_names, settings)

    heading = frames["picture_heading_deg"].to_numpy()
    answers = {
        "exact step (recipe E, held-out sheet)": outputs["recipe_e"]["exact"][:, :2],
        "chunk, 1 step ahead (recipe E)": outputs["recipe_e"]["chunk"].reshape(len(frames), 5, 2)[:, 0],
        "chunk, 3 steps ahead (recipe E)": outputs["recipe_e"]["chunk"].reshape(len(frames), 5, 2)[:, 2],
        "chunk, 5 steps ahead (recipe E)": outputs["recipe_e"]["chunk"].reshape(len(frames), 5, 2)[:, 4],
        "exact step (recipe A, held-out sheet)": outputs["recipe_a"]["exact"][:, :2],
        "exact step (backup model)": outputs["backup"]["exact"][:, :2],
        "exact step (model E itself, pictures it was trained on)": outputs["all_sheets_e"]["exact"][:, :2],
        "chunk, 3 steps ahead (model E itself, trained on)": outputs["all_sheets_e"]["chunk"].reshape(len(frames), 5, 2)[:, 2]
    }
    directions = {name: direction_of(vector) for name, vector in answers.items()}
    directions["names, most likely (backup)"] = names["top_deg"].to_numpy()
    directions["names, mean direction (backup)"] = names["expected_deg"].to_numpy()
    directions["always forward"] = numpy.zeros(len(frames))
    directions["the demonstrated step itself"] = frames["step_picture_deg"].to_numpy()

    is_scored = (~frames["is_stop"]).to_numpy() & (frames["place"] != "no fit").to_numpy()
    places = frames["place"].to_numpy()
    report(f"held-out pictures scored: {int(is_scored.sum())} (not the stop row, with a line position); path rows among them: {int((is_scored & path['is_path_row'].to_numpy()).sum())}")
    report("every number is measured on demonstration pictures; recipe E, recipe A and the backup were trained WITHOUT the picture's sheet")

    # ---- step length ----
    report("")
    report("== 1. step length per kind of place (cm) ==")
    rows = []
    for place in common.PLACE_NAMES:
        mask = is_scored & (places == place)
        row = {"place": place, "pictures": int(mask.sum()),
               "demonstrated median": round(float(numpy.median(frames["step_cm"].to_numpy()[mask])), 2),
               "demonstrated under 0.3": round(float((frames["step_cm"].to_numpy()[mask] < 0.3).mean()), 3)}
        for name in ("exact step (recipe E, held-out sheet)", "exact step (recipe A, held-out sheet)", "exact step (backup model)",
                     "exact step (model E itself, pictures it was trained on)", "chunk, 1 step ahead (recipe E)"):
            length = numpy.linalg.norm(answers[name][mask], axis = 1)
            short_name = name.replace("exact step ", "").replace(", held-out sheet", "").strip("()")
            row[f"{short_name}: median"] = round(float(numpy.median(length)), 2)
            row[f"{short_name}: under 0.3"] = round(float((length < 0.3).mean()), 3)
        rows.append(row)
    length_table = pandas.DataFrame(rows)
    length_table.to_csv(common.ANALYSIS_FOLDER / "s03_length_by_place.csv", index = False)
    report(length_table.to_string(index = False))

    report("")
    report("== 1b. how far ahead: the chunk's place 3 and 5 steps ahead against the demonstrated place (median distance from the tip, cm) ==")
    for place in common.PLACE_NAMES:
        mask = is_scored & (places == place)
        demonstrated_3 = numpy.hypot(path["ahead_3_forward_cm"].to_numpy()[mask], path["ahead_3_left_cm"].to_numpy()[mask])
        demonstrated_5 = numpy.hypot(path["ahead_5_forward_cm"].to_numpy()[mask], path["ahead_5_left_cm"].to_numpy()[mask])
        predicted_3 = numpy.linalg.norm(answers["chunk, 3 steps ahead (recipe E)"][mask], axis = 1)
        predicted_5 = numpy.linalg.norm(answers["chunk, 5 steps ahead (recipe E)"][mask], axis = 1)
        report(f"{place:12s} | 3 ahead: demonstrated {numpy.median(demonstrated_3):.2f}, chunk {numpy.median(predicted_3):.2f}, chunk under 0.5 cm: {float((predicted_3 < 0.5).mean()):.3f} | "
               f"5 ahead: demonstrated {numpy.median(demonstrated_5):.2f}, chunk {numpy.median(predicted_5):.2f}, chunk under 0.5 cm: {float((predicted_5 < 0.5).mean()):.3f}")

    # ---- direction against the path ----
    report("")
    report("== 2. direction against where the demonstration went over the next 3 steps (path rows only) ==")
    rows = []
    path_deg = path["path_deg"].to_numpy()
    is_path_row = path["is_path_row"].to_numpy()
    for place in common.PLACE_NAMES + ["all places"]:
        mask = is_scored & is_path_row & ((places == place) | (place == "all places"))
        for name, predicted in directions.items():
            rows.append({"place": place, "answer": name, "path rows": int(mask.sum()), **direction_scores(predicted[mask], path_deg[mask])})
    direction_table = pandas.DataFrame(rows)
    direction_table.to_csv(common.ANALYSIS_FOLDER / "s03_direction_by_place.csv", index = False)
    for measure in ("within_45", "median_error_deg", "beyond_90"):
        report(f"-- {measure} --")
        report(direction_table.pivot(index = "answer", columns = "place", values = measure)[common.PLACE_NAMES + ["all places"]].loc[list(directions)].to_string())

    # ---- steadiness ----
    report("")
    report("== 3. change of direction from one picture to the next, on the table (pairs of consecutive pictures of one episode) ==")
    is_same_episode = (frames["episode_id"] == frames["episode_id"].shift(-1)).to_numpy()
    is_next_row = (frames["row_in_episode"].shift(-1) - frames["row_in_episode"]).to_numpy() == 1
    has_next = is_same_episode & is_next_row & is_scored & numpy.r_[is_scored[1:], False]
    rows = []
    steady = dict(directions)
    steady["the demonstrated path direction (3 steps ahead)"] = path_deg
    steady.pop("always forward")
    for place in common.PLACE_NAMES + ["all places"]:
        mask = has_next & ((places == place) | (place == "all places"))
        positions = numpy.flatnonzero(mask)
        for name, predicted in steady.items():
            on_table = predicted + heading
            change = common.angle_difference_deg(on_table[positions], on_table[positions + 1])
            rows.append({"place": place, "answer": name, "pairs": int(len(positions)), "median_change_deg": round(float(numpy.median(change)), 1),
                         "over_90": round(float((change > 90).mean()), 3), "over_120": round(float((change > 120).mean()), 3)})
    steady_table = pandas.DataFrame(rows)
    steady_table.to_csv(common.ANALYSIS_FOLDER / "s03_steadiness_by_place.csv", index = False)
    for measure in ("over_90", "median_change_deg"):
        report(f"-- {measure} --")
        report(steady_table.pivot(index = "answer", columns = "place", values = measure)[common.PLACE_NAMES + ["all places"]].loc[list(steady)].to_string())

    # ---- names ----
    report("")
    report("== 4. the backup's move names: is the probability split over opposite names? ==")
    exact_e_length = numpy.linalg.norm(answers["exact step (recipe E, held-out sheet)"], axis = 1)
    exact_backup_length = numpy.linalg.norm(answers["exact step (backup model)"], axis = 1)
    rows = []
    for place in common.PLACE_NAMES + ["all places"]:
        mask = is_scored & ((places == place) | (place == "all places"))
        part = names[mask]
        rows.append({"place": place, "pictures": int(mask.sum()),
                     "top probability, median": round(float(part["top_probability"].median()), 3),
                     "probability over 90 deg from the top name, mean": round(float(part["opposite_mass"].mean()), 3),
                     f"share of pictures with {settings['split_mass']} or more there": round(float((part["opposite_mass"] >= settings["split_mass"]).mean()), 3),
                     "resultant length, median": round(float(part["resultant"].median()), 3),
                     "share with resultant under 0.5": round(float((part["resultant"] < 0.5).mean()), 3),
                     "entropy, median bits": round(float(part["entropy_bits"].median()), 2)})
    names_table = pandas.DataFrame(rows)
    names_table.to_csv(common.ANALYSIS_FOLDER / "s03_names_by_place.csv", index = False)
    report(names_table.to_string(index = False))
    report("")
    report("-- the names where the exact step of the SAME (backup) model is short or long --")
    for label, mask in (("exact under 0.3 cm", exact_backup_length < 0.3), ("exact 0.3 to 0.7 cm", (exact_backup_length >= 0.3) & (exact_backup_length < 0.7)),
                        ("exact 0.7 cm or more", exact_backup_length >= 0.7)):
        part = names[mask & is_scored]
        report(f"{label:22s} | pictures {len(part):5d} | top probability median {part['top_probability'].median():.3f} | probability over 90 deg from the top name: mean "
               f"{part['opposite_mass'].mean():.3f}, share of pictures with {settings['split_mass']} or more {float((part['opposite_mass'] >= settings['split_mass']).mean()):.3f} | "
               f"resultant median {part['resultant'].median():.3f}")

    # ---- short predicted steps: what did the demonstration do? ----
    report("")
    report("== 5. where recipe E's exact step is short: what the demonstration did at the same picture ==")
    demonstrated = frames["step_cm"].to_numpy()
    twist = frames["twist_deg"].to_numpy()
    bins = settings["length_bins_cm"]
    rows = []
    for place in common.PLACE_NAMES + ["all places"]:
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = is_scored & ((places == place) | (place == "all places")) & (exact_e_length >= lower) & (exact_e_length < upper)
            if mask.sum() == 0:
                continue
            is_real = demonstrated[mask] >= settings["long_step_cm"]
            error_step = common.angle_difference_deg(directions["exact step (recipe E, held-out sheet)"][mask], frames["step_picture_deg"].to_numpy()[mask])
            path_mask = mask & is_path_row
            error_path = common.angle_difference_deg(directions["exact step (recipe E, held-out sheet)"][path_mask], path_deg[path_mask])
            rows.append({"place": place, "predicted length": f"{lower} to {upper}", "pictures": int(mask.sum()),
                         "share of the place": round(float(mask.sum() / (is_scored & ((places == place) | (place == "all places"))).sum()), 3),
                         "demonstrated: twist in place": round(float(((demonstrated[mask] < 0.3) & (numpy.abs(twist[mask]) >= 3.0)).mean()), 3),
                         "demonstrated: under 0.3 cm": round(float((demonstrated[mask] < 0.3).mean()), 3),
                         "demonstrated: 0.7 cm or more": round(float(is_real.mean()), 3),
                         "demonstrated median cm": round(float(numpy.median(demonstrated[mask])), 2),
                         "direction beyond 90 of the demonstrated step (real moves)": round(float((error_step[is_real] > 90).mean()), 3) if is_real.any() else numpy.nan,
                         "path within 45": round(float((error_path <= 45).mean()), 3) if path_mask.any() else numpy.nan,
                         "path beyond 90": round(float((error_path > 90).mean()), 3) if path_mask.any() else numpy.nan})
    short_table = pandas.DataFrame(rows)
    short_table.to_csv(common.ANALYSIS_FOLDER / "s03_short_steps.csv", index = False)
    report(short_table.to_string(index = False))

    # ---- twist and bend side ----
    report("")
    report("== 6. twist and sideways move against the side the line bends to (pictures where the line turns 60 degrees or more within 6 cm) ==")
    bend = frames["bend_turn_deg"].to_numpy()
    side = numpy.sign(bend)
    is_bend = is_scored & (numpy.abs(bend) >= settings["bend_side_from_deg"])
    rows = []
    sources = {"demonstration": (frames["twist_deg"].to_numpy(), frames["step_left_cm"].to_numpy()),
               "recipe E (held-out sheet)": (outputs["recipe_e"]["exact"][:, 2], outputs["recipe_e"]["exact"][:, 1]),
               "recipe A (held-out sheet)": (outputs["recipe_a"]["exact"][:, 2], outputs["recipe_a"]["exact"][:, 1]),
               "model E itself (trained on)": (outputs["all_sheets_e"]["exact"][:, 2], outputs["all_sheets_e"]["exact"][:, 1])}
    for place in ("sharp bend", "hairpin"):
        mask = is_bend & (places == place)
        for name, (twist_values, left_values) in sources.items():
            # a positive forearm change turns the camera to the right, so into a right bend (bend_turn negative)
            into_bend = -twist_values[mask] * side[mask]
            outward = -left_values[mask] * side[mask]
            rows.append({"place": place, "source": name, "pictures": int(mask.sum()),
                         "twist into the bend, mean deg": round(float(into_bend.mean()), 2),
                         "share twisting into the bend (2 deg or more)": round(float((into_bend >= settings["twist_counts_from_deg"]).mean()), 3),
                         "share twisting out of it (2 deg or more)": round(float((into_bend <= -settings["twist_counts_from_deg"]).mean()), 3),
                         "sideways move to the outside, mean cm": round(float(outward.mean()), 3),
                         "share moving to the inside by 0.2 cm or more": round(float((outward <= -0.2).mean()), 3)})
    side_table = pandas.DataFrame(rows)
    side_table.to_csv(common.ANALYSIS_FOLDER / "s03_bend_side.csv", index = False)
    report(side_table.to_string(index = False))

    # ---- the second seed of recipe E, to see which differences are larger than the change between two trainings ----
    second_path = Path(settings["cache_folder"]) / "fold_outputs_recipe_e_seed_2.npz"
    if second_path.exists():
        second = numpy.load(second_path)
        second_chunk = second["chunk"].reshape(len(frames), 5, 2)
        seeds = {"seed 1": (outputs["recipe_e"]["exact"][:, :2], outputs["recipe_e"]["chunk"].reshape(len(frames), 5, 2)),
                 "seed 2": (second["exact"][:, :2], second_chunk)}
        report("")
        report("== 6b. recipe E trained twice (two seeds), held-out sheets: step length and path within 45 for the exact step and the chunk ==")
        rows = []
        for place in common.PLACE_NAMES + ["all places"]:
            mask = is_scored & ((places == place) | (place == "all places"))
            path_mask = mask & is_path_row
            row = {"place": place}
            for seed_name, (exact_step, chunk_places) in seeds.items():
                row[f"{seed_name}: length median"] = round(float(numpy.median(numpy.linalg.norm(exact_step[mask], axis = 1))), 2)
                row[f"{seed_name}: under 0.3"] = round(float((numpy.linalg.norm(exact_step[mask], axis = 1) < 0.3).mean()), 3)
                row[f"{seed_name}: exact"] = direction_scores(direction_of(exact_step[path_mask]), path_deg[path_mask])["within_45"]
                row[f"{seed_name}: chunk-3"] = direction_scores(direction_of(chunk_places[path_mask, 2]), path_deg[path_mask])["within_45"]
                row[f"{seed_name}: chunk-5"] = direction_scores(direction_of(chunk_places[path_mask, 4]), path_deg[path_mask])["within_45"]
            rows.append(row)
        report(pandas.DataFrame(rows).to_string(index = False))

    report("")
    report("== 7. twist size per kind of place (median of the absolute forearm change per step, degrees) ==")
    for place in common.PLACE_NAMES:
        mask = is_scored & (places == place)
        report(f"{place:12s} | demonstrated {numpy.median(numpy.abs(twist[mask])):.2f} | recipe E {numpy.median(numpy.abs(outputs['recipe_e']['exact'][mask, 2])):.2f} | "
               f"recipe A {numpy.median(numpy.abs(outputs['recipe_a']['exact'][mask, 2])):.2f} | model E itself {numpy.median(numpy.abs(outputs['all_sheets_e']['exact'][mask, 2])):.2f} | "
               f"sum predicted / sum demonstrated (recipe E): {numpy.abs(outputs['recipe_e']['exact'][mask, 2]).sum() / numpy.abs(twist[mask]).sum():.2f}")



if __name__ == "__main__":
    main()
