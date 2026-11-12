"""
The verifier's own answer to "were the robot's pictures like the training pictures, before each run went wrong?".

Independent of the analyst's novelty scripts: own descriptors (extract_own.py), own nearest-neighbour code, and own
failure steps. Three things are checked that the analyst's headline rests on:

    1. the analyst's numbers with the analyst's own failure steps (does my code give the same percentiles?);
    2. the same split with the failure steps I read off the pictures and measured (puck_on_line.py): when the first
       move left what the demonstrations did, and when the puck stopped touching the line;
    3. whether the answer depends on the yardstick: 5 nearest or the single nearest picture, cosine or Euclidean
       distance, the whole sheet's pictures or only pictures of the same part of an episode.

Distance of a picture = mean cosine distance of its descriptor (the 6,144 numbers the prediction head receives) to
its 5 nearest demonstration pictures of the same sheet. Yardstick = the same number for every demonstration picture
of the sheet against the sheet's OTHER episodes. "Fair" = descriptors of the fold model that never saw the sheet.
Percentile = share of the yardstick below the robot picture's distance.

Run from this folder, after extract_own.py model_e and extract_own.py fold 1 to 6 (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe own_novelty.py
Output: own_novelty.out.txt, own_novelty_per_picture.csv
"""

import numpy
import pandas
import torch

from verify_common import RUN_FOLDERS, RUN_MODELS, RUN_SHEETS, VERIFY_FOLDER, VERIFY_PARAMETERS, Report, training_frames

OWN_NOVELTY_PARAMETERS = {
    "neighbour_count": 5,
    # the analyst's failure steps (common.FAILURE_STEPS): first wrong decision, puck off the line
    "analyst_steps": {3: (0, 5), 5: (2, 5), 6: (14, 16), 8: (7, 11), 9: (2, 5), 10: (2, 5)},
    # the verifier's: "left the demonstrated move" = the first picture whose answer, as carried out, departs from what
    # the nearest demonstrations did (read off own montages and own_key_frames.out.txt); "puck off the line" = first
    # of three pictures in a row with a gap of 0.5 cm or more (puck_on_line.out.txt), run 8 by eye (dashed line)
    "verifier_steps": {3: (0, 5), 5: (2, 18), 6: (8, 10), 8: (7, 20), 9: (2, 17), 10: (2, 37)},
    "start_rows": [1, 3],
    "start_kinds": ["STD", "OL1", "OL1.5", "OR1", "OR1.5", "TL10", "TL15", "TR10", "TR15", "SQP", "SQL"]
}



def load_table(name: str) -> dict:
    """Read one of the verifier's descriptor tables; the half-precision "grid" comes back as float32."""

    with numpy.load(VERIFY_PARAMETERS["own_cache_folder"] / f"{name}.npz") as saved:
        table = {key: saved[key] for key in saved.files}
    table["grid"] = table["grid"].astype(numpy.float32)
    return table



def unit(rows: numpy.ndarray) -> torch.Tensor:
    """Scale every row to length 1."""

    tensor = torch.from_numpy(numpy.ascontiguousarray(rows, dtype = numpy.float32))
    return tensor / tensor.norm(dim = 1, keepdim = True).clamp_min(1e-12)



def nearest(query: numpy.ndarray, reference: numpy.ndarray, query_groups = None, reference_groups = None, count: int = 5,
            kind: str = "cosine") -> tuple:
    """
    Return the ``count`` smallest distances (and their reference rows) of every query row.

    Parameters
    ----------
    query, reference : numpy.ndarray
        Descriptor rows.
    query_groups, reference_groups : numpy.ndarray, optional
        When given, reference rows of the query row's own group (its episode) are left out.
    count : int, optional
        How many neighbours.
    kind : str, optional
        "cosine" (1 minus the cosine) or "euclidean".

    Returns
    -------
    tuple of numpy.ndarray
        (queries, count) distances in rising order, and the reference rows.
    """

    torch.set_num_threads(VERIFY_PARAMETERS["threads"])
    if kind == "cosine":
        query_tensor, reference_tensor = unit(query), unit(reference)
    else:
        query_tensor = torch.from_numpy(numpy.ascontiguousarray(query, dtype = numpy.float32))
        reference_tensor = torch.from_numpy(numpy.ascontiguousarray(reference, dtype = numpy.float32))
        reference_squares = (reference_tensor ** 2).sum(dim = 1)
    distances, indexes = [], []
    for start in range(0, len(query_tensor), 400):
        part = query_tensor[start:start + 400]
        if kind == "cosine":
            distance = 1.0 - part @ reference_tensor.T
        else:
            distance = ((part ** 2).sum(dim = 1, keepdim = True) + reference_squares[None, :] - 2.0 * part @ reference_tensor.T).clamp_min(0.0).sqrt()
        if query_groups is not None:
            same = torch.from_numpy(numpy.asarray(query_groups[start:start + 400])[:, None] == numpy.asarray(reference_groups)[None, :])
            distance = distance.masked_fill(same, float("inf"))
        best = torch.topk(distance, count, dim = 1, largest = False)
        distances.append(best.values.numpy())
        indexes.append(best.indices.numpy())
    return numpy.concatenate(distances), numpy.concatenate(indexes)



def percentile(values: numpy.ndarray, yardstick: numpy.ndarray) -> numpy.ndarray:
    """Share (0 to 100) of the yardstick strictly below every value."""

    return 100.0 * numpy.searchsorted(numpy.sort(yardstick), values, side = "left") / len(yardstick)



def phase_names(run_number: int, steps: numpy.ndarray, failure_steps: dict) -> numpy.ndarray:
    """Name the phase of every step: before (up to the first wrong move), between, after (puck off the line)."""

    if run_number not in failure_steps:
        return numpy.array(["whole run"] * len(steps), dtype = object)
    wrong, off_line = failure_steps[run_number]
    return numpy.where(steps <= wrong, "before", numpy.where(steps < off_line, "between", "after")).astype(object)



def trained_rows() -> pandas.DataFrame:
    """Return the rows model E trained on (the audit's cleaning as its recipe set it)."""

    from config import TRAINING_TARGET_PARAMETERS
    from train_policy import merged_run_settings
    from training_targets import load_training_frames

    settings = merged_run_settings("resnet18_model_4_chunk_stop_25_all_sheets")
    parameters = dict(TRAINING_TARGET_PARAMETERS)
    parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    return load_training_frames(parameters)



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_novelty.out.txt")
    settings = OWN_NOVELTY_PARAMETERS
    count = settings["neighbour_count"]
    frames = training_frames()
    trained = trained_rows()
    is_trained = frames["frame_index"].isin(trained["frame_index"]).to_numpy()
    report.say(f"frames.csv: {len(frames)} pictures from {frames['episode_id'].nunique()} episodes; model E trained on {int(is_trained.sum())} "
               f"pictures from {frames.loc[is_trained, 'episode_id'].nunique()} episodes")

    model_e_training, model_e_robot = load_table("model_e_training"), load_table("model_e_robot")
    rows = []
    yardsticks = {}
    for sheet in sorted(set(RUN_SHEETS.values())):
        sheet_frames = frames[is_trained & (frames["sheet"] == sheet).to_numpy()].reset_index(drop = True)
        episodes = pandas.factorize(sheet_frames["episode_id"])[0]
        runs_here = [run_number for run_number, run_sheet in RUN_SHEETS.items() if run_sheet == sheet]
        fold_sheet, fold_robot = load_table(f"fold_without_{sheet}_sheet"), load_table(f"fold_without_{sheet}_robot")
        position = numpy.searchsorted(fold_sheet["frame_index"], sheet_frames["frame_index"].to_numpy())
        spaces = {
            "fair": (fold_sheet["grid"][position], fold_robot),
            "model_e": (model_e_training["grid"][sheet_frames["frame_index"].to_numpy()], model_e_robot)
        }
        is_start = (sheet_frames["row_in_episode"].between(*settings["start_rows"]) & sheet_frames["start_kind"].isin(settings["start_kinds"])).to_numpy()
        for space, (demonstrations, robot) in spaces.items():
            own_distance, own_index = nearest(demonstrations, demonstrations, episodes, episodes, count)
            own_mean = own_distance.mean(axis = 1)
            own_euclidean = nearest(demonstrations, demonstrations, episodes, episodes, count, "euclidean")[0].mean(axis = 1)
            yardsticks[(sheet, space)] = {"mean": own_mean, "nearest": own_distance[:, 0], "euclidean": own_euclidean, "is_start": is_start,
                                          "row": sheet_frames["row_in_episode"].to_numpy(), "steps_to_end": sheet_frames["steps_to_end"].to_numpy()}
            is_robot = numpy.isin(robot["run"], runs_here)
            robot_distance, robot_index = nearest(robot["grid"][is_robot], demonstrations, count = count)
            robot_euclidean = nearest(robot["grid"][is_robot], demonstrations, count = count, kind = "euclidean")[0].mean(axis = 1)
            robot_mean = robot_distance.mean(axis = 1)
            # the local ratio: the robot picture's distance over the distances its own neighbours have to their neighbours
            local_ratio = robot_mean / own_mean[robot_index].mean(axis = 1)
            # the place-matched yardstick: only demonstration pictures whose row in the episode is within 3 of the
            # median row of the robot picture's five neighbours
            neighbour_rows = numpy.median(sheet_frames["row_in_episode"].to_numpy()[robot_index], axis = 1)
            place = numpy.array([percentile(numpy.array([value]), own_mean[numpy.abs(sheet_frames["row_in_episode"].to_numpy() - centre) <= 3])[0]
                                 for value, centre in zip(robot_mean, neighbour_rows)])
            for position_in_run, (run_number, step) in enumerate(zip(robot["run"][is_robot], robot["step"][is_robot])):
                rows.append({"run": int(run_number), "step": int(step), "sheet": sheet, "space": space,
                             "distance": float(robot_mean[position_in_run]), "nearest": float(robot_distance[position_in_run, 0]),
                             "percentile": float(percentile(robot_mean[position_in_run:position_in_run + 1], own_mean)[0]),
                             "percentile_nearest": float(percentile(robot_distance[position_in_run:position_in_run + 1, 0], own_distance[:, 0])[0]),
                             "percentile_euclidean": float(percentile(robot_euclidean[position_in_run:position_in_run + 1], own_euclidean)[0]),
                             "percentile_same_place": float(place[position_in_run]),
                             "percentile_among_starts": float(percentile(robot_mean[position_in_run:position_in_run + 1], own_mean[is_start])[0]),
                             "local_ratio": float(local_ratio[position_in_run]), "neighbour_row_median": float(neighbour_rows[position_in_run])})
    table = pandas.DataFrame(rows).sort_values(["space", "run", "step"]).reset_index(drop = True)
    table.round(4).to_csv(VERIFY_FOLDER / "own_novelty_per_picture.csv", index = False)

    report.say("")
    report.say("THE YARDSTICK per sheet (each demonstration picture against the sheet's other episodes, mean of 5 nearest, cosine)")
    report.say("sheet     pictures  fair: median  95 %    99 %   |  model E: median  95 %    99 %")
    for sheet in sorted(set(RUN_SHEETS.values())):
        fair, model_e = yardsticks[(sheet, "fair")]["mean"], yardsticks[(sheet, "model_e")]["mean"]
        report.say(f"{sheet}   {len(fair):8d}  {numpy.median(fair):12.4f}  {numpy.quantile(fair, 0.95):.4f}  {numpy.quantile(fair, 0.99):.4f}  |"
                   f"  {numpy.median(model_e):15.4f}  {numpy.quantile(model_e, 0.95):.4f}  {numpy.quantile(model_e, 0.99):.4f}")

    for title, failure_steps in (("1. WITH THE ANALYST'S FAILURE STEPS", settings["analyst_steps"]), ("2. WITH THE VERIFIER'S FAILURE STEPS", settings["verifier_steps"])):
        report.say("")
        report.say(f"{title} (first wrong move, puck off the line): " + ", ".join(f"run {run}: {steps[0]}/{steps[1]}" for run, steps in failure_steps.items()))
        report.say("run  model  phase       pictures | FAIR: percentile median  largest  above 95 %  local ratio median | MODEL E: percentile median  largest  above 95 %  local ratio median")
        for run_number in RUN_FOLDERS:
            fair = table[(table["space"] == "fair") & (table["run"] == run_number)].reset_index(drop = True)
            model_e = table[(table["space"] == "model_e") & (table["run"] == run_number)].reset_index(drop = True)
            phases = phase_names(run_number, fair["step"].to_numpy(), failure_steps)
            for phase in pandas.unique(phases):
                is_phase = phases == phase
                report.say(f"{run_number:3d}  {RUN_MODELS[run_number]:5s}  {phase:10s}  {int(is_phase.sum()):8d} | "
                           f"{fair.loc[is_phase, 'percentile'].median():23.0f}  {fair.loc[is_phase, 'percentile'].max():7.0f}  {int((fair.loc[is_phase, 'percentile'] > 95).sum()):6d} of {int(is_phase.sum()):<3d}"
                           f"  {fair.loc[is_phase, 'local_ratio'].median():14.2f} | "
                           f"{model_e.loc[is_phase, 'percentile'].median():26.0f}  {model_e.loc[is_phase, 'percentile'].max():7.0f}  {int((model_e.loc[is_phase, 'percentile'] > 95).sum()):6d} of {int(is_phase.sum()):<3d}"
                           f"  {model_e.loc[is_phase, 'local_ratio'].median():14.2f}")

    report.say("")
    report.say("3. DOES THE YARDSTICK MATTER? Pictures above the 95 % point up to and including the first wrong move (verifier's steps), by yardstick")
    report.say("run  pictures | fair: 5 nearest  1 nearest  euclidean  same place | model E: 5 nearest  1 nearest  euclidean  same place")
    for run_number, (wrong, _) in settings["verifier_steps"].items():
        cells = []
        for space in ("fair", "model_e"):
            part = table[(table["space"] == space) & (table["run"] == run_number) & (table["step"] <= wrong)]
            cells.append("  ".join(f"{int((part[column] > 95).sum()):9d}" for column in ("percentile", "percentile_nearest", "percentile_euclidean", "percentile_same_place")))
        report.say(f"{run_number:3d}  {wrong + 1:8d} | {cells[0]} | {cells[1]}")

    report.say("")
    report.say("4. THE FIRST PICTURE ABOVE THE 95 % POINT in each failed run, against when the puck left the line (verifier's steps)")
    report.say("run  first wrong move  puck off the line | first step above 95 %: fair  model E | first of 3 in a row above 95 %: fair  model E | pictures above 95 % while the puck was still on the line: fair  model E")
    for run_number, (wrong, off_line) in settings["verifier_steps"].items():
        cells, firsts, runs_of_three = [], [], []
        for space in ("fair", "model_e"):
            part = table[(table["space"] == space) & (table["run"] == run_number)].reset_index(drop = True)
            above = (part["percentile"] > 95).to_numpy()
            firsts.append(int(numpy.argmax(above)) if above.any() else -1)
            three = [step for step in range(len(above) - 2) if above[step:step + 3].all()]
            runs_of_three.append(three[0] if three else -1)
            cells.append(int(above[:off_line].sum()))
        report.say(f"{run_number:3d}  {wrong:16d}  {off_line:17d} | {firsts[0]:27d}  {firsts[1]:7d} | {runs_of_three[0]:36d}  {runs_of_three[1]:7d} | "
                   f"{cells[0]:61d}  {cells[1]:7d}")

    report.say("")
    report.say("5. THE START OF EVERY RUN: percentile of steps 0, 1, 2 among the demonstrations' own start pictures (rows 1 to 3 of the standard kinds of start)")
    report.say("run  model  sheet    | fair                | model E")
    for run_number in RUN_FOLDERS:
        cells = []
        for space in ("fair", "model_e"):
            part = table[(table["space"] == space) & (table["run"] == run_number) & (table["step"] < 3)]
            cells.append("  ".join(f"{value:4.0f}" for value in part["percentile_among_starts"]))
        report.say(f"{run_number:3d}  {RUN_MODELS[run_number]:5s}  {RUN_SHEETS[run_number]} | {cells[0]:19s} | {cells[1]}")

    report.say("")
    report.say("6. THE WHOLE DAY: are the robot's pictures as close to the demonstrations as those are to each other? (all pictures of a run)")
    report.say("run  model  outcome   pictures | fair: percentile median  local ratio median | model E: percentile median  local ratio median")
    for run_number in RUN_FOLDERS:
        fair = table[(table["space"] == "fair") & (table["run"] == run_number)]
        model_e = table[(table["space"] == "model_e") & (table["run"] == run_number)]
        outcome = "failed" if run_number in settings["verifier_steps"] else "reached"
        report.say(f"{run_number:3d}  {RUN_MODELS[run_number]:5s}  {outcome:8s}  {len(fair):8d} | {fair['percentile'].median():23.0f}  {fair['local_ratio'].median():18.2f} | "
                   f"{model_e['percentile'].median():26.0f}  {model_e['local_ratio'].median():18.2f}")
    is_good_part = numpy.array([(run not in settings["verifier_steps"]) or (step <= settings["verifier_steps"][run][0]) for run, step in zip(table["run"], table["step"])])
    for space in ("fair", "model_e"):
        part = table[(table["space"] == space) & is_good_part]
        report.say(f"{space}: all pictures of the four runs that reached the end plus the pictures up to the first wrong move of the six that failed "
                   f"({len(part)} pictures): percentile median {part['percentile'].median():.0f} (50 would be 'like the demonstrations'), "
                   f"share above the demonstrations' median {100 * (part['percentile'] > 50).mean():.0f} %, local ratio median {part['local_ratio'].median():.2f}")

    report.say("")
    report.say("7. PER STEP, the failed runs' first 24 pictures: fair percentile / model E percentile")
    for run_number in settings["verifier_steps"]:
        fair = table[(table["space"] == "fair") & (table["run"] == run_number)].reset_index(drop = True)
        model_e = table[(table["space"] == "model_e") & (table["run"] == run_number)].reset_index(drop = True)
        report.say(f"run {run_number:2d} fair   : " + " ".join(f"{value:3.0f}" for value in fair["percentile"].to_numpy()[:24]))
        report.say(f"run {run_number:2d} model E: " + " ".join(f"{value:3.0f}" for value in model_e["percentile"].to_numpy()[:24]))
    report.say("")
    report.say("written: own_novelty_per_picture.csv")
    report.save()



if __name__ == "__main__":
    main()
