"""
The fair yardstick: the robot's pictures against the demonstrations, in the descriptors of models that never saw the
sheet.

Model E was trained on every demonstration picture, so its descriptors may hold those pictures closer together than
fresh pictures of the same scenes; a robot picture could then look "unfamiliar" only because it is new. The fold models
have the same recipe as model E but each was trained WITHOUT one sheet. For the fold model without sheet k, the
demonstration pictures of sheet k and the robot's pictures of sheet k are both new, so they are compared on equal
terms:
    yardstick   every demonstration picture of sheet k against the pictures of the OTHER EPISODES of sheet k;
    robot       every robot picture of a run on sheet k against all demonstration pictures of sheet k.
The same within-sheet comparison is repeated in model E's descriptors, to show how much the choice of model matters.

A second number per fold: how far the unseen sheet's demonstration pictures are from the fold model's own training
pictures (the other five sheets), against how far those training pictures are from each other. It says how unfamiliar
a whole unseen sheet is to a model, the situation of the exam, as a scale for the robot day's numbers.

Run from this folder, after extract_features.py fold without_path_0k --all-pictures for k = 1 to 6 and after
extract_features.py model_e (about three minutes on two threads):
    D:/AI_Project/.venv/Scripts/python.exe novelty_fold_models.py
Output: novelty_fold_models.out.txt, novelty_fold_models_per_picture.csv, figures/novelty_fold_models_along_runs.png
"""

from typing import Dict, Optional

import numpy
import pandas

from common import (EVIDENCE_FOLDER, FAILURE_STEPS, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, load_descriptors, trained_frames,
                    training_frames, unit_rows)
from neighbours import nearest_rows, percentile_rank
from novelty_model_e import NOVELTY_PARAMETERS, draw_runs, phase_of_steps

FOLD_NOVELTY_PARAMETERS = {
    "descriptor": "grid",
    "neighbour_count": 5,
    # how many of a fold model's own training pictures are placed against the others (a sample keeps the run short)
    "training_sample": 2000,
    "seed": 11_11_26
}



def within_sheet_distances(
        sheet_rows: pandas.DataFrame,
        sheet_descriptors: numpy.ndarray,
        robot_descriptors: numpy.ndarray,
        count: int
) -> Dict[str, numpy.ndarray]:
    """
    Place the demonstration pictures of one sheet against its other episodes, and the robot pictures against all of them.

    Parameters
    ----------
    sheet_rows : pandas.DataFrame
        The sheet's training rows (with "episode_id"), in the order of ``sheet_descriptors``.
    sheet_descriptors : numpy.ndarray
        Their descriptors.
    robot_descriptors : numpy.ndarray
        The descriptors of the robot pictures taken on this sheet.
    count : int
        How many nearest pictures are averaged.

    Returns
    -------
    dict of str to numpy.ndarray
        "demonstrations" (one mean distance per demonstration picture) and "robot" (one per robot picture).
    """

    reference = unit_rows(sheet_descriptors)
    episode_codes = pandas.factorize(sheet_rows["episode_id"])[0]
    demonstrations = nearest_rows(reference, reference, {"other_episodes": (episode_codes, episode_codes)}, {"count": count})["other_episodes"]
    robot = nearest_rows(unit_rows(robot_descriptors), reference, {"all": None}, {"count": count})["all"]
    return {"demonstrations": demonstrations["distance"].mean(axis = 1), "robot": robot["distance"].mean(axis = 1)}



def fold_tables(
        log: OutputLog,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    Compute the within-sheet distances in the fold models' descriptors and in model E's.

    Returns
    -------
    dict
        "per_picture" (one row per robot picture), "baseline" (one row per demonstration picture), "unseen_sheet"
        (one row per fold) and "settings".
    """

    settings = dict(FOLD_NOVELTY_PARAMETERS)
    settings.update(parameters or {})
    count, name = settings["neighbour_count"], settings["descriptor"]
    frames = training_frames()
    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"])
    reference = frames[frames["frame_index"].isin(trained["frame_index"])].reset_index(drop = True)
    model_e_training = load_descriptors("model_e_training")[name]
    model_e_robot = load_descriptors("model_e_robot")
    randomness_generator = numpy.random.default_rng(settings["seed"])

    per_picture, baseline, unseen_sheet = [], [], []
    for sheet in sorted(reference["sheet"].unique()):
        fold_name = f"without_{sheet}"
        fold_training = load_descriptors(f"fold_{fold_name}_all")
        fold_robot = load_descriptors(f"fold_{fold_name}_robot")
        sheet_rows = reference[reference["sheet"] == sheet].reset_index(drop = True)
        other_rows = reference[reference["sheet"] != sheet].reset_index(drop = True)
        runs_here = [run_number for run_number, run in ROBOT_RUNS.items() if run["sheet"] == sheet]
        is_robot_here = numpy.isin(fold_robot["run"], runs_here)

        fold = within_sheet_distances(sheet_rows, fold_training[name][sheet_rows["frame_index"].to_numpy()], fold_robot[name][is_robot_here], count)
        model_e = within_sheet_distances(sheet_rows, model_e_training[sheet_rows["frame_index"].to_numpy()], model_e_robot[name][is_robot_here], count)
        sheet_baseline = sheet_rows[["frame_index", "episode_id", "sheet", "row_in_episode", "start_kind"]].copy()
        sheet_baseline["fold_distance"] = fold["demonstrations"]
        sheet_baseline["model_e_distance"] = model_e["demonstrations"]
        baseline.append(sheet_baseline)

        rows = pandas.DataFrame({"run": fold_robot["run"][is_robot_here], "step": fold_robot["step"][is_robot_here], "sheet": sheet,
                                 "fold_distance": fold["robot"], "model_e_distance": model_e["robot"]})
        rows["fold_percentile"] = percentile_rank(fold["robot"], fold["demonstrations"])
        rows["model_e_percentile"] = percentile_rank(model_e["robot"], model_e["demonstrations"])
        per_picture.append(rows)

        # the unseen sheet against the fold model's own training pictures, and those against each other
        other_unit = unit_rows(fold_training[name][other_rows["frame_index"].to_numpy()])
        unseen = nearest_rows(unit_rows(fold_training[name][sheet_rows["frame_index"].to_numpy()]), other_unit, {"all": None}, {"count": count})["all"]
        sample = numpy.sort(randomness_generator.choice(len(other_rows), size = min(settings["training_sample"], len(other_rows)), replace = False))
        episode_codes = pandas.factorize(other_rows["episode_id"])[0]
        seen = nearest_rows(other_unit[sample], other_unit, {"other_episodes": (episode_codes[sample], episode_codes)}, {"count": count})["other_episodes"]
        robot_to_training = nearest_rows(unit_rows(fold_robot[name][is_robot_here]), other_unit, {"all": None}, {"count": count})["all"]
        unseen_sheet.append({"sheet": sheet, "pictures": len(sheet_rows),
                             "training_pictures_among_themselves_median": float(numpy.median(seen["distance"].mean(axis = 1))),
                             "training_pictures_among_themselves_95": float(numpy.quantile(seen["distance"].mean(axis = 1), 0.95)),
                             "unseen_sheet_to_training_median": float(numpy.median(unseen["distance"].mean(axis = 1))),
                             "unseen_sheet_to_training_95": float(numpy.quantile(unseen["distance"].mean(axis = 1), 0.95)),
                             "robot_pictures_to_training_median": float(numpy.median(robot_to_training["distance"].mean(axis = 1))) if is_robot_here.any() else numpy.nan})
        log.say(f"sheet {sheet}: fold model {fold_name}, {len(sheet_rows)} demonstration pictures, {int(is_robot_here.sum())} robot pictures (runs {runs_here})")

    per_picture = pandas.concat(per_picture).sort_values(["run", "step"]).reset_index(drop = True)
    per_picture["phase"] = numpy.concatenate([phase_of_steps(run_number, int((per_picture["run"] == run_number).sum())) for run_number in sorted(per_picture["run"].unique())])
    return {"per_picture": per_picture, "baseline": pandas.concat(baseline).reset_index(drop = True), "unseen_sheet": pandas.DataFrame(unseen_sheet), "settings": settings}



def say_tables(
        tables: Dict,
        log: OutputLog
) -> None:
    """Say the fair comparison per run and phase, beside the same comparison in model E's descriptors."""

    per_picture, baseline = tables["per_picture"], tables["baseline"]
    log.say("")
    log.say("THE YARDSTICK WITHIN EACH SHEET (mean cosine distance to the 5 nearest pictures of the sheet's other episodes)")
    log.say("sheet     pictures   fold model (never saw the sheet): median   95 %    99 %   |  model E (trained on it): median   95 %    99 %")
    for sheet, group in baseline.groupby("sheet"):
        log.say(f"{sheet:8s}  {len(group):8d}   {group['fold_distance'].median():38.4f}  {group['fold_distance'].quantile(0.95):.4f}  {group['fold_distance'].quantile(0.99):.4f}  |"
                f"  {group['model_e_distance'].median():28.4f}  {group['model_e_distance'].quantile(0.95):.4f}  {group['model_e_distance'].quantile(0.99):.4f}")

    log.say("")
    log.say("EVERY RUN BY PHASE, within its sheet (percentile = place among the sheet's demonstration pictures, each against the sheet's other episodes)")
    log.say("run  model  sheet     phase                         pictures | FOLD MODEL: distance median  percentile median  largest  above 95 %  above 99 % | MODEL E: distance median  percentile median  largest  above 95 %")
    for run_number in ROBOT_RUNS:
        run_rows = per_picture[per_picture["run"] == run_number]
        for phase in pandas.unique(run_rows["phase"]):
            rows = run_rows[run_rows["phase"] == phase]
            log.say(f"{run_number:3d}  {ROBOT_RUNS[run_number]['model']:5s}  {ROBOT_RUNS[run_number]['sheet']:8s}  {phase:28s}  {len(rows):8d} | "
                    f"{rows['fold_distance'].median():27.4f}  {rows['fold_percentile'].median():17.0f}  {rows['fold_percentile'].max():7.0f}  "
                    f"{100 * (rows['fold_percentile'] > 95).mean():8.0f} %  {100 * (rows['fold_percentile'] > 99).mean():8.0f} % | "
                    f"{rows['model_e_distance'].median():24.4f}  {rows['model_e_percentile'].median():17.0f}  {rows['model_e_percentile'].max():7.0f}  "
                    f"{100 * (rows['model_e_percentile'] > 95).mean():8.0f} %")

    log.say("")
    log.say("THE START OF EVERY RUN in the fold model's descriptors: steps 0, 1, 2 against the demonstrations' rows 1 to 3 of the standard kinds of start")
    first_row, last_row = NOVELTY_PARAMETERS["start_rows"]
    is_start = baseline["row_in_episode"].between(first_row, last_row) & baseline["start_kind"].isin(NOVELTY_PARAMETERS["standard_start_kinds"])
    log.say("run  sheet     robot distance, steps 0 1 2      demonstrations' starts: median  95 %   largest   percentile of the robot's three among the starts")
    for run_number in ROBOT_RUNS:
        rows = per_picture[(per_picture["run"] == run_number) & (per_picture["step"] < NOVELTY_PARAMETERS["start_steps"])]
        yardstick = baseline.loc[is_start & (baseline["sheet"] == ROBOT_RUNS[run_number]["sheet"]), "fold_distance"]
        values = rows["fold_distance"].to_numpy()
        log.say(f"{run_number:3d}  {ROBOT_RUNS[run_number]['sheet']:8s}  {'  '.join(f'{value:.4f}' for value in values):30s}  {yardstick.median():28.4f}  {yardstick.quantile(0.95):.4f}  {yardstick.max():.4f}"
                f"   {'  '.join(f'{rank:.0f}' for rank in percentile_rank(values, yardstick.to_numpy()))}")

    log.say("")
    log.say("HOW UNFAMILIAR A WHOLE UNSEEN SHEET IS (each fold model's descriptors; mean cosine distance to the 5 nearest of the fold's own training pictures)")
    log.say("held-out sheet  training pictures among themselves: median  95 %  |  the unseen sheet's demonstrations: median  95 %  ratio of medians  |  robot pictures of that sheet: median")
    for _, row in tables["unseen_sheet"].iterrows():
        log.say(f"{row['sheet']:14s}  {row['training_pictures_among_themselves_median']:42.4f}  {row['training_pictures_among_themselves_95']:.4f}  |"
                f"  {row['unseen_sheet_to_training_median']:38.4f}  {row['unseen_sheet_to_training_95']:.4f}  "
                f"{row['unseen_sheet_to_training_median'] / row['training_pictures_among_themselves_median']:15.1f}  |  {row['robot_pictures_to_training_median']:.4f}")



def main() -> None:

    log = OutputLog(EVIDENCE_FOLDER / "novelty_fold_models.out.txt")
    tables = fold_tables(log)
    say_tables(tables, log)
    tables["per_picture"].round(5).to_csv(EVIDENCE_FOLDER / "novelty_fold_models_per_picture.csv", index = False)

    figure_folder = EVIDENCE_FOLDER / "figures"
    figure_folder.mkdir(exist_ok = True)
    drawn = dict(tables)
    drawn["settings"] = dict(NOVELTY_PARAMETERS)
    draw_runs(drawn, figure_folder / "novelty_fold_models_along_runs.png", column = "fold_distance", yardstick_column = "fold_distance",
              heading = "The fair yardstick: each robot picture against the demonstrations of its sheet, in the descriptors of a model that never saw that sheet\n"
                        "The yardstick is every demonstration picture of the sheet against the sheet's other episodes. Log scale; higher = less like any demonstration picture.")
    log.say("")
    log.say("written: novelty_fold_models_per_picture.csv, figures/novelty_fold_models_along_runs.png")
    log.save()



if __name__ == "__main__":
    main()
