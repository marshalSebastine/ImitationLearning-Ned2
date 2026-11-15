"""
How unfamiliar was every robot picture of 11.11.26 to model E, measured in model E's own descriptors?

For every robot picture: the cosine distance to its 5 nearest training pictures (the 18,032 pictures model E trained
on), averaged. The yardstick is the same number for every training picture against the pictures of all OTHER
episodes ("leave one episode out"), and against the pictures of all OTHER sheets. A robot picture is then placed in
the yardstick of its own sheet: percentile 50 = as close to the demonstrations as a typical demonstration picture is
to the other demonstrations; above 95 or 99 = unusual.

Two cautions, both checked elsewhere in this folder:
    - model E was trained on the yardstick's pictures, so its descriptors may hold them closer together than fresh
      pictures of the same scenes (novelty_fold_models.py repeats the comparison with models that never saw the sheet);
    - the three successful robot runs (2, 4, 7) are a second yardstick: novelty the model coped with on the day.

Run from this folder, after extract_features.py model_e (about two minutes on two threads):
    D:/AI_Project/.venv/Scripts/python.exe novelty_model_e.py
Output: novelty_model_e.out.txt, novelty_model_e_per_picture.csv, novelty_model_e_baseline.csv,
figures/novelty_model_e_along_runs.png; the neighbours of every robot picture are kept in the cache folder for
key_frames.py.
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import lines
from matplotlib import patches
from matplotlib import pyplot

from common import (EVIDENCE_FOLDER, FAILURE_STEPS, PLOT_STYLE, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, cache_folder,
                    load_descriptors, styled_axes, trained_frames, training_frames, unit_rows)
from neighbours import nearest_rows, percentile_rank

NOVELTY_PARAMETERS = {
    "descriptors": ["grid", "mean", "hidden"],
    "main_descriptor": "grid",
    "neighbour_count": 5,
    "kept_neighbours": 10,
    # the first pictures of a robot run, and the demonstration rows they are compared with (row 0 is not trained on)
    "start_steps": 3,
    "start_rows": [1, 3],
    "standard_start_kinds": ["STD", "OL1", "OL1.5", "OR1", "OR1.5", "TL10", "TL15", "TR10", "TR15", "SQP", "SQL"],
    "percentiles": [50, 95, 99]
}



def phase_of_steps(
        run_number: int,
        step_count: int
) -> numpy.ndarray:
    """
    Name the phase of every picture of a run.

    For a failed run: "seen before it went wrong" (the pictures up to and including the first wrong decision),
    "between" (after it, while the puck still touched the line) and "after the puck left the line". A run that
    reached the end has the single phase "whole run".

    Parameters
    ----------
    run_number : int
        1 to 10.
    step_count : int
        The number of pictures of the run.

    Returns
    -------
    numpy.ndarray
        One phase name per picture.
    """

    if run_number not in FAILURE_STEPS:
        return numpy.array(["whole run"] * step_count, dtype = object)
    wrong, off_line = FAILURE_STEPS[run_number]["first_wrong_decision"], FAILURE_STEPS[run_number]["puck_off_line"]
    steps = numpy.arange(step_count)
    return numpy.where(steps <= wrong, "seen before it went wrong", numpy.where(steps < off_line, "between", "after the puck left the line")).astype(object)



def novelty_tables(
        log: OutputLog,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    Compute the distances of the robot pictures and of the yardstick, for every descriptor.

    Parameters
    ----------
    log : OutputLog
        Where the progress is said.
    parameters : dict, optional
        Overrides of ``NOVELTY_PARAMETERS``.

    Returns
    -------
    dict
        "per_picture" (one row per robot picture), "baseline" (one row per training picture), "reference" (the
        training rows used, in the order of the neighbour indexes) and "neighbours" (per descriptor, the robot
        pictures' nearest reference rows and distances).
    """

    settings = dict(NOVELTY_PARAMETERS)
    settings.update(parameters or {})
    frames = training_frames()
    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"])
    reference = frames[frames["frame_index"].isin(trained["frame_index"])].reset_index(drop = True)
    log.say(f"training pictures in frames.csv: {len(frames)}; model E trained on {len(reference)} of them, from {reference['episode_id'].nunique()} episodes")

    training = load_descriptors("model_e_training")
    robot = load_descriptors("model_e_robot")
    episode_codes = pandas.factorize(reference["episode_id"])[0]
    sheet_codes = pandas.factorize(reference["sheet"])[0]
    count = settings["neighbour_count"]

    per_picture = pandas.DataFrame({"run": robot["run"], "step": robot["step"]})
    per_picture["sheet"] = per_picture["run"].map(lambda run_number: ROBOT_RUNS[run_number]["sheet"])
    per_picture["model"] = per_picture["run"].map(lambda run_number: ROBOT_RUNS[run_number]["model"])
    per_picture["phase"] = numpy.concatenate([phase_of_steps(run_number, int((robot["run"] == run_number).sum())) for run_number in ROBOT_RUNS])
    baseline = reference[["frame_index", "episode_id", "sheet", "row_in_episode", "start_kind", "start_end", "plan_number", "lighting", "steps_to_end"]].copy()
    neighbours = {}

    for name in settings["descriptors"]:
        reference_unit = unit_rows(training[name][reference["frame_index"].to_numpy()])
        robot_unit = unit_rows(robot[name])
        log.say(f"descriptor {name}: {reference_unit.shape[1]} numbers per picture")
        among_demonstrations = nearest_rows(reference_unit, reference_unit,
                                            {"other_episodes": (episode_codes, episode_codes), "other_sheets": (sheet_codes, sheet_codes)},
                                            {"count": count})
        to_robot = nearest_rows(robot_unit, reference_unit, {"all": None}, {"count": settings["kept_neighbours"]})["all"]
        neighbours[name] = to_robot

        for rule, result in among_demonstrations.items():
            baseline[f"{name}_{rule}_nearest"] = result["distance"][:, 0]
            baseline[f"{name}_{rule}_mean_{count}"] = result["distance"].mean(axis = 1)
        own_mean = baseline[f"{name}_other_episodes_mean_{count}"].to_numpy()
        # the local ratio: a picture's distance over the distances its own neighbours have to their neighbours; about
        # 1 inside a well-filled region, far above 1 for a picture that sits alone
        baseline[f"{name}_local_ratio"] = own_mean / own_mean[among_demonstrations["other_episodes"]["index"]].mean(axis = 1)

        robot_mean = to_robot["distance"][:, :count].mean(axis = 1)
        per_picture[f"{name}_nearest"] = to_robot["distance"][:, 0]
        per_picture[f"{name}_mean_{count}"] = robot_mean
        per_picture[f"{name}_local_ratio"] = robot_mean / own_mean[to_robot["index"][:, :count]].mean(axis = 1)
        percentile = numpy.zeros(len(per_picture))
        for sheet in per_picture["sheet"].unique():
            is_sheet = (per_picture["sheet"] == sheet).to_numpy()
            percentile[is_sheet] = percentile_rank(robot_mean[is_sheet], own_mean[(baseline["sheet"] == sheet).to_numpy()])
        per_picture[f"{name}_percentile_in_sheet"] = percentile
        per_picture[f"{name}_percentile_all_sheets"] = percentile_rank(robot_mean, own_mean)
        neighbour_sheets = reference["sheet"].to_numpy()[to_robot["index"][:, :count]]
        per_picture[f"{name}_same_sheet_neighbours"] = (neighbour_sheets == per_picture["sheet"].to_numpy()[:, None]).mean(axis = 1)

    return {"per_picture": per_picture, "baseline": baseline, "reference": reference, "neighbours": neighbours, "settings": settings}



def say_baseline(
        tables: Dict,
        log: OutputLog
) -> None:
    """Say the yardstick: the demonstrations' own distances per sheet, against other episodes and against other sheets."""

    settings, baseline = tables["settings"], tables["baseline"]
    count, main = settings["neighbour_count"], settings["main_descriptor"]
    log.say("")
    log.say(f"THE YARDSTICK (descriptor {main}; mean cosine distance to the {count} nearest pictures)")
    log.say("sheet     pictures   against OTHER EPISODES: median   95 %    99 %    largest |  against OTHER SHEETS: median   95 %    99 %")
    for sheet, group in list(baseline.groupby("sheet")) + [("all", baseline)]:
        episodes = group[f"{main}_other_episodes_mean_{count}"]
        sheets = group[f"{main}_other_sheets_mean_{count}"]
        log.say(f"{sheet:8s}  {len(group):8d}   {episodes.median():27.4f}  {episodes.quantile(0.95):.4f}  {episodes.quantile(0.99):.4f}  {episodes.max():.4f}  |"
                f"  {sheets.median():25.4f}  {sheets.quantile(0.95):.4f}  {sheets.quantile(0.99):.4f}")
    ratio = baseline[f"{main}_local_ratio"]
    log.say("")
    log.say(f"The local ratio of the yardstick itself (a picture's distance over its neighbours' own distances): median {ratio.median():.2f}, "
            f"95 % {ratio.quantile(0.95):.2f}, 99 % {ratio.quantile(0.99):.2f}")
    log.say("")
    log.say("The yardstick at the START of an episode (demonstration rows 1 to 3 of the standard kinds of start, against other episodes)")
    first_row, last_row = settings["start_rows"]
    is_start = baseline["row_in_episode"].between(first_row, last_row) & baseline["start_kind"].isin(settings["standard_start_kinds"])
    log.say("sheet     pictures   median   95 %    largest")
    for sheet, group in baseline[is_start].groupby("sheet"):
        values = group[f"{main}_other_episodes_mean_{count}"]
        log.say(f"{sheet:8s}  {len(group):8d}   {values.median():.4f}  {values.quantile(0.95):.4f}  {values.max():.4f}")



def say_runs(
        tables: Dict,
        log: OutputLog
) -> None:
    """Say, per run and phase, how unfamiliar the robot's pictures were."""

    settings, per_picture, baseline = tables["settings"], tables["per_picture"], tables["baseline"]
    count, main = settings["neighbour_count"], settings["main_descriptor"]
    first_row, last_row = settings["start_rows"]
    is_demonstration_start = baseline["row_in_episode"].between(first_row, last_row) & baseline["start_kind"].isin(settings["standard_start_kinds"])

    log.say("")
    log.say("THE START OF EVERY RUN: the first three robot pictures against the demonstrations' own start pictures of that sheet")
    log.say("run  model  sheet     distance of steps 0, 1, 2            demonstrations' starts: median, largest   robot / median   percentile among the starts")
    for run_number in ROBOT_RUNS:
        rows = per_picture[(per_picture["run"] == run_number) & (per_picture["step"] < settings["start_steps"])]
        yardstick = baseline.loc[is_demonstration_start & (baseline["sheet"] == ROBOT_RUNS[run_number]["sheet"]), f"{main}_other_episodes_mean_{count}"]
        values = rows[f"{main}_mean_{count}"].to_numpy()
        log.say(f"{run_number:3d}  {ROBOT_RUNS[run_number]['model']:5s}  {ROBOT_RUNS[run_number]['sheet']:8s}  "
                f"{'  '.join(f'{value:.4f}' for value in values):35s}  {yardstick.median():22.4f}  {yardstick.max():.4f}"
                f"  {numpy.median(values) / yardstick.median():15.1f}   {'  '.join(f'{rank:.0f}' for rank in percentile_rank(values, yardstick.to_numpy()))}")

    log.say("")
    log.say("EVERY RUN BY PHASE (percentile = place of the robot picture among the demonstration pictures of the same sheet, each against other episodes)")
    log.say("run  model  sheet     phase                         pictures  distance median  percentile median  largest  share above 95 %  above 99 %  local ratio median  same-sheet neighbours")
    for run_number in ROBOT_RUNS:
        run_rows = per_picture[per_picture["run"] == run_number]
        for phase in pandas.unique(run_rows["phase"]):
            rows = run_rows[run_rows["phase"] == phase]
            ranks = rows[f"{main}_percentile_in_sheet"]
            log.say(f"{run_number:3d}  {ROBOT_RUNS[run_number]['model']:5s}  {ROBOT_RUNS[run_number]['sheet']:8s}  {phase:28s}  {len(rows):8d}  "
                    f"{rows[f'{main}_mean_{count}'].median():15.4f}  {ranks.median():17.0f}  {ranks.max():7.0f}  {100 * (ranks > 95).mean():15.0f}  "
                    f"{100 * (ranks > 99).mean():10.0f}  {rows[f'{main}_local_ratio'].median():18.2f}  {100 * rows[f'{main}_same_sheet_neighbours'].mean():20.0f} %")

    log.say("")
    log.say("THE SAME QUESTION WITH THE OTHER TWO DESCRIPTORS (median percentile in the sheet's yardstick; share of pictures above 95 %)")
    log.say("run  phase                         " + "  ".join(f"{name:>22s}" for name in settings["descriptors"]))
    for run_number in ROBOT_RUNS:
        run_rows = per_picture[per_picture["run"] == run_number]
        for phase in pandas.unique(run_rows["phase"]):
            rows = run_rows[run_rows["phase"] == phase]
            cells = [f"{rows[f'{name}_percentile_in_sheet'].median():10.0f} / {100 * (rows[f'{name}_percentile_in_sheet'] > 95).mean():5.0f} %  " for name in settings["descriptors"]]
            log.say(f"{run_number:3d}  {phase:28s}  " + "  ".join(cells))

    log.say("")
    log.say("THE SUCCESSFUL RUNS AS A SECOND YARDSTICK: the novelty model E coped with on the day (runs 2, 4, 7), against what it saw before each failure")
    is_success = per_picture["run"].isin([2, 4, 7])
    coped = per_picture.loc[is_success, f"{main}_mean_{count}"]
    log.say(f"runs 2, 4, 7, all {int(is_success.sum())} pictures: distance median {coped.median():.4f}, 95 % {coped.quantile(0.95):.4f}, largest {coped.max():.4f}; "
            f"local ratio median {per_picture.loc[is_success, f'{main}_local_ratio'].median():.2f}, 95 % {per_picture.loc[is_success, f'{main}_local_ratio'].quantile(0.95):.2f}")
    for run_number in FAILURE_STEPS:
        rows = per_picture[(per_picture["run"] == run_number) & (per_picture["phase"] == "seen before it went wrong")]
        values = rows[f"{main}_mean_{count}"]
        log.say(f"run {run_number:2d}, the {len(rows)} pictures seen before it went wrong: distance largest {values.max():.4f} "
                f"(= percentile {percentile_rank(numpy.array([values.max()]), coped.to_numpy())[0]:.0f} of the successful runs' pictures); "
                f"local ratio largest {rows[f'{main}_local_ratio'].max():.2f}")



def draw_runs(
        tables: Dict,
        figure_path,
        column: Optional[str] = None,
        yardstick_column: Optional[str] = None,
        heading: Optional[str] = None
) -> None:
    """
    Draw the distance along every run, with the sheet's yardstick and the steps where the run went wrong.

    Parameters
    ----------
    tables : dict
        "per_picture" (one row per robot picture, with "run", "step" and the distance column) and "baseline" (one
        row per demonstration picture, with "sheet" and the yardstick column).
    figure_path : pathlib.Path
        Where the figure is saved.
    column, yardstick_column : str, optional
        The robot pictures' distance and the demonstrations' own distance; model E's main descriptor by default.
    heading : str, optional
        The figure's title; the title for model E by default.
    """

    settings, per_picture, baseline = tables["settings"], tables["per_picture"], tables["baseline"]
    count, main = settings["neighbour_count"], settings["main_descriptor"]
    column = column or f"{main}_mean_{count}"
    yardstick_column = yardstick_column or f"{main}_other_episodes_mean_{count}"
    figure, grid = pyplot.subplots(5, 2, figsize = (13, 15.5), facecolor = PLOT_STYLE["surface"])
    lowest = min(per_picture[column].min(), baseline[yardstick_column].quantile(0.25)) * 0.8
    highest = per_picture[column].max() * 1.25
    ticks = [value for value in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8) if lowest <= value <= highest]

    for axes, run_number in zip(grid.flatten(), ROBOT_RUNS):
        styled_axes(axes)
        rows = per_picture[per_picture["run"] == run_number]
        yardstick = baseline.loc[baseline["sheet"] == ROBOT_RUNS[run_number]["sheet"], yardstick_column]
        last_step = rows["step"].max()
        for share, label in ((0.5, "median"), (0.95, "95 %"), (0.99, "99 %")):
            level = yardstick.quantile(share)
            axes.hlines(level, -1, last_step + 0.5, color = PLOT_STYLE["neutral"], linewidth = 1.0)
            axes.text(last_step + 1.0, level, label, fontsize = 7, color = PLOT_STYLE["ink_secondary"], va = "center", ha = "left")
        if run_number in FAILURE_STEPS:
            axes.axvspan(FAILURE_STEPS[run_number]["puck_off_line"] - 0.5, last_step + 0.5, color = PLOT_STYLE["went_wrong"], alpha = 0.08, linewidth = 0)
            axes.axvline(FAILURE_STEPS[run_number]["first_wrong_decision"], color = PLOT_STYLE["went_wrong"], linewidth = 1.4)
        axes.plot(rows["step"], rows[column], color = PLOT_STYLE["series"][0], linewidth = 1.8, marker = "o", markersize = 4.5,
                  markeredgecolor = PLOT_STYLE["surface"], markeredgewidth = 0.8)
        axes.set_yscale("log")
        axes.set_ylim(lowest, highest)
        axes.set_yticks(ticks)
        axes.set_yticklabels([f"{value:g}" for value in ticks])
        axes.minorticks_off()
        axes.set_xlim(-1, last_step * 1.12 + 3)
        outcome = "went wrong" if run_number in FAILURE_STEPS else "reached the end"
        axes.set_title(f"Run {run_number}: sheet {ROBOT_RUNS[run_number]['sheet'][-2:]}, model {ROBOT_RUNS[run_number]['model']}, {outcome} ({ROBOT_RUNS[run_number]['note']})",
                       fontsize = 9, color = PLOT_STYLE["ink"], loc = "left")
        axes.set_xlabel("picture (step) of the run", fontsize = 8, color = PLOT_STYLE["ink_secondary"])
        axes.set_ylabel("distance to the 5 nearest\ndemonstration pictures", fontsize = 8, color = PLOT_STYLE["ink_secondary"])

    keys = [lines.Line2D([], [], color = PLOT_STYLE["series"][0], linewidth = 1.8, marker = "o", markersize = 4.5, label = "robot picture"),
            lines.Line2D([], [], color = PLOT_STYLE["neutral"], linewidth = 1.0, label = "the demonstrations' own distances on that sheet (median, 95 %, 99 %)"),
            lines.Line2D([], [], color = PLOT_STYLE["went_wrong"], linewidth = 1.4, label = "first wrong decision"),
            patches.Patch(color = PLOT_STYLE["went_wrong"], alpha = 0.12, label = "after the puck left the line")]
    figure.legend(handles = keys, loc = "upper left", bbox_to_anchor = (0.008, 0.962), ncol = 4, fontsize = 8.5, frameon = False, labelcolor = PLOT_STYLE["ink"])
    figure.suptitle(heading or "How unfamiliar each robot picture was to model E: cosine distance in model E's own descriptors (what its prediction head sees)\n"
                    "The yardstick is every training picture against the pictures of all other episodes. Log scale; higher = less like any demonstration picture.",
                    fontsize = 10.5, color = PLOT_STYLE["ink"], x = 0.01, ha = "left")
    figure.tight_layout(rect = (0, 0, 1, 0.945))
    figure.savefig(figure_path, dpi = 110, facecolor = PLOT_STYLE["surface"])
    pyplot.close(figure)



def main() -> None:

    log = OutputLog(EVIDENCE_FOLDER / "novelty_model_e.out.txt")
    tables = novelty_tables(log)
    say_baseline(tables, log)
    say_runs(tables, log)

    tables["per_picture"].round(5).to_csv(EVIDENCE_FOLDER / "novelty_model_e_per_picture.csv", index = False)
    count, main_descriptor = tables["settings"]["neighbour_count"], tables["settings"]["main_descriptor"]
    summary = tables["baseline"].groupby("sheet")[[f"{main_descriptor}_other_episodes_mean_{count}", f"{main_descriptor}_other_sheets_mean_{count}"]].describe(
        percentiles = [0.5, 0.95, 0.99]).round(5)
    summary.to_csv(EVIDENCE_FOLDER / "novelty_model_e_baseline.csv")
    numpy.savez(cache_folder() / "model_e_robot_neighbours.npz",
                reference_frame_index = tables["reference"]["frame_index"].to_numpy(),
                baseline_mean = tables["baseline"][f"{main_descriptor}_other_episodes_mean_{count}"].to_numpy(),
                **{f"{name}_{key}": value for name, result in tables["neighbours"].items() for key, value in result.items()})
    figure_folder = EVIDENCE_FOLDER / "figures"
    figure_folder.mkdir(exist_ok = True)
    draw_runs(tables, figure_folder / "novelty_model_e_along_runs.png")
    log.say("")
    log.say("written: novelty_model_e_per_picture.csv, novelty_model_e_baseline.csv, figures/novelty_model_e_along_runs.png")
    log.save()



if __name__ == "__main__":
    main()
