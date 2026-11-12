"""
What the model asked for against the 1 cm the driver did, step by step, for the ten robot runs of 11.11.26, and where
the puck first left the line in each run.

The asked length is hypot(model_tip_forward_cm, model_tip_left_cm) of steps.csv; the length done is the move of the arm
model's tip to the next picture. Where the puck left the line is read from the run's own pictures (line_and_puck.py:
the line map and the puck track on the table), so it does not depend on where the sheet lay.

Run from this folder after line_and_puck.py:
    python 01_asked_against_done.py
Writes 01_asked_against_done.out.txt, 01_asked_against_done_steps.csv and 01_asked_against_done.png.
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import line_and_puck
import step_length_common as common

from config import PUCK_PARAMETERS

ASKED_PARAMETERS = {
    # asked lengths counted as short (cm)
    "short_asked_cm": [0.3, 0.5],
    # the puck is off centre when its centre lies this far outside the ink (the centre line is then 0.5 cm further),
    # and off the line when its rim no longer reaches the ink: half the puck's diameter in config.PUCK_PARAMETERS
    "off_centre_cm": 0.6,
    "off_line_cm": 100.0 * PUCK_PARAMETERS["diameter_m"] / 2.0,
    # a mark needs this many pictures in a row, so that one noisy puck position is not a departure
    "pictures_in_a_row": 2,
    # the decisions before a departure whose asked lengths are summarised
    "decisions_before_departure": 6
}



def first_run_of_true(
        values: numpy.ndarray,
        length: int,
        first_index: int = 0
) -> Optional[int]:
    """Return the index at which the first run of ``length`` True values starts at or after ``first_index``, or None."""

    count = 0
    for index in range(first_index, len(values)):
        count = count + 1 if values[index] else 0
        if count >= length:
            return index - length + 1
    return None



def run_table(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    Return one row per picture of the ten runs: what was asked, what was done, and how far the puck was from the line.

    Returns
    -------
    pandas.DataFrame
        "run", "model", "sheet", "outcome", "step", "action", "asked_cm", "asked_angle_deg", "asked_twist_deg",
        "done_cm", "twist_done_deg", "tip_x_cm", "tip_y_cm", "puck_x_cm", "puck_y_cm", "puck_outside_ink_cm" (0 when
        the puck's centre is on the ink) and "is_direction_kept".
    """

    tables = []
    for run in common.STEP_LENGTH_PARAMETERS["robot_runs"]:
        steps = common.load_robot_run(run)
        track = dict(numpy.load(line_and_puck.run_track_path(run["number"])))
        distance = line_and_puck.distance_to_line_m(track, is_start_added = True)
        puck = track["puck_xy_m"]
        tables.append(pandas.DataFrame({
            "run": run["number"], "model": run["model"], "sheet": run["sheet"], "outcome": run["outcome"], "step": steps["step"], "action": steps["action"],
            "asked_cm": steps["asked_cm"], "asked_angle_deg": steps["model_angle_deg"], "asked_twist_deg": steps["model_twist_deg"],
            "stop_probability": steps["model_stop_probability"], "done_cm": steps["done_cm"], "twist_done_deg": steps["twist_done_deg"],
            "tip_x_cm": 100.0 * steps["tip_x"], "tip_y_cm": 100.0 * steps["tip_y"], "puck_x_cm": 100.0 * puck[:, 0], "puck_y_cm": 100.0 * puck[:, 1],
            "puck_outside_ink_cm": 100.0 * line_and_puck.values_at_points(distance, track, puck, numpy.nan),
            "is_direction_kept": steps["is_direction_kept"].fillna(0).astype(int)}))
    return pandas.concat(tables, ignore_index = True)



def departure_marks(
        table: pandas.DataFrame,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    Return, per run, the first step at which the puck was off centre and off the line, judged from the pictures.

    The first centimetres of a line are hidden under the puck in the first pictures; line_and_puck.distance_to_line_m
    draws them in from the puck's first place, so a run is judged from its first picture.
    """

    settings = dict(ASKED_PARAMETERS)
    settings.update(parameters or {})
    rows = []
    for number, group in table.groupby("run"):
        outside = group["puck_outside_ink_cm"].to_numpy()
        off_centre = first_run_of_true(outside >= settings["off_centre_cm"], settings["pictures_in_a_row"])
        off_line = first_run_of_true(outside > settings["off_line_cm"], settings["pictures_in_a_row"])
        asked = group["asked_cm"].to_numpy()
        is_move = (group["action"] == "model_step").to_numpy()
        before = slice(max(0, off_line - settings["decisions_before_departure"]), off_line) if off_line is not None else slice(0, 0)
        rows.append({"run": number, "model": group["model"].iloc[0], "sheet": group["sheet"].iloc[0], "outcome": group["outcome"].iloc[0],
                     "off_centre_step": off_centre, "off_line_step": off_line,
                     "largest_outside_ink_cm": float(numpy.nanmax(outside)),
                     "asked_before_off_line_median_cm": float(numpy.median(asked[before])) if off_line else numpy.nan,
                     "asked_before_off_line_least_cm": float(numpy.min(asked[before])) if off_line else numpy.nan,
                     "asked_on_line_median_cm": float(numpy.median(asked[is_move & (outside <= settings["off_centre_cm"])])) if (is_move & (outside <= settings["off_centre_cm"])).any() else numpy.nan,
                     "asked_off_line_median_cm": float(numpy.median(asked[is_move & (outside > settings["off_line_cm"])])) if (is_move & (outside > settings["off_line_cm"])).any() else numpy.nan})
    return pandas.DataFrame(rows)



def main() -> None:

    settings = dict(ASKED_PARAMETERS)
    lines: List[str] = []
    table = run_table(settings)
    table.round(3).to_csv(common.EVIDENCE_FOLDER / "01_asked_against_done_steps.csv", index = False)
    pandas.set_option("display.width", 250)

    summary = []
    for number, group in table.groupby("run"):
        moves = group[group["action"] == "model_step"]
        summary.append({"run": number, "model": group["model"].iloc[0], "sheet": group["sheet"].iloc[0][-2:], "outcome": group["outcome"].iloc[0],
                        "pictures": len(group), "moves": len(moves),
                        "asked_median_cm": moves["asked_cm"].median(), "asked_mean_cm": moves["asked_cm"].mean(),
                        "asked_under_0.3_share": (moves["asked_cm"] < settings["short_asked_cm"][0]).mean(),
                        "asked_under_0.5_share": (moves["asked_cm"] < settings["short_asked_cm"][1]).mean(),
                        "asked_path_cm": moves["asked_cm"].sum(), "done_path_cm": group["done_cm"].sum(),
                        "done_median_cm": group["done_cm"].median(), "done_least_cm": group["done_cm"].min(), "done_most_cm": group["done_cm"].max(),
                        "done_over_asked": group["done_cm"].sum() / moves["asked_cm"].sum(),
                        "twist_done_per_cm_done": moves["twist_done_deg"].abs().sum() / group["done_cm"].sum(),
                        "twist_asked_per_cm_asked": moves["asked_twist_deg"].abs().sum() / moves["asked_cm"].sum(),
                        "directions_kept": int(moves["is_direction_kept"].sum())})
    summary = pandas.DataFrame(summary)
    common.print_and_keep(lines, "ASKED AGAINST DONE, per run [measured: steps.csv of the ten runs in 03_data/model_runs]")
    common.print_and_keep(lines, "asked = hypot(model_tip_forward_cm, model_tip_left_cm); done = the arm model's tip move to the next picture")
    common.print_and_keep(lines, summary.round(3).to_string(index = False))
    common.print_and_keep(lines)
    all_moves = table[table["action"] == "model_step"]
    common.print_and_keep(lines, f"all {len(all_moves)} moves: done median {table['done_cm'].median():.3f} cm (5th to 95th percentile "
                                 f"{table['done_cm'].quantile(0.05):.3f} to {table['done_cm'].quantile(0.95):.3f}); asked median {all_moves['asked_cm'].median():.3f} cm; "
                                 f"asked under 0.5 cm on {100 * (all_moves['asked_cm'] < 0.5).mean():.1f} % and over 1.0 cm on {100 * (all_moves['asked_cm'] > 1.0).mean():.1f} % of the moves; "
                                 f"largest asked {all_moves['asked_cm'].max():.2f} cm")
    common.print_and_keep(lines)

    marks = departure_marks(table, settings)
    common.print_and_keep(lines, "WHERE THE PUCK LEFT THE LINE, per run [measured from the run's pictures: line map and puck track, line_and_puck.py]")
    common.print_and_keep(lines, f"off centre: the puck's centre {settings['off_centre_cm']} cm or more outside the ink; off the line: more than {settings['off_line_cm']:.1f} cm "
                                 f"outside (the puck's rim no longer reaches the ink); each on {settings['pictures_in_a_row']} pictures in a row")
    common.print_and_keep(lines, marks.round(2).to_string(index = False))
    common.print_and_keep(lines)
    for _, mark in marks.iterrows():
        if pandas.isna(mark["off_line_step"]):
            continue
        group = table[table["run"] == mark["run"]]
        first = max(0, int(mark["off_line_step"]) - 9)
        shown = group.iloc[first:int(mark["off_line_step"]) + 3]
        common.print_and_keep(lines, f"run {int(mark['run'])} around its first off-the-line step {int(mark['off_line_step'])}:")
        common.print_and_keep(lines, shown[["step", "asked_cm", "asked_angle_deg", "asked_twist_deg", "done_cm", "twist_done_deg", "puck_outside_ink_cm"]].round(2).to_string(index = False))
        common.print_and_keep(lines)

    figure, axes = pyplot.subplots(5, 2, figsize = (17, 19), sharey = True)
    for axis, (number, group) in zip(axes.T.ravel(), table.groupby("run")):
        moves = group[group["action"] == "model_step"]
        axis.bar(moves["step"], moves["asked_cm"], width = 0.85, color = "tab:blue", label = "asked by the model")
        axis.plot(group["step"], group["done_cm"], color = "black", linewidth = 1.2, label = "done by the driver")
        axis.plot(group["step"], group["puck_outside_ink_cm"], color = "tab:red", linewidth = 1.2, label = "puck centre outside the ink")
        axis.axhline(settings["off_line_cm"], color = "tab:red", linewidth = 0.6, linestyle = ":")
        mark = marks[marks["run"] == number].iloc[0]
        if not pandas.isna(mark["off_line_step"]):
            axis.axvline(mark["off_line_step"], color = "tab:red", linewidth = 1.5, linestyle = "--")
            axis.annotate(f"off the line at step {int(mark['off_line_step'])}", (mark["off_line_step"], 2.25), color = "tab:red", fontsize = 9,
                          ha = "left", va = "top", xytext = (4, 0), textcoords = "offset points")
        axis.set_ylim(0, 2.3)
        axis.set_xlim(-1, max(90, len(group)))
        axis.set_title(f"run {number}: model {group['model'].iloc[0]}, sheet {group['sheet'].iloc[0][-2:]} ({group['outcome'].iloc[0]}); "
                       f"asked median {moves['asked_cm'].median():.2f} cm", fontsize = 10)
        axis.set_ylabel("cm")
        axis.grid(True, alpha = 0.3)
    axes[0, 0].legend(fontsize = 8, loc = "upper right")
    for axis in axes[-1]:
        axis.set_xlabel("decision (picture) number")
    figure.suptitle("Asked step length (bars) against the 1 cm done (black), and how far the puck's centre was outside the ink (red; dotted: the puck's rim leaves the ink)",
                    fontsize = 11)
    figure.tight_layout(rect = (0, 0, 1, 0.98))
    figure.savefig(common.EVIDENCE_FOLDER / "01_asked_against_done.png", dpi = 90)
    (common.EVIDENCE_FOLDER / "01_asked_against_done.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")



if __name__ == "__main__":
    main()
