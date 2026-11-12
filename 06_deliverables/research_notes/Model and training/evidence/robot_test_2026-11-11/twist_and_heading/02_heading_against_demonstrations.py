"""
The camera heading of each robot run against the demonstrations AT THE SAME PLACE on the table.

For the seven runs whose sheet lay as in a plan (runs 4 to 10), every robot picture is compared with the rows of the
standard-start demonstrations of the same sheet and end whose magnet tip was within "near_cm" of the robot tip and that
travelled the same way (within "same_way_deg"). Per robot picture:
    off_path_cm        distance from the robot tip to the nearest demonstrated tip position (how far the tip is from
                       where a demonstration ever was);
    heading_gap_deg    robot picture heading minus the demonstrations' mean picture heading at that place (positive =
                       the robot camera is turned further to the left, counter-clockwise seen from above);
    joint_4_gap_deg    the same for joint 4;
    travel_gap_deg     robot travel direction on the table minus the demonstrations' at that place.
The question: does the camera heading fall behind the demonstrations BEFORE the tip leaves the demonstrated path?

Output: 02_heading_against_demonstrations.out.txt, 02_top_view_run_NN.png, 02_heading_gap_timeline.png,
02_heading_gap_per_step.csv
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 02_heading_against_demonstrations.py
"""

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames, robot_run_steps, wrapped_deg

HEADING_GAP_PARAMETERS = {
    "near_cm": 1.5,
    "same_way_deg": 60.0,
    "reference_start_kind": "STD",
    "off_path_limit_cm": 2.0,
    "heading_arrow_cm": 1.2
}



def circular_mean_deg(angles_deg: numpy.ndarray) -> float:
    """Return the mean direction of angles in degrees."""

    radians = numpy.radians(angles_deg)
    return float(numpy.degrees(numpy.arctan2(numpy.sin(radians).mean(), numpy.cos(radians).mean())))



def main(parameters: Optional[Dict] = None) -> None:
    """Compare every robot picture with the demonstrations at the same place and write the tables, top views and timeline."""

    settings = dict(HEADING_GAP_PARAMETERS)
    settings.update(parameters or {})
    report = Report("02_heading_against_demonstrations.out.txt")
    frames = demonstration_frames()
    frames = frames[frames["action"] != "stop"].copy()
    frames["travel_deg"] = wrapped_deg(frames["picture_heading_deg"] + frames["move_angle_deg"])

    report.say("THE ROBOT'S CAMERA HEADING AGAINST THE DEMONSTRATIONS AT THE SAME PLACE (measured: steps.csv of the runs, frames.csv)")
    report.say(f"reference: standard-start demonstrations of the same sheet and end; rows within {settings['near_cm']} cm of the robot tip that travel within {settings['same_way_deg']:.0f} degrees of the robot's way")
    report.say("heading gap > 0: the robot camera is turned further LEFT (counter-clockwise from above) than the demonstrations' there")
    report.say("")

    all_rows = []
    timeline_figure, timeline_axes = pyplot.subplots(7, 1, figsize = (14, 24))
    plotted = 0
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        if run["plan"] is None:
            continue
        steps = robot_run_steps(run)
        start_end = frames.loc[frames["plan_number"] == run["plan"], "start_end"].iloc[0]
        reference = frames[(frames["sheet"] == run["sheet"]) & (frames["start_kind"] == settings["reference_start_kind"]) & (frames["start_end"] == start_end)]
        reference_xy = 100.0 * reference[["tip_x_m", "tip_y_m"]].to_numpy()
        reference_heading = reference["picture_heading_deg"].to_numpy()
        reference_joint_4 = reference["joint_4_deg"].to_numpy()
        reference_travel = reference["travel_deg"].to_numpy()
        reference_long = (reference["step_cm"] >= 0.5).to_numpy()

        rows = []
        for _, step in steps.iterrows():
            tip_xy = 100.0 * numpy.array([step["tip_x"], step["tip_y"]])
            distance = numpy.hypot(*(reference_xy - tip_xy).T)
            heading = step["picture_heading_deg"]
            row = {"run": run["number"], "step": int(step["step"]), "arc_cm": step["arc_cm"], "tip_x_cm": tip_xy[0], "tip_y_cm": tip_xy[1],
                   "off_path_cm": float(distance.min()), "joint_4_deg": step["joint_4_deg"], "picture_heading_deg": heading,
                   "model_angle_deg": step["model_angle_deg"], "model_twist_deg": step["model_twist_deg"], "asked_cm": step["asked_cm"],
                   "stop_probability": step["model_stop_probability"]}
            if numpy.isfinite(heading):
                travel = wrapped_deg(heading + step["model_angle_deg"])
                is_near = (distance <= settings["near_cm"]) & (numpy.abs(wrapped_deg(reference_travel - travel)) <= settings["same_way_deg"]) & reference_long
                if not is_near.any():
                    is_near = (distance <= settings["near_cm"]) & (numpy.abs(wrapped_deg(reference_travel - travel)) <= settings["same_way_deg"])
                row["travel_deg"] = float(travel)
                row["reference_rows"] = int(is_near.sum())
                if is_near.any():
                    row["heading_gap_deg"] = float(wrapped_deg(heading - circular_mean_deg(reference_heading[is_near])))
                    row["joint_4_gap_deg"] = float(step["joint_4_deg"] - reference_joint_4[is_near].mean())
                    row["travel_gap_deg"] = float(wrapped_deg(travel - circular_mean_deg(reference_travel[is_near])))
                    row["reference_heading_spread_deg"] = float(numpy.std(wrapped_deg(reference_heading[is_near] - circular_mean_deg(reference_heading[is_near]))))
            rows.append(row)
        table = pandas.DataFrame(rows)
        all_rows.append(table)

        is_off = table["off_path_cm"] > settings["off_path_limit_cm"]
        # the first picture from which the tip stays off the demonstrated path for 3 pictures in a row
        first_off = None
        for position in range(len(table) - 2):
            if is_off.iloc[position:position + 3].all():
                first_off = position
                break
        report.say(f"run {run['number']} ({run['model']}, {run['sheet']}, as plan {run['plan']}, end {start_end}; {reference['episode_id'].nunique()} reference demonstrations; {run['outcome']})")
        is_started_elsewhere = first_off == 0
        if is_started_elsewhere:
            # the sheet or the hand-placed start did not lie where the demonstrations' did, so the comparison of place holds
            # only for the pictures that come within the limit of a demonstrated tip position later; all are listed
            report.say(f"   the run STARTS {table['off_path_cm'].iloc[0]:.1f} cm from every demonstrated tip position (the sheet or the start lay elsewhere than in the demonstrations): "
                       f"only the pictures within {settings['off_path_limit_cm']} cm of a demonstrated position are compared, and less weight belongs on this run")
            first_off = None
        else:
            report.say(f"   first picture of 3 in a row more than {settings['off_path_limit_cm']} cm from every demonstrated tip position: "
                       f"{'never' if first_off is None else 'step ' + str(first_off)}")
        on_path = table[(table["off_path_cm"] <= settings["off_path_limit_cm"]) & table["heading_gap_deg"].notna()] if "heading_gap_deg" in table else table.iloc[:0]
        if len(on_path):
            before = on_path if first_off is None else on_path[on_path["step"] < first_off]
            report.say(f"   on the path ({len(before)} pictures{'' if first_off is None else ' before leaving it'}): heading gap median {before['heading_gap_deg'].median():+.1f}, "
                       f"median |gap| {before['heading_gap_deg'].abs().median():.1f}, largest |gap| {before['heading_gap_deg'].abs().max():.1f} degrees; "
                       f"demonstrations' own spread of heading at one place (median of the standard deviations) {before['reference_heading_spread_deg'].median():.1f} degrees")
        report.say(f"   {'step':>4s} {'arc cm':>6s} {'off cm':>6s} {'head gap':>8s} {'j4 gap':>7s} {'travel gap':>10s} {'model angle':>11s} {'twist':>6s} {'asked cm':>8s} {'demo rows':>9s}")
        last_listed = len(table) if first_off is None else min(len(table), first_off + 8)
        for _, row in table.iloc[:last_listed].iterrows():
            gap = row.get("heading_gap_deg", float("nan"))
            report.say(f"   {int(row['step']):4d} {row['arc_cm']:6.1f} {row['off_path_cm']:6.1f} {gap:8.1f} {row.get('joint_4_gap_deg', float('nan')):7.1f} {row.get('travel_gap_deg', float('nan')):10.1f} "
                       f"{row['model_angle_deg']:11.1f} {row['model_twist_deg']:6.1f} {row['asked_cm']:8.2f} {row.get('reference_rows', 0):9.0f}"
                       + ("   <- leaves the demonstrated path" if first_off is not None and int(row["step"]) == first_off else ""))
        report.say("")

        # top view
        figure, axis = pyplot.subplots(figsize = (9, 8))
        for episode_id, episode in reference.groupby("episode_id"):
            axis.plot(100.0 * episode["tip_x_m"], 100.0 * episode["tip_y_m"], color = "0.7", linewidth = 0.8)
            sparse = episode.iloc[::4]
            axis.quiver(100.0 * sparse["tip_x_m"], 100.0 * sparse["tip_y_m"], numpy.cos(numpy.radians(sparse["picture_heading_deg"])), numpy.sin(numpy.radians(sparse["picture_heading_deg"])),
                        color = "0.55", angles = "xy", scale_units = "xy", scale = 1.0 / settings["heading_arrow_cm"], width = 0.002)
        axis.plot(100.0 * steps["tip_x"], 100.0 * steps["tip_y"], color = "tab:red", linewidth = 1.5, marker = ".", markersize = 4)
        has_heading = steps["picture_heading_deg"].notna()
        axis.quiver(100.0 * steps.loc[has_heading, "tip_x"], 100.0 * steps.loc[has_heading, "tip_y"], numpy.cos(numpy.radians(steps.loc[has_heading, "picture_heading_deg"])),
                    numpy.sin(numpy.radians(steps.loc[has_heading, "picture_heading_deg"])), color = "tab:red", angles = "xy", scale_units = "xy",
                    scale = 1.0 / (1.5 * settings["heading_arrow_cm"]), width = 0.003)
        for number in range(0, len(steps), 5):
            axis.annotate(str(number), (100.0 * steps["tip_x"].iloc[number], 100.0 * steps["tip_y"].iloc[number]), fontsize = 7, color = "darkred",
                          xytext = (3, 3), textcoords = "offset points")
        axis.set_aspect("equal")
        axis.set_xlabel("x from the base axis, cm (away from the robot)")
        axis.set_ylabel("y, cm (left is up)")
        axis.set_title(f"Run {run['number']} ({run['model']}, {run['sheet']}, {run['outcome']}): magnet tip path and camera heading (arrows = picture top)\n"
                       f"red = robot run with step numbers, grey = {reference['episode_id'].nunique()} standard-start demonstrations of plan {run['plan']}'s end", fontsize = 9)
        axis.grid(alpha = 0.3)
        figure.tight_layout()
        figure.savefig(EVIDENCE_FOLDER / f"02_top_view_run_{run['number']:02d}.png", dpi = 110)
        pyplot.close(figure)

        # timeline
        axis = timeline_axes[plotted]
        plotted += 1
        if "heading_gap_deg" in table:
            axis.plot(table["step"], table["heading_gap_deg"], color = "tab:blue", marker = ".", label = "camera heading gap to the demonstrations at the same place, degrees")
        axis.plot(table["step"], table["model_angle_deg"], color = "tab:green", linewidth = 1, label = "model's travel direction in the picture, degrees (0 = picture top)")
        axis.plot(table["step"], 10.0 * table["off_path_cm"], color = "tab:red", linewidth = 1, label = "tip distance from the nearest demonstrated tip position, mm")
        axis.axhline(0.0, color = "0.5", linewidth = 0.5)
        if first_off is not None:
            axis.axvline(first_off, color = "black", linestyle = "--", linewidth = 1)
        axis.set_ylim(-120, 120)
        axis.set_title(f"run {run['number']} ({run['model']}, {run['sheet']}): {run['outcome']}", fontsize = 9)
        axis.legend(fontsize = 7, loc = "upper right", ncol = 3)
    timeline_axes[-1].set_xlabel("picture number of the run (dashed line: the tip leaves the demonstrated path)")
    timeline_figure.tight_layout()
    timeline_figure.savefig(EVIDENCE_FOLDER / "02_heading_gap_timeline.png", dpi = 100)
    pandas.concat(all_rows, ignore_index = True).to_csv(EVIDENCE_FOLDER / "02_heading_gap_per_step.csv", index = False)
    report.save()



if __name__ == "__main__":
    main()
