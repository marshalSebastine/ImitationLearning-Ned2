"""
Where the camera points relative to the PUSH: the "aim" of each picture, in the demonstrations and in the robot runs.

aim = the direction from the magnet tip to the puck on the table, minus the picture heading (degrees; 0 = the puck is
straight up the picture from the tip, positive = the puck is to the picture's left). Unlike the comparison of place
(02_...), the aim needs no knowledge of where the sheet lay: it comes from each picture and its joint angles alone
(puck_positions_*.csv from 06_puck_positions.py).

Parts:
    1  demonstrations: how strictly the demonstrator kept the camera looking along the push (spread of the aim), and how
       often the tip was not touching the puck at all (re-positioning);
    2  robot runs: the aim over each run, against the demonstrations' spread, and in the pictures before each failure;
    3  the twist as a correction of the aim: by how many degrees per step did the demonstrator turn the forearm for
       each degree of aim, and how much do the models turn (fold models on held-out sheets; models A, B and E on the
       robot)? A weaker reaction of the model is under-turning in the sense that matters for closed-loop driving;
    4  a hand-written rule "turn the forearm towards the puck" scored against the demonstrated twist (it would need the
       professor's permission; scored here only to know what it would be worth).

Output: 08_aim_of_the_camera.out.txt, 08_aim_over_runs.png, 08_aim_histograms.png, 08_twist_against_aim.png
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 08_aim_of_the_camera.py
"""

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames, fold_predictions, robot_run_steps, wrapped_deg

AIM_PARAMETERS = {
    # the tip counts as touching the puck when the puck centre is at most this far from the tip (the robot runs, where
    # the tip pushes all the time, give 2.1 to 2.6 cm: see part 2)
    "touching_cm": 2.8,
    "aim_bins_deg": [-180, -60, -40, -25, -15, -5, 5, 15, 25, 40, 60, 180],
    "linear_range_deg": 45.0,
    "rule_gains": [0.1, 0.15, 0.2, 0.3],
    "rule_twist_most_deg": 10.0,
    "window_steps": 10
}



def with_aim(table: pandas.DataFrame) -> pandas.DataFrame:
    """Add "puck_distance_cm", "push_deg" (table direction from the tip to the puck) and "aim_deg" to a table that has the puck and tip places and the picture heading."""

    table = table.copy()
    table["puck_distance_cm"] = numpy.hypot(table["puck_x_cm"] - table["tip_x_cm"], table["puck_y_cm"] - table["tip_y_cm"])
    table["push_deg"] = numpy.degrees(numpy.arctan2(table["puck_y_cm"] - table["tip_y_cm"], table["puck_x_cm"] - table["tip_x_cm"]))
    table["aim_deg"] = wrapped_deg(table["push_deg"] - table["picture_heading_deg"])
    return table



def spread(values: numpy.ndarray) -> str:
    """Return the median, the quartiles and the 5th and 95th percentiles of ``values`` as text."""

    quantiles = numpy.nanquantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return f"median {quantiles[2]:+6.1f}  quartiles {quantiles[1]:+6.1f} to {quantiles[3]:+6.1f}  5 to 95 % {quantiles[0]:+6.1f} to {quantiles[4]:+6.1f}"



def reaction(
        aim: numpy.ndarray,
        twist: numpy.ndarray,
        linear_range: float
) -> Dict[str, float]:
    """Return the least-squares line twist = slope x aim + offset within ``linear_range`` of the median aim, and the correlation."""

    centre = numpy.nanmedian(aim)
    is_used = numpy.isfinite(aim) & numpy.isfinite(twist) & (numpy.abs(aim - centre) <= linear_range)
    slope, offset = numpy.polyfit(aim[is_used], twist[is_used], 1)
    return {"slope": float(slope), "offset": float(offset), "correlation": float(numpy.corrcoef(aim[is_used], twist[is_used])[0, 1]), "rows": int(is_used.sum()), "centre": float(centre)}



def main(parameters: Optional[Dict] = None) -> None:
    """Measure where the camera points relative to the push in the demonstrations and the robot runs, and how the twist reacts to it."""

    settings = dict(AIM_PARAMETERS)
    settings.update(parameters or {})
    report = Report("08_aim_of_the_camera.out.txt")
    frames = demonstration_frames()
    demonstrations = with_aim(frames.merge(pandas.read_csv(EVIDENCE_FOLDER / "puck_positions_demonstrations.csv"), on = "frame_index", validate = "one_to_one"))
    demonstrations = demonstrations[demonstrations["action"] != "stop"].copy()
    demonstrations["is_touching"] = demonstrations["puck_distance_cm"] <= settings["touching_cm"]
    run_pucks = pandas.read_csv(EVIDENCE_FOLDER / "puck_positions_runs.csv")

    report.say("THE AIM OF THE CAMERA: push direction (tip to puck) minus picture heading, degrees; 0 = the puck straight up the picture from the tip, + = to its left")
    report.say("Measured with the team's puck tracker on every picture (06_puck_positions.py); the puck was found in 99.3 % of the demonstration pictures and in all robot pictures.")
    report.say("")
    report.say("1. Demonstrations")
    seen = demonstrations[demonstrations["aim_deg"].notna()]
    report.say(f"   tip to puck centre, cm: {spread(seen['puck_distance_cm'].to_numpy())}")
    report.say(f"   share of pictures with the tip touching the puck (puck centre within {settings['touching_cm']} cm): {seen['is_touching'].mean():.2f}; "
               f"further than 4 cm away: {(seen['puck_distance_cm'] > 4.0).mean():.2f}")
    report.say(f"   aim, all pictures:       {spread(seen['aim_deg'].to_numpy())}")
    report.say(f"   aim, tip touching:       {spread(seen.loc[seen['is_touching'], 'aim_deg'].to_numpy())}")
    report.say(f"   aim, tip not touching:   {spread(seen.loc[~seen['is_touching'], 'aim_deg'].to_numpy())}")
    touching = seen[seen["is_touching"]]
    centre = touching["aim_deg"].median()
    for limit in (10, 20, 30, 45):
        report.say(f"   touching pictures within {limit:2d} degrees of the median aim ({centre:+.1f}): {((touching['aim_deg'] - centre).abs() <= limit).mean():.2f}")
    for sheet, rows in touching.groupby("sheet"):
        report.say(f"   {sheet} touching ({len(rows)}): {spread(rows['aim_deg'].to_numpy())}")
    report.say("")

    report.say("2. Robot runs (the tip pushes all the time, so every picture is compared with the demonstrations' touching pictures)")
    low, high = touching["aim_deg"].quantile(0.05), touching["aim_deg"].quantile(0.95)
    report.say(f"   demonstrations' 5 to 95 % range of the aim when touching: {low:+.1f} to {high:+.1f} degrees")
    figure, axes = pyplot.subplots(5, 2, figsize = (18, 20))
    run_tables = {}
    for position, run in enumerate(ANALYSIS_PARAMETERS["robot_runs"]):
        steps = robot_run_steps(run).merge(run_pucks[run_pucks["run"] == run["number"]].drop(columns = ["run", "tip_x_cm", "tip_y_cm"]), on = "step")
        steps["tip_x_cm"], steps["tip_y_cm"] = 100.0 * steps["tip_x"], 100.0 * steps["tip_y"]
        # the heading is logged only for carried-out steps; the stop row gets the last one
        steps["picture_heading_deg"] = steps["picture_heading_deg"].ffill()
        steps = with_aim(steps)
        run_tables[run["number"]] = steps
        outside = (steps["aim_deg"] < low) | (steps["aim_deg"] > high)
        first_outside = None
        for index in range(len(steps) - 2):
            if outside.iloc[index:index + 3].all():
                first_outside = index
                break
        report.say(f"   run {run['number']:2d} {run['model']} {run['sheet']}: aim {spread(steps['aim_deg'].to_numpy())}; outside the demonstrations' range in {outside.mean():.2f} of the pictures"
                   f"{'' if first_outside is None else ', first 3 in a row from picture ' + str(first_outside)}; tip to puck median {steps['puck_distance_cm'].median():.1f} cm, "
                   f"over 4 cm in {(steps['puck_distance_cm'] > 4.0).mean():.2f}; {run['outcome']}")
        axis = axes.flat[position]
        axis.axhspan(low, high, color = "0.9", label = "demonstrations, 5 to 95 % when touching")
        axis.axhspan(touching["aim_deg"].quantile(0.25), touching["aim_deg"].quantile(0.75), color = "0.78", label = "demonstrations, quartiles")
        axis.plot(steps["step"], steps["aim_deg"], color = "tab:purple", marker = ".", label = "aim: puck direction in the picture, degrees")
        axis.plot(steps["step"], steps["model_angle_deg"], color = "tab:green", linewidth = 0.8, label = "model's travel direction, degrees")
        axis.plot(steps["step"], 10.0 * steps["model_twist_deg"], color = "tab:blue", linewidth = 0.8, label = "model's twist x 10")
        axis.plot(steps["step"], 20.0 * steps["puck_distance_cm"], color = "tab:red", linewidth = 0.8, label = "tip to puck, mm x 2")
        axis.set_ylim(-185, 185)
        axis.set_title(f"run {run['number']} ({run['model']}, {run['sheet']}): {run['outcome']}", fontsize = 9)
        axis.legend(fontsize = 6, loc = "lower left", ncol = 3)
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "08_aim_over_runs.png", dpi = 100)
    pyplot.close(figure)
    report.say("")
    report.say("   in blocks of 10 pictures: median aim, largest |aim - demonstrations' median|, sum of the twist done")
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        steps = run_tables[run["number"]]
        parts = []
        for start in range(0, len(steps), 10):
            block = steps.iloc[start:start + 10]
            parts.append(f"[{start:3d}: {block['aim_deg'].median():+6.1f} {(block['aim_deg'] - centre).abs().max():5.1f} {block['twist_done_deg'].sum():+6.1f}]")
        report.say(f"   run {run['number']:2d} " + " ".join(parts))
    report.say("")

    report.say("3. The twist as a correction of the aim (twist > 0 turns the camera to the right; a puck to the picture's LEFT, aim > 0, calls for twist < 0)")
    report.say(f"   least-squares line twist = slope x aim + offset over pictures within {settings['linear_range_deg']:.0f} degrees of the median aim")
    report.say(f"   {'who':58s} {'rows':>6s} {'slope, deg twist per deg aim':>29s} {'offset':>7s} {'correlation':>11s}")
    fitted = {}
    line = reaction(touching["aim_deg"].to_numpy(), touching["forearm_change_deg"].fillna(0.0).to_numpy(), settings["linear_range_deg"])
    fitted["demonstrations"] = line
    report.say(f"   {'demonstrated twist, tip touching, all sheets':58s} {line['rows']:6d} {line['slope']:29.3f} {line['offset']:7.2f} {line['correlation']:11.2f}")
    line_all = reaction(seen["aim_deg"].to_numpy(), seen["forearm_change_deg"].fillna(0.0).to_numpy(), settings["linear_range_deg"])
    report.say(f"   {'demonstrated twist, all pictures':58s} {line_all['rows']:6d} {line_all['slope']:29.3f} {line_all['offset']:7.2f} {line_all['correlation']:11.2f}")
    for sheet, rows in touching.groupby("sheet"):
        line_sheet = reaction(rows["aim_deg"].to_numpy(), rows["forearm_change_deg"].fillna(0.0).to_numpy(), settings["linear_range_deg"])
        report.say(f"   {'demonstrated twist, tip touching, ' + sheet:58s} {line_sheet['rows']:6d} {line_sheet['slope']:29.3f} {line_sheet['offset']:7.2f} {line_sheet['correlation']:11.2f}")
    for label in ("A recipe (exact)", "E recipe (chunk, stop 25)"):
        predictions = fold_predictions(ANALYSIS_PARAMETERS["fold_runs"][label], frames)[["frame_index", "predicted_twist_deg"]]
        joined = touching.merge(predictions, on = "frame_index")
        line_model = reaction(joined["aim_deg"].to_numpy(), joined["predicted_twist_deg"].to_numpy(), settings["linear_range_deg"])
        fitted[label] = line_model
        report.say(f"   {'fold models ' + label + ', held-out sheets, same pictures':58s} {line_model['rows']:6d} {line_model['slope']:29.3f} {line_model['offset']:7.2f} {line_model['correlation']:11.2f}"
                   f"   = {line_model['slope'] / line['slope']:.2f} of the demonstrated reaction")
        for sheet, rows in joined.groupby("sheet"):
            line_sheet = reaction(rows["aim_deg"].to_numpy(), rows["predicted_twist_deg"].to_numpy(), settings["linear_range_deg"])
            demonstrated_sheet = reaction(rows["aim_deg"].to_numpy(), rows["forearm_change_deg"].fillna(0.0).to_numpy(), settings["linear_range_deg"])
            report.say(f"   {'    ' + sheet:58s} {line_sheet['rows']:6d} {line_sheet['slope']:29.3f} {line_sheet['offset']:7.2f} {line_sheet['correlation']:11.2f}"
                       f"   = {line_sheet['slope'] / demonstrated_sheet['slope']:.2f} of the demonstrated reaction on this sheet")
    all_runs = pandas.concat([table.assign(run = number) for number, table in run_tables.items()], ignore_index = True)
    all_runs = all_runs[all_runs["action"] == "model_step"]
    for name, chosen in (("robot, all ten runs (models A, B, E trained on all sheets)", all_runs),
                         ("robot, runs that reached the end (1, 2, 4, 7)", all_runs[all_runs["run"].isin([1, 2, 4, 7])]),
                         ("robot, failed runs (3, 5, 6, 8, 9, 10)", all_runs[all_runs["run"].isin([3, 5, 6, 8, 9, 10])])):
        line_robot = reaction(chosen["aim_deg"].to_numpy() - (numpy.nanmedian(chosen["aim_deg"]) - centre) * 0.0, chosen["model_twist_deg"].to_numpy(), settings["linear_range_deg"])
        fitted[name] = line_robot
        report.say(f"   {name:58s} {line_robot['rows']:6d} {line_robot['slope']:29.3f} {line_robot['offset']:7.2f} {line_robot['correlation']:11.2f}"
                   f"   = {line_robot['slope'] / line['slope']:.2f} of the demonstrated reaction (per decision)")
    report.say("")
    report.say("   mean twist per bin of the aim (degrees per step): demonstrated (touching), fold models on the same pictures, and the robot's models")
    edges = settings["aim_bins_deg"]
    joined_models = touching.copy()
    for label in ("A recipe (exact)", "E recipe (chunk, stop 25)"):
        predictions = fold_predictions(ANALYSIS_PARAMETERS["fold_runs"][label], frames)[["frame_index", "predicted_twist_deg"]].rename(columns = {"predicted_twist_deg": label})
        joined_models = joined_models.merge(predictions, on = "frame_index")
    report.say(f"   {'aim':>14s} {'demo rows':>9s} {'demonstrated':>12s} {'A fold':>8s} {'E fold':>8s} {'robot rows':>10s} {'robot (asked)':>13s} {'demo step cm':>12s} {'robot asked cm':>14s}")
    centres, demonstrated_means, fold_means, robot_means = [], [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        chosen = joined_models[(joined_models["aim_deg"] > lower) & (joined_models["aim_deg"] <= upper)]
        robot_chosen = all_runs[(all_runs["aim_deg"] > lower) & (all_runs["aim_deg"] <= upper)]
        report.say(f"   {format(lower, '+d') + ' to ' + format(upper, '+d'):>14s} {len(chosen):9d} {chosen['forearm_change_deg'].mean():+12.2f} {chosen['A recipe (exact)'].mean():+8.2f} "
                   f"{chosen['E recipe (chunk, stop 25)'].mean():+8.2f} {len(robot_chosen):10d} {robot_chosen['model_twist_deg'].mean():+13.2f} {chosen['step_cm'].mean():12.2f} {robot_chosen['asked_cm'].mean():14.2f}")
        centres.append(chosen["aim_deg"].mean())
        demonstrated_means.append(chosen["forearm_change_deg"].mean())
        fold_means.append(chosen["E recipe (chunk, stop 25)"].mean())
        robot_means.append(robot_chosen["model_twist_deg"].mean())
    figure, axis = pyplot.subplots(figsize = (9, 6))
    axis.plot(centres, demonstrated_means, color = "black", marker = "o", linewidth = 2, label = "demonstrated twist per step (tip touching)")
    axis.plot(centres, fold_means, color = "tab:red", marker = "o", label = "E-recipe fold models on the same pictures (sheet held out)")
    axis.plot(centres, robot_means, color = "tab:blue", marker = "o", label = "models on the robot, 11.11.26 (per decision; the tip then moved 1 cm)")
    axis.axhline(0.0, color = "0.5", linewidth = 0.5)
    axis.axvline(centre, color = "0.5", linewidth = 0.5)
    axis.set_xlim(-90, 90)
    axis.set_xlabel("aim: puck direction in the picture, degrees (+ = puck to the left)")
    axis.set_ylabel("mean twist per step, degrees (+ = camera turns right)")
    axis.set_title("The twist as a correction of the aim")
    axis.grid(alpha = 0.3)
    axis.legend(fontsize = 8)
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "08_twist_against_aim.png", dpi = 100)
    pyplot.close(figure)
    report.say("")

    report.say("   how fast the aim comes back: the aim N steps later against the aim now (slope of a line through the median; 1 = no correction, 0 = fully corrected)")
    ordered = seen.sort_values(["episode_id", "row_in_episode"])
    for label, table, group in (("demonstrations", ordered, "episode_id"), ("robot runs", all_runs.sort_values(["run", "step"]), "run")):
        parts = []
        for ahead in (1, 3, 5, 10):
            later = table.groupby(group)["aim_deg"].shift(-ahead)
            is_used = later.notna() & table["aim_deg"].notna() & ((table["aim_deg"] - centre).abs() <= 60)
            now_values, later_values = table.loc[is_used, "aim_deg"].to_numpy() - centre, later[is_used].to_numpy() - centre
            parts.append(f"after {ahead:2d} steps {float((now_values * later_values).sum() / (now_values ** 2).sum()):.2f}")
        report.say(f"   {label:16s} " + ", ".join(parts))
    report.say("")

    report.say("4. A hand-written rule 'turn the forearm towards the puck': twist = -gain x (aim - usual aim), at most 10 degrees per step")
    report.say("   (scored on the demonstrations' touching pictures; it uses the puck tracker, not the model, so it is NOT imitation learning and would need the professor's permission)")
    table = touching.sort_values(["episode_id", "row_in_episode"]).merge(
        fold_predictions(ANALYSIS_PARAMETERS["fold_runs"]["E recipe (chunk, stop 25)"], frames)[["frame_index", "predicted_twist_deg"]], on = "frame_index")
    demonstrated = table["forearm_change_deg"].fillna(0.0).to_numpy()
    report.say(f"   {'predictor':44s} {'correlation with the demonstrated twist':>40s} {'mean |twist|':>12s} {'sign right (demo >= 2 deg)':>27s}")
    is_twisting = numpy.abs(demonstrated) >= 2.0
    report.say(f"   {'E-recipe fold model (held-out sheets)':44s} {numpy.corrcoef(table['predicted_twist_deg'], demonstrated)[0, 1]:40.2f} {table['predicted_twist_deg'].abs().mean():12.2f} "
               f"{(numpy.sign(table['predicted_twist_deg'].to_numpy()[is_twisting]) == numpy.sign(demonstrated[is_twisting])).mean():27.2f}")
    for gain in settings["rule_gains"]:
        rule = numpy.clip(-gain * (table["aim_deg"].to_numpy() - centre), -settings["rule_twist_most_deg"], settings["rule_twist_most_deg"])
        report.say(f"   {'rule with gain ' + format(gain, '.2f'):44s} {numpy.corrcoef(rule, demonstrated)[0, 1]:40.2f} {numpy.abs(rule).mean():12.2f} "
                   f"{(numpy.sign(rule[is_twisting]) == numpy.sign(demonstrated[is_twisting])).mean():27.2f}")
    report.say(f"   demonstrated mean |twist| on these pictures: {numpy.abs(demonstrated).mean():.2f} degrees")

    # histograms of the aim
    figure, axes = pyplot.subplots(2, 3, figsize = (18, 9))
    bins = numpy.arange(-180, 181, 10)
    for position, sheet in enumerate(ANALYSIS_PARAMETERS["sheets"]):
        axis = axes.flat[position]
        rows = touching[touching["sheet"] == sheet]
        axis.hist(rows["aim_deg"], bins = bins, density = True, color = "0.7", label = f"demonstrations, tip touching ({len(rows)})")
        for run in ANALYSIS_PARAMETERS["robot_runs"]:
            if run["sheet"] == sheet:
                axis.hist(run_tables[run["number"]]["aim_deg"], bins = bins, density = True, histtype = "step", linewidth = 1.5, label = f"run {run['number']} ({run['model']}): {run['outcome']}")
        axis.set_title(sheet)
        axis.set_xlabel("aim: puck direction in the picture, degrees (0 = straight up from the tip, + = left)")
        axis.legend(fontsize = 7)
    figure.suptitle("Where the camera points relative to the push: demonstrations (grey) against the robot runs of 11.11.26")
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "08_aim_histograms.png", dpi = 100)
    report.save()



if __name__ == "__main__":
    main()
