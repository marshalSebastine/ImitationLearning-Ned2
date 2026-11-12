"""
The direction of travel in the PICTURE frame (0 = towards the picture's top, positive = towards its left): how the
demonstrations spread around "straight up", how the twist goes with it, and the same for the ten robot runs.

Parts:
    1  demonstrations: the spread of move_angle_deg per sheet, for pushing steps and for re-aiming steps;
    2  demonstrations: which way the forearm twists when the tip travels to the picture's left or right (the sign a
       hand-written rule "turn the forearm towards the direction of travel" would need);
    3  robot runs: the spread of model_angle_deg per run, and its course in the pictures before each failure.

Output: 03_picture_direction.out.txt, 03_picture_direction_histograms.png, 03_model_angle_over_runs.png
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 03_picture_direction.py
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames, robot_run_steps

PICTURE_DIRECTION_PARAMETERS = {
    # a step counts as "pushing" when the tip moved at least this far, and as "re-aiming" when it moved less and the
    # forearm twisted at least "reaim_twist_deg"
    "pushing_step_cm": 0.7,
    "reaim_twist_deg": 3.0,
    "cones_deg": [15.0, 30.0, 45.0, 90.0],
    "smallest_direction_cm": 0.05
}



def spread_line(
        name: str,
        angles_deg: numpy.ndarray,
        cones_deg: List[float]
) -> str:
    """Return one table line with the spread of ``angles_deg`` around zero."""

    absolute = numpy.abs(angles_deg)
    shares = " ".join(f"{(absolute <= cone).mean():6.2f}" for cone in cones_deg)
    return (f"{name:28s} {len(angles_deg):6d} {numpy.median(angles_deg):+7.1f} {numpy.median(absolute):7.1f} {numpy.quantile(absolute, 0.9):7.1f} {shares}")



def main(parameters: Optional[Dict] = None) -> None:
    """Print how the direction of travel in the picture is spread in the demonstrations and in the robot runs, and draw the figures."""

    settings = dict(PICTURE_DIRECTION_PARAMETERS)
    settings.update(parameters or {})
    report = Report("03_picture_direction.out.txt")
    frames = demonstration_frames()
    moving = frames[(frames["action"] != "stop") & (frames["step_cm"] >= settings["smallest_direction_cm"])].copy()
    cones = settings["cones_deg"]
    header = f"{'':28s} {'rows':>6s} {'median':>7s} {'med |a|':>7s} {'90% |a|':>7s} " + " ".join(f"{'<=' + format(cone, '.0f'):>6s}" for cone in cones)

    report.say("THE DIRECTION OF TRAVEL IN THE PICTURE FRAME (degrees; 0 = picture top, + = picture left). Measured: frames.csv (audit cleaning), steps.csv of the runs")
    report.say("")
    report.say("1. Demonstrations: how far from 'straight up the picture' the tip travelled")
    report.say(header)
    is_pushing = moving["step_cm"] >= settings["pushing_step_cm"]
    is_reaiming = (~is_pushing) & (moving["forearm_change_deg"].abs() >= settings["reaim_twist_deg"])
    report.say(spread_line("all steps with a direction", moving["move_angle_deg"].to_numpy(), cones))
    report.say(spread_line(f"pushing (>= {settings['pushing_step_cm']} cm)", moving.loc[is_pushing, "move_angle_deg"].to_numpy(), cones))
    report.say(spread_line(f"re-aiming (< {settings['pushing_step_cm']} cm, twist >= {settings['reaim_twist_deg']:.0f})", moving.loc[is_reaiming, "move_angle_deg"].to_numpy(), cones))
    report.say(spread_line("other short steps", moving.loc[(~is_pushing) & (~is_reaiming), "move_angle_deg"].to_numpy(), cones))
    for sheet, rows in moving.groupby("sheet"):
        report.say(spread_line(f"{sheet} all", rows["move_angle_deg"].to_numpy(), cones))
        report.say(spread_line(f"{sheet} pushing", rows.loc[rows["step_cm"] >= settings["pushing_step_cm"], "move_angle_deg"].to_numpy(), cones))
    all_rows = frames[frames["action"] != "stop"]
    report.say(f"   share of all non-stop rows: pushing {(all_rows['step_cm'] >= settings['pushing_step_cm']).mean():.2f}, "
               f"re-aiming {((all_rows['step_cm'] < settings['pushing_step_cm']) & (all_rows['forearm_change_deg'].abs() >= settings['reaim_twist_deg'])).mean():.2f}; "
               f"share of all the twisting (sum of |twist|) done in re-aiming rows: "
               f"{all_rows.loc[(all_rows['step_cm'] < settings['pushing_step_cm']) & (all_rows['forearm_change_deg'].abs() >= settings['reaim_twist_deg']), 'forearm_change_deg'].abs().sum() / all_rows['forearm_change_deg'].abs().sum():.2f}; "
               f"median tip travel of a re-aiming row {all_rows.loc[(all_rows['step_cm'] < settings['pushing_step_cm']) & (all_rows['forearm_change_deg'].abs() >= settings['reaim_twist_deg']), 'step_cm'].median():.2f} cm, "
               f"median |twist| {all_rows.loc[(all_rows['step_cm'] < settings['pushing_step_cm']) & (all_rows['forearm_change_deg'].abs() >= settings['reaim_twist_deg']), 'forearm_change_deg'].abs().median():.1f} degrees")
    report.say("")

    report.say("2. Demonstrations: which way the forearm twists when the tip travels to the picture's left (+) or right (-)")
    report.say("   (twist > 0 turns the camera to the RIGHT, clockwise seen from above: the picture heading changes by -1.01 x twist, fitted on frames.csv)")
    report.say(f"   {'travel direction in the picture':34s} {'rows':>6s} {'mean twist':>10s} {'median twist':>12s} {'share twist>+2':>14s} {'share twist<-2':>14s} {'median step cm':>14s}")
    edges = [-180, -90, -45, -15, 15, 45, 90, 180]
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = moving[(moving["move_angle_deg"] > low) & (moving["move_angle_deg"] <= high)]
        report.say(f"   {format(low, '+d') + ' to ' + format(high, '+d'):34s} {len(chosen):6d} {chosen['forearm_change_deg'].mean():+10.2f} {chosen['forearm_change_deg'].median():+12.2f} "
                   f"{(chosen['forearm_change_deg'] > 2).mean():14.2f} {(chosen['forearm_change_deg'] < -2).mean():14.2f} {chosen['step_cm'].median():14.2f}")
    correlation = numpy.corrcoef(moving["move_angle_deg"], moving["forearm_change_deg"])[0, 1]
    within = moving[moving["move_angle_deg"].abs() <= 90]
    slope = float((within["move_angle_deg"] * within["forearm_change_deg"]).sum() / (within["move_angle_deg"] ** 2).sum())
    report.say(f"   correlation of twist with the travel direction: {correlation:+.3f} (all), {numpy.corrcoef(within['move_angle_deg'], within['forearm_change_deg'])[0, 1]:+.3f} (within 90 degrees); "
               f"least-squares twist per degree of travel direction (within 90): {slope:+.4f}")
    report.say("   reading: a POSITIVE number means that the camera is turned to the right while the tip travels to the picture's LEFT,")
    report.say("   that is AWAY from the direction of travel: at a bend the tip swings to the outside of the bend to get behind the puck,")
    report.say("   and the camera turns to the inside. A rule 'turn the forearm towards the direction of travel' would need a NEGATIVE number.")
    for sheet, rows in moving.groupby("sheet"):
        rows = rows[rows["move_angle_deg"].abs() <= 90]
        report.say(f"   {sheet}: correlation {numpy.corrcoef(rows['move_angle_deg'], rows['forearm_change_deg'])[0, 1]:+.3f}, "
                   f"twist per degree {float((rows['move_angle_deg'] * rows['forearm_change_deg']).sum() / (rows['move_angle_deg'] ** 2).sum()):+.4f}")
    report.say("")

    report.say("3. Robot runs: the model's direction of travel in the picture (the tip moved 1 cm that way)")
    report.say(header)
    figure, axes = pyplot.subplots(5, 2, figsize = (18, 20))
    for position, run in enumerate(ANALYSIS_PARAMETERS["robot_runs"]):
        steps = robot_run_steps(run)
        done = steps[steps["action"] == "model_step"]
        report.say(spread_line(f"run {run['number']} {run['model']} {run['sheet']}", done["model_angle_deg"].to_numpy(), cones) + f"   {run['outcome']}")
        axis = axes.flat[position]
        axis.plot(steps["step"], steps["model_angle_deg"], color = "tab:green", marker = ".", label = "model's travel direction in the picture, degrees")
        axis.plot(steps["step"], 10.0 * steps["model_twist_deg"], color = "tab:blue", linewidth = 1, label = "model's twist x 10, degrees")
        axis.plot(steps["step"], 100.0 * steps["asked_cm"], color = "tab:orange", linewidth = 1, label = "model's step length, mm x 10")
        axis.axhline(0.0, color = "0.5", linewidth = 0.5)
        for cone in (-45, 45):
            axis.axhline(cone, color = "0.8", linewidth = 0.5, linestyle = ":")
        axis.set_ylim(-185, 185)
        axis.set_title(f"run {run['number']} ({run['model']}, {run['sheet']}): {run['outcome']}", fontsize = 9)
        axis.legend(fontsize = 6, loc = "lower left", ncol = 3)
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "03_model_angle_over_runs.png", dpi = 100)
    pyplot.close(figure)
    report.say("   demonstrations for comparison: see part 1 ('all steps with a direction' and per sheet)")
    report.say("")
    report.say("   the same in blocks of 10 pictures: median |direction|, share beyond 45 degrees, mean asked step length (cm), sum of the twist done (degrees)")
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        steps = robot_run_steps(run)
        done = steps[steps["action"] == "model_step"]
        parts = []
        for start in range(0, len(done), 10):
            block = done.iloc[start:start + 10]
            parts.append(f"[{start:3d}-{start + len(block) - 1:3d}: {block['model_angle_deg'].abs().median():5.1f} {(block['model_angle_deg'].abs() > 45).mean():4.2f} {block['asked_cm'].mean():4.2f} {block['twist_done_deg'].sum():+6.1f}]")
        report.say(f"   run {run['number']:2d} {run['sheet']} " + " ".join(parts))

    # histograms: demonstrations against the robot runs
    figure, axes = pyplot.subplots(2, 3, figsize = (18, 9))
    bins = numpy.arange(-180, 181, 10)
    for position, sheet in enumerate(ANALYSIS_PARAMETERS["sheets"]):
        axis = axes.flat[position]
        rows = moving[moving["sheet"] == sheet]
        axis.hist(rows["move_angle_deg"], bins = bins, density = True, color = "0.7", label = f"demonstrations ({len(rows)} steps)")
        for run in ANALYSIS_PARAMETERS["robot_runs"]:
            if run["sheet"] != sheet:
                continue
            steps = robot_run_steps(run)
            done = steps[steps["action"] == "model_step"]
            axis.hist(done["model_angle_deg"], bins = bins, density = True, histtype = "step", linewidth = 1.5, label = f"run {run['number']} ({run['model']}): {run['outcome']}")
        axis.set_title(sheet)
        axis.set_xlabel("direction of travel in the picture, degrees (0 = top, + = left)")
        axis.legend(fontsize = 7)
    figure.suptitle("Direction of travel in the picture frame: demonstrations (grey) against the robot runs of 11.11.26")
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "03_picture_direction_histograms.png", dpi = 100)
    report.save()



if __name__ == "__main__":
    main()
