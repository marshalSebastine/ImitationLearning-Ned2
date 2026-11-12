"""
Twist per cm travelled: the ten robot runs of 11.11.26 against the demonstrations, per sheet and along the path.

Three ways of counting, because they answer different questions:
    absolute    the sum of |joint-4 change| per cm of tip path (the brief's scout number): it counts back-and-forth
                twisting as well as the turning the bend needs;
    net         |joint-4 change from the start to the end of a 5-cm stretch| per cm: only the turning that stays;
    asked       for the robot, what the MODEL asked for: the sum of |predicted twist| per cm of ASKED tip path (the
                predicted step lengths), against what the driver did with its fixed 1-cm step.

Output: 01_twist_per_cm.out.txt, 01_twist_along_path.png, 01_twist_per_window.csv
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 01_twist_per_cm.py
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames, robot_run_steps

TWIST_PER_CM_PARAMETERS = {
    "window_cm": 5.0,
    # a demonstration is compared along the path with a robot run only when it is a standard start (STD) on the same
    # sheet from the same end as the plan Emranul named
    "reference_start_kind": "STD",
    "smallest_window_share": 0.6
}



def windowed_twist(
        arc_cm: numpy.ndarray,
        step_cm: numpy.ndarray,
        twist_deg: numpy.ndarray,
        window_cm: float,
        smallest_share: float
) -> pandas.DataFrame:
    """
    Split a path into stretches of ``window_cm`` of arc length and sum the twist in each.

    Parameters
    ----------
    arc_cm : numpy.ndarray
        The path length at the START of each step.
    step_cm : numpy.ndarray
        The tip travel of each step.
    twist_deg : numpy.ndarray
        The joint-4 change of each step.
    window_cm : float
        The length of a stretch.
    smallest_share : float
        A last stretch shorter than this share of ``window_cm`` is dropped.

    Returns
    -------
    pandas.DataFrame
        One row per stretch: "window", "steps", "travel_cm", "absolute_twist_deg", "net_twist_deg",
        "absolute_per_cm", "net_per_cm" (signed).
    """

    window = numpy.floor(arc_cm / window_cm).astype(int)
    rows = []
    for number in numpy.unique(window):
        chosen = window == number
        travel = float(step_cm[chosen].sum())
        if travel < smallest_share * window_cm:
            continue
        rows.append({"window": int(number), "steps": int(chosen.sum()), "travel_cm": travel,
                     "absolute_twist_deg": float(numpy.abs(twist_deg[chosen]).sum()), "net_twist_deg": float(twist_deg[chosen].sum()),
                     "absolute_per_cm": float(numpy.abs(twist_deg[chosen]).sum() / travel), "net_per_cm": float(twist_deg[chosen].sum() / travel)})
    return pandas.DataFrame(rows)



def main(parameters: Optional[Dict] = None) -> None:
    """Print the twist per cm of the demonstrations and of the robot runs, per sheet and per 5 cm of path, and draw the figure."""

    settings = dict(TWIST_PER_CM_PARAMETERS)
    settings.update(parameters or {})
    report = Report("01_twist_per_cm.out.txt")
    frames = demonstration_frames()
    moving = frames[frames["action"] != "stop"]

    report.say("TWIST PER CM TRAVELLED (all numbers measured from frames.csv with the audit cleaning and from the runs' steps.csv)")
    report.say("")
    report.say("1. The demonstrations, per sheet (one value per episode, then the median over episodes; 'pooled' = all twist / all travel)")
    report.say(f"{'sheet':8s} {'episodes':>8s} {'abs deg/cm median':>18s} {'q25':>6s} {'q75':>6s} {'pooled':>7s} {'net deg/cm (5 cm) median':>25s} {'net/abs share':>14s} {'median step cm':>15s}")
    demonstration_windows = []
    per_sheet = {}
    for sheet, sheet_rows in moving.groupby("sheet"):
        per_episode, net_values, absolute_values = [], [], []
        for episode_id, rows in sheet_rows.groupby("episode_id"):
            travel = rows["step_cm"].sum()
            if travel < 1.0:
                continue
            per_episode.append(rows["forearm_change_deg"].abs().sum() / travel)
            # the path length at the start of each step, from the labelled step lengths (so that a twist in place stays in its stretch)
            arc = numpy.r_[0.0, numpy.cumsum(rows["step_cm"].to_numpy())[:-1]]
            windows = windowed_twist(arc, rows["step_cm"].to_numpy(), rows["forearm_change_deg"].fillna(0.0).to_numpy(), settings["window_cm"], settings["smallest_window_share"])
            windows["episode_id"] = episode_id
            windows["sheet"] = sheet
            windows["start_kind"] = rows["start_kind"].iloc[0]
            windows["start_end"] = rows["start_end"].iloc[0]
            demonstration_windows.append(windows)
            net_values.extend(windows["net_per_cm"].abs().tolist())
            absolute_values.extend(windows["absolute_per_cm"].tolist())
        pooled = sheet_rows["forearm_change_deg"].abs().sum() / sheet_rows["step_cm"].sum()
        per_sheet[sheet] = {"absolute_median": float(numpy.median(per_episode)), "pooled": float(pooled),
                            "net_median": float(numpy.median(net_values)), "window_absolute_median": float(numpy.median(absolute_values))}
        report.say(f"{sheet:8s} {len(per_episode):8d} {numpy.median(per_episode):18.2f} {numpy.quantile(per_episode, 0.25):6.2f} {numpy.quantile(per_episode, 0.75):6.2f} "
                   f"{pooled:7.2f} {numpy.median(net_values):25.2f} {numpy.sum(numpy.abs(net_values)) / numpy.sum(absolute_values):14.2f} {sheet_rows['step_cm'].median():15.2f}")
    demonstration_windows = pandas.concat(demonstration_windows, ignore_index = True)
    report.say("   'net/abs share': of all the twisting in 5-cm stretches, the share that is still there at the end of the stretch;")
    report.say("   the rest was turned back within the same 5 cm (jitter of the guiding hand and corrections).")
    report.say("")

    report.say("2. The robot runs (done = measured joint-4 change and tip path between pictures; asked = the model's own numbers)")
    report.say(f"{'run':>3s} {'model':>5s} {'sheet':8s} {'moves':>5s} {'path cm':>8s} {'abs done':>9s} {'net(5cm)':>9s} {'asked twist':>11s} {'asked path':>10s} {'asked deg/asked cm':>19s} "
               f"{'demo abs':>9s} {'demo net':>9s} {'done/demo abs':>13s} {'done/demo net':>13s} {'clamped':>7s}  outcome")
    run_windows = []
    summary_rows = []
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        steps = robot_run_steps(run)
        done = steps[steps["action"] == "model_step"]
        # only steps that were carried out (the last model_step row of a run ended by Esc may not have moved)
        done = done[done["moved_cm"] > 0.2]
        windows = windowed_twist(done["arc_cm"].to_numpy(), done["moved_cm"].to_numpy(), done["twist_done_deg"].to_numpy(), settings["window_cm"], settings["smallest_window_share"])
        windows["run"] = run["number"]
        windows["sheet"] = run["sheet"]
        run_windows.append(windows)
        path = done["moved_cm"].sum()
        absolute_done = done["twist_done_deg"].abs().sum() / path
        net_done = float(numpy.median(windows["net_per_cm"].abs()))
        asked_twist = done["model_twist_deg"].abs().sum()
        asked_path = done["asked_cm"].sum()
        reference = per_sheet[run["sheet"]]
        clamped = int(done["is_turn_clamped"].sum()) + int((done["model_twist_deg"].abs() > 10.0).sum())
        summary_rows.append({"run": run["number"], "model": run["model"], "sheet": run["sheet"], "moves": len(done), "path_cm": path,
                             "absolute_done_per_cm": absolute_done, "net_done_per_cm_median": net_done, "asked_twist_deg": asked_twist,
                             "asked_path_cm": asked_path, "asked_per_asked_cm": asked_twist / asked_path,
                             "demonstration_absolute": reference["absolute_median"], "demonstration_net": reference["net_median"]})
        report.say(f"{run['number']:3d} {run['model']:>5s} {run['sheet']:8s} {len(done):5d} {path:8.1f} {absolute_done:9.2f} {net_done:9.2f} {asked_twist:11.1f} {asked_path:10.1f} "
                   f"{asked_twist / asked_path:19.2f} {reference['absolute_median']:9.2f} {reference['net_median']:9.2f} {absolute_done / reference['absolute_median']:13.2f} "
                   f"{net_done / reference['net_median']:13.2f} {clamped:7d}  {run['outcome']}")
    run_windows = pandas.concat(run_windows, ignore_index = True)
    report.say("   'abs done': sum |joint-4 change| / path, degrees per cm. 'net(5cm)': median over 5-cm stretches of |net joint-4 change| / 5 cm.")
    report.say("   'asked deg/asked cm': sum |predicted twist| / sum of predicted step lengths: the twist per cm the MODEL asked for.")
    report.say("   'clamped': steps where the twist met the 10-degree step limit or the 117-degree joint limit.")
    report.say("")

    report.say("3. Along the path, per 5 cm: the robot run against standard-start demonstrations of the same sheet from the same end")
    report.say("   (signed net joint-4 change in each 5-cm stretch, degrees; demonstrations: median [smallest, largest] over the episodes)")
    figure, axes = pyplot.subplots(2, 4, figsize = (20, 8))
    plotted = 0
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        if run["plan"] is None:
            continue
        plan_rows = frames[frames["plan_number"] == run["plan"]]
        start_end = plan_rows["start_end"].iloc[0]
        reference = demonstration_windows[(demonstration_windows["sheet"] == run["sheet"]) & (demonstration_windows["start_kind"] == settings["reference_start_kind"])
                                          & (demonstration_windows["start_end"] == start_end)]
        own = run_windows[run_windows["run"] == run["number"]]
        report.say(f"   run {run['number']} ({run['model']}, {run['sheet']}, as plan {run['plan']}, end {start_end}; {reference['episode_id'].nunique()} reference demonstrations; {run['outcome']})")
        report.say(f"   {'cm':>7s} {'robot net':>10s} {'robot abs':>10s} {'demo net median':>16s} {'demo net range':>18s} {'demo abs median':>16s} {'robot/demo net':>15s}")
        axis = axes.flat[plotted]
        plotted += 1
        centres, robot_net, demonstration_median = [], [], []
        for _, window in own.iterrows():
            same = reference[reference["window"] == window["window"]]
            if len(same) == 0:
                report.say(f"   {int(window['window'] * settings['window_cm']):3d}-{int((window['window'] + 1) * settings['window_cm']):3d} {window['net_twist_deg']:10.1f} {window['absolute_twist_deg']:10.1f} {'(no demonstration this far)':>16s}")
                continue
            median = same["net_twist_deg"].median()
            ratio = window["net_twist_deg"] / median if abs(median) >= 5.0 else float("nan")
            report.say(f"   {int(window['window'] * settings['window_cm']):3d}-{int((window['window'] + 1) * settings['window_cm']):3d} {window['net_twist_deg']:10.1f} {window['absolute_twist_deg']:10.1f} "
                       f"{median:16.1f} {'[' + format(same['net_twist_deg'].min(), '.0f') + ', ' + format(same['net_twist_deg'].max(), '.0f') + ']':>18s} "
                       f"{same['absolute_twist_deg'].median():16.1f} {ratio:15.2f}")
            centres.append((window["window"] + 0.5) * settings["window_cm"])
            robot_net.append(window["net_twist_deg"])
            demonstration_median.append(median)
        for episode_id, episode_windows in reference.groupby("episode_id"):
            axis.plot((episode_windows["window"] + 0.5) * settings["window_cm"], episode_windows["net_twist_deg"], color = "0.75", linewidth = 0.8)
        axis.plot(centres, demonstration_median, color = "black", linewidth = 2, label = "demonstrations, median")
        axis.plot((own["window"] + 0.5) * settings["window_cm"], own["net_twist_deg"], color = "tab:red", marker = "o", linewidth = 2, label = "robot run")
        axis.axhline(0.0, color = "0.5", linewidth = 0.5)
        axis.set_title(f"run {run['number']} ({run['model']}, {run['sheet']}): {run['outcome']}", fontsize = 9)
        axis.set_xlabel("path length of the tip, cm")
        axis.set_ylabel("net joint-4 change per 5 cm, degrees")
        axis.legend(fontsize = 7)
        report.say("")
    for axis in axes.flat[plotted:]:
        axis.axis("off")
    figure.suptitle("Forearm twist along the path, per 5 cm: robot runs of 11.11.26 (red) against standard-start demonstrations of the same sheet and end (grey, median black)")
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "01_twist_along_path.png", dpi = 110)

    pandas.concat([demonstration_windows.assign(source = "demonstration"), run_windows.assign(source = "robot")], ignore_index = True).to_csv(EVIDENCE_FOLDER / "01_twist_per_window.csv", index = False)
    pandas.DataFrame(summary_rows).to_csv(EVIDENCE_FOLDER / "01_twist_per_run.csv", index = False)
    report.save()



if __name__ == "__main__":
    main()
