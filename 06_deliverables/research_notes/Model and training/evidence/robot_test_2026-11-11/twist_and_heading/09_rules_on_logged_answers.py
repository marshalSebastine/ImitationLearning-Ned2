"""
What each driver rule would have commanded on the pictures the robot really saw on 11.11.26 (first order only: the
logged answers are kept, although after the first different step the robot would have seen other pictures).

Parts:
    1  per run: tip travel and twist per decision under each rule, and the twist per cm that results, against the
       demonstrations' twist per cm on that sheet;
    2  the first bend of sheet 03 (runs 5, 9, 10; the decisions before the tip left the demonstrated path): the same,
       against what the demonstrations did over the same stretch of path;
    3  how often the model asks for a step that neither moves nor turns (the risk of standing still when the driver
       follows the asked length), on the robot and on the held-out sheets.

Output: 09_rules_on_logged_answers.out.txt
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 09_rules_on_logged_answers.py
"""

from typing import Dict, Optional

import importlib

import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames, robot_run_steps

LOGGED_ANSWER_PARAMETERS = {
    # the decisions of the sheet-03 runs before the tip left the demonstrated path (02_...out.txt: steps 19, 20 and,
    # for run 10, the first stretch on the path up to the bend)
    "first_bend_decisions": {5: 13, 9: 13, 10: 13},
    "first_bend_path_cm": 10.0,
    "weak_length_cm": 0.3,
    "weak_twist_deg": 1.5
}



def main(parameters: Optional[Dict] = None) -> None:
    """Apply each driver rule to the answers logged on the robot and print the travel and twist that would have resulted."""

    settings = dict(LOGGED_ANSWER_PARAMETERS)
    settings.update(parameters or {})
    report = Report("09_rules_on_logged_answers.out.txt")
    rules_module = importlib.import_module("07_driver_rules")
    rule_settings = dict(rules_module.DRIVER_RULE_PARAMETERS)
    rules = rules_module.rule_table(rule_settings)
    frames = demonstration_frames()
    moving = frames[frames["action"] != "stop"]
    demonstrated_per_cm = {sheet: rows["forearm_change_deg"].abs().sum() / rows["step_cm"].sum() for sheet, rows in moving.groupby("sheet")}

    report.say("DRIVER RULES APPLIED TO THE ANSWERS THE MODELS GAVE ON THE ROBOT (calculated from steps.csv; first order: the pictures would change after the first different step)")
    report.say("")
    report.say("1. Per run: mean tip travel per decision (cm), mean |twist| per decision (degrees), twist per cm, and that as a share of the demonstrations' pooled twist per cm on the sheet")
    names = list(rules)
    report.say(f"   {'run':>3s} {'sheet':8s} {'demo deg/cm':>11s}  " + "  ".join(f"{name[:26]:>28s}" for name in names))
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        steps = robot_run_steps(run)
        done = steps[steps["action"] == "model_step"]
        parts = []
        for name in names:
            commanded = numpy.array([rules[name](length, twist) for length, twist in zip(done["asked_cm"], done["model_twist_deg"])])
            per_cm = numpy.abs(commanded[:, 1]).sum() / commanded[:, 0].sum()
            parts.append(f"{commanded[:, 0].mean():4.2f} {numpy.abs(commanded[:, 1]).mean():4.1f} {per_cm:5.1f} {per_cm / demonstrated_per_cm[run['sheet']]:5.2f}")
        report.say(f"   {run['number']:3d} {run['sheet']:8s} {demonstrated_per_cm[run['sheet']]:11.2f}  " + "  ".join(f"{part:>28s}" for part in parts))
    report.say("   each cell: travel per decision, |twist| per decision, twist per cm, share of the demonstrations' twist per cm")
    report.say("")

    report.say("2. The first bend of sheet 03, where runs 5, 9 and 10 failed")
    windows = pandas.read_csv(EVIDENCE_FOLDER / "01_twist_per_window.csv")
    reference = windows[(windows["source"] == "demonstration") & (windows["sheet"] == "path_03") & (windows["start_kind"] == "STD") & (windows["start_end"] == "A") & (windows["window"] <= 1)]
    per_episode = reference.groupby("episode_id").agg(pictures = ("steps", "sum"), turn = ("net_twist_deg", "sum"), travel = ("travel_cm", "sum"))
    report.say(f"   demonstrations (standard starts from end A, {len(per_episode)} episodes), first {settings['first_bend_path_cm']:.0f} cm of path: median {per_episode['pictures'].median():.0f} pictures, "
               f"net joint-4 change {per_episode['turn'].median():+.0f} degrees (range {per_episode['turn'].min():+.0f} to {per_episode['turn'].max():+.0f}), "
               f"that is {per_episode['travel'].median() / per_episode['pictures'].median():.2f} cm and {per_episode['turn'].median() / per_episode['pictures'].median():+.1f} degrees per picture, "
               f"{per_episode['turn'].median() / per_episode['travel'].median():+.1f} degrees per cm")
    report.say(f"   {'run':>3s} {'decisions':>9s}  " + "  ".join(f"{name[:26]:>28s}" for name in names))
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        if run["number"] not in settings["first_bend_decisions"]:
            continue
        steps = robot_run_steps(run)
        first = steps.iloc[:settings["first_bend_decisions"][run["number"]]]
        parts = []
        for name in names:
            commanded = numpy.array([rules[name](length, twist) for length, twist in zip(first["asked_cm"], first["model_twist_deg"])])
            parts.append(f"{commanded[:, 0].sum():5.1f} cm {commanded[:, 1].sum():+6.1f} deg {commanded[:, 1].sum() / commanded[:, 0].sum():+5.1f}/cm")
        report.say(f"   {run['number']:3d} {len(first):9d}  " + "  ".join(f"{part:>28s}" for part in parts))
    report.say("   each cell: the tip travel, the net twist and the twist per cm that the rule would have made of the first decisions of the run")
    report.say("")

    report.say(f"3. Answers that neither move nor turn (asked length under {settings['weak_length_cm']} cm AND |twist| under {settings['weak_twist_deg']} degrees): the driver that follows the asked length would nearly stand still on them")
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        steps = robot_run_steps(run)
        done = steps[steps["action"] == "model_step"]
        is_weak = (done["asked_cm"] < settings["weak_length_cm"]) & (done["model_twist_deg"].abs() < settings["weak_twist_deg"])
        longest, current = 0, 0
        for value in is_weak:
            current = current + 1 if value else 0
            longest = max(longest, current)
        size = numpy.maximum(done["asked_cm"] / rule_settings["unit_step_cm"], done["model_twist_deg"].abs() / rule_settings["unit_step_deg"])
        report.say(f"   run {run['number']:2d} {run['sheet']}: {int(is_weak.sum()):3d} of {len(done):3d} decisions ({is_weak.mean():.2f}), longest row of them {longest:2d}; "
                   f"asked step size in the recording's measure: median {size.median():.2f}, share under 0.5: {(size < 0.5).mean():.2f}; {run['outcome']}")
    for model in ("E", "A"):
        saved = numpy.load(EVIDENCE_FOLDER / f"fold_steps_{model}.npz")
        answers = pandas.DataFrame({"frame_index": saved["frame_index"], "asked_cm": numpy.hypot(saved["forward_cm"], saved["left_cm"]), "asked_twist_deg": saved["twist_deg"],
                                    "stop_probability": saved["stop_probability"]})
        table = moving.merge(answers, on = "frame_index")
        is_weak = (table["asked_cm"] < settings["weak_length_cm"]) & (table["asked_twist_deg"].abs() < settings["weak_twist_deg"])
        far = table["goal_distance_cm"] > 3.0
        report.say(f"   fold models {model} on held-out sheets (the demonstrations' own pictures, non-stop rows): {is_weak.mean():.3f} of the answers; "
                   f"more than 3 cm before the goal: {is_weak[far].mean():.3f}; within 3 cm of the goal: {is_weak[~far].mean():.3f}")
        for sheet, rows in table.groupby("sheet"):
            weak_sheet = (rows["asked_cm"] < settings["weak_length_cm"]) & (rows["asked_twist_deg"].abs() < settings["weak_twist_deg"])
            report.say(f"      {sheet}: {weak_sheet.mean():.3f} ({int(weak_sheet.sum())} of {len(rows)})")
    report.save()



if __name__ == "__main__":
    main()
