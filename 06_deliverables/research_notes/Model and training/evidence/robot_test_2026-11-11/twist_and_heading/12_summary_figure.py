"""
One figure with the three numbers the plan needs from this folder:
    left    the first bend of sheet 03 (where runs 5, 9 and 10 failed): pictures or decisions per cm, twist per
            picture or decision, and twist per cm, demonstrations against the robot (measured);
    middle  the same bend: the twist per cm each driver rule would have made of the answers the models gave there
            (calculated from the logged answers), against the demonstrations';
    right   unseen sheets: the share of a demonstration's first big turn that each rule gives back in the simulation
            with the fold models (calculated).

It re-computes the left and middle panels from steps.csv and 01_twist_per_window.csv and reads the right panel's
numbers from 07_driver_rules.out.txt, so 01_twist_per_cm.py and 07_driver_rules.py must have been run first.

Output: 12_summary_figure.png
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 12_summary_figure.py
"""

import importlib
import re
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, robot_run_steps

SUMMARY_FIGURE_PARAMETERS = {
    "runs": [5, 9, 10],
    "decisions": 13,
    "shown_rules": ["now (1 cm, twist as predicted)", "gain 1.5", "gain 2.0", "length x 1.3", "length (0.2 to 1.5 cm)", "unit step (1 cm or 5 deg)", "half step (0.5 cm, twist as predicted)"],
    "short_names": ["now", "gain 1.5", "gain 2.0", "asked length\nx 1.3", "asked\nlength", "unit step", "half step\n0.5 cm"]
}



def main(parameters: Optional[Dict] = None) -> None:
    """Draw the summary figure from the logs, the window table and the simulation's printed output."""

    settings = dict(SUMMARY_FIGURE_PARAMETERS)
    settings.update(parameters or {})
    rules_module = importlib.import_module("07_driver_rules")
    rules = rules_module.rule_table(dict(rules_module.DRIVER_RULE_PARAMETERS))
    windows = pandas.read_csv(EVIDENCE_FOLDER / "01_twist_per_window.csv")
    reference = windows[(windows["source"] == "demonstration") & (windows["sheet"] == "path_03") & (windows["start_kind"] == "STD") & (windows["start_end"] == "A") & (windows["window"] <= 1)]
    per_episode = reference.groupby("episode_id").agg(pictures = ("steps", "sum"), turn = ("net_twist_deg", "sum"), travel = ("travel_cm", "sum"))
    demonstrated = {"per_cm": float(per_episode["pictures"].median() / per_episode["travel"].median()), "per_picture": float(abs(per_episode["turn"].median()) / per_episode["pictures"].median()),
                    "twist_per_cm": float(abs(per_episode["turn"].median()) / per_episode["travel"].median())}

    figure, axes = pyplot.subplots(1, 3, figsize = (21, 6.5))
    axis = axes[0]
    groups = ["pictures or decisions\nper cm of tip path", "twist per picture\nor decision, degrees", "twist per cm,\ndegrees"]
    width = 0.2
    axis.bar(numpy.arange(3) - 1.5 * width, [demonstrated["per_cm"], demonstrated["per_picture"], demonstrated["twist_per_cm"]], width, color = "black", label = f"demonstrations ({len(per_episode)}, first 10 cm)")
    by_rule = {name: [] for name in settings["shown_rules"]}
    for position, number in enumerate(settings["runs"]):
        run = [run for run in ANALYSIS_PARAMETERS["robot_runs"] if run["number"] == number][0]
        first = robot_run_steps(run).iloc[:settings["decisions"]]
        travel = first["moved_cm"].sum()
        turn = abs(first["twist_done_deg"].sum())
        axis.bar(numpy.arange(3) + (position - 0.5) * width, [len(first) / travel, turn / len(first), turn / travel], width, label = f"run {number} (model {run['model']}), first {settings['decisions']} decisions")
        for name in settings["shown_rules"]:
            commanded = numpy.array([rules[name](length, twist) for length, twist in zip(first["asked_cm"], first["model_twist_deg"])])
            by_rule[name].append(abs(commanded[:, 1].sum()) / commanded[:, 0].sum())
    axis.set_xticks(numpy.arange(3))
    axis.set_xticklabels(groups)
    axis.set_title("Measured: the first bend of sheet 03\nthe twist per decision was nearly the demonstrated one, the decisions per cm were not", fontsize = 10)
    axis.legend(fontsize = 8)
    axis.grid(alpha = 0.3, axis = "y")

    axis = axes[1]
    positions = numpy.arange(len(settings["shown_rules"]))
    for position, number in enumerate(settings["runs"]):
        axis.bar(positions + (position - 1) * 0.25, [by_rule[name][position] for name in settings["shown_rules"]], 0.25, label = f"answers of run {number}")
    axis.axhline(demonstrated["twist_per_cm"], color = "black", linewidth = 2, label = f"demonstrations: {demonstrated['twist_per_cm']:.1f} degrees per cm")
    axis.set_xticks(positions)
    axis.set_xticklabels(settings["short_names"], fontsize = 8)
    axis.set_ylabel("twist per cm of tip path, degrees")
    axis.set_title("Calculated from the logged answers: the same bend under each driver rule\n(first order: the pictures would change after the first different step)", fontsize = 10)
    axis.legend(fontsize = 8)
    axis.grid(alpha = 0.3, axis = "y")

    axis = axes[2]
    text = (EVIDENCE_FOLDER / "07_driver_rules.out.txt").read_text(encoding = "utf-8")
    shares = {"E": {}, "A": {}}
    model = None
    for line in text.splitlines():
        if line.startswith("MODEL "):
            model = line.split()[1]
        found = re.match(r"\s+(.+?)\s{2,}(\d\.\d\d) \(152 demos", line)
        if found and model:
            shares[model][found.group(1).strip()] = float(found.group(2))
    names = settings["shown_rules"] + ["per picture (reference)"]
    positions = numpy.arange(len(names))
    for position, model in enumerate(("E", "A")):
        axis.bar(positions + (position - 0.5) * 0.38, [shares[model].get(name, numpy.nan) for name in names], 0.38, label = f"fold models, recipe {model}")
    axis.axhline(1.0, color = "black", linewidth = 2, label = "the demonstration's turn")
    axis.set_xticks(positions)
    axis.set_xticklabels(settings["short_names"] + ["one decision\nper demo picture"], fontsize = 8)
    axis.set_ylabel("share of the demonstration's first big turn (median of 152)")
    axis.set_title("Calculated, unseen sheets: a simulated robot walked along held-out demonstrations\n(the model twists less there: even one decision per demonstration picture gives 0.6 to 0.7)", fontsize = 10)
    axis.legend(fontsize = 8)
    axis.grid(alpha = 0.3, axis = "y")
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "12_summary_figure.png", dpi = 100)
    print(f"demonstrations at the first bend of sheet 03: {demonstrated}")
    print({name: [round(value, 1) for value in values] for name, values in by_rule.items()})
    print(shares)



if __name__ == "__main__":
    main()
