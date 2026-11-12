"""
Can the model tell from one picture whether the demonstrator re-aimed (a short tip move with a twist of 5 degrees or
more) or pushed on (about 1 cm of tip travel)? If it cannot, its answer is a blend of both, shorter than a push and
with less twist than a re-aim, and the forearm turns too little per cm whatever the driver does.

Measured with the fold models' whole answers on held-out sheets (fold_steps_E.npz, fold_steps_A.npz).

Output: 11_reaim_or_push.out.txt, 11_reaim_or_push.png
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 11_reaim_or_push.py
"""

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames

REAIM_PARAMETERS = {
    "short_step_cm": 0.7,
    "reaim_twist_deg": 3.0
}



def ranking_score(scores: numpy.ndarray, is_positive: numpy.ndarray) -> float:
    """Return the area under the ROC curve: the chance that a positive row has the higher score than a negative one."""

    order = numpy.argsort(scores)
    ranks = numpy.empty(len(scores))
    ranks[order] = numpy.arange(1, len(scores) + 1)
    positives = int(is_positive.sum())
    negatives = len(scores) - positives
    return float((ranks[is_positive].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))



def main(parameters: Optional[Dict] = None) -> None:
    """Print and draw what the fold models ask for on demonstrated re-aim steps and push steps."""

    settings = dict(REAIM_PARAMETERS)
    settings.update(parameters or {})
    report = Report("11_reaim_or_push.out.txt")
    frames = demonstration_frames()
    moving = frames[frames["action"] != "stop"]
    report.say("RE-AIM OR PUSH ON: what the fold models answer on the two kinds of demonstrated step (held-out sheets; measured)")
    report.say(f"re-aim = tip travel under {settings['short_step_cm']} cm with a twist of {settings['reaim_twist_deg']:.0f} degrees or more; push = tip travel of {settings['short_step_cm']} cm or more")
    report.say("")
    figure, axes = pyplot.subplots(1, 2, figsize = (14, 6))
    for position, model in enumerate(("E", "A")):
        saved = numpy.load(EVIDENCE_FOLDER / f"fold_steps_{model}.npz")
        answers = pandas.DataFrame({"frame_index": saved["frame_index"], "asked_cm": numpy.hypot(saved["forward_cm"], saved["left_cm"]), "asked_twist_deg": saved["twist_deg"]})
        table = moving.merge(answers, on = "frame_index")
        is_reaim = ((table["step_cm"] < settings["short_step_cm"]) & (table["forearm_change_deg"].abs() >= settings["reaim_twist_deg"])).to_numpy()
        is_push = (table["step_cm"] >= settings["short_step_cm"]).to_numpy()
        table = table[is_reaim | is_push]
        is_reaim = is_reaim[is_reaim | is_push]
        report.say(f"MODEL {model}: {int(is_reaim.sum())} re-aim steps, {int((~is_reaim).sum())} push steps")
        report.say(f"   {'':22s} {'demonstrated cm':>15s} {'asked cm':>9s} {'demonstrated |twist|':>20s} {'asked |twist|':>13s} {'demonstrated deg per cm':>23s} {'asked deg per cm':>16s}")
        for name, chosen in (("re-aim steps", table[is_reaim]), ("push steps", table[~is_reaim])):
            report.say(f"   {name:22s} {chosen['step_cm'].median():15.2f} {chosen['asked_cm'].median():9.2f} {chosen['forearm_change_deg'].abs().median():20.2f} {chosen['asked_twist_deg'].abs().median():13.2f} "
                       f"{chosen['forearm_change_deg'].abs().sum() / chosen['step_cm'].sum():23.1f} {chosen['asked_twist_deg'].abs().sum() / chosen['asked_cm'].sum():16.1f}")
        ratio = table["asked_twist_deg"].abs() / numpy.maximum(table["asked_cm"], 0.05)
        report.say(f"   how well the answer tells the two apart (1.0 = always, 0.5 = not at all): by the asked length {ranking_score(-table['asked_cm'].to_numpy(), is_reaim):.2f}, "
                   f"by the asked |twist| {ranking_score(table['asked_twist_deg'].abs().to_numpy(), is_reaim):.2f}, by the asked twist per cm {ranking_score(ratio.to_numpy(), is_reaim):.2f}")
        for sheet, rows in table.groupby("sheet"):
            is_reaim_sheet = ((rows["step_cm"] < settings["short_step_cm"]) & (rows["forearm_change_deg"].abs() >= settings["reaim_twist_deg"])).to_numpy()
            sheet_ratio = rows["asked_twist_deg"].abs() / numpy.maximum(rows["asked_cm"], 0.05)
            report.say(f"      {sheet}: share of re-aim steps {is_reaim_sheet.mean():.2f}; told apart by the asked twist per cm {ranking_score(sheet_ratio.to_numpy(), is_reaim_sheet):.2f}; "
                       f"asked on re-aim steps {rows.loc[is_reaim_sheet, 'asked_cm'].median():.2f} cm and {rows.loc[is_reaim_sheet, 'asked_twist_deg'].abs().median():.1f} degrees, "
                       f"on push steps {rows.loc[~is_reaim_sheet, 'asked_cm'].median():.2f} cm and {rows.loc[~is_reaim_sheet, 'asked_twist_deg'].abs().median():.1f} degrees")
        report.say("")
        axis = axes[position]
        axis.scatter(table.loc[~is_reaim, "asked_cm"][::4], table.loc[~is_reaim, "asked_twist_deg"].abs()[::4], s = 3, alpha = 0.25, color = "tab:blue", label = "demonstrated push steps")
        axis.scatter(table.loc[is_reaim, "asked_cm"][::2], table.loc[is_reaim, "asked_twist_deg"].abs()[::2], s = 3, alpha = 0.35, color = "tab:red", label = "demonstrated re-aim steps")
        axis.scatter([0.40, 1.14], [6.4, 2.4], s = 120, marker = "*", color = "black", zorder = 5, label = "demonstrated medians (re-aim, push)")
        axis.plot([0, 1.0], [5.0, 5.0], color = "0.4", linewidth = 1, linestyle = "--")
        axis.plot([1.0, 1.0], [0.0, 5.0], color = "0.4", linewidth = 1, linestyle = "--", label = "a recorded step: 1 cm or 5 degrees")
        axis.set_xlim(0, 1.8)
        axis.set_ylim(0, 12)
        axis.set_xlabel("asked tip travel, cm")
        axis.set_ylabel("asked |twist|, degrees")
        axis.set_title(f"What fold model {model} asks for on held-out sheets")
        axis.legend(fontsize = 8)
        axis.grid(alpha = 0.3)
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "11_reaim_or_push.png", dpi = 100)
    report.save()



if __name__ == "__main__":
    main()
