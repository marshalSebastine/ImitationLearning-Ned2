"""
Verifier: a crude test of the causal story on all ten runs at once. If "the driver moved much further than the model
asked while the model asked for a big twist" pushes the puck off the line, then decisions of that kind, taken while the
puck was still on the line, should be followed by the puck leaving the centre more often than other decisions.

Uses the analyst's per-decision table 01_asked_against_done_steps.csv for the puck's distance outside the ink (their
measure from the pictures; I checked it by eye on contact sheets of runs 1, 3, 4, 5, 6, 9 and 10) and the logged answers.
Association only: ten runs, decisions within a run are not independent.

Run from this folder:  python v13_association.py > v13_association.out.txt
"""

from pathlib import Path

import numpy
import pandas

TABLE = Path(__file__).resolve().parent.parent / "01_asked_against_done_steps.csv"
pandas.set_option("display.width", 250)
table = pandas.read_csv(TABLE)
table = table[table["action"] == "model_step"].copy()
AHEAD = 4

rows = []
for run, group in table.groupby("run"):
    outside = group["puck_outside_ink_cm"].to_numpy()
    asked = group["asked_cm"].to_numpy()
    twist = numpy.abs(group["asked_twist_deg"].to_numpy())
    for index in range(len(group) - AHEAD):
        if not numpy.isfinite(outside[index]) or outside[index] > 0.3:
            continue
        later = outside[index + 1:index + 1 + AHEAD]
        rows.append({"run": run, "step": int(group["step"].iloc[index]), "asked_cm": asked[index], "twist": twist[index],
                     "leaves_centre": bool(numpy.nanmax(later) >= 0.6), "leaves_line": bool(numpy.nanmax(later) > 1.2), "largest_later": float(numpy.nanmax(later))})
decisions = pandas.DataFrame(rows)
print(f"decisions taken while the puck's centre was at most 0.3 cm outside the ink, with {AHEAD} later decisions logged: {len(decisions)} in {decisions['run'].nunique()} runs")
decisions["kind"] = "other"
decisions.loc[(decisions["asked_cm"] < 0.5) & (decisions["twist"] >= 4.0), "kind"] = "asked under 0.5 cm with a twist of 4 degrees or more"
decisions.loc[(decisions["asked_cm"] < 0.5) & (decisions["twist"] < 4.0), "kind"] = "asked under 0.5 cm, twist under 4 degrees"
decisions.loc[(decisions["asked_cm"] >= 0.5) & (decisions["twist"] >= 4.0), "kind"] = "asked 0.5 cm or more with a twist of 4 degrees or more"
summary = decisions.groupby("kind").agg(decisions = ("run", "size"), runs = ("run", "nunique"), leaves_centre_share = ("leaves_centre", "mean"), leaves_line_share = ("leaves_line", "mean"),
                                        largest_later_median_cm = ("largest_later", "median"))
print(summary.round(2).to_string())
print()
print("per run, the decisions of the kind 'asked under 0.5 cm with a twist of 4 degrees or more' taken on the line:")
kind = decisions[decisions["kind"] == "asked under 0.5 cm with a twist of 4 degrees or more"]
print(kind.groupby("run").agg(decisions = ("step", "size"), first = ("step", "min"), last = ("step", "max"), leaves_centre_share = ("leaves_centre", "mean"),
                              leaves_line_share = ("leaves_line", "mean"), largest_later_cm = ("largest_later", "max")).round(2).to_string())
print()
print("how far the driver moved against the asked length, by what followed (decisions on the line):")
decisions["done_over_asked"] = 1.0 / decisions["asked_cm"].clip(lower = 0.05)
print(decisions.groupby("leaves_centre").agg(decisions = ("run", "size"), asked_median_cm = ("asked_cm", "median"), twist_median_deg = ("twist", "median"),
                                             done_over_asked_median = ("done_over_asked", "median")).round(2).to_string())
