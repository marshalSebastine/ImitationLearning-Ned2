"""
Verifier 15: the "aim of the camera" numbers, recomputed from the analyst's cache of puck positions
(puck_positions_demonstrations.csv and puck_positions_runs.csv: the team's puck tracker; I did not rerun the tracker).
Also: is "the tip is not touching the puck in 56 % of the demonstration pictures" real, or does the distance from the
tip to the puck simply grow with the lean of the rod (the arm model's tip is the bottom of the magnet block)?
"""
from pathlib import Path
import numpy
import pandas
from vcommon import demos, run_log, RUNS, wrap

EVIDENCE = Path(__file__).resolve().parent.parent
frames = demos()
puck = pandas.read_csv(EVIDENCE / "puck_positions_demonstrations.csv")
table = frames.merge(puck, on = "frame_index", how = "inner")
print("demonstration pictures", len(table), "; puck placed on the table in %.3f" % table["puck_x_cm"].notna().mean())
seen = table[table["puck_x_cm"].notna()].copy()
seen["distance"] = numpy.hypot(seen["puck_x_cm"] - seen["tip_x_cm"], seen["puck_y_cm"] - seen["tip_y_cm"])
seen["aim"] = wrap(numpy.degrees(numpy.arctan2(seen["puck_y_cm"] - seen["tip_y_cm"], seen["puck_x_cm"] - seen["tip_x_cm"])) - seen["picture_heading_deg"])
print("tip to puck centre: median %.2f cm, quartiles %.2f to %.2f; within 2.8 cm: %.2f; further than 4 cm: %.2f" % (seen["distance"].median(), seen["distance"].quantile(0.25), seen["distance"].quantile(0.75), (seen["distance"] <= 2.8).mean(), (seen["distance"] > 4).mean()))
print("   touching distance by the config: puck radius 1.2 cm + tip radius 0.65 cm = 1.85 cm")
print("   rod lean (rod_from_vertical_deg): median %.1f, 5 to 95 %%: %.1f to %.1f; correlation of the tip-to-puck distance with the lean: %+.2f" % (
    seen["rod_from_vertical_deg"].median(), seen["rod_from_vertical_deg"].quantile(0.05), seen["rod_from_vertical_deg"].quantile(0.95), numpy.corrcoef(seen["distance"], seen["rod_from_vertical_deg"])[0, 1]))
for low, high in ((0, 15), (15, 25), (25, 35), (35, 60)):
    part = seen[(seen["rod_from_vertical_deg"] >= low) & (seen["rod_from_vertical_deg"] < high)]
    print("   lean %2d to %2d degrees: %5d pictures, tip to puck median %.2f cm, within 2.8 cm in %.2f" % (low, high, len(part), part["distance"].median(), (part["distance"] <= 2.8).mean()))
# the first rows of standard starts: the tip is set behind the puck before the first push; what distance does the tracker give there, and after 5 pushing steps on the straight sheet?
straight = seen[(seen["sheet"] == "path_01")]
print("   sheet 01 (straight pushing): tip to puck median %.2f cm; lean median %.1f" % (straight["distance"].median(), straight["rod_from_vertical_deg"].median()))
touching = seen[seen["distance"] <= 2.8]
print("aim, all pictures: median %+.1f; touching (within 2.8 cm): median %+.1f, quartiles %+.1f to %+.1f; within 20 degrees of the median: %.2f; within 30: %.2f; within 45: %.2f" % (
    seen["aim"].median(), touching["aim"].median(), touching["aim"].quantile(0.25), touching["aim"].quantile(0.75),
    (wrap(touching["aim"] - touching["aim"].median()).__abs__() <= 20).mean(), (numpy.abs(wrap(touching["aim"] - touching["aim"].median())) <= 30).mean(), (numpy.abs(wrap(touching["aim"] - touching["aim"].median())) <= 45).mean()))
loose = seen[seen["distance"] <= 4.0]
print("aim with a looser 'at the puck' (within 4 cm, %.2f of the pictures): median %+.1f, quartiles %+.1f to %+.1f; within 20 degrees of the median: %.2f; within 45: %.2f" % (
    len(loose) / len(seen), loose["aim"].median(), loose["aim"].quantile(0.25), loose["aim"].quantile(0.75), (numpy.abs(wrap(loose["aim"] - loose["aim"].median())) <= 20).mean(), (numpy.abs(wrap(loose["aim"] - loose["aim"].median())) <= 45).mean()))

moving = touching[touching["action"] != "stop"]
near = moving[numpy.abs(wrap(moving["aim"] - touching["aim"].median())) <= 45]
slope, offset = numpy.polyfit(near["aim"], near["twist"], 1)
print("twist as a correction of the aim, demonstrations (touching, within 45 degrees of the median aim): rows %d, slope %+.3f degrees of twist per degree of aim, correlation %+.2f" % (len(near), slope, numpy.corrcoef(near["aim"], near["twist"])[0, 1]))

runs = pandas.read_csv(EVIDENCE / "puck_positions_runs.csv")
rows = []
for number in RUNS:
    log = run_log(number)
    part = runs[runs["run"] == number].merge(log[["step", "picture_heading_deg", "model_twist_deg", "action"]], on = "step")
    part["aim"] = wrap(numpy.degrees(numpy.arctan2(part["puck_y_cm"] - part["tip_y_cm"], part["puck_x_cm"] - part["tip_x_cm"])) - part["picture_heading_deg"])
    part["distance"] = numpy.hypot(part["puck_x_cm"] - part["tip_x_cm"], part["puck_y_cm"] - part["tip_y_cm"])
    rows.append(part.assign(run = number))
    outside = ((part["aim"] < -28.3) | (part["aim"] > 39.6)).mean()
    print("   run %2d: aim median %+.1f; outside -28.3 to +39.6 in %.2f of the pictures; tip to puck median %.1f cm" % (number, part["aim"].median(), outside, part["distance"].median()))
robot = pandas.concat(rows)
robot = robot[robot["action"] == "model_step"]
near = robot[numpy.abs(wrap(robot["aim"] - touching["aim"].median())) <= 45]
slope_robot = numpy.polyfit(near["aim"], near["model_twist_deg"], 1)[0]
print("robot, all ten runs: rows %d, slope %+.3f per decision, correlation %+.2f; that is %.2f of the demonstrated slope" % (len(near), slope_robot, numpy.corrcoef(near["aim"], near["model_twist_deg"])[0, 1], slope_robot / slope))
