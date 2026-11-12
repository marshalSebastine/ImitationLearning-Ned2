"""
Verifier 17: the robot's aim against the demonstrations' aim, like for like (cached puck positions of the analyst).
The analyst compared ALL robot pictures with only those demonstration pictures whose tip was within 2.8 cm of the puck
centre (44 % of them). Here: (a) robot pictures within 2.8 cm against the demonstrations' range within 2.8 cm;
(b) all robot pictures against the range of all demonstration pictures.
"""
from pathlib import Path
import numpy
import pandas
from vcommon import demos, run_log, RUNS, wrap

EVIDENCE = Path(__file__).resolve().parent.parent
frames = demos()
table = frames.merge(pandas.read_csv(EVIDENCE / "puck_positions_demonstrations.csv"), on = "frame_index", how = "inner")
table = table[table["puck_x_cm"].notna() & (table["action"] != "stop")].copy()
table["distance"] = numpy.hypot(table["puck_x_cm"] - table["tip_x_cm"], table["puck_y_cm"] - table["tip_y_cm"])
table["aim"] = wrap(numpy.degrees(numpy.arctan2(table["puck_y_cm"] - table["tip_y_cm"], table["puck_x_cm"] - table["tip_x_cm"])) - table["picture_heading_deg"])
near = table[table["distance"] <= 2.8]
range_near = (near["aim"].quantile(0.05), near["aim"].quantile(0.95))
range_all = (table["aim"].quantile(0.05), table["aim"].quantile(0.95))
print("demonstrations: 5 to 95 %% of the aim, tip within 2.8 cm: %+.1f to %+.1f (%d pictures); all pictures: %+.1f to %+.1f (%d)" % (*range_near, len(near), *range_all, len(table)))
print("demonstrations, all pictures: more than 45 degrees from the median aim in %.2f; more than 30 in %.2f; tip further than 2.8 cm in %.2f" % (
    (numpy.abs(wrap(table["aim"] - table["aim"].median())) > 45).mean(), (numpy.abs(wrap(table["aim"] - table["aim"].median())) > 30).mean(), (table["distance"] > 2.8).mean()))
runs = pandas.read_csv(EVIDENCE / "puck_positions_runs.csv")
for number in RUNS:
    log = run_log(number)
    part = runs[runs["run"] == number].merge(log[["step", "picture_heading_deg"]], on = "step")
    part["aim"] = wrap(numpy.degrees(numpy.arctan2(part["puck_y_cm"] - part["tip_y_cm"], part["puck_x_cm"] - part["tip_x_cm"])) - part["picture_heading_deg"])
    part["distance"] = numpy.hypot(part["puck_x_cm"] - part["tip_x_cm"], part["puck_y_cm"] - part["tip_y_cm"])
    close = part[part["distance"] <= 2.8]
    print("   run %2d: tip within 2.8 cm in %.2f of the pictures; of those, aim outside the demonstrations' near range in %.2f; all pictures outside the range of all demonstration pictures in %.2f (the analyst's mixed comparison: %.2f)" % (
        number, len(close) / len(part), ((close["aim"] < range_near[0]) | (close["aim"] > range_near[1])).mean() if len(close) else float("nan"),
        ((part["aim"] < range_all[0]) | (part["aim"] > range_all[1])).mean(), ((part["aim"] < range_near[0]) | (part["aim"] > range_near[1])).mean()))
