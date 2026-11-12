"""
Plan review, key "evidence": small look-ups in the ten run logs of 11.11.26 (steps.csv, summary.json), read only.
1. How each run ended (summary.json "ending") and the stop chance on its last pictures: which runs the model's own
   stop rule ended, and whether the puck was at the line's end then (the plan counts four finished runs).
2. Run 2 (folder 172421_E_sheet01_run1): the last three moves before the stop, asked length against the fixed 1.0 cm
   (the plan says the fixed step made them "2 to 5 times as long").
3. Run 7 (folder 174448_E_sheet05_run1): asked against driven over steps 47 to 55, the walk round the puck
   (dashed-line check, challenge 2).
"""

import json
from pathlib import Path

import numpy
import pandas

MODEL_RUNS = Path("D:/AI_Project/03_data/model_runs")
names = ["2026-11-11_172021_check_no_puck", "2026-11-11_172421_E_sheet01_run1", "2026-11-11_172743_E_sheet02_run1", "2026-11-11_173036_E_sheet02_run2",
         "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1", "2026-11-11_174448_E_sheet05_run1", "2026-11-11_174936_E_sheet06_run1",
         "2026-11-11_175203_E_sheet03_run2_no_hoodie", "2026-11-11_175544_B_team5_sheet03"]
print("1. how the ten runs ended")
for number, name in enumerate(names, start = 1):
    steps = pandas.read_csv(MODEL_RUNS / name / "steps.csv")
    summary = json.loads((MODEL_RUNS / name / "summary.json").read_text(encoding = "utf-8"))
    stop = steps["model_stop_probability"].to_numpy()
    print(f"   run {number:2d} ({name[11:]}): ending '{summary['ending']}', {len(steps)} pictures, {summary['seconds']} s; stop chance on the last two pictures "
          f"{stop[-2]:.2f} and {stop[-1]:.2f}; largest stop chance before them {stop[:-2].max():.2f}")

print()
print("2. run 2, the last moves: asked length, asked direction in the picture, factor of the fixed 1.0 cm")
steps = pandas.read_csv(MODEL_RUNS / names[1] / "steps.csv")
steps["asked_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
steps["asked_direction_deg"] = numpy.degrees(numpy.arctan2(steps["model_tip_left_cm"], steps["model_tip_forward_cm"]))
steps["factor"] = 1.0 / steps["asked_cm"]
print(steps[["step", "action", "asked_cm", "asked_direction_deg", "factor", "model_stop_probability"]].tail(5).round(2).to_string(index = False))

print()
print("3. run 7, steps 47 to 55 (the walk round the puck)")
steps = pandas.read_csv(MODEL_RUNS / names[6] / "steps.csv")
asked = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
done = 100.0 * numpy.hypot(numpy.diff(steps["tip_x"]), numpy.diff(steps["tip_y"]))
print(f"   asked {asked.iloc[47:56].round(2).tolist()} cm, sum {asked.iloc[47:56].sum():.2f} cm; driven {done[47:56].sum():.2f} cm; "
      f"forearm from {numpy.degrees(steps['joint_4'].iloc[47]):.1f} to {numpy.degrees(steps['joint_4'].iloc[56]):.1f} degrees")
