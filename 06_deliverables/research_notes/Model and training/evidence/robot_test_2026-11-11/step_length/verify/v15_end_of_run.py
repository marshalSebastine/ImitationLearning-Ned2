"""
Verifier: what the model asked in the last decisions before the stop of the runs that reached the line's end (1, 2, 4, 7),
to see what a driver that carries out the asked length would do to the place of the stop. Reads steps.csv only.

Run from this folder:  python v15_end_of_run.py > v15_end_of_run.out.txt
"""

import os

import numpy
import pandas

RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = {1: "2026-11-11_172021_check_no_puck", 2: "2026-11-11_172421_E_sheet01_run1", 4: "2026-11-11_173036_E_sheet02_run2", 7: "2026-11-11_174448_E_sheet05_run1"}
pandas.set_option("display.width", 250)
for number, folder in RUN_FOLDERS.items():
    steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
    steps["asked_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
    tips = 100.0 * steps[["tip_x", "tip_y"]].to_numpy()
    last = steps.iloc[-8:][["step", "asked_cm", "model_twist_deg", "model_stop_probability", "action"]]
    print(f"--- run {number}: the last 8 pictures")
    print(last.round(2).to_string(index = False))
    moves = steps[steps["action"] == "model_step"].iloc[-6:]
    shorter = (1.0 - numpy.minimum(moves["asked_cm"], 1.2)).sum()
    first_half = int(numpy.flatnonzero(steps["model_stop_probability"].to_numpy() >= 0.25)[0]) if (steps["model_stop_probability"] >= 0.25).any() else None
    print(f"   over the last 6 moves the asked lengths add up to {numpy.minimum(moves['asked_cm'], 1.2).sum():.2f} cm against {len(moves)}.0 cm done: with the same answers the tip would be "
          f"{shorter:+.2f} cm short of where it stopped (open loop); the stop chance first reached 0.25 at decision {first_half}, the run stopped at decision {int(steps['step'].iloc[-1])}")
    print()
