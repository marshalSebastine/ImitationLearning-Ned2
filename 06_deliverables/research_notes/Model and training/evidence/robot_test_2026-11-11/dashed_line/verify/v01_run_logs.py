"""
Verifier check 1: what the logs of runs 7 and 8 say by themselves (no camera model, no pictures).

Reads steps.csv of the two robot runs and prints the asked step, the step done, the forearm angle, the stop chance and
the wrist. Nothing here connects to the robot.

Run:  D:/AI_Project/.venv/Scripts/python.exe v01_run_logs.py
"""

from pathlib import Path

import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RUN_FOLDERS = {"run_7 (sheet 05)": "2026-11-11_174448_E_sheet05_run1", "run_8 (sheet 06)": "2026-11-11_174936_E_sheet06_run1"}

pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
pandas.set_option("display.max_columns", 40)

for run_label, folder_name in RUN_FOLDERS.items():
    steps = pandas.read_csv(RUNS_FOLDER / folder_name / "steps.csv")
    steps["asked_length_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
    steps["joint_4_deg"] = numpy.degrees(steps["joint_4"])
    steps["joint_5_deg"] = numpy.degrees(steps["joint_5"])
    steps["joint_6_deg"] = numpy.degrees(steps["joint_6"])
    tip = 100.0 * steps[["tip_x", "tip_y"]].to_numpy()
    done = numpy.hypot(*numpy.diff(tip, axis = 0).T)
    steps["done_length_cm"] = numpy.append(done, numpy.nan)
    steps["is_backward_asked"] = numpy.abs(steps["model_angle_deg"]) > 90.0
    print(f"\n===== {run_label}: {len(steps)} pictures, actions {steps['action'].value_counts().to_dict()} =====")
    print(steps[["step", "model_angle_deg", "asked_length_cm", "done_length_cm", "model_twist_deg", "twist_used_deg",
                 "joint_4_deg", "joint_5_deg", "model_stop_probability", "is_turn_clamped", "is_direction_kept",
                 "action"]].round(2).to_string(index = False))
    print(f"asked length: median {steps['asked_length_cm'].median():.2f} cm, sum {steps['asked_length_cm'].sum():.1f} cm;"
          f" done: sum {numpy.nansum(done):.1f} cm, median {numpy.nanmedian(done):.2f} cm")
    print(f"asked backward (|angle| > 90): {int(steps['is_backward_asked'].sum())} answers,"
          f" rows {steps.index[steps['is_backward_asked']].tolist()}")
    print(f"joint 4: first {steps['joint_4_deg'].iloc[0]:.1f}, least {steps['joint_4_deg'].min():.1f},"
          f" most {steps['joint_4_deg'].max():.1f}, last {steps['joint_4_deg'].iloc[-1]:.1f} deg")
    print(f"joint 5 (wrist): {steps['joint_5_deg'].min():.2f} to {steps['joint_5_deg'].max():.2f} deg;"
          f" joint 6: {steps['joint_6_deg'].min():.2f} to {steps['joint_6_deg'].max():.2f} deg")
    last = steps.iloc[-14:]
    print(f"last 14 answers (rows {last['step'].iloc[0]} to {last['step'].iloc[-1]}): asked {last['asked_length_cm'].sum():.1f} cm,"
          f" done {numpy.nansum(last['done_length_cm']):.1f} cm, asked twist sum {last['model_twist_deg'].sum():.1f} deg")
    last_before_stop = steps.iloc[-14:-1]
    print(f"   the 13 answers before the last picture: asked {last_before_stop['asked_length_cm'].sum():.1f} cm,"
          f" done {numpy.nansum(last_before_stop['done_length_cm']):.1f} cm")
    print(f"twist asked in total: {steps['model_twist_deg'].sum():.1f} deg; used {steps['twist_used_deg'].sum():.1f} deg")
