"""
Verifier's check 10 (11.11.26): the last five pictures of runs 1, 2, 4 and 7 with my own measure: where the puck's
foot stood against the line's end (the end's place: the median over my single reference pictures), how far the puck
and the magnet tip moved along and across the line in each step, and what the model had asked for.
The camera carries the analyst's correction; reads pictures and steps.csv only.
"""

import math
from pathlib import Path

import cv2
import numpy
import pandas

import end_tools
import my_measure

cv2.setNumThreads(2)
RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RUNS = {1: "2026-11-11_172021_check_no_puck", 2: "2026-11-11_172421_E_sheet01_run1",
        4: "2026-11-11_173036_E_sheet02_run2", 7: "2026-11-11_174448_E_sheet05_run1"}
ANALYST_FIRST_STOP_SHORT = {1: 5.4, 2: 6.9, 4: 1.9, 7: 7.9}
CAMERA = my_measure.shifted_camera((9.9, -2.7), 1.26)

for number, folder in RUNS.items():
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    last = steps.iloc[-14:]
    pictures = [cv2.imread(str(RUNS_FOLDER / folder / frame)) for frame in last["frame"]]
    joints = [[row[f"joint_{index}"] for index in range(1, 7)] for _, row in last.iterrows()]
    tips = last[["tip_x", "tip_y"]].to_numpy(dtype = float)
    measured = end_tools.measure_stop(pictures, joints, tips, CAMERA)
    good = [entry for entry in measured["references"] if 8.0 <= entry["width_mm"] <= 11.0 and entry["seen_length_mm"] >= 15.0]
    end_centre = numpy.median(numpy.array([entry["end_centre"] for entry in good]), axis = 0)
    direction = numpy.mean(numpy.array([entry["direction"] for entry in good]), axis = 0)
    direction = direction / numpy.linalg.norm(direction)
    left = numpy.array([-direction[1], direction[0]])
    print(f"run {number} ({folder}): the end's centre from {len(good)} reference pictures at {end_centre.round(4)}, the line's direction there {math.degrees(math.atan2(direction[1], direction[0])):+.1f} degrees in the base frame")
    previous_puck, previous_tip = None, None
    for position in range(len(pictures) - 5, len(pictures)):
        row = last.iloc[position]
        puck = my_measure.puck_foot(pictures[position], joints[position], CAMERA)
        if puck is None:
            continue
        gap = end_centre - puck["xy"]
        short, aside = 1000.0 * float(gap @ direction), -1000.0 * float(gap @ left)
        text = f"  picture {int(row['step']):3d}: puck short {short:+6.1f} mm, left {aside:+6.1f} mm; stop chance {row['model_stop_probability']:.3f}"
        asked = math.hypot(row["model_tip_forward_cm"], row["model_tip_left_cm"])
        text += f"; model asked {asked:.2f} cm"
        if previous_puck is not None:
            puck_move = 1000.0 * (puck["xy"] - previous_puck)
            tip_move = 1000.0 * (tips[position] - previous_tip)
            text += (f"; since the picture before: puck {float(puck_move @ direction):+5.1f} along, {float(puck_move @ left):+5.1f} left;"
                     f" tip {float(tip_move @ direction):+5.1f} along, {float(tip_move @ left):+5.1f} left")
        print(text)
        previous_puck, previous_tip = puck["xy"], tips[position]
    print(f"  the analyst: the puck was {ANALYST_FIRST_STOP_SHORT[number]:.1f} mm short when the stop chance first passed 0.5 (the picture before the last)")
