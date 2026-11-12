"""
Verifier's check 3 (11.11.26): where the robot stopped in runs 1, 2, 4 and 7 against the line's round end, with my own
measure (end_tools.py), once with the camera as fitted on 6.11.26 and once with the analyst's picture shift and roll.
Reads the runs' pictures and steps.csv only.
"""

from pathlib import Path

import cv2
import pandas

import end_tools
import my_measure

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RUNS = {1: "2026-11-11_172021_check_no_puck", 2: "2026-11-11_172421_E_sheet01_run1",
        4: "2026-11-11_173036_E_sheet02_run2", 7: "2026-11-11_174448_E_sheet05_run1"}
ANALYST = {1: (-3.9, -21.3), 2: (-1.9, 13.1), 4: (-7.5, -0.3), 7: (-1.3, 1.9)}
CAMERAS = {"camera as fitted 6.11.26": None, "analyst's correction (9.9, -2.7) px, roll 1.26 deg": my_measure.shifted_camera((9.9, -2.7), 1.26)}

for number, folder in RUNS.items():
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    last = steps.iloc[-14:]
    pictures = [cv2.imread(str(RUNS_FOLDER / folder / frame)) for frame in last["frame"]]
    joints = [[row[f"joint_{index}"] for index in range(1, 7)] for _, row in last.iterrows()]
    tips = last[["tip_x", "tip_y"]].to_numpy(dtype = float)
    print(f"run {number} ({folder}); the analyst: short {ANALYST[number][0]:+.1f} mm, left {ANALYST[number][1]:+.1f} mm")
    for label, camera in CAMERAS.items():
        measured = end_tools.measure_stop(pictures, joints, tips, camera)
        if measured["puck"] is None or not measured["references"]:
            print(f"  {label}: nothing measured")
            continue
        print(f"  {label}: puck at the stop {measured['puck']['xy'].round(4)} (fit residual {measured['puck']['residual_px']:.1f} px)")
        for row in measured["references"]:
            print(f"    end seen {int(row['steps_before']):2d} steps before the stop: short {row['short_mm']:+6.1f} mm, left {row['aside_mm']:+6.1f} mm,"
                  f" centres {row['gap_mm']:5.1f} mm apart; line width there {row['width_mm']:.1f} mm; line seen over {row['seen_length_mm']:.0f} mm")
        summary = end_tools.summary_of(measured["references"])
        if summary is not None:
            print(f"    over the {summary['references']} reference pictures with a line 8 to 11 mm wide seen over 15 mm or more:"
                  f" short median {summary['short_mm']:+.1f} (range {summary['short_range_mm'][0]:+.1f} to {summary['short_range_mm'][1]:+.1f}),"
                  f" left median {summary['aside_mm']:+.1f} (range {summary['aside_range_mm'][0]:+.1f} to {summary['aside_range_mm'][1]:+.1f}),"
                  f" centres {summary['gap_mm']:.1f} mm apart")
