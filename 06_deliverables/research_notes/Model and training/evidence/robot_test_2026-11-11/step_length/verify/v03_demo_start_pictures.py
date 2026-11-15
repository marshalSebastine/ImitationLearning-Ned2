"""
Verifier: the first pictures of the standard-start demonstrations of sheets 03 and 04 (end A) next to the first pictures of
the robot runs on the same sheet, with joint 1 and joint 4 printed, to judge by eye how the sheet lay.
Reads pictures and csv only. Run from this folder:  python v03_demo_start_pictures.py
"""

import os

import cv2
import numpy
import pandas

INDEX = "D:/AI_Project/03_data/processed/v01_training_index/frames.csv"
RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"
HERE = os.path.dirname(os.path.abspath(__file__))

frames = pandas.read_csv(INDEX, low_memory = False, usecols = ["episode_id", "row_in_episode", "sheet", "plan_number", "start_kind", "start_end", "plan_end",
                                                             "frame_path", "joint_1", "joint_4", "tip_x_m", "tip_y_m", "action", "sheet_orientation"])
frames = frames.sort_values(["episode_id", "row_in_episode"])

def tile(picture, text):
    picture = cv2.resize(picture, (400, 300), interpolation = cv2.INTER_AREA)
    cv2.putText(picture, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return picture

for sheet, run_folders in [("path_03", ["2026-11-11_173549_E_sheet03_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie", "2026-11-11_175544_B_team5_sheet03"]),
                           ("path_04", ["2026-11-11_174108_E_sheet04_run1"])]:
    tiles = []
    for folder in run_folders:
        steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
        for step in (0, 3):
            picture = cv2.imread(os.path.join(RUNS, folder, "frames", f"{step:03d}.jpg"))
            row = steps.iloc[step]
            tiles.append(tile(picture, f"{folder[11:17]} s{step} j1 {numpy.degrees(row['joint_1']):.0f} j4 {numpy.degrees(row['joint_4']):.0f} tip {100 * row['tip_x']:.1f},{100 * row['tip_y']:.1f}"))
    chosen = frames[(frames["sheet"] == sheet) & (frames["start_kind"] == "STD")]
    print(sheet, "STD plans:", sorted(chosen["plan_number"].dropna().unique()))
    for plan, group in chosen.groupby("plan_number"):
        print(f"   plan {plan}: start_end {group['start_end'].iloc[0]}, plan_end {group['plan_end'].iloc[0]}, rows {len(group)}, orientation text: {group['sheet_orientation'].iloc[0]}; "
              f"first tip ({100 * group['tip_x_m'].iloc[0]:.1f}, {100 * group['tip_y_m'].iloc[0]:.1f}) cm, joint 4 first {numpy.degrees(group['joint_4'].iloc[0]):.1f}")
        for row_number in (0, 4, 8):
            if row_number >= len(group):
                continue
            row = group.iloc[row_number]
            picture = cv2.imread(os.path.join(RAW, row["frame_path"]))
            if picture is None:
                continue
            tiles.append(tile(picture, f"demo {plan} r{row_number} j1 {numpy.degrees(row['joint_1']):.0f} j4 {numpy.degrees(row['joint_4']):.0f} tip {100 * row['tip_x_m']:.1f},{100 * row['tip_y_m']:.1f}"))
    columns = 6
    while len(tiles) % columns:
        tiles.append(numpy.zeros_like(tiles[0]))
    rows = [numpy.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
    out = os.path.join(HERE, f"v03_start_pictures_{sheet}.png")
    cv2.imwrite(out, numpy.vstack(rows))
    print(out, len(tiles))
