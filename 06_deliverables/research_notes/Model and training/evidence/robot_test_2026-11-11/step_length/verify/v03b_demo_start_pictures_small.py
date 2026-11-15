"""
Verifier: a small version of v03: per sheet, the robot runs' first picture and rows 0, 3, 6, 9 of two demonstrations of each
placement group named by the analyst. Reads pictures and csv only.
Run from this folder:  python v03b_demo_start_pictures_small.py
"""

import os

import cv2
import numpy
import pandas

INDEX = "D:/AI_Project/03_data/processed/v01_training_index/frames.csv"
RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"
HERE = os.path.dirname(os.path.abspath(__file__))

frames = pandas.read_csv(INDEX, low_memory = False, usecols = ["episode_id", "row_in_episode", "sheet", "plan_number", "start_kind", "frame_path", "joint_1", "joint_4",
                                                             "tip_x_m", "tip_y_m", "picture_heading_deg"])
frames = frames.sort_values(["episode_id", "row_in_episode"])

def tile(picture, text):
    picture = cv2.resize(picture, (400, 300), interpolation = cv2.INTER_AREA)
    cv2.putText(picture, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return picture

jobs = [("path_03", ["2026-11-11_173549_E_sheet03_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie", "2026-11-11_175544_B_team5_sheet03"], ["03-01", "03-04", "03-25", "03-27"]),
        ("path_04", ["2026-11-11_174108_E_sheet04_run1"], ["04-01", "04-05", "04-13", "04-17"])]
for sheet, run_folders, plans in jobs:
    tiles = []
    for folder in run_folders:
        steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
        row = steps.iloc[0]
        picture = cv2.imread(os.path.join(RUNS, folder, "frames", "000.jpg"))
        tiles.append(tile(picture, f"{folder[11:17]} s0 j1 {numpy.degrees(row['joint_1']):.0f} j4 {numpy.degrees(row['joint_4']):.0f} head {row['picture_heading_deg']:.0f} tip {100 * row['tip_x']:.1f},{100 * row['tip_y']:.1f}"))
    while len(tiles) % 4:
        tiles.append(numpy.zeros_like(tiles[0]))
    for plan in plans:
        group = frames[frames["plan_number"] == plan]
        for row_number in (0, 3, 6, 9):
            row = group.iloc[row_number]
            picture = cv2.imread(os.path.join(RAW, row["frame_path"]))
            tiles.append(tile(picture, f"demo {plan} r{row_number} j1 {numpy.degrees(row['joint_1']):.0f} j4 {numpy.degrees(row['joint_4']):.0f} head {row['picture_heading_deg']:.0f} tip {100 * row['tip_x_m']:.1f},{100 * row['tip_y_m']:.1f}"))
    rows = [numpy.hstack(tiles[index:index + 4]) for index in range(0, len(tiles), 4)]
    out = os.path.join(HERE, f"v03b_start_pictures_{sheet}.png")
    cv2.imwrite(out, numpy.vstack(rows))
    print(out, len(tiles))
