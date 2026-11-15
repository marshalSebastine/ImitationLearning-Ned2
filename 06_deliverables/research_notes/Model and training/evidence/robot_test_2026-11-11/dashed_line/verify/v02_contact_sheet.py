"""
Verifier check 2: larger tiles of chosen pictures of a logged run or a demonstration, to look at them myself.

Run:  D:/AI_Project/.venv/Scripts/python.exe v02_contact_sheet.py <kind> <name> <first> <last> <columns> <out.jpg>
      kind = run  -> name is the folder under 03_data/model_runs
      kind = demo -> name is the plan number (for example 06-03), pictures from frames.csv
"""

import sys
from pathlib import Path

import cv2
import numpy
import pandas

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")

kind, name, first, last, columns, out_name = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
scale = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
tiles = []
if kind == "run":
    steps = pandas.read_csv(RUNS_FOLDER / name / "steps.csv")
    for row in steps.itertuples():
        if not first <= row.step <= last:
            continue
        picture = cv2.imread(str(RUNS_FOLDER / name / row.frame))
        text = (f"{row.step} a{row.model_angle_deg:+.0f} L{numpy.hypot(row.model_tip_forward_cm, row.model_tip_left_cm):.2f}"
                f" t{row.model_twist_deg:+.1f} j4 {numpy.degrees(row.joint_4):.0f} s{row.model_stop_probability:.2f}")
        tiles.append((picture, text))
else:
    frames = pandas.read_csv(INDEX_FOLDER / "frames.csv")
    rows = frames[frames["plan_number"] == name].sort_values("row_in_episode")
    for row in rows.itertuples():
        if not first <= row.row_in_episode <= last:
            continue
        picture = cv2.imread(str(RAW_FOLDER / row.frame_path))
        text = (f"{row.row_in_episode} a{row.move_angle_deg:+.0f} L{100 * row.move_length_m:.2f} t{row.forearm_change_deg:+.1f}"
                f" j4 {numpy.degrees(row.joint_4):.0f} j5 {numpy.degrees(row.joint_5):.0f} {row.action}")
        tiles.append((picture, text))
width, height = int(640 * scale), int(480 * scale)
rows_count = int(numpy.ceil(len(tiles) / columns))
sheet = numpy.zeros((rows_count * (height + 18), columns * width, 3), dtype = numpy.uint8)
for number, (picture, text) in enumerate(tiles):
    small = cv2.resize(picture, (width, height), interpolation = cv2.INTER_AREA)
    top = (number // columns) * (height + 18)
    left = (number % columns) * width
    sheet[top + 18:top + 18 + height, left:left + width] = small
    cv2.putText(sheet, text, (left + 3, top + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(out_name, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"{out_name}: {len(tiles)} pictures")
