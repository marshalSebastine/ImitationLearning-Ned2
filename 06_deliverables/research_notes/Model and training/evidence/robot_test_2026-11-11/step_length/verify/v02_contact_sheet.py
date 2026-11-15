"""
Verifier: contact sheets of the logged pictures of a robot run, so that a person (or the verifier) can see with their own
eyes where the puck is against the line at each decision. Reads pictures only; nothing connects to the robot.

Run from this folder:  python v02_contact_sheet.py <run number> <first step> <last step> [stride]
Writes v02_run_<NN>_steps_<first>_<last>.png
"""

import os
import sys

import cv2
import numpy

RUNS_FOLDER = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = [
    "2026-11-11_172021_check_no_puck", "2026-11-11_172421_E_sheet01_run1", "2026-11-11_172743_E_sheet02_run1",
    "2026-11-11_173036_E_sheet02_run2", "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1",
    "2026-11-11_174448_E_sheet05_run1", "2026-11-11_174936_E_sheet06_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie",
    "2026-11-11_175544_B_team5_sheet03"
]

number, first, last = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
columns = int(sys.argv[5]) if len(sys.argv) > 5 else 4
folder = os.path.join(RUNS_FOLDER, RUN_FOLDERS[number - 1], "frames")
tiles = []
for step in range(first, last + 1, stride):
    picture = cv2.imread(os.path.join(folder, f"{step:03d}.jpg"))
    if picture is None:
        continue
    picture = cv2.resize(picture, (480, 360), interpolation = cv2.INTER_AREA)
    cv2.putText(picture, f"run {number} step {step}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(picture)
while len(tiles) % columns:
    tiles.append(numpy.zeros_like(tiles[0]))
rows = [numpy.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"v02_run_{number:02d}_steps_{first:03d}_{last:03d}.png")
cv2.imwrite(out, numpy.vstack(rows))
print(out)
