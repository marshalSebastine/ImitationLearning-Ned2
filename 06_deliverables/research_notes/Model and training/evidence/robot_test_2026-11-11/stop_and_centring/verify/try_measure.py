"""
Verifier's first try of my_measure.py on a robot run: overlays to see whether the puck and the line are found.
Usage: python try_measure.py [run folder name] [out.png] [steps ...]
Yellow: pixels taken as line; magenta tint: the hand's mask; cyan: the fitted foot circle of the puck on the table.
"""

import sys
from pathlib import Path

import cv2
import numpy
import pandas

import my_measure

RUN = Path("D:/AI_Project/03_data/model_runs") / (sys.argv[1] if len(sys.argv) > 1 else "2026-11-11_172421_E_sheet01_run1")
steps = pandas.read_csv(RUN / "steps.csv")
chosen = [int(value) for value in sys.argv[3:]] or [8, 16, 21, 22, 23, 24]
tiles = []
for step in chosen:
    row = steps.iloc[step]
    joints = [row[f"joint_{number}"] for number in range(1, 7)]
    picture = cv2.imread(str(RUN / row["frame"]))
    found = my_measure.puck_foot(picture, joints)
    line = my_measure.line_mask(picture, joints, puck = None if found is None else found["mask"])
    shown = picture.copy()
    shown[line] = (0, 255, 255)
    hidden = my_measure.hand_mask(picture.shape[:2], joints)
    shown[hidden] = (0.6 * shown[hidden] + 0.4 * numpy.array([255, 0, 255])).astype(numpy.uint8)
    if found is not None:
        circle = my_measure.foot_circle_px(found["xy"], joints)
        cv2.polylines(shown, [numpy.round(circle).astype(numpy.int32)], True, (255, 255, 0), 1)
        cv2.circle(shown, tuple(numpy.round(found["foot_px"]).astype(int)), 2, (255, 255, 0), -1)
        cv2.circle(shown, tuple(numpy.round(found["top_px"]).astype(int)), 2, (0, 255, 0), -1)
        tip = numpy.array([row["tip_x"], row["tip_y"]])
        print(f"step {step}: puck foot {found['xy'].round(4)}, residual {found['residual_px']:.1f} px over {found['directions']} directions; "
              f"puck minus tip {1000 * numpy.linalg.norm(found['xy'] - tip):.1f} mm; line pixels {int(line.sum())}")
    cv2.putText(shown, f"step {step}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    tiles.append(shown)
while len(tiles) % 3:
    tiles.append(numpy.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else "try_measure_run2.png", numpy.vstack([numpy.hstack(tiles[index:index + 3]) for index in range(0, len(tiles), 3)]))
