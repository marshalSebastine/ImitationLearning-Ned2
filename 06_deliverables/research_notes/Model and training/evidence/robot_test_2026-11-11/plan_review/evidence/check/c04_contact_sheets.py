"""
Check of the plan review, key "evidence_check": contact sheets of logged robot pictures, to look with my own eyes at
(a) whether a person or the blue hoodie is in view in run 10 before the puck is lost (pictures 12 to 37; the check
verified "the hoodie did not cause the failure" for runs 5 and 6 only), and (b) the first pictures of run 3.
Reads the logged JPEG files only; writes two picture sheets into this folder.
"""

from pathlib import Path

import cv2
import numpy

MODEL_RUNS = Path("D:/AI_Project/03_data/model_runs")
HERE = Path(__file__).resolve().parent


def sheet(folder_name: str, steps: list, columns: int, out_name: str) -> None:

    tiles = []
    for step in steps:
        picture = cv2.imread(str(MODEL_RUNS / folder_name / "frames" / f"{step:03d}.jpg"))
        picture = cv2.resize(picture, (320, 240), interpolation = cv2.INTER_AREA)
        cv2.putText(picture, f"{step:03d}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(picture)
    while len(tiles) % columns:
        tiles.append(numpy.zeros_like(tiles[0]))
    rows = [numpy.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
    cv2.imwrite(str(HERE / out_name), numpy.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


sheet("2026-11-11_175544_B_team5_sheet03", list(range(12, 38, 2)) + [37], 5, "c04_run_10_pictures_012_037.jpg")
sheet("2026-11-11_172743_E_sheet02_run1", list(range(0, 10)), 5, "c04_run_03_pictures_000_009.jpg")
print("written")
