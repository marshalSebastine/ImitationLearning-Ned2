"""Verifier 10: picture sheets of runs 3 and 4 (sheet 02) with the asked step on each tile. Reads only; writes v10_runs_3_4.png."""
from pathlib import Path
import cv2
import numpy
from vcommon import run_log, RUNS, RUNS_FOLDER

OUT = Path(__file__).resolve().parent


def tile(picture, text):
    picture = cv2.resize(picture, (320, 240), interpolation = cv2.INTER_AREA)
    cv2.rectangle(picture, (0, 0), (320, 18), (255, 255, 255), -1)
    cv2.putText(picture, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return picture


rows = []
for number, picks in ((3, [0, 2, 4, 6, 8, 10, 13]), (4, [0, 6, 12, 17, 20, 24, 30])):
    s = run_log(number)
    rows.append(numpy.hstack([tile(cv2.imread(str(RUNS_FOLDER / RUNS[number][0] / "frames" / f"{k:03d}.jpg")),
                                   f"run {number} #{k} j4 {s['j4'].iloc[k]:+.0f} ask {s['asked_cm'].iloc[k]:.2f}cm {s['model_angle_deg'].iloc[k]:+.0f}deg tw {s['model_twist_deg'].iloc[k]:+.1f}") for k in picks]))
cv2.imwrite(str(OUT / "v10_runs_3_4.png"), numpy.vstack(rows))
s = run_log(3)
print("run 3 first 16 pictures: angle / length / twist / joint 4")
for k in range(16):
    print(f"  {k:2d}: {s['model_angle_deg'].iloc[k]:+6.1f}  {s['asked_cm'].iloc[k]:.2f}  {s['model_twist_deg'].iloc[k]:+5.1f}  {s['j4'].iloc[k]:+6.1f}")
