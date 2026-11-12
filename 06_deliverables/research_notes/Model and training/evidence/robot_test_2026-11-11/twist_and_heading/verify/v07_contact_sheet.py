"""
Verifier 07: pictures side by side, to look at the first bend of sheet 03 with my own eyes.
Row 1: demonstration plan 03-01 (standard start, end A) at pictures 0, 3, 6, 9, 12, 15.
Rows 2 to 4: robot runs 9, 10 and 5 at pictures 0, 3, 6, 9, 12, 15.
Each tile is labelled with the picture number, joint 4 and the path length of the tip so far.
Reads only; writes v07_bend_sheet_03.png into this folder.
"""
from pathlib import Path
import cv2
import numpy
from vcommon import demos, run_log, RUNS, RUNS_FOLDER

RAW = Path("D:/AI_Project/03_data/raw/main_data")
OUT = Path(__file__).resolve().parent
PICKS = [0, 3, 6, 9, 12, 15, 18]


def tile(picture, text):
    picture = cv2.resize(picture, (320, 240), interpolation = cv2.INTER_AREA)
    cv2.rectangle(picture, (0, 0), (320, 18), (255, 255, 255), -1)
    cv2.putText(picture, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return picture


frames = demos()
rows = []
for plan in ("03-01", "03-25"):
    e = frames[(frames["plan_number"] == plan) & (frames["sheet"] == "path_03")].reset_index(drop = True)
    arc = numpy.r_[0.0, numpy.cumsum(e["len_cm"].to_numpy())]
    rows.append(numpy.hstack([tile(cv2.imread(str(RAW / e["frame_path"].iloc[k])), f"demo {plan} #{k} j4 {e['j4'].iloc[k]:+.0f} path {arc[k]:.1f} cm") for k in PICKS]))
for number in (9, 10, 5):
    s = run_log(number)
    arc = numpy.r_[0.0, numpy.cumsum(s["moved_cm"].to_numpy())]
    rows.append(numpy.hstack([tile(cv2.imread(str(RUNS_FOLDER / RUNS[number][0] / "frames" / f"{k:03d}.jpg")), f"run {number} #{k} j4 {s['j4'].iloc[k]:+.0f} path {arc[k]:.1f} cm") for k in PICKS]))
cv2.imwrite(str(OUT / "v07_bend_sheet_03.png"), numpy.vstack(rows))
print("written", OUT / "v07_bend_sheet_03.png")
