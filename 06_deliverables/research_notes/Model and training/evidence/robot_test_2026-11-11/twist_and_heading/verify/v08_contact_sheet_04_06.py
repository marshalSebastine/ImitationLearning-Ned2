"""
Verifier 08: picture sheets for sheet 04 (run 6) and sheet 06 (run 8) against a standard-start demonstration of the same
sheet and end, to see with my own eyes where the start was and what the camera looked at. Reads only.
Writes v08_sheet_04.png and v08_sheet_06.png into this folder.
"""
from pathlib import Path
import cv2
import numpy
from vcommon import demos, run_log, RUNS, RUNS_FOLDER

RAW = Path("D:/AI_Project/03_data/raw/main_data")
OUT = Path(__file__).resolve().parent


def tile(picture, text):
    picture = cv2.resize(picture, (320, 240), interpolation = cv2.INTER_AREA)
    cv2.rectangle(picture, (0, 0), (320, 18), (255, 255, 255), -1)
    cv2.putText(picture, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return picture


frames = demos()
for name, plan, sheet, number, picks_demo, picks_run in (("v08_sheet_04.png", "04-01", "path_04", 6, [0, 4, 8, 12, 16, 20, 25], [0, 4, 8, 12, 16, 20, 25]),
                                                         ("v08_sheet_06.png", "06-01", "path_06", 8, [0, 4, 8, 12, 16, 20, 25], [0, 3, 6, 9, 12, 15, 18])):
    rows = []
    e = frames[(frames["plan_number"] == plan) & (frames["sheet"] == sheet)].reset_index(drop = True)
    arc = numpy.r_[0.0, numpy.cumsum(e["len_cm"].to_numpy())]
    rows.append(numpy.hstack([tile(cv2.imread(str(RAW / e["frame_path"].iloc[k])), f"demo {plan} #{k} j4 {e['j4'].iloc[k]:+.0f} path {arc[k]:.1f} cm") for k in picks_demo]))
    s = run_log(number)
    arc = numpy.r_[0.0, numpy.cumsum(s["moved_cm"].to_numpy())]
    rows.append(numpy.hstack([tile(cv2.imread(str(RUNS_FOLDER / RUNS[number][0] / "frames" / f"{k:03d}.jpg")), f"run {number} #{k} j4 {s['j4'].iloc[k]:+.0f} path {arc[k]:.1f} asked {s['asked_cm'].iloc[k]:.2f}cm {s['model_twist_deg'].iloc[k]:+.1f}deg") for k in picks_run]))
    cv2.imwrite(str(OUT / name), numpy.vstack(rows))
    print("written", OUT / name)
s = run_log(6)
print("run 6, per picture: asked angle, asked length, asked twist, joint 4")
for k in range(0, 40):
    print(f"  {k:2d}: angle {s['model_angle_deg'].iloc[k]:+7.1f}  length {s['asked_cm'].iloc[k]:.2f}  twist {s['model_twist_deg'].iloc[k]:+5.1f}  joint 4 {s['j4'].iloc[k]:+6.1f}  stop {s['model_stop_probability'].iloc[k]:.3f}  kept {int(s['is_direction_kept'].iloc[k])}")
e = frames[(frames["plan_number"] == "04-01") & (frames["sheet"] == "path_04")].reset_index(drop = True)
print("demonstration 04-01: joint 4 at rows 0, 5, 10, 15, 20, 25, 30:", [round(float(e["j4"].iloc[k]), 1) for k in (0, 5, 10, 15, 20, 25, 30)], "; mean |twist| per picture over the first 60 rows: %.2f" % e["twist"].iloc[:60].abs().mean())
ref = frames[(frames["sheet"] == "path_04") & (frames["start_kind"] == "STD") & (frames["start_end"] == "A") & (frames["action"] != "stop")]
print("sheet 04 standard starts from end A: mean |twist| per picture %.2f; run 6 mean |asked twist| per decision %.2f" % (ref["twist"].abs().mean(), s.loc[s["action"] == "model_step", "model_twist_deg"].abs().mean()))
