"""
Verifier, check 16: look at the start of failing runs beside the plan's own demonstration (pictures only, no numbers
beyond the logged tip and twist): run 8 against plan 06-01, run 6 against plan 04-01, run 5 against plan 03-01. For each
run its pictures 0 and 4, and the demonstration rows whose tip is nearest to those run tips.
    D:/AI_Project/.venv/Scripts/python.exe v16_start_pictures.py
"""
import math
import cv2
import numpy
import pandas

RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"
frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "tip_x_m", "tip_y_m", "joint_4", "joint_5", "frame_path"])
frames["plan"] = frames["episode_id"].str.extract(r"plan_(\d\d-\d\d)")[0]
tiles = []
for run, folder, plan in ((8, "2026-11-11_174936_E_sheet06_run1", "06-01"), (6, "2026-11-11_174108_E_sheet04_run1", "04-01"), (5, "2026-11-11_173549_E_sheet03_run1", "03-01")):
    steps = pandas.read_csv(f"{RUNS}/{folder}/steps.csv")
    demo = frames[frames["plan"] == plan].sort_values("row_in_episode")
    demo = demo[demo["episode_id"] == demo["episode_id"].iloc[0]]
    row_tiles = []
    for picture_number in (0, 4):
        step = steps.iloc[picture_number]
        picture = cv2.imread(f"{RUNS}/{folder}/frames/{picture_number:03d}.jpg")
        cv2.putText(picture, "run %d pic %d: tip %.1f, %.1f cm, twist %+.0f" % (run, picture_number, 100 * step["tip_x"], 100 * step["tip_y"], math.degrees(step["joint_4"])),
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        row_tiles.append(picture)
        distance = numpy.hypot(demo["tip_x_m"] - step["tip_x"], demo["tip_y_m"] - step["tip_y"])
        nearest = demo.loc[distance.idxmin()]
        picture = cv2.imread(f"{RAW}/{nearest['frame_path']}")
        cv2.putText(picture, "plan %s row %d: tip %.1f, %.1f, twist %+.0f, %.1f cm away" % (plan, nearest["row_in_episode"], 100 * nearest["tip_x_m"], 100 * nearest["tip_y_m"],
                    math.degrees(nearest["joint_4"]), 100 * distance.min()), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        row_tiles.append(picture)
    tiles.append(numpy.hstack(row_tiles))
    first = demo.iloc[0]
    print("plan %s first row: tip %.1f, %.1f cm; run %d first picture: tip %.1f, %.1f cm" % (plan, 100 * first["tip_x_m"], 100 * first["tip_y_m"], run, 100 * steps.iloc[0]["tip_x"], 100 * steps.iloc[0]["tip_y"]))
cv2.imwrite("v16_start_pictures.jpg", cv2.resize(numpy.vstack(tiles), None, fx = 0.55, fy = 0.55), [cv2.IMWRITE_JPEG_QUALITY, 85])
