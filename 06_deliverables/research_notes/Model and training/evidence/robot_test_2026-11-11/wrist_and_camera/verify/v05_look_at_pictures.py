"""
Verifier, check 5a: put pictures side by side to look at them (no numbers). First rows of start-mark demonstrations
before and after the claimed camera step, and robot-run pictures (all at wrist 13.5, hand -6.1).
    D:/AI_Project/.venv/Scripts/python.exe v05_look_at_pictures.py
"""
import cv2
import numpy
import pandas

RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"
frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "frame_path", "started", "joint_5", "joint_6"])
first = frames[frames["row_in_episode"] == 0].sort_values("started")
first = first[first["sheet_orientation"].str.contains("START MARK", na = False)]
picks = []
for sheet, position in (("path_01", 0), ("path_03", 5), ("path_04", 3), ("path_04", -1), ("path_05", 3), ("path_06", 3)):
    row = first[first["sheet"] == sheet].iloc[position]
    picks.append((f"{sheet[-2:]} {row['started'][5:16]}", f"{RAW}/{row['frame_path']}"))
picks.append(("run 2 pic 0", f"{RUNS}/2026-11-11_172421_E_sheet01_run1/frames/000.jpg"))
picks.append(("run 5 pic 0", f"{RUNS}/2026-11-11_173549_E_sheet03_run1/frames/000.jpg"))
tiles = []
for label, path in picks:
    picture = cv2.imread(path)
    cv2.putText(picture, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(picture, (320, 0), (320, 479), (0, 255, 0), 1)
    cv2.line(picture, (0, 400), (639, 400), (0, 255, 0), 1)
    tiles.append(picture)
grid = numpy.vstack([numpy.hstack(tiles[0:4]), numpy.hstack(tiles[4:8])])
cv2.imwrite("v05_first_pictures.jpg", cv2.resize(grid, None, fx = 0.6, fy = 0.6), [cv2.IMWRITE_JPEG_QUALITY, 85])
# the bottom strips at full size, stacked, to see the fork's sideways place
strips = [tile[330:480] for tile in tiles]
cv2.imwrite("v05_bottom_strips.jpg", numpy.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved")
