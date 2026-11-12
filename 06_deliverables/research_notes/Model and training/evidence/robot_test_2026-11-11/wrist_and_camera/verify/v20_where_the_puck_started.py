"""
Verifier, check 20: did the sheets lie where they lay in the recordings? The puck stands on the line's start cap in the
first picture of every run and of every start-mark demonstration, so the puck's place in the robot's frame says where
the start cap was. Measured on the pictures: the red puck is found by colour, the middle of its silhouette is turned
into a table point with the project's fitted camera (forearm_camera_model.pixels_to_plane at the table's height plus
14.5 mm, as the project's puck tracking does), using the logged joints of that picture. The same measure on both
sides, so a bias of the method cancels in the difference.

Also: the puck's width, from the silhouette's left and right ends at the height of its top face.
    D:/AI_Project/.venv/Scripts/python.exe v20_where_the_puck_started.py
"""

import glob
import math
import sys

import cv2
import numpy
import pandas

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
import forearm_camera_model
from config import table_height

RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"
JOINTS = [f"joint_{index}" for index in range(1, 7)]
RUN_PLANS = {"172021_check_no_puck": ("path_01", "01-01"), "172421_E_sheet01_run1": ("path_01", "01-01"), "172743_E_sheet02_run1": ("path_02", None),
             "173036_E_sheet02_run2": ("path_02", "02-01"), "173549_E_sheet03_run1": ("path_03", "03-01"), "174108_E_sheet04_run1": ("path_04", "04-01"),
             "174448_E_sheet05_run1": ("path_05", "05-01"), "174936_E_sheet06_run1": ("path_06", "06-01"), "175203_E_sheet03_run2_no_hoodie": ("path_03", "03-01"),
             "175544_B_team5_sheet03": ("path_03", "03-01")}


def puck_in_picture(path: str):
    """The red puck's silhouette: centre (u, v), left and right ends (u), and its area in px; None when not found."""

    picture = cv2.imdecode(numpy.fromfile(path, dtype = numpy.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 70)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(red.astype(numpy.uint8), connectivity = 8)
    if count < 2:
        return None
    biggest = 1 + int(numpy.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 400:
        return None
    left = stats[biggest, cv2.CC_STAT_LEFT]
    return {"u": float(centres[biggest][0]), "v": float(centres[biggest][1]), "left": float(left), "right": float(left + stats[biggest, cv2.CC_STAT_WIDTH] - 1),
            "top": float(stats[biggest, cv2.CC_STAT_TOP]), "area": int(stats[biggest, cv2.CC_STAT_AREA])}


def puck_on_table(path: str, joints: list):
    found = puck_in_picture(path)
    if found is None:
        return None
    guess = forearm_camera_model.pixels_to_plane(numpy.array([[found["u"], found["v"]]]), joints, 0.0145)[0]
    height = table_height(float(guess[0]), float(guess[1]))
    centre = forearm_camera_model.pixels_to_plane(numpy.array([[found["u"], found["v"]]]), joints, height + 0.0145)[0]
    ends = forearm_camera_model.pixels_to_plane(numpy.array([[found["left"], found["top"] + 0.35 * (found["v"] - found["top"])],
                                                             [found["right"], found["top"] + 0.35 * (found["v"] - found["top"])]]), joints, height + 0.029)
    return {"x_cm": 100 * float(centre[0]), "y_cm": 100 * float(centre[1]), "width_cm": 100 * float(numpy.linalg.norm(ends[1] - ends[0])), "width_px": found["right"] - found["left"] + 1}


frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "frame_path", "tip_x_m", "tip_y_m"] + JOINTS)
corrected = pandas.read_csv("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv")
first = frames[frames["row_in_episode"] == 0].merge(corrected[["episode_id", "plan", "corrected_end", "start_kind"]], on = "episode_id")
first = first[first["sheet_orientation"].str.contains("START MARK", na = False) & (first["start_kind"] == "STD")]
rows = []
for _, row in first.iterrows():
    place = puck_on_table(f"{RAW}/{row['frame_path']}", row[JOINTS].to_numpy(dtype = float).tolist())
    if place is not None:
        rows.append({"sheet": row["sheet"], "end": row["corrected_end"], "plan": row["plan"], **place})
demos = pandas.DataFrame(rows)
pandas.set_option("display.width", 220)
print("the puck at the start of the STD start-mark demonstrations (first picture; tip at 14.0, -1.5 cm in all of them):")
print(demos.groupby(["sheet", "end"]).agg(episodes = ("x_cm", "size"), x_median = ("x_cm", "median"), x_least = ("x_cm", "min"), x_most = ("x_cm", "max"),
                                          y_median = ("y_cm", "median"), y_least = ("y_cm", "min"), y_most = ("y_cm", "max"), width_cm = ("width_cm", "median")).round(1).to_string())
print("all STD demonstrations: puck x median %.1f (5th %.1f, 95th %.1f), y median %.1f (5th %.1f, 95th %.1f) cm; puck width median %.1f cm (%.0f px at 640)" % (
    demos["x_cm"].median(), demos["x_cm"].quantile(0.05), demos["x_cm"].quantile(0.95), demos["y_cm"].median(), demos["y_cm"].quantile(0.05), demos["y_cm"].quantile(0.95),
    demos["width_cm"].median(), demos["width_px"].median()))

print("\nthe puck in the first picture of each robot run, against the demonstrations of the same sheet (and end, where the plan is known):")
for folder in sorted(glob.glob(f"{RUNS}/2026-11-11_17*")):
    name = folder.replace("\\", "/").split("/")[-1][11:]
    sheet, plan = RUN_PLANS[name]
    step = pandas.read_csv(f"{folder}/steps.csv").iloc[0]
    place = puck_on_table(f"{folder}/frames/000.jpg", [step[joint] for joint in JOINTS])
    same = demos[demos["sheet"] == sheet]
    if plan is not None:
        same = same[same["end"] == corrected.loc[corrected["plan"] == plan, "corrected_end"].iloc[0]]
    if place is None:
        print("  %-34s puck not found" % name)
        continue
    print("  %-34s puck at %5.1f, %5.1f cm (tip %4.1f, %4.1f; puck ahead of the tip by %.1f cm); demonstrations: %5.1f, %5.1f; the run's puck lies %+.1f cm further out and %+.1f cm to the left (gap %.1f cm)" % (
        name, place["x_cm"], place["y_cm"], 100 * step["tip_x"], 100 * step["tip_y"], math.hypot(place["x_cm"] - 100 * step["tip_x"], place["y_cm"] - 100 * step["tip_y"]),
        same["x_cm"].median(), same["y_cm"].median(), place["x_cm"] - same["x_cm"].median(), place["y_cm"] - same["y_cm"].median(),
        math.hypot(place["x_cm"] - same["x_cm"].median(), place["y_cm"] - same["y_cm"].median())))
