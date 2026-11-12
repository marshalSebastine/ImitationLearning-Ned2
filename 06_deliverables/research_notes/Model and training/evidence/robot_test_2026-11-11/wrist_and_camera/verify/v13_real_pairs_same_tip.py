"""
Verifier, check 13: the calculated view shift against REAL picture pairs that differ as the driver's view differs from
a demonstration: two different episodes of the same sheet, same placement (start mark, kind STD, same corrected end),
the tip at the same point (within 3 mm), the forearm twisted alike (within 1.5 degrees), but the wrist different by 8
degrees or more. The table content (paper, line) of the two real pictures is matched by normalised cross-correlation
on the picture's upper part, and its measured move is compared with the move the arm model and the fitted camera
calculate for a table point 5 cm ahead of the tip. This tests the 2.8 px per degree on real pictures of the same
place, which the analyst's check (neighbouring pictures of one episode, where the camera hardly moved) does not.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe v13_real_pairs_same_tip.py
"""

import math
import sys

import cv2
import numpy
import pandas

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
import forearm_camera_model
from config import table_height

RAW = "D:/AI_Project/03_data/raw/main_data"
CORRECTED_ENDS = "D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv"
JOINTS = [f"joint_{index}" for index in range(1, 7)]
REGION = (slice(70, 250), slice(110, 530))      # rows, columns of the reference patch (table ahead of the block)
SEARCH = 70                                     # px of search room up and down
SIDE = 40                                       # px of search room left and right

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "tip_x_m", "tip_y_m", "tip_z_m", "frame_path", "started"] + JOINTS)
corrected = pandas.read_csv(CORRECTED_ENDS)[["episode_id", "corrected_end", "start_kind"]]
frames = frames.merge(corrected, on = "episode_id", how = "left")
frames = frames[frames["sheet_orientation"].str.contains("START MARK", na = False) & (frames["start_kind"] == "STD")].reset_index(drop = True)
frames["wrist"] = numpy.degrees(frames["joint_5"])
frames["twist"] = numpy.degrees(frames["joint_4"])
frames["is_after_step"] = frames["started"] >= "2026-11-08T19:33:39"

generator = numpy.random.default_rng(13)
pairs = []
for (sheet, end, after), group in frames.groupby(["sheet", "corrected_end", "is_after_step"]):
    group = group.reset_index(drop = True)
    xy = group[["tip_x_m", "tip_y_m"]].to_numpy()
    far_wrist = group.index[(group["wrist"] - 13.5).abs() > 8.0]
    found = []
    for position in generator.permutation(far_wrist):
        row = group.loc[position]
        distance = numpy.hypot(xy[:, 0] - row["tip_x_m"], xy[:, 1] - row["tip_y_m"])
        candidates = group[(distance < 0.003) & ((group["twist"] - row["twist"]).abs() < 1.5) & ((group["wrist"] - 13.5).abs() < 1.0)
                           & (group["episode_id"] != row["episode_id"]) & ((group["tip_z_m"] - row["tip_z_m"]).abs() < 0.004)]
        if len(candidates):
            other = candidates.iloc[int(generator.integers(len(candidates)))]
            found.append((row, other))
        if len(found) >= 14:
            break
    pairs.extend(found)
print("pairs found:", len(pairs))

rows = []
for far, near in pairs:
    joints_far, joints_near = far[JOINTS].to_numpy(dtype = float).tolist(), near[JOINTS].to_numpy(dtype = float).tolist()
    # a table point 5 cm ahead of the tip, along the viewing direction of the far picture, on the fitted table
    centre, axes = forearm_camera_model.camera_pose_in_base(joints_far)
    ahead = numpy.array([far["tip_x_m"], far["tip_y_m"], 0.0]) + 0.05 * numpy.array([axes[0, 2], axes[1, 2], 0.0]) / math.hypot(axes[0, 2], axes[1, 2])
    ahead[2] = table_height(ahead[0], ahead[1])
    pixel_far = forearm_camera_model.project_points(ahead, joints_far)[0]
    pixel_near = forearm_camera_model.project_points(ahead, joints_near)[0]
    calculated = pixel_near - pixel_far                    # where the table point moves from the far-wrist picture to the 13.5 picture
    picture_far = cv2.cvtColor(cv2.imread(f"{RAW}/{far['frame_path']}"), cv2.COLOR_BGR2GRAY)
    picture_near = cv2.cvtColor(cv2.imread(f"{RAW}/{near['frame_path']}"), cv2.COLOR_BGR2GRAY)
    template = numpy.ascontiguousarray(picture_far[REGION])
    top, bottom = max(0, REGION[0].start - SEARCH), min(480, REGION[0].stop + SEARCH)
    left, right = REGION[1].start - SIDE, REGION[1].stop + SIDE
    scores = cv2.matchTemplate(numpy.ascontiguousarray(picture_near[top:bottom, left:right]), template, cv2.TM_CCOEFF_NORMED)
    _, best, _, location = cv2.minMaxLoc(scores)
    measured = numpy.array([location[0] + left - REGION[1].start, location[1] + top - REGION[0].start], dtype = float)
    rows.append({"sheet": far["sheet"], "wrist_far": far["wrist"], "wrist_near": near["wrist"], "wrist_difference": near["wrist"] - far["wrist"], "twist": far["twist"],
                 "calculated_du": calculated[0], "calculated_dv": calculated[1], "measured_du": measured[0], "measured_dv": measured[1], "score": best,
                 "at_search_edge": bool(location[1] in (0, scores.shape[0] - 1) or location[0] in (0, scores.shape[1] - 1)),
                 "far_picture": far["frame_path"], "near_picture": near["frame_path"]})
table = pandas.DataFrame(rows)
table.to_csv("v13_real_pairs_same_tip.csv", index = False)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
print(table.drop(columns = ["far_picture", "near_picture"]).round(2).to_string(index = False))
good = table[(table["score"] >= 0.6) & ~table["at_search_edge"]]
print("\npairs with a match score of 0.6 or more, away from the search edge: %d of %d" % (len(good), len(table)))
if len(good) >= 5:
    slope = numpy.polyfit(good["calculated_dv"], good["measured_dv"], 1)
    print("measured against calculated move down the picture: slope %.3f, intercept %+.1f px, correlation %.3f; measured minus calculated: median %+.1f, median |x| %.1f px" % (
        slope[0], slope[1], numpy.corrcoef(good["calculated_dv"], good["measured_dv"])[0, 1], (good["measured_dv"] - good["calculated_dv"]).median(),
        (good["measured_dv"] - good["calculated_dv"]).abs().median()))
    per_degree_measured = numpy.polyfit(good["wrist_difference"], good["measured_dv"], 1)[0]
    per_degree_calculated = numpy.polyfit(good["wrist_difference"], good["calculated_dv"], 1)[0]
    print("px (at 640) down the picture per degree of (13.5-view wrist minus far wrist): measured %.2f, calculated %.2f" % (per_degree_measured, per_degree_calculated))
    print("sideways: measured median |x| %.1f px, calculated median |x| %.1f px" % (good["measured_du"].abs().median(), good["calculated_du"].abs().median()))

# robust summary: the ratio of measured to calculated move per pair (the dashed sheet 06 can lock onto the wrong dash)
ratio = good["measured_dv"] / good["calculated_dv"]
print("\nmeasured / calculated move down the picture per pair: median %.2f, 25th %.2f, 75th %.2f; same sign in %d of %d pairs; without sheet 06: median %.2f, same sign in %d of %d" % (
    ratio.median(), ratio.quantile(0.25), ratio.quantile(0.75), int((ratio > 0).sum()), len(ratio),
    ratio[good["sheet"] != "path_06"].median(), int((ratio[good["sheet"] != "path_06"] > 0).sum()), int((good["sheet"] != "path_06").sum())))
per_degree = (good["measured_dv"] / good["wrist_difference"])[good["sheet"] != "path_06"]
print("measured px (at 640) per degree of wrist, per pair, without sheet 06: median %.2f (calculated for the same pairs: %.2f)" % (
    per_degree.median(), (good["calculated_dv"] / good["wrist_difference"])[good["sheet"] != "path_06"].median()))
