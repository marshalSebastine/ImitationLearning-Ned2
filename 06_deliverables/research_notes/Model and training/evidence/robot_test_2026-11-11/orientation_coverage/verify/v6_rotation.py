"""
The verifier's own measure of the point a picture turns about when the forearm twists in place.

Own method (the analyst used an ECC alignment): pairs of consecutive recorded pictures in which the tip moved under
0.15 cm while joint 4 changed by 5 degrees or more; ORB key points in the table part of both pictures (rows above
330, the robot's own parts are below), matched and fitted with a rotation-plus-shift (RANSAC); the fixed point of that
motion is the point the picture turns about. Compared with the tip's pixel from the fitted camera and with the
picture's centre. Also: how far a table point moves in the picture when the picture is rotated about its centre
instead of about the tip's pixel, and how much of the table part of a rotated picture is empty.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v6_rotation.py > v6_rotation.out.txt
"""

import math
import sys
from pathlib import Path

import cv2
import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from config import table_height  # noqa: E402
from forearm_camera_model import project_points  # noqa: E402
from side_view_test import magnet_tip_and_lean  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]
TABLE_ROWS = 330

frames = load_training_frames({"is_audit_cleaning": True}).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
grouped = frames.groupby("episode_id", sort = False)
next_row = grouped["row_in_episode"].shift(-1)
moved_cm = 100.0 * numpy.hypot(grouped["tip_x_m"].shift(-1) - frames["tip_x_m"], grouped["tip_y_m"].shift(-1) - frames["tip_y_m"])
twist_deg = numpy.degrees(grouped["joint_4"].shift(-1) - frames["joint_4"])
heading_change = (grouped["picture_heading_deg"].shift(-1) - frames["picture_heading_deg"] + 180.0) % 360.0 - 180.0
is_pair = (next_row == frames["row_in_episode"] + 1) & (moved_cm < 0.15) & (twist_deg.abs() >= 5.0)
candidates = frames[is_pair].copy()
candidates["twist_deg"] = twist_deg[is_pair]
candidates["heading_change_deg"] = heading_change[is_pair]
candidates["next_frame_path"] = grouped["frame_path"].shift(-1)[is_pair]
print(f"pairs with the tip moving under 0.15 cm and joint 4 turning 5 degrees or more: {len(candidates)}")
candidates = candidates.sample(min(220, len(candidates)), random_state = 11)

orb = cv2.ORB_create(nfeatures = 1500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
mask = numpy.zeros((480, 640), numpy.uint8)
mask[:TABLE_ROWS, :] = 255
results = []
for _, row in candidates.iterrows():
    first = cv2.imread(str(RAW_FOLDER / row["frame_path"]), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(RAW_FOLDER / row["next_frame_path"]), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        continue
    points_first, descriptors_first = orb.detectAndCompute(first, mask)
    points_second, descriptors_second = orb.detectAndCompute(second, mask)
    if descriptors_first is None or descriptors_second is None or len(points_first) < 20 or len(points_second) < 20:
        continue
    matches = matcher.match(descriptors_first, descriptors_second)
    if len(matches) < 20:
        continue
    source = numpy.float32([points_first[match.queryIdx].pt for match in matches])
    target = numpy.float32([points_second[match.trainIdx].pt for match in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method = cv2.RANSAC, ransacReprojThreshold = 2.0)
    if matrix is None or inliers.sum() < 15:
        continue
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    # a counter-clockwise turn of the table content on the screen (y down) is a negative atan2 here
    rotation_deg = -math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    linear = matrix[:, :2]
    try:
        fixed = numpy.linalg.solve(numpy.eye(2) - linear, matrix[:, 2])
    except numpy.linalg.LinAlgError:
        continue
    joints = row[JOINT_NAMES].to_numpy(dtype = float)
    tip = magnet_tip_and_lean(list(joints))[0]
    tip_pixel = project_points(numpy.array([tip[0], tip[1], table_height(float(tip[0]), float(tip[1]))]), joints)[0]
    results.append({"twist_deg": row["twist_deg"], "heading_change_deg": row["heading_change_deg"], "rotation_deg": rotation_deg, "scale": scale,
                    "fixed_u": fixed[0], "fixed_v": fixed[1], "tip_u": tip_pixel[0], "tip_v": tip_pixel[1], "inliers": int(inliers.sum())})
table = pandas.DataFrame(results)
table.to_csv(VERIFY_FOLDER / "v6_rotation_pairs.csv", index = False)
good = table[(table["inliers"] >= 25) & (table["rotation_deg"].abs() >= 3.0)]
print(f"pairs fitted: {len(table)}; with 25 inliers or more and a fitted rotation of 3 degrees or more: {len(good)}")
ratio = good["rotation_deg"] / good["heading_change_deg"]
print(f"fitted picture rotation over the calculated heading change: median {ratio.median():+.2f} (25th to 75th percentile {ratio.quantile(0.25):+.2f} to {ratio.quantile(0.75):+.2f});"
      f" the content turns the opposite way to the camera, so -1 is the expected value; scale median {good['scale'].median():.3f}")
print(f"point the picture turns about: u median {good['fixed_u'].median():.0f} (25th to 75th {good['fixed_u'].quantile(0.25):.0f} to {good['fixed_u'].quantile(0.75):.0f}),"
      f" v median {good['fixed_v'].median():.0f} ({good['fixed_v'].quantile(0.25):.0f} to {good['fixed_v'].quantile(0.75):.0f})")
print(f"tip's pixel in the same pictures (fitted camera): u median {good['tip_u'].median():.0f}, v median {good['tip_v'].median():.0f}; picture centre (320, 240)")
from_tip = numpy.hypot(good["fixed_u"] - good["tip_u"], good["fixed_v"] - good["tip_v"])
from_centre = numpy.hypot(good["fixed_u"] - 320.0, good["fixed_v"] - 240.0)
print(f"distance of the fitted fixed point from the tip's pixel: median {from_tip.median():.0f} px; from the picture's centre: median {from_centre.median():.0f} px;"
      f" nearer the tip's pixel than the centre in {(from_tip < from_centre).mean() * 100:.0f} % of pairs")

# plain geometry: rotating about the centre instead of about the tip's pixel moves the tip's own pixel by 2 d sin(angle / 2)
distance = math.hypot(313.0 - 319.5, 371.0 - 239.5)
print(f"\ntip's pixel (313, 371) lies {distance:.0f} px from the picture's centre; rotated about the centre it moves "
      + ", ".join(f"{2 * distance * math.sin(math.radians(angle) / 2):.0f} px at {angle} degrees" for angle in (5, 10, 20, 30, 45)))

# how much of the table part of the picture (rows above 330) is empty after rotating about the tip's pixel
height, width = 480, 640
vs, us = numpy.mgrid[0:TABLE_ROWS, 0:width]
for angle in (5, 10, 20, 30, 45):
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    # where each output pixel comes from
    du, dv = us - 313.0, vs - 371.0
    source_u = 313.0 + cosine * du - sine * dv
    source_v = 371.0 + sine * du + cosine * dv
    outside = (source_u < 0) | (source_u > width - 1) | (source_v < 0) | (source_v > height - 1)
    from_parts = ~outside & (source_v >= TABLE_ROWS)
    print(f"  rotation by {angle} degrees about the tip's pixel: {100 * outside.mean():.1f} % of the table part comes from outside the picture,"
          f" {100 * from_parts.mean():.1f} % from the rows of the robot's parts; together {100 * (outside | from_parts).mean():.1f} %")
