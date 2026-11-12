"""
Verifier: how was the sheet of each robot run turned against the demonstrations' sheets? An estimate that shares
nothing with the analyst's registration but the fitted camera model of 02_code:

  - ONE picture per episode (the first), my own ink mask (dark pixels inside the bright sheet, not red, not the robot);
  - the ink pixels are put onto the table with forearm_camera_model.viewing_rays_in_base and the taught table plane;
  - the demonstration's ink points are turned and shifted until they lie on the run's ink points (brute-force search,
    mean capped distance to the nearest ink cell).

The analyst laid whole puck tracks of the demonstrations onto a line map built from all pictures of the run.
Reads pictures only; nothing connects to the robot.

Run from this folder:  python v12_sheet_placement.py > v12_sheet_placement.out.txt
"""

import math
import os
import sys

import cv2
import numpy
import pandas

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
from config import WORKSPACE_PARAMETERS
from forearm_camera_model import viewing_rays_in_base

cv2.setNumThreads(2)
INDEX = "D:/AI_Project/03_data/processed/v01_training_index/frames.csv"
RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = {5: "2026-11-11_173549_E_sheet03_run1", 9: "2026-11-11_175203_E_sheet03_run2_no_hoodie", 10: "2026-11-11_175544_B_team5_sheet03",
               6: "2026-11-11_174108_E_sheet04_run1"}
PLANS = {5: ["03-01", "03-04", "03-25", "03-27"], 9: ["03-01", "03-04", "03-25", "03-27"], 10: ["03-01", "03-04", "03-25", "03-27"], 6: ["04-01", "04-05", "04-13", "04-17"]}
ANALYST_TURN = {(5, "03-01"): 0.0, (5, "03-04"): 7.0, (5, "03-25"): -9.5, (5, "03-27"): -9.5, (9, "03-01"): 3.0, (9, "03-04"): 8.0, (9, "03-25"): -8.5, (9, "03-27"): -7.0,
                (10, "03-01"): -8.5, (10, "03-04"): 0.5, (10, "03-25"): -18.0, (10, "03-27"): -16.5, (6, "04-01"): 28.5, (6, "04-05"): 29.5, (6, "04-13"): 17.0, (6, "04-17"): 19.0}
JOINTS = [f"joint_{number}" for number in range(1, 7)]
CELL = 0.002
X_RANGE, Y_RANGE = (0.05, 0.60), (-0.40, 0.40)
CAP = 0.02


def ink_points_on_table(picture_path, joints):
    """Ink pixels of one picture as table points (N, 2) in metres."""
    picture = cv2.imread(picture_path)
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    paper = ((value >= 195) & (saturation <= 120)).astype(numpy.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(paper)
    if count < 2:
        return numpy.zeros((0, 2))
    largest = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    big = [label for label in range(1, count) if statistics[label, cv2.CC_STAT_AREA] >= 0.2 * statistics[largest, cv2.CC_STAT_AREA]]
    contours, _ = cv2.findContours(numpy.isin(labels, big).astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = numpy.zeros(paper.shape, numpy.uint8)
    cv2.drawContours(hull, [cv2.convexHull(numpy.vstack(contours))], -1, 1, thickness = -1)
    hull = cv2.erode(hull, numpy.ones((15, 15), numpy.uint8))
    is_red = ((hue <= 12) | (hue >= 168)) & (saturation >= 90)
    is_red = cv2.dilate(is_red.astype(numpy.uint8), numpy.ones((25, 25), numpy.uint8)) > 0
    # the ink: clearly darker than the paper, bluish, inside the sheet; the robot's own parts fill the bottom middle
    is_ink = (hull > 0) & (value <= 185) & (saturation >= 110) & (hue >= 90) & (hue <= 125) & ~is_red
    is_ink[330:, 150:] = False
    is_ink[400:, :] = False
    rows, columns = numpy.nonzero(is_ink[::3, ::3])
    pixels = numpy.column_stack([3 * columns, 3 * rows]).astype(float)
    if not len(pixels):
        return numpy.zeros((0, 2))
    plane = WORKSPACE_PARAMETERS["table_plane"]
    camera, rays, is_valid = viewing_rays_in_base(pixels, list(joints))
    denominator = rays[:, 2] - plane["a"] * rays[:, 0] - plane["b"] * rays[:, 1]
    numerator = plane["a"] * camera[0] + plane["b"] * camera[1] + plane["c"] - camera[2]
    distance = numerator / denominator
    points = camera[None, :2] + distance[:, None] * rays[:, :2]
    keep = is_valid & numpy.isfinite(distance) & (distance > 0)
    return points[keep]


def distance_map(points):
    shape = (int(round((X_RANGE[1] - X_RANGE[0]) / CELL)), int(round((Y_RANGE[1] - Y_RANGE[0]) / CELL)))
    raster = numpy.ones(shape, numpy.uint8)
    cells_x = ((points[:, 0] - X_RANGE[0]) / CELL).astype(int)
    cells_y = ((points[:, 1] - Y_RANGE[0]) / CELL).astype(int)
    inside = (cells_x >= 0) & (cells_x < shape[0]) & (cells_y >= 0) & (cells_y < shape[1])
    raster[cells_x[inside], cells_y[inside]] = 0
    return numpy.minimum(cv2.distanceTransform(raster, cv2.DIST_L2, 5) * CELL, CAP)


def cost(distance, points):
    cells_x = ((points[:, 0] - X_RANGE[0]) / CELL).astype(int)
    cells_y = ((points[:, 1] - Y_RANGE[0]) / CELL).astype(int)
    inside = (cells_x >= 0) & (cells_x < distance.shape[0]) & (cells_y >= 0) & (cells_y < distance.shape[1])
    values = numpy.full(len(points), CAP)
    values[inside] = distance[cells_x[inside], cells_y[inside]]
    return float(values.mean())


def register(moving, fixed_distance):
    """Turn (about the centroid) and shift ``moving`` onto the fixed cloud; returns the best turn, shift and cost."""
    centre = moving.mean(axis = 0)
    best = (math.inf, 0.0, 0.0, 0.0)
    for turn_steps, shift_reach, shift_step, around in [(numpy.arange(-45.0, 45.1, 2.5), 0.08, 0.01, None), (None, 0.012, 0.002, True)]:
        turns = turn_steps if turn_steps is not None else numpy.arange(best[1] - 2.5, best[1] + 2.51, 0.5)
        base_shift = (0.0, 0.0) if around is None else (best[2], best[3])
        offsets = numpy.arange(-shift_reach, shift_reach + shift_step / 2, shift_step)
        for turn in turns:
            angle = math.radians(turn)
            rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            turned = (moving - centre) @ rotation.T + centre
            for shift_x in base_shift[0] + offsets:
                for shift_y in base_shift[1] + offsets:
                    value = cost(fixed_distance, turned + numpy.array([shift_x, shift_y]))
                    if value < best[0]:
                        best = (value, float(turn), float(shift_x), float(shift_y))
    return best


frames = pandas.read_csv(INDEX, low_memory = False, usecols = ["episode_id", "row_in_episode", "plan_number", "frame_path"] + JOINTS)
frames = frames.sort_values(["episode_id", "row_in_episode"])
rows = []
clouds = {}
for number, folder in RUN_FOLDERS.items():
    steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
    run_points = ink_points_on_table(os.path.join(RUNS, folder, steps["frame"].iloc[0]), steps.loc[0, JOINTS].to_numpy(dtype = float))
    run_distance = distance_map(run_points)
    for plan in PLANS[number]:
        if plan not in clouds:
            group = frames[frames["plan_number"] == plan]
            clouds[plan] = ink_points_on_table(os.path.join(RAW, group["frame_path"].iloc[0]), group[JOINTS].iloc[0].to_numpy(dtype = float))
        value, turn, shift_x, shift_y = register(clouds[plan], run_distance)
        rows.append({"run": number, "plan": plan, "ink_points_run": len(run_points), "ink_points_demonstration": len(clouds[plan]), "my_turn_deg": turn,
                     "my_shift_x_cm": 100 * shift_x, "my_shift_y_cm": 100 * shift_y, "mean_capped_distance_mm": 1000 * value, "analyst_turn_deg": ANALYST_TURN[(number, plan)]})
table = pandas.DataFrame(rows)
table["difference_deg"] = table["my_turn_deg"] - table["analyst_turn_deg"]
pandas.set_option("display.width", 250)
print("the demonstration's sheet turned onto the run's sheet (positive = counter-clockwise seen from above); my shift is after a turn about the ink points' centroid, so it is not")
print("comparable with the analyst's shift (turn about the puck's first place)")
print(table.round(2).to_string(index = False))
print()
print("between demonstrations (first pictures, same robot pose): the second laid onto the first")
for first, second in [("03-01", "03-04"), ("03-01", "03-25"), ("03-01", "03-27"), ("03-25", "03-27"), ("04-01", "04-05"), ("04-01", "04-13"), ("04-01", "04-17"), ("04-13", "04-17")]:
    value, turn, shift_x, shift_y = register(clouds[second], distance_map(clouds[first]))
    print(f"   {second} onto {first}: turn {turn:+.1f} degrees, mean capped distance {1000 * value:.1f} mm")
