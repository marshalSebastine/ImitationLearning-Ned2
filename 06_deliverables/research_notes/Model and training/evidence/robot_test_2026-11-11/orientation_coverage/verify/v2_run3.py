"""
The verifier's own look at robot run 3 (sheet 02 square, portrait) against the six SQP demonstrations.

Own code, not the analyst's: a plain red-blob finder for the puck and a dark-pixel finder for the line, both put on
the table with the fitted forearm camera (forearm_camera_model.pixels_to_plane). Nothing connects to the robot.

1. What the model asked for in run 3 (steps.csv) and where the asked steps would have taken the tip.
2. The puck's place and the line's direction where it leaves the puck in the first picture of run 3, run 4 and the
   six SQP demonstrations (is it the same lay-out?).
3. In the SQP demonstrations: when the block reached the puck, and how far the forearm had turned by then.
4. How far the puck would be off the line when it is pushed the way the model asked, with 1-cm steps and with the
   asked step lengths (the traced sheet-02 line laid through the puck's start).

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v2_run3.py > v2_run3.out.txt
"""

import sys
from pathlib import Path

import cv2
import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from config import table_height  # noqa: E402
from forearm_camera_model import pixels_to_plane, project_points  # noqa: E402
from setup_visuals import load_sheet_lines  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 400)

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
PUCK_POINT_HEIGHT_M = 0.0145


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


def red_blob(picture_bgr):
    """Return the centroid (u, v) and pixel count of the largest red blob, or None."""
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 168)) & (hsv[:, :, 1] >= 110) & (hsv[:, :, 2] >= 70)).astype(numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    if count <= 1:
        return None
    biggest = 1 + int(numpy.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 300:
        return None
    return float(centroids[biggest, 0]), float(centroids[biggest, 1]), int(stats[biggest, cv2.CC_STAT_AREA])


def puck_on_table(picture_bgr, joints):
    """Return the puck's (x, y) in cm on the table from the red blob's centroid, or None."""
    blob = red_blob(picture_bgr)
    if blob is None:
        return None
    # the blob's centroid is taken as the middle of the puck's silhouette, half the puck's height above the table
    guess = pixels_to_plane(numpy.array([[blob[0], blob[1]]]), joints, PUCK_POINT_HEIGHT_M)[0]
    height = table_height(float(guess[0]), float(guess[1])) + PUCK_POINT_HEIGHT_M
    point = pixels_to_plane(numpy.array([[blob[0], blob[1]]]), joints, height)[0]
    return 100.0 * point


def line_direction_near_puck(picture_bgr, joints, puck_cm, inner_cm = 2.5, outer_cm = 6.0):
    """
    Return the table direction (degrees, 0 = +x, counter-clockwise) from the puck to the dark line pixels that lie
    between ``inner_cm`` and ``outer_cm`` from the puck, their count and the spread of their directions.
    """
    grey = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    # the printed line shows bluish and strongly coloured in this camera (hue about 100 to 106, saturation 150 to 190,
    # sampled by hand in run 3's and plan 02-13's first pictures; paper: saturation 50 to 80; table: hue about 34);
    # rows from 330 down hold the robot's own blue and black parts and are left out
    is_dark = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] >= 110) & (grey < 170)
    is_dark[330:, :] = False
    # keep only the blobs that touch the puck's red blob grown by 12 pixels: the line runs into the cap under the puck,
    # while shadows, the table's edge and the robot's parts do not
    blob = red_blob(picture_bgr)
    red = numpy.zeros(grey.shape, numpy.uint8)
    cv2.circle(red, (int(round(blob[0])), int(round(blob[1]))), int(numpy.sqrt(blob[2] / numpy.pi)) + 12, 1, -1)
    count, labels = cv2.connectedComponents(is_dark.astype(numpy.uint8), connectivity = 8)
    touching = numpy.unique(labels[(red > 0) & (labels > 0)])
    is_dark = numpy.isin(labels, touching) & (labels > 0)
    rows, columns = numpy.nonzero(is_dark)
    if len(rows) == 0:
        return None
    pixels = numpy.column_stack([columns, rows]).astype(float)
    points = 100.0 * pixels_to_plane(pixels, joints, table_height(puck_cm[0] / 100.0, puck_cm[1] / 100.0))
    offset = points - numpy.asarray(puck_cm)[None, :]
    distance = numpy.hypot(offset[:, 0], offset[:, 1])
    is_ring = (distance >= inner_cm) & (distance <= outer_cm)
    if is_ring.sum() < 30:
        return None
    angles = numpy.arctan2(offset[is_ring, 1], offset[is_ring, 0])
    mean = numpy.degrees(numpy.arctan2(numpy.sin(angles).mean(), numpy.cos(angles).mean()))
    spread = float(numpy.degrees(numpy.sqrt(-2.0 * numpy.log(max(numpy.hypot(numpy.sin(angles).mean(), numpy.cos(angles).mean()), 1e-9)))))
    return float(mean), int(is_ring.sum()), spread


# ---- 1. what the model asked for in run 3 -------------------------------------------------------------------------
print("=== 1. Run 3 (2026-11-11_172743_E_sheet02_run1): what the model asked for (steps.csv, measured) ===")
steps = pandas.read_csv(RUNS_FOLDER / "2026-11-11_172743_E_sheet02_run1" / "steps.csv")
steps["asked_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
steps["asked_in_picture_deg"] = numpy.degrees(numpy.arctan2(steps["model_tip_left_cm"], steps["model_tip_forward_cm"]))
steps["asked_on_table_deg"] = wrap(steps["picture_heading_deg"] + steps["asked_in_picture_deg"])
steps["joint_4_deg"] = numpy.degrees(steps["joint_4"])
steps["done_cm"] = 100.0 * numpy.hypot(steps["tip_next_x"] - steps["tip_x"], steps["tip_next_y"] - steps["tip_y"])
steps["done_on_table_deg"] = numpy.degrees(numpy.arctan2(steps["tip_next_y"] - steps["tip_y"], steps["tip_next_x"] - steps["tip_x"]))
first = steps.head(10)
print(first[["asked_cm", "asked_in_picture_deg", "asked_on_table_deg", "model_twist_deg", "joint_4_deg", "picture_heading_deg", "done_cm", "done_on_table_deg"]].round(2).to_string())
print(f"first 10 pictures: asked step median {first['asked_cm'].median():.2f} cm (least {first['asked_cm'].min():.2f}, most {first['asked_cm'].max():.2f}),"
      f" asked twist median {first['model_twist_deg'].median():.2f} degrees, sum {first['model_twist_deg'].sum():.1f};"
      f" asked travel sum {first['asked_cm'].sum():.2f} cm; done {first['done_cm'].sum():.2f} cm")
print(f"  asked direction in the picture: median {first['asked_in_picture_deg'].median():.1f} degrees (least {first['asked_in_picture_deg'].min():.1f}, most {first['asked_in_picture_deg'].max():.1f});"
      f" on the table: median {first['asked_on_table_deg'].median():.1f}")
# where the asked steps would have taken the tip (the logged answers and headings; in a real run the pictures would differ)
heading = numpy.radians(first["picture_heading_deg"].to_numpy())
forward = first["model_tip_forward_cm"].to_numpy()
left = first["model_tip_left_cm"].to_numpy()
asked_dx = (forward * numpy.cos(heading) - left * numpy.sin(heading))
asked_dy = (forward * numpy.sin(heading) + left * numpy.cos(heading))
print(f"  net of the ten asked steps on the table: {asked_dx.sum():.2f} cm away from the robot, {asked_dy.sum():+.2f} cm to its left"
      f" (direction {numpy.degrees(numpy.arctan2(asked_dy.sum(), asked_dx.sum())):.1f} degrees, length {numpy.hypot(asked_dx.sum(), asked_dy.sum()):.2f} cm)")
done_dx = 100.0 * (first["tip_next_x"] - first["tip_x"]).sum()
done_dy = 100.0 * (first["tip_next_y"] - first["tip_y"]).sum()
print(f"  net of the ten steps done: {done_dx:.2f} cm away, {done_dy:+.2f} cm to the left (direction {numpy.degrees(numpy.arctan2(done_dy, done_dx)):.1f} degrees)")
twist_30 = int(numpy.argmax(numpy.abs(steps["joint_4_deg"] - steps["joint_4_deg"].iloc[0]) >= 30.0))
print(f"  joint 4 had turned 30 degrees at picture {twist_30} (after {steps['done_cm'].iloc[:twist_30].sum():.1f} cm done; the asked steps up to there sum to {steps['asked_cm'].iloc[:twist_30].sum():.2f} cm)")

# ---- 2. the lay-out in the first pictures --------------------------------------------------------------------------
print("\n=== 2. Puck and line in first pictures (own red-blob and dark-pixel finders, fitted camera) ===")
raw = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False).sort_values(["episode_id", "row_in_episode"])
joint_names = [f"joint_{number}" for number in range(1, 7)]
rows = []
for name, folder in (("run 3", "2026-11-11_172743_E_sheet02_run1"), ("run 4", "2026-11-11_173036_E_sheet02_run2")):
    run_steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    joints = run_steps.loc[0, joint_names].to_numpy(dtype = float)
    picture = cv2.imread(str(RUNS_FOLDER / folder / "frames" / "000.jpg"))
    puck = puck_on_table(picture, joints)
    tip = 100.0 * run_steps.loc[0, ["tip_x", "tip_y"]].to_numpy(dtype = float)
    line = line_direction_near_puck(picture, joints, puck)
    rows.append({"what": name, "tip_x": tip[0], "tip_y": tip[1], "puck_x": puck[0], "puck_y": puck[1], "puck_from_tip_cm": float(numpy.hypot(*(puck - tip))),
                 "line_from_puck_deg": line[0] if line else numpy.nan, "line_pixels": line[1] if line else 0, "spread_deg": line[2] if line else numpy.nan,
                 "joint_4_deg": float(numpy.degrees(joints[3]))})
sheet_02 = raw[raw["sheet"] == "path_02"]
for plan in ["02-13", "02-14", "02-15", "02-16", "02-17", "02-18", "02-36", "02-37", "02-38", "02-39", "02-40", "02-41", "02-42", "02-43", "02-44", "02-45", "02-01", "02-04", "02-31"]:
    episode = sheet_02[sheet_02["plan_number"] == plan]
    if len(episode) == 0:
        continue
    row = episode.iloc[0]
    joints = row[joint_names].to_numpy(dtype = float)
    picture = cv2.imread(str(RAW_FOLDER / row["frame_path"]))
    puck = puck_on_table(picture, joints)
    tip = 100.0 * row[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    line = line_direction_near_puck(picture, joints, puck) if puck is not None else None
    rows.append({"what": f"plan {plan} ({row['start_kind']})", "tip_x": tip[0], "tip_y": tip[1],
                 "puck_x": puck[0] if puck is not None else numpy.nan, "puck_y": puck[1] if puck is not None else numpy.nan,
                 "puck_from_tip_cm": float(numpy.hypot(*(puck - tip))) if puck is not None else numpy.nan,
                 "line_from_puck_deg": line[0] if line else numpy.nan, "line_pixels": line[1] if line else 0, "spread_deg": line[2] if line else numpy.nan,
                 "joint_4_deg": float(numpy.degrees(joints[3]))})
print(pandas.DataFrame(rows).round(1).to_string(index = False))

# ---- 3. the SQP demonstrations: when the block reached the puck -----------------------------------------------------
print("\n=== 3. The six SQP demonstrations, row by row: tip, puck, distance, forearm (raw index, rows 0 to 20) ===")
summary = []
for plan in ["02-13", "02-14", "02-15", "02-16", "02-17", "02-18"]:
    episode = sheet_02[sheet_02["plan_number"] == plan].head(24)
    puck_start = None
    lines = []
    for _, row in episode.iterrows():
        joints = row[joint_names].to_numpy(dtype = float)
        picture = cv2.imread(str(RAW_FOLDER / row["frame_path"]))
        puck = puck_on_table(picture, joints)
        tip = 100.0 * row[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
        if puck is None:
            continue
        if puck_start is None:
            puck_start = puck
        lines.append({"row": int(row["row_in_episode"]), "tip_x": tip[0], "tip_y": tip[1], "puck_x": puck[0], "puck_y": puck[1],
                      "puck_from_tip_cm": float(numpy.hypot(*(puck - tip))), "puck_moved_cm": float(numpy.hypot(*(puck - puck_start))),
                      "joint_4_deg": float(numpy.degrees(joints[3]))})
    table = pandas.DataFrame(lines)
    # the first row at which the puck has clearly moved from where it lay (1 cm; the camera fit is good to a few mm near the tip)
    moved = table[table["puck_moved_cm"] >= 1.0]
    first_moved = moved.iloc[0] if len(moved) else None
    before = table[table["row"] < first_moved["row"]].iloc[-1] if first_moved is not None else table.iloc[-1]
    travel = float(numpy.hypot(numpy.diff(table.loc[table["row"] <= before["row"], "tip_x"]), numpy.diff(table.loc[table["row"] <= before["row"], "tip_y"])).sum())
    summary.append({"plan": plan, "puck_from_tip_at_start_cm": table.iloc[0]["puck_from_tip_cm"], "last_row_before_the_puck_moved": int(before["row"]),
                    "tip_travel_until_then_cm": travel, "joint_4_then_deg": before["joint_4_deg"], "tip_x_then": before["tip_x"], "tip_y_then": before["tip_y"],
                    "puck_x_start": table.iloc[0]["puck_x"], "puck_y_start": table.iloc[0]["puck_y"]})
    print(f"--- plan {plan} ---")
    print(table.round(1).to_string(index = False))
print("\nSummary (the puck 'moved' = 1 cm or more from where the first picture shows it):")
print(pandas.DataFrame(summary).round(1).to_string(index = False))

# ---- 4. how far off the line the puck gets when pushed as the model asked -----------------------------------------
print("\n=== 4. Run 3: the puck against the line, as done (1-cm steps) and with the asked step lengths (calculated) ===")
run_joints = steps.loc[0, joint_names].to_numpy(dtype = float)
run_picture = cv2.imread(str(RUNS_FOLDER / "2026-11-11_172743_E_sheet02_run1" / "frames" / "000.jpg"))
puck_start = puck_on_table(run_picture, run_joints)
sheet_line = load_sheet_lines()["02"]["line"][::-1]          # from end B
# lay the traced line: end B on the puck's start, turned so that it leaves the puck as measured in the picture
measured = line_direction_near_puck(run_picture, run_joints, puck_start)[0]
offset = sheet_line - sheet_line[0][None, :]
distance_along = numpy.hypot(offset[:, 0], offset[:, 1])
is_ring = (distance_along >= 2.5) & (distance_along <= 6.0) & (numpy.arange(len(sheet_line)) < 100)
own_direction = numpy.degrees(numpy.arctan2(offset[is_ring, 1].mean(), offset[is_ring, 0].mean()))
turn = numpy.radians(measured - own_direction)
rotation = numpy.array([[numpy.cos(turn), -numpy.sin(turn)], [numpy.sin(turn), numpy.cos(turn)]])
table_line = offset @ rotation.T + puck_start[None, :]
start_tangent = numpy.degrees(numpy.arctan2(*(table_line[5] - table_line[0])[::-1]))
print(f"puck at the start ({puck_start[0]:.1f}, {puck_start[1]:.1f}) cm; the line leaves it at {measured:.0f} degrees over 2.5 to 6 cm (measured in the picture);"
      f" laid traced line: sheet turn {numpy.degrees(turn):.0f} degrees, tangent at the cap {start_tangent:.0f} degrees")


def off_line_cm(point):
    return float(numpy.min(numpy.hypot(table_line[:, 0] - point[0], table_line[:, 1] - point[1])))


def pushed_puck(tips):
    """The puck as the pictures of run 3 show it: touching the block from the start, it is carried along by the tip's own displacement."""
    return puck_start[None, :] + (tips - tips[0][None, :])


tips_done = 100.0 * numpy.column_stack([numpy.r_[steps["tip_x"].to_numpy(), steps["tip_next_x"].iloc[-1]], numpy.r_[steps["tip_y"].to_numpy(), steps["tip_next_y"].iloc[-1]]])
heading_all = numpy.radians(steps["picture_heading_deg"].to_numpy())
asked_x = steps["model_tip_forward_cm"].to_numpy() * numpy.cos(heading_all) - steps["model_tip_left_cm"].to_numpy() * numpy.sin(heading_all)
asked_y = steps["model_tip_forward_cm"].to_numpy() * numpy.sin(heading_all) + steps["model_tip_left_cm"].to_numpy() * numpy.cos(heading_all)
tips_asked = tips_done[0][None, :] + numpy.vstack([[0.0, 0.0], numpy.cumsum(numpy.column_stack([asked_x, asked_y]), axis = 0)])
pucks_done = pushed_puck(tips_done)
pucks_asked = pushed_puck(tips_asked)
print(" picture  tip_done            puck_off_line_done_cm  tip_with_asked_lengths  puck_off_line_asked_cm  forearm_turned_deg")
for index in range(0, 14):
    print(f"   {index:2d}     ({tips_done[index, 0]:5.1f}, {tips_done[index, 1]:5.1f})        {off_line_cm(pucks_done[index]):5.2f}              ({tips_asked[index, 0]:5.1f}, {tips_asked[index, 1]:5.1f})           {off_line_cm(pucks_asked[index]):5.2f}"
          f"              {steps['joint_4_deg'].iloc[min(index, len(steps) - 1)] - steps['joint_4_deg'].iloc[0]:6.1f}")
first_off_done = int(numpy.argmax(numpy.array([off_line_cm(point) for point in pucks_done]) >= 1.2))
first_off_asked = int(numpy.argmax(numpy.array([off_line_cm(point) for point in pucks_asked]) >= 1.2))
print(f"puck's centre 1.2 cm (half a puck) or more off the line: from picture {first_off_done} on as done, from picture {first_off_asked} on with the asked lengths"
      f" (the logged answers; a real run would see other pictures)")
# check the crude puck against the pictures: where the red blob really was
print("check of the carried puck against the pictures (red blob on the table, as done):")
for index in (0, 3, 5, 8, 10):
    joints = steps.loc[index, joint_names].to_numpy(dtype = float)
    picture = cv2.imread(str(RUNS_FOLDER / "2026-11-11_172743_E_sheet02_run1" / "frames" / f"{index:03d}.jpg"))
    seen = puck_on_table(picture, joints)
    print(f"  picture {index}: puck seen at ({seen[0]:.1f}, {seen[1]:.1f}), {off_line_cm(seen):.2f} cm off the laid line; carried puck ({pucks_done[index, 0]:.1f}, {pucks_done[index, 1]:.1f}), {off_line_cm(pucks_done[index]):.2f} cm off")
