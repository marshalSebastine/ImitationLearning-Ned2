"""
The verifier's own check of the physics claims (nothing connects to the robot; only the arm model and the index).

1. How far the picture's top can be turned from the radial direction with joint 4 at 117 degrees and the wrist held at
   13.5 degrees, by reach: once with the recording tool's heading rule (forearm_camera_heading_rad) and once, as an
   independent route, with the fitted camera (which table direction runs straight up the picture at the tip's pixel).
2. The rod's lean by reach with the forearm straight and fully twisted (the driver refuses above 58 degrees).
3. The tip's pixel in the picture.
4. What the demonstrations show: the picture's top from radial where |joint 4| is 100 or more; the lean and reach of
   the rows that push back towards the base; how many pictures lie near the base with the forearm turned far (the
   state a pre-turned start would begin in).

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v4_physics.py > v4_physics.out.txt
"""

import math
import sys
from pathlib import Path

import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height, table_height  # noqa: E402
from forearm_camera_model import project_points  # noqa: E402
from record_episodes import forearm_camera_heading_rad  # noqa: E402  (a pure function; nothing connects at import)
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 400)


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


def joints_for(reach_cm, joint_4_deg, bearing_deg = 0.0):
    """The driver's joints for a tip at ``reach_cm`` on the bearing, joint 4 as given, wrist held; None when unreachable."""
    x = reach_cm / 100.0 * math.cos(math.radians(bearing_deg))
    y = reach_cm / 100.0 * math.sin(math.radians(bearing_deg))
    target = numpy.array([x, y, pushing_height(x, y) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    for start in ([math.atan2(y, x), -0.8, -0.6], [math.atan2(y, x), -0.3, -0.9], [math.atan2(y, x), -1.0, 0.0], [math.atan2(y, x), 0.2, -1.2]):
        joints = solve_with_wrist_fixed(target, math.radians(joint_4_deg), math.radians(MODEL_MODE_PARAMETERS["wrist_held_deg"]), start)
        if joints is not None:
            joints[5] = math.radians(MODEL_MODE_PARAMETERS["hand_held_deg"])
            return joints
    return None


def camera_heading_deg(joints):
    """The table direction that runs straight up the picture at the tip's pixel, from the fitted camera alone."""
    tip = magnet_tip_and_lean(list(joints))[0]
    height = table_height(float(tip[0]), float(tip[1]))
    centre = numpy.array([tip[0], tip[1], height])
    pixel_centre = project_points(centre, joints)[0]
    best, best_score = None, -1e9
    for angle in numpy.arange(-180.0, 180.0, 0.5):
        point = centre + 0.02 * numpy.array([math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0])
        shift = project_points(point, joints)[0] - pixel_centre
        if not numpy.all(numpy.isfinite(shift)):
            continue
        score = -shift[1] / (numpy.hypot(*shift) + 1e-9)          # 1 when the shift is straight up the picture
        if score > best_score:
            best, best_score = angle, score
    return best, pixel_centre


print("=== 1 to 3. Arm model: picture's top from radial, rod lean and the tip's pixel (calculated; wrist held at 13.5 degrees) ===")
rows = []
for reach in [12.0, 15.0, 18.0, 21.0, 25.0, 30.0, 35.0, 40.0, 45.0, 47.0, 47.5, 48.0, 49.0, 49.5]:
    row = {"reach_cm": reach}
    for joint_4 in (-117.0, 0.0, 117.0):
        joints = joints_for(reach, joint_4)
        if joints is None:
            row[f"tool_heading_j4_{joint_4:+.0f}"] = numpy.nan
            continue
        tool_heading = math.degrees(forearm_camera_heading_rad(list(joints)))
        camera_heading, pixel = camera_heading_deg(joints)
        lean = magnet_tip_and_lean(joints)[1]
        row[f"tool_heading_j4_{joint_4:+.0f}"] = float(wrap(tool_heading))
        row[f"camera_heading_j4_{joint_4:+.0f}"] = camera_heading
        row[f"lean_j4_{joint_4:+.0f}"] = lean
        if joint_4 == 0.0:
            row["tip_pixel_u"], row["tip_pixel_v"] = pixel
    rows.append(row)
table = pandas.DataFrame(rows)
print(table.round(1).to_string(index = False))
limit = MODEL_MODE_PARAMETERS["rod_lean_most_deg"]
straight = table[["reach_cm", "lean_j4_+0"]].dropna()
over = straight[straight["lean_j4_+0"] > limit]
print(f"rod lean limit of the driver: {limit}; with the forearm straight the lean first passes it at reach {over['reach_cm'].min() if len(over) else 'never'} cm")
fine = numpy.arange(45.0, 49.6, 0.1)
leans = [(reach, magnet_tip_and_lean(joints_for(reach, 0.0))[1]) for reach in fine if joints_for(reach, 0.0) is not None]
first_over = next((reach for reach, lean in leans if lean > limit), None)
print(f"  finer: the lean passes {limit} degrees at reach {first_over:.1f} cm (forearm straight)")
leans_twisted = [(reach, magnet_tip_and_lean(joints_for(reach, 117.0))[1]) for reach in fine if joints_for(reach, 117.0) is not None]
first_over_twisted = next((reach for reach, lean in leans_twisted if lean > limit), None)
print(f"  with joint 4 at 117 degrees the lean passes it at reach {first_over_twisted if first_over_twisted is None else round(first_over_twisted, 1)} cm (None = not within 49.5 cm)")

print("\n=== 4. The demonstrations (measured from the audit-cleaned index) ===")
frames = load_training_frames({"is_audit_cleaning": True})
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
joint_4 = numpy.degrees(frames["joint_4"].to_numpy())
x = 100.0 * frames["tip_x_m"].to_numpy()
y = 100.0 * frames["tip_y_m"].to_numpy()
reach = numpy.hypot(x, y)
radial = numpy.degrees(numpy.arctan2(y, x))
heading_from_radial = wrap(frames["picture_heading_deg"].to_numpy() - radial)
far_twist = numpy.abs(joint_4) >= 100.0
print(f"|joint 4| of 100 or more: {far_twist.sum()} pictures; |picture top from radial| median {numpy.median(numpy.abs(heading_from_radial[far_twist])):.1f}, largest {numpy.abs(heading_from_radial).max():.1f}")
slope = numpy.polyfit(joint_4, heading_from_radial, 1)[0]
print(f"least-squares slope of the picture's turn from radial against joint 4: {slope:.3f}")
# the index's own heading against the fitted camera's on a sample of rows (two routes to the same quantity)
sample = frames.sample(150, random_state = 3)
differences = []
for _, row in sample.iterrows():
    joints = row[[f"joint_{number}" for number in range(1, 7)]].to_numpy(dtype = float)
    camera_heading, _ = camera_heading_deg(joints)
    differences.append(float(wrap(camera_heading - row["picture_heading_deg"])))
differences = numpy.array(differences)
print(f"fitted camera's 'straight up at the tip' minus the index's picture_heading_deg on 150 random rows: median {numpy.median(differences):+.1f} degrees,"
      f" 10th to 90th percentile {numpy.percentile(differences, 10):+.1f} to {numpy.percentile(differences, 90):+.1f}")

grouped = frames.groupby("episode_id", sort = False)
x_ahead = 100.0 * grouped["tip_x_m"].shift(-3).to_numpy()
y_ahead = 100.0 * grouped["tip_y_m"].shift(-3).to_numpy()
is_path = ((grouped["row_in_episode"].shift(-3).to_numpy() - frames["row_in_episode"].to_numpy()) == 3) & (numpy.hypot(x_ahead - x, y_ahead - y) >= 0.9)
path_from_radial = wrap(numpy.degrees(numpy.arctan2(y_ahead - y, x_ahead - x)) - radial)
path_in_picture = wrap(numpy.degrees(numpy.arctan2(y_ahead - y, x_ahead - x)) - frames["picture_heading_deg"].to_numpy())
lean = frames["rod_from_vertical_deg"].to_numpy()
wrist = numpy.degrees(frames["joint_5"].to_numpy())
back = is_path & (numpy.abs(path_from_radial) > 135.0)
print(f"\nrows pushing back towards the base (path beyond 135 degrees from radial): {back.sum()}")
print(f"  reach: median {numpy.median(reach[back]):.1f} cm, 10th to 90th percentile {numpy.percentile(reach[back], 10):.1f} to {numpy.percentile(reach[back], 90):.1f}; beyond 40 cm: {(reach[back] > 40).sum()} rows")
print(f"  rod lean there: median {numpy.median(lean[back]):.1f} degrees, 90th percentile {numpy.percentile(lean[back], 90):.1f}; wrist (joint 5): median {numpy.median(wrist[back]):.1f}, 10th to 90th {numpy.percentile(wrist[back], 10):.1f} to {numpy.percentile(wrist[back], 90):.1f}")
print(f"  all rows: wrist (joint 5) median {numpy.median(wrist):.1f}, 10th to 90th {numpy.percentile(wrist, 10):.1f} to {numpy.percentile(wrist, 90):.1f}; the driver holds 13.5")
for low, high in ((12, 20), (20, 27.5), (27.5, 35), (35, 42.5), (42.5, 49.5)):
    cell = back & (reach > low) & (reach <= high)
    if cell.sum():
        expected = magnet_tip_and_lean(joints_for(0.5 * (low + high), 0.0))[1]
        print(f"    reach {low} to {high}: {cell.sum()} rows, demonstrated lean median {numpy.median(lean[cell]):.1f}, wrist median {numpy.median(wrist[cell]):.1f};"
              f" the driver's lean at {0.5 * (low + high):.1f} cm with the forearm straight {expected:.1f}")
print(f"  looking along the path there (within 30 degrees of the picture's top): {100 * (numpy.abs(path_in_picture[back]) <= 30).mean():.1f} %;"
      f" more than 90 off: {100 * (numpy.abs(path_in_picture[back]) > 90).mean():.1f} %; |joint 4| median {numpy.median(numpy.abs(joint_4[back])):.1f}")

near_turned = (reach <= 20.0) & (numpy.abs(joint_4) >= 60.0)
print(f"\npictures within 20 cm of the base axis with |joint 4| of 60 or more: {near_turned.sum()} (of {(reach <= 20.0).sum()} pictures within 20 cm);"
      f" with |joint 4| of 80 or more: {((reach <= 20.0) & (numpy.abs(joint_4) >= 80.0)).sum()}; episodes: {frames.loc[near_turned, 'episode_id'].nunique()}")
looking_along_near = near_turned & is_path & (numpy.abs(path_in_picture) <= 30.0)
print(f"  of these, moving along the picture's top (path within 30 degrees): {looking_along_near.sum()} rows in {frames.loc[looking_along_near, 'episode_id'].nunique()} episodes")
first_rows = frames.groupby("episode_id", sort = False).head(1)
first_joint_4 = numpy.degrees(first_rows["joint_4"].to_numpy())
first_reach = 100.0 * numpy.hypot(first_rows["tip_x_m"].to_numpy(), first_rows["tip_y_m"].to_numpy())
print(f"episodes that START with |joint 4| of 40 or more: {(numpy.abs(first_joint_4) >= 40).sum()}; of these within 25 cm of the base: {((numpy.abs(first_joint_4) >= 40) & (first_reach <= 25)).sum()}")
print(first_rows.loc[numpy.abs(first_joint_4) >= 40, ["plan_number", "start_kind"]].assign(joint_4 = first_joint_4[numpy.abs(first_joint_4) >= 40].round(1), reach = first_reach[numpy.abs(first_joint_4) >= 40].round(1)).to_string(index = False))
