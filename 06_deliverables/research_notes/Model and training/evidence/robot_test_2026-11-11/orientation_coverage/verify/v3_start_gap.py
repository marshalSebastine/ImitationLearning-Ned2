"""
The verifier's own measure of how far the puck lay from the tip in the first picture of every demonstration and of
the ten robot runs, and of the distance while the block pushes (own red-blob finder, fitted forearm camera).

It also checks the analyst's lay-out fits: the puck seen in the first picture against the fitted start cap of
01_placements_per_episode.csv, and the analyst's own distances in 03_start_gaps.csv.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v3_start_gap.py > v3_start_gap.out.txt
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
from forearm_camera_model import pixels_to_plane  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 400)

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
PUCK_POINT_HEIGHT_M = 0.0145
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]
ROBOT_RUNS = ["2026-11-11_172021_check_no_puck", "2026-11-11_172421_E_sheet01_run1", "2026-11-11_172743_E_sheet02_run1", "2026-11-11_173036_E_sheet02_run2",
              "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1", "2026-11-11_174448_E_sheet05_run1", "2026-11-11_174936_E_sheet06_run1",
              "2026-11-11_175203_E_sheet03_run2_no_hoodie", "2026-11-11_175544_B_team5_sheet03"]


def red_blob(picture_bgr):
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


def puck_on_table(picture_path, joints):
    picture = cv2.imread(str(picture_path))
    if picture is None:
        return None
    blob = red_blob(picture)
    if blob is None:
        return None
    guess = pixels_to_plane(numpy.array([[blob[0], blob[1]]]), joints, PUCK_POINT_HEIGHT_M)[0]
    if not numpy.all(numpy.isfinite(guess)):
        return None
    height = table_height(float(guess[0]), float(guess[1])) + PUCK_POINT_HEIGHT_M
    return 100.0 * pixels_to_plane(numpy.array([[blob[0], blob[1]]]), joints, height)[0]


frames = load_training_frames({"is_audit_cleaning": True})
frames["start_kind"] = frames["start_kind"].fillna("STD")
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
taught_kinds = ["NE", "MID", "X1", "X2"]

# ---- the first kept picture of every demonstration ----------------------------------------------------------------
rows = []
for episode_id, group in frames.groupby("episode_id", sort = False):
    first = group.iloc[0]
    joints = first[JOINT_NAMES].to_numpy(dtype = float)
    puck = puck_on_table(RAW_FOLDER / first["frame_path"], joints)
    tip = 100.0 * first[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    rows.append({"episode_id": episode_id, "plan": first["plan_number"], "sheet": first["sheet"][-2:], "start_kind": first["start_kind"],
                 "puck_x": puck[0] if puck is not None else numpy.nan, "puck_y": puck[1] if puck is not None else numpy.nan,
                 "puck_from_tip_cm": float(numpy.hypot(*(puck - tip))) if puck is not None else numpy.nan,
                 "puck_bearing_from_radial_deg": float((numpy.degrees(numpy.arctan2(puck[1] - tip[1], puck[0] - tip[0]) - numpy.arctan2(tip[1], tip[0])) + 180) % 360 - 180) if puck is not None else numpy.nan})
starts = pandas.DataFrame(rows)
starts.to_csv(VERIFY_FOLDER / "v3_start_gap.csv", index = False)
mark = starts[~starts["start_kind"].isin(taught_kinds)]
taught = starts[starts["start_kind"].isin(taught_kinds)]

# ---- the distance while the block pushes: forward steps of 0.8 cm or more, rows 10 to 40, five per episode ---------
forward_cm = 100.0 * frames["tip_forward_m"].fillna(0.0)
left_cm = 100.0 * frames["tip_left_m"].fillna(0.0)
is_forward_push = (forward_cm >= 0.8) & (left_cm.abs() <= 0.25) & (frames["forearm_change_deg"].fillna(0.0).abs() < 3.0) & frames["row_in_episode"].between(10, 60)
pushing = frames[is_forward_push].groupby("episode_id", sort = False).head(5)
push_distances = []
for _, row in pushing.iterrows():
    puck = puck_on_table(RAW_FOLDER / row["frame_path"], row[JOINT_NAMES].to_numpy(dtype = float))
    if puck is not None:
        push_distances.append(float(numpy.hypot(puck[0] - 100.0 * row["tip_x_m"], puck[1] - 100.0 * row["tip_y_m"])))
push_distances = numpy.array(push_distances)
touching = float(numpy.median(push_distances))
print(f"Distance from the tip to the puck while the block pushes straight forward ({len(push_distances)} pictures of {pushing['episode_id'].nunique()} episodes; measured):"
      f" median {touching:.2f} cm, 10th to 90th percentile {numpy.percentile(push_distances, 10):.2f} to {numpy.percentile(push_distances, 90):.2f}")

print(f"\nStart-mark demonstrations: puck found in {mark['puck_from_tip_cm'].notna().sum()} of {len(mark)} first pictures")
found = mark["puck_from_tip_cm"].dropna()
print(f"  puck from the tip: median {found.median():.2f} cm, 10th to 90th percentile {found.quantile(0.1):.2f} to {found.quantile(0.9):.2f}, least {found.min():.2f}, most {found.max():.2f}")
print(f"  free travel before the block touches (distance minus {touching:.2f}): median {found.median() - touching:.2f} cm")
print(f"  under 3.5 cm (within 1 cm of touching by the analyst's rule): {(found < 3.5).sum()};  under touching + 1.0 = {touching + 1.0:.2f} cm: {(found < touching + 1.0).sum()};  over 5.5 cm: {(found > 5.5).sum()}")
print(mark.groupby("sheet")["puck_from_tip_cm"].agg(["count", "min", "median", "max"]).round(1).to_string())
print("  sheet 02 by start kind:")
print(mark[mark["sheet"] == "02"].groupby("start_kind")["puck_from_tip_cm"].agg(["count", "min", "median", "max"]).round(1).to_string())
print(f"  where the puck lay as seen from the tip (direction from radial, degrees): median {mark['puck_bearing_from_radial_deg'].median():.1f},"
      f" 10th to 90th percentile {mark['puck_bearing_from_radial_deg'].quantile(0.1):.1f} to {mark['puck_bearing_from_radial_deg'].quantile(0.9):.1f}")
found_taught = taught["puck_from_tip_cm"].dropna()
print(f"Taught starts: found in {len(found_taught)} of {len(taught)}; median {found_taught.median():.2f} cm, 10th to 90th percentile {found_taught.quantile(0.1):.2f} to {found_taught.quantile(0.9):.2f}")

# ---- against the analyst's numbers --------------------------------------------------------------------------------
analyst = pandas.read_csv(VERIFY_FOLDER.parent / "03_start_gaps.csv")
both = starts.merge(analyst[analyst["what"] == "demonstration"][["episode_id", "puck_from_tip_cm"]], on = "episode_id", suffixes = ("", "_analyst"))
difference = (both["puck_from_tip_cm"] - both["puck_from_tip_cm_analyst"]).dropna()
print(f"\nAgainst the analyst's 03_start_gaps.csv ({len(difference)} episodes with both): own minus analyst median {difference.median():+.2f} cm,"
      f" median |difference| {difference.abs().median():.2f}, 90th percentile {difference.abs().quantile(0.9):.2f}, over 1 cm in {(difference.abs() > 1.0).sum()}")
placements = pandas.read_csv(VERIFY_FOLDER.parent / "01_placements_per_episode.csv")
caps = starts.merge(placements[["episode_id", "start_cap_x_cm", "start_cap_y_cm", "line_start_direction_deg"]], on = "episode_id")
caps = caps[caps["start_kind"].isin(["STD", "SQP", "SQL", "TL10", "TL15", "TR10", "TR15"])]
cap_distance = numpy.hypot(caps["puck_x"] - caps["start_cap_x_cm"], caps["puck_y"] - caps["start_cap_y_cm"]).dropna()
print(f"Puck seen in the first picture against the analyst's fitted start cap ({len(cap_distance)} start-mark episodes without a puck offset):"
      f" median {cap_distance.median():.2f} cm, 90th percentile {cap_distance.quantile(0.9):.2f}, most {cap_distance.max():.2f}")
print(f"  where the pucks lay at the start (own measure): x median {caps['puck_x'].median():.1f} cm (10th to 90th {caps['puck_x'].quantile(0.1):.1f} to {caps['puck_x'].quantile(0.9):.1f}),"
      f" y median {caps['puck_y'].median():.1f} ({caps['puck_y'].quantile(0.1):.1f} to {caps['puck_y'].quantile(0.9):.1f})")

# ---- the ten robot runs -------------------------------------------------------------------------------------------
print("\nThe ten robot runs, first picture (own measure):")
for folder in ROBOT_RUNS:
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    joints = steps.loc[0, JOINT_NAMES].to_numpy(dtype = float)
    puck = puck_on_table(RUNS_FOLDER / folder / "frames" / "000.jpg", joints)
    tip = 100.0 * steps.loc[0, ["tip_x", "tip_y"]].to_numpy(dtype = float)
    if puck is None:
        print(f"  {folder}: no puck found")
        continue
    print(f"  {folder}: tip ({tip[0]:.1f}, {tip[1]:.1f}), puck ({puck[0]:.1f}, {puck[1]:.1f}), puck from tip {numpy.hypot(*(puck - tip)):.2f} cm, joint 4 {numpy.degrees(joints[3]):+.1f}")
