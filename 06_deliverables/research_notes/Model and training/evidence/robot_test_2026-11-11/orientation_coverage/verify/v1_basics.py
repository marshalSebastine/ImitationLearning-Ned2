"""
The verifier's own recomputation of the basic coverage numbers (not the analyst's code).

Reads the audit-cleaned index through training_targets.load_training_frames and computes, with its own lines of code:
the start-kind counts, the ends of the SQP episodes, the start pose of the start-mark episodes, joint 4, the direction
of travel from radial and in the picture, the reach-by-direction cells, and the step length of the twisting rows.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v1_basics.py > v1_basics.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from training_targets import load_training_frames  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 400)


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


frames = load_training_frames({"is_audit_cleaning": True})
frames["start_kind"] = frames["start_kind"].fillna("STD")
print("rows", len(frames), "episodes", frames["episode_id"].nunique())

# ---- 1. start kinds and ends -------------------------------------------------------------------------------------
first_rows = frames.groupby("episode_id", sort = False).head(1).copy()
print("\n1. episodes per start kind:")
print(first_rows["start_kind"].value_counts().to_string())
print("pictures per start kind:")
print(frames["start_kind"].value_counts().to_string())
offsets = first_rows["start_kind"].isin(["OL1", "OL1.5", "OR1", "OR1.5"]).sum()
turned = first_rows["start_kind"].isin(["TL10", "TL15", "TR10", "TR15"]).sum()
print("offset starts", offsets, "turned starts", turned)
sqp = first_rows[first_rows["start_kind"] == "SQP"]
print("\nSQP episodes: plan, start_end (after cleaning), plan_end:")
print(sqp[["plan_number", "start_end", "plan_end", "sheet_orientation"]].to_string())
sql = first_rows[first_rows["start_kind"] == "SQL"]
print("SQL episodes:")
print(sql[["plan_number", "start_end", "plan_end", "sheet_orientation"]].to_string())

# ---- 2. the start pose -------------------------------------------------------------------------------------------
taught_kinds = ["NE", "MID", "X1", "X2"]
is_taught = first_rows["start_kind"].isin(taught_kinds)
mark = first_rows[~is_taught]
taught = first_rows[is_taught]
print("\n2. start-mark episodes:", len(mark), " taught:", len(taught))
for name, group in (("start mark", mark), ("taught", taught)):
    reach = 100 * numpy.hypot(group["tip_x_m"], group["tip_y_m"])
    joint_4 = numpy.degrees(group["joint_4"])
    print(f"  {name}: tip x {100 * group['tip_x_m'].min():.2f} to {100 * group['tip_x_m'].max():.2f} (median {100 * group['tip_x_m'].median():.2f}),"
          f" tip y {100 * group['tip_y_m'].min():.2f} to {100 * group['tip_y_m'].max():.2f} (median {100 * group['tip_y_m'].median():.2f}),"
          f" reach {reach.min():.1f} to {reach.max():.1f}, joint 4 {joint_4.min():.1f} to {joint_4.max():.1f},"
          f" |joint 4| > 10 in {(joint_4.abs() > 10).sum()}")
# note: the cleaning drops row 0, so the first kept row is not the very first pose; check the raw index too
raw = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False)
raw_first = raw.sort_values(["episode_id", "row_in_episode"]).groupby("episode_id", sort = False).head(1)
raw_first = raw_first[raw_first["episode_id"].isin(mark["episode_id"])]
print(f"  raw row 0 of the same start-mark episodes: tip x {100 * raw_first['tip_x_m'].min():.2f} to {100 * raw_first['tip_x_m'].max():.2f},"
      f" tip y {100 * raw_first['tip_y_m'].min():.2f} to {100 * raw_first['tip_y_m'].max():.2f},"
      f" joint 4 {numpy.degrees(raw_first['joint_4']).min():.1f} to {numpy.degrees(raw_first['joint_4']).max():.1f}")
outliers = mark[(100 * mark["tip_x_m"] - 14.0).abs() > 0.3]
print("  start-mark episodes whose first kept tip x differs from 14.0 by more than 0.3 cm:")
print(outliers[["plan_number", "start_kind", "tip_x_m", "tip_y_m", "row_in_episode"]].to_string())

# ---- 3. joint 4 --------------------------------------------------------------------------------------------------
joint_4 = numpy.degrees(frames["joint_4"])
print(f"\n3. joint 4: min {joint_4.min():.1f}, max {joint_4.max():.1f}, median |j4| {joint_4.abs().median():.1f},"
      f" beyond 60: {100 * (joint_4.abs() > 60).mean():.1f} %, beyond 90: {100 * (joint_4.abs() > 90).mean():.1f} %,"
      f" beyond 110: {100 * (joint_4.abs() > 110).mean():.1f} %, beyond 117: {100 * (joint_4.abs() > 117).mean():.2f} % ({(joint_4.abs() > 117).sum()} pictures)")

# ---- 4. direction of travel from radial, own three-step path -----------------------------------------------------
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
grouped = frames.groupby("episode_id", sort = False)
x = 100 * frames["tip_x_m"].to_numpy()
y = 100 * frames["tip_y_m"].to_numpy()
ahead = 3
x_ahead = 100 * grouped["tip_x_m"].shift(-ahead).to_numpy()
y_ahead = 100 * grouped["tip_y_m"].shift(-ahead).to_numpy()
row_ahead = grouped["row_in_episode"].shift(-ahead).to_numpy()
is_consecutive = (row_ahead - frames["row_in_episode"].to_numpy()) == ahead
path_cm = numpy.hypot(x_ahead - x, y_ahead - y)
is_path = is_consecutive & (path_cm >= 0.9)
path_table = numpy.degrees(numpy.arctan2(y_ahead - y, x_ahead - x))
radial = numpy.degrees(numpy.arctan2(y, x))
from_radial = wrap(path_table - radial)
in_picture = wrap(path_table - frames["picture_heading_deg"].to_numpy())
valid = from_radial[is_path]
print(f"\n4. three-step paths (own rule: rows consecutive, at least 0.9 cm net): {is_path.sum()} rows")
print(f"   within 45 of radial {100 * (numpy.abs(valid) <= 45).mean():.1f} %, 45 to 135 {100 * ((numpy.abs(valid) > 45) & (numpy.abs(valid) <= 135)).mean():.1f} %,"
      f" beyond 135 {100 * (numpy.abs(valid) > 135).mean():.1f} %")
sideways = valid[(numpy.abs(valid) > 45) & (numpy.abs(valid) <= 135)]
print(f"   sideways left {(sideways > 0).sum()}, right {(sideways < 0).sum()}")
valid_picture = in_picture[is_path]
print(f"   in the picture: within 30 of the top {100 * (numpy.abs(valid_picture) <= 30).mean():.1f} %, 30 to 60 {100 * ((numpy.abs(valid_picture) > 30) & (numpy.abs(valid_picture) <= 60)).mean():.1f} %,"
      f" 60 to 120 {100 * ((numpy.abs(valid_picture) > 60) & (numpy.abs(valid_picture) <= 120)).mean():.1f} %, beyond 120 {100 * (numpy.abs(valid_picture) > 120).mean():.1f} %")

# ---- 5. the labelled steps (one step, in the picture frame plus heading) and the cells ----------------------------
forward = 100 * frames["tip_forward_m"].fillna(0.0).to_numpy()
left = 100 * frames["tip_left_m"].fillna(0.0).to_numpy()
step_cm = numpy.hypot(forward, left)
step_table = wrap(frames["picture_heading_deg"].to_numpy() + numpy.degrees(numpy.arctan2(left, forward)))
step_from_radial = wrap(step_table - radial)
is_moving = (step_cm >= 0.3) & (frames["action"] != "stop").to_numpy()
print(f"\n5. moving steps (0.3 cm or more, no stop rows): {is_moving.sum()} ({100 * is_moving.mean():.1f} %)")
reach = numpy.hypot(x, y)
print(f"   reach of all pictures: min {reach.min():.1f}, max {reach.max():.1f}; beyond 42.5: {(reach > 42.5).sum()} pictures, beyond 47.5: {(reach > 47.5).sum()}")
reach_edges = [12, 20, 27.5, 35, 42.5, 49.5]
direction_bin = (numpy.round(step_from_radial / 30.0) * 30).astype(int)
direction_bin[direction_bin == -180] = 180
table = pandas.crosstab(pandas.cut(reach[is_moving], reach_edges), direction_bin[is_moving])
print(table.to_string())
print("   moving steps outside 12 to 49.5 cm reach:", int(((reach[is_moving] <= 12) | (reach[is_moving] > 49.5)).sum()))
mid = table.iloc[1:4]
print(f"   mid reach (20 to 42.5): smallest cell {mid.to_numpy().min()}; near (12 to 20) sideways/back cells: "
      f"{table.iloc[0][[-150, -120, -90, -60, 120, 150, 180]].to_list()}; far (42.5 to 49.5): {table.iloc[4].to_list()} sum {table.iloc[4].sum()}")
thin = (table.to_numpy() < 100).sum()
print(f"   cells under 100 pictures: {thin} of {table.size}; share of moving steps in them {100 * table.to_numpy()[table.to_numpy() < 100].sum() / is_moving.sum():.1f} %")

# ---- 6. twisting rows --------------------------------------------------------------------------------------------
twist = frames["forearm_change_deg"].fillna(0.0).abs().to_numpy()
is_twisting = (twist >= 6) & (frames["action"] != "stop").to_numpy()
print(f"\n6. rows twisting 6 degrees or more: {is_twisting.sum()}, median step {numpy.median(step_cm[is_twisting]):.2f} cm, under 0.5 cm {100 * (step_cm[is_twisting] < 0.5).mean():.0f} %")
not_stop = (frames["action"] != "stop").to_numpy()
print(f"   all non-stop rows: median step {numpy.median(step_cm[not_stop]):.2f} cm, mean {step_cm[not_stop].mean():.2f}, under 0.5 cm {100 * (step_cm[not_stop] < 0.5).mean():.1f} %, under 0.2 cm {100 * (step_cm[not_stop] < 0.2).mean():.1f} %")

# ---- 7. first five moving steps of the start-mark episodes: net direction ------------------------------------------
frames["step_cm"] = step_cm
frames["is_moving"] = is_moving
rows = []
for episode_id, group in frames.groupby("episode_id", sort = False):
    moving = group[group["is_moving"]].head(5)
    if len(moving) == 0:
        continue
    heading = numpy.radians(moving["picture_heading_deg"].to_numpy())
    f = 100 * moving["tip_forward_m"].to_numpy()
    l = 100 * moving["tip_left_m"].to_numpy()
    dx = (f * numpy.cos(heading) - l * numpy.sin(heading)).sum()
    dy = (f * numpy.sin(heading) + l * numpy.cos(heading)).sum()
    first = group.iloc[0]
    start_radial = numpy.degrees(numpy.arctan2(first["tip_y_m"], first["tip_x_m"]))
    rows.append({"episode_id": episode_id, "plan_number": first["plan_number"], "start_kind": first["start_kind"], "sheet": first["sheet"],
                 "net_from_radial": float(wrap(numpy.degrees(numpy.arctan2(dy, dx)) - start_radial)), "net_cm": float(numpy.hypot(dx, dy)),
                 "joint_4_start": float(numpy.degrees(first["joint_4"]))})
five = pandas.DataFrame(rows)
five_mark = five[~five["start_kind"].isin(taught_kinds)]
absolute = five_mark["net_from_radial"].abs()
print(f"\n7. start-mark episodes ({len(five_mark)}): net direction of the first five moving steps from radial: within 20: {(absolute <= 20).sum()},"
      f" 20 to 45: {((absolute > 20) & (absolute <= 45)).sum()}, 45 to 90: {((absolute > 45) & (absolute <= 90)).sum()}, over 90: {(absolute > 90).sum()}")
print(five_mark.groupby("start_kind")["net_from_radial"].agg(["count", "min", "median", "max"]).round(1).to_string())
five_taught = five[five["start_kind"].isin(taught_kinds)]
print(f"   taught starts: {len(five_taught)}, |joint 4| at the start over 10 degrees in {(five_taught['joint_4_start'].abs() > 10).sum()},"
      f" joint 4 from {five_taught['joint_4_start'].min():.1f} to {five_taught['joint_4_start'].max():.1f}")
