"""
Verifier, check 2: the camera's view with the wrist held, against the demonstrated view, recomputed with the PROJECT's
own functions (side_view_test.magnet_tip_and_lean, solve_with_wrist_fixed, forearm_camera_model.camera_pose_in_base and
project_points), not the analyst's numpy copies. All calculated from the logged joints; nothing measured on the robot.

For every demonstration row: same tip point, same joint 4, wrist 13.5 degrees, hand -6.1 degrees (variant "constant"),
or the wrist of the episode's first row (variant "first_row"). The shift is where the SAME table/tip point lands in the
two pictures (640 px wide).

Run from this folder (a few minutes, one core):
    D:/AI_Project/.venv/Scripts/python.exe v02_held_wrist_geometry.py
"""

import math
import sys

import numpy
import pandas

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
import forearm_camera_model
from config import table_height
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed

INDEX_FOLDER = "D:/AI_Project/03_data/processed/v01_training_index"
HELD_WRIST_DEG = 13.5
HELD_HAND_DEG = -6.1
JOINTS = [f"joint_{index}" for index in range(1, 7)]

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)

frames = pandas.read_csv(f"{INDEX_FOLDER}/frames.csv", usecols = ["episode_id", "row_in_episode", "sheet", "frame_index", "tip_x_m", "tip_y_m", "tip_z_m",
                                                                    "rod_from_vertical_deg", "sheet_orientation", "action"] + JOINTS)
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
first_wrist = frames.groupby("episode_id")["joint_5"].transform("first").to_numpy()
first_hand = frames.groupby("episode_id")["joint_6"].transform("first").to_numpy()
joint_values = frames[JOINTS].to_numpy(dtype = float)

rows = []
for position in range(len(frames)):
    joints = joint_values[position].tolist()
    tip, lean = magnet_tip_and_lean(joints)
    centre, axes = forearm_camera_model.camera_pose_in_base(joints)
    table_point = numpy.array([tip[0], tip[1], table_height(tip[0], tip[1])])
    ahead = table_point + 0.05 * numpy.array([axes[0, 2], axes[1, 2], 0.0]) / max(1e-9, math.hypot(axes[0, 2], axes[1, 2]))
    ahead[2] = table_height(ahead[0], ahead[1])
    pixel_tip = forearm_camera_model.project_points(tip, joints)[0]
    pixel_table = forearm_camera_model.project_points(table_point, joints)[0]
    pixel_ahead = forearm_camera_model.project_points(ahead, joints)[0]
    row = {"tip_height_above_table_mm": 1000 * (tip[2] - table_point[2]), "lean_fk_deg": lean, "tip_u": pixel_tip[0], "tip_v": pixel_tip[1],
           "fk_tip_x": tip[0], "fk_tip_y": tip[1], "fk_tip_z": tip[2]}
    for variant, wrist_rad, hand_rad in (("constant", math.radians(HELD_WRIST_DEG), math.radians(HELD_HAND_DEG)),
                                         ("first_row", first_wrist[position], first_hand[position])):
        held = solve_with_wrist_fixed(tip, joints[3], wrist_rad, joints[:3])
        if held is None:
            row[f"{variant}_solved"] = False
            continue
        held[5] = hand_rad
        held_tip, held_lean = magnet_tip_and_lean(held)
        held_centre, held_axes = forearm_camera_model.camera_pose_in_base(held)
        shift_tip = forearm_camera_model.project_points(tip, held)[0] - pixel_tip
        shift_table = forearm_camera_model.project_points(table_point, held)[0] - pixel_table
        shift_ahead = forearm_camera_model.project_points(ahead, held)[0] - pixel_ahead
        row.update({f"{variant}_solved": True,
                    f"{variant}_tip_error_mm": 1000 * float(numpy.linalg.norm(held_tip - tip)),
                    f"{variant}_camera_move_mm": 1000 * float(numpy.linalg.norm(held_centre - centre)),
                    f"{variant}_camera_turn_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(axes[:, 2] @ held_axes[:, 2]))))),
                    f"{variant}_tip_du": shift_tip[0], f"{variant}_tip_dv": shift_tip[1],
                    f"{variant}_table_du": shift_table[0], f"{variant}_table_dv": shift_table[1],
                    f"{variant}_ahead_du": shift_ahead[0], f"{variant}_ahead_dv": shift_ahead[1],
                    f"{variant}_lean_deg": held_lean,
                    f"{variant}_joint_1_deg": math.degrees(held[0]), f"{variant}_joint_2_deg": math.degrees(held[1]),
                    f"{variant}_joint_3_deg": math.degrees(held[2])})
    rows.append(row)
result = pandas.concat([frames, pandas.DataFrame(rows)], axis = 1)
result["joint_4_deg"] = numpy.degrees(result["joint_4"])
result["joint_5_deg"] = numpy.degrees(result["joint_5"])
result["wrist_offset_deg"] = result["joint_5_deg"] - HELD_WRIST_DEG
result["reach_cm"] = 100 * numpy.hypot(result["tip_x_m"], result["tip_y_m"])
compact = ["frame_index", "sheet", "joint_4_deg", "joint_5_deg", "wrist_offset_deg", "reach_cm", "lean_fk_deg", "tip_height_above_table_mm", "tip_u", "tip_v",
           "constant_camera_move_mm", "constant_camera_turn_deg", "constant_tip_du", "constant_tip_dv", "constant_ahead_du", "constant_ahead_dv", "constant_lean_deg",
           "first_row_tip_du", "first_row_tip_dv"]
result[compact].to_csv("v02_held_wrist_geometry_rows.csv", index = False, float_format = "%.2f")

print("rows", len(result))
tip_difference = numpy.abs(result[["fk_tip_x", "fk_tip_y", "fk_tip_z"]].to_numpy() - result[["tip_x_m", "tip_y_m", "tip_z_m"]].to_numpy())
print("my tip against the index's tip columns (mm): largest %.3f" % (1000 * tip_difference.max()))
print("my lean against the index's rod_from_vertical_deg: largest difference %.3f degrees" % (result["lean_fk_deg"] - result["rod_from_vertical_deg"]).abs().max())
print("tip above the fitted table (mm): median %.1f, 5th %.1f, 95th %.1f" % (result["tip_height_above_table_mm"].median(),
      result["tip_height_above_table_mm"].quantile(0.05), result["tip_height_above_table_mm"].quantile(0.95)))

for variant in ("constant", "first_row"):
    solved = result[f"{variant}_solved"].fillna(False).astype(bool)
    part = result[solved]
    print(f"\n=== variant {variant}: solved {int(solved.sum())} of {len(result)}; tip kept within {part[f'{variant}_tip_error_mm'].max():.4f} mm ===")
    for name, column in (("camera move, mm", f"{variant}_camera_move_mm"), ("camera turn, deg", f"{variant}_camera_turn_deg")):
        values = part[column]
        print("  %-18s median %.2f, 90th %.2f, 95th %.2f, most %.2f" % (name, values.median(), values.quantile(0.9), values.quantile(0.95), values.max()))
    for point in ("tip", "table", "ahead"):
        shift = numpy.hypot(part[f"{variant}_{point}_du"], part[f"{variant}_{point}_dv"])
        print("  %-6s shift px at 640: median %.2f, 90th %.1f, 95th %.1f, most %.1f; over 8 px %.1f %%, over 16 px %.1f %%, over 20 px %.1f %%, over 40 px %.1f %%" % (
            point, shift.median(), shift.quantile(0.9), shift.quantile(0.95), shift.max(),
            100 * (shift > 8).mean(), 100 * (shift > 16).mean(), 100 * (shift > 20).mean(), 100 * (shift > 40).mean()))

part = result[result["constant_solved"].fillna(False).astype(bool)].copy()
part["tip_shift"] = numpy.hypot(part["constant_tip_du"], part["constant_tip_dv"])
part["over_20"] = part["tip_shift"] > 20
print("\nconstant 13.5: share of rows with the tip shifted over 20 px (at 640)")
print("by sheet:\n", (100 * part.groupby("sheet")["over_20"].mean()).round(1).to_string())
part["twist_bin"] = pandas.cut(part["joint_4_deg"].abs(), [0, 30, 60, 90, 120], include_lowest = True)
part["reach_bin"] = pandas.cut(part["reach_cm"], [12, 20, 26, 32, 38, 50], include_lowest = True)
print("by |joint 4|:\n", part.groupby("twist_bin", observed = True).agg(
    rows = ("over_20", "size"), over_20 = ("over_20", "mean"), median_px = ("tip_shift", "median"),
    p90_px = ("tip_shift", lambda v: v.quantile(0.9)), abs_offset_median = ("wrist_offset_deg", lambda v: v.abs().median())).round(3).to_string())
print("by reach:\n", part.groupby("reach_bin", observed = True).agg(
    rows = ("over_20", "size"), over_20 = ("over_20", "mean"), median_px = ("tip_shift", "median"),
    p90_px = ("tip_shift", lambda v: v.quantile(0.9)), abs_offset_median = ("wrist_offset_deg", lambda v: v.abs().median())).round(3).to_string())

# is the shift a function of the wrist alone, and is it vertical?
moved = part[part["wrist_offset_deg"].abs() > 1.0]
slope_v, intercept_v = numpy.polyfit(moved["wrist_offset_deg"], moved["constant_tip_dv"], 1)
slope_u, intercept_u = numpy.polyfit(moved["wrist_offset_deg"], moved["constant_tip_du"], 1)
residual = moved["constant_tip_dv"] - (slope_v * moved["wrist_offset_deg"] + intercept_v)
print("\ntip shift against (demonstrated wrist - 13.5), rows over 1 degree off (%d rows):" % len(moved))
print("  down the picture: %.3f px per degree at 640 (%.3f at 160), intercept %.2f, residual sd %.2f px, r %.4f" % (
    slope_v, slope_v / 4, intercept_v, residual.std(), numpy.corrcoef(moved["wrist_offset_deg"], moved["constant_tip_dv"])[0, 1]))
print("  across the picture: %.3f px per degree, |du| median %.2f px, 95th %.2f px" % (
    slope_u, moved["constant_tip_du"].abs().median(), moved["constant_tip_du"].abs().quantile(0.95)))
slope_t = numpy.polyfit(moved["wrist_offset_deg"], moved["constant_table_dv"], 1)[0]
slope_a = numpy.polyfit(moved["wrist_offset_deg"], moved["constant_ahead_dv"], 1)[0]
print("  table point under the tip: %.3f px per degree; table point 5 cm ahead: %.3f px per degree" % (slope_t, slope_a))
print("  sign: a demonstrated wrist ABOVE 13.5 gives dv %s 0, so in the driver's view the tip sits %s in the picture than in the demonstration" % (
    ">" if slope_v > 0 else "<", "lower" if slope_v > 0 else "higher"))

lean_demo = part["lean_fk_deg"]
lean_held = part["constant_lean_deg"]
print("\nrod lean: demonstrated median %.1f, 95th %.1f, most %.1f; held: median %.1f, 95th %.1f, most %.1f; held over 58 degrees in %d rows" % (
    lean_demo.median(), lean_demo.quantile(0.95), lean_demo.max(), lean_held.median(), lean_held.quantile(0.95), lean_held.max(), int((lean_held > 58).sum())))
print("demonstrated lean over 58 degrees in %d rows" % int((lean_demo > 58).sum()))
print("reach: most %.1f cm; rows beyond 46 cm: %d" % (part["reach_cm"].max(), int((part["reach_cm"] > 46).sum())))
