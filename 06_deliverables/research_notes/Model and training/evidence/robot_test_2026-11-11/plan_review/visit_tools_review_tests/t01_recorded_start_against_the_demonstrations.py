"""Review check (reads only): is "recorded_start_joints_deg" the pose from which 149 of 210 demonstrations begin, and where is its tip?"""

import math
import sys

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

import numpy
import pandas

from config import MODEL_MODE_PARAMETERS, TRAINING_DATA_PARAMETERS, PUSHER_PARAMETERS, pushing_height
from side_view_test import magnet_tip_and_lean, ned2_fk

frames = pandas.read_csv(TRAINING_DATA_PARAMETERS["processed_folder"] / "frames.csv")
print("columns:", [name for name in frames.columns][:60])
first_rows = frames.sort_values(["episode_id", "row_in_episode"]).groupby("episode_id").head(1)
print("episodes:", len(first_rows))
joint_columns = [f"joint_{index}" for index in range(1, 7)]
first_deg = numpy.degrees(first_rows[joint_columns].to_numpy(dtype = float))
recorded = numpy.array(MODEL_MODE_PARAMETERS["recorded_start_joints_deg"])
difference = numpy.abs(first_deg - recorded)
for tolerance in (0.1, 0.25, 0.5, 1.0, 2.0):
    print(f"first rows within {tolerance} degrees on all six joints: {int((difference.max(axis = 1) <= tolerance).sum())} of {len(first_rows)}")
print("median of first rows, degrees:", numpy.round(numpy.median(first_deg, axis = 0), 2))
close = difference.max(axis = 1) <= 0.5
print("mean of the close first rows, degrees:", numpy.round(first_deg[close].mean(axis = 0), 3))

recorded_rad = [math.radians(value) for value in recorded]
tip, lean = magnet_tip_and_lean(recorded_rad)
print("recorded start tip (m):", numpy.round(tip, 4), "lean", round(lean, 1))
print("pushing height there minus extension:", round(pushing_height(tip[0], tip[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"], 4), " tip z:", round(float(tip[2]), 4))

# the same tip from ned2_fk directly, written out here, not through magnet_tip_and_lean
fk = ned2_fk.fk(recorded_rad)
tool = numpy.array(fk["tool"][0])
rod = numpy.array(fk["joint_6"][1])[:, 2]
print("tool:", numpy.round(tool, 4), "rod direction:", numpy.round(rod, 4), "tip:", numpy.round(tool + PUSHER_PARAMETERS["extension_below_rod_end_m"] * rod, 4))

# the robot's own reported pose in the first rows against the model
pose_columns = [name for name in ("x", "y", "z") if name in frames.columns]
print("pose columns in frames.csv:", pose_columns, [name for name in frames.columns if "tip" in name or name in ("x", "y", "z", "pose_x", "pose_y")])
tips_first = numpy.array([magnet_tip_and_lean(list(numpy.radians(row)))[0] for row in first_deg])
print("first-row tips: x cm 5/50/95 pct", numpy.round(100 * numpy.percentile(tips_first[:, 0], [5, 50, 95]), 2), " y cm", numpy.round(100 * numpy.percentile(tips_first[:, 1], [5, 50, 95]), 2))
distance = 100 * numpy.hypot(tips_first[:, 0] - tip[0], tips_first[:, 1] - tip[1])
print("first-row tip distance from the recorded start, cm: within 0.3:", int((distance <= 0.3).sum()), "within 1.0:", int((distance <= 1.0).sum()), "max:", round(float(distance.max()), 1))
