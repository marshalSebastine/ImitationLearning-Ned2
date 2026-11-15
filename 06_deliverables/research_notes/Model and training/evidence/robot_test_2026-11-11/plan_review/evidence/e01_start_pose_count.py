"""
Plan review, key "evidence": how many recorded episodes begin from the one start pose (tip 14.0, -1.5 cm, forearm
straight)? The plan says 170 of 210; two verifiers say 148 or 149. Reads frames.csv only; nothing is changed.
"""

import math

import numpy
import pandas

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False)
first_rows = frames.sort_values(["episode_id", "step"]).groupby("episode_id").first()
print("episodes with rows:", len(first_rows))

distance_cm = 100.0 * numpy.hypot(first_rows["tip_x_m"] - 0.140, first_rows["tip_y_m"] - (-0.015))
joint_4_deg = numpy.degrees(first_rows["joint_4"]) if first_rows["joint_4"].abs().max() < 7 else first_rows["joint_4"]
is_on_mark = (distance_cm < 0.3) & (joint_4_deg.abs() < 2.0)
print("first row within 0.3 cm of (14.0, -1.5) cm and forearm within 2 degrees of 0:", int(is_on_mark.sum()), "of", len(first_rows))
print(pandas.crosstab(first_rows["start_kind"], is_on_mark).to_string())
print("distance of the others, cm: ", distance_cm[~is_on_mark].describe().round(2).to_dict())
print("joint 4 of the others, degrees:", joint_4_deg[~is_on_mark].describe().round(1).to_dict())
print("wrist (joint 5) at the first row, degrees, by on-mark:")
joint_5_deg = numpy.degrees(first_rows["joint_5"]) if first_rows["joint_5"].abs().max() < 7 else first_rows["joint_5"]
print(joint_5_deg.groupby(is_on_mark).describe().round(1).to_string())
print("wrist at the first row of the taught starts, by kind:")
print(joint_5_deg[~is_on_mark].groupby(first_rows["start_kind"][~is_on_mark]).describe().round(1).to_string())
