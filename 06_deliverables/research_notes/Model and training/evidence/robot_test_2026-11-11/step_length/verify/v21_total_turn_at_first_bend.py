"""
Verifier: how FAR the forearm turned at the first bend of sheet 03 (not how fast): the largest turn of joint 4 from its
start value within the first 30 cm of tip travel, demonstrations against the robot runs, and against the same logged
answers summed as asked. Reads frames.csv and steps.csv only.

Run from this folder:  python v21_total_turn_at_first_bend.py > v21_total_turn_at_first_bend.out.txt
"""

import os

import numpy
import pandas

INDEX = "D:/AI_Project/03_data/processed/v01_training_index/frames.csv"
RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = {5: "2026-11-11_173549_E_sheet03_run1", 9: "2026-11-11_175203_E_sheet03_run2_no_hoodie", 10: "2026-11-11_175544_B_team5_sheet03"}
PLANS = ["03-01", "03-02", "03-03", "03-04", "03-05", "03-25", "03-26", "03-27"]


def largest_turn(tip_cm, joint_4_deg, travel_most_cm = 30.0):
    travelled = numpy.r_[0.0, numpy.cumsum(numpy.linalg.norm(numpy.diff(tip_cm, axis = 0), axis = 1))]
    turned = (joint_4_deg - joint_4_deg[0])[travelled <= travel_most_cm]
    return float(turned.min()), float(travelled[int(turned.argmin())])


frames = pandas.read_csv(INDEX, low_memory = False, usecols = ["episode_id", "row_in_episode", "plan_number", "tip_x_m", "tip_y_m", "joint_4", "picture_heading_deg"])
frames = frames.sort_values(["episode_id", "row_in_episode"])
values = []
for plan in PLANS:
    group = frames[frames["plan_number"] == plan]
    turn, at = largest_turn(100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy(), numpy.degrees(group["joint_4"].to_numpy()))
    heading = group["picture_heading_deg"].to_numpy()
    values.append(turn)
    print(f"demonstration {plan}: largest forearm turn {turn:.0f} degrees after {at:.1f} cm of tip travel; picture heading from {heading[0]:.0f} to at most {heading[:60].max():.0f} degrees")
print(f"demonstrations: median {numpy.median(values):.0f} degrees ({max(values):.0f} to {min(values):.0f})")
for number, folder in RUN_FOLDERS.items():
    steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
    turn, at = largest_turn(100.0 * steps[["tip_x", "tip_y"]].to_numpy(), numpy.degrees(steps["joint_4"].to_numpy()))
    asked_turn = numpy.cumsum(numpy.clip(steps["model_twist_deg"].to_numpy(), -10.0, 10.0))
    heading = steps["picture_heading_deg"].ffill().to_numpy()
    print(f"robot run {number}: largest forearm turn {turn:.0f} degrees after {at:.1f} cm of tip travel (decision {int(round(at))}); the asked twists add up to at most {asked_turn[:40].min():.0f} degrees; "
          f"picture heading from {heading[0]:.0f} to at most {heading[:40].max():.0f} degrees")
