"""Verifier, first look: what the run logs and the training index hold (measured, nothing changed)."""
import sys
from pathlib import Path
import numpy
import pandas

RUNS = Path("D:/AI_Project/03_data/model_runs")
for folder in sorted(RUNS.glob("2026-11-11_17*")):
    steps = pandas.read_csv(folder / "steps.csv")
    print(folder.name, len(steps), steps["action"].value_counts().to_dict(), "refusals", steps["refusal"].notna().sum(),
          "clamped", int(steps["is_turn_clamped"].sum()), "kept", int(steps["is_direction_kept"].sum()),
          "joint 4 first %.1f min %.1f max %.1f" % (numpy.degrees(steps["joint_4"].iloc[0]), numpy.degrees(steps["joint_4"].min()), numpy.degrees(steps["joint_4"].max())),
          "joint 5 %.1f..%.1f joint 6 %.1f..%.1f" % (numpy.degrees(steps["joint_5"].min()), numpy.degrees(steps["joint_5"].max()), numpy.degrees(steps["joint_6"].min()), numpy.degrees(steps["joint_6"].max())))
frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False)
print(frames.shape, frames["episode_id"].nunique())
print(frames["action"].value_counts().to_dict())
print(frames.groupby("sheet")["episode_id"].nunique().to_dict())
print(frames[["tip_forward_m", "tip_left_m", "forearm_change_deg", "move_length_m", "move_angle_deg", "picture_heading_deg", "forearm_turn_deg"]].describe().T)
