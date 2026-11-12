"""
Verifier: two checks on the logged pictures of the robot runs, by running the saved models again (no robot, no training).

A. Do the logged answers come back? Model E on the pictures of run 5 and the team B on the pictures of run 10, through
   model_mode.ModelPolicy.decide, against steps.csv.
B. How much shorter is the TEAM's step than its members' steps? The team's exact step is the mean of five vectors, so
   members that disagree about the direction shorten it. This matters for a driver that carries out the predicted length.

Run from this folder:  python v11_team_length.py > v11_team_length.out.txt
"""

import os
import sys
from pathlib import Path

import numpy
import pandas
import torch

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
torch.set_num_threads(2)

from model_mode import ModelPolicy

RUNS = Path("D:/AI_Project/03_data/model_runs")
GPU = ROOT / "04_runs" / "gpu_ki014"
pandas.set_option("display.width", 250)

model_e = ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", GPU / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "checkpoints" / "all_sheets.pt",
                      GPU / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "metrics.json")
member_folders = ["2026-11-10_resnet18_model_1_all_sheets"] + [f"2026-11-10_resnet18_model_1_all_sheets_seed_{seed}" for seed in range(2, 6)]
member_names = ["resnet18_model_1_all_sheets"] + [f"resnet18_model_1_all_sheets_seed_{seed}" for seed in range(2, 6)]
members = [ModelPolicy(name, GPU / folder / "checkpoints" / "all_sheets.pt", GPU / folder / "metrics.json") for name, folder in zip(member_names, member_folders)]


def answers(policy, folder, steps):
    rows = []
    for name in steps["frame"]:
        decision = policy.decide((RUNS / folder / name).read_bytes())
        rows.append([decision["tip_forward_cm"], decision["tip_left_cm"], decision["twist_deg"], decision["stop_probability"]])
    return numpy.array(rows)


print("A. the logged answers against the models run again on the logged pictures")
folder = "2026-11-11_173549_E_sheet03_run1"
steps = pandas.read_csv(RUNS / folder / "steps.csv")
again = answers(model_e, folder, steps)
logged = steps[["model_tip_forward_cm", "model_tip_left_cm", "model_twist_deg", "model_stop_probability"]].to_numpy()
print(f"   run 5 (model E), {len(steps)} pictures: largest difference forward {numpy.abs(again[:, 0] - logged[:, 0]).max():.4f} cm, left {numpy.abs(again[:, 1] - logged[:, 1]).max():.4f} cm, "
      f"twist {numpy.abs(again[:, 2] - logged[:, 2]).max():.4f} degrees, stop chance {numpy.abs(again[:, 3] - logged[:, 3]).max():.4f}")

folder = "2026-11-11_175544_B_team5_sheet03"
steps = pandas.read_csv(RUNS / folder / "steps.csv")
logged = steps[["model_tip_forward_cm", "model_tip_left_cm", "model_twist_deg", "model_stop_probability"]].to_numpy()
member_answers = numpy.array([answers(member, folder, steps) for member in members])
team = member_answers.mean(axis = 0)
print(f"   run 10 (team B = mean of 5), {len(steps)} pictures: largest difference forward {numpy.abs(team[:, 0] - logged[:, 0]).max():.4f} cm, left {numpy.abs(team[:, 1] - logged[:, 1]).max():.4f} cm, "
      f"twist {numpy.abs(team[:, 2] - logged[:, 2]).max():.4f} degrees, stop chance {numpy.abs(team[:, 3] - logged[:, 3]).max():.4f}")

print()
print("B. the team's length against its members' lengths on the 74 pictures of run 10")
member_length = numpy.linalg.norm(member_answers[:, :, :2], axis = 2)
team_length = numpy.linalg.norm(team[:, :2], axis = 1)
mean_member_length = member_length.mean(axis = 0)
ratio = team_length / mean_member_length
member_angle = numpy.degrees(numpy.arctan2(member_answers[:, :, 1], member_answers[:, :, 0]))
spread = numpy.array([numpy.degrees(numpy.sqrt(-2.0 * numpy.log(max(1e-9, numpy.abs(numpy.exp(1j * numpy.radians(member_angle[:, index])).mean()))))) for index in range(len(steps))])
print(f"   members' length, median over pictures: " + ", ".join(f"{numpy.median(member_length[index]):.3f}" for index in range(5)) + f"; mean of the five per picture, median {numpy.median(mean_member_length):.3f} cm")
print(f"   team length median {numpy.median(team_length):.3f} cm; team / mean member length: median {numpy.median(ratio):.3f}, 10th percentile {numpy.percentile(ratio, 10):.3f}, least {ratio.min():.3f}; "
      f"under 0.9 on {100 * (ratio < 0.9).mean():.0f} % of pictures, under 0.7 on {100 * (ratio < 0.7).mean():.0f} %")
short = team_length < 0.3
print(f"   pictures where the team asks under 0.3 cm: {int(short.sum())}; there the members' mean length is {mean_member_length[short].mean():.3f} cm (team {team_length[short].mean():.3f}), "
      f"and the members' directions spread by {numpy.median(spread[short]):.0f} degrees (circular standard deviation, median) against {numpy.median(spread[~short]):.0f} elsewhere")
print(f"   first 15 decisions: team {numpy.round(team_length[:15], 2).tolist()}")
print(f"                       mean of members {numpy.round(mean_member_length[:15], 2).tolist()}")
print(f"   twist: team median size {numpy.median(numpy.abs(team[:, 2])):.2f} degrees; mean of the members' sizes {numpy.median(numpy.abs(member_answers[:, :, 2]).mean(axis = 0)):.2f}")
