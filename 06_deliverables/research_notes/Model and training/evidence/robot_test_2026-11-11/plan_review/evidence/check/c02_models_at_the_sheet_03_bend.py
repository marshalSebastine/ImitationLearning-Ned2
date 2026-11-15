"""
Check of the plan review, key "evidence_check" (objection P7): do the models answer alike at the first bend of
sheet 03? The critic's e03 says that on the same logged pictures (2 to 6 of runs 5, 9 and 10) model E asks a mean
direction of -40, -36, -34 degrees, model A -60, -60, -56 and the team of five -57, -51, -49, and that the team's
summed twist is 18 to 25 % larger. This script recomputes those numbers with its own loop and adds two things the
critic did not print: the spread between the five single members of the team (is the difference larger than the
seed-to-seed spread?) and the second seed of the E recipe is not available as an all-sheets model, so the plain
chunk model stands in as the second chunk-recipe model.

Inference only. Nothing connects to the robot, nothing is trained, nothing outside this folder is written.
"""

import sys
from pathlib import Path

import numpy
import pandas
import torch

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
from model_mode import ModelPolicy  # noqa: E402

torch.set_num_threads(2)
RUNS = ROOT / "04_runs" / "gpu_ki014"
MODEL_RUNS = Path("D:/AI_Project/03_data/model_runs")
suffixes = ("", "_seed_2", "_seed_3", "_seed_4", "_seed_5")
team_paths = [RUNS / f"2026-11-10_resnet18_model_1_all_sheets{suffix}" / "checkpoints" / "all_sheets.pt" for suffix in suffixes]
a_metrics = RUNS / "2026-11-10_resnet18_model_1_all_sheets" / "metrics.json"
policies = {
    "E": ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "checkpoints" / "all_sheets.pt",
                     RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "metrics.json"),
    "chunk": ModelPolicy("resnet18_model_2_chunk_all_sheets", RUNS / "2026-11-10_resnet18_model_2_chunk_all_sheets" / "checkpoints" / "all_sheets.pt",
                         RUNS / "2026-11-10_resnet18_model_2_chunk_all_sheets" / "metrics.json"),
    "team of five": ModelPolicy("resnet18_model_1_all_sheets", team_paths, a_metrics)
}
for number, path in enumerate(team_paths, start = 1):
    policies[f"member {number}"] = ModelPolicy("resnet18_model_1_all_sheets", path, a_metrics)
torch.set_num_threads(2)

RUN_FOLDERS = {5: "2026-11-11_173549_E_sheet03_run1", 9: "2026-11-11_175203_E_sheet03_run2_no_hoodie", 10: "2026-11-11_175544_B_team5_sheet03"}
FIRST, LAST = 2, 6
summary = []
for run_number, folder_name in RUN_FOLDERS.items():
    folder = MODEL_RUNS / folder_name
    steps = pandas.read_csv(folder / "steps.csv")
    part = steps[(steps["step"] >= FIRST) & (steps["step"] <= LAST)]
    line = {"run": run_number, "logged direction mean": part["model_angle_deg"].mean(), "logged twist sum": part["model_twist_deg"].sum()}
    for name, policy in policies.items():
        answers = [policy.decide((folder / frame).read_bytes()) for frame in part["frame"]]
        angles = numpy.array([numpy.degrees(numpy.arctan2(answer["tip_left_cm"], answer["tip_forward_cm"])) for answer in answers])
        line[f"{name}: direction"] = angles.mean()
        line[f"{name}: twist"] = sum(answer["twist_deg"] for answer in answers)
    summary.append(line)
table = pandas.DataFrame(summary).set_index("run")
pandas.set_option("display.width", 250)
print(f"pictures {FIRST} to {LAST} of each run; direction = mean asked direction in the picture (degrees, negative = to the right), twist = summed asked twist (degrees)")
print(table.round(1).T.to_string())
print()
members = [f"member {number}: direction" for number in range(1, 6)]
print("spread of the five single members' mean direction (smallest to largest), against E:")
for run_number in RUN_FOLDERS:
    values = table.loc[run_number, members].to_numpy(dtype = float)
    print(f"  run {run_number}: members {values.min():+.1f} to {values.max():+.1f} (mean {values.mean():+.1f}), team {table.loc[run_number, 'team of five: direction']:+.1f}, "
          f"E {table.loc[run_number, 'E: direction']:+.1f}, plain chunk model {table.loc[run_number, 'chunk: direction']:+.1f}")
members = [f"member {number}: twist" for number in range(1, 6)]
print("spread of the five single members' summed twist, against E:")
for run_number in RUN_FOLDERS:
    values = table.loc[run_number, members].to_numpy(dtype = float)
    print(f"  run {run_number}: members {values.min():+.1f} to {values.max():+.1f}, team {table.loc[run_number, 'team of five: twist']:+.1f}, "
          f"E {table.loc[run_number, 'E: twist']:+.1f}, plain chunk model {table.loc[run_number, 'chunk: twist']:+.1f}")
