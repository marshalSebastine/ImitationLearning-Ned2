"""
Check of the plan review, key "evidence_check" (objections P6 and P7): at the first bend of sheet 03, do the models
ask for the recorded turn when they see the RECORDED pictures (standard starts from end A, first 16 rows, as the
bends checker counted: recordings 11.1 cm and 93 degrees there)? On the robot's own pictures model E asked 61, 74 and
61 degrees in all where the recordings turn 96 (step-length check, missed 5). If the models ask for the full turn on
the recorded pictures, the shortfall on the robot belongs to the states the runs were in, not to the models' fit.
Teacher forced: every model sees the recording's own pictures. Models: E and the team of five (trained on these
pictures), and the E-recipe and A-recipe models that never saw sheet 03 (the exam case).

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
pandas.set_option("display.width", 250)
RUNS = ROOT / "04_runs" / "gpu_ki014"
RAW = Path("D:/AI_Project/03_data/raw/main_data")
ROWS = 16

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "sheet", "start_kind", "plan_number", "frame_path", "row_in_episode", "forearm_change_deg",
                                    "tip_forward_m", "tip_left_m"])
ends = pandas.read_csv(ROOT / "06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv")[["episode_id", "corrected_end"]]
frames = frames.merge(ends, on = "episode_id", how = "left")
chosen = frames[(frames["sheet"] == "path_03") & (frames["start_kind"] == "STD") & (frames["corrected_end"] == "A")]
print("standard-start episodes of sheet 03 from end A:", chosen["episode_id"].nunique())

suffixes = ("", "_seed_2", "_seed_3", "_seed_4", "_seed_5")
policies = {
    "E (all sheets)": ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "checkpoints" / "all_sheets.pt",
                                  RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "metrics.json"),
    "team of five (all sheets)": ModelPolicy("resnet18_model_1_all_sheets",
                                             [RUNS / f"2026-11-10_resnet18_model_1_all_sheets{suffix}" / "checkpoints" / "all_sheets.pt" for suffix in suffixes],
                                             RUNS / "2026-11-10_resnet18_model_1_all_sheets" / "metrics.json"),
    "E recipe, never saw sheet 03": ModelPolicy("resnet18_chunk_stop_weight_25", RUNS / "2026-11-10_resnet18_chunk_stop_weight_25" / "checkpoints" / "without_path_03.pt",
                                                RUNS / "2026-11-10_resnet18_chunk_stop_weight_25" / "metrics.json"),
    "A recipe, never saw sheet 03": ModelPolicy("resnet18_final_candidate", RUNS / "2026-11-09_resnet18_final_candidate" / "checkpoints" / "without_path_03.pt",
                                                RUNS / "2026-11-09_resnet18_final_candidate" / "metrics.json")
}
torch.set_num_threads(2)

rows_out = []
for episode_id, group in chosen.groupby("episode_id", sort = False):
    group = group.sort_values("row_in_episode").iloc[:ROWS]
    demonstrated_twist = group["forearm_change_deg"].to_numpy()
    demonstrated_length = 100.0 * numpy.hypot(group["tip_forward_m"], group["tip_left_m"]).to_numpy()
    line = {"plan": group["plan_number"].iloc[0], "demonstrated twist": demonstrated_twist.sum(), "demonstrated travel cm": demonstrated_length.sum()}
    pictures = [(RAW / path).read_bytes() for path in group["frame_path"]]
    for name, policy in policies.items():
        answers = [policy.decide(picture) for picture in pictures]
        line[f"{name}: twist"] = sum(answer["twist_deg"] for answer in answers)
        line[f"{name}: travel cm"] = sum(float(numpy.hypot(answer["tip_forward_cm"], answer["tip_left_cm"])) for answer in answers)
    rows_out.append(line)
table = pandas.DataFrame(rows_out)
print(table.round(1).to_string(index = False))
print()
print(f"medians over the {len(table)} episodes, first {ROWS} rows:")
print(f"  demonstrated: twist {table['demonstrated twist'].median():+.1f} deg, travel {table['demonstrated travel cm'].median():.1f} cm")
for name in policies:
    share = table[f"{name}: twist"] / table["demonstrated twist"]
    print(f"  {name}: twist {table[f'{name}: twist'].median():+.1f} deg (share of the demonstrated: median {share.median():.2f}, {share.min():.2f} to {share.max():.2f}), "
          f"asked travel {table[f'{name}: travel cm'].median():.1f} cm")
print()
print("for comparison, the robot runs on sheet 03 (steps.csv), first 16 pictures: asked twist summed, asked travel")
for run_number, folder in ((5, "2026-11-11_173549_E_sheet03_run1"), (9, "2026-11-11_175203_E_sheet03_run2_no_hoodie"), (10, "2026-11-11_175544_B_team5_sheet03")):
    steps = pandas.read_csv(Path("D:/AI_Project/03_data/model_runs") / folder / "steps.csv").iloc[:ROWS]
    print(f"  run {run_number}: {steps['model_twist_deg'].sum():+.1f} deg, {numpy.hypot(steps['model_tip_forward_cm'], steps['model_tip_left_cm']).sum():.1f} cm asked, "
          f"joint 4 from {numpy.degrees(steps['joint_4'].iloc[0]):+.1f} deg")
