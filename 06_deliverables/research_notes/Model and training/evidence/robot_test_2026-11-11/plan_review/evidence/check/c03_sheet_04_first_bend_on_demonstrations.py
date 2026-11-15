"""
Check of the plan review, key "evidence_check" (objection P3): is run 6's failure at the first S-bend of sheet 04 a
matter of the model's fit or of the state the run was in (sheet 15 to 25 degrees off, start 5 cm off)?
The two checkers disagree. A cheap pointer that needs no robot: give model E (trained on all sheets, so also on these
pictures) and the E-recipe model that never saw sheet 04 the pictures of the eleven standard-start demonstrations of
sheet 04 from end A, and add up the twist each asks for over the rows in which the demonstrator went through the
first S-bend (the first 16 rows, as the bends checker used). Teacher forced: every model sees the demonstration's own
pictures, so this says what the model asks in the recorded states, not where it would drive.

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
                                    "tip_forward_m", "tip_left_m", "joint_4", "action"])
ends = pandas.read_csv(ROOT / "06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv")[["episode_id", "corrected_end"]]
frames = frames.merge(ends, on = "episode_id", how = "left")
chosen = frames[(frames["sheet"] == "path_04") & (frames["start_kind"] == "STD") & (frames["corrected_end"] == "A")]
print("standard-start episodes of sheet 04 from end A:", chosen["episode_id"].nunique())

policies = {
    "E (all sheets)": ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "checkpoints" / "all_sheets.pt",
                                  RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "metrics.json"),
    "E recipe, never saw sheet 04": ModelPolicy("resnet18_chunk_stop_weight_25", RUNS / "2026-11-10_resnet18_chunk_stop_weight_25" / "checkpoints" / "without_path_04.pt",
                                                RUNS / "2026-11-10_resnet18_chunk_stop_weight_25" / "metrics.json")
}
torch.set_num_threads(2)

rows_out = []
for episode_id, group in chosen.groupby("episode_id", sort = False):
    group = group.sort_values("row_in_episode").head(ROWS + 1)
    # the label of row k is the move made after picture k; the last row of the window is not needed
    group = group.iloc[:ROWS]
    demonstrated_twist = group["forearm_change_deg"].to_numpy()
    demonstrated_length = 100.0 * numpy.hypot(group["tip_forward_m"], group["tip_left_m"]).to_numpy()
    line = {"plan": group["plan_number"].iloc[0], "rows": len(group), "demonstrated twist, summed": demonstrated_twist.sum(),
            "demonstrated travel cm": demonstrated_length.sum(),
            "joint 4 after the window": numpy.degrees(group["joint_4"].iloc[-1]) if abs(group["joint_4"].iloc[-1]) < 7 else group["joint_4"].iloc[-1]}
    for name, policy in policies.items():
        answers = [policy.decide((RAW / path).read_bytes()) for path in group["frame_path"]]
        twist = numpy.array([answer["twist_deg"] for answer in answers])
        length = numpy.array([float(numpy.hypot(answer["tip_forward_cm"], answer["tip_left_cm"])) for answer in answers])
        line[f"{name}: twist summed"] = twist.sum()
        line[f"{name}: share of the demonstrated"] = twist.sum() / demonstrated_twist.sum() if abs(demonstrated_twist.sum()) > 1 else numpy.nan
        line[f"{name}: asked travel cm"] = length.sum()
    rows_out.append(line)

table = pandas.DataFrame(rows_out)
print(table.round(2).to_string(index = False))
print()
turning = table[table["demonstrated twist, summed"].abs() >= 60.0]
other = table[table["demonstrated twist, summed"].abs() < 60.0]
print(f"episodes whose demonstrator turned 60 degrees or more within the first {ROWS} rows: {len(turning)}; the others: {len(other)}")
for name in policies:
    print(f"  {name}: median share of the demonstrated twist in the turning episodes {turning[f'{name}: share of the demonstrated'].median():.2f} "
          f"(range {turning[f'{name}: share of the demonstrated'].min():.2f} to {turning[f'{name}: share of the demonstrated'].max():.2f}); "
          f"median asked twist {turning[f'{name}: twist summed'].median():+.1f} against {turning['demonstrated twist, summed'].median():+.1f} demonstrated; "
          f"in the other episodes asked {other[f'{name}: twist summed'].median() if len(other) else float('nan'):+.1f} against {other['demonstrated twist, summed'].median() if len(other) else float('nan'):+.1f}")
print()
print("for comparison, run 6 on the robot (steps.csv): twist asked over the first 16 pictures")
steps = pandas.read_csv("D:/AI_Project/03_data/model_runs/2026-11-11_174108_E_sheet04_run1/steps.csv")
print(f"  summed asked twist {steps['model_twist_deg'].iloc[:ROWS].sum():+.1f} deg, sum of sizes {steps['model_twist_deg'].iloc[:ROWS].abs().sum():.1f} deg, "
      f"asked travel {numpy.hypot(steps['model_tip_forward_cm'], steps['model_tip_left_cm']).iloc[:ROWS].sum():.1f} cm")
