"""
Plan review, key "evidence": the plan says "A, the team of five, E and the plain chunk model behave alike at the
failures" (2.2, cause 2; section 8, "no switch of model"). The check's cross-model comparison (dashed_line/verify/v16)
covers runs 7 and 8 only. This script gives every all-sheets model, and the two fold models that never saw the sheet
(E recipe and A recipe), the RECORDED pictures of the other failed runs: 5 and 9 (sheet 03, E drove), 10 (sheet 03, the
team B drove), 6 (sheet 04) and 3 (sheet 02 lying square). Same pictures for every model, so it compares answers on
the states the driving model produced; it does not say where each model would have driven itself (first order only).

Inference only: nothing connects to the robot, nothing is trained, no file outside this folder is written.
Run:  D:/AI_Project/.venv/Scripts/python.exe e03_models_on_failed_runs.py > e03_models_on_failed_runs.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas
import torch

REVIEW_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from model_mode import ModelPolicy  # noqa: E402

torch.set_num_threads(2)
pandas.set_option("display.width", 320)
pandas.set_option("display.max_rows", 300)
RUNS = ROOT_FOLDER / "04_runs" / "gpu_ki014"
MODEL_RUNS = Path("D:/AI_Project/03_data/model_runs")
team = [RUNS / f"2026-11-10_resnet18_model_1_all_sheets{suffix}" / "checkpoints" / "all_sheets.pt" for suffix in ("", "_seed_2", "_seed_3", "_seed_4", "_seed_5")]
ALL_SHEETS_MODELS = {
    "E": ("resnet18_model_4_chunk_stop_25_all_sheets", RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "checkpoints" / "all_sheets.pt",
          RUNS / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "metrics.json"),
    "A": ("resnet18_model_1_all_sheets", team[0], RUNS / "2026-11-10_resnet18_model_1_all_sheets" / "metrics.json"),
    "B": ("resnet18_model_1_all_sheets", team, RUNS / "2026-11-10_resnet18_model_1_all_sheets" / "metrics.json"),
    "chunk": ("resnet18_model_2_chunk_all_sheets", RUNS / "2026-11-10_resnet18_model_2_chunk_all_sheets" / "checkpoints" / "all_sheets.pt",
              RUNS / "2026-11-10_resnet18_model_2_chunk_all_sheets" / "metrics.json")
}


def fold_models(sheet_number: int) -> dict:
    """The fold models that never saw the sheet: E recipe, A recipe and the team of the five A-recipe seeds."""

    name = f"without_path_{sheet_number:02d}.pt"
    a_seeds = [RUNS / f"2026-11-{day}_resnet18_final_candidate{suffix}" / "checkpoints" / name
               for day, suffix in (("09", ""), ("09", "_seed_2"), ("10", "_seed_3"), ("10", "_seed_4"), ("10", "_seed_5"))]
    models = {
        "E_unseen": ("resnet18_chunk_stop_weight_25", RUNS / "2026-11-10_resnet18_chunk_stop_weight_25" / "checkpoints" / name,
                     RUNS / "2026-11-10_resnet18_chunk_stop_weight_25" / "metrics.json"),
        "A_unseen": ("resnet18_final_candidate", a_seeds[0], RUNS / "2026-11-09_resnet18_final_candidate" / "metrics.json")
    }
    if all(path.exists() for path in a_seeds):
        models["B_unseen"] = ("resnet18_final_candidate", a_seeds, RUNS / "2026-11-09_resnet18_final_candidate" / "metrics.json")
    return models


def loaded(models: dict) -> dict:

    policies = {}
    for name, (run_name, checkpoint, metrics) in models.items():
        try:
            policies[name] = ModelPolicy(run_name, checkpoint, metrics)
        except Exception as problem:
            print(f"model {name} not loaded: {problem}")
    torch.set_num_threads(2)
    return policies


def answers_table(folder: Path, policies: dict) -> pandas.DataFrame:

    steps = pandas.read_csv(folder / "steps.csv")
    table = pandas.DataFrame({"step": steps["step"], "joint_4": numpy.degrees(steps["joint_4"]).round(1)})
    for name, policy in policies.items():
        answers = [policy.decide(open(folder / frame, "rb").read()) for frame in steps["frame"]]
        table[f"{name}_angle"] = [answer["angle_deg"] for answer in answers]
        table[f"{name}_len"] = [float(numpy.hypot(answer["tip_forward_cm"], answer["tip_left_cm"])) for answer in answers]
        table[f"{name}_twist"] = [answer["twist_deg"] for answer in answers]
    return table


def summary(table: pandas.DataFrame, names: list, first: int, last: int, reference: str) -> None:

    part = table[(table["step"] >= first) & (table["step"] <= last)]
    print(f"  steps {first} to {last} ({len(part)} pictures): twist asked summed / asked length median / asked direction median / direction differs from {reference}'s by (mean)")
    for name in names:
        gap = numpy.abs((part[f"{name}_angle"] - part[f"{reference}_angle"] + 180.0) % 360.0 - 180.0)
        print(f"    {name:9s}: {part[f'{name}_twist'].sum():+7.1f} deg / {part[f'{name}_len'].median():.2f} cm / {part[f'{name}_angle'].median():+7.1f} deg / {gap.mean():5.1f} deg")


all_sheets = loaded(ALL_SHEETS_MODELS)
CASES = [
    ("run 5 (sheet 03, E drove)", "2026-11-11_173549_E_sheet03_run1", 3, [(0, 12), (2, 6), (0, 16)], "E"),
    ("run 9 (sheet 03, E drove)", "2026-11-11_175203_E_sheet03_run2_no_hoodie", 3, [(0, 12), (2, 6), (0, 16)], "E"),
    ("run 10 (sheet 03, team B drove)", "2026-11-11_175544_B_team5_sheet03", 3, [(0, 12), (2, 6), (0, 16), (25, 37)], "B"),
    ("run 6 (sheet 04, E drove)", "2026-11-11_174108_E_sheet04_run1", 4, [(0, 12), (4, 8), (8, 12), (19, 50)], "E"),
    ("run 3 (sheet 02 lying square, E drove)", "2026-11-11_172743_E_sheet02_run1", 2, [(0, 5), (0, 9)], "E")
]
for label, folder_name, sheet_number, windows, reference in CASES:
    policies = dict(all_sheets)
    policies.update(loaded(fold_models(sheet_number)))
    table = answers_table(MODEL_RUNS / folder_name, policies)
    table.round(3).to_csv(REVIEW_FOLDER / f"e03_models_on_{folder_name[11:17]}.csv", index = False)
    print(f"\n===== {label}: every model on the recorded pictures =====")
    logged = pandas.read_csv(MODEL_RUNS / folder_name / "steps.csv")
    twist_column = next((column for column in logged.columns if "twist" in column and "model" in column), None)
    if twist_column is not None:
        difference = numpy.abs(logged[twist_column].to_numpy() - table[f"{reference}_twist"].to_numpy())
        print(f"  check: my {reference} twist against the logged column {twist_column}: largest difference {numpy.nanmax(difference):.4f} deg")
    for first, last in windows:
        summary(table, list(policies), first, last, reference)
    shown = table[table["step"] <= 16]
    print(shown[["step", "joint_4"] + [f"{name}_{kind}" for name in policies for kind in ("angle", "len", "twist")]].round(1).to_string(index = False))
