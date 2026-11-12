"""
Proofreader sitting_a, check 1: do the three models that sitting A names load through the driver's own loader, with the
values a checklist would put on the command line, and are they the models of 11.11?

Nothing connects to the robot: only ModelPolicy (file reads) and logged pictures of 11.11 are used.

Run:  D:/AI_Project/.venv/Scripts/python.exe -B p01_models_load.py > p01_models_load.out.txt
"""

import hashlib
import os
import sys
from pathlib import Path

import pandas

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
RUNS = Path("D:/AI_Project/03_data/model_runs")
sys.path.insert(0, str(CODE))
os.chdir(CODE)   # the checklist types the commands from 02_code, so relative paths are tried as typed

from config import MODEL_MODE_PARAMETERS, TRAINING_RUNS   # noqa: E402
from model_mode import ModelPolicy                         # noqa: E402

G = Path("..") / "04_runs" / "gpu_ki014"
MODELS = {
    "model E (all six sheets)": {
        "run_name": "resnet18_model_4_chunk_stop_25_all_sheets",
        "checkpoint": [G / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "checkpoints" / "all_sheets.pt"],
        "metrics": G / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets" / "metrics.json",
        "logged_run": "2026-11-11_173549_E_sheet03_run1"},
    "team of five (model B)": {
        "run_name": "resnet18_model_1_all_sheets",
        "checkpoint": [G / f"2026-11-10_resnet18_model_1_all_sheets{suffix}" / "checkpoints" / "all_sheets.pt"
                       for suffix in ("", "_seed_2", "_seed_3", "_seed_4", "_seed_5")],
        "metrics": G / "2026-11-10_resnet18_model_1_all_sheets" / "metrics.json",
        "logged_run": "2026-11-11_175544_B_team5_sheet03"},
    "model trained without sheet 03 (E recipe, first seed)": {
        "run_name": "resnet18_chunk_stop_weight_25",
        "checkpoint": [G / "2026-11-10_resnet18_chunk_stop_weight_25" / "checkpoints" / "without_path_03.pt"],
        "metrics": G / "2026-11-10_resnet18_chunk_stop_weight_25" / "metrics.json",
        "logged_run": None},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


print(f"default model of the driver today: {MODEL_MODE_PARAMETERS['run_name']}")
print()
frames_of_run_5 = RUNS / "2026-11-11_173549_E_sheet03_run1" / "frames"
for name, model in MODELS.items():
    print(name)
    print(f"   --run-name {model['run_name']}   (a key of TRAINING_RUNS: {model['run_name'] in TRAINING_RUNS})")
    for checkpoint in model["checkpoint"]:
        print(f"   --checkpoint {str(checkpoint)}   exists {checkpoint.exists()}, {checkpoint.stat().st_size / 1e6:.1f} MB, sha256 {sha(checkpoint)}")
    print(f"   --metrics {str(model['metrics'])}   exists {model['metrics'].exists()}")
    policy = ModelPolicy(model["run_name"], model["checkpoint"], model["metrics"])
    print(f"   loaded: design {policy.settings['design']}, split {policy.settings.get('split')}, {len(policy.models)} network(s), "
          f"stop weight {policy.settings.get('stop_positive_weight')}, picture size {policy.settings['picture_size']}")
    for picture in ("000.jpg", "008.jpg", "020.jpg"):
        answer = policy.decide((frames_of_run_5 / picture).read_bytes())
        print(f"   run 5 picture {picture}: forward {answer['tip_forward_cm']:+.2f} cm, left {answer['tip_left_cm']:+.2f} cm, "
              f"twist {answer['twist_deg']:+.1f} deg, stop chance {answer['stop_probability']:.3f}")
    if model["logged_run"]:
        folder = RUNS / model["logged_run"]
        steps = pandas.read_csv(folder / "steps.csv")
        worst = 0.0
        for index in (0, 5, 10, 20, 30, len(steps) - 1):
            row = steps.iloc[index]
            answer = policy.decide((folder / row["frame"]).read_bytes())
            worst = max(worst, abs(answer["tip_forward_cm"] - row["model_tip_forward_cm"]), abs(answer["tip_left_cm"] - row["model_tip_left_cm"]),
                        abs(answer["twist_deg"] - row["model_twist_deg"]), abs(answer["stop_probability"] - row["model_stop_probability"]))
        print(f"   against the answers logged on 11.11 in {model['logged_run']} (6 pictures): largest difference {worst:.5f}  -> "
              f"{'the same model as on 11.11' if worst < 1e-3 else 'NOT the model of 11.11'}")
    print()

print("wrong pairings, to see what the loader notices:")
trials = {
    "model E's checkpoint with the default run name (model 1, design exact)": ("resnet18_model_1_all_sheets", MODELS["model E (all six sheets)"]["checkpoint"], MODELS["team of five (model B)"]["metrics"]),
    "the without-sheet-03 checkpoint under model E's run name": ("resnet18_model_4_chunk_stop_25_all_sheets", MODELS["model trained without sheet 03 (E recipe, first seed)"]["checkpoint"], MODELS["model E (all six sheets)"]["metrics"]),
    "the seed-2 without-sheet-03 checkpoint under the first seed's run name": ("resnet18_chunk_stop_weight_25", [G / "2026-11-10_resnet18_chunk_stop_weight_25_seed_2" / "checkpoints" / "without_path_03.pt"], MODELS["model trained without sheet 03 (E recipe, first seed)"]["metrics"]),
    "without_path_04.pt in place of without_path_03.pt": ("resnet18_chunk_stop_weight_25", [G / "2026-11-10_resnet18_chunk_stop_weight_25" / "checkpoints" / "without_path_04.pt"], MODELS["model trained without sheet 03 (E recipe, first seed)"]["metrics"]),
}
for label, (run_name, checkpoint, metrics) in trials.items():
    try:
        ModelPolicy(run_name, checkpoint, metrics)
        print(f"   {label}: LOADS WITHOUT A WORD")
    except Exception as error:
        print(f"   {label}: refused ({type(error).__name__}: {str(error).splitlines()[0][:110]})")
