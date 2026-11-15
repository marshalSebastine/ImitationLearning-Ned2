"""
Robot-session critic: can the driver's ModelPolicy load the model that was trained WITHOUT sheet 03 (block B of the
plan), and with which arguments? Loads the weights and answers three logged pictures of run 5 (sheet 03). No robot
connection: only ModelPolicy is built, never RobotLink.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe fold_model_loads.py > fold_model_loads.out.txt
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
torch.set_num_threads(2)

from config import MODEL_MODE_PARAMETERS, TRAINING_RUNS  # noqa: E402
from model_mode import ModelPolicy  # noqa: E402

RUNS = ROOT / "04_runs" / "gpu_ki014"
PICTURES = Path("D:/AI_Project/03_data/model_runs/2026-11-11_173549_E_sheet03_run1/frames")
candidates = {
    "E recipe, without sheet 03": ("resnet18_chunk_stop_weight_25", "2026-11-10_resnet18_chunk_stop_weight_25", "without_path_03"),
    "E recipe seed 2, without sheet 03": ("resnet18_chunk_stop_weight_25_seed_2", "2026-11-10_resnet18_chunk_stop_weight_25_seed_2", "without_path_03"),
    "A recipe, without sheet 03": ("resnet18_final_candidate", "2026-11-09_resnet18_final_candidate", "without_path_03"),
    "model E, all sheets (used on 11.11)": ("resnet18_model_4_chunk_stop_25_all_sheets", "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets", "all_sheets")
}
print("default model of the driver:", MODEL_MODE_PARAMETERS["run_name"])
for label, (run_name, folder, fold) in candidates.items():
    checkpoint = RUNS / folder / "checkpoints" / f"{fold}.pt"
    metrics = RUNS / folder / "metrics.json"
    print(f"\n{label}: run name in TRAINING_RUNS {run_name in TRAINING_RUNS}; checkpoint exists {checkpoint.exists()} "
          f"({checkpoint.stat().st_size / 1e6:.1f} MB); metrics.json exists {metrics.exists()}")
    saved = json.loads(metrics.read_text(encoding = "utf-8"))
    print(f"   metrics.json: design {saved['settings']['design']}, split {saved['settings'].get('split')}, normalisation {saved['settings']['normalisation']}, "
          f"stop weight {saved['settings'].get('stop_positive_weight')}, folds {[fold_result.get('fold') for fold_result in saved['folds']]}")
    policy = ModelPolicy(run_name, checkpoint, metrics)
    for name in ("000.jpg", "008.jpg", "020.jpg"):
        decision = policy.decide((PICTURES / name).read_bytes())
        print(f"   run 5 picture {name}: forward {decision['tip_forward_cm']:+.2f} cm, left {decision['tip_left_cm']:+.2f} cm, twist {decision['twist_deg']:+.1f} deg, "
              f"stop chance {decision['stop_probability']:.3f}")
    print(f"   command: model_mode.py robot --label <name> --run-name {run_name} --checkpoint ..\04_runs\gpu_ki014\{folder}\checkpoints\{fold}.pt "
          f"--metrics ..\04_runs\gpu_ki014\{folder}\metrics.json")
