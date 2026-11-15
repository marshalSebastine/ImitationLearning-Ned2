"""
Print per-fold numbers of the probe experiments and end-to-end runs that the targets section needs (read only).
"""

import json
import sys
from pathlib import Path

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
PROBE_FOLDER = RUNS_FOLDER / "2026-11-09_frozen_feature_probes"

KEYS = ["test_episodes", "direction_rows", "within_45_deg", "path_within_45_deg", "vector_within_45_deg",
        "path_vector_within_45_deg", "path_expected_within_45_deg", "expected_within_45_deg",
        "class_balanced_accuracy", "episodes_stopped_right", "episodes_stopped_early", "episodes_never_stopped",
        "tuned_episodes_stopped_right", "tuned_episodes_stopped_early", "tuned_episodes_never_stopped", "tuned_stop_threshold",
        "tuned_two_episodes_stopped_right", "tuned_two_episodes_stopped_early", "tuned_two_episodes_never_stopped",
        "tuned_two_stop_threshold", "stop_average_precision", "stop_recall", "stop_precision",
        "twist_rows", "twist_sign_accuracy", "twist_error_mean_deg", "lean_rows", "lean_sign_accuracy", "lean_error_mean_deg",
        "hand_rows", "hand_sign_accuracy", "chunk_error_step_1_cm", "chunk_error_step_3_cm", "chunk_error_step_5_cm"]


def print_result(name, folds):
    print(f"=== {name}")
    fold_names = [fold.get("fold", "?") for fold in folds]
    print("fold".ljust(34), " ".join(f"{fold_name[-10:]:>10}" for fold_name in fold_names))
    for key in KEYS:
        values = [fold.get(key) for fold in folds]
        if all(value is None for value in values):
            continue
        print(key.ljust(34), " ".join(f"{(value if value is not None else float('nan')):>10.4g}" for value in values))


def main(experiments):
    for experiment in experiments:
        path = PROBE_FOLDER / f"{experiment}.json"
        if not path.exists():
            print("missing", path)
            continue
        data = json.loads(path.read_text(encoding = "utf-8"))
        for result in data["results"]:
            print_result(f"{experiment} / {result['name']}", result["folds"])
    for run_folder in sorted(RUNS_FOLDER.glob("2026-11-09_*")):
        if run_folder.name.endswith("frozen_feature_probes"):
            continue
        folds = []
        for fold_path in sorted(run_folder.glob("fold_*.json")):
            folds.append(json.loads(fold_path.read_text(encoding = "utf-8"))["measures"])
        if folds:
            print_result(f"end-to-end / {run_folder.name}", folds)


if __name__ == "__main__":
    main(sys.argv[1:] or ["designs", "stop", "stop_distance", "splits", "seeds"])
