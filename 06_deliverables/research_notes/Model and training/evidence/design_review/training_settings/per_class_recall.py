"""
Per-direction recall and predicted shares of the end-to-end runs and the main probes, on the same test rows (read only).

Shows whether a design without class weights (the exact step) under-predicts the rare directions (back moves) on
held-out sheets. Run with python -B.
"""

import sys

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

from pathlib import Path

import numpy

from config import TRAINING_TARGET_PARAMETERS
from training_metrics import nearest_direction
from training_targets import DIRECTION_NAMES, load_training_frames

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
PROBE_FOLDER = RUNS_FOLDER / "2026-11-09_frozen_feature_probes" / "predictions"
FOLDS = ["without_path_03", "without_path_05", "without_path_06"]
ALL_FOLDS = [f"without_path_0{number}" for number in range(1, 7)]


def recall_table(frames, sources, folds):
    """Print recall per true direction and the predicted share of each direction, pooled over the folds."""

    position_of_frame = {index: position for position, index in enumerate(frames["frame_index"].to_numpy())}
    print("source".ljust(44) + "".join(name[:11].rjust(12) for name in DIRECTION_NAMES) + "   balanced")
    for label, path_of_fold in sources.items():
        true_classes, predicted_classes = [], []
        for fold in folds:
            path = path_of_fold(fold)
            if not path.exists():
                continue
            predictions = dict(numpy.load(path))
            rows = frames.iloc[[position_of_frame[index] for index in predictions["frame_index"]]]
            is_moving = rows["action"].isin(DIRECTION_NAMES).to_numpy()
            true_classes.append(nearest_direction(rows["move_angle_deg"].to_numpy()[is_moving]))
            predicted_classes.append(nearest_direction(predictions["angle_deg"][is_moving]))
        true_classes = numpy.concatenate(true_classes)
        predicted_classes = numpy.concatenate(predicted_classes)
        recalls = [float((predicted_classes[true_classes == index] == index).mean()) for index in range(len(DIRECTION_NAMES))]
        print(f"{label[:42]:44s}" + "".join(f"{value:12.3f}" for value in recalls) + f"   {numpy.mean(recalls):.3f}")
        shares = [float((predicted_classes == index).mean()) for index in range(len(DIRECTION_NAMES))]
        print(f"{'   predicted share':44s}" + "".join(f"{value:12.3f}" for value in shares))
    true_shares = [float((true_classes == index).mean()) for index in range(len(DIRECTION_NAMES))]
    print(f"{'   true share':44s}" + "".join(f"{value:12.3f}" for value in true_shares) + f"   rows {len(true_classes)}")


def main():
    frames = load_training_frames(dict(TRAINING_TARGET_PARAMETERS))
    print("=== End to end, held-out sheets 03, 05, 06 pooled")
    recall_table(frames, {
        "end-to-end ResNet-18 classes (sqrt weights)": lambda fold: RUNS_FOLDER / "2026-11-09_resnet18_finetune_classes" / f"predictions_{fold}.npz",
        "end-to-end ResNet-18 exact (no weights)": lambda fold: RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / f"predictions_{fold}.npz",
        "end-to-end small CNN classes": lambda fold: RUNS_FOLDER / "2026-11-09_small_cnn_scratch_classes" / f"predictions_{fold}.npz",
    }, FOLDS)
    print("\n=== Frozen probes, all six held-out sheets pooled")
    recall_table(frames, {
        "probe classes, square-root weights": lambda fold: PROBE_FOLDER / f"scores_1_2_3_{fold}.npz",
        "probe classes, no class weights": lambda fold: PROBE_FOLDER / f"no_class_weights_{fold}.npz",
        "probe classes, full inverse weights": lambda fold: PROBE_FOLDER / f"full_inverse_weights_{fold}.npz",
        "probe exact step (no weights)": lambda fold: PROBE_FOLDER / f"exact_step_{fold}.npz",
    }, ALL_FOLDS)


if __name__ == "__main__":
    main()
