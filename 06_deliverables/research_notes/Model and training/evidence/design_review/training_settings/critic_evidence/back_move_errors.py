"""
Angle errors of the end-to-end runs on the true back moves of held-out sheets 03, 05 and 06 (read only).

Checks whether the exact model's 0.3 % recall of straight-back moves means large direction errors, or predictions
in the neighbouring back-left and back-right bins (within 45 degrees). Same frames and predictions as
per_class_recall.py.
"""

import sys

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

from pathlib import Path

import numpy

from config import TRAINING_TARGET_PARAMETERS
from training_metrics import nearest_direction
from training_targets import DIRECTION_NAMES, load_training_frames

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FOLDS = ["without_path_03", "without_path_05", "without_path_06"]
RUNS = {
    "exact": "2026-11-09_resnet18_finetune_exact",
    "classes": "2026-11-09_resnet18_finetune_classes",
    "mirror (03 and 05 only)": "2026-11-09_resnet18_finetune_mirror",
}


def angle_difference_deg(first_angle, second_angle):
    """Return the absolute angle between two directions in degrees, from 0 to 180."""
    difference = numpy.abs((numpy.asarray(first_angle) - numpy.asarray(second_angle) + 180.0) % 360.0 - 180.0)
    return difference


def main():
    frames = load_training_frames(dict(TRAINING_TARGET_PARAMETERS))
    print("is_audit_cleaning in default parameters:", TRAINING_TARGET_PARAMETERS.get("is_audit_cleaning"))
    position_of_frame = {index: position for position, index in enumerate(frames["frame_index"].to_numpy())}
    back_names = ["back", "back_left", "back_right"]
    for label, run_folder in RUNS.items():
        true_angles, predicted_angles = [], []
        for fold in FOLDS:
            path = RUNS_FOLDER / run_folder / f"predictions_{fold}.npz"
            if not path.exists():
                continue
            predictions = dict(numpy.load(path))
            rows = frames.iloc[[position_of_frame[index] for index in predictions["frame_index"]]]
            is_moving = rows["action"].isin(DIRECTION_NAMES).to_numpy()
            true_angles.append(rows["move_angle_deg"].to_numpy()[is_moving])
            predicted_angles.append(predictions["angle_deg"][is_moving])
        true_angles = numpy.concatenate(true_angles)
        predicted_angles = numpy.concatenate(predicted_angles)
        true_classes = nearest_direction(true_angles)
        predicted_classes = nearest_direction(predicted_angles)
        errors = angle_difference_deg(true_angles, predicted_angles)
        print("=" * 90)
        print(label, "| moving rows", len(true_angles), "| within 45 overall %.3f" % (errors <= 45.0).mean())
        for name in back_names:
            index = DIRECTION_NAMES.index(name)
            is_class = true_classes == index
            predicted_names, counts = numpy.unique(predicted_classes[is_class], return_counts = True)
            spread = ", ".join(f"{DIRECTION_NAMES[value]} {count}" for value, count in zip(predicted_names, counts))
            print(f"   true {name:10s} rows {is_class.sum():5d}  within 45 {(errors[is_class] <= 45.0).mean():.3f}"
                  f"  within 90 {(errors[is_class] <= 90.0).mean():.3f}  median error {numpy.median(errors[is_class]):6.1f}  predicted: {spread}")
        is_any_back = numpy.isin(true_classes, [DIRECTION_NAMES.index(name) for name in back_names])
        is_predicted_back = numpy.isin(predicted_classes, [DIRECTION_NAMES.index(name) for name in back_names])
        print(f"   any back class: rows {is_any_back.sum()}, predicted any back class for {(is_predicted_back[is_any_back]).mean():.3f} of them,"
              f" within 45 {(errors[is_any_back] <= 45.0).mean():.3f}, predicted share of any back {is_predicted_back.mean():.3f}"
              f" against true {is_any_back.mean():.3f}")


if __name__ == "__main__":
    main()
