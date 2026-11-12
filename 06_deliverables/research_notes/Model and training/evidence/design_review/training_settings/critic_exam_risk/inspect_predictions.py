"""
Print the keys, shapes and a few values of the saved prediction files used in the exam-risk critique (read only).
Run with python -B.
"""

from pathlib import Path

import numpy

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")

FILES = [
    RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_01.npz",
    RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_02.npz",
    RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_05.npz",
    RUNS_FOLDER / "2026-11-09_resnet18_finetune_classes" / "predictions_without_path_05.npz",
]


def main():
    for path in FILES:
        print("==", path)
        predictions = numpy.load(path)
        for key in predictions.files:
            values = predictions[key]
            print(f"   {key:28s} {str(values.shape):16s} {values.dtype}  first: {values.ravel()[:5]}")


if __name__ == "__main__":
    main()
