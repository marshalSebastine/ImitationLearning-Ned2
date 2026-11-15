"""
List the arrays saved in a few end-to-end prediction files (read only; writes nothing).
"""

from pathlib import Path

import numpy

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
FILES = [
    RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / "predictions_without_path_05.npz",
    RUNS_FOLDER / "2026-11-09_resnet18_finetune_chunk" / "predictions_without_path_03.npz",
    RUNS_FOLDER / "2026-11-09_resnet18_finetune_classes" / "predictions_without_path_03.npz",
    RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / "predictions_without_path_03.npz",
]

for path in FILES:
    contents = numpy.load(path)
    print(path.parent.name, path.name)
    for key in contents.files:
        print("   ", key, contents[key].shape, contents[key].dtype)
