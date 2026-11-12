"""
Verifier: for the E recipe (design "chunk") the saved predictions hold the chunk and an angle taken from the chunk, not
the exact step's angle; check that the analyst's npz gives back the saved chunk on every row.

Run from this folder:  python v06b_check_e_recipe_chunk.py >> v06_check_fold_outputs.out.txt
"""

from pathlib import Path

import numpy

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
data = numpy.load(ROOT / "06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/step_length/fold_outputs_E_recipe.npz")
position_of = {int(value): position for position, value in enumerate(data["frame_index"])}
for number in range(1, 7):
    saved = numpy.load(ROOT / f"04_runs/gpu_ki014/2026-11-10_resnet18_chunk_stop_weight_25/predictions_without_path_{number:02d}.npz")
    positions = numpy.array([position_of[int(value)] for value in saved["frame_index"]])
    exact = data["exact"][positions]
    angle = numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0]))
    difference = numpy.abs((angle - saved["angle_vector_deg"] + 180) % 360 - 180)
    chunk = numpy.abs(data["chunk"][positions] - saved["chunk_cm"]).max()
    print(number, "E recipe: exact angle against saved angle_vector_deg: largest difference", round(float(difference.max()), 4),
          "degrees; chunk against saved chunk_cm: largest difference", round(float(chunk), 5), "cm")
