"""
Verifier: are the analyst's fold_outputs_<recipe>.npz right? Two checks, no training, nothing connects to the robot.

1. Every row: the angle of the npz's exact step against the angle saved when the fold model was scored
   (04_runs/gpu_ki014/<run>/predictions_<fold>.npz "angle_deg" or "angle_vector_deg"), and the twist.
2. A random sample of rows per fold: the exact step computed here again, through a DIFFERENT input path than the
   analyst's (the raw 640 x 480 JPEG through model_mode.ModelPolicy.decide, as on the robot, instead of the cached
   160 x 120 pictures), against the npz.

Run from this folder:  python v06_check_fold_outputs.py > v06_check_fold_outputs.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas
import torch

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
torch.set_num_threads(2)

from model_mode import ModelPolicy

STEP_LENGTH = ROOT / "06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/step_length"
GPU_RUNS = ROOT / "04_runs/gpu_ki014"
RAW = Path("D:/AI_Project/03_data/raw/main_data")
RECIPES = {"E_recipe": ("resnet18_chunk_stop_weight_25", "2026-11-10_resnet18_chunk_stop_weight_25"),
           "A_recipe": ("resnet18_final_candidate", "2026-11-09_resnet18_final_candidate")}
SAMPLE_PER_FOLD = 60
randomness = numpy.random.default_rng(111126)

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False, usecols = ["frame_index", "frame_path", "sheet"])
frames = frames.set_index("frame_index")

for recipe, (run_name, run_folder) in RECIPES.items():
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    position_of = {int(frame_index): position for position, frame_index in enumerate(data["frame_index"])}
    print(f"=== {recipe}: {len(data['frame_index'])} rows in the npz, {len(position_of)} different frame indices")
    for fold_number in range(1, 7):
        fold = f"without_path_{fold_number:02d}"
        saved = numpy.load(GPU_RUNS / run_folder / f"predictions_{fold}.npz")
        positions = numpy.array([position_of[int(value)] for value in saved["frame_index"]])
        exact = data["exact"][positions]
        sheets = frames.loc[saved["frame_index"], "sheet"].unique()
        angle = numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0]))
        saved_angle = saved["angle_vector_deg"] if "angle_vector_deg" in saved.files and recipe != "E_recipe" else saved["angle_deg"]
        angle_difference = numpy.abs((angle - saved_angle + 180.0) % 360.0 - 180.0)
        twist_difference = numpy.abs(exact[:, 2] - saved["twist_deg"])
        # the sample through the robot's input path
        policy = ModelPolicy(run_name, GPU_RUNS / run_folder / "checkpoints" / f"{fold}.pt", GPU_RUNS / run_folder / "metrics.json")
        chosen = randomness.choice(len(saved["frame_index"]), size = SAMPLE_PER_FOLD, replace = False)
        differences = []
        lengths = []
        for row in chosen:
            frame_index = int(saved["frame_index"][row])
            decision = policy.decide((RAW / frames.loc[frame_index, "frame_path"]).read_bytes())
            mine = numpy.array([decision["tip_forward_cm"], decision["tip_left_cm"], decision["twist_deg"]])
            differences.append(numpy.abs(mine - exact[row, :3]))
            lengths.append((numpy.hypot(mine[0], mine[1]), numpy.hypot(exact[row, 0], exact[row, 1])))
        differences = numpy.array(differences)
        lengths = numpy.array(lengths)
        print(f"   {fold} (held-out sheets {list(sheets)}), {len(positions)} rows: angle against the saved predictions: largest difference {angle_difference.max():.4f} degrees; "
              f"twist {twist_difference.max():.5f}; sample of {SAMPLE_PER_FOLD} through the JPEG path: largest difference forward {differences[:, 0].max():.4f} cm, "
              f"left {differences[:, 1].max():.4f} cm, twist {differences[:, 2].max():.4f} degrees; length: mine median {numpy.median(lengths[:, 0]):.3f}, npz {numpy.median(lengths[:, 1]):.3f}, "
              f"largest length difference {numpy.abs(lengths[:, 0] - lengths[:, 1]).max():.4f} cm")
