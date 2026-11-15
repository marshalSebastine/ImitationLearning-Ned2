"""
Robot-session critic: which of the plan's driver switches are only numbers today, and which need new code? The whole
loop is run against the SIMULATED robot (model_mode.simulated_run, SimulatedRobotLink; the real RobotLink is never
built) with the model trained without sheet 03 and with the plan's numbers passed as overrides of
MODEL_MODE_PARAMETERS. Logs go into a folder beside this script, not into 03_data. The simulated pictures look nothing
like real ones, so this checks the software path only.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe driver_switches_without_robot.py > driver_switches_without_robot.out.txt
"""

import argparse
import inspect
import shutil
import sys
from pathlib import Path

import pandas
import torch

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
torch.set_num_threads(2)

import model_mode  # noqa: E402
from config import MODEL_MODE_PARAMETERS  # noqa: E402

HERE = Path(__file__).resolve().parent
LOGS = HERE / "simulated_logs"
shutil.rmtree(LOGS, ignore_errors = True)
RUNS = ROOT / "04_runs" / "gpu_ki014"
FOLD = RUNS / "2026-11-10_resnet18_chunk_stop_weight_25"

print("1. what the command line of model_mode.py offers today (read from its source):")
source = inspect.getsource(model_mode.main)
flags = [line.split('"')[1] for line in source.splitlines() if "add_argument(" in line]
print("   ", flags)
for wanted, key in (("the fixed step's length", "step_m"), ("the stall rule's numbers", "stall_steps"), ("a start pose to drive to", "start"), ("a travel budget for short runs", "travel")):
    print(f"   a flag for {wanted}: {'yes' if any(key.replace('_', '-') in flag for flag in flags) else 'NO (config.py or new code)'}")
print(f"   settings in config.py today: step_m {MODEL_MODE_PARAMETERS['step_m']}, step_length_rule {MODEL_MODE_PARAMETERS['step_length_rule']}, step_most_m {MODEL_MODE_PARAMETERS['step_most_m']}, "
      f"stall {MODEL_MODE_PARAMETERS['stall_steps']} pictures / {MODEL_MODE_PARAMETERS['stall_extent_m']} m / {MODEL_MODE_PARAMETERS['stall_twist_deg']} deg, step budget {MODEL_MODE_PARAMETERS['step_budget']}, "
      f"executor check from {MODEL_MODE_PARAMETERS['executor_check_smallest_step_m']} m")
print(f"   robot_run's start: {'hand-placed only (set_free_motion, the person guides the tip)' if 'set_free_motion(True)' in inspect.getsource(model_mode.robot_run) else '?'}; "
      f"a function that drives the real arm to a start pose: {'yes' if any('start' in name and name not in ('start_pose',) for name, _ in inspect.getmembers(model_mode, inspect.isfunction) if name not in ('start_pose',)) else 'NO'}")
print(f"   summary.json records the code version: {'yes' if 'commit' in inspect.getsource(model_mode.ModelRun.run) else 'NO'}; the sheet or the start: "
      f"{'yes' if 'sheet' in inspect.getsource(model_mode.ModelRun.run) else 'NO (only the free-text label)'}")

print("\n2. the model trained without sheet 03 through the whole loop on the simulated robot, the plan's numbers as overrides:")
policy = model_mode.ModelPolicy("resnet18_chunk_stop_weight_25", FOLD / "checkpoints" / "without_path_03.pt", FOLD / "metrics.json")
settings = {
    "fixed 1.0 cm (the control)": {"step_length_rule": "fixed", "step_m": 0.010},
    "fixed 0.65 cm": {"step_length_rule": "fixed", "step_m": 0.0065},
    "fixed 0.2 cm (block 0)": {"step_length_rule": "fixed", "step_m": 0.002},
    "the model's own length": {"step_length_rule": "predicted", "step_length_gain": 1.0, "step_most_m": 0.010},
}
for name, overrides in settings.items():
    overrides = dict(overrides, run_folder = LOGS, step_budget = 14, stall_steps = 15, stall_extent_m = 0.04, stall_twist_deg = 15.0)
    summary = model_mode.simulated_run(overrides, policy)
    steps = pandas.read_csv(Path(summary["folder"]) / "steps.csv")
    moved = steps[steps["action"] == "model_step"]
    following = steps.loc[[index + 1 for index in moved.index if index + 1 in steps.index]]
    kept = moved.loc[[index - 1 for index in following.index]]
    reached = ((following["tip_x"].to_numpy() - kept["tip_x"].to_numpy()) ** 2 + (following["tip_y"].to_numpy() - kept["tip_y"].to_numpy()) ** 2) ** 0.5 * 1000.0
    print(f"   {name:28s}: ending {summary['ending']:20s} pictures {summary['pictures']:2d}; commanded step {moved['step_cm'].min() * 10:.1f} to {moved['step_cm'].max() * 10:.1f} mm; "
          f"tip moved (simulated arm) {reached.min():.1f} to {reached.max():.1f} mm; checkpoint logged: {Path(summary['checkpoints'][0]).name}")
shutil.rmtree(LOGS, ignore_errors = True)
print("\n   so: step_m, the stall numbers and the budget are numbers the loop already takes (as overrides or in config.py); only the command-line flag is missing.")
