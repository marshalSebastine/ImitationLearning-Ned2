"""
Proofreader sitting_a, check 11: do the command lines of sitting A parse and run with the flags that exist today?

Each command is given to model_mode.main() exactly as it would be typed from 02_code, but with the part "simulated"
in place of "robot" (the same flags, the simulated robot, the simulated pictures: it checks the software path and the
model files, not the model's skill). The run folder is pointed into this folder first, so nothing is written into
03_data. Nothing connects to the robot.

Run:  D:/AI_Project/.venv/Scripts/python.exe -B p11_command_lines.py > p11_command_lines.out.txt
"""

import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
os.chdir(CODE)

import config                # noqa: E402
import model_mode            # noqa: E402

OUTPUT = HERE / "simulated_runs"
config.MODEL_MODE_PARAMETERS["run_folder"] = OUTPUT      # the same dictionary object model_mode reads

G = r"..\04_runs\gpu_ki014"
E = rf"--run-name resnet18_model_4_chunk_stop_25_all_sheets --checkpoint {G}\2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets\checkpoints\all_sheets.pt --metrics {G}\2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets\metrics.json"
TEAM = (rf"--run-name resnet18_model_1_all_sheets --checkpoint {G}\2026-11-10_resnet18_model_1_all_sheets\checkpoints\all_sheets.pt "
        + " ".join(rf"{G}\2026-11-10_resnet18_model_1_all_sheets_seed_{seed}\checkpoints\all_sheets.pt" for seed in (2, 3, 4, 5))
        + rf" --metrics {G}\2026-11-10_resnet18_model_1_all_sheets\metrics.json")
FOLD = rf"--run-name resnet18_chunk_stop_weight_25 --checkpoint {G}\2026-11-10_resnet18_chunk_stop_weight_25\checkpoints\without_path_03.pt --metrics {G}\2026-11-10_resnet18_chunk_stop_weight_25\metrics.json"
COMMANDS = {
    "A3 and A5: model E, the driver of 11.11 (fixed 1.0 cm)": f"--label A3_E_sheet01_fixed10 {E} --step-length fixed",
    "A4: model E, fixed 1.0 cm, short run": f"--label A4_E_fixed10 {E} --step-length fixed --step-budget 27",
    "A4: model E, its own length, short run": f"--label A4_E_own {E} --step-length predicted --step-budget 90",
    "A4: team of five, fixed 1.0 cm, short run": f"--label A4_team_fixed10 {TEAM} --step-length fixed --step-budget 27",
    "A4: model without sheet 03, fixed 1.0 cm, short run": f"--label A4_fold03_fixed10 {FOLD} --step-length fixed --step-budget 27",
}
for name, flags in COMMANDS.items():
    arguments = ["model_mode.py", "simulated"] + flags.split()
    sys.argv = arguments
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            model_mode.main()
        text = printed.getvalue()
        summary = json.loads(text[text.index("{"):])
        parameters = summary["parameters"]
        print(f"{name}\n   model_mode.py robot {flags}\n   -> runs: ending {summary['ending']} after {summary['pictures']} pictures; run_name {summary['run_name']}; {len(summary['checkpoints'])} checkpoint(s); "
              f"step rule {parameters['step_length_rule']}, step_m {parameters['step_m']}, budget {parameters['step_budget']}, stall rule {parameters['stall_steps']} pictures / {parameters['stall_extent_m']} m")
    except SystemExit as error:
        print(f"{name}\n   model_mode.py robot {flags}\n   -> THE COMMAND LINE IS REFUSED (exit {error.code})")
    except Exception as error:
        print(f"{name}\n   model_mode.py robot {flags}\n   -> FAILS: {type(error).__name__}: {str(error).splitlines()[0][:160]}")

print("\nflags that sitting A needs and that do not exist today:")
for flag in ("--step-cm 0.65", "--travel-budget-cm 27", "--start recorded", "--stall log", "--scripted small_steps", "--smallest-step-cm 0.2"):
    sys.argv = ["model_mode.py", "simulated"] + flag.split()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model_mode.main()
        print(f"   {flag}: accepted")
    except SystemExit:
        print(f"   {flag}: not known to the command line (new work)")
shutil.rmtree(OUTPUT, ignore_errors = True)
