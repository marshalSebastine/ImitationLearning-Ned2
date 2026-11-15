"""
Review attack 10: visit_runs.py at its own Enter question, for a ROBOT line, with the starter replaced by one that only
notes what it was given (nothing is started, no robot can be reached): Ctrl+C, no keyboard, Enter.
"""

import builtins
import contextlib
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

import visit_runs
from config import MODEL_MODE_PARAMETERS

CHECKPOINT = MODEL_MODE_PARAMETERS["checkpoint_path"]
line = {"label": "robot_line", "part": "robot", "sheet": "sheet 03", "lay_out": "STD", "model_key": "E", "run_name": MODEL_MODE_PARAMETERS["run_name"],
        "checkpoints": os.path.relpath(CHECKPOINT, visit_runs.CODE_FOLDER), "checkpoint_sha256": visit_runs.sha256_of(CHECKPOINT)[:12],
        "metrics": os.path.relpath(MODEL_MODE_PARAMETERS["metrics_path"], visit_runs.CODE_FOLDER), "step_rule": "fixed", "step_cm": "0.65", "step_budget": "",
        "stall": "", "ghost": "", "place_help": "yes", "note": ""}


def raising(error):

    def answer(prompt = ""):
        raise error

    return answer


with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "list.csv"
    with open(path, "w", newline = "", encoding = "utf-8") as file:
        writer = csv.DictWriter(file, fieldnames = visit_runs.RUN_LIST_COLUMNS)
        writer.writeheader()
        writer.writerow(line)
    for name, answer in (("Ctrl+C at the question", raising(KeyboardInterrupt())), ("no keyboard (end of input)", raising(EOFError())), ("Enter", lambda prompt = "": "")):
        started = []
        real = (visit_runs.run_in_this_terminal, builtins.input)
        visit_runs.run_in_this_terminal = lambda command: started.append(command) or 0
        builtins.input = answer
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                code = visit_runs.main([str(path), "robot_line", "--run-folder", folder])
        finally:
            visit_runs.run_in_this_terminal, builtins.input = real
        print(f"--- {name}: exit code {code}; started {len(started)} command(s)" + (f": ...{' '.join(started[0][2:5])} ... {' '.join(started[0][-4:])}" if started else ""))
        print("    printed:", [text for text in printed.getvalue().splitlines() if "REAL ROBOT moves" in text or "not started" in text or "starting model_mode" in text])
