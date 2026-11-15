"""
Safety review of the visit tools, check 7: can visit_runs.py start a robot run with a wrong model, a wrong sheet line or
the same label twice without the person noticing? Lines that should worry a person at the robot are given to it, and
what it says is printed.

NOTHING can be started here: every call is dry (--dry), and on top of that the starter and input() are replaced by
functions that refuse. The lists are written into the computer's temporary folder; the run folder under 03_data is only
looked at (for the label that was used on 11.11.26).

Also: what the metrics files of the ten all-sheets models hold, to judge what a mixed-up metrics file would do.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe s07_visit_runs_worrying_lines.py > s07_visit_runs_worrying_lines.out.txt
"""

import builtins
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))

import visit_runs  # noqa: E402
from config import MODEL_MODE_PARAMETERS  # noqa: E402


def refused_start(command):
    """Stands in the starter's place: nothing may be started in this check."""

    raise AssertionError(f"something tried to start {command[:3]}")


def refused_input(prompt = ""):
    """Stands in input's place: nothing may get as far as asking for Enter."""

    raise AssertionError("something asked for Enter")


visit_runs.run_in_this_terminal = refused_start
builtins.input = refused_input

MODEL_1 = MODEL_MODE_PARAMETERS["checkpoint_path"]
RUNS = MODEL_1.parent.parent.parent
MODEL_3 = RUNS / "2026-11-10_resnet18_model_3_16_rounds_all_sheets" / "checkpoints" / "all_sheets.pt"
MODEL_2_METRICS = RUNS / "2026-11-10_resnet18_model_2_chunk_all_sheets" / "metrics.json"


def from_code_folder(path):
    """``path`` as a run list gives it: relative to 02_code."""

    return os.path.relpath(path, visit_runs.CODE_FOLDER)


def sha256_start(path):
    """The first 12 characters of the file's sha256."""

    return visit_runs.sha256_of(path)[:12]


def list_line(**changes):
    """A good "robot" line of a run list, with ``changes`` laid over it."""

    line = {"label": "v1_check", "part": "robot", "sheet": "sheet 03", "lay_out": "STD", "model_key": "E", "run_name": MODEL_MODE_PARAMETERS["run_name"],
            "checkpoints": from_code_folder(MODEL_1), "checkpoint_sha256": sha256_start(MODEL_1), "metrics": from_code_folder(MODEL_MODE_PARAMETERS["metrics_path"]),
            "step_rule": "fixed", "step_cm": "0.65", "step_budget": "", "stall": "log", "ghost": "", "place_help": "yes", "note": ""}
    line.update(changes)
    return line


def said_about(line, folder, is_default_run_folder = False):
    """What visit_runs.py --dry says about ``line`` (its exit code and its words, without the long command)."""

    path = Path(folder) / "run_list.csv"
    with open(path, "w", newline = "", encoding = "utf-8") as file:
        writer = csv.DictWriter(file, fieldnames = visit_runs.RUN_LIST_COLUMNS)
        writer.writeheader()
        writer.writerow(line)
    arguments = [str(path), line["label"], "--dry"] + ([] if is_default_run_folder else ["--run-folder", folder])
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exit_code = visit_runs.main(arguments)
    words = printed.getvalue().split("dry: nothing is started")[0].rstrip()
    return exit_code, words


print("1. what visit_runs.py --dry says (exit code 0 = it would wait for Enter and then start the run):")
with tempfile.TemporaryDirectory() as folder:
    cases = (("a good robot line", list_line(), False),
             ("a robot line with no sheet, no lay-out and no model key", list_line(sheet = "", lay_out = "", model_key = ""), False),
             ("the model key says E, the file and its sha256 are model 3's", list_line(checkpoints = from_code_folder(MODEL_3), checkpoint_sha256 = sha256_start(MODEL_3)), False),
             ("model 1's checkpoint with the metrics file of model 2 (another recipe)", list_line(metrics = from_code_folder(MODEL_2_METRICS)), False),
             ("the same checkpoint twice as a team of 2", list_line(checkpoints = from_code_folder(MODEL_1) + ";" + from_code_folder(MODEL_1),
                                                                   checkpoint_sha256 = sha256_start(MODEL_1) + ";" + sha256_start(MODEL_1)), False),
             ("a step budget of 100000", list_line(step_budget = "100000"), False),
             ("the right path, the sha256 of another file", list_line(checkpoint_sha256 = sha256_start(MODEL_3)), False),
             ("a label that a robot run of 11.11.26 already has (the real run folder is looked at)", list_line(label = "E_sheet03_run1"), True),
             ("the same label in small letters", list_line(label = "e_sheet03_run1"), True))
    for words, line, is_default_run_folder in cases:
        exit_code, said = said_about(line, folder, is_default_run_folder)
        print(f"\n   -- {words}: exit code {exit_code}")
        for said_line in said.splitlines():
            print("      " + said_line[:210])

print("\n2. the metrics files of the all-sheets models (what a mixed-up metrics file would change):")
seen = set()
for folder in sorted(RUNS.iterdir()):
    if "all_sheets" not in folder.name or not (folder / "metrics.json").is_file():
        continue
    metrics = json.loads((folder / "metrics.json").read_text(encoding = "utf-8"))
    settings = metrics["settings"]
    used = json.dumps([settings["normalisation"], metrics["class_names"]])
    seen.add(used)
    print(f"   {folder.name}: name \"{metrics.get('name')}\", design {settings.get('design')}, backbone {settings.get('backbone')}, picture {settings.get('picture_size')}")
print(f"   what model_mode.py takes from a metrics file (the picture normalisation and the class names) has {len(seen)} form(s) over these files")
