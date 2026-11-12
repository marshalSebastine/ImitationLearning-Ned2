"""
Review check: break a COPY of visit_runs.py and run its whole test file against each breakage. A breakage that no test
notices marks a behaviour the tests do not really check. Nothing here can start a robot line: the test file only runs
"simulated" lines for real.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
HERE = Path(tempfile.mkdtemp(prefix = "review_breakages_"))
source = (CODE / "visit_runs.py").read_text(encoding = "utf-8")
# the copy must still find the real model_mode.py and resolve the list's paths from the real 02_code
source = source.replace('CODE_FOLDER = Path(__file__).resolve().parent', f'CODE_FOLDER = Path(r"{CODE}")')
(HERE / "test_visit_runs.py").write_text((CODE / "test_visit_runs.py").read_text(encoding = "utf-8"), encoding = "utf-8")

MUTATIONS = [
    ("Ctrl+C at the Enter question starts the run all the same", '''    except (KeyboardInterrupt, EOFError):
        # no Enter from a person (Ctrl+C, or no keyboard to read from): nothing is started
        print("\\nnot started")
        return 1''', '''    except (KeyboardInterrupt, EOFError):
        pass'''),
    ("keys typed ahead are not thrown away before the Enter question", '''    if discard_typed_keys():
        print("keys pressed before this question were ignored")
''', ''),
    ("on Ctrl+C the starter kills the run instead of waiting for it", '''        except KeyboardInterrupt:
            continue''', '''        except KeyboardInterrupt:
            process.kill()
            return 1'''),
    ("only the first 4 characters of the sha256 are compared", 'if not found.startswith(characters):', 'if not found.startswith(characters[:4]):'),
    ("only a team's first file is checked against its sha256", 'for path, characters in zip(run["checkpoints"], listed_characters):', 'for path, characters in list(zip(run["checkpoints"], listed_characters))[:1]:'),
    ("--again is not needed: a label with a run folder is started again", 'if earlier_runs and not arguments.again:', 'if False:'),
    ("a robot line is started without --fresh-keys", '''    if run["part"] == "robot":
        # an Enter pressed while the model loads must not answer the first question, which switches the motors off
        command.append("--fresh-keys")
''', ''),
    ("the run is started from another folder than 02_code", 'process = subprocess.Popen(command, cwd = str(CODE_FOLDER))', 'process = subprocess.Popen(command)'),
    ("a missing metrics file is not noticed", '''    if run["metrics"] is None or not run["metrics"].is_file():''', '''    if run["metrics"] is None:'''),
    ("--dry starts the run after printing the command", '''        print(subprocess.list2cmdline(command))
        return 0''', '''        print(subprocess.list2cmdline(command))'''),
    ("the label is not passed on (model_mode names the folder itself)", '''run["part"], "--label", run["label"], "--run-name"''', '''run["part"], "--run-name"'''),
    ("the real-robot warning is not printed", '''    if run["part"] == "robot":
        print("THE REAL ROBOT moves in this run: the checklist done, nothing else connected to the robot, the stop button in reach.")
''', ''),
]

(HERE / "visit_runs.py").write_text(source, encoding = "utf-8")


def failing() -> str:

    try:
        finished = subprocess.run([sys.executable, "test_visit_runs.py"], cwd = str(HERE), capture_output = True, text = True, env = dict(os.environ, PYTHONPATH = str(CODE)), timeout = 300)
    except subprocess.TimeoutExpired:
        return "the tests did not end within 300 s"
    if finished.returncode == 0 and "all tests passed" in finished.stdout:
        return ""
    return (finished.stderr.strip().splitlines() or ["?"])[-1][:140]


print("unbroken copy:", failing() or "all tests passed", flush = True)
for title, old, new in MUTATIONS:
    if source.count(old) != 1:
        print(f"SKIPPED (text found {source.count(old)} times): {title}", flush = True)
        continue
    (HERE / "visit_runs.py").write_text(source.replace(old, new), encoding = "utf-8")
    outcome = failing()
    print(f"{'caught' if outcome else 'NOT CAUGHT'}: {title}" + (f" -> {outcome}" if outcome else ""), flush = True)
(HERE / "visit_runs.py").write_text(source, encoding = "utf-8")
