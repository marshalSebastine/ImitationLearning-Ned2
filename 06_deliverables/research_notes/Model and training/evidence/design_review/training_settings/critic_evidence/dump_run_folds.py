"""Print the scalar fields of end-to-end fold files, their settings and loss histories.

Read only: opens the fold and metrics JSON files of 04_runs and prints them.
"""

import json
import pathlib
import sys

RUNS_FOLDER = pathlib.Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")


def print_scalars(dictionary, prefix = ""):
    """Print every scalar of a nested dictionary, one per line, with its dotted key."""
    for key, value in dictionary.items():
        full_key = prefix + str(key)
        if isinstance(value, dict):
            print_scalars(value, prefix = full_key + ".")
        elif isinstance(value, list):
            text = str(value)
            if len(text) > 2400:
                text = text[:2400] + " ... (%d items)" % len(value)
            print("   %s = %s" % (full_key, text))
        else:
            print("   %s = %s" % (full_key, value))


def print_run(run_name):
    """Print every fold file of one run folder, then its metrics file if present."""
    run_folder = RUNS_FOLDER / run_name
    for path in sorted(run_folder.glob("*.json")):
        print("=" * 100)
        print(path.name, "in", run_name)
        content = json.loads(path.read_text(encoding = "utf-8"))
        print_scalars(content)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print_run(name)
