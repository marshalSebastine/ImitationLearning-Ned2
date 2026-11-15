"""Scratch helper: run a Python script in a new console that is not shown, wait for it, and print what it wrote."""

import subprocess
import sys
from pathlib import Path

script, result_path = sys.argv[1], Path(sys.argv[2])
if result_path.exists():
    result_path.unlink()
startup = subprocess.STARTUPINFO()
startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
startup.wShowWindow = 0
finished = subprocess.run([sys.executable, script, str(result_path)] + sys.argv[3:], creationflags = subprocess.CREATE_NEW_CONSOLE, startupinfo = startup, timeout = 240)
print("exit code of the hidden console's script:", finished.returncode)
print(result_path.read_text(encoding = "utf-8") if result_path.exists() else "(no result file was written)")
