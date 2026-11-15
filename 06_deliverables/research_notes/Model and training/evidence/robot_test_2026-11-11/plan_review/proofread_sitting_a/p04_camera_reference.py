"""
Proofreader sitting_a, check 4: which picture can serve as the camera reference for sitting A?

The driver holds the wrist at 13.5 and the hand at -6.1 degrees in every picture, and the camera sits on the forearm
cover (it turns with joint 4), so the robot's own hand and the block should sit at the same place in every picture of a
model run, whatever the arm's pose. If so, a first picture of 11.11.26 can be the reference, and a camera move between
11.11 and sitting A would show. Measured here with the recording tool's own check (record_episodes.camera_shift_px,
the bottom strip, phase correlation): the runs of 11.11 against run 2's first picture, and demonstrations from before
and after the camera moved (8.11.26 about 19:00) against the same picture.

Reads pictures only. Run:
    D:/AI_Project/.venv/Scripts/python.exe -B p04_camera_reference.py > p04_camera_reference.out.txt
"""

import math
import sys
from pathlib import Path

import numpy
import pandas

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
sys.path.insert(0, str(CODE))

from config import RECORDING_PARAMETERS          # noqa: E402
from record_episodes import camera_shift_px      # noqa: E402

RUNS = Path("D:/AI_Project/03_data/model_runs")
RAW = Path("D:/AI_Project/03_data/raw/main_data")
REGION = RECORDING_PARAMETERS["camera_check_region"]
reference_path = RUNS / "2026-11-11_172421_E_sheet01_run1" / "frames" / "000.jpg"
reference = reference_path.read_bytes()
print(f"reference: {reference_path} (run 2 of 11.11.26, first picture); region {REGION}; the recording tool warns above {RECORDING_PARAMETERS['camera_shift_warning_px']} px")


def shift(picture_path: Path) -> str:
    x, y = camera_shift_px(reference, picture_path.read_bytes(), REGION)
    return f"{math.hypot(x, y):5.1f} px (x {x:+5.1f}, y {y:+5.1f})"


print("\nthe ten runs of 11.11.26 against it (first picture, picture 10, last picture), with joints 4, 5, 6 in degrees at the first picture:")
values = []
for folder in sorted(RUNS.glob("2026-11-11_17*")):
    steps = pandas.read_csv(folder / "steps.csv")
    first = steps.iloc[0]
    texts = []
    for index in (0, min(10, len(steps) - 1), len(steps) - 1):
        x, y = camera_shift_px(reference, (folder / steps.iloc[index]["frame"]).read_bytes(), REGION)
        values.append(math.hypot(x, y))
        texts.append(f"{math.hypot(x, y):4.1f}")
    print(f"   {folder.name:42s} {' / '.join(texts)} px   joints 4, 5, 6: {math.degrees(first['joint_4']):+6.1f} {math.degrees(first['joint_5']):+5.1f} {math.degrees(first['joint_6']):+5.1f}")
print(f"   all 30 readings: median {numpy.median(values):.1f} px, largest {max(values):.1f} px")

print("\ndemonstrations against it (camera_check.jpg, taken at the recorded start pose):")
episodes = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/episodes.csv")
for plan in ("01-03", "01-04", "03-01", "03-04", "03-25", "04-01", "04-05", "04-13", "04-31", "04-33", "05-03", "05-14", "06-01", "06-14"):
    rows = episodes[episodes["episode_id"].str.contains(f"plan_{plan}_")]
    if rows.empty:
        print(f"   {plan}: not in the index")
        continue
    row = rows.iloc[-1]
    folder = RAW / row["episode_folder"]
    picture = folder / "camera_check.jpg"
    if not picture.exists():
        print(f"   {plan}: no camera_check.jpg in {folder}")
        continue
    print(f"   plan {plan} ({row['started']}, tool's own reading {row['camera_shift_px']:5.1f} px): {shift(picture)}")
