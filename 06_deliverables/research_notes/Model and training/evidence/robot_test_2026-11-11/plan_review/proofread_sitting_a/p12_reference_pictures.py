"""
Proofreader sitting_a, check 12: which demonstration's first picture can serve as the ghost for laying a sheet?

For the standard starts of sheets 01, 03 and 04 from end A: is the arm at the stored start joints in the first
picture, is the puck in that picture (red pixels by the recording tool's own colour limits), and where is it?
A first picture without the puck cannot show where the puck stood.

Reads pictures and frames.csv only. Run:
    D:/AI_Project/.venv/Scripts/python.exe -B p12_reference_pictures.py > p12_reference_pictures.out.txt
"""

import sys
from pathlib import Path

import cv2
import numpy
import pandas

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
sys.path.insert(0, str(CODE))
from config import RECORDING_PARAMETERS    # noqa: E402

RAW = Path("D:/AI_Project/03_data/raw/main_data")
INDEX = Path("D:/AI_Project/03_data/processed/v01_training_index")
STORED = numpy.array([-6.06, -41.87, -61.93, 0.0, 13.53, -6.15])
episodes = pandas.read_csv(INDEX / "episodes.csv")
frames = pandas.read_csv(INDEX / "frames.csv", usecols = ["episode_id", "row_in_episode", "frame_path"] + [f"joint_{index}" for index in range(1, 7)], low_memory = False)
low, high = RECORDING_PARAMETERS["puck_red_hue_edges"]


def red(path: Path):
    hsv = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] < low) | (hsv[..., 0] > high)) & (hsv[..., 1] >= RECORDING_PARAMETERS["puck_red_min_saturation"]) & (hsv[..., 2] >= RECORDING_PARAMETERS["puck_red_min_value"])
    if int(mask.sum()) < RECORDING_PARAMETERS["puck_red_min_area_px"]:
        return None
    ys, xs = numpy.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


for plans in (["01-01", "01-02", "01-03", "01-04"], ["03-01", "03-02", "03-03", "03-04", "03-05"], ["04-01", "04-02", "04-03", "04-04", "04-05"]):
    for plan in plans:
        row = episodes[episodes["episode_id"].str.contains(f"plan_{plan}_")].iloc[-1]
        rows = frames[frames["episode_id"] == row["episode_id"]].sort_values("row_in_episode")
        first = rows.iloc[0]
        off = float(numpy.abs(numpy.degrees([first[f"joint_{index}"] for index in range(1, 7)]) - STORED).max())
        picture = RAW / first["frame_path"]
        puck = red(picture)
        check = RAW / row["episode_folder"] / "camera_check.jpg"
        puck_check = red(check) if check.exists() else None
        print(f"plan {plan} (score {row['score']}, {row['lighting']}, camera reading then {row['camera_shift_px']} px): first picture {first['frame_path']}; "
              f"joints {off:.2f} deg from the stored start; puck in the first picture: {'no' if puck is None else f'yes, at ({puck[0]:.0f}, {puck[1]:.0f}) px'}; "
              f"in camera_check.jpg: {'no' if puck_check is None else f'yes, at ({puck_check[0]:.0f}, {puck_check[1]:.0f}) px'}")
    print()
