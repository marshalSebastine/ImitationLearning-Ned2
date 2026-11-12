"""
The verifier's own nearest-picture check: how well do the first 13 pictures of each robot run match their best
demonstration picture (grey 40 x 30 pictures, correlation after removing each picture's mean)? The analyst gave 0.75
for run 3 and 0.84 for run 4; here all ten runs are scored the same way, for scale.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v13_nearest_pictures.py > v13_nearest_pictures.out.txt
"""

import sys
from pathlib import Path

import cv2
import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from training_targets import load_training_frames  # noqa: E402

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
ROBOT_RUNS = [(1, "2026-11-11_172021_check_no_puck"), (2, "2026-11-11_172421_E_sheet01_run1"), (3, "2026-11-11_172743_E_sheet02_run1"),
              (4, "2026-11-11_173036_E_sheet02_run2"), (5, "2026-11-11_173549_E_sheet03_run1"), (6, "2026-11-11_174108_E_sheet04_run1"),
              (7, "2026-11-11_174448_E_sheet05_run1"), (8, "2026-11-11_174936_E_sheet06_run1"), (9, "2026-11-11_175203_E_sheet03_run2_no_hoodie"),
              (10, "2026-11-11_175544_B_team5_sheet03")]


def small_grey(picture_rgb):
    grey = cv2.cvtColor(picture_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, (40, 30), interpolation = cv2.INTER_AREA).astype(numpy.float32).ravel()


def unit(vectors):
    vectors = vectors - vectors.mean(axis = 1, keepdims = True)
    return vectors / (numpy.linalg.norm(vectors, axis = 1, keepdims = True) + 1e-9)


frames = load_training_frames({"is_audit_cleaning": True})
frames["start_kind"] = frames["start_kind"].fillna("STD")
cache = numpy.load("D:/AI_Project/03_data/processed/v01_training_index/images_wide_160.npy", mmap_mode = "r")
indices = frames["frame_index"].to_numpy()
library = unit(numpy.stack([small_grey(numpy.asarray(cache[index])) for index in indices]))
kinds = frames["start_kind"].to_numpy()
plans = frames["plan_number"].astype(str).to_numpy()
print(f"demonstration pictures in the library: {len(library)}")

for number, folder in ROBOT_RUNS:
    best, best_kinds = [], []
    for index in range(13):
        picture = cv2.imread(str(RUNS_FOLDER / folder / "frames" / f"{index:03d}.jpg"))
        if picture is None:
            continue
        small = cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (160, 120), interpolation = cv2.INTER_AREA)
        query = unit(small_grey(small)[None, :])[0]
        correlation = library @ query
        top = int(numpy.argmax(correlation))
        best.append(float(correlation[top]))
        best_kinds.append(f"{plans[top]} {kinds[top]}")
    counts = pandas.Series(best_kinds).str.split(" ").str[1].value_counts().to_dict()
    print(f"run {number:2d}: best correlation of the first {len(best)} pictures: median {numpy.median(best):.3f} (least {min(best):.3f}, most {max(best):.3f}); kinds of the best matches: {counts}")

# for scale: demonstration episodes against the rest of the library (their own episode left out), first 13 kept rows
generator = numpy.random.default_rng(2)
episodes = frames["episode_id"].to_numpy()
medians = []
for episode_id in generator.choice(pandas.unique(episodes), 40, replace = False):
    own = numpy.nonzero(episodes == episode_id)[0][:13]
    others = episodes != episode_id
    medians.append(float(numpy.median((library[own] @ library[others].T).max(axis = 1))))
print(f"for scale, 40 random demonstration episodes, first 13 kept rows against all other episodes: median of the medians {numpy.median(medians):.3f},"
      f" 10th to 90th percentile {numpy.percentile(medians, 10):.3f} to {numpy.percentile(medians, 90):.3f}")
