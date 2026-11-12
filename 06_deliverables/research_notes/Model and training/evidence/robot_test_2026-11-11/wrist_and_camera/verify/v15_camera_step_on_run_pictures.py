"""
Verifier, check 15: would model E have answered differently on the robot test's own pictures if the camera still sat
where it was for sheets 01 to 03 and most of 04 (before the step of 8.11.26)? The real run pictures (640 x 480) are
moved back by the measured step (15 px to the left and 5 px up at 640, the whole picture: a camera pan moves the fork
too), then go through the same decoding, shrinking and normalisation as on the robot. Control: the same move the other
way. With no move my inference must give back the answers logged in steps.csv.

This is an imitation on real pictures; inference only, 2 threads.
    D:/AI_Project/.venv/Scripts/python.exe v15_camera_step_on_run_pictures.py
"""

import glob
import sys

import cv2
import numpy
import pandas
import torch

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
import model_mode
from train_policy import augmented_pictures

RUNS = "D:/AI_Project/03_data/model_runs"
MODEL = "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014/2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets"
STEP_BACK = (-15, -5)        # px at 640: content to the left and up = the camera back at its place before the step

policy = model_mode.ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", f"{MODEL}/checkpoints/all_sheets.pt", f"{MODEL}/metrics.json")
torch.set_num_threads(2)
width, height = policy.settings["picture_size"]


def answers(pictures_bgr: list, move: tuple) -> numpy.ndarray:
    """Forward cm, left cm, twist, stop chance for 640 x 480 BGR pictures moved by ``move`` px (x right, y down)."""

    small = []
    for picture in pictures_bgr:
        if move != (0, 0):
            picture = cv2.warpAffine(picture, numpy.float32([[1, 0, move[0]], [0, 1, move[1]]]), (picture.shape[1], picture.shape[0]), flags = cv2.INTER_LINEAR,
                                     borderMode = cv2.BORDER_REPLICATE)
        small.append(cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (width, height), interpolation = cv2.INTER_AREA))
    inputs, _ = augmented_pictures(torch.from_numpy(numpy.stack(small)), policy.settings, numpy.random.default_rng(0), is_training = False)
    with torch.inference_mode():
        outputs = policy.models[0](inputs)
    exact = outputs["exact"].numpy()
    return numpy.column_stack([exact[:, 0], exact[:, 1], exact[:, 2], torch.sigmoid(outputs["stop"][:, 0]).numpy()])


pandas.set_option("display.width", 250)
rows = []
for folder in sorted(glob.glob(f"{RUNS}/2026-11-11_17*_E_*")):
    name = folder.replace("\\", "/").split("/")[-1][11:]
    steps = pandas.read_csv(f"{folder}/steps.csv")
    pictures = [cv2.imdecode(numpy.fromfile(f"{folder}/{frame}", dtype = numpy.uint8), cv2.IMREAD_COLOR) for frame in steps["frame"]]
    plain = answers(pictures, (0, 0))
    back = answers(pictures, STEP_BACK)
    other = answers(pictures, (-STEP_BACK[0], -STEP_BACK[1]))
    logged = steps[["model_tip_forward_cm", "model_tip_left_cm", "model_twist_deg", "model_stop_probability"]].to_numpy()
    angle = lambda values: numpy.degrees(numpy.arctan2(values[:, 1], values[:, 0]))
    length = lambda values: numpy.hypot(values[:, 0], values[:, 1])
    turn = lambda a, b: numpy.abs((angle(a) - angle(b) + 180.0) % 360.0 - 180.0)
    first = numpy.arange(len(steps)) < 15
    rows.append({"run": name, "pictures": len(steps), "check_largest_gap_to_log": float(numpy.abs(plain - logged).max()),
                 "direction_turn_median_deg": float(numpy.median(turn(back, plain))), "direction_turn_p90_deg": float(numpy.quantile(turn(back, plain), 0.9)),
                 "other_way_turn_median_deg": float(numpy.median(turn(other, plain))),
                 "twist_mean_abs_plain": float(numpy.abs(plain[:, 2]).mean()), "twist_mean_abs_back": float(numpy.abs(back[:, 2]).mean()),
                 "twist_first15_plain": float(plain[first, 2].sum()), "twist_first15_back": float(back[first, 2].sum()), "twist_first15_other": float(other[first, 2].sum()),
                 "length_median_plain_cm": float(numpy.median(length(plain))), "length_median_back_cm": float(numpy.median(length(back))),
                 "stop_pictures_plain": int((plain[:, 3] >= 0.5).sum()), "stop_pictures_back": int((back[:, 3] >= 0.5).sum())})
table = pandas.DataFrame(rows)
print(table.round(2).T.to_string())
print("\nsummed twist asked over the first 15 pictures, as seen against the camera moved back: sheets recorded before the step are 01, 02, 03 and 72 % of 04")
