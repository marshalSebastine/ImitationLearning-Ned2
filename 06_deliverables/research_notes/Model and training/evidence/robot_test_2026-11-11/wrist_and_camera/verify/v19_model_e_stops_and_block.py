"""
Verifier, check 19:
    a) model E (trained on all six sheets, the model used on the robot) on the last two rows of every episode, as
       recorded and with my imitated held-wrist view (v09's "replicate" and "null"): would a wrist sweep with model E
       on one of the six sheets show anything?
    b) what moves by how much in the picture per degree of wrist when ONLY the wrist turns (forearm and camera still):
       the rod's end and the magnet tip, from the arm model and the fitted camera (the analyst's 1.81 and 2.63).

Inference with 2 threads on about 400 pictures.
    D:/AI_Project/.venv/Scripts/python.exe v19_model_e_stops_and_block.py
"""

import math
import sys

import cv2
import numpy
import pandas
import torch

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
import forearm_camera_model
import model_mode
import ned2_fk
from side_view_test import magnet_tip_and_lean
from train_policy import augmented_pictures
from training_targets import load_training_frames

MODEL = "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014/2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets"
CACHE = "D:/AI_Project/03_data/processed/v01_training_index/images_wide_160.npy"

frames = load_training_frames()
frames["offset"] = numpy.degrees(frames["joint_5"]) - 13.5
frames["shift_px"] = numpy.rint(0.701 * frames["offset"]).astype(int)
frames["from_end"] = frames.groupby("episode_id").cumcount(ascending = False)
last_two = frames[frames["from_end"] <= 1].copy()
images = numpy.load(CACHE, mmap_mode = "r")

generator = numpy.random.default_rng(11)
blue_count = numpy.zeros(images.shape[1:3])
sample = numpy.sort(generator.choice(len(images), 600, replace = False))
for index in sample:
    hsv = cv2.cvtColor(numpy.ascontiguousarray(images[index]), cv2.COLOR_RGB2HSV)
    blue_count += (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 126) & (hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 60)
fork = blue_count / len(sample) >= 0.8


def moved(picture, shift, way):
    height = picture.shape[0]
    if shift == 0:
        return picture.copy()
    if way == "null":
        result = picture.copy()
        if shift > 0:
            result[height - shift:] = picture[height - shift - 1]
        else:
            result[:-shift] = picture[-shift]
    else:
        result = cv2.warpAffine(numpy.ascontiguousarray(picture), numpy.float32([[1, 0, 0], [0, 1, shift]]), (picture.shape[1], height), flags = cv2.INTER_NEAREST,
                                borderMode = cv2.BORDER_REPLICATE)
    result[fork] = picture[fork]
    return result


policy = model_mode.ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", f"{MODEL}/checkpoints/all_sheets.pt", f"{MODEL}/metrics.json")
torch.set_num_threads(2)


def stop_chance(pictures):
    collected = []
    with torch.inference_mode():
        for start in range(0, len(pictures), 128):
            inputs, _ = augmented_pictures(torch.from_numpy(numpy.ascontiguousarray(pictures[start:start + 128])), policy.settings, numpy.random.default_rng(0), is_training = False)
            collected.append(torch.sigmoid(policy.models[0](inputs)["stop"][:, 0]).numpy())
    return numpy.concatenate(collected)


originals = numpy.stack([images[index] for index in last_two["frame_index"]])
shifts = last_two["shift_px"].to_numpy()
last_two["recorded"] = stop_chance(originals)
for way in ("replicate", "null"):
    last_two[way] = stop_chance(numpy.stack([moved(picture, int(shift), way) for picture, shift in zip(originals, shifts)]))
last_two["moved_bin"] = pandas.cut(last_two["shift_px"].abs(), [-1, 0, 4, 9, 40], labels = ["0", "1 to 4", "5 to 9", "10 or more"])
print("a) model E (all sheets), the last two rows of every episode: share with a stop chance of 0.5 or more")
for name, part in (("the stop row", last_two[last_two["from_end"] == 0]), ("the row before", last_two[last_two["from_end"] == 1])):
    print(f"   {name}:")
    print(part.groupby("moved_bin", observed = True).agg(rows = ("shift_px", "size"), recorded = ("recorded", lambda v: round(float((v >= 0.5).mean()), 3)),
                                                         replicate = ("replicate", lambda v: round(float((v >= 0.5).mean()), 3)),
                                                         null = ("null", lambda v: round(float((v >= 0.5).mean()), 3)),
                                                         chance_recorded = ("recorded", lambda v: round(float(v.mean()), 3)),
                                                         chance_replicate = ("replicate", lambda v: round(float(v.mean()), 3))).to_string())
both = last_two.pivot(index = "episode_id", columns = "from_end", values = ["recorded", "replicate", "null"])
for way in ("recorded", "replicate", "null"):
    print("   episodes with 0.5 or more on BOTH last rows (%s): %d of %d" % (way, int(((both[way][0] >= 0.5) & (both[way][1] >= 0.5)).sum()), len(both)))

print("\nb) only the wrist turns (forearm and camera still): move in the picture per degree of wrist, px at 640")
rows = frames.sample(300, random_state = 19)
tip_rates, end_rates = [], []
for _, row in rows.iterrows():
    joints = [row[f"joint_{index}"] for index in range(1, 7)]
    more = list(joints)
    more[4] += math.radians(1.0)
    tip_rates.append(forearm_camera_model.project_points(magnet_tip_and_lean(more)[0], joints)[0][1] - forearm_camera_model.project_points(magnet_tip_and_lean(joints)[0], joints)[0][1])
    end_rates.append(forearm_camera_model.project_points(numpy.array(ned2_fk.fk(more)["tool"][0]), joints)[0][1] - forearm_camera_model.project_points(numpy.array(ned2_fk.fk(joints)["tool"][0]), joints)[0][1])
print("   magnet tip: median %.2f (5th %.2f, 95th %.2f); rod's end: median %.2f (5th %.2f, 95th %.2f); negative = up the picture when the wrist grows" % (
    numpy.median(tip_rates), numpy.quantile(tip_rates, 0.05), numpy.quantile(tip_rates, 0.95), numpy.median(end_rates), numpy.quantile(end_rates, 0.05), numpy.quantile(end_rates, 0.95)))
