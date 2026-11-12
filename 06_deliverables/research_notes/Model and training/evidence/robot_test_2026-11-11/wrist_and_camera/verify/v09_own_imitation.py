"""
Verifier, check 9: my OWN imitation of the driver's view (wrist held at 13.5 degrees) for the fold models of the E recipe,
to see how far the analyst's "right stops fall from 103 to 90 of 207" depends on how the view is imitated.

Every held-out picture (160 x 120 from the training cache) is moved up or down by its own whole number of pixels,
0.701 px per degree of (demonstrated wrist - 13.5), calculated in v02. Three ways of filling what the move uncovers:
    replicate   the edge row is repeated (what a plain shift does)
    reflect     the picture is mirrored at the edge (keeps table texture instead of smear)
    null        NO move at all, but the rows a move would have pushed out of the picture are smeared in place
                (control: how much is lost only because real content at the edge is destroyed)
The blue fork (part of the forearm, fixed in the picture) is kept where it is in "replicate" and "reflect": pixels that
are saturated blue in at least 80 % of 600 random pictures.
Rows whose move rounds to 0 px keep their saved prediction. Inference with 2 threads; the first part checks that my
inference gives back the saved stop chances.

Run from this folder (about 4 to 6 minutes):
    D:/AI_Project/.venv/Scripts/python.exe v09_own_imitation.py
"""

import sys
import time

import cv2
import numpy
import pandas
import torch

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
import model_mode
from train_policy import augmented_pictures
from training_metrics import stop_metrics
from training_probes import predictions_from_outputs
from training_targets import load_training_frames

RUN_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014/2026-11-10_resnet18_chunk_stop_weight_25"
RUN_NAME = "resnet18_chunk_stop_weight_25"
CACHE = "D:/AI_Project/03_data/processed/v01_training_index/images_wide_160.npy"
PX_PER_DEGREE_160 = 0.701
HELD_WRIST_DEG = 13.5
BATCH = 128

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
frames = load_training_frames()
frames["offset"] = numpy.degrees(frames["joint_5"]) - HELD_WRIST_DEG
frames["shift_px"] = numpy.rint(PX_PER_DEGREE_160 * frames["offset"]).astype(int)      # positive: content moves DOWN in the held view
frames["is_start_mark"] = frames["sheet_orientation"].str.contains("START MARK", na = False)
images = numpy.load(CACHE, mmap_mode = "r")
print("cache", images.shape, images.dtype)

generator = numpy.random.default_rng(11)
sample = numpy.sort(generator.choice(len(images), 600, replace = False))
blue_count = numpy.zeros(images.shape[1:3])
for index in sample:
    hsv = cv2.cvtColor(numpy.ascontiguousarray(images[index]), cv2.COLOR_RGB2HSV)
    blue_count += (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 126) & (hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 60)
fork = blue_count / len(sample) >= 0.8
print("fork mask: %d pixels (%.1f %% of the picture), rows %d to %d" % (int(fork.sum()), 100 * fork.mean(), numpy.flatnonzero(fork.any(axis = 1)).min(), numpy.flatnonzero(fork.any(axis = 1)).max()))


def moved(picture: numpy.ndarray, shift: int, way: str) -> numpy.ndarray:
    """The picture with its content moved ``shift`` px down (negative: up), the uncovered rows filled as ``way`` says."""

    height = picture.shape[0]
    if way == "null":
        result = picture.copy()
        if shift > 0:
            result[height - shift:] = picture[height - shift - 1]
        elif shift < 0:
            result[:-shift] = picture[-shift]
        result[fork] = picture[fork]
        return result
    border = cv2.BORDER_REPLICATE if way == "replicate" else cv2.BORDER_REFLECT_101
    matrix = numpy.float32([[1, 0, 0], [0, 1, shift]])
    result = cv2.warpAffine(numpy.ascontiguousarray(picture), matrix, (picture.shape[1], height), flags = cv2.INTER_NEAREST, borderMode = border)
    result[fork] = picture[fork]
    return result


def answers(policy, pictures: numpy.ndarray) -> numpy.ndarray:
    """Stop chance and exact-step direction (degrees) for uint8 RGB pictures (N, 120, 160, 3)."""

    collected = []
    with torch.inference_mode():
        for start in range(0, len(pictures), BATCH):
            inputs, _ = augmented_pictures(torch.from_numpy(numpy.ascontiguousarray(pictures[start:start + BATCH])), policy.settings, numpy.random.default_rng(0), is_training = False)
            outputs = policy.models[0](inputs)
            predictions = predictions_from_outputs(outputs, policy.class_names, policy.settings["stop_target"])
            exact = outputs["exact"].numpy()
            collected.append(numpy.column_stack([numpy.asarray(predictions["stop_probability"]), numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0]))]))
    return numpy.concatenate(collected)


results = []
started = time.time()
for sheet_number in range(1, 7):
    fold = f"without_path_{sheet_number:02d}"
    saved = numpy.load(f"{RUN_FOLDER}/predictions_{fold}.npz")
    part = frames.merge(pandas.DataFrame({"frame_index": saved["frame_index"], "stop_recorded": saved["stop_probability"]}), on = "frame_index", how = "inner")
    part = part.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    policy = model_mode.ModelPolicy(RUN_NAME, f"{RUN_FOLDER}/checkpoints/{fold}.pt", f"{RUN_FOLDER}/metrics.json")
    torch.set_num_threads(2)
    check_rows = part.iloc[numpy.linspace(0, len(part) - 1, 64).astype(int)]
    check = answers(policy, numpy.stack([images[index] for index in check_rows["frame_index"]]))
    print("%s: %d rows; my inference against the saved stop chances on 64 rows: largest gap %.6f" % (fold, len(part), numpy.abs(check[:, 0] - check_rows["stop_recorded"].to_numpy()).max()))
    is_moved = part["shift_px"].to_numpy() != 0
    originals = numpy.stack([images[index] for index in part.loc[is_moved, "frame_index"]])
    shifts = part.loc[is_moved, "shift_px"].to_numpy()
    recorded_angle = answers(policy, originals)[:, 1]
    part["angle_recorded"] = numpy.nan
    part.loc[is_moved, "angle_recorded"] = recorded_angle
    for way in ("replicate", "reflect", "null"):
        changed = numpy.stack([moved(picture, int(shift), way) for picture, shift in zip(originals, shifts)])
        output = answers(policy, changed)
        part[f"stop_{way}"] = part["stop_recorded"]
        part.loc[is_moved, f"stop_{way}"] = output[:, 0]
        part[f"angle_{way}"] = numpy.nan
        part.loc[is_moved, f"angle_{way}"] = output[:, 1]
    results.append(part)
    print("   moved rows: %d (%.0f %%); elapsed %.0f s" % (int(is_moved.sum()), 100 * is_moved.mean(), time.time() - started))
scored = pandas.concat(results, ignore_index = True)
scored.insert(0, "episode_number", pandas.factorize(scored["episode_id"])[0] + 1)
keep = ["episode_number", "row_in_episode", "frame_index", "sheet", "is_start_mark", "action", "goal_distance_cm", "offset", "shift_px", "move_angle_deg", "move_length_m",
        "stop_recorded", "stop_replicate", "stop_reflect", "stop_null", "angle_recorded", "angle_replicate", "angle_reflect", "angle_null"]
scored[keep].to_csv("v09_own_imitation_rows.csv", index = False, float_format = "%.4f")

print("\n=== episodes by the stop rule (0.5 on two pictures in a row; the project's stop_metrics) ===")
is_stop = (scored["action"] == "stop").to_numpy()
count = scored["episode_id"].nunique()
for way in ("recorded", "replicate", "reflect", "null"):
    for label, mask in (("all", numpy.ones(len(scored), dtype = bool)), ("start mark", scored["is_start_mark"].to_numpy()), ("taught start", ~scored["is_start_mark"].to_numpy())):
        metrics = stop_metrics(is_stop[mask], scored.loc[mask, f"stop_{way}"].to_numpy(), scored.loc[mask, "goal_distance_cm"].to_numpy(), scored.loc[mask, "episode_id"].to_numpy(),
                               threshold = 0.5, consecutive_pictures = 2)
        episodes = scored.loc[mask, "episode_id"].nunique()
        print("  %-10s %-13s right %3.0f, early %3.0f, never %3.0f of %d" % (way, label, metrics["episodes_stopped_right"] * episodes, metrics["episodes_stopped_early"] * episodes,
                                                                             metrics["episodes_never_stopped"] * episodes, episodes))

print("\n=== stop rows: share with a stop chance of 0.5 or more, by how far the picture is moved ===")
stops = scored[is_stop].copy()
stops["moved_bin"] = pandas.cut(stops["shift_px"].abs(), [-1, 0, 4, 9, 40], labels = ["0", "1 to 4", "5 to 9", "10 or more"])
table = stops.groupby("moved_bin", observed = True).agg(rows = ("shift_px", "size"), **{way: (f"stop_{way}", lambda v: round(float((v >= 0.5).mean()), 3)) for way in ("recorded", "replicate", "reflect", "null")})
print(table.to_string())
stops["direction"] = numpy.where(stops["shift_px"] > 0, "content down (wrist above 13.5)", numpy.where(stops["shift_px"] < 0, "content up (wrist below 13.5)", "not moved"))
print(stops.groupby("direction").agg(rows = ("shift_px", "size"), **{way: (f"stop_{way}", lambda v: round(float((v >= 0.5).mean()), 3)) for way in ("recorded", "replicate", "reflect", "null")}).to_string())

print("\n=== direction: exact step within 45 degrees of the demonstrated step (moved rows that move the tip 5 mm or more) ===")
steps = scored[(scored["shift_px"] != 0) & (scored["action"] != "stop") & (scored["move_length_m"] >= 0.005)].copy()
for way in ("recorded", "replicate", "reflect", "null"):
    gap = (steps[f"angle_{way}"] - steps["move_angle_deg"] + 180.0) % 360.0 - 180.0
    big = steps["shift_px"].abs() > 4
    print("  %-10s within 45 degrees: %.1f %% of %d moved rows; rows moved by more than 4 px: %.1f %% of %d" % (
        way, 100 * (gap.abs() <= 45).mean(), len(steps), 100 * (gap[big].abs() <= 45).mean(), int(big.sum())))
print("total seconds: %.0f" % (time.time() - started))
