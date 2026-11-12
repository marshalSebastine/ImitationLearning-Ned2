"""
Stress test of the final candidate's saved fold models on their own held-out sheets (read only; nothing is trained).

The fold models of sheets 01 and 02 (resnet18_final_candidate, checkpoints/without_path_0*.pt) are run on their
held-out pictures as they are, and with changes the exam could bring and the training never showed:
    light       brightness x0.5 and x1.5, a warm and a cool colour cast of about 15 %;
    camera      the whole picture shifted by 4 px (inside the training shift) and by 8 px (twice the training shift,
                about 32 px at 640 x 480), a 3 degree roll, a blur of sigma 1 px at 160 x 120.
For each it reports the direction (within 45 degrees of the demonstrated move on moving pictures), the stop outcomes
of three rules, and the predicted step length, which no offline measure reports so far.

Run with python -B. Two processor threads, so the training queue keeps most of the processor.
"""

import math
import sys
import time

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

from pathlib import Path

import numpy
import pandas
import torch

from config import TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import PictureToMoveModel, merged_run_settings
from training_metrics import angle_difference_deg
from training_probes import design_outputs, predictions_from_outputs
from training_targets import DIRECTION_NAMES

RUN_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_resnet18_final_candidate")
FRAMES_PATH = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")
SHEETS = ["01", "02"]
TUNED_TWO_THRESHOLD = {"01": 0.4, "02": 0.1}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STANDARD_DEVIATION = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def shifted(batch: torch.Tensor, right_px: int, down_px: int) -> torch.Tensor:
    """Shift the whole picture, repeating the edge pixels, as the training shift does."""

    pad = max(abs(right_px), abs(down_px))
    padded = torch.nn.functional.pad(batch, (pad, pad, pad, pad), mode = "replicate")
    height, width = batch.shape[2:]
    top = pad - down_px
    left = pad - right_px
    return padded[:, :, top:top + height, left:left + width]


def rotated(batch: torch.Tensor, degrees: float) -> torch.Tensor:
    """Turn the whole picture about its centre, repeating the border."""

    angle = math.radians(degrees)
    height, width = batch.shape[2:]
    aspect = width / height
    matrix = torch.tensor([[math.cos(angle), -math.sin(angle) / aspect, 0.0],
                           [math.sin(angle) * aspect, math.cos(angle), 0.0]], dtype = torch.float32)
    grid = torch.nn.functional.affine_grid(matrix.unsqueeze(0).expand(batch.shape[0], 2, 3), list(batch.shape), align_corners = False)
    return torch.nn.functional.grid_sample(batch, grid, padding_mode = "border", align_corners = False)


def blurred(batch: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur with a separable Gaussian kernel."""

    radius = max(1, int(math.ceil(3 * sigma)))
    positions = torch.arange(-radius, radius + 1, dtype = torch.float32)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = batch.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    batch = torch.nn.functional.conv2d(torch.nn.functional.pad(batch, (radius, radius, 0, 0), mode = "replicate"), horizontal, groups = channels)
    return torch.nn.functional.conv2d(torch.nn.functional.pad(batch, (0, 0, radius, radius), mode = "replicate"), vertical, groups = channels)


CONDITIONS = {
    "clean": lambda batch: batch,
    "brightness x0.5": lambda batch: batch * 0.5,
    "brightness x1.5": lambda batch: batch * 1.5,
    "warm cast (R x1.15, B x0.85)": lambda batch: batch * torch.tensor([1.15, 1.0, 0.85]).view(1, 3, 1, 1),
    "cool cast (R x0.87, B x1.15)": lambda batch: batch * torch.tensor([0.87, 1.0, 1.15]).view(1, 3, 1, 1),
    "shift 4 px right and down": lambda batch: shifted(batch, 4, 4),
    "shift 8 px right and down": lambda batch: shifted(batch, 8, 8),
    "shift 8 px left and up": lambda batch: shifted(batch, -8, -8),
    "roll 3 degrees": lambda batch: rotated(batch, 3.0),
    "blur sigma 1 px": lambda batch: blurred(batch, 1.0),
}


def episode_outcomes(stop_probability, goal_distance_cm, episode_ids, threshold, consecutive_pictures):
    """Return (right, early, never) episode counts, as training_metrics.stop_metrics counts them."""

    right = early = never = 0
    for episode_id in numpy.unique(episode_ids):
        rows = numpy.flatnonzero(episode_ids == episode_id)
        is_passing = stop_probability[rows] >= threshold
        run_length = numpy.zeros(len(rows), dtype = int)
        for position in range(len(rows)):
            run_length[position] = ((run_length[position - 1] + 1) if position > 0 else 1) if is_passing[position] else 0
        passing = rows[run_length >= consecutive_pictures]
        if len(passing) == 0:
            never += 1
        elif goal_distance_cm[passing[0]] <= 2.0:
            right += 1
        elif goal_distance_cm[passing[0]] > 3.0:
            early += 1
    return right, early, never


def main():
    torch.set_num_threads(2)
    settings = merged_run_settings("resnet18_final_candidate")
    settings["is_pretrained"] = False
    settings["chunk_steps"] = TRAINING_TARGET_PARAMETERS["chunk_steps"]
    class_names = DIRECTION_NAMES + ["stop"]
    output_sizes = design_outputs(settings["design"], len(class_names), settings["chunk_steps"], settings.get("is_stop_separate", False))
    frames = pandas.read_csv(FRAMES_PATH, low_memory = False, usecols = [
        "frame_index", "episode_id", "action", "goal_distance_cm", "move_angle_deg", "move_length_m"]).set_index("frame_index")
    cache = numpy.load(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / f"images_{settings['picture_cache']}.npy", mmap_mode = "r")
    for sheet in SHEETS:
        fold = f"without_path_{sheet}"
        saved = dict(numpy.load(RUN_FOLDER / f"predictions_{fold}.npz"))
        model = PictureToMoveModel(settings, output_sizes)
        model.load_state_dict(torch.load(RUN_FOLDER / "checkpoints" / f"{fold}.pt", map_location = "cpu"))
        model.eval()
        rows = frames.loc[saved["frame_index"]]
        pictures = torch.from_numpy(numpy.array(cache[saved["frame_index"]]))
        is_moving = rows["action"].isin(DIRECTION_NAMES).to_numpy()
        true_angle = rows["move_angle_deg"].to_numpy()
        goal_distance = rows["goal_distance_cm"].fillna(0.0).to_numpy()
        episode_ids = rows["episode_id"].to_numpy()
        demonstrated_length_cm = 100.0 * rows["move_length_m"].to_numpy()
        print(f"\n== sheet {sheet}: {len(rows)} held-out pictures, {is_moving.sum()} moving, {len(numpy.unique(episode_ids))} episodes", flush = True)
        for name, change in CONDITIONS.items():
            started = time.perf_counter()
            parts = []
            with torch.inference_mode():
                for start in range(0, len(pictures), 128):
                    batch = pictures[start:start + 128].permute(0, 3, 1, 2).float().div(255.0)
                    batch = change(batch).clamp(0.0, 1.0)
                    parts.append(model((batch - MEAN) / STANDARD_DEVIATION))
            outputs = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
            predictions = predictions_from_outputs(outputs, class_names, settings["stop_target"])
            exact = outputs["exact"].numpy()
            length_cm = numpy.hypot(exact[:, 0], exact[:, 1])
            errors = angle_difference_deg(true_angle[is_moving], predictions["angle_deg"][is_moving])
            stop_probability = predictions["stop_probability"]
            fixed_one = episode_outcomes(stop_probability, goal_distance, episode_ids, 0.5, 1)
            fixed_two = episode_outcomes(stop_probability, goal_distance, episode_ids, 0.3, 2)
            tuned_two = episode_outcomes(stop_probability, goal_distance, episode_ids, TUNED_TWO_THRESHOLD[sheet], 2)
            line = (f"   {name:30s} within 45 {numpy.mean(errors <= 45):.3f} | stop right/early/never (0.5, 1) {fixed_one}"
                    f" (0.3, 2) {fixed_two} (tuned {TUNED_TWO_THRESHOLD[sheet]}, 2) {tuned_two}"
                    f" | median predicted step on moving pictures {numpy.median(length_cm[is_moving]):.2f} cm"
                    f" | {time.perf_counter() - started:.0f} s")
            if name == "clean":
                line += (f"\n      check against the saved predictions: largest stop difference"
                         f" {numpy.abs(stop_probability - saved['stop_probability']).max():.2e}, largest angle difference"
                         f" {angle_difference_deg(predictions['angle_deg'], saved['angle_deg']).max():.2e} deg")
                for low, high in ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 1000.0)):
                    is_band = is_moving & (goal_distance > low) & (goal_distance <= high)
                    if is_band.any():
                        line += (f"\n      moving pictures {low:.0f} to {high:.0f} cm from the goal: {is_band.sum()};"
                                 f" predicted step median {numpy.median(length_cm[is_band]):.2f} cm, demonstrated"
                                 f" {numpy.median(demonstrated_length_cm[is_band]):.2f} cm; predicted under 0.3 cm"
                                 f" {numpy.mean(length_cm[is_band] < 0.3):.3f}, under 0.5 cm {numpy.mean(length_cm[is_band] < 0.5):.3f}")
                is_stop = rows["action"].to_numpy() == "stop"
                line += (f"\n      stop pictures: {is_stop.sum()}; predicted step median {numpy.median(length_cm[is_stop]):.2f} cm;"
                         f" all moving pictures: predicted median {numpy.median(length_cm[is_moving]):.2f} cm against"
                         f" demonstrated {numpy.median(demonstrated_length_cm[is_moving]):.2f} cm,"
                         f" ratio of means {length_cm[is_moving].mean() / demonstrated_length_cm[is_moving].mean():.2f}")
            print(line, flush = True)


if __name__ == "__main__":
    main()
