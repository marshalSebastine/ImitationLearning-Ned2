"""
Saved models on many pictures at once, without the robot (dashed-line analysis, robot test of 11.11.26).

A model is loaded exactly as the robot driver loads it (model_mode.ModelPolicy: the run's recipe, the picture
normalisation saved with its metrics), and gives for every picture what the driver would read: the exact step
(tip forward and left in cm in the picture's frame, forearm twist in degrees), the stop chance, and for a chunk model
also the five future tip places. Only two processor threads are used.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy
import torch

from dashed_common import ANALYSIS_PARAMETERS
from model_mode import ModelPolicy
from train_policy import augmented_pictures

MODEL_PARAMETERS = {
    "threads": 2,
    "batch_size": 128,
    # the saved models used here: folder under 04_runs/gpu_ki014, the run's name in TRAINING_RUNS, the checkpoint
    "models": {
        "E all sheets": ("2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets", "resnet18_model_4_chunk_stop_25_all_sheets", "all_sheets"),
        "E recipe without 05": ("2026-11-10_resnet18_chunk_stop_weight_25", "resnet18_chunk_stop_weight_25", "without_path_05"),
        "E recipe without 06": ("2026-11-10_resnet18_chunk_stop_weight_25", "resnet18_chunk_stop_weight_25", "without_path_06"),
        "E recipe without 05, seed 2": ("2026-11-10_resnet18_chunk_stop_weight_25_seed_2", "resnet18_chunk_stop_weight_25_seed_2", "without_path_05"),
        "E recipe without 06, seed 2": ("2026-11-10_resnet18_chunk_stop_weight_25_seed_2", "resnet18_chunk_stop_weight_25_seed_2", "without_path_06"),
        "A recipe without 05": ("2026-11-09_resnet18_final_candidate", "resnet18_final_candidate", "without_path_05"),
        "A recipe without 06": ("2026-11-09_resnet18_final_candidate", "resnet18_final_candidate", "without_path_06"),
        "A all sheets": ("2026-11-10_resnet18_model_1_all_sheets", "resnet18_model_1_all_sheets", "all_sheets")
    }
}



def loaded_policy(model_name: str) -> ModelPolicy:
    """Return one of MODEL_PARAMETERS["models"] loaded as the robot driver loads it."""

    folder_name, run_name, checkpoint_name = MODEL_PARAMETERS["models"][model_name]
    folder = Path(ANALYSIS_PARAMETERS["gpu_runs_folder"]) / folder_name
    policy = ModelPolicy(run_name, folder / "checkpoints" / f"{checkpoint_name}.pt", folder / "metrics.json")
    torch.set_num_threads(MODEL_PARAMETERS["threads"])
    return policy



def saved_predictions(model_name: str) -> Optional[Dict[str, numpy.ndarray]]:
    """Return the predictions saved when a fold model was scored on its held-out sheet (None for an all-sheets model)."""

    folder_name, _, checkpoint_name = MODEL_PARAMETERS["models"][model_name]
    path = Path(ANALYSIS_PARAMETERS["gpu_runs_folder"]) / folder_name / f"predictions_{checkpoint_name}.npz"
    return dict(numpy.load(path)) if path.exists() else None



def small_picture(picture_bgr: numpy.ndarray) -> numpy.ndarray:
    """Return a 640 x 480 camera picture as the 160 x 120 RGB picture the models were trained on (area averaging)."""

    return cv2.resize(cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2RGB), (160, 120), interpolation = cv2.INTER_AREA)



def answers(
        policy: ModelPolicy,
        pictures_rgb: numpy.ndarray
) -> Dict[str, numpy.ndarray]:
    """
    Return a model's answers to many pictures.

    Parameters
    ----------
    policy : model_mode.ModelPolicy
        From ``loaded_policy`` (a single model, not a team).
    pictures_rgb : numpy.ndarray
        (pictures, 120, 160, 3) uint8 RGB, as in the training cache.

    Returns
    -------
    dict
        "forward_cm", "left_cm", "twist_deg" (the exact step the driver carries out), "angle_deg" and "length_cm" of
        that step, "stop_probability", and for a chunk model "chunk_cm" (pictures, 5, 2).
    """

    torch.set_num_threads(MODEL_PARAMETERS["threads"])
    model = policy.models[0]
    collected = {"exact": [], "stop": [], "chunk": []}
    with torch.inference_mode():
        for first in range(0, len(pictures_rgb), MODEL_PARAMETERS["batch_size"]):
            batch = torch.from_numpy(numpy.ascontiguousarray(pictures_rgb[first:first + MODEL_PARAMETERS["batch_size"]]))
            inputs, _ = augmented_pictures(batch, policy.settings, numpy.random.default_rng(0), is_training = False)
            outputs = model(inputs)
            collected["exact"].append(outputs["exact"].numpy())
            collected["stop"].append(torch.sigmoid(outputs["stop"][:, 0]).numpy())
            if "chunk" in outputs:
                collected["chunk"].append(outputs["chunk"].numpy().reshape(len(batch), -1, 2))
    exact = numpy.concatenate(collected["exact"])
    result = {
        "forward_cm": exact[:, 0], "left_cm": exact[:, 1], "twist_deg": exact[:, 2],
        "angle_deg": numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0])), "length_cm": numpy.hypot(exact[:, 0], exact[:, 1]),
        "stop_probability": numpy.concatenate(collected["stop"])
    }
    if collected["chunk"]:
        result["chunk_cm"] = numpy.concatenate(collected["chunk"])
    return result



def angle_error_deg(
        first_deg: numpy.ndarray,
        second_deg: numpy.ndarray
) -> numpy.ndarray:
    """Return the size of the difference between two directions, 0 to 180 degrees."""

    return numpy.abs((numpy.asarray(first_deg) - numpy.asarray(second_deg) + 180.0) % 360.0 - 180.0)
