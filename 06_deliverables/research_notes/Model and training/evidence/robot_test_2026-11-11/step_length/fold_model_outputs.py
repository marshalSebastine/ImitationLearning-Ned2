"""
Run saved fold models on the pictures of the sheet each one never saw, and keep their raw exact step (no robot, no
training).

The run files of 04_runs/gpu_ki014 keep only the angle of the predicted step, so the predicted LENGTH has to be computed
again: every fold model (trained without one sheet) is run on the cached 160 x 120 pictures of its held-out sheet, on
exactly the rows it was scored on (the audit cleaning of training_targets.load_training_frames). The outputs are checked
against the saved predictions (twist and stop chance must agree) and written to fold_outputs_<recipe>.npz in this
folder, which the other scripts read.

Run from this folder:
    python fold_model_outputs.py                  both recipes (E: chunk with stop weight 25; A: the exact step)
    python fold_model_outputs.py E_recipe         one recipe
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import torch

import step_length_common as common

from config import TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import PictureToMoveModel, augmented_pictures, merged_run_settings
from training_probes import design_outputs
from training_targets import leave_one_sheet_out_folds, load_training_frames

FOLD_OUTPUT_PARAMETERS = {
    # the recipes of the models used on the robot on 11.11.26, as fold models: E (runs 2 to 9) and A (run 1; the team B
    # of run 10 is five seeds of the A recipe)
    "recipes": {
        "E_recipe": {"run_name": "resnet18_chunk_stop_weight_25", "run_folder": "2026-11-10_resnet18_chunk_stop_weight_25"},
        "E_recipe_seed_2": {"run_name": "resnet18_chunk_stop_weight_25_seed_2", "run_folder": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"},
        "A_recipe": {"run_name": "resnet18_final_candidate", "run_folder": "2026-11-09_resnet18_final_candidate"}
    },
    "default_recipes": ["E_recipe", "A_recipe"],
    "threads": 2,
    "batch_size": 128
}



def fold_outputs(
        recipe_name: str,
        parameters: Optional[Dict] = None
) -> Dict[str, numpy.ndarray]:
    """
    Run the six fold models of one recipe on their held-out sheets.

    Parameters
    ----------
    recipe_name : str
        A key of "recipes".
    parameters : dict, optional
        Overrides of ``FOLD_OUTPUT_PARAMETERS``.

    Returns
    -------
    dict of str to numpy.ndarray
        "frame_index", "exact" (rows, 5: tip forward and left in cm, forearm, lean and hand changes in degrees),
        "chunk" (rows, 5, 2; only for a chunk recipe), "stop_probability", and the largest differences from the saved
        predictions ("twist_difference_most_deg", "stop_difference_most").
    """

    settings_here = dict(FOLD_OUTPUT_PARAMETERS)
    settings_here.update(parameters or {})
    recipe = settings_here["recipes"][recipe_name]
    run_folder = Path(common.STEP_LENGTH_PARAMETERS["gpu_runs_folder"]) / recipe["run_folder"]
    metrics = json.loads((run_folder / "metrics.json").read_text(encoding = "utf-8"))
    settings = merged_run_settings(recipe["run_name"])
    settings["normalisation"] = metrics["settings"]["normalisation"]
    settings["is_pretrained"] = False
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    settings["chunk_steps"] = target_parameters["chunk_steps"]
    frames = load_training_frames(target_parameters)
    cache = numpy.load(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / f"images_{settings['picture_cache']}.npy", mmap_mode = "r")
    output_sizes = design_outputs(settings["design"], len(metrics["class_names"]), settings["chunk_steps"], settings.get("is_stop_separate", False))
    torch.set_num_threads(settings_here["threads"])

    kept = {"frame_index": [], "exact": [], "chunk": [], "stop_probability": []}
    twist_difference_most, stop_difference_most = 0.0, 0.0
    for fold in leave_one_sheet_out_folds(frames):
        started = time.perf_counter()
        test_rows = numpy.flatnonzero(fold["test"])
        frame_index = frames["frame_index"].to_numpy()[test_rows]
        model = PictureToMoveModel(settings, output_sizes)
        model.load_state_dict(torch.load(run_folder / "checkpoints" / f"{fold['name']}.pt", map_location = "cpu"))
        model.eval()
        parts = []
        with torch.inference_mode():
            for start in range(0, len(test_rows), settings_here["batch_size"]):
                pictures = torch.from_numpy(numpy.array(cache[numpy.sort(frame_index[start:start + settings_here["batch_size"]])]))
                # the cache is read in ascending order (faster on a memory map); put the batch back in row order
                order = numpy.argsort(numpy.argsort(frame_index[start:start + settings_here["batch_size"]]))
                inputs, _ = augmented_pictures(pictures[order], settings, numpy.random.default_rng(0), is_training = False)
                parts.append({name: value.numpy() for name, value in model(inputs).items()})
        outputs = {name: numpy.concatenate([part[name] for part in parts]) for name in parts[0]}
        stop_probability = 1.0 / (1.0 + numpy.exp(-outputs["stop"][:, 0]))
        saved = numpy.load(run_folder / f"predictions_{fold['name']}.npz")
        if not numpy.array_equal(saved["frame_index"], frame_index):
            raise ValueError(f"{recipe_name} {fold['name']}: the rows differ from the saved predictions")
        twist_difference = float(numpy.abs(saved["twist_deg"] - outputs["exact"][:, 2]).max())
        stop_difference = float(numpy.abs(saved["stop_probability"] - stop_probability).max())
        twist_difference_most = max(twist_difference_most, twist_difference)
        stop_difference_most = max(stop_difference_most, stop_difference)
        kept["frame_index"].append(frame_index)
        kept["exact"].append(outputs["exact"])
        kept["stop_probability"].append(stop_probability)
        if "chunk" in outputs:
            kept["chunk"].append(outputs["chunk"].reshape(len(test_rows), -1, 2))
        print(f"{recipe_name} | {fold['name']} | {len(test_rows)} pictures in {time.perf_counter() - started:.0f} s | largest difference from the "
              f"saved predictions: twist {twist_difference:.4f} degrees, stop chance {stop_difference:.5f}", flush = True)

    result = {name: numpy.concatenate(values) for name, values in kept.items() if values}
    result["twist_difference_most_deg"] = numpy.array(twist_difference_most)
    result["stop_difference_most"] = numpy.array(stop_difference_most)
    return result



def main(recipe_names: List[str]) -> None:

    for recipe_name in recipe_names or FOLD_OUTPUT_PARAMETERS["default_recipes"]:
        result = fold_outputs(recipe_name)
        numpy.savez_compressed(common.EVIDENCE_FOLDER / f"fold_outputs_{recipe_name}.npz", **result)
        print(f"{recipe_name}: {len(result['frame_index'])} pictures written to fold_outputs_{recipe_name}.npz", flush = True)



if __name__ == "__main__":
    main(sys.argv[1:])
