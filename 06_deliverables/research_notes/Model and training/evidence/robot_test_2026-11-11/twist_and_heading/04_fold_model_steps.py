"""
The fold models' whole exact step on their held-out sheets (the saved prediction files hold only the angle and the
twist, not the step LENGTH the model asked for).

Each fold checkpoint is run once on the cached 160 x 120 pictures of the sheet it never saw (inference only, two
threads, no training). The twist must come out as in the saved predictions; the largest difference is printed.

Output: fold_steps_<label>.npz (frame_index, forward_cm, left_cm, twist_deg, stop_probability and, for chunk models,
chunk_cm) and 04_fold_model_steps.out.txt
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 04_fold_model_steps.py
(about 5 minutes per model on two threads, 11 minutes in all on 11.11.26)
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy
import torch

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames

from config import TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import PictureToMoveModel, augmented_pictures, merged_run_settings
from training_probes import design_outputs, predictions_from_outputs

FOLD_STEP_PARAMETERS = {
    # label: (run folder under 04_runs/gpu_ki014, run name in TRAINING_RUNS)
    "runs": {
        "E": ("2026-11-10_resnet18_chunk_stop_weight_25", "resnet18_chunk_stop_weight_25"),
        "A": ("2026-11-09_resnet18_final_candidate", "resnet18_final_candidate")
    },
    "batch_size": 128,
    "threads": 2
}



def main(parameters: Optional[Dict] = None) -> None:
    """Run every fold checkpoint on its held-out sheet and save the whole exact step it asks for."""

    settings = dict(FOLD_STEP_PARAMETERS)
    settings.update(parameters or {})
    report = Report("04_fold_model_steps.out.txt")
    torch.set_num_threads(settings["threads"])
    frames = demonstration_frames()
    pictures = numpy.load(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "images_wide_160.npy", mmap_mode = "r")
    report.say("FOLD MODELS RUN AGAIN ON THEIR HELD-OUT SHEETS, to get the step length each one asked for (inference only)")

    for label, (run_folder, run_name) in settings["runs"].items():
        folder = Path(ANALYSIS_PARAMETERS["trained_runs_folder"]) / run_folder
        metrics = json.loads((folder / "metrics.json").read_text(encoding = "utf-8"))
        run_settings = merged_run_settings(run_name)
        run_settings["chunk_steps"] = TRAINING_TARGET_PARAMETERS["chunk_steps"]
        run_settings["normalisation"] = metrics["settings"]["normalisation"]
        run_settings["is_pretrained"] = False
        class_names = metrics["class_names"]
        output_sizes = design_outputs(run_settings["design"], len(class_names), run_settings["chunk_steps"], run_settings.get("is_stop_separate", False))
        collected = {"frame_index": [], "exact": [], "stop_probability": [], "chunk": []}
        started = time.perf_counter()
        for sheet in ANALYSIS_PARAMETERS["sheets"]:
            model = PictureToMoveModel(run_settings, output_sizes)
            model.load_state_dict(torch.load(folder / "checkpoints" / f"without_{sheet}.pt", map_location = "cpu"))
            model.eval()
            torch.set_num_threads(settings["threads"])
            frame_index = frames.loc[frames["sheet"] == sheet, "frame_index"].to_numpy()
            exact_parts, stop_parts, chunk_parts = [], [], []
            with torch.inference_mode():
                for start in range(0, len(frame_index), settings["batch_size"]):
                    chosen = numpy.sort(frame_index[start:start + settings["batch_size"]])
                    batch = torch.from_numpy(numpy.ascontiguousarray(pictures[chosen]))
                    inputs, _ = augmented_pictures(batch, run_settings, numpy.random.default_rng(0), is_training = False)
                    outputs = model(inputs)
                    predictions = predictions_from_outputs(outputs, class_names, run_settings["stop_target"])
                    order = numpy.argsort(numpy.argsort(frame_index[start:start + settings["batch_size"]]))
                    exact_parts.append(outputs["exact"].numpy()[order])
                    stop_parts.append(numpy.asarray(predictions["stop_probability"])[order])
                    if "chunk" in outputs:
                        chunk_parts.append(outputs["chunk"].numpy()[order])
            exact = numpy.concatenate(exact_parts)
            saved = numpy.load(folder / f"predictions_without_{sheet}.npz")
            saved_twist = dict(zip(saved["frame_index"].tolist(), saved["twist_deg"].tolist()))
            difference = numpy.abs(exact[:, 2] - numpy.array([saved_twist[index] for index in frame_index.tolist()]))
            report.say(f"{label} {run_folder} without_{sheet}: {len(frame_index)} pictures; twist against the saved predictions: largest difference {difference.max():.4f} degrees; "
                       f"asked step length median {numpy.median(numpy.hypot(exact[:, 0], exact[:, 1])):.2f} cm")
            collected["frame_index"].append(frame_index)
            collected["exact"].append(exact)
            collected["stop_probability"].append(numpy.concatenate(stop_parts))
            if chunk_parts:
                collected["chunk"].append(numpy.concatenate(chunk_parts))
        exact = numpy.concatenate(collected["exact"])
        arrays = {"frame_index": numpy.concatenate(collected["frame_index"]), "forward_cm": exact[:, 0], "left_cm": exact[:, 1], "twist_deg": exact[:, 2],
                  "lean_deg": exact[:, 3], "hand_deg": exact[:, 4], "stop_probability": numpy.concatenate(collected["stop_probability"])}
        if collected["chunk"]:
            arrays["chunk_cm"] = numpy.concatenate(collected["chunk"]).reshape(len(exact), -1, 2)
        numpy.savez_compressed(EVIDENCE_FOLDER / f"fold_steps_{label}.npz", **arrays)
        report.say(f"{label}: saved fold_steps_{label}.npz with {len(exact)} rows in {time.perf_counter() - started:.0f} s")
    report.save()



if __name__ == "__main__":
    main()
