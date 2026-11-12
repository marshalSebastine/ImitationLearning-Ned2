"""
Verifier 14 (inference only, two threads): what would a model that NEVER saw the sheet have answered on the pictures the
robot really took on 11.11.26? For every run the fold checkpoint of the same recipe trained without that run's sheet is
run on the run's own logged pictures and compared with the logged answers of the all-sheets model.
First order only: the pictures come from a run steered by the all-sheets model. It shows how far the answers of the
exam case (unseen line) are from the answers behind the ten runs. Nothing connects to the robot; no training.
"""
import json
import cv2
import numpy
import torch
from vcommon import run_log, RUNS, RUNS_FOLDER, TRAINED, wrap

from config import TRAINING_TARGET_PARAMETERS
from train_policy import PictureToMoveModel, augmented_pictures, merged_run_settings
from training_probes import design_outputs

torch.set_num_threads(2)
RECIPES = {"E": ("2026-11-10_resnet18_chunk_stop_weight_25", "resnet18_chunk_stop_weight_25"),
           "A": ("2026-11-09_resnet18_final_candidate", "resnet18_final_candidate"),
           "B": ("2026-11-09_resnet18_final_candidate", "resnet18_final_candidate")}


def loaded(run_folder, run_name, checkpoint):
    folder = TRAINED / run_folder
    metrics = json.loads((folder / "metrics.json").read_text(encoding = "utf-8"))
    settings = merged_run_settings(run_name)
    settings["chunk_steps"] = TRAINING_TARGET_PARAMETERS["chunk_steps"]
    settings["normalisation"] = metrics["settings"]["normalisation"]
    settings["is_pretrained"] = False
    sizes = design_outputs(settings["design"], len(metrics["class_names"]), settings["chunk_steps"], settings.get("is_stop_separate", False))
    model = PictureToMoveModel(settings, sizes)
    model.load_state_dict(torch.load(folder / "checkpoints" / checkpoint, map_location = "cpu"))
    model.eval()
    torch.set_num_threads(2)
    return model, settings


def answers(model, settings, small_rgb):
    parts = []
    with torch.inference_mode():
        for start in range(0, len(small_rgb), 64):
            inputs, _ = augmented_pictures(torch.from_numpy(numpy.ascontiguousarray(small_rgb[start:start + 64])), settings, numpy.random.default_rng(0), is_training = False)
            parts.append(model(inputs)["exact"].numpy())
    return numpy.concatenate(parts)


print("run  model  sheet    pictures | all-sheets (logged): mean length, mean |twist| | unseen-sheet fold: mean length, mean |twist| | twist: correlation, fold on logged slope | direction: median |difference|, share over 45 | first 13 pictures: net twist logged / fold")
for number, (folder, model_label, sheet) in RUNS.items():
    log = run_log(number)
    log = log[log["action"] == "model_step"].reset_index(drop = True)
    run_folder, run_name = RECIPES[model_label]
    model, settings = loaded(run_folder, run_name, f"without_{sheet}.pt")
    width, height = settings["picture_size"]
    pictures = numpy.stack([cv2.resize(cv2.cvtColor(cv2.imread(str(RUNS_FOLDER / folder / "frames" / f"{int(k):03d}.jpg")), cv2.COLOR_BGR2RGB), (width, height), interpolation = cv2.INTER_AREA) for k in log["step"]])
    out = answers(model, settings, pictures)
    length = numpy.hypot(out[:, 0], out[:, 1])
    angle = numpy.degrees(numpy.arctan2(out[:, 1], out[:, 0]))
    logged_twist = log["model_twist_deg"].to_numpy()
    both_long = (length > 0.2) & (log["asked_cm"].to_numpy() > 0.2)
    difference = numpy.abs(wrap(angle - log["model_angle_deg"].to_numpy()))[both_long]
    slope = float(numpy.sum(out[:, 2] * logged_twist) / numpy.sum(logged_twist ** 2))
    first = slice(0, 13)
    print(f"{number:3d}  {model_label:5s}  {sheet}  {len(log):8d} | {log['asked_cm'].mean():19.2f} {numpy.abs(logged_twist).mean():14.2f} | {length.mean():26.2f} {numpy.abs(out[:, 2]).mean():14.2f} |"
          f" {numpy.corrcoef(out[:, 2], logged_twist)[0, 1]:18.2f} {slope:21.2f} | {numpy.median(difference):26.1f} {(difference > 45).mean():14.2f} | {logged_twist[first].sum():+22.1f} / {out[first, 2].sum():+.1f}")
print("note: run 10 was steered by team B (five all-sheets models of recipe A); it is compared with one fold model of recipe A; run 1 was model A.")
