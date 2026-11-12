"""
Verifier 05 (inference only, two threads, no training, nothing connects to the robot):
  (a) check of my pipeline: the all-sheets model E on the logged pictures of run 9 must give back the logged answers;
  (b) what a model that NEVER saw sheet 03 (fold checkpoint without_path_03, recipes E and A) asks for on the robot's own
      pictures at the first bend of sheet 03 (runs 5, 9, 10): asked length and twist;
  (c) the same fold model and the all-sheets model E on the DEMONSTRATION pictures of the eight standard-start
      demonstrations up to the picture where joint 4 reaches -70 degrees: do the asked lengths add up to the demonstrated
      path (with the hand's jitter) or to the net displacement?
"""
import json
from pathlib import Path
import cv2
import numpy
import pandas
import torch
from vcommon import demos, run_log, RUNS, RUNS_FOLDER, TRAINED, ROOT

from config import TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import PictureToMoveModel, augmented_pictures, merged_run_settings
from training_probes import design_outputs

torch.set_num_threads(2)


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
    """small_rgb: (n, height, width, 3) uint8. Returns the exact step (n, 5): forward cm, left cm, twist, lean, hand."""
    parts = []
    with torch.inference_mode():
        for start in range(0, len(small_rgb), 64):
            inputs, _ = augmented_pictures(torch.from_numpy(numpy.ascontiguousarray(small_rgb[start:start + 64])), settings, numpy.random.default_rng(0), is_training = False)
            parts.append(model(inputs)["exact"].numpy())
    return numpy.concatenate(parts)


def run_pictures(number, count, settings):
    width, height = settings["picture_size"]
    pictures = []
    for k in range(count):
        picture = cv2.imread(str(RUNS_FOLDER / RUNS[number][0] / "frames" / f"{k:03d}.jpg"), cv2.IMREAD_COLOR)
        pictures.append(cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (width, height), interpolation = cv2.INTER_AREA))
    return numpy.stack(pictures)


all_E, settings_E = loaded("2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets", "resnet18_model_4_chunk_stop_25_all_sheets", "all_sheets.pt")
fold_E, settings_fold_E = loaded("2026-11-10_resnet18_chunk_stop_weight_25", "resnet18_chunk_stop_weight_25", "without_path_03.pt")
fold_A, settings_fold_A = loaded("2026-11-09_resnet18_final_candidate", "resnet18_final_candidate", "without_path_03.pt")
all_A, settings_A = loaded("2026-11-10_resnet18_model_1_all_sheets", "resnet18_model_1_all_sheets", "all_sheets.pt")

print("(a) pipeline check: all-sheets model E on the logged pictures of run 9 (first 30) against the log")
log = run_log(9)
mine = answers(all_E, settings_E, run_pictures(9, 30, settings_E))
print("    largest difference: forward %.4f cm, left %.4f cm, twist %.4f degrees" % (numpy.abs(mine[:, 0] - log["model_tip_forward_cm"].iloc[:30]).max(), numpy.abs(mine[:, 1] - log["model_tip_left_cm"].iloc[:30]).max(), numpy.abs(mine[:, 2] - log["model_twist_deg"].iloc[:30]).max()))

print("\n(b) the robot pictures at the first bend of sheet 03: the model used on the robot (trained on sheet 03 too) against fold models that never saw sheet 03")
for number, count in ((5, 17), (9, 13), (10, 13)):
    whole = run_log(number)
    log = whole.iloc[:count]
    print(f"    run {number}, pictures 0 to {count - 1} (the forearm went from {log['j4'].iloc[0]:.1f} to {whole['j4'].iloc[count]:.1f} degrees)")
    print(f"       logged ({RUNS[number][1]}, all sheets): mean asked length {log['asked_cm'].mean():.2f} cm, sum {log['asked_cm'].sum():.1f}; mean twist {log['model_twist_deg'].mean():+.2f}, sum {log['model_twist_deg'].sum():+.1f}; twist per asked cm {log['model_twist_deg'].sum() / log['asked_cm'].sum():+.1f}")
    for label, model, settings in (("fold E without sheet 03", fold_E, settings_fold_E), ("fold A without sheet 03", fold_A, settings_fold_A)):
        out = answers(model, settings, run_pictures(number, count, settings))
        length = numpy.hypot(out[:, 0], out[:, 1])
        print(f"       {label}: mean asked length {length.mean():.2f} cm, sum {length.sum():.1f}; mean twist {out[:, 2].mean():+.2f}, sum {out[:, 2].sum():+.1f}; twist per asked cm {out[:, 2].sum() / length.sum():+.1f};"
              f" twists {numpy.round(out[:, 2], 1).tolist()}")

print("\n(c) the demonstration pictures of the eight standard-start demonstrations of sheet 03 (end A), from the first picture to the one where joint 4 reaches -70")
frames = demos()
reference = frames[(frames["sheet"] == "path_03") & (frames["start_kind"] == "STD") & (frames["start_end"] == "A")]
cache = numpy.load(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "images_wide_160.npy", mmap_mode = "r")
saved_fold = numpy.load(ROOT / "06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/twist_and_heading/fold_steps_E.npz")
saved_map = {int(i): k for k, i in enumerate(saved_fold["frame_index"])}
rows = []
largest = 0.0
for episode, e in reference.groupby("episode_id"):
    e = e.reset_index(drop = True)
    k = int(numpy.flatnonzero(e["j4"].to_numpy() <= -70.0)[0])
    part = e.iloc[:k]
    pictures = numpy.stack([cache[int(i)] for i in part["frame_index"]])
    path = part["len_cm"].sum()
    net = 100.0 * numpy.hypot(e["tip_x_m"].iloc[k] - e["tip_x_m"].iloc[0], e["tip_y_m"].iloc[k] - e["tip_y_m"].iloc[0])
    row = {"pictures": k, "demo_path": path, "demo_net": net, "demo_twist_sum": part["twist"].sum()}
    for label, model, settings in (("all E", all_E, settings_E), ("fold E", fold_E, settings_fold_E), ("all A", all_A, settings_A), ("fold A", fold_A, settings_fold_A)):
        out = answers(model, settings, pictures)
        row[label + " length sum"] = numpy.hypot(out[:, 0], out[:, 1]).sum()
        row[label + " twist sum"] = out[:, 2].sum()
        if label == "fold E":
            theirs = numpy.array([[saved_fold["forward_cm"][saved_map[int(i)]], saved_fold["left_cm"][saved_map[int(i)]], saved_fold["twist_deg"][saved_map[int(i)]]] for i in part["frame_index"]])
            largest = max(largest, float(numpy.abs(theirs - out[:, :3]).max()))
    rows.append(row)
table = pandas.DataFrame(rows)
pandas.set_option("display.width", 250)
print(table.round(1).to_string())
print("    medians:", table.median().round(2).to_dict())
print("    the analyst's fold_steps_E.npz against my own run of the same checkpoint on these pictures: largest difference %.4f" % largest)
