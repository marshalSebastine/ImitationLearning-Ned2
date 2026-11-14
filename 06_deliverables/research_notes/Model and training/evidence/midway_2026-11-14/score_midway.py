"""
Score the mid-way check of round 2 (14.11.26) by the rule fixed in config.py before the results: on each new track held
out, model E's recipe trained on round 1 plus the other new tracks against the same recipe trained on round 1 only.

    path within 45   share of path pictures whose predicted direction is within 45 degrees of where the line goes;
    right stops      episodes stopped on the end (fixed threshold, two pictures in a row).

Run from 02_code, after the runs are copied back into 04_runs/gpu_ki014:
    ..\\.venv\\Scripts\\python.exe "../06_deliverables/research_notes/Model and training/evidence/midway_2026-11-14/score_midway.py"
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "02_code"))

from config import ROUND_2_INDEX_FOLDER, TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import merged_run_settings
from training_probes import fold_measures, training_arrays
from training_targets import load_training_frames, new_track_shapes


MIDWAY_PARAMETERS = {
    "runs_folder": Path(__file__).resolve().parents[5] / "04_runs" / "gpu_ki014",
    "held_out_run": "2026-11-14_resnet18_round_2_part_new_shapes",
    "round_1_run": "2026-11-14_resnet18_round_1_to_new_shapes",
    "recipe_run_name": "resnet18_round_2_part_new_shapes",
    "output_path": Path(__file__).resolve().parent / "midway_scores.json"
}



def main(parameters: Dict = MIDWAY_PARAMETERS) -> None:

    settings = merged_run_settings(parameters["recipe_run_name"])
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    settings["chunk_steps"] = target_parameters["chunk_steps"]
    data_parameters = dict(TRAINING_DATA_PARAMETERS)
    data_parameters["processed_folder"] = ROUND_2_INDEX_FOLDER
    frames = load_training_frames(target_parameters, data_parameters)
    arrays, _, is_moving = training_arrays(frames, settings, target_parameters)
    shapes = new_track_shapes(frames)
    row_of_frame = {int(frame_index): row for row, frame_index in enumerate(frames["frame_index"].to_numpy())}

    round_1_folder = parameters["runs_folder"] / parameters["round_1_run"]
    saved = numpy.load(round_1_folder / "predictions_round_1_only.npz")
    round_1_rows = numpy.array([row_of_frame[int(index)] for index in saved["frame_index"]])
    predictions = {key: saved[key] for key in saved.files if key != "frame_index"}

    scores = {}
    totals = {"held_out": {"right": 0, "episodes": 0}, "round_1_only": {"right": 0, "episodes": 0}}
    for fold_path in sorted((parameters["runs_folder"] / parameters["held_out_run"]).glob("fold_without_T*.json")):
        shape = fold_path.stem.split("_")[-1]
        held_out = json.loads(fold_path.read_text(encoding = "utf-8"))["measures"]
        is_shape = shapes[round_1_rows] == shape
        subset = {key: value[is_shape] for key, value in predictions.items()}
        round_1 = fold_measures(subset, frames, round_1_rows[is_shape], arrays, is_moving)
        scores[shape] = {}
        for name, measures in (("held_out", held_out), ("round_1_only", round_1)):
            right = int(round(measures["fixed_two_episodes_stopped_right"] * measures["test_episodes"]))
            scores[shape][name] = {"path_within_45": round(float(measures["path_within_45_deg"]), 3), "right_stops": right,
                                   "episodes": int(measures["test_episodes"])}
            totals[name]["right"] += right
            totals[name]["episodes"] += int(measures["test_episodes"])
        print(shape, scores[shape], flush = True)
    for name in totals:
        totals[name]["mean_path_within_45"] = round(float(numpy.mean([scores[shape][name]["path_within_45"] for shape in scores])), 3)
    scores["totals"] = totals
    print("totals:", totals)
    parameters["output_path"].write_text(json.dumps(scores, indent = 1), encoding = "utf-8")



if __name__ == "__main__":
    main()
