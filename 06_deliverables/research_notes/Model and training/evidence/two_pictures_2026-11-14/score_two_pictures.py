"""
Score the two-picture model against model E's recipe by the rule fixed in config.py before the results (14.11.26).

For each run (model E's recipe and the two-picture model, two seeds each, all six sheets held out once on ki-014):
    path within 45      the mean over the six folds, as metrics.json gives it;
    right stops         episodes stopped on the end (fixed threshold, two pictures in a row), counted over the folds;
    turning pictures    path within 45 only on the pictures where the demonstrated path direction turns by
                        "turn_threshold_deg" or more from the step before; a model that repeats the last move scores
                        well elsewhere and badly here (causal confusion, de Haan and others 2019);
    steady pictures     the same on all other path pictures, for comparison.

Run from 02_code, after the runs are copied back into 04_runs/gpu_ki014:
    ..\\.venv\\Scripts\\python.exe "../06_deliverables/research_notes/Model and training/evidence/two_pictures_2026-11-14/score_two_pictures.py"
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "02_code"))

from config import TRAINING_TARGET_PARAMETERS
from train_policy import merged_run_settings
from training_metrics import angle_difference_deg
from training_probes import path_directions, training_arrays
from training_targets import load_training_frames, previous_rows


SCORING_PARAMETERS = {
    "runs_folder": Path(__file__).resolve().parents[5] / "04_runs" / "gpu_ki014",
    "runs": {
        "model E recipe, seed 1": "2026-11-10_resnet18_chunk_stop_weight_25",
        "model E recipe, seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2",
        "two pictures, seed 1": "2026-11-14_resnet18_two_pictures",
        "two pictures, seed 2": "2026-11-14_resnet18_two_pictures_seed_2"
    },
    "recipe_run_name": "resnet18_chunk_stop_weight_25",
    "turn_threshold_deg": 20.0,
    "within_deg": 45.0,
    "output_path": Path(__file__).resolve().parent / "two_pictures_scores.json"
}



def run_scores(
        run_folder: Path,
        frames,
        path_angle: numpy.ndarray,
        is_path_row: numpy.ndarray,
        is_turning: numpy.ndarray,
        parameters: Dict
) -> Dict:
    """
    Return the four measures of one run from its metrics.json, its fold files and its saved predictions.

    Parameters
    ----------
    run_folder : pathlib.Path
        The run's folder with metrics.json, fold_*.json and predictions_*.npz.
    frames : pandas.DataFrame
        The training frames, in the order the path angles were computed.
    path_angle, is_path_row, is_turning : numpy.ndarray
        Per frame row: the demonstrated path direction, whether it is scored, and whether it turns.
    parameters : dict
        The scoring settings.

    Returns
    -------
    dict
        "path_within_45", "right_stops", "turning_within_45", "steady_within_45" and the picture counts.
    """

    metrics = json.loads((run_folder / "metrics.json").read_text(encoding = "utf-8"))
    row_of_frame = {int(frame_index): row for row, frame_index in enumerate(frames["frame_index"].to_numpy())}
    right_stops = 0
    is_close = []
    turning = []
    for fold_path in sorted(run_folder.glob("fold_*.json")):
        measures = json.loads(fold_path.read_text(encoding = "utf-8"))["measures"]
        right_stops += int(round(measures["fixed_two_episodes_stopped_right"] * measures["test_episodes"]))
        saved = numpy.load(run_folder / f"predictions_{fold_path.stem[len('fold_'):]}.npz")
        rows = numpy.array([row_of_frame[int(frame_index)] for frame_index in saved["frame_index"]])
        is_scored = is_path_row[rows]
        difference = numpy.abs(angle_difference_deg(saved["angle_deg"][is_scored], path_angle[rows][is_scored]))
        is_close.append(difference <= parameters["within_deg"])
        turning.append(is_turning[rows][is_scored])
    is_close = numpy.concatenate(is_close)
    turning = numpy.concatenate(turning)
    return {
        "path_within_45": round(float(metrics["summary"]["path_within_45_deg_mean"]), 4),
        "right_stops": right_stops,
        "turning_within_45": round(float(is_close[turning].mean()), 4),
        "steady_within_45": round(float(is_close[~turning].mean()), 4),
        "turning_pictures": int(turning.sum()),
        "steady_pictures": int((~turning).sum())
    }



def main(parameters: Dict = SCORING_PARAMETERS) -> None:

    settings = merged_run_settings(parameters["recipe_run_name"])
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    frames = load_training_frames(target_parameters)
    arrays, _, _ = training_arrays(frames, settings, target_parameters)
    all_rows = numpy.arange(len(frames))
    path_angle, is_path_row = path_directions(frames, all_rows, arrays)
    # the step before in the same episode (an episode's first row is its own step before, so it never turns)
    previous = previous_rows(frames, 1)
    is_turning = (numpy.abs(angle_difference_deg(path_angle, path_angle[previous])) >= parameters["turn_threshold_deg"]) & is_path_row[previous]

    scores = {}
    for label, folder_name in parameters["runs"].items():
        run_folder = parameters["runs_folder"] / folder_name
        if not (run_folder / "metrics.json").exists():
            print(f"{label}: {run_folder.name} not here yet")
            continue
        scores[label] = run_scores(run_folder, frames, path_angle, is_path_row, is_turning, parameters)
        print(f"{label}: {scores[label]}", flush = True)
    parameters["output_path"].write_text(json.dumps(scores, indent = 1), encoding = "utf-8")



if __name__ == "__main__":
    main()
