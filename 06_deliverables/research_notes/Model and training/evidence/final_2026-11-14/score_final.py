"""
Score the search for the final model (14.11.26) on its yardstick: each of the 16 new tracks held out once, 55 test
episodes on lines the model has not seen. Per run, over the folds that are finished:

    path within 45   mean over the folds of the share of path pictures whose predicted direction is within 45 degrees
                     of where the line goes;
    right stops      episodes stopped on the end (fixed threshold 0.5, two pictures in a row), with the too early and
                     the never stopped ones, for all tracks and for the six whose line ends in the middle of the paper;
    against the reference on the same folds, so that an unfinished run can already be read.

The run trained on round 1 only is one training tested on every new track; its pictures are scored per track here.

Run from 02_code, after the result files are copied into 04_runs/gpu_ki014:
    ..\.venv\Scripts\python.exe "../06_deliverables/research_notes/Model and training/evidence/final_2026-11-14/score_final.py"
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "02_code"))

from config import FULL_INDEX_FOLDER, TRAINING_DATA_PARAMETERS, TRAINING_RUN_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import merged_run_settings
from training_probes import fold_measures, training_arrays
from training_targets import load_training_frames, new_track_shapes


FINAL_SCORE_PARAMETERS = {
    "runs_folder": Path(__file__).resolve().parents[5] / "04_runs" / "gpu_ki014",
    "run_prefix": "2026-11-14_resnet18_full_",
    "reference_run": "2026-11-14_resnet18_full_new_shapes",
    "round_1_run": "2026-11-14_resnet18_full_round_1_to_new",
    "recipe_run_name": "resnet18_full_new_shapes",
    "mid_paper_end_tracks": TRAINING_RUN_PARAMETERS["mid_paper_end_tracks"],
    "output_path": Path(__file__).resolve().parent / "final_scores.json"
}



def track_scores_of_measures(measures: Dict) -> Dict:
    """Return the few numbers kept per held-out track from one fold's measures."""

    episodes = int(measures["test_episodes"])
    return {"path_within_45": float(measures["path_within_45_deg"]), "episodes": episodes,
            "right": int(round(measures["fixed_two_episodes_stopped_right"] * episodes)),
            "early": int(round(measures["fixed_two_episodes_stopped_early"] * episodes)),
            "never": int(round(measures["fixed_two_episodes_never_stopped"] * episodes))}



def round_1_track_scores(parameters: Dict) -> Dict[str, Dict]:
    """Return the per-track scores of the run trained on round 1 only, from its saved predictions of all new tracks."""

    prediction_path = parameters["runs_folder"] / parameters["round_1_run"] / "predictions_round_1_only.npz"
    if not prediction_path.is_file():
        return {}
    settings = merged_run_settings(parameters["recipe_run_name"])
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    settings["chunk_steps"] = target_parameters["chunk_steps"]
    data_parameters = dict(TRAINING_DATA_PARAMETERS)
    data_parameters["processed_folder"] = FULL_INDEX_FOLDER
    frames = load_training_frames(target_parameters, data_parameters)
    arrays, _, is_moving = training_arrays(frames, settings, target_parameters)
    shapes = new_track_shapes(frames)
    row_of_frame = {int(frame_index): row for row, frame_index in enumerate(frames["frame_index"].to_numpy())}
    saved = numpy.load(prediction_path)
    rows = numpy.array([row_of_frame[int(index)] for index in saved["frame_index"]])
    predictions = {key: saved[key] for key in saved.files if key != "frame_index"}
    scores = {}
    for shape in sorted(set(shapes[rows])):
        is_shape = shapes[rows] == shape
        subset = {key: value[is_shape] for key, value in predictions.items()}
        scores[shape] = track_scores_of_measures(fold_measures(subset, frames, rows[is_shape], arrays, is_moving))
    return scores



def summary_of_tracks(track_scores: Dict[str, Dict], tracks: Optional[list] = None) -> Dict:
    """Return the mean path and the summed stop counts over the given tracks (all when not given)."""

    names = [name for name in sorted(track_scores) if tracks is None or name in tracks]
    if not names:
        return {"tracks": 0}
    summary = {"tracks": len(names), "path_within_45": round(float(numpy.mean([track_scores[name]["path_within_45"] for name in names])), 4)}
    for key in ("episodes", "right", "early", "never"):
        summary[key] = int(sum(track_scores[name][key] for name in names))
    return summary



def main(parameters: Dict = FINAL_SCORE_PARAMETERS) -> None:

    runs = {}
    for run_folder in sorted(parameters["runs_folder"].glob(parameters["run_prefix"] + "*")):
        track_scores = {}
        for fold_path in sorted(run_folder.glob("fold_without_T*.json")):
            track_scores[fold_path.stem.split("_")[-1]] = track_scores_of_measures(json.loads(fold_path.read_text(encoding = "utf-8"))["measures"])
        if track_scores:
            runs[run_folder.name.replace(parameters["run_prefix"], "")] = track_scores
    round_1 = round_1_track_scores(parameters)
    if round_1:
        runs["round_1_to_new"] = round_1

    reference_name = parameters["reference_run"].replace(parameters["run_prefix"], "")
    reference = runs.get(reference_name, {})
    results = {}
    print(f"{'run':38s} {'folds':>5s} {'path45':>7s} {'right':>8s} {'early':>6s} {'never':>6s} {'mid-paper right':>16s} | reference on the same folds: path45, right, mid-paper right")
    for name, track_scores in runs.items():
        everything = summary_of_tracks(track_scores)
        mid_paper = summary_of_tracks(track_scores, parameters["mid_paper_end_tracks"])
        same_folds = {track: reference[track] for track in track_scores if track in reference}
        reference_all = summary_of_tracks(same_folds)
        reference_mid = summary_of_tracks(same_folds, parameters["mid_paper_end_tracks"])
        results[name] = {"all": everything, "mid_paper": mid_paper, "reference_same_folds": reference_all,
                         "reference_same_folds_mid_paper": reference_mid, "tracks": track_scores}
        mid_text = f"{mid_paper.get('right', 0)} of {mid_paper.get('episodes', 0)}"
        reference_text = "" if not same_folds else (f"{reference_all['path_within_45']:.3f}, {reference_all['right']} of {reference_all['episodes']}, "
                                                    f"{reference_mid.get('right', 0)} of {reference_mid.get('episodes', 0)}")
        print(f"{name:38s} {everything['tracks']:5d} {everything['path_within_45']:7.3f} {everything['right']:3d} of {everything['episodes']:2d} "
              f"{everything['early']:6d} {everything['never']:6d} {mid_text:>16s} | {reference_text}")
    parameters["output_path"].write_text(json.dumps(results, indent = 1), encoding = "utf-8")



if __name__ == "__main__":
    main()
