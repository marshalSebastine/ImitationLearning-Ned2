"""
Score every run of the final model search with the corrections by hand (the h key) in view, 15.11.26.

In the test episodes, too, the last steps before an h press show the replay going on while the puck drifts: on those
pictures the recorded move is the drift, not the right move, so a model that has learned to come back is marked wrong
there. This script reads each run's saved predictions and gives the path measure (share of path pictures whose
predicted direction is within 45 degrees of where the tip goes) on four groups of test pictures:

    all             every test picture, as score_final.py does;
    without before  the same without the last "before_correction_rows" steps before an h press: the fair yardstick;
    after           only the "after_correction_rows" replayed steps after an h press: does the model bring a drifted
                    puck back the way the demonstration did?
    before          only the steps before an h press (a model that copies the drift scores high here).

"mean" is the mean over the folds, as in the rule of the search; "pooled" counts all pictures of all folds together
and is steadier for the small groups. Both yardsticks are read: the 16 new tracks and the six course sheets.

Run from 02_code, after the result files are copied into 04_runs/gpu_ki014:
    ../.venv/Scripts/python.exe "../06_deliverables/research_notes/Model and training/evidence/final_2026-11-14/score_hand_corrections.py"
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "02_code"))

from config import TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy_emranul import merged_run_settings
from training_metrics import direction_metrics
from training_probes import correction_rows, path_directions, training_arrays
from training_targets import load_training_frames


HAND_CORRECTION_SCORE_PARAMETERS = {
    "runs_folder": Path(__file__).resolve().parents[5] / "04_runs" / "gpu_ki014",
    "run_prefix": "2026-11-14_resnet18_full_",
    "recipe_run_name": "resnet18_full_new_shapes",
    "yardsticks": {
        "new tracks": {"fold_pattern": "predictions_without_T*.npz", "reference_run": "new_shapes"},
        "course sheets": {"fold_pattern": "predictions_without_path_*.npz", "reference_run": "course_sheets"}
    },
    "before_correction_rows": 5,
    "after_correction_rows": 8,
    "angle_key": "angle_deg",
    "output_path": Path(__file__).resolve().parent / "hand_correction_scores.json"
}



def test_table(parameters: Dict) -> Dict:
    """
    Load the training index once: the path direction of every row and the groups it belongs to.

    Returns
    -------
    dict
        "row_of_frame" (picture number to row), "path_angle", "is_path_row" and the masks "is_before" and "is_after".
    """

    settings = merged_run_settings(parameters["recipe_run_name"])
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    settings["chunk_steps"] = target_parameters["chunk_steps"]
    data_parameters = dict(TRAINING_DATA_PARAMETERS)
    data_parameters["processed_folder"] = settings["processed_folder"]
    frames = load_training_frames(target_parameters, data_parameters)
    arrays, _, _ = training_arrays(frames, settings, target_parameters)
    rows = numpy.arange(len(frames))
    path_angle, is_path_row = path_directions(frames, rows, arrays)
    is_before, is_after = correction_rows(frames, parameters)
    return {"row_of_frame": {int(frame_index): row for row, frame_index in enumerate(frames["frame_index"].to_numpy())},
            "path_angle": path_angle, "is_path_row": is_path_row, "is_before": is_before, "is_after": is_after}



def fold_counts(prediction_path: Path, table: Dict, parameters: Dict) -> Dict[str, list]:
    """Return, for one fold's saved predictions, the pictures within 45 degrees and the pictures scored, per group."""

    saved = numpy.load(prediction_path)
    rows = numpy.array([table["row_of_frame"][int(frame_index)] for frame_index in saved["frame_index"]])
    predicted = saved[parameters["angle_key"]]
    groups = {"all": numpy.ones(len(rows), dtype = bool), "without_before": ~table["is_before"][rows],
              "after": table["is_after"][rows] & ~table["is_before"][rows], "before": table["is_before"][rows]}
    counts = {}
    for name, is_in_group in groups.items():
        is_scored = table["is_path_row"][rows] & is_in_group
        scored = int(is_scored.sum())
        share = direction_metrics(table["path_angle"][rows], predicted, is_scored)["within_45_deg"] if scored else 0.0
        counts[name] = [int(round(share * scored)), scored]
    return counts



def run_scores(fold_tables: Dict[str, Dict[str, list]]) -> Dict:
    """Return the mean over the folds and the pooled share of each group, from the folds' counts."""

    scores = {"folds": len(fold_tables)}
    for name in ("all", "without_before", "after", "before"):
        within = [counts[name][0] for counts in fold_tables.values()]
        scored = [counts[name][1] for counts in fold_tables.values()]
        shares = [right / total for right, total in zip(within, scored) if total]
        scores[name] = {"mean": round(float(numpy.mean(shares)), 4) if shares else None,
                        "pooled": round(sum(within) / sum(scored), 4) if sum(scored) else None, "pictures": int(sum(scored))}
    return scores



def main(parameters: Dict = HAND_CORRECTION_SCORE_PARAMETERS) -> None:
    """Print one table per yardstick and write all numbers to "output_path"."""

    table = test_table(parameters)
    results = {}
    for yardstick, choice in parameters["yardsticks"].items():
        runs = {}
        for run_folder in sorted(parameters["runs_folder"].glob(parameters["run_prefix"] + "*")):
            fold_tables = {path.stem.replace("predictions_", ""): fold_counts(path, table, parameters)
                           for path in sorted(run_folder.glob(choice["fold_pattern"]))}
            if fold_tables:
                runs[run_folder.name.replace(parameters["run_prefix"], "")] = fold_tables
        reference = runs.get(choice["reference_run"], {})
        results[yardstick] = {}
        print(f"\n{yardstick}: path within 45 degrees (mean over the folds / pooled), then the reference on the same folds")
        print(f"{'run':44s} {'folds':>5s} {'all':>13s} {'without before':>15s} {'after (pooled, n)':>19s} {'before (pooled, n)':>19s} | {'reference: without before, after':>32s}")
        for name, fold_tables in runs.items():
            scores = run_scores(fold_tables)
            same_folds = run_scores({fold: reference[fold] for fold in fold_tables if fold in reference})
            results[yardstick][name] = {"scores": scores, "reference_same_folds": same_folds, "folds": fold_tables}

            def text(group: str, source: Dict = scores) -> str:
                values = source.get(group, {})
                return "-" if not values or values["mean"] is None else f"{values['mean']:.3f} / {values['pooled']:.3f}"

            def small(group: str, source: Dict = scores) -> str:
                values = source.get(group, {})
                return "-" if not values or values["pooled"] is None else f"{values['pooled']:.3f} ({values['pictures']})"

            reference_text = "-" if not same_folds["folds"] else f"{text('without_before', same_folds)}, {small('after', same_folds)}"
            print(f"{name:44s} {scores['folds']:5d} {text('all'):>13s} {text('without_before'):>15s} {small('after'):>19s} {small('before'):>19s} | {reference_text:>32s}")
    parameters["output_path"].write_text(json.dumps(results, indent = 1), encoding = "utf-8")



if __name__ == "__main__":
    main()
