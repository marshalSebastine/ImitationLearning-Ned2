"""
Raw outputs of the saved models on pictures they were not trained on, and of model E on the robot's own pictures.

The run files of 04_runs keep only the direction's angle, so the predicted step LENGTH, the whole chunk and the names'
probabilities must be computed again. Inference only, nothing is trained:
    fold models   each demonstration picture goes through the fold model that was trained WITHOUT its sheet (the
                  held-out-sheet outputs), for the recipes named in "fold_jobs";
    all sheets    model E (the one on the robot on 11.11.26) over every demonstration picture, with its hidden layer
                  (its own description of a picture, used later to find look-alike demonstration pictures), and over
                  the logged pictures of the ten robot runs.
Every fold is checked against the predictions saved when it was scored on ki-014 (the angle must agree).

Run from this folder (each job takes about 3 minutes on two processor threads):
    python s02_model_outputs.py recipe_e recipe_e_seed_2 backup recipe_a all_sheets_e robot_runs
Output: one .npz per job in the cache folder (ANALYSIS_PARAMETERS["cache_folder"]) and s02_model_outputs.out.txt.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas

import common

OUTPUT_PARAMETERS = {
    "fold_jobs": {
        "recipe_e": {"run_name": "resnet18_chunk_stop_weight_25", "folder": "2026-11-10_resnet18_chunk_stop_weight_25"},
        "recipe_e_seed_2": {"run_name": "resnet18_chunk_stop_weight_25_seed_2", "folder": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"},
        "backup": {"run_name": "resnet18_backup_names_plus_exact", "folder": "2026-11-09_resnet18_backup_names_plus_exact"},
        "recipe_a": {"run_name": "resnet18_final_candidate", "folder": "2026-11-09_resnet18_final_candidate"}
    },
    "all_sheets_e": {"run_name": "resnet18_model_4_chunk_stop_25_all_sheets", "folder": "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets"},
    "robot_run_prefix": "2026-11-11_17",
    "check_angle_deg": 0.05
}



def fold_job(
        job_name: str,
        frames: pandas.DataFrame,
        report,
        parameters: Optional[Dict] = None
) -> None:
    """
    Run the six fold models of one recipe over their held-out sheets and save the raw outputs.

    Parameters
    ----------
    job_name : str
        A key of "fold_jobs".
    frames : pandas.DataFrame
        From ``common.load_demonstration_frames``.
    report : callable
        Prints a line to the screen and the output file.
    parameters : dict, optional
        Overrides for ``OUTPUT_PARAMETERS`` and ``common.ANALYSIS_PARAMETERS``.
    """

    settings = common.merged_parameters(dict(OUTPUT_PARAMETERS, **(parameters or {})))
    job = settings["fold_jobs"][job_name]
    run_folder = Path(settings["fold_runs_folder"]) / job["folder"]
    collected = {}
    for sheet in sorted(frames["sheet"].unique()):
        fold_name = f"without_{sheet}"
        rows = numpy.flatnonzero((frames["sheet"] == sheet).to_numpy())
        loaded = common.load_model(job["run_name"], run_folder, fold_name)
        started = time.perf_counter()
        outputs = common.outputs_of_pictures(loaded, common.cached_pictures(frames["frame_index"].to_numpy()[rows]), parameters = settings)
        for name, values in outputs.items():
            if name not in collected:
                collected[name] = numpy.zeros((len(frames),) + values.shape[1:], dtype = numpy.float32)
            collected[name][rows] = values
        # the saved predictions of the same fold: the angle of the chunk's first step (chunk design), of the most likely
        # name (mixed design) or of the exact step (exact design)
        saved = numpy.load(run_folder / f"predictions_{fold_name}.npz")
        if not numpy.array_equal(saved["frame_index"], frames["frame_index"].to_numpy()[rows]):
            raise ValueError(f"{job_name} {fold_name}: the saved predictions cover other pictures than the cleaned table")
        if "chunk" in outputs:
            angle = numpy.degrees(numpy.arctan2(outputs["chunk"][:, 1], outputs["chunk"][:, 0]))
            reference = saved["angle_deg"]
        elif "direction" in outputs:
            angle = numpy.degrees(numpy.arctan2(outputs["exact"][:, 1], outputs["exact"][:, 0]))
            reference = saved["angle_vector_deg"]
        else:
            angle = numpy.degrees(numpy.arctan2(outputs["exact"][:, 1], outputs["exact"][:, 0]))
            reference = saved["angle_deg"]
        worst = float(common.angle_difference_deg(angle, reference).max())
        report(f"{job_name} | {fold_name} | {len(rows)} pictures | {time.perf_counter() - started:.0f} s | "
               f"largest difference to the saved angle: {worst:.4f} deg | {'ok' if worst <= settings['check_angle_deg'] else 'MISMATCH'}")
    Path(settings["cache_folder"]).mkdir(parents = True, exist_ok = True)
    numpy.savez_compressed(Path(settings["cache_folder"]) / f"fold_outputs_{job_name}.npz", frame_index = frames["frame_index"].to_numpy(), **collected)



def all_sheets_job(
        frames: pandas.DataFrame,
        report,
        parameters: Optional[Dict] = None
) -> None:
    """Run model E (trained on all sheets) over every demonstration picture and save its outputs and hidden layer."""

    settings = common.merged_parameters(dict(OUTPUT_PARAMETERS, **(parameters or {})))
    job = settings["all_sheets_e"]
    loaded = common.load_model(job["run_name"], Path(settings["fold_runs_folder"]) / job["folder"], "all_sheets")
    started = time.perf_counter()
    outputs = common.outputs_of_pictures(loaded, common.cached_pictures(frames["frame_index"].to_numpy()), is_hidden_kept = True, parameters = settings)
    Path(settings["cache_folder"]).mkdir(parents = True, exist_ok = True)
    numpy.savez_compressed(Path(settings["cache_folder"]) / "all_sheets_e_on_demonstrations.npz", frame_index = frames["frame_index"].to_numpy(), **outputs)
    report(f"all_sheets_e | {len(frames)} demonstration pictures | {time.perf_counter() - started:.0f} s | outputs {sorted(outputs)}")



def robot_runs_job(
        report,
        parameters: Optional[Dict] = None
) -> None:
    """
    Run model E over the logged pictures of every robot run of 11.11.26 and check its exact step against the log.

    The log holds what the robot's model answered at the time; for the runs with model E the recomputed exact step must
    agree with it, which proves that the pictures are fed in the same way here as on the robot. For runs 1 and 10 (models
    A and B) the outputs say what model E would have answered to the same pictures.
    """

    settings = common.merged_parameters(dict(OUTPUT_PARAMETERS, **(parameters or {})))
    job = settings["all_sheets_e"]
    loaded = common.load_model(job["run_name"], Path(settings["fold_runs_folder"]) / job["folder"], "all_sheets")
    saved = {}
    for run_folder in sorted(Path(settings["model_runs_folder"]).glob(settings["robot_run_prefix"] + "*")):
        steps = pandas.read_csv(run_folder / "steps.csv")
        outputs = common.outputs_of_pictures(loaded, common.robot_run_pictures(run_folder), is_hidden_kept = True, parameters = settings)
        difference = numpy.abs(outputs["exact"][:, :2] - steps[["model_tip_forward_cm", "model_tip_left_cm"]].to_numpy())
        report(f"robot run {run_folder.name} | {len(steps)} pictures | largest difference to the logged exact step: {difference.max():.4f} cm")
        for name, values in outputs.items():
            saved[f"{run_folder.name}__{name}"] = values
    Path(settings["cache_folder"]).mkdir(parents = True, exist_ok = True)
    numpy.savez_compressed(Path(settings["cache_folder"]) / "all_sheets_e_on_robot_runs.npz", **saved)



def main(job_names: List[str]) -> None:

    report = common.TeeOutput(common.ANALYSIS_FOLDER / f"s02_model_outputs.{'_'.join(job_names)}.out.txt")
    frames = common.load_demonstration_frames()
    report(f"demonstration pictures after the audit's cleaning: {len(frames)}")
    for job_name in job_names:
        if job_name == "all_sheets_e":
            all_sheets_job(frames, report)
        elif job_name == "robot_runs":
            robot_runs_job(report)
        else:
            fold_job(job_name, frames, report)



if __name__ == "__main__":
    main(sys.argv[1:])
