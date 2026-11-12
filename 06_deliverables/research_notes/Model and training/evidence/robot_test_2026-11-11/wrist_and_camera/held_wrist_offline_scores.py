"""
The offline scores again, with every held-out picture changed to the view the driver's held wrist would give
(key: wrist_and_camera, part 8).

The training runs scored the fold models on the demonstration pictures as recorded. On the robot the wrist is held
at 13.5 degrees, so the hand, the block and the table sit up to tens of pixels higher or lower in the picture than
in about a third of those pictures, and in more than half of the stop pictures (stop_views.py). Here every held-out
row is moved by its own amount (0.70 px at 160 per degree between its demonstrated wrist and 13.5; the fork kept; the
imitation of shift_sensitivity.py, crude for large moves), the fold model of model E's recipe that never saw the sheet
answers it, and the two offline scores are worked out with the project's own functions and rules:

    stops        training_metrics.stop_metrics, threshold 0.5 on two pictures in a row (the driver's rule)
    direction    the exact step (what the driver carries out) within 45 degrees of the demonstrated step

The answers to the unchanged pictures are the ones the training runs saved (the inference path is checked against
them on a sample). Four to seven minutes on two threads, depending on what else runs. Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe held_wrist_offline_scores.py
"""

from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas
import torch

import wrist_geometry
import model_mode
import shift_sensitivity
import training_targets
from training_metrics import stop_metrics


SCORE_PARAMETERS = {
    "stop_threshold": 0.5,
    "stop_pictures_in_a_row": 2,
    "check_rows_per_sheet": 60
}



def episode_outcomes(
        rows: pandas.DataFrame,
        stop_probability: numpy.ndarray,
        parameters: Dict
) -> Dict[str, int]:
    """How many of the rows' episodes stop right, too early, at another distance, or never (the project's stop rule)."""

    result = stop_metrics((rows["action"] == "stop").to_numpy().astype(float), stop_probability, rows["goal_distance_cm"].fillna(0.0).to_numpy(),
                          rows["episode_id"].to_numpy(), threshold = parameters["stop_threshold"], consecutive_pictures = parameters["stop_pictures_in_a_row"])
    episodes = rows["episode_id"].nunique()
    right = int(round(result["episodes_stopped_right"] * episodes))
    early = int(round(result["episodes_stopped_early"] * episodes))
    never = int(round(result["episodes_never_stopped"] * episodes))
    return {"episodes": episodes, "stopped_right": right, "stopped_early": early, "never_stopped": never, "stopped_between": episodes - right - early - never}



def main(parameters: Optional[Dict] = None) -> None:
    """Score the fold models on the pictures as recorded and in the imitated held-wrist view, per sheet and in all."""

    settings = dict(shift_sensitivity.SENSITIVITY_PARAMETERS)
    settings.update(SCORE_PARAMETERS)
    settings.update(parameters or {})
    torch.set_num_threads(settings["threads"])
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    held = wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]

    frames = training_targets.load_training_frames()
    frames["joint_5_deg"] = numpy.degrees(frames["joint_5"])
    frames["is_on_mark"] = frames["sheet_orientation"].str.contains("START MARK")
    frames["demonstrated_angle_deg"] = numpy.degrees(numpy.arctan2(frames["tip_left_m"].fillna(0.0), frames["tip_forward_m"].fillna(0.0)))
    frames["is_moving"] = (frames["action"] != "stop") & (numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0)) >= settings["smallest_step_m"])
    images = numpy.load(settings["cache_file"], mmap_mode = "r")
    mask = shift_sensitivity.fork_mask(images, settings)

    tables = []
    for sheet in settings["sheets"]:
        saved = numpy.load(Path(settings["fold_run_folder"]) / f"predictions_without_{sheet}.npz")
        saved_stop = pandas.Series(saved["stop_probability"].astype(float), index = saved["frame_index"].astype(int))
        rows = frames[(frames["sheet"] == sheet) & frames["frame_index"].isin(saved_stop.index)].copy()
        policy = model_mode.ModelPolicy(settings["fold_run_name"], Path(settings["fold_run_folder"]) / "checkpoints" / f"without_{sheet}.pt",
                                        Path(settings["fold_run_folder"]) / "metrics.json")
        torch.set_num_threads(settings["threads"])
        pictures = numpy.stack([images[index] for index in rows["frame_index"]])
        shifts = numpy.rint(settings["pixels_per_wrist_degree_160"] * (rows["joint_5_deg"].to_numpy() - held)).astype(int)

        check = numpy.linspace(0, len(rows) - 1, settings["check_rows_per_sheet"]).astype(int)
        checked = shift_sensitivity.model_answers(policy, pictures[check], settings)
        largest_gap = float(numpy.abs(checked["stop_probability"] - saved_stop.reindex(rows["frame_index"].to_numpy()[check]).to_numpy()).max())

        held_view = shift_sensitivity.held_wrist_pictures(pictures, shifts, mask)
        answers = shift_sensitivity.model_answers(policy, held_view, settings)
        rows["stop_as_recorded"] = saved_stop.reindex(rows["frame_index"].to_numpy()).to_numpy()
        rows["stop_held_view"] = answers["stop_probability"]
        rows["shift_160_px"] = shifts
        rows["exact_angle_held_view_deg"] = answers["exact_angle_deg"]
        # the saved file holds the chunk's first step, not the exact step, so the unchanged pictures are answered again for the direction
        rows["exact_angle_as_recorded_deg"] = shift_sensitivity.model_answers(policy, pictures, settings)["exact_angle_deg"]
        rows["twist_held_view_deg"] = answers["twist_deg"]
        tables.append(rows)
        print(f"{sheet}: {len(rows)} rows, {rows['episode_id'].nunique()} episodes; inference check against the saved stop chances: largest gap {largest_gap:.6f}; "
              f"rows moved by more than 4 px: {100.0 * (numpy.abs(shifts) > 4).mean():.1f} %")
    scored = pandas.concat(tables, ignore_index = True)
    # a compact row table (the episode and the picture are found again through frame_index in frames.csv)
    scored[["frame_index", "sheet", "row_in_episode", "action", "goal_distance_cm", "joint_5_deg", "shift_160_px", "stop_as_recorded", "stop_held_view",
            "exact_angle_as_recorded_deg", "exact_angle_held_view_deg", "demonstrated_angle_deg", "is_on_mark"]].round(4).to_csv(output_folder / "held_wrist_offline_rows.csv", index = False)

    print(f"\n=== episodes by where the stop rule (stop chance {settings['stop_threshold']} on {settings['stop_pictures_in_a_row']} pictures in a row) first fires ===")
    summary = []
    groups = [("all sheets", numpy.ones(len(scored), dtype = bool)), ("start-mark episodes", scored["is_on_mark"].to_numpy()), ("taught-start episodes", ~scored["is_on_mark"].to_numpy())]
    groups += [(sheet, (scored["sheet"] == sheet).to_numpy()) for sheet in settings["sheets"]]
    for label, is_group in groups:
        part = scored[is_group]
        for view, column in [("as recorded", "stop_as_recorded"), ("held-wrist view", "stop_held_view")]:
            summary.append(dict({"rows": label, "view": view}, **episode_outcomes(part, part[column].to_numpy(), settings)))
    summary = pandas.DataFrame(summary)
    print(summary.to_string(index = False))
    summary.to_csv(output_folder / "held_wrist_offline_stops.csv", index = False)

    print("\n=== direction: exact steps within 45 degrees of the demonstrated step (rows that move the tip 5 mm or more) ===")
    moving = scored[scored["is_moving"]]
    direction = []
    is_far = (moving["shift_160_px"].abs() > 4).to_numpy()
    for label, is_group in [("all sheets", numpy.ones(len(moving), dtype = bool)), ("rows moved by more than 4 px", is_far), ("rows moved by 4 px or less", ~is_far)] \
            + [(sheet, (moving["sheet"] == sheet).to_numpy()) for sheet in settings["sheets"]]:
        part = moving[is_group]
        recorded = shift_sensitivity.angle_gap(part["exact_angle_as_recorded_deg"], part["demonstrated_angle_deg"]) <= 45.0
        changed = shift_sensitivity.angle_gap(part["exact_angle_held_view_deg"], part["demonstrated_angle_deg"]) <= 45.0
        direction.append({"rows": label, "count": len(part), "within_45_as_recorded": float(recorded.mean()), "within_45_held_view": float(changed.mean()),
                          "change_points": 100.0 * float(changed.mean() - recorded.mean())})
    direction = pandas.DataFrame(direction)
    print(direction.round(3).to_string(index = False))
    direction.to_csv(output_folder / "held_wrist_offline_direction.csv", index = False)
    print(f"\nsaved held_wrist_offline_rows.csv, held_wrist_offline_stops.csv and held_wrist_offline_direction.csv in {output_folder}")



if __name__ == "__main__":
    main()
