"""
Per-situation results of the recovery ablation (recovery_ablation.py), read only.

For every row rule (all, no_corrections, no_matched_first_pass) and seed, the held-out predictions saved in
recovery_ablation/predictions are scored per situation with the same hits as situation_errors.py; the table gives the
mean over seeds and the seed range. It also counts where the off-line pictures of the data sit (first pass or
correction replay).

Usage: python recovery_ablation_situations.py
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/02_code"
sys.path.insert(0, CODE_FOLDER)
sys.path.insert(0, str(HERE))

import numpy
import pandas

from config import TRAINING_ERROR_ANALYSIS_PARAMETERS, TRAINING_METRIC_PARAMETERS
from training_error_analysis import situation_columns
from training_metrics import angle_difference_deg
from training_targets import chunk_targets, load_training_frames
from situation_errors import registration_situations

PARAMETERS = {
    "prediction_folder": HERE / "recovery_ablation" / "predictions",
    "results_file": HERE / "recovery_ablation" / "recovery_ablation.json",
    "rules": ["all", "no_corrections", "no_matched_first_pass"],
    "seeds": [1, 2, 3],
    "folds": [f"without_path_0{number}" for number in range(1, 7)],
}


def main(parameters = None):
    """
    Print the tables.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    frames = load_training_frames()
    situations = situation_columns(frames, TRAINING_ERROR_ANALYSIS_PARAMETERS)
    situations = pandas.concat([situations, registration_situations(frames)], axis = 1)
    is_start = ((frames["waypoint"].fillna(-1) == 0) & (frames["row_in_episode"] > 0)).astype(int)
    situations["pass"] = numpy.where(is_start.groupby(frames["episode_id"]).cumsum() > 0, "correction replay", "first pass")
    is_moving = frames["action"].isin(["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"]).to_numpy()
    print("=== where the moving pictures sit (all rows, registered good fits) ===")
    counts = pandas.crosstab(situations.loc[is_moving, "off_line"], situations.loc[is_moving, "pass"])
    print(counts.to_string())
    print()
    index_of_frame = pandas.Series(numpy.arange(len(frames)), index = frames["frame_index"].to_numpy())
    chunk = chunk_targets(frames, TRAINING_METRIC_PARAMETERS["path_steps"])[:, -1, :]
    path_angle = numpy.degrees(numpy.arctan2(chunk[:, 1], chunk[:, 0]))
    has_path = (numpy.linalg.norm(chunk, axis = 1) >= TRAINING_METRIC_PARAMETERS["path_minimum_cm"]) & (frames["action"] != "stop").to_numpy()
    rows = []
    for rule in settings["rules"]:
        for seed in settings["seeds"]:
            parts = []
            for fold in settings["folds"]:
                path = settings["prediction_folder"] / f"recovery_{rule}_seed_{seed}_{fold}.npz"
                if not path.exists():
                    continue
                predictions = numpy.load(path)
                positions = index_of_frame.reindex(predictions["frame_index"]).to_numpy()
                is_known = ~numpy.isnan(positions)
                positions = positions[is_known].astype(int)
                predicted = predictions["angle_deg"][is_known]
                part = situations.iloc[positions].copy()
                part["step_hit"] = numpy.where(is_moving[positions], angle_difference_deg(predicted, frames["move_angle_deg"].to_numpy()[positions]) <= 45, numpy.nan)
                part["path_hit"] = numpy.where(has_path[positions], angle_difference_deg(predicted, path_angle[positions]) <= 45, numpy.nan)
                parts.append(part)
            if not parts:
                continue
            scored = pandas.concat(parts)
            for column in ["sheet", "place", "shape", "off_line", "crossing", "line_radius", "line_to_go", "pass"]:
                for value, group in scored.groupby(column):
                    rows.append({"rule": rule, "seed": seed, "situation": column, "value": value,
                                 "moving_pictures": int(group["step_hit"].notna().sum()),
                                 "step_within_45": float(group["step_hit"].mean()),
                                 "path_within_45": float(group["path_hit"].mean())})
            rows.append({"rule": rule, "seed": seed, "situation": "all", "value": "all",
                         "moving_pictures": int(scored["step_hit"].notna().sum()),
                         "step_within_45": float(scored["step_hit"].mean()),
                         "path_within_45": float(scored["path_hit"].mean())})
    table = pandas.DataFrame(rows)
    table.to_csv(HERE / "recovery_ablation_situations.csv", index = False)
    summary = table.groupby(["situation", "value", "rule"]).agg(
        pictures = ("moving_pictures", "first"),
        seeds = ("seed", "count"),
        step_mean = ("step_within_45", "mean"),
        step_low = ("step_within_45", "min"),
        step_high = ("step_within_45", "max"),
        path_mean = ("path_within_45", "mean")).reset_index()
    wide = summary.pivot_table(index = ["situation", "value"], columns = "rule", values = "step_mean")
    if "no_corrections" in wide.columns and "all" in wide.columns:
        wide["no_corrections minus all"] = wide["no_corrections"] - wide["all"]
    if "no_matched_first_pass" in wide.columns and "all" in wide.columns:
        wide["matched minus all"] = wide["no_matched_first_pass"] - wide["all"]
    seeds_done = table.groupby("rule")["seed"].nunique().to_dict()
    print("seeds scored per rule:", seeds_done)
    pictures = summary[summary["rule"] == "all"].set_index(["situation", "value"])["pictures"]
    wide["pictures"] = pictures
    print("=== step within 45, mean of seeds, per situation ===")
    print(wide.round(3).to_string())
    print()
    print(summary.round(3).to_string(index = False))
    if settings["results_file"].exists():
        results = json.loads(settings["results_file"].read_text(encoding = "utf-8"))["results"]
        print()
        print("=== fold means from the probe results (within 45 / path within 45 / stopped right tuned two) ===")
        for result in results:
            summary_values = result["summary"]
            print(result["name"], summary_values.get("within_45_deg_mean"), summary_values.get("path_within_45_deg_mean"),
                  summary_values.get("tuned_two_episodes_stopped_right_mean"))


if __name__ == "__main__":
    main()
