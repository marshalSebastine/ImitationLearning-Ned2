"""
A check on the coverage tables of 02_coverage.py: do the held-out-sheet scores of the saved fold models fall where the
demonstrations are thin?

No model is run: the saved predictions of the fold models (each trained without one sheet, 04_runs/gpu_ki014) give the
predicted step direction of every held-out picture. A picture is a hit when that direction lies within 45 degrees of
the path of the next three steps (the report's "path within 45"). The hits are grouped by
    - the travel cell (direction from radial against reach) and how many training pictures of the other five sheets
      the cell held for that fold,
    - where the path lies in the picture and how far the forearm is twisted,
    - the measured lay-out of the episode (01_placements.py) and, for sheet 02, the start kind (square lay-outs),
    - the first rows of an episode.
"Always forward" (the picture's top) is scored beside the model, because some cells are easy for it and others not.

Printed and saved: 07_offline_score_by_coverage.out.txt, 07_score_by_cell.csv.

Run from this folder (after 01_placements.py):
    D:/AI_Project/.venv/Scripts/python.exe 07_offline_score_by_coverage.py
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas

import orientation_common
from orientation_common import EVIDENCE_FOLDER, ROOT_FOLDER, demonstration_frames, wrapped_deg

SCORE_PARAMETERS = {
    "runs_folder": ROOT_FOLDER / "04_runs" / "gpu_ki014",
    # the fold runs scored: label -> run folder (the recipes of models A and E, two seeds each)
    "fold_runs": {
        "A recipe": "2026-11-09_resnet18_final_candidate",
        "A recipe, seed 2": "2026-11-09_resnet18_final_candidate_seed_2",
        "E recipe": "2026-11-10_resnet18_chunk_stop_weight_25",
        "E recipe, seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"
    },
    "sheets": ["01", "02", "03", "04", "05", "06"],
    "hit_within_deg": 45.0,
    # the path of the next three steps must be at least this long to be scored (TRAINING_METRIC_PARAMETERS)
    "path_minimum_cm": 1.5,
    "direction_bin_deg": 30.0,
    "reach_edges_cm": [12.0, 20.0, 27.5, 35.0, 42.5, 49.5],
    "training_picture_edges": [0, 50, 150, 400, 1000, 100000],
    "first_rows": 10
}



def scored_pictures(
        frames: pandas.DataFrame,
        run_folder: Path,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Join one fold run's saved predictions to the demonstration rows and mark the hits.

    Returns
    -------
    pandas.DataFrame
        The held-out rows that have a path direction, with "predicted_deg" (the predicted step direction in the
        picture), "is_hit" (within "hit_within_deg" of the path) and "is_forward_hit" (the picture's top within it).
    """

    predicted = pandas.Series(numpy.nan, index = frames["frame_index"].to_numpy())
    for sheet in parameters["sheets"]:
        file = run_folder / f"predictions_without_path_{sheet}.npz"
        saved = numpy.load(file)
        known = numpy.isin(saved["frame_index"], predicted.index.to_numpy())
        predicted.loc[saved["frame_index"][known]] = saved["angle_deg"][known]
    scored = frames.copy()
    scored["predicted_deg"] = predicted.to_numpy()
    grouped = scored.groupby("episode_id", sort = False)
    ahead = parameters["path_steps_ahead"]
    path_cm = 100.0 * numpy.hypot(grouped["tip_x_m"].shift(-ahead) - scored["tip_x_m"], grouped["tip_y_m"].shift(-ahead) - scored["tip_y_m"])
    is_scored = scored["path_table_deg"].notna() & scored["predicted_deg"].notna() & (path_cm >= parameters["path_minimum_cm"]) & (scored["action"] != "stop")
    scored = scored[is_scored].copy()
    scored["is_hit"] = numpy.abs(wrapped_deg(scored["predicted_deg"] - scored["path_picture_deg"])) <= parameters["hit_within_deg"]
    scored["is_forward_hit"] = scored["path_picture_deg"].abs() <= parameters["hit_within_deg"]
    return scored



def main(parameters: Optional[Dict] = None) -> None:
    """Print the scores by coverage and save the per-cell table."""

    settings = orientation_common.merged_parameters(SCORE_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 500)
    pandas.set_option("display.max_columns", 40)
    frames = demonstration_frames(settings)
    layouts = pandas.read_csv(EVIDENCE_FOLDER / "01_placements_per_episode.csv", dtype = {"sheet": str}).set_index("episode_id")
    frames["layout"] = frames["episode_id"].map(layouts["layout"])
    bin_deg = settings["direction_bin_deg"]
    frames["direction_bin"] = wrapped_deg(numpy.round(frames["path_from_radial_deg"] / bin_deg) * bin_deg)
    frames.loc[frames["direction_bin"] == -180.0, "direction_bin"] = 180.0
    labels = [f"{low:g} to {high:g}" for low, high in zip(settings["reach_edges_cm"][:-1], settings["reach_edges_cm"][1:])]
    frames["reach_bin"] = pandas.cut(frames["reach_cm"], settings["reach_edges_cm"], include_lowest = True, labels = labels).astype(str)

    # the pictures each fold trained on, per cell: those of the other five sheets
    with_path = frames[frames["path_table_deg"].notna()]
    per_sheet = with_path.groupby(["sheet_number", "reach_bin", "direction_bin"]).size().rename("pictures").reset_index()
    all_sheets = per_sheet.groupby(["reach_bin", "direction_bin"])["pictures"].sum().rename("all_pictures").reset_index()
    training = per_sheet.merge(all_sheets, on = ["reach_bin", "direction_bin"], how = "right")
    training["training_pictures"] = training["all_pictures"] - training["pictures"]

    all_cells = []
    for label, folder in settings["fold_runs"].items():
        run_folder = Path(settings["runs_folder"]) / folder
        if not run_folder.exists():
            print(f"{label}: {run_folder} is missing, skipped")
            continue
        scored = scored_pictures(frames, run_folder, settings)
        per_sheet_score = scored.groupby("sheet_number")["is_hit"].mean()
        print(f"\n=== {label} ({folder}) ===")
        print(f"Pictures scored {len(scored)}; path within {settings['hit_within_deg']:.0f} degrees: all pictures {scored['is_hit'].mean():.3f}, mean of the six sheets "
              f"{per_sheet_score.mean():.3f} (always forward {scored.groupby('sheet_number')['is_forward_hit'].mean().mean():.3f})")

        cells = scored.groupby(["sheet_number", "reach_bin", "direction_bin"]).agg(
            pictures = ("is_hit", "size"), hits = ("is_hit", "sum"), forward_hits = ("is_forward_hit", "sum")).reset_index()
        cells = cells.merge(training[["sheet_number", "reach_bin", "direction_bin", "training_pictures"]], on = ["sheet_number", "reach_bin", "direction_bin"], how = "left")
        cells["training_pictures"] = cells["training_pictures"].fillna(0)
        cells["run"] = label
        all_cells.append(cells)
        cells["training_bin"] = pandas.cut(cells["training_pictures"], settings["training_picture_edges"], include_lowest = True)
        by_training = cells.groupby("training_bin", observed = True).agg(cells = ("pictures", "size"), pictures = ("pictures", "sum"), hits = ("hits", "sum"),
                                                                         forward_hits = ("forward_hits", "sum"))
        by_training["path_within_45"] = (by_training["hits"] / by_training["pictures"]).round(3)
        by_training["always_forward"] = (by_training["forward_hits"] / by_training["pictures"]).round(3)
        print("By how many training pictures (of the other five sheets) the picture's travel cell held:")
        print(by_training[["cells", "pictures", "path_within_45", "always_forward"]].to_string())
        big = cells[cells["pictures"] >= 20]
        rank_training = big["training_pictures"].rank()
        rank_score = (big["hits"] / big["pictures"]).rank()
        print(f"  rank correlation between a cell's training pictures and its score, over {len(big)} sheet cells of 20 pictures or more: "
              f"{numpy.corrcoef(rank_training, rank_score)[0, 1]:+.2f}")

        by_direction = scored.groupby("direction_bin").agg(pictures = ("is_hit", "size"), path_within_45 = ("is_hit", "mean"), always_forward = ("is_forward_hit", "mean"))
        print("By travel direction from radial (all reaches):")
        print(by_direction.round(3).T.to_string())
        by_reach = scored.groupby("reach_bin").agg(pictures = ("is_hit", "size"), path_within_45 = ("is_hit", "mean"), always_forward = ("is_forward_hit", "mean")).reindex(labels)
        print("By reach (cm):")
        print(by_reach.round(3).T.to_string())

        scored["path_in_picture"] = pandas.cut(scored["path_picture_deg"].abs(), [0, 30, 60, 120, 180], include_lowest = True)
        by_picture = scored.groupby("path_in_picture", observed = True).agg(pictures = ("is_hit", "size"), path_within_45 = ("is_hit", "mean"),
                                                                            always_forward = ("is_forward_hit", "mean"))
        print("By where the path lies in the picture (degrees off its top):")
        print(by_picture.round(3).T.to_string())
        scored["twist"] = pandas.cut(scored["joint_4_deg"].abs(), [0, 30, 60, 90, 120], include_lowest = True)
        by_twist = scored.groupby("twist", observed = True).agg(pictures = ("is_hit", "size"), path_within_45 = ("is_hit", "mean"), always_forward = ("is_forward_hit", "mean"))
        print("By forearm twist (degrees, either side):")
        print(by_twist.round(3).T.to_string())

        layout_sizes = scored.groupby("layout")["episode_id"].nunique()
        scored["layout_episodes"] = scored["layout"].map(layout_sizes)
        scored["layout_size"] = pandas.cut(scored["layout_episodes"], [0, 2, 5, 10, 100], labels = ["1 to 2 episodes", "3 to 5", "6 to 10", "over 10"])
        by_layout = scored.groupby("layout_size", observed = True).agg(pictures = ("is_hit", "size"), episodes = ("episode_id", "nunique"),
                                                                        path_within_45 = ("is_hit", "mean"), always_forward = ("is_forward_hit", "mean"))
        print("By how many episodes share the picture's measured lay-out (the held-out sheet is unseen in every lay-out):")
        print(by_layout.round(3).to_string())
        sheet_02 = scored[scored["sheet_number"] == "02"].copy()
        sheet_02["kind"] = numpy.where(sheet_02["start_kind"].isin(["SQP", "SQL"]), sheet_02["start_kind"], "other start kinds")
        by_kind = sheet_02.groupby("kind").agg(pictures = ("is_hit", "size"), episodes = ("episode_id", "nunique"), path_within_45 = ("is_hit", "mean"),
                                              always_forward = ("is_forward_hit", "mean"))
        early = sheet_02[sheet_02["row_in_episode"] <= settings["first_rows"]].groupby("kind").agg(
            first_rows_pictures = ("is_hit", "size"), first_rows_path_within_45 = ("is_hit", "mean"), first_rows_always_forward = ("is_forward_hit", "mean"))
        print(f"Sheet 02 held out, by start kind (all rows, and the first {settings['first_rows']} rows):")
        print(by_kind.join(early).round(3).to_string())
        groups = {"STD": "standard", "OL1": "puck beside the line", "OR1": "puck beside the line", "OL1.5": "puck beside the line", "OR1.5": "puck beside the line",
                  "TL10": "sheet turned 10 or 15", "TR10": "sheet turned 10 or 15", "TL15": "sheet turned 10 or 15", "TR15": "sheet turned 10 or 15",
                  "SQP": "square, portrait (sheet 02)", "SQL": "square, landscape (sheet 02)", "NE": "taught start", "MID": "taught start", "X1": "taught start",
                  "X2": "taught start"}
        first = scored[scored["row_in_episode"] <= settings["first_rows"]].copy()
        first["start"] = first["start_kind"].map(groups)
        by_start = first.groupby("start").agg(pictures = ("is_hit", "size"), episodes = ("episode_id", "nunique"), path_within_45 = ("is_hit", "mean"),
                                              always_forward = ("is_forward_hit", "mean"),
                                              path_off_picture_top_median = ("path_picture_deg", lambda values: float(values.abs().median())))
        print(f"All six held-out sheets, the first {settings['first_rows']} rows of every episode, by kind of start:")
        print(by_start.round(3).to_string())

    if all_cells:
        table = pandas.concat(all_cells, ignore_index = True).drop(columns = ["training_bin"], errors = "ignore")
        table["path_within_45"] = (table["hits"] / table["pictures"]).round(3)
        table.to_csv(EVIDENCE_FOLDER / "07_score_by_cell.csv", index = False)
        print("\nSaved 07_score_by_cell.csv")



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding = "utf-8")
    main()
