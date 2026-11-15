"""
How much the per-place direction scores move between seeds, from the predictions saved on ki-014 (no model is run).

For every saved six-sheet run of the exact recipe (five seeds), the chunk recipe (two seeds) and the chunk recipe with
the stop weighted 25 (recipe E, two seeds) the script scores the saved directions against the demonstration's path
direction (3 steps ahead) per kind of place. Exact runs save the exact step's direction; chunk runs save the direction
of the chunk's first step ("angle_deg") and of its place 3 steps ahead ("angle_vector_deg"). A difference between two
ways of steering counts only when it is larger than the spread between seeds of one way.

Run from this folder:  python s09_seed_spread.py
Output: s09_seed_spread.out.txt, s09_seed_spread.csv.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas

import common

SEED_PARAMETERS = {
    "runs": {
        "exact recipe (A)": ["2026-11-09_resnet18_final_candidate", "2026-11-09_resnet18_final_candidate_seed_2", "2026-11-10_resnet18_final_candidate_seed_3",
                             "2026-11-10_resnet18_final_candidate_seed_4", "2026-11-10_resnet18_final_candidate_seed_5"],
        "chunk recipe": ["2026-11-10_resnet18_final_candidate_chunk", "2026-11-10_resnet18_final_candidate_chunk_seed_2"],
        "chunk recipe, stop weight 25 (E)": ["2026-11-10_resnet18_chunk_stop_weight_25", "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"]
    }
}



def main(parameters: Optional[Dict] = None) -> None:

    settings = common.merged_parameters(dict(SEED_PARAMETERS, **(parameters or {})))
    report = common.TeeOutput(common.ANALYSIS_FOLDER / "s09_seed_spread.out.txt")
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    frames = common.load_demonstration_frames(settings)
    path = common.path_direction(frames, settings)
    frames["path_deg"] = path["path_deg"].to_numpy()
    frames["is_path_row"] = path["is_path_row"].to_numpy()
    indexed = frames.set_index("frame_index")

    rows = []
    for recipe, folders in settings["runs"].items():
        for seed_number, folder in enumerate(folders, start = 1):
            parts = []
            for sheet in sorted(frames["sheet"].unique()):
                saved = numpy.load(Path(settings["fold_runs_folder"]) / folder / f"predictions_without_{sheet}.npz")
                part = {"frame_index": saved["frame_index"], "first": saved["angle_deg"]}
                if "angle_vector_deg" in saved.files:
                    part["ahead"] = saved["angle_vector_deg"]
                parts.append(pandas.DataFrame(part))
            joined = indexed.join(pandas.concat(parts).set_index("frame_index"), how = "inner")
            joined = joined[joined["is_path_row"] & (joined["place"] != "no fit")]
            for place in common.PLACE_NAMES + ["all places"]:
                part = joined if place == "all places" else joined[joined["place"] == place]
                row = {"recipe": recipe, "seed": seed_number, "place": place, "path rows": len(part),
                       "first step within 45": round(float((common.angle_difference_deg(part["first"], part["path_deg"]) <= 45).mean()), 3)}
                if "ahead" in part.columns:
                    row["place 3 steps ahead within 45"] = round(float((common.angle_difference_deg(part["ahead"], part["path_deg"]) <= 45).mean()), 3)
                rows.append(row)
    table = pandas.DataFrame(rows)
    table.to_csv(common.ANALYSIS_FOLDER / "s09_seed_spread.csv", index = False)
    report("== path within 45 degrees per kind of place, every saved seed (held-out sheets; measured from the saved predictions) ==")
    report("-- the direction of the exact step (exact recipe) or of the chunk's first step (chunk recipes) --")
    report(table.pivot(index = ["recipe", "seed"], columns = "place", values = "first step within 45")[common.PLACE_NAMES + ["all places"]].to_string())
    report("-- the direction to the chunk's place 3 steps ahead (chunk recipes only) --")
    ahead = table.dropna(subset = ["place 3 steps ahead within 45"])
    report(ahead.pivot(index = ["recipe", "seed"], columns = "place", values = "place 3 steps ahead within 45")[common.PLACE_NAMES + ["all places"]].to_string())
    report("-- the 3-steps-ahead direction minus the first-step direction of the same model (points of path within 45) --")
    difference = ahead.copy()
    difference["gain"] = (100.0 * (difference["place 3 steps ahead within 45"] - difference["first step within 45"])).round(1)
    report(difference.pivot(index = ["recipe", "seed"], columns = "place", values = "gain")[common.PLACE_NAMES + ["all places"]].to_string())
    exact_only = table[table["recipe"] == "exact recipe (A)"]
    spread = exact_only.groupby("place")["first step within 45"].agg(["min", "max"])
    spread["range in points"] = (100.0 * (spread["max"] - spread["min"])).round(1)
    report("-- the spread between the five seeds of the exact recipe --")
    report(spread.loc[common.PLACE_NAMES + ["all places"]].to_string())



if __name__ == "__main__":
    main()
