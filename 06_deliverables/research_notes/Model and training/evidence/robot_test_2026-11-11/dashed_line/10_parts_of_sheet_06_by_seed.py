"""
Step 10 of the dashed-line analysis: how much the unseen-sheet score of sheet 06 moves between seeds and recipes,
by part of the line.

Step 4 found the tail after the loop of sheet 06 scored at 0.50 and 0.58 by the two E-recipe fold models and at 0.89
by the A-recipe one. Here the predictions saved on ki-014 for every fold run without sheet 06 are scored the same way
(path within 45 degrees; for chunk recipes the saved direction is the chunk's first step, which step 4 showed to
agree with the exact step within about 2 points), split by part of the line and by the end the episode started at.
Needs 04_fold_rows.csv (step 4).

Run: D:/AI_Project/.venv/Scripts/python.exe 10_parts_of_sheet_06_by_seed.py > 10_parts_of_sheet_06_by_seed.out.txt
"""

from pathlib import Path

import numpy
import pandas

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER
from model_answers import angle_error_deg

SEED_PARAMETERS = {
    "fold": "without_path_06",
    # runs whose saved direction is the executed vector ("angle_deg"); the names-plus-exact backup saves it as "angle_vector_deg"
    "vector_key_runs": ["resnet18_backup_names_plus_exact"]
}



def main() -> None:

    settings = SEED_PARAMETERS
    pandas.set_option("display.width", 250)
    rows = pandas.read_csv(EVIDENCE_FOLDER / "04_fold_rows.csv", low_memory = False)
    rows = rows[(rows["sheet"] == 6) & rows["is_path_row"] & (rows["part"] != "not placed")].set_index("frame_index")
    table = []
    for folder in sorted(Path(ANALYSIS_PARAMETERS["gpu_runs_folder"]).iterdir()):
        saved_path = folder / f"predictions_{settings['fold']}.npz"
        if not saved_path.exists():
            continue
        saved = numpy.load(saved_path)
        key = "angle_vector_deg" if any(name in folder.name for name in settings["vector_key_runs"]) else "angle_deg"
        asked = pandas.Series(saved[key], index = saved["frame_index"]).reindex(rows.index)
        is_within = pandas.Series(angle_error_deg(asked, rows["path_angle_deg"]) <= 45.0, index = rows.index)
        entry = {"fold run": folder.name[11:]}
        for part in ("before the crossing", "loop", "after the loop"):
            entry[part] = float(is_within[rows["part"] == part].mean())
        for end in ("A", "B"):
            entry[f"after the loop, from end {end}"] = float(is_within[(rows["part"] == "after the loop") & (rows["start_end"] == end)].mean())
        table.append(entry)
    table = pandas.DataFrame(table)
    counts = rows.groupby("part").size().to_dict()
    print(f"Sheet 06 held out, share of path pictures within 45 degrees, {len(table)} fold runs; path pictures per part: {counts}; "
          f"after the loop from end A {int(((rows['part'] == 'after the loop') & (rows['start_end'] == 'A')).sum())}, from end B "
          f"{int(((rows['part'] == 'after the loop') & (rows['start_end'] == 'B')).sum())}")
    print(table.round(3).to_string(index = False))
    print("\nspread over the fold runs (least, median, most):")
    print(table.drop(columns = ["fold run"]).agg(["min", "median", "max"]).round(3).to_string())
    tail = pandas.read_csv(EVIDENCE_FOLDER / "04_fold_rows.csv", low_memory = False)
    tail = tail[(tail["sheet"] == 6) & (tail["part"] == "after the loop") & ~tail["is_stop"]]
    print("\nwhat the demonstrations do after the loop of sheet 06 (cleaned rows): step direction in the picture, forearm and wrist")
    print(tail.groupby("start_end").agg(pictures = ("frame_index", "size"), step_direction_median_deg = ("demo_angle_deg", "median"),
                                        forearm_median_deg = ("joint_4_deg", "median"), wrist_median_deg = ("joint_5_deg", "median")).round(1).to_string())



if __name__ == "__main__":
    main()
