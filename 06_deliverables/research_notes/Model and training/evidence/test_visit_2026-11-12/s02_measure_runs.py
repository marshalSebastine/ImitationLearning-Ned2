"""
Step 2: the picture measures of 11.11.26 (run_measures.measure_steps, unchanged) on every run of the visit.

Per picture: where the puck's foot stands on the table (a cylinder of the puck's size fitted to the red blob through the
forearm camera model, corrected as on 11.11), and its sideways offset from the line's middle against the line map voted
by the last 12 pictures ("method A"). At the last picture: how the puck stands on the line's end (end_cover), used for
sheet 01. The votes of every picture of each run are kept (in the scratch folder, they are large) for step 3, which lays
the printed drawing onto them.

Run from this folder:  python s02_measure_runs.py
Writes s02_steps_method_a.csv and s02_ends_method_a.json here, and <label>_votes.npz in the scratch folder.
"""

import json
import multiprocessing
from pathlib import Path

import numpy
import pandas

import visit_common
import run_measures


MEASURE_RUN_PARAMETERS = {
    "processes": 5
}



def measure_one(label: str) -> dict:
    """Measure one run; returns its rows and its end, and writes its votes to the scratch folder."""

    parameters = visit_common.merged_visit_parameters()
    folder = visit_common.run_folders()[label]
    steps = run_measures.robot_run_steps(folder)
    camera_parameters = run_measures.camera_parameters_for(parameters["camera_period"])
    measured = run_measures.measure_steps(steps, camera_parameters = camera_parameters)
    rows = measured["rows"].copy()
    rows.insert(0, "label", label)
    rows["tip_x"] = steps["tip_x"].to_numpy()[rows["step"].to_numpy()]
    rows["tip_y"] = steps["tip_y"].to_numpy()[rows["step"].to_numpy()]
    line_map = measured["line_map"]
    scratch = Path(parameters["scratch_folder"])
    scratch.mkdir(parents = True, exist_ok = True)
    numpy.savez_compressed(scratch / f"{label}_votes.npz", line_votes = measured["all_line_votes"], paper_votes = measured["all_paper_votes"],
                           end_line_votes = line_map.line_votes, end_paper_votes = line_map.paper_votes,
                           x_range = numpy.array(line_map.x_range), y_range = numpy.array(line_map.y_range))
    end = measured["end"]
    if end is not None:
        end = {key: (value.tolist() if isinstance(value, numpy.ndarray) else value) for key, value in end.items()}
    return {"label": label, "rows": rows, "end": end}



def main() -> None:

    labels = list(visit_common.run_folders().keys())
    with multiprocessing.Pool(MEASURE_RUN_PARAMETERS["processes"]) as pool:
        results = pool.map(measure_one, labels)
    table = pandas.concat([result["rows"] for result in results], ignore_index = True)
    table.to_csv(visit_common.EVIDENCE_FOLDER / "s02_steps_method_a.csv", index = False, float_format = "%.5g")
    ends = {result["label"]: result["end"] for result in results}
    (visit_common.EVIDENCE_FOLDER / "s02_ends_method_a.json").write_text(json.dumps(ends, indent = 1, default = float), encoding = "utf-8")
    for result in results:
        rows = result["rows"]
        print(f"{result['label']:5s} pictures {len(rows):3d}  puck fitted {int(rows['puck_x'].notna().sum()):3d}  "
              f"fit quality >= 0.8 {int((rows['fit_quality'] >= 0.8).sum()):3d}  offset measured {int(rows['offset_mm'].notna().sum()):3d}")



if __name__ == "__main__":
    main()
