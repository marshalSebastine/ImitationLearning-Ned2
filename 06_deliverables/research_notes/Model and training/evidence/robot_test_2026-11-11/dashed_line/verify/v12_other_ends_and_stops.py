"""
Verifier check 12: two things the analyst's answer does not cover.

(a) The other two start ends, which were demonstrated but not driven on 11.11.26: sheet 06 from end B (8 whole-line
    demonstrations) and sheet 05 from end A (5). How the wrist (joint 5) and the forearm (joint 4) are used there, to
    see whether a robot run from those ends would separate "dashed" from "wrist bent in the demonstrations".
(b) Stop chance of the models that never saw a dashed line (folds without sheet 06) on sheet 06's pictures, by what
    lies just ahead of the puck (v10's classes): do the gaps look like a line's end to them?

Run:  D:/AI_Project/.venv/Scripts/python.exe v12_other_ends_and_stops.py > v12_other_ends_and_stops.out.txt
Needs v05, v07 and v10 outputs. A few seconds.
"""

import numpy
import pandas

from verify_common import ROOT_FOLDER, VERIFY_FOLDER

pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
RUNS_FOLDER = ROOT_FOLDER / "04_runs" / "gpu_ki014"

print("(a) Wrist and forearm in the whole-line demonstrations, by sheet and start end (the driver holds the wrist at 13.5 deg)")
rows = []
for sheet in ("05", "06"):
    tracks = pandas.read_csv(VERIFY_FOLDER / f"v05_demonstration_tracks_{sheet}.csv")
    progress = tracks.groupby("plan")["from_start_cm"].agg(lambda values: values.max() - values.min())
    whole = tracks[tracks["plan"].map(progress) >= 35.0]
    for travel, part in whole.groupby("travel_from"):
        end = {("05", "last_end"): "B", ("05", "first_end"): "A", ("06", "first_end"): "A", ("06", "last_end"): "B"}[(sheet, travel)]
        loop = part[(part["along_cm"] >= 0) & (part["along_cm"] <= part["loop_length_cm"])]
        after = part[part["along_cm"] > part["loop_length_cm"]]
        is_backward = (part["move_angle_deg"].abs() > 90) & (part["move_length_cm"] > 0.3)
        rows.append({
            "sheet": sheet, "start end": end, "demonstrations": part["plan"].nunique(), "pictures": len(part),
            "wrist over 5 deg from 13.5, all": ((part["joint_5_deg"] - 13.5).abs() > 5).mean(),
            "in the loop": ((loop["joint_5_deg"] - 13.5).abs() > 5).mean(),
            "after the loop": ((after["joint_5_deg"] - 13.5).abs() > 5).mean(),
            "wrist median in the loop": loop["joint_5_deg"].median(), "wrist 90th in the loop": loop["joint_5_deg"].quantile(0.9),
            "wrist median after the loop": after["joint_5_deg"].median(),
            "forearm least": part.groupby("plan")["joint_4_deg"].min().median(), "forearm most": part.groupby("plan")["joint_4_deg"].max().median(),
            "backward steps per demonstration": is_backward.groupby(part["plan"]).sum().median(),
            "pictures per demonstration": part.groupby("plan").size().median(),
            "puck over 1 cm beside the line": (part["beside_cm"].abs() > 1.0).mean()
        })
print(pandas.DataFrame(rows).round(2).T.to_string())

print("\n(b) Stop chance of the folds that never saw sheet 06, on sheet 06's pictures more than 4 cm of travel before the stop row, by what lies ahead of the puck")
classes = pandas.read_csv(VERIFY_FOLDER / "v10_gap_ahead_rows.csv")
index = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False, usecols = ["frame_index", "goal_distance_cm", "steps_to_end"])
classes = classes.merge(index, on = "frame_index")
for label, folder in (("E recipe", "2026-11-10_resnet18_chunk_stop_weight_25"), ("E recipe, seed 2", "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"),
                      ("A recipe", "2026-11-09_resnet18_final_candidate")):
    saved = numpy.load(RUNS_FOLDER / folder / "predictions_without_path_06.npz")
    merged = classes.merge(pandas.DataFrame({"frame_index": saved["frame_index"], "stop_probability": saved["stop_probability"]}), on = "frame_index")
    far = merged[(merged["action"] != "stop") & (merged["steps_to_end"] >= 5)]
    table = far.groupby("ahead")["stop_probability"].agg(pictures = "size", mean = "mean", over_half = lambda values: (values >= 0.5).mean())
    table2 = far.groupby("view")["stop_probability"].agg(pictures = "size", mean = "mean", over_half = lambda values: (values >= 0.5).mean())
    print(f"  {label}:")
    print(pandas.concat([table, table2]).round(3).to_string())
    by_part = far.groupby("part")["stop_probability"].agg(pictures = "size", mean = "mean", over_half = lambda values: (values >= 0.5).mean())
    print(by_part.round(3).to_string())
for sheet in ("05", "06"):
    saved = numpy.load(RUNS_FOLDER / "2026-11-10_resnet18_chunk_stop_weight_25" / f"predictions_without_path_{sheet}.npz")
    rows_all = pandas.read_csv(VERIFY_FOLDER / "v07_rows_05_06.csv").merge(index, on = "frame_index").merge(
        pandas.DataFrame({"frame_index": saved["frame_index"], "stop_probability": saved["stop_probability"]}), on = "frame_index")
    far = rows_all[(rows_all["action"] != "stop") & (rows_all["steps_to_end"] >= 5)]
    print(f"  E recipe fold without sheet {sheet}, all its pictures 5 or more rows before the end: {len(far)} pictures, stop chance 0.5 or more in {100 * (far['stop_probability'] >= 0.5).mean():.1f} %")
