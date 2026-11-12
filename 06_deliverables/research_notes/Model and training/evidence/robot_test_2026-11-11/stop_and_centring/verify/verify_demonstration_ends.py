"""
Verifier's check 4 (11.11.26): how the demonstrations end, with my own measure (end_tools.py), for every episode of
the sheets given: the puck's foot in the stop picture against the line's round end as single earlier pictures show it.

Usage: python verify_demonstration_ends.py <camera> <first sheet> <last sheet>
    camera: "fitted" (the camera of 6.11.26 as it stands in config.py), "corrected" (the analyst's picture shift and
    roll per period, read from ../camera_shift.json) or "swapped" (each period with the OTHER period's correction)
Writes verify_demonstration_ends_<camera>_<first>_<last>.csv. Reads frames.csv and the raw pictures only.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy
import pandas

import end_tools
import my_measure

cv2.setNumThreads(2)
HERE = Path(__file__).resolve().parent
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
CAMERA_MOVED_AT = "2026-11-08T19:00:00"
LAST_PICTURES = 18


def main(camera_name: str, first_sheet: int, last_sheet: int) -> None:

    frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False)
    shifts = json.loads((HERE.parent / "camera_shift.json").read_text(encoding = "utf-8"))
    sheets = [f"path_{number:02d}" for number in range(first_sheet, last_sheet + 1)]
    rows = []
    for episode_id, steps in frames[frames["sheet"].isin(sheets)].groupby("episode_id", sort = False):
        steps = steps.sort_values("row_in_episode")
        period = "after_move" if str(steps["started"].iloc[0]) >= CAMERA_MOVED_AT else "before_move"
        camera = None
        if camera_name == "corrected":
            camera = my_measure.shifted_camera(shifts[period]["shift_px"], shifts[period]["roll_deg"])
        if camera_name == "swapped":
            # the other period's correction, as a check that the two periods really need different ones
            other = "before_move" if period == "after_move" else "after_move"
            camera = my_measure.shifted_camera(shifts[other]["shift_px"], shifts[other]["roll_deg"])
        last = steps.iloc[-LAST_PICTURES:]
        pictures = [cv2.imread(str(RAW_FOLDER / frame_path)) for frame_path in last["frame_path"]]
        entry = {"episode_id": episode_id, "sheet": steps["sheet"].iloc[0], "period": period, "pictures": len(steps), "last_action": steps["action"].iloc[-1]}
        if any(picture is None for picture in pictures) or steps["action"].iloc[-1] != "stop":
            rows.append(entry)
            continue
        joints = [[row[f"joint_{index}"] for index in range(1, 7)] for _, row in last.iterrows()]
        tips = last[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
        measured = end_tools.measure_stop(pictures, joints, tips, camera, reference_offsets = range(2, LAST_PICTURES))
        entry["is_puck_found"] = measured["puck"] is not None
        if measured["puck"] is not None:
            entry.update({"puck_x": float(measured["puck"]["xy"][0]), "puck_y": float(measured["puck"]["xy"][1]), "puck_residual_px": measured["puck"]["residual_px"]})
        summary = end_tools.summary_of(measured["references"])
        entry["references"] = 0 if summary is None else summary["references"]
        if summary is not None:
            entry.update({"short_mm": summary["short_mm"], "aside_mm": summary["aside_mm"], "gap_mm": summary["gap_mm"],
                          "short_low_mm": summary["short_range_mm"][0], "short_high_mm": summary["short_range_mm"][1],
                          "aside_low_mm": summary["aside_range_mm"][0], "aside_high_mm": summary["aside_range_mm"][1]})
        rows.append(entry)
        if len(rows) % 10 == 0:
            print(f"{len(rows)} episodes measured", flush = True)
    table = pandas.DataFrame(rows)
    out_path = HERE / f"verify_demonstration_ends_{camera_name}_{first_sheet:02d}_{last_sheet:02d}.csv"
    table.to_csv(out_path, index = False, float_format = "%.3f")
    print(f"written {out_path.name}: {len(table)} episodes, end measured in {int((table['references'] > 0).sum())}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
