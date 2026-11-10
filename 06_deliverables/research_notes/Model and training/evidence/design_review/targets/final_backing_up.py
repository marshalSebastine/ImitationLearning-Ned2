"""
Backward moves for the final candidate's finished folds (read only): the share of moving pictures within 45 deg on
backward moves and on the others, per held-out sheet. Writes final_backing_up.out.txt.
"""

import sys

import numpy
import pandas

from common_frames import DIRECTION_NAMES, OUTPUT_FOLDER, RUNS_FOLDER, read_frames
from per_class_direction import nearest

BACK = ["back", "back_left", "back_right"]


def main(folds) -> None:
    frames = read_frames()[["frame_index", "action", "move_angle_deg"]]
    lines = []
    for fold in folds:
        data = numpy.load(RUNS_FOLDER / "2026-11-09_resnet18_final_candidate" / f"predictions_{fold}.npz")
        table = pandas.DataFrame({"frame_index": data["frame_index"], "angle": data["angle_deg"]}).merge(frames, on = "frame_index", how = "left")
        table = table[table["action"].isin(DIRECTION_NAMES)]
        error = numpy.abs((table["angle"] - table["move_angle_deg"] + 180.0) % 360.0 - 180.0)
        is_back = numpy.isin(nearest(table["move_angle_deg"].to_numpy()), BACK)
        lines.append(f"{fold}: backward moves {int(is_back.sum())} of {len(table)}; within 45 deg {(error[is_back] <= 45).mean():.3f} on backward moves, "
                     f"{(error[~is_back] <= 45).mean():.3f} on the others; predicted straight back {(nearest(table['angle'].to_numpy()) == 'back').mean():.3f}")
    print("\n".join(lines))
    (OUTPUT_FOLDER / "final_backing_up.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["without_path_01", "without_path_02"])
