"""
Fold spread of the key situations (read only): step within 45 per held-out sheet for the puck's distance from the
line, the line left to go and the line radius, for the end-to-end exact run and the frozen exact probe.

Usage: python situation_errors_by_sheet.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
sys.path.insert(0, str(HERE))

import numpy
import pandas

from config import TRAINING_ERROR_ANALYSIS_PARAMETERS
from training_error_analysis import prediction_files, situation_columns
from training_metrics import angle_difference_deg
from training_targets import load_training_frames
from situation_errors import registration_situations


def main():
    """
    Print one table per run and situation: rows are situation values, columns held-out sheets.
    """
    frames = load_training_frames()
    situations = situation_columns(frames, TRAINING_ERROR_ANALYSIS_PARAMETERS)
    situations = pandas.concat([situations, registration_situations(frames)], axis = 1)
    is_moving = frames["action"].isin(["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"]).to_numpy()
    index_of_frame = pandas.Series(numpy.arange(len(frames)), index = frames["frame_index"].to_numpy())
    for name in ("resnet18_finetune_exact", "exact_step"):
        parts = []
        for file in prediction_files(name):
            predictions = numpy.load(file)
            positions = index_of_frame.reindex(predictions["frame_index"]).to_numpy()
            is_known = ~numpy.isnan(positions)
            positions = positions[is_known].astype(int)
            part = situations.iloc[positions].copy()
            error = angle_difference_deg(predictions["angle_deg"][is_known], frames["move_angle_deg"].to_numpy()[positions])
            part["step_hit"] = numpy.where(is_moving[positions], error <= 45, numpy.nan)
            parts.append(part)
        scored = pandas.concat(parts)
        scored = scored[scored["step_hit"].notna()]
        for column in ("off_line", "line_to_go", "line_radius", "shape"):
            table = scored.groupby([column, "sheet"])["step_hit"].agg(["mean", "count"]).unstack("sheet")
            print(f"=== {name}: {column} (step within 45; count of moving pictures) ===")
            means = table["mean"].round(3)
            counts = table["count"].fillna(0).astype(int)
            combined = means.astype(str) + " (" + counts.astype(str) + ")"
            print(combined.to_string())
            print()


if __name__ == "__main__":
    main()
