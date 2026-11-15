"""
Robot time per kept demonstration by start kind (read only), from episode_timing_episodes.csv (this folder) joined
with the index's episodes.csv (start kind). Near-end (NE) starts are the cheapest source of stop pictures.

Usage: python timing_by_start_kind.py
"""

from pathlib import Path

import numpy
import pandas

PARAMETERS = {
    "timing": Path(__file__).resolve().parent / "episode_timing_episodes.csv",
    "index_episodes": Path("D:/AI_Project/03_data/processed/v01_training_index/episodes.csv"),
}


def main(parameters = None):
    """
    Print medians per start kind.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    timing = pandas.read_csv(settings["timing"])
    index = pandas.read_csv(settings["index_episodes"], usecols = ["episode_folder", "start_kind", "sheet"])
    index["folder"] = index["episode_folder"]
    merged = timing.merge(index[["folder", "start_kind"]], on = "folder", how = "left")
    kept = merged[merged["is_kept"]]
    kept = kept.assign(start_kind = kept["start_kind"].fillna("STD (01-01, 01-02)"))
    summary = kept.groupby("start_kind").agg(
        episodes = ("folder", "count"),
        rows_median = ("rows", "median"),
        duration_median_min = ("duration_min", "median"),
        cycle_median_min = ("cycle_min", "median"),
        corrections_mean = ("corrections_found", "mean"))
    print(summary.round(2).to_string())
    print()
    near_end = kept[kept["start_kind"] == "NE"]
    print("NE episodes:", len(near_end), "; rows median", near_end["rows"].median(), "; start to next start median",
          round(float(near_end["cycle_min"].median()), 2), "min; start to finish median",
          round(float(near_end["duration_min"].median()), 2), "min")
    print("NE by sheet:", near_end.groupby("sheet")["folder"].count().to_dict())


if __name__ == "__main__":
    main()
