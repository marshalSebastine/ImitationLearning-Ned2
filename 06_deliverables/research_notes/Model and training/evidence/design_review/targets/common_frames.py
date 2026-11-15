"""
Shared helpers for the targets section: read frames.csv read-only and apply the audit's cleaning the way
training_targets.audit_cleaned_frames does (re-implemented here so that no project module is imported).
"""

import json
from pathlib import Path

import numpy
import pandas

PROCESSED_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
AUDIT_FOLDER = Path("C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/audit")
OUTPUT_FOLDER = Path(__file__).resolve().parent
DIRECTION_NAMES = ["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"]


def read_frames() -> pandas.DataFrame:
    """Return frames.csv sorted by episode and row."""

    frames = pandas.read_csv(PROCESSED_FOLDER / "frames.csv", low_memory = False)
    return frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)


def audit_cleaned(frames: pandas.DataFrame, is_clipped: bool = True) -> pandas.DataFrame:
    """Apply the audit cleaning (drops, stop relabelling, clips) and the score and mistake rules of the final candidate."""

    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True).copy()
    for episode_id in frames.loc[frames["is_scored"] == 0, "episode_id"].unique():
        folder = frames.loc[frames["episode_id"] == episode_id, "episode_folder"].iloc[0]
        review_path = RAW_FOLDER / folder / "review.json"
        if review_path.exists():
            score = json.loads(review_path.read_text(encoding = "utf-8")).get("score")
            if score is not None:
                frames.loc[frames["episode_id"] == episode_id, ["score", "is_scored"]] = [score, 1]
    joint_columns = [f"joint_{number}" for number in range(1, 7)]
    grouped = frames.groupby("episode_id", sort = False)
    largest_change_deg = numpy.degrees((grouped[joint_columns].shift(-1) - frames[joint_columns]).abs().max(axis = 1, skipna = False))
    is_before_stop = grouped["action"].shift(-1) == "stop"
    is_relabelled = (largest_change_deg < 0.2) & is_before_stop
    flags = frames["picture_flags"].fillna("")
    is_dropped = (frames["row_in_episode"] == 0) | ((frames["waypoint"].fillna(-1) == 0) & (frames["row_in_episode"] > 0))
    is_dropped |= flags.str.contains("sudden change")
    is_dropped |= frames["episode_folder"].str.contains("_plan_05-16_", regex = False)
    is_dropped |= frames["episode_folder"].str.contains("_plan_04-01_try_6", regex = False) & frames["row_in_episode"].between(169, 175)
    frames["is_relabelled_stop"] = (is_relabelled & ~is_dropped).astype(int)
    frames.loc[frames["is_relabelled_stop"] == 1, "action"] = "stop"
    frames = frames[~is_dropped].copy()
    if is_clipped:
        length_m = numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
        scale = numpy.minimum(1.0, 0.02 / numpy.maximum(length_m, 1e-9))
        frames["tip_forward_m"] = frames["tip_forward_m"] * scale
        frames["tip_left_m"] = frames["tip_left_m"] * scale
        frames["forearm_change_deg"] = frames["forearm_change_deg"].clip(-15.0, 15.0)
    frames["rows_before_scores"] = len(frames)
    frames = frames[frames["score"].isin([1, 2, 3]) & (frames["is_mistake_row"] == 0)].copy()
    return frames.reset_index(drop = True)
