"""
How centred the recorded demonstrations are: the sideways offset between the puck's centre and the line's middle at
every picture of a sample of episodes (the same measure as for the robot's runs in robot_runs.py).

The sample is drawn with a fixed seed: per sheet a few standard starts, one start with the puck put to the left, one
with it put to the right, and one other start, so that both the ordinary pushing and the demonstrated recoveries from an
off-centre puck are seen.

Run from this folder:    python demonstration_centring.py <sheet number>
Writes demonstration_centring_<sheet>.csv; summary.py reads all parts. Nothing connects to the robot and nothing is
written into 03_data.
"""

import sys
from pathlib import Path
from typing import List

import numpy
import pandas

import run_measures

from config import TRAINING_DATA_PARAMETERS
from puck_detector import PuckDetector


DEMONSTRATION_CENTRING_PARAMETERS = {
    # how many episodes of each group of start kinds are drawn per sheet
    "episodes_per_group": {"standard": 3, "puck_left": 1, "puck_right": 1, "other": 1},
    "start_kind_groups": {
        "standard": ["STD"],
        "puck_left": ["OL1", "OL1.5"],
        "puck_right": ["OR1", "OR1.5"],
        "other": ["TL10", "TL15", "TR10", "TR15", "MID", "NE", "X1", "X2", "SQP", "SQL"]
    },
    "seed": 11_11_26
}



def sampled_episodes(
        frames: pandas.DataFrame,
        sheet: str
) -> List[str]:
    """The episodes of one sheet drawn for the sample, in a fixed order."""

    episodes = frames[frames["sheet"] == sheet].groupby("episode_id").agg(start_kind = ("start_kind", "first")).reset_index()
    randomness_generator = numpy.random.default_rng(DEMONSTRATION_CENTRING_PARAMETERS["seed"])
    chosen = []
    for group, start_kinds in DEMONSTRATION_CENTRING_PARAMETERS["start_kind_groups"].items():
        candidates = sorted(episodes.loc[episodes["start_kind"].isin(start_kinds), "episode_id"])
        count = min(DEMONSTRATION_CENTRING_PARAMETERS["episodes_per_group"][group], len(candidates))
        if count:
            chosen.extend(randomness_generator.choice(candidates, size = count, replace = False).tolist())
    return chosen



def main(sheet_number: int) -> None:

    frames = pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", low_memory = False)
    sheet = f"path_{sheet_number:02d}"
    detector = PuckDetector()
    tables = []
    for episode_id in sampled_episodes(frames, sheet):
        steps = run_measures.demonstration_steps(frames, episode_id)
        period = run_measures.camera_period(steps.iloc[0]["started"])
        measured = run_measures.measure_steps(steps, run_measures.camera_parameters_for(period), detector = detector)
        rows = measured["rows"]
        for column in ("action", "move_length_m", "forearm_change_deg", "source", "waypoint", "is_mistake_row", "steps_to_end", "goal_distance_cm", "tip_x", "tip_y"):
            rows[column] = steps[column].to_numpy()[rows["step"].to_numpy()]
        rows.insert(0, "episode_id", episode_id)
        rows.insert(1, "sheet", sheet)
        rows.insert(2, "start_kind", steps.iloc[0]["start_kind"])
        rows.insert(3, "score", steps.iloc[0]["score"])
        rows.insert(4, "hand_corrections", steps.iloc[0]["hand_corrections"])
        rows.insert(5, "camera_period", period)
        tables.append(rows)
        offsets = rows["offset_mm"].dropna().abs()
        print(f"{episode_id.split('/')[-1]}: {len(rows)} pictures, offset measured in {len(offsets)}, "
              f"median size {offsets.median():.1f} mm, 90th percentile {offsets.quantile(0.9):.1f} mm, largest {offsets.max():.1f} mm", flush = True)
    table = pandas.concat(tables, ignore_index = True)
    out_path = run_measures.EVIDENCE_FOLDER / f"demonstration_centring_{sheet_number:02d}.csv"
    table.to_csv(out_path, index = False, float_format = "%.5f")
    print(f"written: {out_path.name} ({table['episode_id'].nunique()} episodes, {len(table)} pictures)")



if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
