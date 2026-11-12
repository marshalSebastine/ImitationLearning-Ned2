"""
The verifier's own count of how rare run 3's start state is in the demonstrations: tip within 22 cm of the base axis,
forearm nearly straight (|joint 4| of 15 degrees or less) and the path over the next three rows 45 degrees or more
off the picture's top (the block pushing sideways in the picture, near the base, with a straight forearm).

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v14_start_state_rarity.py > v14_start_state_rarity.out.txt
"""

import sys
from pathlib import Path

import numpy

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from training_targets import load_training_frames  # noqa: E402


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


frames = load_training_frames({"is_audit_cleaning": True}).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
frames["start_kind"] = frames["start_kind"].fillna("STD")
grouped = frames.groupby("episode_id", sort = False)
x = 100.0 * frames["tip_x_m"].to_numpy()
y = 100.0 * frames["tip_y_m"].to_numpy()
x_ahead = 100.0 * grouped["tip_x_m"].shift(-3).to_numpy()
y_ahead = 100.0 * grouped["tip_y_m"].shift(-3).to_numpy()
is_path = ((grouped["row_in_episode"].shift(-3).to_numpy() - frames["row_in_episode"].to_numpy()) == 3) & (numpy.hypot(x_ahead - x, y_ahead - y) >= 0.9)
in_picture = wrap(numpy.degrees(numpy.arctan2(y_ahead - y, x_ahead - x)) - frames["picture_heading_deg"].to_numpy())
reach = numpy.hypot(x, y)
joint_4 = numpy.degrees(frames["joint_4"].to_numpy())
near = reach <= 22.0
straight = numpy.abs(joint_4) <= 15.0
print(f"pictures within 22 cm of the base axis: {near.sum()}; of these with |joint 4| of 15 or less: {(near & straight).sum()}")
for limit in (30.0, 45.0, 60.0):
    state = near & straight & is_path & (numpy.abs(in_picture) >= limit)
    kinds = frames.loc[state, "start_kind"].value_counts().to_dict()
    print(f"  ... and the path {limit:.0f} degrees or more off the picture's top: {state.sum()} pictures in {frames.loc[state, 'episode_id'].nunique()} episodes; by start kind: {kinds}")
state = near & straight & is_path & (numpy.abs(in_picture) >= 45.0)
order = grouped.cumcount().to_numpy()
print(f"  of the 45-degree group, within the first 8 kept rows of their episode: {(state & (order < 8)).sum()}; these are mostly the free approach to a puck laid beside the line or on a square sheet")
