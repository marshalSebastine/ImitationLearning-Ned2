"""
Robot-session critic: would our own demonstrations pass the plan's block A rule ("the puck stays within 12 mm of the
line's middle to 10 cm past the bend")? The six sheet-03 demonstrations the check measured picture by picture
(stop_and_centring/demonstration_centring_03.csv, the same measure as for the robot runs) are scored the same way as
the three robot runs in block_a_rule_on_the_three_runs.py. A rule that a careful human demonstration fails cannot be
asked of a model that copies it. Reads one CSV and the index. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe block_a_rule_on_demonstrations.py > block_a_rule_on_demonstrations.out.txt
"""

from pathlib import Path

import numpy
import pandas

CENTRING = Path(__file__).resolve().parents[2] / "stop_and_centring" / "demonstration_centring_03.csv"
EPISODES = Path("D:/AI_Project/03_data/processed/v01_training_index/episodes.csv")

table = pandas.read_csv(CENTRING)
episodes = pandas.read_csv(EPISODES, low_memory = False)
end_column = next((name for name in episodes.columns if name in ("start_end", "start_end_corrected")), None)
print(f"columns of episodes.csv that name the start end: {[name for name in episodes.columns if 'end' in name]}")
for episode_id, rows in table.groupby("episode_id", sort = False):
    rows = rows.sort_values("step").reset_index(drop = True)
    puck = rows[["puck_x", "puck_y"]].to_numpy() * 100.0
    is_found = numpy.isfinite(puck[:, 0])
    if is_found.sum() < 20:
        continue
    filled = pandas.DataFrame(puck).ffill().bfill().to_numpy()
    path = numpy.r_[0.0, numpy.cumsum(numpy.hypot(numpy.diff(filled[:, 0]), numpy.diff(filled[:, 1])))]
    if path[-1] < 14.0:
        print(f"{episode_id.split('/')[-1]} ({rows['start_kind'].iloc[0]}): the puck travelled only {path[-1]:.1f} cm; left out")
        continue
    first = filled[numpy.searchsorted(path, 2.0)] - filled[0]
    first = first / numpy.linalg.norm(first)
    bend = None
    for index in range(3, len(filled)):
        base = int(numpy.searchsorted(path, max(0.0, path[index] - 1.5)))
        move = filled[index] - filled[base]
        if numpy.linalg.norm(move) > 0.8 and numpy.degrees(numpy.arccos(numpy.clip(move @ first / numpy.linalg.norm(move), -1.0, 1.0))) > 45.0:
            bend = index
            break
    if bend is None:
        print(f"{episode_id.split('/')[-1]}: no bend found; left out")
        continue
    end = int(numpy.searchsorted(path, path[bend] + 10.0, side = "right"))
    window = rows.iloc[:end]
    offset = window["offset_mm"].abs()
    is_sound = window["is_clean"].astype(bool) & window["is_aligned"].astype(bool) & (window["extrapolated_mm"] < 5.0)
    is_pushed = numpy.r_[False, numpy.hypot(numpy.diff(filled[:end, 0]), numpy.diff(filled[:end, 1])) > 0.2]
    info = episodes[episodes["episode_id"] == episode_id]
    start_end = info[end_column].iloc[0] if end_column and len(info) else "?"
    print(f"{episode_id.split('/')[-1]} (start {rows['start_kind'].iloc[0]}, end {start_end}, {int(rows['hand_corrections'].iloc[0])} hand corrections, score {rows['score'].iloc[0]:.0f}): "
          f"bend at picture {bend} (puck path {path[bend]:.1f} cm), window to picture {end - 1} of {len(rows) - 1}")
    sound = offset[is_sound]
    print(f"   sound readings in the window {len(sound)} of {len(window)}: over 7 mm {int((sound > 7).sum())}, over 12 mm {int((sound > 12).sum())}, over 17 mm {int((sound > 17).sum())}; "
          f"largest {sound.max():.1f} mm; while the puck was moving: over 12 mm {int(((offset > 12) & is_sound & is_pushed).sum())}")
    for label, limit, in_a_row in (("12 mm, any sound picture", 12.0, 1), ("12 mm, two sound pictures in a row", 12.0, 2), ("17 mm, two sound pictures in a row", 17.0, 2)):
        over = ((offset > limit) & is_sound).to_numpy()
        fails = [int(window["step"].iloc[index]) for index in range(in_a_row - 1, len(over)) if all(over[index - in_a_row + 1:index + 1])]
        print(f"      {label:36s}: {'PASS' if not fails else 'FAIL at picture ' + str(fails[0])}")
