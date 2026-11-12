"""
A stall rule that does not depend on the step length, tried under the same three calculations as stall_rule_grid.py:
    part 1 "shuffling"   over the pictures that hold the last PATH cm of the tip's own path (at least 15 pictures), the
                         tips stay within EXTENT cm and the forearm turns less than 15 degrees. With 1 cm steps and
                         PATH 15 this is the plan's rule (15 pictures, 4 cm), which fires in none of the recordings;
    part 2 "on the spot" the last SPOT pictures stay within 1.5 cm with less than 15 degrees of turn.
Calculated; not a robot run; no training.

Run:  D:/AI_Project/.venv/Scripts/python.exe stall_rule_by_path.py > stall_rule_by_path.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
from config import TRAINING_TARGET_PARAMETERS  # noqa: E402
from train_policy import merged_run_settings  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

STEP_LENGTH = ROOT / "06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/step_length"
RUN_ROOT = Path("D:/AI_Project/03_data/model_runs")
TURN_DEG = 15.0

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)


def first_stall(position, turn, path_cm, extent_cm, spot_pictures, spot_cm = 1.5, least_pictures = 15):
    """The first picture at which either part fires, and which part ("shuffling" or "on the spot")."""

    steps = numpy.r_[0.0, numpy.linalg.norm(numpy.diff(position, axis = 0), axis = 1)]
    travelled = numpy.cumsum(steps)
    for last in range(len(position)):
        # part 1: the window that holds the last path_cm of path
        first = int(numpy.searchsorted(travelled, travelled[last] - path_cm, side = "right")) - 1
        if first >= 0 and last - first + 1 >= least_pictures:
            window = position[first:last + 1]
            extent = numpy.hypot(window[:, 0].max() - window[:, 0].min(), window[:, 1].max() - window[:, 1].min())
            if extent < extent_cm and turn[first:last + 1].max() - turn[first:last + 1].min() < TURN_DEG:
                return last, "shuffling"
        if last >= spot_pictures - 1:
            window = position[last - spot_pictures + 1:last + 1]
            extent = numpy.hypot(window[:, 0].max() - window[:, 0].min(), window[:, 1].max() - window[:, 1].min())
            if extent < spot_cm and turn[last - spot_pictures + 1:last + 1].max() - turn[last - spot_pictures + 1:last + 1].min() < TURN_DEG:
                return last, "on the spot"
    return None, None


def chained(group, step_rule):
    heading = numpy.radians(group["picture_heading_deg"].to_numpy())
    forward, left = group["p_forward"].to_numpy(), group["p_left"].to_numpy()
    twist = numpy.clip(group["p_twist"].to_numpy(), -10.0, 10.0)
    vector = numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])
    length = numpy.linalg.norm(vector, axis = 1)
    unit = vector / numpy.maximum(length, 1e-9)[:, None]
    vector = unit * (numpy.minimum(length, 1.0)[:, None] if step_rule == "own" else step_rule)
    return numpy.vstack([numpy.zeros((1, 2)), numpy.cumsum(vector, axis = 0)])[:-1], numpy.r_[0.0, numpy.cumsum(twist)][:-1]


def paced_walk(group, step_rule, budget = 1500):
    tips = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    forearm = numpy.degrees(group["joint_4"].to_numpy(dtype = float))
    demonstrated = 100.0 * numpy.hypot(group["tip_forward_m"].fillna(0.0), group["tip_left_m"].fillna(0.0)).to_numpy()
    asked = numpy.minimum(numpy.hypot(group["p_forward"], group["p_left"]).to_numpy(), 1.0)
    count, place, index_axis = len(group), 0.0, numpy.arange(len(group))
    positions, turns = [], []
    while place < count - 1 and len(positions) < budget:
        positions.append([numpy.interp(place, index_axis, tips[:, 0]), numpy.interp(place, index_axis, tips[:, 1])])
        turns.append(numpy.interp(place, index_axis, forearm))
        nearest = int(round(place))
        step = asked[nearest] if step_rule == "own" else step_rule
        place += 1.0 if demonstrated[nearest] < 0.3 else max(step, 0.02) / demonstrated[nearest]
    return numpy.array(positions), numpy.array(turns)


def recorded(group):
    return 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float), numpy.degrees(group["joint_4"].to_numpy(dtype = float))


run_tracks = []
for folder in sorted(RUN_ROOT.glob("2026-11-11_17*")):
    steps = pandas.read_csv(folder / "steps.csv")
    moving = steps[steps["action"] != "stop"]
    run_tracks.append((100.0 * moving[["tip_x", "tip_y"]].to_numpy(), numpy.degrees(moving["joint_4"].to_numpy())))

for recipe in ("E_recipe", "A_recipe"):
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1], "p_twist": data["exact"][:, 2]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    table = table[table["action"] != "stop"]
    groups = [group for _, group in table.groupby("episode_id", sort = False)]
    tracks = {"recorded steps": [recorded(group) for group in groups]}
    for name, rule in (("own", "own"), ("0.65", 0.65), ("0.5", 0.5)):
        tracks[f"chained {name}"] = [chained(group, rule) for group in groups]
        tracks[f"paced walk {name}"] = [paced_walk(group, rule) for group in groups]
    print(f"================ {recipe}, {len(groups)} demonstrations")
    for path_cm, extent_cm, spot_pictures in ((15.0, 4.0, 40), (12.0, 4.0, 40), (15.0, 4.0, 30), (10.0, 4.0, 40)):
        print(f"   rule: last {path_cm:.0f} cm of path within {extent_cm:.0f} cm (at least 15 pictures), or last {spot_pictures} pictures within 1.5 cm; forearm under {TURN_DEG:.0f} degrees")
        for name, items in tracks.items():
            results = [first_stall(position, turn, path_cm, extent_cm, spot_pictures) for position, turn in items]
            shuffling = sum(1 for _, part in results if part == "shuffling")
            spot = sum(1 for _, part in results if part == "on the spot")
            print(f"      {name:18s}: ends {shuffling + spot:3d} of {len(items)} ({shuffling} shuffling, {spot} on the spot)")
        if recipe == "E_recipe":
            fired = {number + 1: first_stall(position, turn, path_cm, extent_cm, spot_pictures) for number, (position, turn) in enumerate(run_tracks)}
            print("      the ten runs as driven: " + (", ".join(f"run {number} at picture {where} ({part})" for number, (where, part) in fired.items() if where is not None) or "none"))
    print()
