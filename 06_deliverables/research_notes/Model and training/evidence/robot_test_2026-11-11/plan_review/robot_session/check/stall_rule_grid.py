"""
Check of the critic's fix for P5 ("25 pictures, 4 cm, 15 degrees for every setting"): a grid of stall rules under three
calculations, to see which rule ends few healthy walks at short steps and still ends run 6 of 11.11.26.
    A  the critic's method: the fold models' answers chained picture by picture along each demonstration;
    B  the paced walk of stall_rule_paced_walk.py: more pictures per centimetre where the model asks short steps;
       also split by how slow the ended window was: "creeping" = the walker still made 1.5 cm or more of path in the
       window (slow but moving on), "stuck" = less than that (a real hesitation that a stall rule should end);
    C  the ten logged runs as driven.
Calculated; not a robot run; no training.

Run:  D:/AI_Project/.venv/Scripts/python.exe stall_rule_grid.py > stall_rule_grid.out.txt
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
RULES = [(15, 4.0), (20, 4.0), (25, 4.0), (30, 4.0), (40, 4.0), (25, 3.0), (30, 3.0), (40, 3.0), (30, 2.0), (40, 2.0)]
TURN_DEG = 15.0

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)


def first_stall(position, turn, pictures, extent_cm):
    for last in range(pictures - 1, len(position)):
        window = position[last - pictures + 1:last + 1]
        extent = numpy.hypot(window[:, 0].max() - window[:, 0].min(), window[:, 1].max() - window[:, 1].min())
        turned = turn[last - pictures + 1:last + 1].max() - turn[last - pictures + 1:last + 1].min()
        if extent < extent_cm and turned < TURN_DEG:
            return last
    return None


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
    positions, turns, steps = [], [], []
    while place < count - 1 and len(positions) < budget:
        positions.append([numpy.interp(place, index_axis, tips[:, 0]), numpy.interp(place, index_axis, tips[:, 1])])
        turns.append(numpy.interp(place, index_axis, forearm))
        nearest = int(round(place))
        step = asked[nearest] if step_rule == "own" else step_rule
        steps.append(step)
        place += 1.0 if demonstrated[nearest] < 0.3 else max(step, 0.02) / demonstrated[nearest]
    return numpy.array(positions), numpy.array(turns), numpy.array(steps)


data = numpy.load(STEP_LENGTH / "fold_outputs_E_recipe.npz")
predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1], "p_twist": data["exact"][:, 2]})
table = cleaned.merge(predicted, on = "frame_index", how = "left")
table = table[table["action"] != "stop"]
groups = [group for _, group in table.groupby("episode_id", sort = False)]
count = len(groups)
chains = {name: [chained(group, rule) for group in groups] for name, rule in (("own", "own"), ("0.65", 0.65), ("0.5", 0.5))}
walks = {name: [paced_walk(group, rule) for group in groups] for name, rule in (("own", "own"), ("0.65", 0.65), ("0.5", 0.5))}

run_tracks = []
for folder in sorted(RUN_ROOT.glob("2026-11-11_17*")):
    steps = pandas.read_csv(folder / "steps.csv")
    moving = steps[steps["action"] != "stop"]
    run_tracks.append((100.0 * moving[["tip_x", "tip_y"]].to_numpy(), numpy.degrees(moving["joint_4"].to_numpy()), len(steps)))

print(f"E recipe, {count} demonstrations, each answered by the model that never saw its sheet; forearm limit {TURN_DEG:.0f} degrees in every rule")
print("rule (pictures, cm) | A chained: own, 0.65, 0.5 | B paced walk: own (creeping + stuck), 0.65 (creeping + stuck), 0.5 (creeping + stuck) | C the ten runs as driven: run -> picture")
for pictures, extent_cm in RULES:
    chained_counts = [sum(first_stall(position, turn, pictures, extent_cm) is not None for position, turn in chains[name]) for name in ("own", "0.65", "0.5")]
    walk_text = []
    for name in ("own", "0.65", "0.5"):
        creeping, stuck = 0, 0
        for position, turn, step in walks[name]:
            where = first_stall(position, turn, pictures, extent_cm)
            if where is None:
                continue
            window = position[where - pictures + 1:where + 1]
            path = float(numpy.linalg.norm(numpy.diff(window, axis = 0), axis = 1).sum())
            if path >= 1.5:
                creeping += 1
            else:
                stuck += 1
        walk_text.append(f"{creeping + stuck} ({creeping} + {stuck})")
    fired = {number + 1: first_stall(position, turn, pictures, extent_cm) for number, (position, turn, _) in enumerate(run_tracks)}
    fired_text = ", ".join(f"run {number} -> {where}" for number, where in fired.items() if where is not None) or "none"
    print(f"   {pictures:2d}, {extent_cm:.0f} | {chained_counts[0]:3d}, {chained_counts[1]:3d}, {chained_counts[2]:3d} | {walk_text[0]:>14s}, {walk_text[1]:>12s}, {walk_text[2]:>12s} | {fired_text}")
