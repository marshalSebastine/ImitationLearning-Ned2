"""
Check of the critic's P5 and its fix with a second, different calculation. The critic chained the fold models' answers
along each demonstration picture by picture (one answer per demonstration picture). On the robot a short step means
MORE pictures per centimetre, so a slow place lasts longer. Here each demonstration is walked at the model's own pace:

    the walker stands at a (fractional) picture of the demonstration; it takes the step length the fold model (which
    never saw that sheet) asked at the nearest picture, at most 1.0 cm, or a fixed length; it moves on along the
    demonstration by that length divided by the demonstrated step length there (where the demonstration itself moved
    less than 0.3 cm, a twist in place, the walker moves on one picture: the model is assumed to twist as shown);
    its tip and its forearm angle are the demonstration's at that place (so it follows the line perfectly).

The stall rules are then applied to the walker's tips and forearm angles, as model_mode.ModelRun.is_stalled does.
A perfect follower that is ended by the rule is a healthy run cut short. Calculated; not a robot run; no training.

Run:  D:/AI_Project/.venv/Scripts/python.exe stall_rule_paced_walk.py > stall_rule_paced_walk.out.txt
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
RULES = {"plan: 15 pictures, 4 cm, 15 deg": (15, 4.0, 15.0), "critic: 25 pictures, 4 cm, 15 deg": (25, 4.0, 15.0),
         "today: 30 pictures, 3 cm, 15 deg": (30, 3.0, 15.0), "40 pictures, 4 cm, 15 deg": (40, 4.0, 15.0)}
STEP_RULES = (("own length", "own"), ("fixed 0.65 cm", 0.65), ("fixed 1.0 cm", 1.0))

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
print("columns used:", [name for name in ("tip_x_m", "tip_y_m", "joint_4", "tip_forward_m", "tip_left_m", "action", "sheet", "start_kind") if name in cleaned.columns])


def first_stall(position, turn, pictures, extent_cm, turn_deg):
    for last in range(pictures - 1, len(position)):
        window = position[last - pictures + 1:last + 1]
        extent = numpy.hypot(window[:, 0].max() - window[:, 0].min(), window[:, 1].max() - window[:, 1].min())
        turned = turn[last - pictures + 1:last + 1].max() - turn[last - pictures + 1:last + 1].min()
        if extent < extent_cm and turned < turn_deg:
            return last
    return None


def paced_walk(group, step_rule, budget = 1500):
    tips = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    forearm = numpy.degrees(group["joint_4"].to_numpy(dtype = float))
    demonstrated = 100.0 * numpy.hypot(group["tip_forward_m"].fillna(0.0), group["tip_left_m"].fillna(0.0)).to_numpy()
    asked = numpy.minimum(numpy.hypot(group["p_forward"], group["p_left"]).to_numpy(), 1.0)
    count = len(group)
    place = 0.0
    positions, turns = [], []
    index_axis = numpy.arange(count)
    while place < count - 1 and len(positions) < budget:
        positions.append([numpy.interp(place, index_axis, tips[:, 0]), numpy.interp(place, index_axis, tips[:, 1])])
        turns.append(numpy.interp(place, index_axis, forearm))
        nearest = int(round(place))
        step = asked[nearest] if step_rule == "own" else step_rule
        if demonstrated[nearest] < 0.3:
            place += 1.0
        else:
            place += max(step, 0.02) / demonstrated[nearest]
    return numpy.array(positions), numpy.array(turns)


for recipe in ("E_recipe", "A_recipe"):
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    table = table[table["action"] != "stop"]
    count = table["episode_id"].nunique()
    print(f"\n================ {recipe}: {count} demonstrations walked at the pace of the model that never saw their sheet")
    for step_name, step_rule in STEP_RULES:
        walks = [(group["sheet"].iloc[0][-2:], len(group), *paced_walk(group, step_rule)) for _, group in table.groupby("episode_id", sort = False)]
        pictures = numpy.array([len(walk[2]) for walk in walks])
        recorded = numpy.array([walk[1] for walk in walks])
        print(f"   {step_name}: the walker needs a median {numpy.median(pictures / recorded):.2f} times the demonstration's pictures (largest run {pictures.max()} pictures)")
        for rule_name, (window, extent_cm, turn_deg) in RULES.items():
            fired = [(sheet, where, len(positions)) for sheet, _, positions, turns in walks
                     for where in [first_stall(positions, turns, window, extent_cm, turn_deg)] if where is not None]
            early = sum(1 for _, where, total in fired if where < 0.9 * total)
            per_sheet = {sheet: sum(1 for item in fired if item[0] == sheet) for sheet in sorted({walk[0] for walk in walks})}
            print(f"      {rule_name:34s}: ends {len(fired):3d} of {count} ({100 * len(fired) / count:4.1f} %), {early:3d} before the last tenth; per sheet {per_sheet}")
