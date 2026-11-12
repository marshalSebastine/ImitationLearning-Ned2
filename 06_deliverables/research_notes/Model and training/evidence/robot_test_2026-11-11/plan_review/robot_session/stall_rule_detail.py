"""
Robot-session critic, follow-up to stall_rule_with_short_steps.py: where in a demonstration does the plan's stall rule
(15 pictures, 4 cm, 15 degrees) fire when the steps are the model's own length, and do other forms of the rule keep
healthy demonstrations alive? Same method (fold models on the sheet they never saw, chained along each demonstration,
teacher forced; not a closed loop). No training, nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe stall_rule_detail.py > stall_rule_detail.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
from config import TRAINING_TARGET_PARAMETERS
from train_policy import merged_run_settings
from training_targets import load_training_frames

STEP_LENGTH = ROOT / "06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/step_length"
RULES = {"plan: 15 pictures, 4 cm, 15 degrees": (15, 4.0, 15.0),
         "today: 30 pictures, 3 cm, 15 degrees": (30, 3.0, 15.0),
         "20 pictures, 3 cm, 15 degrees": (20, 3.0, 15.0),
         "25 pictures, 4 cm, 15 degrees": (25, 4.0, 15.0),
         "15 pictures, 2 cm, 15 degrees": (15, 2.0, 15.0)}

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)


def first_stall(position, turn, pictures, extent_cm, turn_deg):
    for last in range(pictures - 1, len(position)):
        window = position[last - pictures + 1:last + 1]
        extent = numpy.hypot(window[:, 0].max() - window[:, 0].min(), window[:, 1].max() - window[:, 1].min())
        turned = turn[last - pictures + 1:last + 1].max() - turn[last - pictures + 1:last + 1].min()
        if extent < extent_cm and turned < turn_deg:
            return last
    return None


def chained(group, step_rule):
    heading = numpy.radians(group["picture_heading_deg"].to_numpy())
    forward, left = group["p_forward"].to_numpy(), group["p_left"].to_numpy()
    twist = numpy.clip(group["p_twist"].to_numpy(), -10.0, 10.0)
    vector = numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])
    length = numpy.linalg.norm(vector, axis = 1)
    unit = vector / numpy.maximum(length, 1e-9)[:, None]
    if step_rule == "own":
        vector = unit * numpy.minimum(length, 1.0)[:, None]
    else:
        vector = unit * step_rule
    position = numpy.vstack([numpy.zeros((1, 2)), numpy.cumsum(vector, axis = 0)])[:-1]
    turn = numpy.r_[0.0, numpy.cumsum(twist)][:-1]
    return position, turn, length


for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1], "p_twist": data["exact"][:, 2]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    table = table[table["action"] != "stop"]
    count = table["episode_id"].nunique()
    print(f"================ {recipe}: {count} demonstrations, each answered by the model that never saw its sheet")
    for rule_name, (pictures, extent_cm, turn_deg) in RULES.items():
        for step_name, step_rule in (("own length", "own"), ("fixed 0.65 cm", 0.65), ("fixed 0.5 cm", 0.5)):
            fired = []
            for episode_id, group in table.groupby("episode_id", sort = False):
                position, turn, length = chained(group, step_rule)
                where = first_stall(position, turn, pictures, extent_cm, turn_deg)
                if where is not None:
                    fired.append((group["sheet"].iloc[0][-2:], group["start_kind"].iloc[0], where, len(group), float(numpy.median(length[max(0, where - pictures + 1):where + 1]))))
            early = sum(1 for item in fired if item[2] < 0.9 * item[3])
            print(f"   {rule_name:38s} | {step_name:14s} | ends {len(fired):3d} of {count} ({100 * len(fired) / count:4.1f} %), {early:3d} before the last tenth")
            if rule_name.startswith("plan") and step_name == "own length":
                for sheet, kind, where, total, median_length in sorted(fired):
                    print(f"         sheet {sheet} start {kind:6s}: fires at picture {where:3d} of {total:3d}; median asked length in that window {median_length:.2f} cm")
    # the asked length itself: how often are 15 pictures in a row short enough to stay within 4 cm even on a straight path?
    lengths = numpy.minimum(numpy.hypot(table["p_forward"], table["p_left"]).to_numpy(), 1.0)
    print(f"   asked length on unseen sheets: median {numpy.median(lengths):.2f} cm, share under 0.27 cm (15 such steps on a straight path stay within 4 cm): {100 * numpy.mean(lengths < 0.27):.1f} %")
    print()
