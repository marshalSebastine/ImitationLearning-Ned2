"""
Robot-session critic: how often would the plan's stall rule (the last 15 pictures within a 4 cm box diagonal AND less
than 15 degrees of forearm turn) end a healthy run when the driver moves SHORT steps?

The plan's "fires in none of the 210 recordings" was counted on the recorded steps (median 1.05 cm). Blocks A and B
drive with the model's own length or a fixed 0.65 cm. Here the fold models' answers on the demonstrations' pictures
(each sheet answered by the model that never saw it; step_length/fold_outputs_*.npz) are chained along each
demonstration (teacher forced, as the verifier's v16 did for another guard) under three step rules, and the rule of
model_mode.ModelRun.is_stalled is applied with the plan's numbers and with today's (30 pictures, 3 cm, 15 degrees).

Not a closed loop: on the robot the pictures follow the model's own moves. No training, nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe stall_rule_with_short_steps.py > stall_rule_with_short_steps.out.txt
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
STALL_RULES = {"the plan's rule (15 pictures, 4 cm, 15 degrees)": (15, 4.0, 15.0), "today's rule (30 pictures, 3 cm, 15 degrees)": (30, 3.0, 15.0)}
STEP_RULES = ["own length, at most 1.0 cm", "fixed 0.65 cm", "fixed 1.0 cm", "recorded steps"]

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)


def first_stall(position: numpy.ndarray, turn: numpy.ndarray, pictures: int, extent_cm: float, turn_deg: float):
    """The index of the first picture at which the rule fires (None when never), as ModelRun.is_stalled counts."""

    for last in range(pictures - 1, len(position)):
        window = position[last - pictures + 1:last + 1]
        extent = numpy.hypot(window[:, 0].max() - window[:, 0].min(), window[:, 1].max() - window[:, 1].min())
        turned = turn[last - pictures + 1:last + 1].max() - turn[last - pictures + 1:last + 1].min()
        if extent < extent_cm and turned < turn_deg:
            return last
    return None


for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1],
                                  "p_twist": data["exact"][:, 2], "p_stop": data["stop_probability"]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    table = table[table["action"] != "stop"]
    print(f"================ {recipe}: fold models on the sheets they never saw, {table['episode_id'].nunique()} demonstrations")
    for step_rule in STEP_RULES:
        for rule_name, (pictures, extent_cm, turn_deg) in STALL_RULES.items():
            fired, fired_early, per_sheet = 0, 0, {}
            for episode_id, group in table.groupby("episode_id", sort = False):
                heading = numpy.radians(group["picture_heading_deg"].to_numpy())
                if step_rule == "recorded steps":
                    forward, left = 100.0 * group["tip_forward_m"].fillna(0.0).to_numpy(), 100.0 * group["tip_left_m"].fillna(0.0).to_numpy()
                    twist = group["forearm_change_deg"].fillna(0.0).to_numpy()
                else:
                    forward, left = group["p_forward"].to_numpy(), group["p_left"].to_numpy()
                    twist = numpy.clip(group["p_twist"].to_numpy(), -10.0, 10.0)
                vector = numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])
                length = numpy.linalg.norm(vector, axis = 1)
                unit = vector / numpy.maximum(length, 1e-9)[:, None]
                if step_rule == "own length, at most 1.0 cm":
                    vector = unit * numpy.minimum(length, 1.0)[:, None]
                elif step_rule.startswith("fixed"):
                    vector = unit * float(step_rule.split()[1])
                # the tip before each picture's move, as the driver logs it
                position = numpy.vstack([numpy.zeros((1, 2)), numpy.cumsum(vector, axis = 0)])[:-1]
                turn = numpy.r_[0.0, numpy.cumsum(twist)][:-1]
                where = first_stall(position, turn, pictures, extent_cm, turn_deg)
                sheet = group["sheet"].iloc[0]
                per_sheet.setdefault(sheet, [0, 0])
                per_sheet[sheet][1] += 1
                if where is not None:
                    fired += 1
                    per_sheet[sheet][0] += 1
                    fired_early += int(where < 0.9 * len(group))
            count = table["episode_id"].nunique()
            sheets = ", ".join(f"{sheet[-2:]}: {values[0]}/{values[1]}" for sheet, values in sorted(per_sheet.items()))
            print(f"   {step_rule:28s} | {rule_name:46s} | ends {fired:3d} of {count} demonstrations ({100 * fired / count:4.1f} %), "
                  f"{fired_early:3d} before their last tenth | per sheet {sheets}")
    print()
