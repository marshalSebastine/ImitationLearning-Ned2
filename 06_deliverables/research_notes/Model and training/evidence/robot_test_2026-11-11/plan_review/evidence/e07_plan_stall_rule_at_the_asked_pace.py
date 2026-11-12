"""
Plan review, key "evidence": the plan (step 1, point 2) wants a stall rule that ends a run whose last 15 pictures stay
within 4 cm (diagonal of the box round the tips, as model_mode.is_stalled measures it) with less than 15 degrees of
forearm turn, and says it "fires in none of the 210 recordings [calculated]". That count treats the RECORDED tip
positions as a run (bends_and_two_ways/verify/v08). The rule is meant for runs at the model's own pace, so this script
repeats the count with the fold models' own steps chained along each held-out demonstration (teacher forced, as
step_length/verify/v16 did for another guard), for the three settings of the plan's block A:
fixed 1.0 cm, the model's own length (at most 1.0 cm, as step_most_m in config.py), fixed 0.65 cm.

Not a closed loop: on the robot the pictures follow the model's own moves. Nothing is trained, nothing connects to the
robot, nothing outside this folder is written.
Run:  D:/AI_Project/.venv/Scripts/python.exe e07_plan_stall_rule_at_the_asked_pace.py > e07_plan_stall_rule_at_the_asked_pace.out.txt
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
pandas.set_option("display.width", 250)
WINDOW, EXTENT_CM, TURN_DEG = 15, 4.0, 15.0


def stalled_windows(position_cm: numpy.ndarray, forearm_deg: numpy.ndarray) -> numpy.ndarray:
    """One flag per window of WINDOW pictures in a row: box diagonal under EXTENT_CM and forearm range under TURN_DEG."""

    count = len(position_cm) - WINDOW + 1
    flags = numpy.zeros(max(count, 0), dtype = bool)
    for index in range(max(count, 0)):
        part = position_cm[index:index + WINDOW]
        extent = float(numpy.hypot(part[:, 0].max() - part[:, 0].min(), part[:, 1].max() - part[:, 1].min()))
        turned = float(forearm_deg[index:index + WINDOW].max() - forearm_deg[index:index + WINDOW].min())
        flags[index] = extent < EXTENT_CM and turned < TURN_DEG
    return flags


target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
print("cleaned rows:", len(cleaned), "episodes:", cleaned["episode_id"].nunique())
print("columns used:", [name for name in ("tip_x_m", "tip_y_m", "joint_4", "picture_heading_deg", "action", "sheet") if name in cleaned.columns])

# 0. the recorded positions themselves (the plan's count), on the cleaned rows
fired = 0
for episode_id, group in cleaned.groupby("episode_id", sort = False):
    position = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
    joint_4 = group["joint_4"].to_numpy()
    forearm = numpy.degrees(joint_4) if numpy.abs(joint_4).max() < 7 else joint_4
    fired += int(stalled_windows(position, forearm).any())
print(f"0. recorded tip positions treated as a run: the rule fires in {fired} of {cleaned['episode_id'].nunique()} demonstrations")

for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1], "p_twist": data["exact"][:, 2],
                                  "p_stop": data["stop_probability"]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    print()
    print(f"================ {recipe} (fold models, each demonstration seen as an unseen sheet): rows without a prediction {int(table['p_forward'].isna().sum())}")
    for setting, fixed_cm in (("fixed 1.0 cm", 1.0), ("the model's own length, at most 1.0 cm", None), ("fixed 0.65 cm", 0.65)):
        rows = []
        for episode_id, group in table.groupby("episode_id", sort = False):
            group = group[group["p_forward"].notna()]
            stop = group["p_stop"].to_numpy()
            two_in_a_row = numpy.flatnonzero((stop[1:] >= 0.5) & (stop[:-1] >= 0.5))
            run_length = int(two_in_a_row[0] + 1) if len(two_in_a_row) else len(group)   # decisions made before the stop rule ends the run
            group = group.iloc[:run_length]
            if len(group) < WINDOW:
                rows.append({"episode_id": episode_id, "sheet": group["sheet"].iloc[0] if len(group) else "", "decisions": len(group), "any": False, "before_last_tenth": False, "first": None})
                continue
            heading = numpy.radians(group["picture_heading_deg"].to_numpy())
            forward, left = group["p_forward"].to_numpy(), group["p_left"].to_numpy()
            vector = numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])
            length = numpy.linalg.norm(vector, axis = 1)
            wanted = numpy.minimum(length, 1.0) if fixed_cm is None else numpy.full(len(length), fixed_cm)
            vector = vector * (wanted / numpy.maximum(length, 1e-9))[:, None]
            twist = numpy.clip(group["p_twist"].to_numpy(), -10.0, 10.0)
            position = numpy.vstack([numpy.zeros((1, 2)), numpy.cumsum(vector, axis = 0)])[:-1]   # the tip at each picture
            forearm = numpy.r_[0.0, numpy.cumsum(twist)][:-1]
            flags = stalled_windows(position, forearm)
            last_tenth = int(0.9 * len(flags))
            first = int(numpy.flatnonzero(flags)[0] + WINDOW - 1) if flags.any() else None
            rows.append({"episode_id": episode_id, "sheet": group["sheet"].iloc[0], "decisions": len(group), "any": bool(flags.any()),
                         "before_last_tenth": bool(flags[:last_tenth].any()), "first": first})
        episodes = pandas.DataFrame(rows)
        print(f"  {setting:40s}: the rule fires in {int(episodes['any'].sum()):3d} of {len(episodes)} demonstrations ({100 * episodes['any'].mean():4.1f} %), "
              f"in {int(episodes['before_last_tenth'].sum()):3d} before the last tenth of the run ({100 * episodes['before_last_tenth'].mean():4.1f} %)")
        by_sheet = episodes.groupby("sheet").agg(demonstrations = ("any", "size"), fires = ("any", "sum"), fires_before_the_last_tenth = ("before_last_tenth", "sum"))
        print("      per sheet (demonstrations / fires / fires before the last tenth): " + "; ".join(
            f"{sheet[-2:]}: {int(row['demonstrations'])} / {int(row['fires'])} / {int(row['fires_before_the_last_tenth'])}" for sheet, row in by_sheet.iterrows()))
