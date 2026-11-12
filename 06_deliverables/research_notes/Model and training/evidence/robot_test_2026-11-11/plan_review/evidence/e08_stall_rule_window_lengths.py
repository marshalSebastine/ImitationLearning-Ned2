"""
Plan review, key "evidence": follow-up to e07. The same stall rule (box diagonal under 4 cm, forearm range under
15 degrees) with windows of 15, 20, 25 and 30 pictures: (a) when it would have ended run 6 as driven on 11.11.26 and
whether it fires in any other of the ten runs, (b) how often it fires at the fold models' own asked pace along the
held-out demonstrations (teacher forced, own length at most 1.0 cm), before the last tenth of the run.
Reads logs and saved predictions only.
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
MODEL_RUNS = Path("D:/AI_Project/03_data/model_runs")
EXTENT_CM, TURN_DEG = 4.0, 15.0


def first_fire(position_cm: numpy.ndarray, forearm_deg: numpy.ndarray, window: int):

    for index in range(len(position_cm) - window + 1):
        part = position_cm[index:index + window]
        extent = float(numpy.hypot(part[:, 0].max() - part[:, 0].min(), part[:, 1].max() - part[:, 1].min()))
        turned = float(forearm_deg[index:index + window].max() - forearm_deg[index:index + window].min())
        if extent < EXTENT_CM and turned < TURN_DEG:
            return index + window - 1
    return None


print("(a) the ten runs as driven (fixed 1 cm): picture at which the rule would have ended the run")
folders = sorted(folder for folder in MODEL_RUNS.glob("2026-11-11_17*") if (folder / "steps.csv").exists())
for window in (15, 20, 25, 30):
    results = []
    for folder in folders:
        steps = pandas.read_csv(folder / "steps.csv")
        steps = steps[steps["action"] != "stop"] if "action" in steps.columns else steps
        position = 100.0 * steps[["tip_x", "tip_y"]].to_numpy()
        forearm = numpy.degrees(steps["joint_4"].to_numpy())
        results.append(f"{folder.name[11:17]}: {first_fire(position, forearm, window)}")
    print(f"   window {window}: " + ", ".join(results))

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
print()
print("(b) the fold models' own asked pace along the held-out demonstrations (own length, at most 1.0 cm)")
for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1], "p_twist": data["exact"][:, 2],
                                  "p_stop": data["stop_probability"]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    for window in (15, 20, 25, 30):
        fires, early = 0, 0
        for episode_id, group in table.groupby("episode_id", sort = False):
            stop = group["p_stop"].to_numpy()
            two_in_a_row = numpy.flatnonzero((stop[1:] >= 0.5) & (stop[:-1] >= 0.5))
            group = group.iloc[:int(two_in_a_row[0] + 1) if len(two_in_a_row) else len(group)]
            if len(group) < window:
                continue
            heading = numpy.radians(group["picture_heading_deg"].to_numpy())
            forward, left = group["p_forward"].to_numpy(), group["p_left"].to_numpy()
            vector = numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])
            length = numpy.linalg.norm(vector, axis = 1)
            vector = vector * (numpy.minimum(length, 1.0) / numpy.maximum(length, 1e-9))[:, None]
            position = numpy.vstack([numpy.zeros((1, 2)), numpy.cumsum(vector, axis = 0)])[:-1]
            forearm = numpy.r_[0.0, numpy.cumsum(numpy.clip(group["p_twist"].to_numpy(), -10.0, 10.0))][:-1]
            picture = first_fire(position, forearm, window)
            fires += int(picture is not None)
            early += int(picture is not None and picture < 0.9 * len(group))
        print(f"   {recipe}, window {window}: fires in {fires} of {table['episode_id'].nunique()} demonstrations, {early} of them before the last tenth of the run")
