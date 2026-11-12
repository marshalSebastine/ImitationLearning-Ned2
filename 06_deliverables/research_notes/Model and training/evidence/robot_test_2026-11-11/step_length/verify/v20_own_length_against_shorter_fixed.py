"""
Verifier: what does carrying out the model's OWN length add over simply making the fixed step shorter?

On held-out sheets (fold models, teacher forced) the own-length rule is compared with a fixed step of the same mean
length (about 0.67 cm) and of 0.5 cm: the net forearm turn per cm in sharp stretches, the landing error one step
ahead, and how much the predicted length really changes between bends and straight parts. On the robot's logs of
11.11.26 (models that had seen the sheets) the same split is shown for comparison.

No training, nothing connects to the robot.
Run from this folder:  python v20_own_length_against_shorter_fixed.py > v20_own_length_against_shorter_fixed.out.txt
"""

import os
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
RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = [
    "2026-11-11_172021_check_no_puck", "2026-11-11_172421_E_sheet01_run1", "2026-11-11_172743_E_sheet02_run1",
    "2026-11-11_173036_E_sheet02_run2", "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1",
    "2026-11-11_174448_E_sheet05_run1", "2026-11-11_174936_E_sheet06_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie",
    "2026-11-11_175544_B_team5_sheet03"
]
pandas.set_option("display.width", 250)

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)

for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1], "p_twist": data["exact"][:, 2]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    table = table[table["action"] != "stop"].copy()
    dv = 100.0 * table[["tip_forward_m", "tip_left_m"]].fillna(0.0).to_numpy()
    pv = table[["p_forward", "p_left"]].to_numpy()
    d, p = numpy.linalg.norm(dv, axis = 1), numpy.linalg.norm(pv, axis = 1)
    dt, pt = table["forearm_change_deg"].fillna(0.0).to_numpy(), table["p_twist"].to_numpy()
    bend, straight = numpy.abs(dt) >= 5.0, numpy.abs(dt) < 2.0
    predicted_bend = numpy.abs(pt) >= 4.0
    print(f"================ {recipe}")
    print(f"   predicted length, mean: rows with a demonstrated twist of 5 degrees or more {p[bend].mean():.3f} cm, under 2 degrees {p[straight].mean():.3f} cm (ratio {p[bend].mean() / p[straight].mean():.2f}); "
          f"demonstrated length: {d[bend].mean():.3f} against {d[straight].mean():.3f} cm (ratio {d[bend].mean() / d[straight].mean():.2f})")
    print(f"   predicted length where the model itself asks for a twist of 4 degrees or more: {p[predicted_bend].mean():.3f} cm ({100 * predicted_bend.mean():.0f} % of rows), elsewhere {p[~predicted_bend].mean():.3f} cm; "
          f"correlation of predicted length with predicted twist size {numpy.corrcoef(p, numpy.abs(pt))[0, 1]:+.2f}, with demonstrated twist size {numpy.corrcoef(p, numpy.abs(dt))[0, 1]:+.2f}")
    own = numpy.minimum(p, 1.2)
    direction = pv / numpy.maximum(p, 1e-9)[:, None]
    rules = {"fixed 1.0 cm": numpy.ones_like(p), f"fixed {own.mean():.2f} cm (the own length's mean)": numpy.full_like(p, own.mean()), "fixed 0.5 cm": numpy.full_like(p, 0.5),
             "own length, at most 1.2 cm": own}
    episode = table["episode_id"].to_numpy()
    for name, length in rules.items():
        landing = numpy.linalg.norm(direction * length[:, None] - dv, axis = 1)
        # net turn per cm in sharp windows of 10 rows (demonstrated net turn of 30 degrees or more)
        signed_sum, length_sum, demonstrated_sum, demonstrated_length = 0.0, 0.0, 0.0, 0.0
        start = 0
        for episode_id, group in table.groupby("episode_id", sort = False):
            count = len(group)
            for first in range(start, start + count - 9, 10):
                part = slice(first, first + 10)
                net = dt[part].sum()
                if abs(net) >= 30.0:
                    signed_sum += numpy.clip(pt[part], -10.0, 10.0).sum() * numpy.sign(net)
                    length_sum += length[part].sum()
                    demonstrated_sum += abs(net)
                    demonstrated_length += d[part].sum()
            start += count
        share = (signed_sum / length_sum) / (demonstrated_sum / demonstrated_length)
        print(f"   {name:36s}: landing error one step ahead, mean: bend rows {landing[bend].mean():.3f}, straight rows {landing[straight].mean():.3f}, all {landing.mean():.3f} cm; "
              f"net turn per cm in sharp windows as a share of the demonstrated {share:.2f}")
    print()

print("================ on the robot, 11.11.26 (all-sheets models, sheets they were trained on): asked length by asked twist")
rows = []
for number, folder in enumerate(RUN_FOLDERS, start = 1):
    steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
    steps = steps[steps["action"] == "model_step"]
    asked = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
    big = steps["model_twist_deg"].abs() >= 4.0
    rows.append({"run": number, "moves": len(steps), "asked_mean_twist_4_or_more": asked[big].mean() if big.any() else numpy.nan, "moves_with_such_a_twist": int(big.sum()),
                 "asked_mean_elsewhere": asked[~big].mean()})
print(pandas.DataFrame(rows).round(2).to_string(index = False))
