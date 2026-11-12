"""
Verifier: when the predicted step is short, which direction is nearer to where the demonstration's tip really went
over the next five rows: the model's current direction, or the last direction it was sure of (predicted length of
0.7 cm or more)? The analyst's numbers: 50 % against 29 to 34 % within 45 degrees for predicted lengths under 0.3 cm.
Also: a nudge along the SUM of the last six predicted steps, as the proposed stall guard would take.

Fold models on held-out sheets, teacher forced. No training, nothing connects to the robot.
Run from this folder:  python v17_short_step_direction.py > v17_short_step_direction.out.txt
"""

import math
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
pandas.set_option("display.width", 250)

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)


def angle_gap(first_deg, second_deg):
    return abs((first_deg - second_deg + 180.0) % 360.0 - 180.0)


for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    records = []
    for episode_id, group in table.groupby("episode_id", sort = False):
        heading = numpy.radians(group["picture_heading_deg"].to_numpy())
        forward, left = group["p_forward"].to_numpy(), group["p_left"].to_numpy()
        vector = numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])
        length = numpy.linalg.norm(vector, axis = 1)
        angle = numpy.degrees(numpy.arctan2(vector[:, 1], vector[:, 0]))
        tip = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
        is_stop = (group["action"] == "stop").to_numpy()
        last_sure = math.nan
        for index in range(len(group) - 1):
            ahead = min(index + 5, len(group) - 1)
            moved = tip[ahead] - tip[index]
            if not is_stop[index] and numpy.hypot(moved[0], moved[1]) >= 1.0:
                path_angle = math.degrees(math.atan2(moved[1], moved[0]))
                summed = vector[max(0, index - 5):index + 1].sum(axis = 0)
                records.append({"predicted_cm": length[index], "current": angle_gap(angle[index], path_angle),
                                "last_sure": angle_gap(last_sure, path_angle) if not math.isnan(last_sure) else math.nan,
                                "sum_of_six": angle_gap(math.degrees(math.atan2(summed[1], summed[0])), path_angle),
                                "sum_of_six_cm": float(numpy.linalg.norm(summed))})
            if length[index] >= 0.7:
                last_sure = angle[index]
    result = pandas.DataFrame(records)
    print(f"================ {recipe}: {len(result)} rows with a demonstrated path of 1 cm or more over the next 5 rows")
    for name, mask in [("predicted under 0.15 cm", result["predicted_cm"] < 0.15), ("predicted under 0.3 cm", result["predicted_cm"] < 0.3),
                       ("predicted 0.3 to 0.7 cm", (result["predicted_cm"] >= 0.3) & (result["predicted_cm"] < 0.7)), ("predicted 0.7 cm or more", result["predicted_cm"] >= 0.7)]:
        part = result[mask]
        known = part["last_sure"].notna()
        print(f"   {name:24s} {len(part):6d} rows: within 45 degrees of the demonstrated path: current direction {100 * (part['current'] <= 45).mean():.0f} % (median error {part['current'].median():.0f}); "
              f"last sure direction {100 * (part.loc[known, 'last_sure'] <= 45).mean():.0f} % ({part.loc[known, 'last_sure'].median():.0f}); "
              f"sum of the last six predicted steps {100 * (part['sum_of_six'] <= 45).mean():.0f} % ({part['sum_of_six'].median():.0f})")
    # the situation of the guard itself: the last six predicted steps add up to less than 1 cm
    stalled = result[result["sum_of_six_cm"] < 1.0]
    print(f"   rows where the last six predicted steps add up to under 1.0 cm (the guard's own situation): {len(stalled)}; the sum's direction is within 45 degrees of the demonstrated path on "
          f"{100 * (stalled['sum_of_six'] <= 45).mean():.0f} % (median error {stalled['sum_of_six'].median():.0f} degrees), within 90 degrees on {100 * (stalled['sum_of_six'] <= 90).mean():.0f} %")
    print()
