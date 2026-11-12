"""
The verifier's own check of what a pre-turned start could expect, without the robot.

1. Saved predictions of the fold models that never saw sheet 02: in the six SQP demonstrations, how good is the
   predicted direction in the rows AFTER the start manoeuvre, where the forearm is already turned 45 degrees or more
   and the tip is still within 25 cm of the base axis (the state a pre-turned start would begin in)? Compared with
   the first ten rows and with the standard-start episodes of sheet 02 in the same rows.
2. Model E (all sheets) on the first pictures of robot run 3: the five future tip points it predicts (the chunk), to
   see where it expected the path to go (inference on 13 pictures, two threads).

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v12_after_the_start.py > v12_after_the_start.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas
import torch

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from training_targets import load_training_frames  # noqa: E402

torch.set_num_threads(2)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)

RUNS_FOLDER = ROOT_FOLDER / "04_runs" / "gpu_ki014"
FOLD_RUNS = {"A": "2026-11-09_resnet18_final_candidate", "A seed 2": "2026-11-09_resnet18_final_candidate_seed_2",
             "E": "2026-11-10_resnet18_chunk_stop_weight_25", "E seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"}


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


frames = load_training_frames({"is_audit_cleaning": True})
frames["start_kind"] = frames["start_kind"].fillna("STD")
frames = frames[frames["sheet"] == "path_02"].sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
grouped = frames.groupby("episode_id", sort = False)
x = 100.0 * frames["tip_x_m"].to_numpy()
y = 100.0 * frames["tip_y_m"].to_numpy()
x_ahead = 100.0 * grouped["tip_x_m"].shift(-3).to_numpy()
y_ahead = 100.0 * grouped["tip_y_m"].shift(-3).to_numpy()
is_chain = (grouped["row_in_episode"].shift(-3).to_numpy() - frames["row_in_episode"].to_numpy()) == 3
path_cm = numpy.hypot(x_ahead - x, y_ahead - y)
frames["path_in_picture"] = wrap(numpy.degrees(numpy.arctan2(y_ahead - y, x_ahead - x)) - frames["picture_heading_deg"].to_numpy())
frames["is_scored"] = is_chain & (path_cm >= 1.5) & (frames["action"] != "stop").to_numpy()
frames["order"] = grouped.cumcount()
frames["reach_cm"] = numpy.hypot(x, y)
frames["joint_4_deg"] = numpy.degrees(frames["joint_4"])

print("=== 1. Fold models that never saw sheet 02, on the sheet-02 demonstrations (saved predictions; path within 45 degrees) ===")
rows = []
for label, folder in FOLD_RUNS.items():
    saved = numpy.load(RUNS_FOLDER / folder / "predictions_without_path_02.npz")
    predicted = pandas.Series(saved["angle_deg"], index = saved["frame_index"])
    scored = frames[frames["is_scored"] & frames["frame_index"].isin(predicted.index)].copy()
    scored["hit"] = numpy.abs(wrap(predicted.loc[scored["frame_index"]].to_numpy() - scored["path_in_picture"])) <= 45.0
    scored["forward_hit"] = scored["path_in_picture"].abs() <= 45.0
    sqp = scored[scored["start_kind"] == "SQP"]
    standard = scored[scored["start_kind"] == "STD"]
    turned_near = sqp[(sqp["joint_4_deg"].abs() >= 45.0) & (sqp["reach_cm"] <= 25.0)]
    for name, part in (("SQP, first 10 rows", sqp[sqp["order"] < 10]), ("SQP, rows 10 to 29", sqp[sqp["order"].between(10, 29)]),
                       ("SQP, forearm turned 45 or more within 25 cm of the base", turned_near),
                       ("SQP, all later rows (30 on)", sqp[sqp["order"] >= 30]),
                       ("STD sheet 02, first 10 rows", standard[standard["order"] < 10]), ("STD sheet 02, rows 10 to 29", standard[standard["order"].between(10, 29)])):
        rows.append({"model": label, "rows": name, "pictures": len(part), "episodes": part["episode_id"].nunique(),
                     "path_within_45": round(float(part["hit"].mean()), 3) if len(part) else numpy.nan,
                     "always_forward": round(float(part["forward_hit"].mean()), 3) if len(part) else numpy.nan})
print(pandas.DataFrame(rows).to_string(index = False))

print("\n=== 2. Model E (all sheets) on the first pictures of robot run 3: the predicted next five tip points (cm, picture frame) ===")
from config import TRAINING_RUNS  # noqa: E402,F401
from model_mode import ModelPolicy  # noqa: E402

run_folder = RUNS_FOLDER / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets"
policy = ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", run_folder / "checkpoints" / "all_sheets.pt", run_folder / "metrics.json")
torch.set_num_threads(2)
steps = pandas.read_csv("D:/AI_Project/03_data/model_runs/2026-11-11_172743_E_sheet02_run1/steps.csv")
for index in range(0, 13, 2):
    picture_bytes = (Path("D:/AI_Project/03_data/model_runs/2026-11-11_172743_E_sheet02_run1/frames") / f"{index:03d}.jpg").read_bytes()
    decision = policy.decide(picture_bytes)
    with torch.inference_mode():
        chunk = policy.models[0](policy.picture_input(picture_bytes))["chunk"][0].numpy().reshape(-1, 2)
    last = chunk[-1]
    print(f"  picture {index:2d}: step asked ({decision['tip_forward_cm']:+.2f} forward, {decision['tip_left_cm']:+.2f} left) cm, logged ({steps.loc[index, 'model_tip_forward_cm']:+.2f}, {steps.loc[index, 'model_tip_left_cm']:+.2f});"
          f" chunk points (forward, left): " + " ".join(f"({point[0]:+.1f},{point[1]:+.1f})" for point in chunk)
          + f"; fifth point {numpy.hypot(*last):.1f} cm away at {numpy.degrees(numpy.arctan2(last[1], last[0])):+.0f} degrees from the picture's top")
