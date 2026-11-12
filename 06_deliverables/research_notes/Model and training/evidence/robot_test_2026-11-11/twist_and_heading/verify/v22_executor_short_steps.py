"""
Verifier 22: the driver's own no-robot executor check (model_mode.executor_check: recorded steps carried out ON PAPER by
model_step, nothing moves, nothing connects), repeated for the steps the configured check leaves out. As configured it
takes only recorded steps of 0.5 cm or more ("executor_check_smallest_step_m": 0.005); the predicted-length rule would
command exactly the shorter ones. How often would a recorded short step be refused with the wrist held, and how far is
the planned move from the recorded one?
"""
import math
import sys
import numpy
import pandas
from vcommon import ROOT

import model_mode
from config import MODEL_MODE_PARAMETERS, TRAINING_DATA_PARAMETERS
from side_view_test import magnet_tip_and_lean

for label, overrides in (("as configured (fixed 1 cm, recorded steps of 0.5 cm or more)", {}),
                         ("predicted length, recorded steps of 0.5 cm or more", {"step_length_rule": "predicted", "step_most_m": 0.015}),
                         ("predicted length, ALL recorded steps", {"step_length_rule": "predicted", "step_most_m": 0.015, "executor_check_smallest_step_m": 0.0, "executor_check_rows": 1200})):
    results = model_mode.executor_check(overrides)
    refused = results["refusal"].notna()
    print(f"{label}: {len(results)} steps; refused {int(refused.sum())} ({100 * refused.mean():.1f} %); direction error median {results['direction_error_deg'].median():.2f}, 95th percentile {results['direction_error_deg'].quantile(0.95):.2f} degrees;"
          f" twist error median {results['twist_error_deg'].median():.3f}; clamped turns {int(results['is_turn_clamped'].sum())}")
    if refused.any():
        print("     refusals: " + "; ".join(f"{reason}: {count}" for reason, count in results.loc[refused, "refusal"].str.split(",").str[0].str.slice(0, 60).value_counts().items()))

# the short steps alone, with the predicted rule: planned tip move against the recorded tip move (length and direction)
frames = pandas.read_csv(TRAINING_DATA_PARAMETERS["processed_folder"] / "frames.csv" if hasattr(TRAINING_DATA_PARAMETERS["processed_folder"], "joinpath") else str(TRAINING_DATA_PARAMETERS["processed_folder"]) + "/frames.csv",
                         usecols = ["episode_id", "row_in_episode", "action", "tip_forward_m", "tip_left_m", "forearm_change_deg"] + [f"joint_{k}" for k in range(1, 7)])
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
has_next = (frames["episode_id"].shift(-1) == frames["episode_id"]).to_numpy() & (frames["action"] != "stop").to_numpy()
length = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0)).to_numpy()
short = numpy.flatnonzero(has_next & (length < 0.5) & (length >= 0.05) & frames["forearm_change_deg"].notna().to_numpy())
chosen = numpy.sort(numpy.random.default_rng(22).choice(short, size = 500, replace = False))
settings = dict(MODEL_MODE_PARAMETERS)
settings.update({"step_length_rule": "predicted", "step_most_m": 0.015})
joint_columns = [f"joint_{k}" for k in range(1, 7)]
refusals, length_error, direction_error, twist_asked = [], [], [], []
for position in chosen:
    now = frames.loc[position, joint_columns].to_numpy(dtype = float).tolist()
    after = frames.loc[position + 1, joint_columns].to_numpy(dtype = float).tolist()
    decision = {"tip_forward_cm": 100.0 * frames.loc[position, "tip_forward_m"], "tip_left_cm": 100.0 * frames.loc[position, "tip_left_m"], "twist_deg": float(frames.loc[position, "forearm_change_deg"])}
    planned = model_mode.model_step(now, decision, None, settings)
    refusals.append(planned["refusal"])
    if planned["joints_next"] is None:
        continue
    recorded = 100.0 * (magnet_tip_and_lean(after)[0][:2] - magnet_tip_and_lean(now)[0][:2])
    made = 100.0 * (magnet_tip_and_lean(planned["joints_next"])[0][:2] - magnet_tip_and_lean(now)[0][:2])
    length_error.append(numpy.hypot(*made) - numpy.hypot(*recorded))
    direction_error.append(abs((math.degrees(math.atan2(made[1], made[0]) - math.atan2(recorded[1], recorded[0])) + 180.0) % 360.0 - 180.0))
    twist_asked.append(abs(decision["twist_deg"]))
refusals = pandas.Series(refusals)
print(f"recorded steps of 0.05 to 0.5 cm through model_step with the predicted length: {len(chosen)} steps (median |twist| {numpy.median(twist_asked):.1f} degrees); refused {int(refusals.notna().sum())} ({100 * refusals.notna().mean():.1f} %);"
      f" planned minus recorded tip move: median {numpy.median(length_error):+.3f} cm, 95th percentile of |error| {numpy.quantile(numpy.abs(length_error), 0.95):.3f} cm; direction error median {numpy.median(direction_error):.1f}, 95th percentile {numpy.quantile(direction_error, 0.95):.1f} degrees")
if refusals.notna().any():
    print("     refusals: " + "; ".join(f"{reason}: {count}" for reason, count in refusals.dropna().str.split(",").str[0].str.slice(0, 70).value_counts().items()))
