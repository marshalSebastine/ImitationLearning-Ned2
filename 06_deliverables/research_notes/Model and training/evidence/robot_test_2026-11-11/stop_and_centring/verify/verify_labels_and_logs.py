"""
Verifier's check 1 (11.11.26): the labels at the stop rows and the driver's last steps, recomputed from frames.csv,
the episode folders and the robot runs' steps.csv. Reads only; nothing connects to the robot.
"""

import json
import math
import sys
from pathlib import Path

import numpy
import pandas

ROOT_FOLDER = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

import training_targets
from config import TRAINING_DATA_PARAMETERS

INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RUNS = {1: "2026-11-11_172021_check_no_puck", 2: "2026-11-11_172421_E_sheet01_run1",
        4: "2026-11-11_173036_E_sheet02_run2", 7: "2026-11-11_174448_E_sheet05_run1"}

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False)
episodes = pandas.read_csv(INDEX_FOLDER / "episodes.csv")
print("PART 1, labels (frames.csv)")
print("rows", len(frames), "episodes with rows", frames["episode_id"].nunique(), "episodes.csv rows", len(episodes))
missing = sorted(set(episodes["episode_id"]) - set(frames["episode_id"])) if "episode_id" in episodes else []
print("episodes of episodes.csv without rows:", missing)
stops = frames[frames["action"] == "stop"]
print("stop rows", len(stops), "episodes with exactly one stop row", int((stops.groupby("episode_id").size() == 1).sum()))
print("goal_distance_cm at stop rows: min", stops["goal_distance_cm"].min(), "max", stops["goal_distance_cm"].max(),
      "; is_at_goal at stop rows:", int(stops["is_at_goal"].sum()), "of", len(stops))
at_goal = frames.groupby("episode_id")["is_at_goal"].sum()
print("rows with is_at_goal per episode: median", at_goal.median(), "min", at_goal.min(), "max", at_goal.max(), "; all", int(frames["is_at_goal"].sum()))
ordered = frames.sort_values(["episode_id", "row_in_episode"])
last_moves = []
for episode_id, rows in ordered.groupby("episode_id", sort = False):
    actions = rows["action"].tolist()
    if actions[-1] != "stop" or len(rows) < 2:
        continue
    tips = rows[["tip_x_m", "tip_y_m"]].to_numpy()
    last_moves.append(100.0 * float(numpy.hypot(*(tips[-1] - tips[-2]))))
last_moves = numpy.array(last_moves)
print("tip travel from the row before the stop row to the stop row (cm): episodes", len(last_moves), "median", round(float(numpy.median(last_moves)), 2),
      "under 0.5:", int((last_moves < 0.5).sum()), "under 0.2:", int((last_moves < 0.2).sum()))
before_stop = ordered.groupby("episode_id").nth(-2)
print("goal_distance_cm one row before the stop row: median", before_stop["goal_distance_cm"].median())

raw_folder = Path(TRAINING_DATA_PARAMETERS["raw_folder"])
outcomes = {}
for episode_folder in frames["episode_folder"].unique():
    path = raw_folder / episode_folder / "episode.json"
    if path.exists():
        outcome = json.loads(path.read_text(encoding = "utf-8")).get("outcome")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
print("episode.json outcome:", outcomes)

print("the training rows of the robot's models (cleaning settings read from each run's metrics.json):")
for run_folder in ("2026-11-10_resnet18_model_1_all_sheets", "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets"):
    settings = json.loads((ROOT_FOLDER / "04_runs" / "gpu_ki014" / run_folder / "metrics.json").read_text(encoding = "utf-8"))["settings"]
    keys = ("kept_scores", "is_unscored_kept", "is_mistake_dropped", "is_flagged_dropped", "is_static_dropped", "is_audit_cleaning")
    overrides = {key: settings[key] for key in keys if key in settings}
    cleaned = training_targets.load_training_frames(overrides)
    is_stop_row = cleaned["action"] == "stop"
    print(f"  {run_folder} {overrides}: rows {len(cleaned)}, episodes {cleaned['episode_id'].nunique()}, stop rows {int(is_stop_row.sum())}"
          f" (relabelled {int(cleaned['is_relabelled_stop'].sum())}), share {100.0 * float(is_stop_row.mean()):.2f} %, design {settings.get('design')}")

print()
print("PART 2, the driver's last steps (steps.csv of the four runs that reached the end)")
for number, folder in RUNS.items():
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    print(f"run {number} ({folder}): rows {len(steps)}, last action {steps['action'].iloc[-1]}")
    for index in range(max(0, len(steps) - 5), len(steps)):
        row = steps.iloc[index]
        asked = math.hypot(row["model_tip_forward_cm"], row["model_tip_left_cm"])
        angle = math.degrees(math.atan2(row["model_tip_left_cm"], row["model_tip_forward_cm"]))
        if index + 1 < len(steps):
            following = steps.iloc[index + 1]
            done = 1000.0 * math.hypot(following["tip_x"] - row["tip_x"], following["tip_y"] - row["tip_y"])
        else:
            done = float("nan")
        planned = 1000.0 * math.hypot(row["tip_next_x"] - row["tip_x"], row["tip_next_y"] - row["tip_y"]) if not pandas.isna(row["tip_next_x"]) else float("nan")
        print(f"  step {int(row['step']):3d}: asked forward {row['model_tip_forward_cm']:+.2f} left {row['model_tip_left_cm']:+.2f} cm"
              f" (length {asked:.2f} cm, {angle:+.0f} deg left of picture up), stop chance {row['model_stop_probability']:.3f},"
              f" action {row['action']}, planned step {planned:.1f} mm, tip really moved {done:.1f} mm, kept direction {row['is_direction_kept']}")
