"""
Verifier's check 2b: the analyst's 'short of the end's centre' at the first stop-like picture against the arm model's
tip distance to the demonstrator's stop place, on the same episodes and pictures (E recipe and A recipe).
"""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RUNS_FOLDER = HERE.parents[6] / "04_runs" / "gpu_ki014"

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False).set_index("frame_index")
last_steps = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_last_steps_01_03.csv", "demonstration_last_steps_04_06.csv")])
ends = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_ends_01_03.csv", "demonstration_ends_04_06.csv")])
print("analyst's last-step rows:", len(last_steps), "episodes", last_steps["episode_id"].nunique(), "; ends rows", len(ends))
print("columns of ends:", list(ends.columns))
for label, folder in {"E recipe": "2026-11-10_resnet18_chunk_stop_weight_25", "A recipe": "2026-11-09_resnet18_final_candidate"}.items():
    chance = {}
    for number in range(1, 7):
        saved = numpy.load(RUNS_FOLDER / folder / f"predictions_without_path_{number:02d}.npz")
        chance.update(dict(zip(saved["frame_index"].tolist(), saved["stop_probability"].tolist())))
    table = last_steps.copy()
    table["stop_probability"] = table["frame_index"].map(chance)
    table["tip_distance_mm"] = 10.0 * table["frame_index"].map(frames["goal_distance_cm"])
    table = table.dropna(subset = ["stop_probability"]).sort_values(["episode_id", "step"])
    rows = []
    for episode_id, group in table.groupby("episode_id", sort = False):
        like = group[group["stop_probability"] >= 0.5]
        if len(like):
            first = like.iloc[0]
            rows.append({"episode_id": episode_id, "steps_before_stop": first["steps_before_stop"], "short_mm": first["short_mm"], "tip_distance_mm": first["tip_distance_mm"]})
    found = pandas.DataFrame(rows)
    print(f"{label}: episodes in the analyst's table {table['episode_id'].nunique()}, first stop-like picture found in {len(found)}")
    print("   analyst's short_mm: median %.1f, 10th to 90th %.1f to %.1f (rows with a value %d)" % (found["short_mm"].median(), found["short_mm"].quantile(0.1), found["short_mm"].quantile(0.9), found["short_mm"].notna().sum()))
    print("   arm model's tip distance to the stop place at the same pictures: median %.1f, 10th to 90th %.1f to %.1f" % (found["tip_distance_mm"].median(), found["tip_distance_mm"].quantile(0.1), found["tip_distance_mm"].quantile(0.9)))
    print("   pictures before the stop at the first stop-like picture:", found["steps_before_stop"].value_counts().sort_index().to_dict())
    print("   difference tip distance minus short_mm: median %.1f mm" % (found["tip_distance_mm"] - found["short_mm"]).median())
