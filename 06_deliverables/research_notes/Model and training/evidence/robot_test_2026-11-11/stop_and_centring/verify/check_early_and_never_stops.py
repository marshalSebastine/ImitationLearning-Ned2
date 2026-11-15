"""
Verifier's check 18 (11.11.26): what the analyst's stop analysis leaves out. It looks at the last eight pictures of
every demonstration; here the driver's rule (stop chance 0.5 or more on two pictures in a row) is run over ALL the
pictures of every held-out episode (fold models, E recipe and A recipe): does the rule fire at all, and if so where:
within the last eight pictures or earlier on the line. For the earlier ones the analyst's puck places do not exist,
so the magnet tip's distance to its stop place is given (arm model), which overstates only where the tip moves round
a standing puck at the end. Reads frames.csv and the saved predictions.
"""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RUNS_FOLDER = HERE.parents[6] / "04_runs" / "gpu_ki014"

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False).set_index("frame_index")
for label, folder in {"E recipe": "2026-11-10_resnet18_chunk_stop_weight_25", "E recipe, seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2",
                      "A recipe": "2026-11-09_resnet18_final_candidate"}.items():
    scored = []
    for number in range(1, 7):
        saved = numpy.load(RUNS_FOLDER / folder / f"predictions_without_path_{number:02d}.npz")
        part = frames.loc[saved["frame_index"], ["episode_id", "row_in_episode", "action", "goal_distance_cm", "sheet"]].copy()
        part["stop_probability"] = saved["stop_probability"]
        scored.append(part)
    scored = pandas.concat(scored).sort_values(["episode_id", "row_in_episode"])
    rows = []
    for episode_id, episode in scored.groupby("episode_id", sort = False):
        if episode["action"].iloc[-1] != "stop":
            continue
        is_like = (episode["stop_probability"] >= 0.5).to_numpy()
        fired = numpy.flatnonzero(is_like[:-1] & is_like[1:])
        entry = {"episode_id": episode_id, "sheet": episode["sheet"].iloc[0], "pictures": len(episode)}
        if len(fired) == 0:
            entry["kind"] = "never"
        else:
            second = int(fired[0]) + 1
            entry["pictures_before_end"] = len(episode) - 1 - second
            entry["tip_distance_cm"] = float(episode["goal_distance_cm"].iloc[second])
            entry["kind"] = "within the last 8 pictures" if entry["pictures_before_end"] <= 7 else "earlier"
        rows.append(entry)
    table = pandas.DataFrame(rows)
    counts = table["kind"].value_counts().to_dict()
    print(f"{label}: {len(table)} held-out episodes; the two-in-a-row rule fires {counts}")
    early = table[table["kind"] == "earlier"]
    if len(early):
        print(f"  the earlier ones: tip then a median {early['tip_distance_cm'].median():.1f} cm from its stop place (10th to 90th percentile {early['tip_distance_cm'].quantile(0.1):.1f} to {early['tip_distance_cm'].quantile(0.9):.1f} cm);"
              f" by sheet {early.groupby('sheet').size().to_dict()}")
    print("  by sheet (never / earlier / within the last 8):", {sheet: (int((part["kind"] == "never").sum()), int((part["kind"] == "earlier").sum()), int((part["kind"] == "within the last 8 pictures").sum()))
                                                           for sheet, part in table.groupby("sheet")})
