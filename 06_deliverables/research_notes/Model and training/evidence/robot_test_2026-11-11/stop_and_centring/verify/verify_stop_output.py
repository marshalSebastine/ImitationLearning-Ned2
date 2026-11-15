"""
Verifier's check 2 (11.11.26): where would the held-out fold models stop, measured WITHOUT any picture measure: the
magnet tip's distance to the demonstrator's own stop place (frames.csv "goal_distance_cm", from the arm model) at the
first picture with a stop chance of 0.5 or more. Reads frames.csv and the saved predictions only.
"""

from pathlib import Path

import numpy
import pandas

INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RUNS_FOLDER = Path(__file__).resolve().parents[7] / "04_runs" / "gpu_ki014"
RUNS = {"A recipe": "2026-11-09_resnet18_final_candidate", "A recipe, seed 2": "2026-11-09_resnet18_final_candidate_seed_2",
        "E recipe": "2026-11-10_resnet18_chunk_stop_weight_25", "E recipe, seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"}
THRESHOLD = 0.5
LAST_PICTURES = 8

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False).set_index("frame_index")
for label, folder in RUNS.items():
    scored = []
    for number in range(1, 7):
        saved = numpy.load(RUNS_FOLDER / folder / f"predictions_without_path_{number:02d}.npz")
        part = frames.loc[saved["frame_index"], ["episode_id", "row_in_episode", "action", "goal_distance_cm", "steps_to_end", "sheet"]].copy()
        part["stop_probability"] = saved["stop_probability"]
        scored.append(part)
    scored = pandas.concat(scored).sort_values(["episode_id", "row_in_episode"])
    first_any, first_last, confirmed, never, early = [], [], [], 0, 0
    for episode_id, rows in scored.groupby("episode_id", sort = False):
        if rows["action"].iloc[-1] != "stop":
            continue
        is_like = (rows["stop_probability"] >= THRESHOLD).to_numpy()
        distance = rows["goal_distance_cm"].to_numpy() * 10.0
        tail = slice(max(0, len(rows) - LAST_PICTURES), len(rows))
        if is_like[tail].any():
            index = int(numpy.flatnonzero(is_like[tail])[0]) + tail.start
            first_last.append(distance[index])
            if index + 1 < len(rows) and is_like[index + 1]:
                confirmed.append(distance[index + 1])
        else:
            never += 1
        if is_like.any():
            first = int(numpy.flatnonzero(is_like)[0])
            first_any.append(distance[first])
            early += int(distance[first] > 30.0)
    first_last = numpy.array(first_last)
    confirmed = numpy.array(confirmed)
    first_any = numpy.array(first_any)
    episodes = len(first_last) + never
    print(f"{label} ({folder}): {episodes} held-out episodes with a stop row")
    print(f"  first of the last {LAST_PICTURES} pictures with a stop chance of {THRESHOLD} or more: found in {len(first_last)}, never in {never}")
    print(f"    the tip is then this far from the demonstrator's stop place: median {numpy.median(first_last):.1f} mm, mean {first_last.mean():.1f} mm,"
          f" 10th to 90th percentile {numpy.percentile(first_last, 10):.1f} to {numpy.percentile(first_last, 90):.1f} mm; over 7 mm in {int((first_last > 7).sum())}")
    print(f"    the next picture is also stop-like in {len(confirmed)}: tip then median {numpy.median(confirmed):.1f} mm from the stop place")
    for push in (0.0, 5.0, 8.0, 10.0):
        share = float((numpy.abs(first_last - push) <= 7.0).mean())
        print(f"    a last straight push of {push:4.1f} mm after that picture: within 7 mm of the demonstrator's stop place in {100 * share:.0f} % of the {len(first_last)}"
              f" ({100 * share * len(first_last) / episodes:.0f} % of all {episodes})")
    print(f"  first stop-like picture anywhere in the episode: {len(first_any)} episodes, more than 3 cm before the stop place in {early}")
