"""
Verifier, check 7: is a stop picture taken with the driver's wrist (13.5 degrees) harder for the fold models than a
stop picture taken with another wrist? REAL recorded pictures only, saved predictions of the three fold-model sets
(each sheet scored by the model that never saw it); no imitated view.

    a) stop rows by sheet and by the wrist's distance from 13.5: share with a stop chance of 0.5 or more;
    b) the same inside "same goal" groups (sheet x corrected end x start kind family), so that the line's end in the
       picture is the same and mainly the wrist differs;
    c) the last three rows before the stop row (the rule needs two pictures in a row);
    d) where the stop rule fires in start-mark episodes, by the wrist at the end.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe v07_stops_on_real_pictures.py
"""

import sys

import numpy
import pandas

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
from training_targets import load_training_frames

RUNS_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014"
RUNS = {"E": "2026-11-10_resnet18_chunk_stop_weight_25", "E2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2", "A": "2026-11-09_resnet18_final_candidate",
        "A2": "2026-11-09_resnet18_final_candidate_seed_2", "C": "2026-11-10_resnet18_final_candidate_chunk"}
pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 60)
pandas.set_option("display.max_rows", 200)

frames = load_training_frames()
frames["wrist_deg"] = numpy.degrees(frames["joint_5"])
frames["offset"] = frames["wrist_deg"] - 13.5
frames["is_start_mark"] = frames["sheet_orientation"].str.contains("START MARK", na = False)
for label, folder in RUNS.items():
    parts = []
    for sheet_number in range(1, 7):
        saved = numpy.load(f"{RUNS_FOLDER}/{folder}/predictions_without_path_{sheet_number:02d}.npz")
        parts.append(pandas.DataFrame({"frame_index": saved["frame_index"], f"stop_{label}": saved["stop_probability"]}))
    frames = frames.merge(pandas.concat(parts), on = "frame_index", how = "left")
frames = frames.dropna(subset = ["stop_E"]).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
models = list(RUNS)
frames["stop_mean"] = frames[[f"stop_{label}" for label in models]].mean(axis = 1)
frames["group"] = numpy.where(frames["offset"].abs() <= 5.7, "near", numpy.where(frames["offset"] < 0, "below", "above"))

stops = frames[frames["action"] == "stop"].copy()
print("stop rows scored:", len(stops))
print("\na) stop rows as recorded: share with a stop chance of 0.5 or more, per model set, by the wrist's group")
table = stops.groupby("group").agg(rows = ("offset", "size"), **{label: (f"stop_{label}", lambda v: round(float((v >= 0.5).mean()), 3)) for label in models})
table["mean_of_models"] = table[models].mean(axis = 1).round(3)
print(table.to_string())
print("\n   by sheet (mean over the five model sets of the share with 0.5 or more; rows in brackets):")
rows = []
for sheet, part in stops.groupby("sheet"):
    row = {"sheet": sheet}
    for group, sub in part.groupby("group"):
        row[group] = "%.2f (%d)" % (numpy.mean([(sub[f"stop_{label}"] >= 0.5).mean() for label in models]), len(sub))
    rows.append(row)
print(pandas.DataFrame(rows).fillna("-").to_string(index = False))

print("\nb) inside same-goal groups (sheet, corrected start end, start-mark or taught): near against below, where both have 3 or more stop rows")
stops["family"] = stops["sheet"] + " " + stops["start_end"].astype(str) + numpy.where(stops["is_start_mark"], " mark", " taught")
differences = []
for family, part in stops.groupby("family"):
    near, below = part[part["group"] == "near"], part[part["group"] == "below"]
    if len(near) >= 3 and len(below) >= 3:
        near_share = numpy.mean([(near[f"stop_{label}"] >= 0.5).mean() for label in models])
        below_share = numpy.mean([(below[f"stop_{label}"] >= 0.5).mean() for label in models])
        differences.append({"family": family, "near_rows": len(near), "below_rows": len(below), "near_share": round(near_share, 3), "below_share": round(below_share, 3),
                            "near_minus_below": round(near_share - below_share, 3), "near_mean_chance": round(near["stop_mean"].mean(), 3), "below_mean_chance": round(below["stop_mean"].mean(), 3)})
differences = pandas.DataFrame(differences)
print(differences.to_string(index = False))
weights = numpy.minimum(differences["near_rows"], differences["below_rows"])
print("   weighted mean of (near minus below): %+.3f; families where near is lower: %d of %d" % (
    float((differences["near_minus_below"] * weights).sum() / weights.sum()), int((differences["near_minus_below"] < 0).sum()), len(differences)))

print("\n   the wrist's distance from 13.5 at the stop row against the stop chance (mean of the model sets), rank correlation per sheet:")
for sheet, part in stops.groupby("sheet"):
    print("     %s: %d rows, Spearman of stop chance with (wrist - 13.5): %+.2f; with |wrist - 13.5|: %+.2f" % (
        sheet, len(part), part["stop_mean"].corr(part["offset"], method = "spearman"), part["stop_mean"].corr(part["offset"].abs(), method = "spearman")))

print("\nc) the three rows before the stop row (stop chance 0.5 or more, mean over the model sets), by the STOP row's wrist group")
frames["steps_before_stop"] = frames.groupby("episode_id").cumcount(ascending = False)
stop_group = stops.set_index("episode_id")["group"]
frames["end_group"] = frames["episode_id"].map(stop_group)
for back in (0, 1, 2, 3):
    part = frames[frames["steps_before_stop"] == back]
    shares = part.groupby("end_group").apply(lambda sub: round(float(numpy.mean([(sub[f"stop_{label}"] >= 0.5).mean() for label in models])), 3))
    print("   %d rows before the end: %s; goal distance median %.1f cm" % (back, shares.to_dict(), part["goal_distance_cm"].median()))

print("\nd) start-mark episodes: where the rule (0.5 twice in a row) first fires, by the wrist's group at the end, per model set")
for label in models:
    outcomes = []
    for episode_id, part in frames[frames["is_start_mark"]].groupby("episode_id"):
        passing = (part[f"stop_{label}"].to_numpy() >= 0.5)
        fired = [position for position in range(1, len(passing)) if passing[position] and passing[position - 1]]
        if not fired:
            outcome = "never"
        else:
            distance = float(part["goal_distance_cm"].to_numpy()[fired[0]])
            rows_left = len(passing) - 1 - fired[0]
            outcome = "at the end (last 2 rows)" if rows_left <= 1 else ("within 2 cm of the goal" if distance <= 2.0 else "before the goal")
        outcomes.append({"end_group": part["end_group"].iloc[0], "outcome": outcome, "rows": len(part)})
    outcomes = pandas.DataFrame(outcomes)
    print(f"   model set {label}:")
    print(outcomes.groupby("end_group")["outcome"].value_counts().unstack(fill_value = 0).assign(episode_rows_median = outcomes.groupby("end_group")["rows"].median()).to_string())
print("\noperators of the start-mark episodes by end group:")
episode_table = frames[frames["is_start_mark"]].groupby("episode_id").agg(operator = ("operator", "first"), end_group = ("end_group", "first"), sheet = ("sheet", "first"))
print(episode_table.groupby(["end_group", "operator"]).size().unstack(fill_value = 0).to_string())
