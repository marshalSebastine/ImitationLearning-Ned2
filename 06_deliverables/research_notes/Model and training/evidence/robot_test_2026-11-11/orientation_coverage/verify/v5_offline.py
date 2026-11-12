"""
The verifier's own recomputation of the offline scores by start kind, direction and reach from the saved predictions
of the fold models (no model is run, nothing is trained).

A picture is a hit when the saved predicted step direction ("angle_deg", in the picture's frame) lies within 45
degrees of the path over the next three rows (own computation from the tip's positions; path at least 1.5 cm).

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v5_offline.py > v5_offline.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from training_targets import load_training_frames  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 400)

RUNS_FOLDER = ROOT_FOLDER / "04_runs" / "gpu_ki014"
FOLD_RUNS = {"A": "2026-11-09_resnet18_final_candidate", "A seed 2": "2026-11-09_resnet18_final_candidate_seed_2",
             "E": "2026-11-10_resnet18_chunk_stop_weight_25", "E seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"}


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


frames = load_training_frames({"is_audit_cleaning": True})
frames["start_kind"] = frames["start_kind"].fillna("STD")
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
grouped = frames.groupby("episode_id", sort = False)
x = 100.0 * frames["tip_x_m"].to_numpy()
y = 100.0 * frames["tip_y_m"].to_numpy()
x_ahead = 100.0 * grouped["tip_x_m"].shift(-3).to_numpy()
y_ahead = 100.0 * grouped["tip_y_m"].shift(-3).to_numpy()
# as the analyst's rule: the three next rows must be the next three pictures of the same replay
is_chain = (grouped["row_in_episode"].shift(-3).to_numpy() - frames["row_in_episode"].to_numpy()) == 3
path_cm = numpy.hypot(x_ahead - x, y_ahead - y)
path_table = numpy.degrees(numpy.arctan2(y_ahead - y, x_ahead - x))
frames["path_in_picture"] = wrap(path_table - frames["picture_heading_deg"].to_numpy())
frames["path_from_radial"] = wrap(path_table - numpy.degrees(numpy.arctan2(y, x)))
frames["reach_cm"] = numpy.hypot(x, y)
frames["is_scored"] = is_chain & (path_cm >= 1.5) & (frames["action"] != "stop").to_numpy()
frames["order_in_episode"] = grouped.cumcount()
kind = frames["start_kind"]
frames["start_group"] = numpy.select(
    [kind == "STD", kind.isin(["TL10", "TL15", "TR10", "TR15"]), kind.isin(["OL1", "OL1.5", "OR1", "OR1.5"]), kind == "SQL", kind == "SQP"],
    ["standard", "turned 10 or 15", "puck beside the line", "square landscape", "square portrait"], default = "taught")

for label, folder in FOLD_RUNS.items():
    predicted = pandas.Series(numpy.nan, index = frames["frame_index"].to_numpy())
    for sheet in ["01", "02", "03", "04", "05", "06"]:
        saved = numpy.load(RUNS_FOLDER / folder / f"predictions_without_path_{sheet}.npz")
        held_out = frames.loc[frames["sheet"] == f"path_{sheet}", "frame_index"].to_numpy()
        known = numpy.isin(saved["frame_index"], held_out)
        predicted.loc[saved["frame_index"][known]] = saved["angle_deg"][known]
    scored = frames.copy()
    scored["predicted"] = predicted.to_numpy()
    scored = scored[scored["is_scored"] & scored["predicted"].notna()].copy()
    scored["hit"] = numpy.abs(wrap(scored["predicted"] - scored["path_in_picture"])) <= 45.0
    scored["forward_hit"] = scored["path_in_picture"].abs() <= 45.0
    print(f"\n=== {label} ({folder}) ===")
    print(f"scored pictures {len(scored)}; path within 45: {scored['hit'].mean():.3f}; mean of the six sheets {scored.groupby('sheet')['hit'].mean().mean():.3f}; always forward {scored['forward_hit'].mean():.3f}")
    first = scored[scored["order_in_episode"] < 10]
    table = first.groupby("start_group").agg(pictures = ("hit", "size"), episodes = ("episode_id", "nunique"), path_within_45 = ("hit", "mean"), always_forward = ("forward_hit", "mean"))
    print("first 10 kept rows of every episode, by kind of start:")
    print(table.round(3).to_string())
    sqp = first[first["start_group"] == "square portrait"]
    per_episode = sqp.groupby("plan_number").agg(pictures = ("hit", "size"), hits = ("hit", "sum"), forward_hits = ("forward_hit", "sum"))
    print("  the six SQP episodes one by one (first 10 rows):")
    print(per_episode.to_string())
    direction = (numpy.round(scored["path_from_radial"] / 30.0) * 30).astype(int).replace(-180, 180)
    by_direction = scored.groupby(direction).agg(pictures = ("hit", "size"), path_within_45 = ("hit", "mean"), always_forward = ("forward_hit", "mean"))
    print("by direction from radial:")
    print(by_direction.round(3).T.to_string())
    reach_bin = pandas.cut(scored["reach_cm"], [12, 20, 27.5, 35, 42.5, 49.5])
    by_reach = scored.groupby(reach_bin, observed = True).agg(pictures = ("hit", "size"), path_within_45 = ("hit", "mean"), always_forward = ("forward_hit", "mean"))
    print("by reach:")
    print(by_reach.round(3).T.to_string())
    # reach and direction together: is the fall with reach still there inside one direction class?
    away = scored[scored["path_from_radial"].abs() <= 45]
    back = scored[scored["path_from_radial"].abs() > 135]
    print("by reach, only paths within 45 degrees of straight away:", away.groupby(pandas.cut(away["reach_cm"], [12, 20, 27.5, 35, 42.5, 49.5]), observed = True)["hit"].mean().round(3).to_dict())
    print("by reach, only paths back towards the base (beyond 135):", back.groupby(pandas.cut(back["reach_cm"], [12, 20, 27.5, 35, 42.5, 49.5]), observed = True)["hit"].mean().round(3).to_dict())
    # leaving the first ten rows of every episode out (the easy starts all lie near the base)
    later = scored[scored["order_in_episode"] >= 10]
    print("by reach, without the first 10 rows of every episode:", later.groupby(pandas.cut(later["reach_cm"], [12, 20, 27.5, 35, 42.5, 49.5]), observed = True)["hit"].agg(["size", "mean"]).round(3).to_dict())
