"""
The verifier's own check of (1) how fast the six course lines bend, (2) the forearm flips in the demonstrations,
(3) whether the six SQP episodes all ran from end B, from the shape of the tip's path alone, (4) the analyst's rule
that a lay-out is impossible when a paper corner meets the base, against the lay-outs that were really recorded, and
(5) the time an episode took.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v7_bends_flips_layouts.py > v7_bends_flips_layouts.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from setup_visuals import load_sheet_lines  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 400)


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


# ---- 1. bends of the traced course lines ----------------------------------------------------------------------------
print("=== 1. How fast the course lines bend (traced lines, own measure: turn of the tangent over a 1-cm window) ===")
for number, sheet in sorted(load_sheet_lines().items()):
    line = numpy.asarray(sheet["line"], dtype = float)
    segments = numpy.linalg.norm(numpy.diff(line, axis = 0), axis = 1)
    arc = numpy.concatenate([[0.0], numpy.cumsum(segments)])
    samples = numpy.arange(0.0, arc[-1], 0.1)
    points = numpy.column_stack([numpy.interp(samples, arc, line[:, 0]), numpy.interp(samples, arc, line[:, 1])])
    # tangent over +-0.5 cm (smooths the tracing's jitter), then its turn over 1 cm
    tangent = numpy.unwrap(numpy.arctan2(points[10:, 1] - points[:-10, 1], points[10:, 0] - points[:-10, 0]))
    bend = numpy.abs(numpy.degrees(tangent[10:] - tangent[:-10]))          # degrees per cm
    print(f"  sheet {number}: length {arc[-1]:.1f} cm, sharpest bend {bend.max():.1f} degrees per cm (radius {180.0 / numpy.pi / max(bend.max(), 1e-6):.1f} cm),"
          f" centimetres bending faster than 10 degrees per cm: {0.1 * (bend > 10).sum():.1f}, faster than 20: {0.1 * (bend > 20).sum():.1f}")

# ---- 2. forearm flips -----------------------------------------------------------------------------------------------
print("\n=== 2. Forearm flips (joint 4 from beyond 60 degrees on one side to beyond 60 on the other; own count) ===")
frames = load_training_frames({"is_audit_cleaning": True}).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
frames["start_kind"] = frames["start_kind"].fillna("STD")
flips = []
for episode_id, group in frames.groupby("episode_id", sort = False):
    joint_4 = numpy.degrees(group["joint_4"].to_numpy())
    x = 100.0 * group["tip_x_m"].to_numpy()
    y = 100.0 * group["tip_y_m"].to_numpy()
    side, last_index = 0, None
    for index, value in enumerate(joint_4):
        now = 1 if value > 60 else (-1 if value < -60 else 0)
        if now == 0:
            continue
        if side != 0 and now != side:
            travel = float(numpy.hypot(numpy.diff(x[last_index:index + 1]), numpy.diff(y[last_index:index + 1])).sum())
            flips.append({"episode_id": episode_id, "sheet": group["sheet"].iloc[0][-2:], "rows": index - last_index, "degrees": abs(joint_4[index] - joint_4[last_index]), "travel_cm": travel})
        side, last_index = now, index
flips = pandas.DataFrame(flips)
print(f"  flips: {len(flips)} in {flips['episode_id'].nunique()} episodes; per sheet: {flips.groupby('sheet').size().to_dict()}")
print(f"  median rows {flips['rows'].median():.0f}, degrees {flips['degrees'].median():.0f}, tip travel {flips['travel_cm'].median():.1f} cm;"
      f" degrees per cm of travel: median {(flips['degrees'] / flips['travel_cm'].clip(lower = 0.1)).median():.1f}")
print(f"  flips made over less than 12.7 cm of travel (which 127 degrees at 10 degrees per 1-cm step needs at least): {(flips['travel_cm'] < 12.7).sum()} of {len(flips)}")

# ---- 3. the SQP episodes' end, from the path's shape ---------------------------------------------------------------
print("\n=== 3. The six SQP episodes: which way the pushed path bends (own measure; end B starts to the left and bends right) ===")
for plan in ["02-13", "02-14", "02-15", "02-16", "02-17", "02-18", "02-01", "02-29"]:
    group = frames[frames["plan_number"] == plan]
    if len(group) == 0:
        continue
    x = 100.0 * group["tip_x_m"].to_numpy()
    y = 100.0 * group["tip_y_m"].to_numpy()
    # the pushed part: from where the tip is 8 cm of path from the start to the end, sampled every 8 cm of path
    arc = numpy.concatenate([[0.0], numpy.cumsum(numpy.hypot(numpy.diff(x), numpy.diff(y)))])
    marks = numpy.arange(8.0, arc[-1], 8.0)
    px, py = numpy.interp(marks, arc, x), numpy.interp(marks, arc, y)
    directions = numpy.degrees(numpy.arctan2(numpy.diff(py), numpy.diff(px)))
    print(f"  plan {plan} ({group['start_kind'].iloc[0]}, end {group['start_end'].iloc[0]} after the audit, plan said {group['plan_end'].iloc[0]}):"
          f" path direction every 8 cm: {numpy.round(directions).astype(int).tolist()}; most leftward tip y {y.max():.1f} cm, last tip ({x[-1]:.1f}, {y[-1]:.1f})")

# ---- 4. the 'paper on the base' rule against what was recorded -----------------------------------------------------
print("\n=== 4. The analyst's fitted lay-outs: does a paper corner meet the base's square in lay-outs that were really recorded? ===")
placements = pandas.read_csv(VERIFY_FOLDER.parent / "01_placements_per_episode.csv", dtype = {"sheet": str})
half_base = 9.4
rows = []
for _, row in placements.iterrows():
    if not numpy.isfinite(row["turn_deg"]):
        continue
    turn = numpy.radians(row["turn_deg"])
    rotation = numpy.array([[numpy.cos(turn), -numpy.sin(turn)], [numpy.sin(turn), numpy.cos(turn)]])
    corners = numpy.array([[-10.5, -14.85], [10.5, -14.85], [10.5, 14.85], [-10.5, 14.85]]) @ rotation.T + numpy.array([row["paper_centre_x_cm"], row["paper_centre_y_cm"]])
    # points along the paper's edges every 0.25 cm: how deep does the paper reach into the base's square?
    edge_points = numpy.vstack([numpy.linspace(corners[index], corners[(index + 1) % 4], 120) for index in range(4)])
    depth = numpy.minimum(half_base - numpy.abs(edge_points[:, 0]), half_base - numpy.abs(edge_points[:, 1]))
    rows.append({"sheet": row["sheet"], "end": row["end"], "start_kind": row["start_kind"], "layout": row["layout"], "deepest_in_base_cm": float(depth.max()),
                 "nearest_paper_x_cm": float(edge_points[:, 0].min())})
overlap = pandas.DataFrame(rows)
summary = overlap.groupby(["sheet", "end"]).agg(episodes = ("layout", "size"), paper_in_base_square = ("deepest_in_base_cm", lambda values: int((values > 0).sum())),
                                                 deepest_cm_median = ("deepest_in_base_cm", "median"), deepest_cm_most = ("deepest_in_base_cm", "max"),
                                                 nearest_paper_x_median = ("nearest_paper_x_cm", "median"))
print(summary.round(1).to_string())
print(f"  recorded episodes whose fitted paper reaches into the base's square (half width {half_base} cm): {(overlap['deepest_in_base_cm'] > 0).sum()} of {len(overlap)};"
      f" by more than 0.5 cm: {(overlap['deepest_in_base_cm'] > 0.5).sum()}; by more than 2 cm: {(overlap['deepest_in_base_cm'] > 2.0).sum()}")

# ---- 5. minutes per episode ----------------------------------------------------------------------------------------
print("\n=== 5. Minutes from the start of one episode to the start of the next (episodes.csv, same session and sheet) ===")
episodes = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/episodes.csv")
episodes["started"] = pandas.to_datetime(episodes["started"])
episodes["finished"] = pandas.to_datetime(episodes["finished"])
episodes = episodes.sort_values(["session", "started"])
gap = (episodes.groupby("session")["started"].shift(-1) - episodes["started"]).dt.total_seconds() / 60.0
same_sheet = episodes.groupby("session")["sheet"].shift(-1) == episodes["sheet"]
valid = gap[same_sheet & (gap < 30)]
duration = (episodes["finished"] - episodes["started"]).dt.total_seconds() / 60.0
print(f"  start to start: median {valid.median():.1f} minutes (25th to 75th percentile {valid.quantile(0.25):.1f} to {valid.quantile(0.75):.1f}; {len(valid)} gaps under 30 minutes);"
      f" an episode itself: median {duration.median():.1f} minutes")
print("  by sheet (median start to start):", valid.groupby(episodes.loc[valid.index, "sheet"]).median().round(1).to_dict())
days = episodes.groupby("session").agg(episodes = ("episode_id", "size"), first = ("started", "min"), last = ("finished", "max"))
days["hours"] = (days["last"] - days["first"]).dt.total_seconds() / 3600.0
days["episodes_per_hour"] = days["episodes"] / days["hours"]
print(days[["episodes", "hours", "episodes_per_hour"]].round(1).to_string())
print(f"  all sessions: {days['episodes'].sum()} kept episodes in {days['hours'].sum():.1f} hours of session time = {days['episodes'].sum() / days['hours'].sum():.1f} per hour"
      f" (the index holds only kept episodes; discarded tries and breaks are inside the session time)")
