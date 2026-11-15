"""
Plan review, key "evidence": the plan quotes the unseen-sheet stop as "right place in 106 of 209 episodes, never in 71,
too early in 32" and treats it as one number. The verifier's own table (stop_and_centring/verify/
check_early_and_never_stops.out.txt) shows it differs strongly by sheet. This script breaks the same count
(E recipe fold models, driver's rule: stop chance 0.5 or more on two pictures in a row, over all pictures) down by
sheet, by how far from the base the line ends (reach of the tip at the stop row), by the wrist at the stop row
(the driver holds 13.5 degrees) and by the kind of start. Reads frames.csv and saved predictions only.
"""

from pathlib import Path

import numpy
import pandas

INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014")
pandas.set_option("display.width", 250)

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False).set_index("frame_index")
for label, folder in {"E recipe": "2026-11-10_resnet18_chunk_stop_weight_25", "A recipe": "2026-11-09_resnet18_final_candidate"}.items():
    scored = []
    for number in range(1, 7):
        saved = numpy.load(RUNS_FOLDER / folder / f"predictions_without_path_{number:02d}.npz")
        part = frames.loc[saved["frame_index"], ["episode_id", "row_in_episode", "action", "sheet", "start_kind", "tip_x_m", "tip_y_m", "joint_5", "joint_4"]].copy()
        part["stop_probability"] = saved["stop_probability"]
        scored.append(part)
    scored = pandas.concat(scored).sort_values(["episode_id", "row_in_episode"])
    rows = []
    for episode_id, episode in scored.groupby("episode_id", sort = False):
        if episode["action"].iloc[-1] != "stop":
            continue
        is_like = (episode["stop_probability"] >= 0.5).to_numpy()
        fired = numpy.flatnonzero(is_like[:-1] & is_like[1:])
        if len(fired) == 0:
            kind = "never"
        else:
            kind = "right" if len(episode) - 1 - (int(fired[0]) + 1) <= 7 else "early"
        last = episode.iloc[-1]
        wrist = float(numpy.degrees(last["joint_5"])) if abs(last["joint_5"]) < 7 else float(last["joint_5"])
        forearm = float(numpy.degrees(last["joint_4"])) if abs(last["joint_4"]) < 7 else float(last["joint_4"])
        rows.append({"sheet": episode["sheet"].iloc[0][-2:], "start_kind": episode["start_kind"].iloc[0], "kind": kind,
                     "reach_cm": 100.0 * float(numpy.hypot(last["tip_x_m"], last["tip_y_m"])), "wrist_deg": wrist, "forearm_deg": forearm})
    table = pandas.DataFrame(rows)
    print(f"================ {label}: {len(table)} held-out episodes: " + ", ".join(f"{kind} {int((table['kind'] == kind).sum())}" for kind in ("right", "never", "early")))

    def breakdown(column: str, title: str) -> None:
        counts = table.groupby([column, "kind"], observed = True).size().unstack(fill_value = 0).reindex(columns = ["right", "never", "early"], fill_value = 0)
        counts["episodes"] = counts.sum(axis = 1)
        counts["right_share"] = (counts["right"] / counts["episodes"]).round(2)
        counts["never_share"] = (counts["never"] / counts["episodes"]).round(2)
        print(title)
        print(counts.to_string())

    breakdown("sheet", "by sheet")
    table["reach_bin"] = pandas.cut(table["reach_cm"], [0, 30, 35, 40, 42.5, 45, 60])
    breakdown("reach_bin", "by reach of the tip at the stop row (cm from the base axis)")
    table["wrist_bin"] = pandas.cut(table["wrist_deg"], [-40, 7.8, 19.2, 60], labels = ["below 7.8 (more than 5.7 under 13.5)", "7.8 to 19.2 (the driver's view)", "above 19.2"])
    breakdown("wrist_bin", "by wrist at the stop row (degrees)")
    table["forearm_bin"] = pandas.cut(table["forearm_deg"].abs(), [-1, 30, 60, 90, 130])
    breakdown("forearm_bin", "by size of the forearm twist at the stop row (degrees)")
    table["start"] = table["start_kind"].where(table["start_kind"].isin(["STD", "NE", "MID", "SQL", "SQP"]), "offset, turned or other")
    breakdown("start", "by kind of start")
    print("sheet x reach bin, right / episodes:")
    cross = table.groupby(["sheet", "reach_bin"], observed = True).agg(episodes = ("kind", "size"), right = ("kind", lambda values: int((values == "right").sum())))
    print(cross.to_string())
    print()

# added: the forearm twist at the stop row inside each sheet (E recipe and A recipe), to see whether the pattern is only a sheet effect
for label, folder in {"E recipe": "2026-11-10_resnet18_chunk_stop_weight_25", "A recipe": "2026-11-09_resnet18_final_candidate"}.items():
    scored = []
    for number in range(1, 7):
        saved = numpy.load(RUNS_FOLDER / folder / f"predictions_without_path_{number:02d}.npz")
        part = frames.loc[saved["frame_index"], ["episode_id", "row_in_episode", "action", "sheet", "joint_4"]].copy()
        part["stop_probability"] = saved["stop_probability"]
        scored.append(part)
    scored = pandas.concat(scored).sort_values(["episode_id", "row_in_episode"])
    rows = []
    for episode_id, episode in scored.groupby("episode_id", sort = False):
        if episode["action"].iloc[-1] != "stop":
            continue
        is_like = (episode["stop_probability"] >= 0.5).to_numpy()
        fired = numpy.flatnonzero(is_like[:-1] & is_like[1:])
        kind = "never" if len(fired) == 0 else ("right" if len(episode) - 1 - (int(fired[0]) + 1) <= 7 else "early")
        last = episode.iloc[-1]
        forearm = float(numpy.degrees(last["joint_4"])) if abs(last["joint_4"]) < 7 else float(last["joint_4"])
        rows.append({"sheet": episode["sheet"].iloc[0][-2:], "kind": kind, "twisted": "forearm within 30 deg of straight" if abs(forearm) <= 30 else "forearm twisted more than 30 deg"})
    table = pandas.DataFrame(rows)
    cross = table.groupby(["sheet", "twisted"]).agg(episodes = ("kind", "size"), right = ("kind", lambda values: int((values == "right").sum())),
                                                     never = ("kind", lambda values: int((values == "never").sum())), early = ("kind", lambda values: int((values == "early").sum())))
    print(f"================ {label}: forearm twist at the stop row inside each sheet")
    print(cross.to_string())
    total = table.groupby("twisted").agg(episodes = ("kind", "size"), right = ("kind", lambda values: int((values == "right").sum())))
    print(total.to_string())
