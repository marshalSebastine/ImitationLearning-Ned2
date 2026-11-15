"""
Plan review (key: exam_and_schedule): unit times measured from files, for the robot-hour sum.

Reads only. Nothing connects to the robot, nothing is trained, no git command is run.
Run:  D:/AI_Project/.venv/Scripts/python.exe timing_checks.py > timing_checks.out.txt
"""

import datetime
import json
import re
import statistics
from pathlib import Path

import pandas

RUN_FOLDER = Path("D:/AI_Project/03_data/model_runs")
EPISODE_TABLE = Path(__file__).resolve().parents[3] / "design_review" / "improvement" / "episode_timing_episodes.csv"


def model_run_cycle() -> None:
    """Start-to-start time of the ten robot runs of 11.11.26, from each summary.json."""

    rows = []
    for folder in sorted(RUN_FOLDER.glob("2026-11-11_17*")):
        summary = json.loads((folder / "summary.json").read_text(encoding = "utf-8"))
        rows.append((folder.name, datetime.datetime.fromisoformat(summary["started"]), summary["seconds"], summary["moves"], summary["ending"]))
    print("=== the ten robot runs of 11.11.26 (summary.json) ===")
    for name, started, seconds, moves, ending in rows:
        print(f"  {name}: started {started.time()}, {moves} moves in {seconds:.1f} s = {seconds / moves:.2f} s per move, ending {ending}")
    gaps = [(rows[index + 1][1] - rows[index][1]).total_seconds() / 60 for index in range(len(rows) - 1)]
    print(f"  start-to-start gaps (min): {[round(gap, 2) for gap in gaps]}")
    print(f"  mean {statistics.mean(gaps):.2f}, median {statistics.median(gaps):.2f} minutes per run, start to start")
    whole = (rows[-1][1] - rows[0][1]).total_seconds() + rows[-1][2]
    print(f"  whole test: {whole / 60:.1f} minutes for {len(rows)} runs")
    all_seconds = sum(row[2] for row in rows)
    all_moves = sum(row[3] for row in rows)
    print(f"  driving: {all_seconds:.0f} s for {all_moves} moves = {all_seconds / all_moves:.2f} s per move; driving is {100 * all_seconds / whole:.0f} % of the session")


def demonstration_cycle() -> None:
    """Time per demonstration: median gap against whole-sitting mean, per operator, new plan number against repeat."""

    table = pandas.read_csv(EPISODE_TABLE)
    table["plan"] = table["folder"].str.extract(r"_plan_(\d\d-\d\d)_")[0]
    table["date"] = table["folder"].str.slice(0, 10)
    kept = table[table["is_kept"]]
    print("\n=== demonstrations 6 to 9.11.26 (episode_timing_episodes.csv, written by the earlier analysis) ===")
    print(f"  folders {len(table)}, kept {len(kept)}; discarded or failed {len(table) - len(kept)} = {100 * (len(table) - len(kept)) / len(table):.0f} % of folders")
    print(f"  operators of kept episodes: {kept['operator'].value_counts().to_dict()}")
    print(f"  kept episodes per date: {kept['date'].value_counts().sort_index().to_dict()}")
    cycle = kept["cycle_min"].dropna()
    print(f"  start to next start, kept episodes: median {cycle.median():.2f} min, mean {cycle.mean():.2f} min, 75th percentile {cycle.quantile(0.75):.2f}, 90th {cycle.quantile(0.9):.2f}")
    sitting_minutes = 0.0
    for sitting, group in table.groupby("sitting"):
        span = (group["finished_s"].max() - group["started_s"].min()) / 60
        sitting_minutes += span
        print(f"  sitting {sitting}: {span:.0f} min, kept {int(group['is_kept'].sum())}, {span / max(1, group['is_kept'].sum()):.2f} min per kept episode")
    print(f"  all sittings: {sitting_minutes / 60:.1f} h for {len(kept)} kept episodes = {sitting_minutes / len(kept):.2f} min per kept episode (the number to plan hours with)")

    # a new plan number against a repeat of the same plan number (next kept episode in time order, same sitting)
    ordered = table.sort_values("started_s").reset_index(drop = True)
    new_plan, same_plan = [], []
    for index in range(1, len(ordered)):
        now, before = ordered.loc[index], ordered.loc[index - 1]
        if now["sitting"] != before["sitting"]:
            continue
        gap = (now["started_s"] - before["started_s"]) / 60
        (same_plan if now["plan"] == before["plan"] else new_plan).append(gap)
    print(f"  start-to-start gap when the plan number stays (a retry): n {len(same_plan)}, median {statistics.median(same_plan):.2f}, mean {statistics.mean(same_plan):.2f} min")
    print(f"  start-to-start gap when the plan number changes: n {len(new_plan)}, median {statistics.median(new_plan):.2f}, mean {statistics.mean(new_plan):.2f} min")

    # the first five kept episodes of each sheet against the rest (learning a new shape)
    print("  first five kept episodes of a sheet against the later ones (cycle, min):")
    for sheet, group in kept.sort_values("started_s").groupby("sheet"):
        first, later = group["cycle_min"].head(5).dropna(), group["cycle_min"].iloc[5:].dropna()
        print(f"    {sheet}: first five mean {first.mean():.1f} (median {first.median():.1f}); later mean {later.mean():.1f} (median {later.median():.1f}); n {len(group)}")


if __name__ == "__main__":
    model_run_cycle()
    demonstration_cycle()
