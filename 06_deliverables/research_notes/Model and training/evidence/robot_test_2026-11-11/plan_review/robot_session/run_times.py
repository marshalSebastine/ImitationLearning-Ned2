"""
Robot-session critic, 11.11.26: how long did the ten runs of the first robot test take, with set-up?
Reads only the logs (summary.json, steps.csv); nothing connects to the robot.
"""

import csv
import datetime
import json
from pathlib import Path

RUN_ROOT = Path("D:/AI_Project/03_data/model_runs")

rows = []
for folder in sorted(RUN_ROOT.glob("2026-11-11_17*")):
    summary = json.loads((folder / "summary.json").read_text(encoding = "utf-8"))
    with open(folder / "steps.csv", newline = "", encoding = "utf-8") as file:
        steps = list(csv.DictReader(file))
    times = [float(step["unix_time"]) for step in steps]
    started = datetime.datetime.fromisoformat(summary["started"])
    first_picture = datetime.datetime.fromtimestamp(times[0])
    last_picture = datetime.datetime.fromtimestamp(times[-1])
    rows.append({"label": summary["label"], "ending": summary["ending"], "pictures": summary["pictures"], "moves": summary["moves"],
                 "seconds": summary["seconds"], "started": started, "first_picture": first_picture, "last_picture": last_picture,
                 "step_rule": summary["parameters"].get("step_length_rule"), "step_m": summary["parameters"].get("step_m"),
                 "run_name": summary["run_name"], "checkpoints": len(summary["checkpoints"])})

print("label | ending | pictures | moves | seconds logged | seconds per picture | started | last picture | gap to next start (s)")
for index, row in enumerate(rows):
    per_picture = (row["last_picture"] - row["first_picture"]).total_seconds() / max(1, row["pictures"] - 1)
    gap = (rows[index + 1]["started"] - row["last_picture"]).total_seconds() if index + 1 < len(rows) else float("nan")
    cycle = (rows[index + 1]["started"] - row["started"]).total_seconds() if index + 1 < len(rows) else float("nan")
    print(f"{row['label']:28s} | {row['ending']:8s} | {row['pictures']:3d} | {row['moves']:3d} | {row['seconds']:6.1f} | {per_picture:5.2f} | "
          f"{row['started']:%H:%M:%S} | {row['last_picture']:%H:%M:%S} | gap {gap:6.0f} | start to start {cycle:6.0f} | {row['run_name']} x{row['checkpoints']} | {row['step_rule']} {row['step_m']}")

total = (rows[-1]["last_picture"] - rows[0]["started"]).total_seconds()
print(f"\nfirst start to last picture: {total / 60:.1f} minutes for {len(rows)} runs = {total / 60 / len(rows):.1f} minutes per run")
cycles = [(rows[index + 1]["started"] - rows[index]["started"]).total_seconds() for index in range(len(rows) - 1)]
print("start-to-start cycles (min):", ", ".join(f"{value / 60:.1f}" for value in cycles))
print(f"median cycle {sorted(cycles)[len(cycles) // 2] / 60:.1f} min, mean {sum(cycles) / len(cycles) / 60:.1f} min")
motion = [row["seconds"] for row in rows]
print("seconds logged inside the run loop:", motion, "sum", round(sum(motion), 1))
