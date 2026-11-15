"""
Robot-session critic: how long would robot session 2 of the plan take? Calculated from measured pieces:
    the ten runs of 11.11.26 (run_times.out.txt in this folder: seconds per picture, gap between runs);
    the demonstrations' picture counts and tip travel per sheet (the training index), which say how many pictures a
    finished run needs at a given step length;
    the recording days' own pace (start-to-start time inside one sheet, from the episodes' start times).
Reads logs and the index only. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe session_time.py > session_time.out.txt
"""

import csv
import datetime
import json
from pathlib import Path

import numpy
import pandas

RUN_ROOT = Path("D:/AI_Project/03_data/model_runs")
INDEX = Path("D:/AI_Project/03_data/processed/v01_training_index")

# 1. the ten runs: seconds per picture and the gap between the last picture of one run and the first of the next
rows = []
for folder in sorted(RUN_ROOT.glob("2026-11-11_17*")):
    summary = json.loads((folder / "summary.json").read_text(encoding = "utf-8"))
    with open(folder / "steps.csv", newline = "", encoding = "utf-8") as file:
        steps = list(csv.DictReader(file))
    times = [float(step["unix_time"]) for step in steps]
    rows.append({"label": summary["label"], "pictures": len(steps), "first": times[0], "last": times[-1]})
per_picture = [(row["last"] - row["first"]) / (row["pictures"] - 1) for row in rows]
gaps = [rows[index + 1]["first"] - rows[index]["last"] for index in range(len(rows) - 1)]
print("11.11.26, ten runs [measured from steps.csv]")
print(f"   seconds per picture: median {numpy.median(per_picture):.2f}, range {min(per_picture):.2f} to {max(per_picture):.2f}")
print(f"   gap from the last picture of a run to the first picture of the next (lift, new sheet or puck, placing the tip by hand, typing the command, "
      f"loading the model, the go-ahead): median {numpy.median(gaps):.0f} s, mean {numpy.mean(gaps):.0f} s, range {min(gaps):.0f} to {max(gaps):.0f} s")
print(f"   gaps in order (s): {[round(value) for value in gaps]}")

# 2. the demonstrations: pictures and tip travel per whole-line episode, per sheet
frames = pandas.read_csv(INDEX / "frames.csv", usecols = ["episode_id", "sheet", "start_kind", "row_in_episode", "move_length_m", "action"], low_memory = False)
episodes = frames.groupby("episode_id").agg(sheet = ("sheet", "first"), start_kind = ("start_kind", "first"), pictures = ("row_in_episode", "size"),
                                            travel_cm = ("move_length_m", lambda values: 100.0 * values.fillna(0.0).sum()))
standard = episodes[episodes["start_kind"] == "STD"]
print("\ndemonstrations with the standard start [measured from frames.csv]: pictures and tip travel per episode")
travel_by_sheet = {}
for sheet, group in standard.groupby("sheet"):
    travel_by_sheet[sheet] = float(group["travel_cm"].median())
    print(f"   {sheet}: {len(group)} episodes, pictures median {group['pictures'].median():.0f} (largest {group['pictures'].max()}), "
          f"tip travel median {group['travel_cm'].median():.0f} cm")

# 3. the recording days' pace: start to start inside one sheet (the check quotes a median of 5.8 minutes)
episode_table = pandas.read_csv(INDEX / "episodes.csv", low_memory = False)
time_column = next((name for name in episode_table.columns if "start" in name and "time" in name), None) or next((name for name in episode_table.columns if name in ("started", "started_at")), None)
print(f"\nepisodes.csv time column used: {time_column}")
if time_column is not None:
    episode_table["when"] = pandas.to_datetime(episode_table[time_column], errors = "coerce")
    episode_table = episode_table.dropna(subset = ["when"]).sort_values("when")
    same_sheet = episode_table["sheet"].eq(episode_table["sheet"].shift())
    gap_minutes = episode_table["when"].diff().dt.total_seconds() / 60.0
    usable = gap_minutes[same_sheet & (gap_minutes < 30)]
    print(f"   recording: start to start inside one sheet, gaps under 30 minutes: median {usable.median():.1f} minutes over {len(usable)} gaps")

# 4. session 2 as the plan lists it, with three paces
seconds_per_picture = float(numpy.median(per_picture))
sheet_03_travel = travel_by_sheet.get("path_03", 60.0)
loop_travel = float(numpy.mean([travel_by_sheet.get("path_05", 70.0), travel_by_sheet.get("path_06", 60.0)]))
arc_travel = travel_by_sheet.get("path_02", 35.0)
blocks = [
    # name, runs, tip travel in cm per run, step in cm, extra seconds per run (phone photo and the like)
    ("0  dry run without the puck, twice (the 11.11 checklist did the same)", 2, 15.0, 0.2, 0),
    ("A  fixed 1.0 cm, 4 short runs (5.5 cm to the puck, 4 cm to the bend, 10 cm past it)", 4, 20.0, 1.0, 0),
    ("A  model's own length, 4 short runs (median asked 0.61 cm)", 4, 20.0, 0.61, 0),
    ("A  fixed 0.65 cm, 4 short runs", 4, 20.0, 0.65, 0),
    ("B  fold model without sheet 03 on sheet 03, 2 settings x 3 full runs", 6, sheet_03_travel, 0.65, 60),
    ("B  model E on two new tracks, 3 full runs each (track length taken as sheet 03's)", 6, sheet_03_travel, 0.65, 60),
    ("C  solid shape 06 and dashed shape 05, 3 full runs each, driver of 11.11 (1.0 cm)", 6, loop_travel, 1.0, 60),
    ("C  sheets 05 and 06 once each", 2, loop_travel, 1.0, 60),
    ("D  sheet 02 lying square, 2 starts x 2 runs, tip placed by hand (full runs, as the plan counts them)", 4, arc_travel, 1.0, 60),
]
print(f"\nsession 2 as planned [calculated]: {seconds_per_picture:.2f} s per picture; a full run is taken as long as the standard-start demonstrations' median tip travel")
print(f"   (sheet 03: {sheet_03_travel:.0f} cm, loops: {loop_travel:.0f} cm, sheet 02: {arc_travel:.0f} cm); 60 s more per full run for the phone photo of block E and the note of the result")
for pace_name, gap_seconds in (("gap as the fastest on 11.11 (107 s)", 107.0), ("gap as the median on 11.11 (180 s)", float(numpy.median(gaps))), ("gap as the slowest on 11.11 (275 s)", 275.0)):
    total = 0.0
    lines = []
    for name, runs, travel_cm, step_cm, extra in blocks:
        pictures = travel_cm / step_cm
        minutes = runs * (pictures * seconds_per_picture + gap_seconds + extra) / 60.0
        total += minutes
        lines.append(f"      {name}: {runs} runs x {pictures:.0f} pictures = {minutes:.0f} min")
    print(f"   {pace_name}: runs alone {total:.0f} min = {total / 60:.1f} h")
    if "median" in pace_name:
        print("\n".join(lines))
sheets_to_tape = 1 + 2 + 4 + 1
print(f"\n   on top of the runs [guess, nothing like it was timed]: switching on, the offline tests and the automatic calibration 15 min; tape marks for about "
      f"{sheets_to_tape} sheet lay-outs (sheet 03, two new tracks, four loop prints, sheet 02 square; blocks A and B share sheet 03) at 5 min each = {5 * sheets_to_tape} min; "
      f"scoring block A before block B (12 runs) 15 min; the checklist read aloud and the go for each new movement 10 min; voided runs repeated, 1 in 10: 4 runs = about 20 min")
