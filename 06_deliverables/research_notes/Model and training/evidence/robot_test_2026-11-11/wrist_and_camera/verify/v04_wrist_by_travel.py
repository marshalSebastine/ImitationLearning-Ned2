"""
Verifier, check 4: how soon after the start does the demonstrated wrist leave 13.5 degrees? Measured from frames.csv.
Independent of any matching of places: rows of the 149 start-mark episodes, binned by the tip's travelled distance.
    D:/AI_Project/.venv/Scripts/python.exe v04_wrist_by_travel.py
"""

import numpy
import pandas

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "tip_x_m", "tip_y_m", "joint_4", "joint_5"])
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
frames["wrist_deg"] = numpy.degrees(frames["joint_5"])
frames["twist_deg"] = numpy.degrees(frames["joint_4"])
step = numpy.hypot(frames.groupby("episode_id")["tip_x_m"].diff().fillna(0.0), frames.groupby("episode_id")["tip_y_m"].diff().fillna(0.0))
frames["travel_cm"] = 100 * step.groupby(frames["episode_id"]).cumsum()
frames["is_far"] = (frames["wrist_deg"] - 13.5).abs() > 5.7
start_mark = frames[frames["sheet_orientation"].str.contains("START MARK", na = False)].copy()
start_mark["travel_bin"] = pandas.cut(start_mark["travel_cm"], [0, 5, 10, 15, 20, 30, 40, 60, 120], include_lowest = True)
pandas.set_option("display.width", 220)
print("start-mark episodes: share of rows (%) with the wrist over 5.7 degrees from 13.5, by sheet and travelled distance (cm)")
print((100 * start_mark.pivot_table(index = "sheet", columns = "travel_bin", values = "is_far", aggfunc = "mean", observed = True)).round(1).to_string())
print("\nrows")
print(start_mark.pivot_table(index = "sheet", columns = "travel_bin", values = "is_far", aggfunc = "size", observed = True).to_string())
print("\nmedian |joint 4| (degrees) in the same cells: how far the forearm is already twisted")
print(start_mark.assign(a = start_mark["twist_deg"].abs()).pivot_table(index = "sheet", columns = "travel_bin", values = "a", aggfunc = "median", observed = True).round(0).to_string())
# first row at which each episode's wrist leaves 13.5 by more than 5.7 degrees
first_far = start_mark[start_mark["is_far"]].groupby("episode_id")["travel_cm"].min()
episodes = start_mark.groupby("episode_id")["sheet"].first().to_frame().join(first_far.rename("first_far_travel_cm"))
print("\ntravel at which an episode's wrist first leaves 13.5 by more than 5.7 degrees (start-mark episodes):")
print(episodes.groupby("sheet")["first_far_travel_cm"].agg(episodes = "size", never_left = lambda v: int(v.isna().sum()), least = "min", median = "median").round(1).to_string())
