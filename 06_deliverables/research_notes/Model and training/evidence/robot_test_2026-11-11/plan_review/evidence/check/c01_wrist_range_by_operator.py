"""
Check of the plan review, key "evidence_check": can the wrist be held still by hand?
The critic (P4) says Emranul's wrist moves over a median range of 8.0 degrees per episode, "so it cannot be held at
13.5 by hand". The wrist checker's verdict 4 gives 8.0 for Emranul and 0.7 for Marshal. This script recounts both and
splits them by sheet, to see whether Marshal's small range comes from easy sheets only. Reads frames.csv only.
"""

import numpy
import pandas

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "operator", "sheet", "start_kind", "joint_5", "step"])
frames["wrist_deg"] = numpy.degrees(frames["joint_5"])
per_episode = frames.groupby("episode_id").agg(operator = ("operator", "first"), sheet = ("sheet", "first"), start_kind = ("start_kind", "first"),
                                               wrist_range = ("wrist_deg", lambda values: values.max() - values.min()),
                                               wrist_first = ("wrist_deg", "first"), rows = ("wrist_deg", "size"))
print("episodes:", len(per_episode))
print(per_episode.groupby("operator")["wrist_range"].describe().round(2).to_string())
print()
print("by operator and sheet: episodes, median wrist range (degrees), share of episodes with a range under 2 degrees")
table = per_episode.groupby(["operator", "sheet"]).agg(episodes = ("wrist_range", "size"), median_range = ("wrist_range", "median"),
                                                       under_2 = ("wrist_range", lambda values: (values < 2.0).mean()))
print(table.round(2).to_string())
print()
on_mark = per_episode[(per_episode["wrist_first"] > 13.0) & (per_episode["wrist_first"] < 14.5)]
print("start-mark episodes only (first wrist 13.0 to 14.5): by operator and sheet")
table = on_mark.groupby(["operator", "sheet"]).agg(episodes = ("wrist_range", "size"), median_range = ("wrist_range", "median"),
                                                   under_2 = ("wrist_range", lambda values: (values < 2.0).mean()),
                                                   under_5 = ("wrist_range", lambda values: (values < 5.0).mean()))
print(table.round(2).to_string())
