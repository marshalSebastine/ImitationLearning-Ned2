"""
Verifier's check 19 (11.11.26): the stop pictures of the nine demonstrations whose end the analyst measured furthest
from covered (centres 13 to 18 mm apart), as the camera took them and without any overlay: does the camera's own stop
picture show the missed end? Writes look_uncovered_stop_pictures.png (to be looked at) and prints the nine.
"""

from pathlib import Path

import pandas

from contact_sheet import contact_sheet

HERE = Path(__file__).resolve().parent
frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False)
ends = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_ends_01_03.csv", "demonstration_ends_04_06.csv")], ignore_index = True)
ends["is_reliable"] = (ends["is_end_found"] & (ends["is_end_seen"] == True) & ends["line_width_mm"].between(8.0, 12.5) & (ends["fit_quality"] >= 0.8))
worst = ends[ends["is_reliable"]].sort_values("centre_gap_mm", ascending = False).head(9)
entries = []
for _, end in worst.iterrows():
    row = frames[frames["episode_id"] == end["episode_id"]].sort_values("row_in_episode").iloc[-1]
    caption = f"{end['sheet'][-2:]} {end['episode_id'].split('episode_')[1][:3]}: short {end['short_mm']:+.0f} aside {end['aside_mm']:+.0f} gap {end['centre_gap_mm']:.0f}"
    entries.append(f"D:/AI_Project/03_data/raw/main_data/{row['frame_path']}|{caption}")
    print(caption, "| the analyst's ink ahead, square mm:", end["ink_seen_ahead_mm2"], "| operator's verdict:", end["outcome"], "| score", end["score"])
contact_sheet(entries, 3, HERE / "look_uncovered_stop_pictures.png", scale = 0.625)
print("looked at by the verifier: a stub of line shows beside the puck or the block in 3 of the 9 (02-022, 02-015, 02-031); nothing of the line shows in the other 6")
