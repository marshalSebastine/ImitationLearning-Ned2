"""
Verifier's check 13 (11.11.26): the sideways offsets of the analyst's sampled demonstrations on sheets 01 (straight)
and 02 (arc), where the line never comes near itself, measured again with my own one-picture measure
(centring_tools.py: puck and line from the same picture) and laid beside the analyst's 12-picture offsets
(../demonstration_centring_01.csv and _02.csv). Writes verify_demonstration_centring.csv.
"""

import json
from pathlib import Path

import cv2
import numpy
import pandas

import centring_tools
import my_measure

cv2.setNumThreads(2)
HERE = Path(__file__).resolve().parent
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False)
shifts = json.loads((HERE.parent / "camera_shift.json").read_text(encoding = "utf-8"))
analyst = pandas.concat([pandas.read_csv(HERE.parent / f"demonstration_centring_{number:02d}.csv") for number in (1, 2)], ignore_index = True)
rows = []
for episode_id, part in analyst.groupby("episode_id", sort = False):
    steps = frames[frames["episode_id"] == episode_id].sort_values("row_in_episode").reset_index(drop = True)
    period = part["camera_period"].iloc[0]
    camera = my_measure.shifted_camera(shifts[period]["shift_px"], shifts[period]["roll_deg"])
    for _, chosen in part.iterrows():
        row = steps.iloc[int(chosen["step"])]
        joints = [row[f"joint_{index}"] for index in range(1, 7)]
        found = centring_tools.same_picture_offset(cv2.imread(str(RAW_FOLDER / row["frame_path"])), joints, camera)
        entry = {"episode_id": episode_id, "sheet": chosen["sheet"], "step": int(chosen["step"]), "analyst_offset_mm": chosen["offset_mm"],
                 "analyst_single_offset_mm": chosen["single_offset_mm"], "analyst_puck_x": chosen["puck_x"], "analyst_puck_y": chosen["puck_y"]}
        if found is not None:
            entry.update({"my_offset_mm": found["offset_mm"], "my_followed_back_mm": found["extrapolated_mm"], "my_bend_per_m": found["bend_per_m"],
                          "my_seen_mm": found["seen_mm"], "my_puck_x": found["puck_xy"][0], "my_puck_y": found["puck_xy"][1]})
        rows.append(entry)
table = pandas.DataFrame(rows)
table.to_csv(HERE / "verify_demonstration_centring.csv", index = False, float_format = "%.3f")

puck_gap = 1000.0 * numpy.hypot(table["my_puck_x"] - table["analyst_puck_x"], table["my_puck_y"] - table["analyst_puck_y"])
print(f"pictures {len(table)} of {table['episode_id'].nunique()} demonstrations (sheets 01 and 02); my puck against the analyst's: median {puck_gap.median():.1f} mm apart, 90th percentile {puck_gap.quantile(0.9):.1f} mm")
usable = table[table["my_offset_mm"].notna() & table["analyst_offset_mm"].notna()
               & ((table["my_followed_back_mm"] <= 10.0) | ((table["my_followed_back_mm"] <= 35.0) & (table["my_bend_per_m"] < 8.0) & (table["my_seen_mm"] >= 30.0)))]
difference = usable["my_offset_mm"].abs() - usable["analyst_offset_mm"].abs()
print(f"pictures where my one-picture offset is usable (line beside the puck, or a nearly straight line followed back at most 35 mm): {len(usable)}")
print(f"  size of mine minus size of the analyst's 12-picture offset: median {difference.median():+.1f} mm, median size {difference.abs().median():.1f} mm,"
      f" 90th percentile of the size {difference.abs().quantile(0.9):.1f} mm; over 5 mm apart in {int((difference.abs() > 5).sum())}")
for label, sizes in (("mine", usable["my_offset_mm"].abs()), ("the analyst's 12-picture", usable["analyst_offset_mm"].abs()), ("the analyst's one-picture", usable["analyst_single_offset_mm"].abs().dropna())):
    print(f"  {label}: {len(sizes)} pictures, median {sizes.median():.1f} mm, within 7 mm {100 * (sizes <= 7).mean():.0f} %, over 12 mm {100 * (sizes > 12).mean():.0f} %, over 17 mm {100 * (sizes > 17).mean():.0f} %")
for sheet, part in usable.groupby("sheet"):
    difference = part["my_offset_mm"].abs() - part["analyst_offset_mm"].abs()
    print(f"  {sheet}: {len(part)} pictures; mine median {part['my_offset_mm'].abs().median():.1f} mm, the analyst's {part['analyst_offset_mm'].abs().median():.1f} mm; difference median size {difference.abs().median():.1f} mm")
big = usable[usable["analyst_offset_mm"].abs() > 12.0]
print(f"  of the {len(big)} usable pictures the analyst puts over 12 mm, mine puts {int((big['my_offset_mm'].abs() > 12).sum())} over 12 mm and {int((big['my_offset_mm'].abs() > 9).sum())} over 9 mm")
small = usable[usable["analyst_offset_mm"].abs() <= 7.0]
print(f"  of the {len(small)} usable pictures the analyst puts within 7 mm, mine puts {int((small['my_offset_mm'].abs() <= 7).sum())} within 7 mm and {int((small['my_offset_mm'].abs() <= 10).sum())} within 10 mm")
