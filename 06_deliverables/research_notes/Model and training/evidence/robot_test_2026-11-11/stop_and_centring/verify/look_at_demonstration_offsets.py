"""
Verifier's check 6 (11.11.26): look at demonstration pictures the analyst measured as far off the line and as centred.

A seeded random sample is drawn from the analyst's tables (../demonstration_centring_0N.csv): standard starts after
the first 15 pictures, by the size of the analyst's offset. Each picture is shown with MY fit of the puck's foot (cyan
circle, 12 mm radius on the table, top-face fit of my_measure.py) and nothing else, so that the eye can judge whether
the foot stands on the line. The caption gives the analyst's offset and my one-picture offset where I have one.
Writes look_far_off.png, look_off.png, look_centred.png and look_at_demonstration_offsets.csv.
"""

import glob
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
SEED = 20261111

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False)
shifts = json.loads((HERE.parent / "camera_shift.json").read_text(encoding = "utf-8"))
table = pandas.concat([pandas.read_csv(path) for path in sorted(glob.glob(str(HERE.parent / "demonstration_centring_0*.csv")))], ignore_index = True)
table = table[(table["start_kind"] == "STD") & (table["step"] > 15) & table["offset_mm"].notna()]
groups = {"far_off": table[table["offset_mm"].abs() > 17.0], "off": table[table["offset_mm"].abs().between(12.0, 17.0)], "centred": table[table["offset_mm"].abs() < 3.0]}
counts = {"far_off": 15, "off": 9, "centred": 9}
rows = []
for name, group in groups.items():
    sample = group.sample(n = counts[name], random_state = SEED).sort_values(["sheet", "episode_id", "step"])
    tiles = []
    for _, chosen in sample.iterrows():
        row = frames[(frames["episode_id"] == chosen["episode_id"]) & (frames["row_in_episode"] == chosen["step"])].iloc[0]
        period = chosen["camera_period"]
        camera = my_measure.shifted_camera(shifts[period]["shift_px"], shifts[period]["roll_deg"])
        joints = [row[f"joint_{index}"] for index in range(1, 7)]
        picture = cv2.imread(str(RAW_FOLDER / row["frame_path"]))
        found = centring_tools.same_picture_offset(picture, joints, camera)
        puck = my_measure.puck_foot(picture, joints, camera)
        shown = picture.copy()
        if puck is not None:
            cv2.polylines(shown, [numpy.round(my_measure.foot_circle_px(puck["xy"], joints, camera)).astype(numpy.int32)], True, (255, 255, 0), 1)
        mine = "none" if found is None else f"{found['offset_mm']:.0f} (back {found['extrapolated_mm']:.0f})"
        caption = f"{chosen['sheet'][-2:]} {chosen['episode_id'].split('episode_')[1][:3]} pic {int(chosen['step'])}: analyst {chosen['offset_mm']:+.0f}, mine {mine}"
        cv2.putText(shown, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(shown, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(shown)
        rows.append({"group": name, "episode_id": chosen["episode_id"], "step": int(chosen["step"]), "analyst_offset_mm": chosen["offset_mm"],
                     "my_offset_mm": None if found is None else found["offset_mm"], "my_followed_back_mm": None if found is None else found["extrapolated_mm"],
                     "my_nearest_ink_mm": None if found is None else found["nearest_ink_mm"]})
    while len(tiles) % 3:
        tiles.append(numpy.zeros_like(tiles[0]))
    cv2.imwrite(str(HERE / f"look_{name}.png"), numpy.vstack([numpy.hstack(tiles[index:index + 3]) for index in range(0, len(tiles), 3)]))
result = pandas.DataFrame(rows)
result.to_csv(HERE / "look_at_demonstration_offsets.csv", index = False, float_format = "%.1f")
pandas.set_option("display.width", 250)
print(f"standard starts after picture 15 with an offset from the analyst: {len(table)} pictures; over 17 mm {len(groups['far_off'])}, 12 to 17 mm {len(groups['off'])}, under 3 mm {len(groups['centred'])}")
print(result.assign(episode = result["episode_id"].str[-40:]).drop(columns = "episode_id").to_string(index = False))
