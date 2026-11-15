"""
Verifier 16: a look at the puck tracker's distances. Twelve demonstration pictures drawn at random (seed 16): six where the
cached tip-to-puck distance is over 4 cm and six where it is under 2.4 cm, with the tracker's puck centre marked.
Is the tip really away from the puck when the tracker says so? Reads only; writes v16_tracker_spot_check.png.
"""
from pathlib import Path
import cv2
import numpy
import pandas
from vcommon import demos

RAW = Path("D:/AI_Project/03_data/raw/main_data")
OUT = Path(__file__).resolve().parent
frames = demos()
puck = pandas.read_csv(OUT.parent / "puck_positions_demonstrations.csv")
table = frames.merge(puck, on = "frame_index", how = "inner")
table = table[table["puck_x_cm"].notna() & (table["action"] != "stop")].copy()
table["distance"] = numpy.hypot(table["puck_x_cm"] - table["tip_x_cm"], table["puck_y_cm"] - table["tip_y_cm"])
rng = numpy.random.default_rng(16)
rows = []
for label, part in (("far", table[table["distance"] > 4.0]), ("near", table[table["distance"] < 2.4])):
    chosen = part.iloc[rng.choice(len(part), size = 6, replace = False)]
    tiles = []
    for _, r in chosen.iterrows():
        picture = cv2.imread(str(RAW / r["frame_path"]))
        cv2.circle(picture, (int(r["puck_u"]), int(r["puck_v"])), 6, (0, 255, 0), 2)
        picture = cv2.resize(picture, (320, 240), interpolation = cv2.INTER_AREA)
        cv2.rectangle(picture, (0, 0), (320, 18), (255, 255, 255), -1)
        cv2.putText(picture, f"{label} {r['distance']:.1f} cm {r['sheet'][-2:]} row {int(r['row_in_episode'])} len {r['len_cm']:.2f} tw {r['twist']:+.1f} lean {r['rod_from_vertical_deg']:.0f}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(picture)
    rows.append(numpy.hstack(tiles))
cv2.imwrite(str(OUT / "v16_tracker_spot_check.png"), numpy.vstack(rows))
print("far share %.3f, near share %.3f" % ((table["distance"] > 4.0).mean(), (table["distance"] < 2.4).mean()))
