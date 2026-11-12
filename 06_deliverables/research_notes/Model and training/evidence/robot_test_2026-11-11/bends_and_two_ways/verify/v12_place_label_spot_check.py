"""
Verifier's spot check of the kinds of place: do pictures labelled "hairpin", "sharp bend" and "straight" show the puck
at such a place? The labels rest on the audit's puck positions along the line (inputs/audit_line_positions.csv), whose
making is not in this folder, so a look at the pictures is the only independent check.

Draws pictures at random (fixed seed) per kind of place and writes one contact sheet per kind. Read-only on the data.
"""

from pathlib import Path

import cv2
import numpy

import verify_common

SPOT_PARAMETERS = {
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "per_place": 10,
    "columns": 5,
    "tile_width": 384,
    "tile_height": 288,
    "seed": 111126
}


if __name__ == "__main__":
    settings = dict(SPOT_PARAMETERS)
    frames = verify_common.demonstration_rows()
    generator = numpy.random.default_rng(settings["seed"])
    for place in ["hairpin", "sharp bend", "straight"]:
        pool = frames[(frames["place"] == place) & ~frames["is_stop"]]
        chosen = pool.iloc[generator.choice(len(pool), settings["per_place"], replace = False)]
        tiles = []
        for _, row in chosen.iterrows():
            picture = cv2.imread(str(settings["raw_folder"] / row["frame_path"]), cv2.IMREAD_COLOR)
            tile = cv2.resize(picture, (settings["tile_width"], settings["tile_height"]), interpolation = cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (settings["tile_width"], 22), (0, 0, 0), -1)
            text = f"{row['sheet'][-2:]} arc {row['arc_cm']:.1f} of {row['line_cm']:.0f} {'A>B' if row['travel_sign'] > 0 else 'B>A'} step {row['step_cm']:.2f} tw {row['twist_deg']:+.1f}"
            cv2.putText(tile, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows = [numpy.hstack(tiles[start:start + settings["columns"]]) for start in range(0, len(tiles), settings["columns"])]
        output_path = Path(__file__).resolve().parent / f"v12_place_{place.replace(' ', '_')}.jpg"
        cv2.imwrite(str(output_path), numpy.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(output_path)
