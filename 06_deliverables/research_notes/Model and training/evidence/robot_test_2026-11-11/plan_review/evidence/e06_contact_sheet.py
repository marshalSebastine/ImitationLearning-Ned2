"""
Plan review, key "evidence": a small contact sheet of logged pictures of one run, to look at by eye.
Usage: python e06_contact_sheet.py <run folder name> <first> <last> <output name>
Reads pictures only; writes one JPEG into this folder.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

run_folder = Path("D:/AI_Project/03_data/model_runs") / sys.argv[1] / "frames"
first, last = int(sys.argv[2]), int(sys.argv[3])
names = [f"{number:03d}.jpg" for number in range(first, last + 1)]
tiles = []
for name in names:
    picture = Image.open(run_folder / name).convert("RGB").resize((480, 360))
    ImageDraw.Draw(picture).text((8, 6), name[:3], fill = (255, 255, 0))
    tiles.append(picture)
columns = 4
rows = (len(tiles) + columns - 1) // columns
sheet = Image.new("RGB", (480 * columns, 360 * rows), (255, 255, 255))
for index, tile in enumerate(tiles):
    sheet.paste(tile, (480 * (index % columns), 360 * (index // columns)))
sheet.save(Path(__file__).resolve().parent / sys.argv[4], quality = 85)
print("written", sys.argv[4], sheet.size)
