"""Measure how wide and how far the forearm camera sees, from the scripted-push pictures."""

import json
import sys

import numpy
from PIL import Image

folder = "D:/AI_Project/03_data/raw/workspace/push_straight/"
steps = json.load(open(folder + "2026-10-25_1430_steps.json", encoding = "utf-8"))

# Line width per row in the start picture: the black line is 10 mm wide, so dark pixels per row
# give pixels per centimetre at that row, and 640 px divided by that is the visible table width.
start = numpy.asarray(Image.open(folder + "2026-10-25_1430_step_00.jpg").convert("L"), dtype = float)
print("start picture", start.shape)
for row in [5, 40, 80, 120, 160, 200, 240, 280, 320]:
    dark = numpy.where(start[row] < 90)[0]
    if dark.size:
        width_px = dark.max() - dark.min() + 1
        print(f"row {row:3d}: line {width_px:3d} px wide -> {640 / width_px * 1.0:5.1f} cm visible across, "
              f"line centre at column {int(dark.mean())}")

# Wood fraction in the top rows of each step picture: the paper's far edge lies at x = 0.42 m and the
# table edge at about 0.44 m, so the first step whose top rows show wood tells how far ahead the
# camera sees.
print("\nstep  tip_x  puck_x  wood fraction in top 30 rows")
for step in steps:
    image = numpy.asarray(Image.open(folder + f"2026-10-25_1430_step_{step['step']:02d}.jpg").convert("RGB"), dtype = float)
    top = image[:30]
    red, green, blue = top[..., 0], top[..., 1], top[..., 2]
    wood = (red > green + 15) & (green > blue + 10) & (red - blue > 40)
    print(f"{step['step']:4d}  {step['x']:.3f}  {step['x'] + 0.0185:.3f}  {wood.mean():.2f}")
