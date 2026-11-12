"""
Review check (reads only): how close did the ten robot runs of 11.11.26 come to today's stall rule? For every 30
pictures in a row: the diagonal of the box round the tips and how far the forearm turned. [calculated from the logs]
"""

import glob
import json
import math
from pathlib import Path

import numpy
import pandas

folders = sorted(glob.glob("D:/AI_Project/03_data/model_runs/2026-11-11_17*"))
print("stall rule: 30 pictures in a row within 3.0 cm AND the forearm turned less than 15 degrees")
for folder in folders:
    steps = pandas.read_csv(Path(folder) / "steps.csv")
    steps = steps[steps["action"] != "stop"]
    tips = steps[["tip_x", "tip_y"]].to_numpy()
    twist = numpy.degrees(steps["joint_4"].to_numpy())
    name = Path(folder).name[18:]
    if len(steps) < 30:
        print(f"{name:28s} fewer than 30 pictures")
        continue
    smallest = None
    for start in range(0, len(steps) - 29):
        window = tips[start:start + 30]
        extent = 100 * math.hypot(window[:, 0].max() - window[:, 0].min(), window[:, 1].max() - window[:, 1].min())
        turned = twist[start:start + 30].max() - twist[start:start + 30].min()
        if smallest is None or extent < smallest[0]:
            smallest = (extent, turned, start)
    print(f"{name:28s} smallest box over 30 pictures: {smallest[0]:5.1f} cm (pictures {smallest[2]} to {smallest[2] + 29}), the forearm turned {smallest[1]:5.1f} degrees there; "
          f"with 0.65 cm steps on the same path that box would be about {0.65 * smallest[0]:4.1f} cm")
