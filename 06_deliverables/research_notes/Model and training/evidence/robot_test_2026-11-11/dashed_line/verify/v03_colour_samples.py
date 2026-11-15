"""
Verifier check 3 (helper): colour statistics of a few pictures, to choose my own ink, paper and puck thresholds.

Run:  D:/AI_Project/.venv/Scripts/python.exe v03_colour_samples.py
"""

from pathlib import Path

import cv2
import numpy

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
for folder_name, numbers in (("2026-11-11_174936_E_sheet06_run1", (0, 10, 20, 30)), ("2026-11-11_174448_E_sheet05_run1", (0, 30, 60))):
    for number in numbers:
        picture = cv2.imread(str(RUNS_FOLDER / folder_name / "frames" / f"{number:03d}.jpg"))
        hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
        hue, saturation, value = [hsv[:, :, channel].astype(int) for channel in range(3)]
        blue, green, red = [picture[:, :, channel].astype(int) for channel in range(3)]
        print(f"\n{folder_name[-12:]} picture {number}: joint histogram of (hue bin of 10, value bin of 40), share of pixels in % (rows hue, columns value)")
        histogram, _, _ = numpy.histogram2d(hue.ravel(), value.ravel(), bins = [numpy.arange(0, 190, 10), numpy.arange(0, 281, 40)])
        print(numpy.round(100.0 * histogram / histogram.sum(), 1))
        is_bluish_dark = (hue >= 90) & (hue <= 125) & (value >= 60) & (value <= 200) & (numpy.arange(480)[:, None] < 300)
        if is_bluish_dark.sum() > 50:
            print("  dark bluish pixels above row 300:", int(is_bluish_dark.sum()),
                  "hue 5/50/95 %:", numpy.percentile(hue[is_bluish_dark], [5, 50, 95]),
                  "sat:", numpy.percentile(saturation[is_bluish_dark], [5, 50, 95]),
                  "val:", numpy.percentile(value[is_bluish_dark], [5, 50, 95]),
                  "B-R:", numpy.percentile((blue - red)[is_bluish_dark], [5, 50, 95]))
        is_bright = (value > 200) & (numpy.arange(480)[:, None] < 300)
        print("  bright pixels above row 300:", int(is_bright.sum()), "hue:", numpy.percentile(hue[is_bright], [5, 50, 95]),
              "sat:", numpy.percentile(saturation[is_bright], [5, 50, 95]))
        is_red = (red > 120) & (red > 1.7 * green) & (red > 1.4 * blue)
        rows, columns = numpy.nonzero(is_red)
        if len(rows):
            print(f"  red pixels: {len(rows)}, centre ({columns.mean():.0f}, {rows.mean():.0f}), columns {columns.min()} to {columns.max()}, rows {rows.min()} to {rows.max()}")
