"""
Verifier: is the puck 2.4 cm (config.PUCK_PARAMETERS) or 4 cm (the brief) across? From pictures alone: the width of the
red blob against the width of the 1-cm line near it, in the first pictures of six robot runs. The puck's top is 2.9 cm
nearer to the camera than the paper, so it looks somewhat larger than it is; a 4-cm puck would look more than four
line widths across.

The line's width is twice the largest distance of an ink pixel from the ink's rim (so a slanted line does not read
wider), in a window of 140 px around the puck.

Run from this folder:  python v19_puck_size.py > v19_puck_size.out.txt
"""

import os

import cv2
import numpy

RUNS = "D:/AI_Project/03_data/model_runs"
FOLDERS = ["2026-11-11_172421_E_sheet01_run1", "2026-11-11_173036_E_sheet02_run2", "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1",
           "2026-11-11_174448_E_sheet05_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie"]
ratios = []
for folder in FOLDERS:
    picture = cv2.imread(os.path.join(RUNS, folder, "frames", "000.jpg"))
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    red = (((hue <= 15) | (hue >= 160)) & (saturation >= 80) & (value >= 60)).astype(numpy.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, numpy.ones((7, 7), numpy.uint8))
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(red)
    blob = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    left, top, width, height, area = (int(statistics[blob, key]) for key in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
    ink = ((hue >= 90) & (hue <= 125) & (saturation >= 110) & (value <= 185)).astype(numpy.uint8)
    ink[330:, :] = 0
    window = ink[max(0, top - 140):top + height, max(0, left - 140):left + width + 140]
    line_width = 2.0 * float(cv2.distanceTransform(window, cv2.DIST_L2, 5).max()) if window.any() else float("nan")
    ratios.append(width / line_width)
    print(f"{folder}: red blob {width} px wide, {height} px high, {area} px; the line near it {line_width:.0f} px wide; puck width / line width = {width / line_width:.2f}")
print(f"median ratio {numpy.median(ratios):.2f}: with the top 2.9 cm nearer to the camera a 2.4-cm puck looks roughly 2.6 to 3 line widths across, a 4-cm puck 4.4 to 5")
