"""
The verifier's own estimate of how much of the picture the robot's own parts cover, from the cached 160 x 120 pictures.

The median picture over 1,500 random cached pictures keeps what never moves (the block, the rod, the jaws) sharp and
blurs the table and the sheets into a pale ground. Own rule: a pixel of the median picture belongs to the robot when it
is dark (grey under 95) or strongly blue (hue 95 to 130 of OpenCV's 180, saturation over 110). A first try with the
spread of each pixel over the pictures failed (the lighting changes between sessions move every pixel).

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v11_robot_parts.py > v11_robot_parts.out.txt
"""

from pathlib import Path

import cv2
import numpy

VERIFY_FOLDER = Path(__file__).resolve().parent
pictures = numpy.load("D:/AI_Project/03_data/processed/v01_training_index/images_wide_160.npy", mmap_mode = "r")
generator = numpy.random.default_rng(5)
chosen = numpy.sort(generator.choice(len(pictures), 1500, replace = False))
median = numpy.median(numpy.asarray(pictures[chosen]), axis = 0).astype(numpy.uint8)          # RGB
cv2.imwrite(str(VERIFY_FOLDER / "v11_median_picture.png"), cv2.resize(cv2.cvtColor(median, cv2.COLOR_RGB2BGR), (640, 480), interpolation = cv2.INTER_NEAREST))
hsv = cv2.cvtColor(median, cv2.COLOR_RGB2HSV)
grey = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
is_part = (grey < 95) | ((hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 110))
count, labels, stats, _ = cv2.connectedComponentsWithStats(is_part.astype(numpy.uint8), connectivity = 8)
# keep the blobs that reach the lower third of the picture (the robot's parts all do)
kept = numpy.zeros_like(is_part)
for label in range(1, count):
    top, height = stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_HEIGHT]
    if top + height >= 100 and stats[label, cv2.CC_STAT_AREA] >= 20:
        kept |= labels == label
rows = numpy.nonzero(kept.any(axis = 1))[0]
print(f"robot's own parts in the median picture: {100 * kept.mean():.1f} % of the picture; highest row {rows.min()} of 120 (= row {4 * rows.min()} of 480);"
      f" share of rows 80 to 120 (320 to 480): {100 * kept[80:].mean():.1f} %")
overlay = cv2.resize(cv2.cvtColor(median, cv2.COLOR_RGB2BGR), (640, 480), interpolation = cv2.INTER_NEAREST)
overlay[cv2.resize(kept.astype(numpy.uint8), (640, 480), interpolation = cv2.INTER_NEAREST) > 0] = (0, 255, 255)
cv2.imwrite(str(VERIFY_FOLDER / "v11_robot_parts_mask.png"), overlay)
