"""
Verifier: shrink the contact sheets of this folder (v02_*.png, v03b_*.png) to 1500 px width and 128 colours, so that the
evidence folder stays small; the puck's place against the line stays readable. Run after v02_contact_sheet.py and
v03b_demo_start_pictures_small.py.

Run from this folder:  python v02b_shrink_pictures.py
"""

import glob
import os

import cv2

from PIL import Image

for path in sorted(glob.glob("*.png")):
    picture = cv2.imread(path)
    height, width = picture.shape[:2]
    if width > 1500:
        picture = cv2.resize(picture, (1500, int(round(height * 1500.0 / width))), interpolation = cv2.INTER_AREA)
    image = Image.fromarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)).convert("P", palette = Image.ADAPTIVE, colors = 128)
    image.save(path, optimize = True)
    print(path, os.path.getsize(path) // 1024, "kB")
