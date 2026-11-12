"""
Verifier's helper: a contact sheet of raw pictures (no overlays), so that claims can be checked by eye.
Usage: python contact_sheet.py out.png columns path1 path2 ...   (a path may carry a caption after '|')
"""

import sys
from pathlib import Path

import cv2
import numpy


def contact_sheet(entries, columns, out_path, scale = 1.0):
    tiles = []
    for entry in entries:
        path, _, caption = entry.partition("|")
        picture = cv2.imread(path)
        if picture is None:
            picture = numpy.zeros((480, 640, 3), dtype = numpy.uint8)
        if scale != 1.0:
            picture = cv2.resize(picture, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
        cv2.putText(picture, caption or Path(path).name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(picture, caption or Path(path).name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(picture)
    while len(tiles) % columns:
        tiles.append(numpy.zeros_like(tiles[0]))
    rows = [numpy.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
    cv2.imwrite(str(out_path), numpy.vstack(rows))


if __name__ == "__main__":
    contact_sheet(sys.argv[3:], int(sys.argv[2]), sys.argv[1], scale = 0.625)
