"""Extract one frame every half second from a video and tile them into contact sheets."""

import sys
from pathlib import Path

import cv2
import numpy

video_path, output_folder = Path(sys.argv[1]), Path(sys.argv[2])
output_folder.mkdir(parents = True, exist_ok = True)
capture = cv2.VideoCapture(str(video_path))
fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"fps {fps:.1f}, frames {count}, duration {count / fps:.1f} s, size {width}x{height}")

step = max(1, int(round(fps * 0.5)))
frames = []
index = 0
while True:
    ok, frame = capture.read()
    if not ok:
        break
    if index % step == 0:
        seconds = index / fps
        path = output_folder / f"t{seconds:05.1f}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        frames.append((seconds, frame))
    index += 1
capture.release()
print("frames saved:", len(frames))

# contact sheets: 6 frames per sheet, scaled so each tile is 400 px wide
tile_width = 400
scale = tile_width / width
tile_height = int(height * scale)
per_sheet = 6
for sheet_index in range(0, len(frames), per_sheet):
    group = frames[sheet_index:sheet_index + per_sheet]
    tiles = []
    for seconds, frame in group:
        tile = cv2.resize(frame, (tile_width, tile_height))
        cv2.putText(tile, f"{seconds:.1f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) < per_sheet:
        tiles.append(numpy.zeros_like(tiles[0]))
    rows = [numpy.hstack(tiles[row * 3:(row + 1) * 3]) for row in range(2)]
    sheet = numpy.vstack(rows)
    sheet_path = output_folder / f"sheet_{sheet_index // per_sheet + 1:02d}.jpg"
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheet", sheet_path.name, "covers", f"{group[0][0]:.1f}", "to", f"{group[-1][0]:.1f}", "s")
