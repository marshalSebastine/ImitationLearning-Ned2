"""Render the exported STL from four directions, for a visual check."""

import pathlib
import struct
import sys

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_binary_stl(path: pathlib.Path) -> numpy.ndarray:
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    records = numpy.frombuffer(data[84:84 + count * 50], dtype = numpy.dtype([
        ("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]))
    return records["vertices"].astype(float)


# F by default; "F2" on the command line renders the corrective version of 2.11.26.
STEM = f"ned2_camera_bracket_{sys.argv[1] if len(sys.argv) > 1 else 'F'}"
triangles = read_binary_stl(pathlib.Path(f"{STEM}.stl"))
print("triangles", len(triangles))
views = [("front (+x, camera side)", 0, 0), ("side (+z)", 0, 90), ("back (-x)", 0, 180), ("isometric", 30, 45)]
figure = pyplot.figure(figsize = (14, 8))
for index, (title, elevation, azimuth) in enumerate(views, start = 1):
    axis = figure.add_subplot(1, 4, index, projection = "3d")
    collection = Poly3DCollection(triangles, facecolor = "lightsteelblue", edgecolor = "none", alpha = 1.0)
    axis.add_collection3d(collection)
    axis.set_xlim(-20, 80); axis.set_ylim(-20, 80); axis.set_zlim(-50, 50)
    axis.set_box_aspect((100, 100, 100))
    axis.view_init(elev = elevation, azim = azimuth)
    axis.set_title(title); axis.set_xlabel("x fwd"); axis.set_ylabel("y up"); axis.set_zlabel("z")
figure.tight_layout()
figure.savefig(f"{STEM}_views.png", dpi = 110)
