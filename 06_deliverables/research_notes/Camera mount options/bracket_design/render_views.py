"""Render the exported STL from four directions, for a visual check."""

import pathlib
import struct

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_binary_stl(path: pathlib.Path) -> numpy.ndarray:
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    records = numpy.frombuffer(data[84:84 + count * 50], dtype = numpy.dtype([
        ("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]))
    return records["vertices"].astype(float)


triangles = read_binary_stl(pathlib.Path("ned2_camera_bracket_F.stl"))
print("triangles", len(triangles))
views = [("front (+x, camera side)", 0, 0), ("side (+z)", 0, 90), ("back (-x)", 0, 180), ("isometric", 30, 45)]
figure = pyplot.figure(figsize = (14, 8))
for index, (title, elevation, azimuth) in enumerate(views, start = 1):
    axis = figure.add_subplot(1, 4, index, projection = "3d")
    collection = Poly3DCollection(triangles, facecolor = "lightsteelblue", edgecolor = "none", alpha = 1.0)
    axis.add_collection3d(collection)
    axis.set_xlim(-20, 60); axis.set_ylim(-20, 80); axis.set_zlim(-40, 40)
    axis.set_box_aspect((80, 100, 80))
    axis.view_init(elev = elevation, azim = azimuth)
    axis.set_title(title); axis.set_xlabel("x fwd"); axis.set_ylabel("y up"); axis.set_zlabel("z")
figure.tight_layout()
figure.savefig("ned2_camera_bracket_F_views.png", dpi = 110)
