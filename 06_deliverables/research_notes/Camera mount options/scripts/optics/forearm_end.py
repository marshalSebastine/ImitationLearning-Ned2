"""Forearm mesh near the wrist: extents in the forearm frame (joint-5 origin at z = 0.17 m; joint-5 axis = forearm +y)."""
import sys, struct, numpy as np
path = sys.argv[1]
with open(path, "rb") as f:
    f.read(80); n = struct.unpack("<I", f.read(4))[0]
    data = np.frombuffer(f.read(n*50), dtype=np.dtype([("n","<3f4"),("v","<9f4"),("a","<u2")]))
v = data["v"].reshape(-1, 3).astype(float) * 1000.0   # mm
print("forearm bbox mm: x", v[:,0].min().round(1), v[:,0].max().round(1), " y", v[:,1].min().round(1), v[:,1].max().round(1), " z", v[:,2].min().round(1), v[:,2].max().round(1))
# slices along z (the forearm's long axis); the wrist axis is at z = 170 mm
for z0 in (100, 120, 140, 150, 160, 170, 180, 190, 200):
    m = np.abs(v[:,2] - z0) < 3
    if m.sum() < 5: print(f" z={z0:3d}: no vertices"); continue
    print(f" z={z0:3d} mm: x [{v[m,0].min():6.1f}, {v[m,0].max():6.1f}]  y [{v[m,1].min():6.1f}, {v[m,1].max():6.1f}]  ({m.sum()} verts)")
