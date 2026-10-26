"""Slice a binary STL (metres) with planes z = const and print/plot the section polygons in mm.
Usage: py -3.13 -I stl_slices.py file.stl z1,z2,... out.png [scale]"""
import sys, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]))
    return rec['v'].astype(np.float64)

def slice_z(tri, z):
    segs = []
    for t in tri:
        pts = []
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            if (a[2] - z) * (b[2] - z) < 0:
                f = (z - a[2]) / (b[2] - a[2])
                pts.append(a[:2] + f * (b[:2] - a[:2]))
            elif abs(a[2] - z) < 1e-9:
                pts.append(a[:2])
        if len(pts) >= 2:
            segs.append((pts[0], pts[1]))
    return segs

path = sys.argv[1]
zs = [float(z) for z in sys.argv[2].split(',')]
out = sys.argv[3]
SC = float(sys.argv[4]) if len(sys.argv) > 4 else 1000.0
tri = read_stl(path) * SC
n = len(zs)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = np.array(axes).reshape(-1)
for ax, z in zip(axes, zs):
    segs = slice_z(tri, z)
    allp = np.array([p for s in segs for p in s]) if segs else np.zeros((0, 2))
    for a, b in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], 'k-', lw=0.8)
    ax.set_aspect('equal'); ax.grid(True, lw=0.3)
    ax.set_title(f'z = {z} mm  ({len(segs)} segs)')
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12)
    if len(allp):
        r = np.hypot(allp[:, 0], allp[:, 1])
        ang = np.degrees(np.arctan2(allp[:, 1], allp[:, 0]))
        print(f'z={z:6.2f}: segs={len(segs):4d} x[{allp[:,0].min():6.2f},{allp[:,0].max():6.2f}] y[{allp[:,1].min():6.2f},{allp[:,1].max():6.2f}] r[{r.min():5.2f},{r.max():5.2f}]')
        # angular profile of the outer radius: for 24 bins print max r
        bins = np.linspace(-180, 180, 25)
        prof = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ang >= lo) & (ang < hi)
            prof.append(f'{r[m].max():4.1f}' if m.any() else ' -- ')
        print('     outer r by 15deg bins from -180:', ' '.join(prof))
        # inner radius profile
        prof = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ang >= lo) & (ang < hi)
            prof.append(f'{r[m].min():4.1f}' if m.any() else ' -- ')
        print('     inner r by 15deg bins          :', ' '.join(prof))
for ax in axes[n:]:
    ax.axis('off')
fig.tight_layout(); fig.savefig(out, dpi=90)
print('written', out)
