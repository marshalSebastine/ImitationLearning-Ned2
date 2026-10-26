"""Hand link STL: print the vertices that lie inside the cylinder (r < 10.4) between z = 9 and 28.6 mm, i.e. the slot
and the bore, grouped by z; and the angular gap in the outer surface per slice (polar profile)."""
import sys, struct
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]))
    return rec['v'].astype(np.float64)
tri = read_stl(sys.argv[1]) * 1000.0
v = np.unique(tri.reshape(-1, 3).round(4), axis=0)
r = np.hypot(v[:, 0], v[:, 1]); az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
m = (v[:, 2] > 9.0) & (v[:, 2] < 28.6) & (r > 6.6) & (r < 10.45)
print('vertices between the bore and the outer surface, z 9..28.6 (slot walls):')
for p, rr, a in sorted(zip(v[m].tolist(), r[m], az[m]), key=lambda t: (t[0][2], t[2])):
    print(f'   z={p[2]:6.2f}  x={p[0]:6.2f}  y={p[1]:6.2f}   r={rr:5.2f}  az={a:7.1f}')
# slices through triangles for z in list: polar outer profile with 5 deg bins
def slice_z(tri, z):
    pts = []
    for t in tri:
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            if (a[2] - z) * (b[2] - z) < 0:
                f = (z - a[2]) / (b[2] - a[2]); pts.append(a[:2] + f * (b[:2] - a[:2]))
    return np.array(pts)
for z in (9.5, 11.0, 12.0, 13.0, 14.5, 18.0, 22.0, 26.0, 27.2, 28.0):
    p = slice_z(tri, z)
    rr = np.hypot(p[:, 0], p[:, 1]); aa = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    outer = rr > 7.0
    bins = np.arange(-180, 181, 10)
    prof = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = outer & (aa >= lo) & (aa < hi)
        prof.append(f'{rr[mm].min():4.1f}' if mm.any() else ' .. ')
    print(f'z={z:5.1f} outer-surface min r per 10deg bin from -180: ' + ' '.join(prof))
    inner = rr <= 7.0
    print(f'        inner (bore) r range: {rr[inner].min():.2f}..{rr[inner].max():.2f}' if inner.any() else '        no bore here')
