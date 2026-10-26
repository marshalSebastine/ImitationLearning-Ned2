"""Wrist link STL (metres, mesh frame): convert to link frame, then about the joint-6 (rod) axis report the housing's
radial extent versus azimuth for vertices within a band of heights above the hand face. Also the hand STL slot geometry."""
import sys, struct
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]))
    return rec['v'].astype(np.float64)

wrist = read_stl(sys.argv[1]) * 1000.0   # mm, mesh frame
v = wrist.reshape(-1, 3)
# mesh -> link: x_link = y_mesh, y_link = -x_mesh, z_link = z_mesh
link = np.c_[v[:, 1], -v[:, 0], v[:, 2]]
print('link-frame bbox', link.min(0).round(2), link.max(0).round(2))
# rod axis: x = 9.25, z = 0 ; tool direction = -y ; hand face at y = -19.7
ax_x, ax_z, face_y = 9.25, 0.0, -19.7
dx, dz = link[:, 0] - ax_x, link[:, 2] - ax_z
r = np.hypot(dx, dz)
az = np.degrees(np.arctan2(dz, dx))   # 0 deg = +x (camera side, away from the base)
h = link[:, 1] - face_y                # height above the hand face (positive = up into the housing)
print('vertices with h < 0 (below the hand face):', (h < -1e-6).sum(), ' min h', h.min().round(2))
for band in ((0, 1), (0, 5), (0, 10), (0, 15), (0, 20)):
    m = (h >= band[0] - 1e-6) & (h <= band[1])
    print(f'-- housing vertices with height {band[0]}..{band[1]} mm above the hand face: n={m.sum()}')
    bins = np.arange(-180, 181, 30)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mm = m & (az >= lo) & (az < hi)
        out.append(f'{lo:4d}..{hi:4d}: rmax {r[mm].max():5.1f}' if mm.any() else f'{lo:4d}..{hi:4d}:   --  ')
    print('   ' + ' | '.join(out))
# which side is +x? print the mesh's camera-side features: extents in x_link at the hand face plane
m = np.abs(h) < 0.5
print('hand-face-plane vertices: x_link range', link[m, 0].min().round(2), link[m, 0].max().round(2), ' z range', link[m, 2].min().round(2), link[m, 2].max().round(2))
# also: tri-based: for each azimuth sector, the lowest housing point (min h) at radius bands
print('-- lowest housing point (min height above the hand face) by radius band, all azimuths')
for rlo, rhi in ((0, 12), (12, 20), (20, 25), (25, 30), (30, 40), (40, 60)):
    mm = (r >= rlo) & (r < rhi)
    if mm.any():
        print(f'   r {rlo:2d}..{rhi:2d} mm: lowest h = {h[mm].min():6.2f} mm  (n={mm.sum()})')
