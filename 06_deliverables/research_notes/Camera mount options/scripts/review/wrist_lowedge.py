"""Wrist STL: for azimuth sectors about the rod axis and radius bands, the lowest housing point (height above the hand face),
to see where a camera body swinging at r 15..45 mm would touch at each pan angle. Azimuth 0 = +x_link (camera side)."""
import sys, struct
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]))
    return rec['v'].astype(np.float64)
tri = read_stl(sys.argv[1]) * 1000.0
# densify: sample points on triangles (barycentric grid) so long triangles are represented
pts = []
for t in tri:
    for a in np.linspace(0, 1, 6):
        for b in np.linspace(0, 1 - a, 6):
            pts.append(t[0] + a * (t[1] - t[0]) + b * (t[2] - t[0]))
v = np.array(pts)
link = np.c_[v[:, 1], -v[:, 0], v[:, 2]]
dx, dz = link[:, 0] - 9.25, link[:, 2]
r = np.hypot(dx, dz); az = np.degrees(np.arctan2(dz, dx)); h = link[:, 1] + 19.7
print('lowest housing point (mm above the hand face) by azimuth sector (rows) and radius band (cols); "-" = no housing there')
bands = [(12, 18), (18, 24), (24, 30), (30, 36), (36, 45), (45, 60)]
print('   az      ' + ''.join(f'{lo:3d}-{hi:<3d}  ' for lo, hi in bands))
for lo in range(-180, 180, 20):
    m0 = (az >= lo) & (az < lo + 20)
    row = []
    for rlo, rhi in bands:
        m = m0 & (r >= rlo) & (r < rhi)
        row.append(f'{h[m].min():6.1f}  ' if m.any() else '    -    ')
    print(f'{lo:5d}..{lo+20:4d} ' + ''.join(row))
