"""List cylindrical surfaces (radius, axis point, axis direction) in a STEP AP203 file, optionally filtered by a bounding box; also overall CARTESIAN_POINT extents and PRODUCT names."""
import sys, re
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]
box = [float(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else None   # xmin,xmax,ymin,ymax,zmin,zmax
rmax = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
ents = {}
for m in re.finditer(r'#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;', txt, re.S):
    ents[int(m.group(1))] = (m.group(2), m.group(3))
print('entities:', len(ents))
print('PRODUCT names:', sorted({re.findall(r"'([^']*)'", b)[0] for t, b in ents.values() if t == 'PRODUCT' and re.findall(r"'([^']*)'", b)}))
pts = {k: np.array([float(x) for x in re.findall(r'\(\s*([^()]*)\)', b)[0].split(',')]) for k, (t, b) in ents.items() if t == 'CARTESIAN_POINT'}
dirs = {k: np.array([float(x) for x in re.findall(r'\(\s*([^()]*)\)', b)[0].split(',')]) for k, (t, b) in ents.items() if t == 'DIRECTION'}
P = np.array(list(pts.values()))
print('CARTESIAN_POINT extents: min', P.min(0).round(2), 'max', P.max(0).round(2), 'n', len(P))
# cylinders
rows = []
for k, (t, b) in ents.items():
    if t != 'CYLINDRICAL_SURFACE':
        continue
    refs = [int(r) for r in re.findall(r'#(\d+)', b)]
    r = float(b.strip().split(',')[-1])
    pl = ents.get(refs[0])
    if not pl or pl[0] != 'AXIS2_PLACEMENT_3D':
        continue
    prefs = [int(r_) for r_ in re.findall(r'#(\d+)', pl[1])]
    p = pts.get(prefs[0]); d = dirs.get(prefs[1])
    if p is None or d is None:
        continue
    if box and not (box[0] <= p[0] <= box[1] and box[2] <= p[1] <= box[3] and box[4] <= p[2] <= box[5]):
        continue
    if r > rmax:
        continue
    rows.append((r, *p.round(3), *d.round(3)))
rows.sort()
# collapse duplicates
seen = set()
print(f'cylindrical surfaces with r<={rmax}' + (f' inside box {box}' if box else '') + ': (r, point xyz, axis dir)')
for row in rows:
    key = (round(row[0], 3), tuple(np.round(row[1:4], 1)), tuple(np.abs(np.round(row[4:7], 2))))
    if key in seen: continue
    seen.add(key)
    print(f'  r={row[0]:6.3f}  p=({row[1]:8.2f},{row[2]:8.2f},{row[3]:8.2f})  axis=({row[4]:6.2f},{row[5]:6.2f},{row[6]:6.2f})')
