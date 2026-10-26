"""STEP: for every MANIFOLD_SOLID_BREP, BFS its entity graph, report the bounding box of its vertex points, and list its cylindrical surfaces (radius, axis point, axis direction) up to a radius limit."""
import sys, re
from collections import deque
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]
rmax = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
only = sys.argv[3] if len(sys.argv) > 3 else None
SC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
ents = {}
for m in re.finditer(r'#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;', txt, re.S):
    ents[int(m.group(1))] = (m.group(2), m.group(3))
refs = {k: [int(r) for r in re.findall(r'#(\d+)', b)] for k, (t, b) in ents.items()}
def vec(b):
    inner = re.findall(r'\(\s*([^()]*)\)', b)
    return np.array([float(x) for x in inner[0].split(',')]) if inner else None
pts = {k: vec(b) for k, (t, b) in ents.items() if t == 'CARTESIAN_POINT'}
dirs = {k: vec(b) for k, (t, b) in ents.items() if t == 'DIRECTION'}
solids = [k for k, (t, b) in ents.items() if t == 'MANIFOLD_SOLID_BREP']
print(f'{len(ents)} entities, {len(solids)} solids')
for s in solids:
    name = re.findall(r"'([^']*)'", ents[s][1])
    name = name[0] if name else ''
    q = deque([s]); seen = {s}
    while q:
        e = q.popleft()
        for r in refs.get(e, []):
            if r not in seen and r in ents:
                seen.add(r); q.append(r)
    P = np.array([pts[e] for e in seen if e in pts and pts[e] is not None and len(pts[e]) == 3])
    lo, hi = P.min(0), P.max(0)
    cyls = []
    for e in seen:
        if ents[e][0] != 'CYLINDRICAL_SURFACE': continue
        b = ents[e][1]; r = float(b.strip().split(',')[-1])
        pl = refs[e][0]; pr = refs[pl]
        c, d = pts.get(pr[0]), dirs.get(pr[1])
        if c is None or d is None: continue
        cyls.append((r, c, d))
    print(f'\n== solid #{s} "{name}": {len(seen)} entities, {len(P)} points, bbox min {lo.round(2)} max {hi.round(2)} size {(hi-lo).round(2)}; {len(cyls)} cyl surfaces')
    if only and only not in name: continue
    seen_keys = set()
    radii = sorted({round(r, 2) for r, c, d in cyls})
    print('   distinct radii:', radii)
    for r, c, d in sorted(cyls, key=lambda z: (z[0], tuple(z[1].round(1)))):
        if r*SC > rmax: continue
        key = (round(r, 3), tuple(np.round(c, 1)), tuple(np.abs(np.round(d, 2))))
        if key in seen_keys: continue
        seen_keys.add(key)
        print(f'   r={r*SC:6.3f}  axis point=({c[0]*SC:8.2f},{c[1]*SC:8.2f},{c[2]*SC:8.2f})  axis dir=({d[0]:5.2f},{d[1]:5.2f},{d[2]:5.2f})')
