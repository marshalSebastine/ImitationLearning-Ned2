"""For one solid in a STEP file, list each cylindrical face's axis, radius and the z-range (along its axis) spanned by its vertices, to tell blind-hole depth and which outer face a hole starts on."""
import sys, re
from collections import deque
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
path, want, rmax = sys.argv[1], sys.argv[2], float(sys.argv[3])
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
for s, (t, b) in ents.items():
    if t != 'MANIFOLD_SOLID_BREP': continue
    name = re.findall(r"'([^']*)'", b); name = name[0] if name else ''
    if want not in name: continue
    print(f'== solid "{name}"')
    shell = refs[s][0]
    for face in refs[shell]:
        if ents[face][0] != 'ADVANCED_FACE': continue
        fr = refs[face]
        surf = fr[-1]
        if ents[surf][0] != 'CYLINDRICAL_SURFACE': continue
        r = float(ents[surf][1].strip().split(',')[-1])
        if r > rmax: continue
        pl = refs[surf][0]; pr = refs[pl]
        c, d = pts[pr[0]], dirs[pr[1]]
        # collect vertex points of the face bounds
        q = deque(fr[:-1]); seen = set(); P = []
        while q:
            e = q.popleft()
            if e in seen or e not in ents: continue
            seen.add(e)
            if ents[e][0] == 'VERTEX_POINT':
                P.append(pts[refs[e][0]])
            else:
                q.extend(refs[e])
        if not P: continue
        P = np.array(P)
        along = (P - c) @ d
        print(f'  r={r:5.2f} (d={2*r:5.2f})  axis dir=({d[0]:5.2f},{d[1]:5.2f},{d[2]:5.2f})  axis pt=({c[0]:7.2f},{c[1]:7.2f},{c[2]:7.2f})  face spans along axis [{along.min():7.2f},{along.max():7.2f}] -> abs [{(c + d*along.min()).round(2)} .. {(c + d*along.max()).round(2)}]')
