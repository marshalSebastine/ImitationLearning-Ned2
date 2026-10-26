"""STEP: for each MANIFOLD_SOLID_BREP (optionally name filter), list every face with surface type, placement
(point, axis dir, ref dir), radius/semi-angle, and bbox of its vertex points. Also prints the assembly
transforms (ITEM_DEFINED_TRANSFORMATION) so part frames can be mapped. Units printed as stored * SC.
Usage: py -3.13 -I step_allfaces.py file.step [name_filter] [SC]"""
import sys, re
from collections import deque
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '' else None
SC = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
ents = {}
for m in re.finditer(r'#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;', txt, re.S):
    ents[int(m.group(1))] = (m.group(2), m.group(3))
# complex entities like #22=( ... ) have no type name; skip
refs = {k: [int(r) for r in re.findall(r'#(\d+)', b)] for k, (t, b) in ents.items()}
def vec(b):
    inner = re.findall(r'\(\s*([^()]*)\)', b)
    return np.array([float(x) for x in inner[0].split(',')]) if inner else None
pts = {k: vec(b) for k, (t, b) in ents.items() if t == 'CARTESIAN_POINT'}
dirs = {k: vec(b) for k, (t, b) in ents.items() if t == 'DIRECTION'}
def placement(pl):
    pr = refs[pl]
    p = pts.get(pr[0]); d = dirs.get(pr[1]) if len(pr) > 1 else None; x = dirs.get(pr[2]) if len(pr) > 2 else None
    return p, d, x
print('== transforms')
for k, (t, b) in ents.items():
    if t == 'ITEM_DEFINED_TRANSFORMATION':
        a, bb = refs[k][0], refs[k][1]
        pa, da, xa = placement(a); pb, db, xb = placement(bb)
        print(f'  #{k}: from #{a} p={np.round(pa*SC,3)} z={da} x={xa}  ->  #{bb} p={np.round(pb*SC,3)} z={db} x={xb}')
for s, (t, b) in ents.items():
    if t != 'MANIFOLD_SOLID_BREP': continue
    name = re.findall(r"'([^']*)'", b); name = name[0] if name else ''
    if want and want not in name: continue
    shell = refs[s][0]
    faces = [f for f in refs[shell] if ents.get(f, ('',))[0] == 'ADVANCED_FACE']
    # all vertex points of solid
    q = deque([s]); seen = {s}
    while q:
        e = q.popleft()
        for r in refs.get(e, []):
            if r not in seen and r in ents:
                seen.add(r); q.append(r)
    P = np.array([pts[e] for e in seen if e in pts and pts[e] is not None and len(pts[e]) == 3]) * SC
    print(f'\n== solid #{s} "{name}": {len(faces)} faces; vertex bbox min {P.min(0).round(2)} max {P.max(0).round(2)}')
    rows = []
    for face in faces:
        fr = refs[face]
        surf = fr[-1]
        st, sb = ents[surf]
        q = deque(fr[:-1]); seen2 = set(); V = []
        while q:
            e = q.popleft()
            if e in seen2 or e not in ents: continue
            seen2.add(e)
            if ents[e][0] == 'VERTEX_POINT':
                V.append(pts[refs[e][0]])
            else:
                q.extend(refs[e])
        V = np.array(V) * SC if V else np.zeros((0, 3))
        desc = st
        if st in ('PLANE', 'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE', 'TOROIDAL_SURFACE', 'SPHERICAL_SURFACE', 'DEGENERATE_TOROIDAL_SURFACE'):
            p, d, x = placement(refs[surf][0])
            nums = [float(v) for v in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?', sb.split(')')[-1] if ')' in sb else sb)]
            # parameters after the placement ref
            tail = sb.split(',', 2)[2] if sb.count(',') >= 2 else ''
            params = [float(v) * (SC if st != 'CONICAL_SURFACE' or i == 0 else 1) for i, v in enumerate(re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', tail))]
            desc += f' p={np.round(p*SC,2)} ax={np.round(d,2)} ref={np.round(x,2) if x is not None else None} params={np.round(params,3)}'
        if len(V):
            desc += f'  | verts n={len(V)} bbox [{V.min(0).round(2)} .. {V.max(0).round(2)}]'
        rows.append(desc)
    for r in sorted(rows):
        print('  ', r)
