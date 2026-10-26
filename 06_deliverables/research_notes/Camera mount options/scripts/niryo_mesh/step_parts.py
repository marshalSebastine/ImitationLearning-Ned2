"""STEP AP203/AP242: attribute cylindrical surfaces and planes to their PRODUCT by traversing the shape representation graph; print small cylinders (holes/bosses) per product."""
import sys, re
from collections import deque
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
path = sys.argv[1]
rmax = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
want = sys.argv[3].split('|') if len(sys.argv) > 3 else None
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
# product name per PRODUCT_DEFINITION
pname = {}
for k, (t, b) in ents.items():
    if t == 'PRODUCT':
        pname[k] = re.findall(r"'([^']*)'", b)[0]
pdf_prod = {k: refs[k][-1] for k, (t, b) in ents.items() if t == 'PRODUCT_DEFINITION_FORMATION' or t.startswith('PRODUCT_DEFINITION_FORMATION')}
pd_prod = {}
for k, (t, b) in ents.items():
    if t == 'PRODUCT_DEFINITION':
        for r in refs[k]:
            if r in pdf_prod: pd_prod[k] = pname.get(pdf_prod[r], '?')
pds_pd = {k: next((r for r in refs[k] if r in pd_prod), None) for k, (t, b) in ents.items() if t == 'PRODUCT_DEFINITION_SHAPE'}
# shape representation per product
rep_prod = {}
for k, (t, b) in ents.items():
    if t == 'SHAPE_DEFINITION_REPRESENTATION':
        pds, rep = refs[k][0], refs[k][1]
        pd = pds_pd.get(pds)
        if pd is not None: rep_prod[rep] = pd_prod.get(pd, '?')
# BFS from each shape representation that holds breps
owner = {}
for rep, prod in rep_prod.items():
    t = ents[rep][0]
    q = deque([rep]); seen = {rep}
    while q:
        e = q.popleft()
        for r in refs.get(e, []):
            if r not in seen and r in ents:
                seen.add(r); q.append(r)
    for e in seen:
        if ents[e][0] in ('CYLINDRICAL_SURFACE', 'PLANE', 'MANIFOLD_SOLID_BREP'):
            owner[e] = prod
prods = sorted(set(owner.values()))
print('products with geometry:', prods)
for prod in prods:
    if want and not any(w in prod for w in want): continue
    cyls = []
    for e, p in owner.items():
        if p != prod or ents[e][0] != 'CYLINDRICAL_SURFACE': continue
        b = ents[e][1]; r = float(b.strip().split(',')[-1])
        pl = refs[e][0]
        pr = refs[pl]
        c, d = pts.get(pr[0]), dirs.get(pr[1])
        if c is None or d is None: continue
        cyls.append((r, c, d))
    # bounding box of all points reachable? approximate by vertex points owned: collect CARTESIAN_POINT under this product
    print(f'\n== {prod}: {len(cyls)} cylindrical surfaces')
    seen = set()
    for r, c, d in sorted(cyls, key=lambda z: (z[0], tuple(z[1].round(1)))):
        if r > rmax: continue
        key = (round(r, 3), tuple(np.round(c, 1)), tuple(np.abs(np.round(d, 2))))
        if key in seen: continue
        seen.add(key)
        print(f'   r={r:6.3f}  centre=({c[0]:8.2f},{c[1]:8.2f},{c[2]:8.2f})  axis=({d[0]:5.2f},{d[1]:5.2f},{d[2]:5.2f})')
