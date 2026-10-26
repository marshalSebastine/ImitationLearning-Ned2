"""For a binary STL: list planar faces normal to each axis and the radii of their vertices about a chosen centre; detect hole circles by grouping vertices at equal distance from candidate centres."""
import sys, struct
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]))
    return rec['v'].astype(np.float64), rec['n'].astype(np.float64)

def fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0))
    resid = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, resid.max()

def circles_from_edges(tris2d, tol=0.00005):
    """Find loops of connected edges that lie on one plane face and fit circles to each loop."""
    # collect boundary edges of the planar triangle set (edges used once)
    edges = {}
    for t in tris2d:
        for i in range(3):
            a = tuple(np.round(t[i], 6)); b = tuple(np.round(t[(i + 1) % 3], 6))
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
    boundary = [e for e, c in edges.items() if c == 1]
    # build adjacency
    adj = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    seen = set(); loops = []
    for start in adj:
        if start in seen: continue
        loop = [start]; seen.add(start); cur = start; prev = None
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt: break
            prev, cur = cur, nxt[0]; loop.append(cur); seen.add(cur)
        loops.append(np.array(loop))
    out = []
    for lp in loops:
        if len(lp) < 6: continue
        cx, cy, r, res = fit_circle(lp)
        out.append((len(lp), cx, cy, r, res))
    return out

path = sys.argv[1]
min_tris = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
tri, nrm = read_stl(path)
n_unit = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
offs = np.einsum('ij,ij->i', n_unit, tri[:, 0, :])
keys = np.c_[n_unit.round(2), offs.round(4)]
uk, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
order = np.argsort(-cnt)
print(f'== {path}: {len(tri)} tris; {len(uk)} (normal, offset) groups')
for k in order:
    if cnt[k] < min_tris: break
    nn = uk[k][:3]
    if np.abs(np.abs(nn).max() - 1) > 0.02:   # only axis-aligned planes
        continue
    ax = int(np.argmax(np.abs(nn)))
    oth = [i for i in range(3) if i != ax]
    t = tri[inv == k]
    t2 = t[:, :, oth]
    loops = circles_from_edges(t2)
    verts = np.unique(t.reshape(-1, 3).round(6), axis=0)
    ext = verts[:, oth]
    print(f'-- plane normal {"xyz"[ax]}{"+" if nn[ax]>0 else "-"} at {"xyz"[ax]}={uk[k][3]*np.sign(nn[ax])*SCALE:8.2f} mm  tris={cnt[k]:4d}  extent {"xyz"[oth[0]]}[{ext[:,0].min()*SCALE:.1f},{ext[:,0].max()*SCALE:.1f}] {"xyz"[oth[1]]}[{ext[:,1].min()*SCALE:.1f},{ext[:,1].max()*SCALE:.1f}]')
    for (npts, cx, cy, r, res) in sorted(loops, key=lambda z: -z[0]):
        tag = 'CIRCLE' if res < 0.00015 else 'loop'
        print(f'     {tag:6s} n={npts:3d} r={r*SCALE:6.2f} mm (d={2*r*SCALE:6.2f}) centre {"xyz"[oth[0]]}={cx*SCALE:7.2f} {"xyz"[oth[1]]}={cy*SCALE:7.2f}  maxres={res*SCALE:.3f}')
