"""Radial profile along z and planar-face / circular-hole detection for a binary STL (metres)."""
import sys, struct
import numpy as np

def read_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]))
    return rec['v'].astype(np.float64), rec['n'].astype(np.float64)

def fit_circle(pts):
    # algebraic least squares circle fit in 2D
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    resid = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, resid.max()

def cluster_2d(pts, eps):
    # simple single-linkage clustering by grid flood
    n = len(pts)
    labels = -np.ones(n, dtype=int)
    cur = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        stack = [i]; labels[i] = cur
        while stack:
            j = stack.pop()
            d = np.linalg.norm(pts - pts[j], axis=1)
            nb = np.where((d < eps) & (labels < 0))[0]
            labels[nb] = cur
            stack.extend(nb.tolist())
        cur += 1
    return labels

path = sys.argv[1]
axis = int(sys.argv[2]) if len(sys.argv) > 2 else 2
tri, nrm = read_stl(path)
v = tri.reshape(-1, 3)
uv = np.unique(v.round(6), axis=0)
others = [i for i in range(3) if i != axis]
print(f'== {path}  (axis={axis}, {len(uv)} unique verts)')

# radial profile: for slices along axis, max radius from axis line through centroid of extremes
lo, hi = uv[:, axis].min(), uv[:, axis].max()
print('-- profile along axis (slice centre, n verts, min/max of other two axes, max radius about 0,0)')
edges = np.linspace(lo, hi, 36)
for a, b in zip(edges[:-1], edges[1:]):
    m = (uv[:, axis] >= a) & (uv[:, axis] < b + 1e-9)
    if m.sum() == 0:
        continue
    s = uv[m]
    r = np.hypot(s[:, others[0]], s[:, others[1]])
    print(f'  {((a+b)/2)*1000:7.2f} mm  n={m.sum():4d}  {others[0]}:[{s[:,others[0]].min()*1000:6.1f},{s[:,others[0]].max()*1000:6.1f}]  {others[1]}:[{s[:,others[1]].min()*1000:6.1f},{s[:,others[1]].max()*1000:6.1f}]  rmax={r.max()*1000:5.1f} rmin={r.min()*1000:5.1f}')

# planar faces: group triangles by normal (rounded) and plane offset
print('-- dominant planar faces (normal, offset, triangle count, vertex count)')
n_unit = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
offs = np.einsum('ij,ij->i', n_unit, tri[:, 0, :])
keys = np.c_[n_unit.round(3), offs.round(4)]
uk, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
order = np.argsort(-cnt)
for k in order[:12]:
    sel = tri[inv == k].reshape(-1, 3)
    print(f'  n={uk[k][:3]}  d={uk[k][3]*1000:7.2f} mm  tris={cnt[k]:4d}  verts={len(np.unique(sel.round(6), axis=0))}')

# circles on the most populated planes: cluster vertices, fit circles
print('-- circle fits on top planar faces (clusters of plane vertices)')
for k in order[:6]:
    nn = uk[k][:3]
    sel = np.unique(tri[inv == k].reshape(-1, 3).round(6), axis=0)
    # project onto plane basis
    a = np.array([1, 0, 0]) if abs(nn[0]) < 0.9 else np.array([0, 1, 0])
    e1 = np.cross(nn, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(nn, e1)
    p2 = np.c_[sel @ e1, sel @ e2]
    labels = cluster_2d(p2, eps=0.0012)
    print(f'  plane n={nn} d={uk[k][3]*1000:.2f} mm: {labels.max()+1} clusters')
    for c in range(labels.max() + 1):
        pts = p2[labels == c]
        if len(pts) < 6:
            continue
        cx, cy, r, res = fit_circle(pts)
        centre3 = cx * e1 + cy * e2 + nn * uk[k][3]
        flag = 'CIRCLE' if res < 0.00025 else 'not-circ'
        print(f'    cluster {c}: {len(pts):3d} pts  r={r*1000:6.2f} mm  centre xyz={np.round(centre3*1000,2)}  maxres={res*1000:.3f} mm  {flag}')
