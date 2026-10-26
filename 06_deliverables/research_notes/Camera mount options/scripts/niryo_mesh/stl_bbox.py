"""Parse binary STL files and print bounding boxes (units as stored)."""
import sys, struct
import numpy as np

def read_stl(path):
    with open(path, 'rb') as f:
        data = f.read()
    header = data[:80]
    n = struct.unpack('<I', data[80:84])[0]
    expected = 84 + n * 50
    if len(data) != expected:
        # try ASCII
        if data[:5] == b'solid':
            verts = []
            for line in data.decode('ascii', 'ignore').splitlines():
                s = line.strip().split()
                if s and s[0] == 'vertex':
                    verts.append([float(s[1]), float(s[2]), float(s[3])])
            v = np.array(verts).reshape(-1, 3, 3)
            return header, v, None
        raise ValueError(f'size mismatch: {len(data)} vs {expected}')
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')]))
    return header, rec['v'].astype(np.float64), rec['n'].astype(np.float64)

for path in sys.argv[1:]:
    header, tri, normals = read_stl(path)
    v = tri.reshape(-1, 3)
    lo, hi = v.min(axis=0), v.max(axis=0)
    print(f'== {path}')
    print(f'   header: {header[:60]!r}')
    print(f'   triangles: {len(tri)}   unique verts: {len(np.unique(v.round(6), axis=0))}')
    print(f'   min xyz: {lo.round(5)}')
    print(f'   max xyz: {hi.round(5)}')
    print(f'   size xyz: {(hi - lo).round(5)}')
