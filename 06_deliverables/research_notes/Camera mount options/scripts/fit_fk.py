"""Fit the Ned2 kinematic chain conventions to recorded (joints, pose) pairs, then answer geometry questions."""
import itertools, json, math, sys
import numpy as np

PI = math.pi

def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def T_origin(xyz, rpy):
    T = np.eye(4); T[:3, :3] = rpy_matrix(*rpy); T[:3, 3] = xyz; return T

def Rz_T(q):
    T = np.eye(4); c, s = math.cos(q), math.sin(q); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c; return T

DATA = [  # (joints, xyz)
    ([0.00079, 0.49941, -1.25062, 0.00009, -0.00009, -0.00144], [0.14017, 0.00011, 0.20327]),
    ([-0.17727, -0.67316, -1.04459, -0.29903, -0.19951, 0.12128], [0.13509, -0.02435, 0.04590]),
    ([-0.20162, -0.87162, -0.96429, -0.75156, -0.09520, 0.52625], [0.13545, -0.02450, 0.01641]),
    ([0.3782, -0.9595, -0.3644, 0.4956, -0.402, 0.0032], [0.2581, 0.1079, 0.0146]),
    ([-0.5136, -1.011, -0.2523, -0.1518, -0.402, 0.0001], [0.261, -0.149, 0.0105]),
    ([0.5517, -0.8807, -0.6749, 0.3314, -0.402, 0.0032], [0.1729, 0.1104, 0.0151]),
    ([-0.6902, -0.8943, -0.6113, -0.2315, -0.402, 0.0016], [0.1686, -0.1423, 0.0144]),
    ([-0.3873, -1.1943, 0.1795, -0.1855, -0.402, 0.0016], [0.3495, -0.1446, 0.0078]),
    ([0.3219, -1.1549, 0.0947, 0.0875, -0.4005, 0.0032], [0.3457, 0.1162, 0.0086]),
]
Q = np.array([d[0] for d in DATA]); P = np.array([d[1] for d in DATA]); N = len(DATA)

half = [(0, PI/2, 0), (0, -PI/2, 0), (PI/2, 0, 0), (-PI/2, 0, 0)]
cand2 = [(r, p, 0) for r in (PI/2, -PI/2) for p in (0, PI/2, -PI/2, PI)]
cand3_xyz = [(sx*a, sy*b, 0) for (a, b) in ((0.012, 0.221), (0.221, 0.012)) for sx in (1, -1) for sy in (1, -1)]
cand3_rpy = [(0, 0, 0), (0, 0, PI/2), (0, 0, -PI/2), (0, 0, PI)]
cand4_xyz = [(sx*0.0325, sy*0.065, 0) for sx in (1, -1) for sy in (1, -1)]
cand6_xyz = [(sx*0.00925, sy*0.0197, 0) for sx in (1, -1) for sy in (1, -1)]

def stage(prev, originsT, qcol):
    """prev: (K, N, 4, 4); originsT: list of 4x4; returns (K*len, N, 4, 4)"""
    out = []
    Rq = np.array([Rz_T(q) for q in Q[:, qcol]])  # (N,4,4)
    for To in originsT:
        step = np.einsum('ij,njk->nik', To, Rq)   # (N,4,4)
        out.append(np.einsum('knij,njl->knil', prev, step))
    return np.concatenate(out, axis=0)

T1 = np.array([[T_origin((0, 0, 0.1013), (0, 0, 0)) @ Rz_T(q)] for q in Q[:, 0]]).reshape(1, N, 4, 4)
o2 = [T_origin((0, 0, 0.065), rpy) for rpy in cand2]
T12 = stage(T1, o2, 1)
o3 = [T_origin(xyz, rpy) for xyz in cand3_xyz for rpy in cand3_rpy]
T123 = stage(T12, o3, 2)
o4 = [T_origin(xyz, rpy) for xyz in cand4_xyz for rpy in half]
T1234 = stage(T123, o4, 3)
o5 = [T_origin((0, 0, 0.17), rpy) for rpy in half]
T12345 = stage(T1234, o5, 4)
o6 = [T_origin(xyz, rpy) for xyz in cand6_xyz for rpy in half]
T123456 = stage(T12345, o6, 5)
tool = np.array([0, 0, 0.0285, 1.0])
tip = T123456 @ tool  # (K,N,4)
err = np.linalg.norm(tip[:, :, :3] - P[None], axis=2)  # (K,N)
rms = np.sqrt((err**2).mean(axis=1))
order = np.argsort(rms)
K = len(rms)
n6, n5, n4, n3, n2 = len(o6), len(o5), len(o4), len(o3), len(o2)
def decode(k):
    i2 = k % n2; k //= n2
    i3 = k % n3; k //= n3
    i4 = k % n4; k //= n4
    i5 = k % n5; k //= n5
    i6 = k
    return i2, i3, i4, i5, i6
print("combos", K)
for k in order[:8]:
    i2, i3, i4, i5, i6 = decode(int(k))
    print(f"rms {1000*rms[k]:.2f} mm  max {1000*err[k].max():.2f} mm | j2 rpy {cand2[i2]} | j3 xyz {cand3_xyz[i3//4]} rpy {cand3_rpy[i3%4]} | j4 xyz {cand4_xyz[i4//4]} rpy {half[i4%4]} | j5 rpy {half[i5]} | j6 xyz {cand6_xyz[i6//4]} rpy {half[i6%4]}")
best = int(order[0])
i2, i3, i4, i5, i6 = decode(best)
model = {"j2_rpy": cand2[i2], "j3_xyz": cand3_xyz[i3//4], "j3_rpy": cand3_rpy[i3%4], "j4_xyz": cand4_xyz[i4//4], "j4_rpy": half[i4%4],
         "j5_rpy": half[i5], "j6_xyz": cand6_xyz[i6//4], "j6_rpy": half[i6%4], "rms_mm": float(1000*rms[best]), "per_point_mm": (1000*err[best]).round(2).tolist()}
json.dump(model, open(sys.argv[1], "w"), indent=1)
print(json.dumps(model, indent=1))

# verify decoded model directly
def fk_direct(q):
    T = T_origin((0,0,0.1013),(0,0,0)) @ Rz_T(q[0])
    T = T @ T_origin((0,0,0.065), model["j2_rpy"]) @ Rz_T(q[1])
    T = T @ T_origin(model["j3_xyz"], model["j3_rpy"]) @ Rz_T(q[2])
    T = T @ T_origin(model["j4_xyz"], model["j4_rpy"]) @ Rz_T(q[3])
    T = T @ T_origin((0,0,0.17), model["j5_rpy"]) @ Rz_T(q[4])
    T = T @ T_origin(model["j6_xyz"], model["j6_rpy"]) @ Rz_T(q[5])
    return (T @ tool)[:3]
print("direct check (mm):", [round(1000*float(np.linalg.norm(fk_direct(Q[i]) - P[i])),2) for i in range(N)])
