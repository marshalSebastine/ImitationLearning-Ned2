"""Geometry questions on the fitted Ned2 chain: pushing poses, forearm (camera) direction, reach of a vertical forearm."""
import json, math, sys
import numpy as np
from scipy.optimize import least_squares

PI = math.pi
M = json.load(open(sys.argv[1]))

def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]); Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def T_origin(xyz, rpy):
    T = np.eye(4); T[:3, :3] = rpy_matrix(*rpy); T[:3, 3] = xyz; return T

def Rz_T(q):
    T = np.eye(4); c, s = math.cos(q), math.sin(q); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c; return T

ORIGINS = [T_origin((0, 0, 0.1013), (0, 0, 0)), T_origin((0, 0, 0.065), M["j2_rpy"]), T_origin(M["j3_xyz"], M["j3_rpy"]),
           T_origin(M["j4_xyz"], M["j4_rpy"]), T_origin((0, 0, 0.17), M["j5_rpy"]), T_origin(M["j6_xyz"], M["j6_rpy"])]
LIMITS_DEG = [(-171.88, 171.88), (-105.0, 34.96), (-76.78, 85.5), (-119.75, 119.75), (-110.01, 110.17), (-144.96, 144.96)]

def frames(q):
    """World transforms of shoulder, arm, elbow, forearm, wrist, hand, tool frames."""
    T = np.eye(4); out = []
    for i in range(6):
        T = T @ ORIGINS[i] @ Rz_T(q[i]); out.append(T.copy())
    out.append(T @ T_origin((0, 0, 0.0285), (0, 0, 0)))
    return out

def summary(q):
    F = frames(q)
    tip = F[6][:3, 3]
    rod_axis = F[5][:3, 2]           # hand roll axis (joint 6 axis) = rod axis, pointing from flange to tip?
    forearm_axis = F[3][:3, 2]       # joint 4 axis (forearm roll axis)
    elbow = F[2][:3, 3]; wrist = F[4][:3, 3]
    fdir = wrist - elbow             # elbow -> wrist, where the forearm camera looks (approximately)
    return tip, rod_axis, forearm_axis, elbow, wrist, fdir

def within(q):
    return all(lo - 1e-6 <= math.degrees(qi) <= hi + 1e-6 for qi, (lo, hi) in zip(q, LIMITS_DEG))

# sanity: rod axis vs reported pitch
print("sanity: rod axis z-component vs reported pitch")
for q, pitch in [([0.00079, 0.49941, -1.25062, 0.00009, -0.00009, -0.00144], 0.7513), ([-0.17727, -0.67316, -1.04459, -0.29903, -0.19951, 0.12128], 1.2282),
                 ([0.3782, -0.9595, -0.3644, 0.4956, -0.402, 0.0032], 1.3532), ([-0.3873, -1.1943, 0.1795, -0.1855, -0.402, 0.0016], 1.3962)]:
    tip, rod, fa, elbow, wrist, fdir = summary(q)
    print(f"  reported pitch {pitch:.3f} -> rod axis {np.round(rod,3)}  (elevation of rod axis {math.degrees(math.asin(-rod[2])):.1f} deg below horizontal; -sin(pitch) would be {-math.sin(pitch):.3f})")

def ik(target, rod_dir=(0, 0, -1), q0=None, fix=None):
    """Solve q1..q6 for tip position and rod direction; fix: dict of joint index -> value."""
    q0 = np.array(q0 if q0 is not None else [0, -0.5, -0.5, 0, -0.5, 0], float)
    free = [i for i in range(6) if not (fix and i in fix)]
    def res(v):
        q = q0.copy()
        for i, vi in zip(free, v): q[i] = vi
        if fix:
            for i, val in fix.items(): q[i] = val
        tip, rod, *_ = summary(q)
        return np.concatenate([(tip - target) * 100, (rod - np.array(rod_dir)) * 10])
    lo = [math.radians(LIMITS_DEG[i][0]) for i in free]; hi = [math.radians(LIMITS_DEG[i][1]) for i in free]
    best = None
    for start in ([0, -0.5, -0.5, 0, -0.5, 0], [0, -0.8, -0.2, 0, -0.6, 0], [0, -1.2, 0.2, 0, -0.6, 0], [0, -0.3, -1.0, 0, 0.2, 0], [0, -1.0, 0.8, 0, -1.3, 0]):
        v0 = np.clip([start[i] for i in free], lo, hi)
        sol = least_squares(res, v0, bounds=(lo, hi))
        if best is None or sol.cost < best.cost: best = sol
    q = q0.copy()
    for i, vi in zip(free, best.x): q[i] = vi
    if fix:
        for i, val in fix.items(): q[i] = val
    return q, best.cost

print("\nPushing poses along the line (tip z 0.039 = rod end with magnet block; rod straight down; y -0.02):")
print("  x_tip  q1..q6 (deg)                          q4   forearm elev(deg below horiz)  forearm azimuth(deg)  elbow x,z    cost")
for x in [0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.40, 0.42]:
    q, cost = ik(np.array([x, -0.02, 0.039]))
    tip, rod, fa, elbow, wrist, fdir = summary(q)
    elev = math.degrees(math.atan2(-fdir[2], math.hypot(fdir[0], fdir[1])))
    az = math.degrees(math.atan2(fdir[1], fdir[0]))
    print(f"  {x:.2f}  {np.round(np.degrees(q),1)}  q4={math.degrees(q[3]):6.1f}  elev={elev:6.1f}  az={az:6.1f}  elbow=({elbow[0]:.3f},{elbow[2]:.3f})  cost={cost:.2e}  limits_ok={within(q)}")

print("\nSame with bare rod height (tip z 0.0165):")
for x in [0.14, 0.26, 0.38]:
    q, cost = ik(np.array([x, -0.02, 0.0165]))
    tip, rod, fa, elbow, wrist, fdir = summary(q)
    elev = math.degrees(math.atan2(-fdir[2], math.hypot(fdir[0], fdir[1])))
    print(f"  {x:.2f}  q4={math.degrees(q[3]):6.1f}  forearm elev={elev:6.1f}  cost={cost:.2e} limits_ok={within(q)}")

print("\nPushing poses off-axis (y = +0.08 and -0.11, x 0.2): azimuth of forearm vs azimuth of tip")
for (x, y) in [(0.2, 0.08), (0.2, -0.11), (0.4, 0.08), (0.15, -0.11)]:
    q, cost = ik(np.array([x, y, 0.039]))
    tip, rod, fa, elbow, wrist, fdir = summary(q)
    az = math.degrees(math.atan2(fdir[1], fdir[0])); az_tip = math.degrees(math.atan2(y, x))
    print(f"  tip ({x},{y}): forearm azimuth {az:6.1f}, tip azimuth {az_tip:6.1f}, q4={math.degrees(q[3]):.1f}, q1={math.degrees(q[0]):.1f}, cost {cost:.1e}")

print("\nRod leaning 20 deg (pitch 1.23, lean about y): what q4 does the solver need? (compare with lean20 json: -17 and -43 deg)")
for z in (0.0459, 0.0164):
    rd = (math.cos(1.2282 + PI/2), 0, math.sin(-1.2282))  # rod axis tilted 20 deg from vertical
    rd = np.array([-math.cos(1.2282), 0, -math.sin(1.2282)]); rd = rd / np.linalg.norm(rd)
    for sgn in (1, -1):
        r = rd.copy(); r[0] *= sgn
        q, cost = ik(np.array([0.135, -0.024, z]), rod_dir=tuple(r))
        print(f"  z={z}: rod dir {np.round(r,3)} -> q4={math.degrees(q[3]):6.1f} q5={math.degrees(q[5-1]):6.1f} cost={cost:.1e}")

print("\nVertical forearm (camera straight down), rod along forearm (q5=0), q1=0, q4=0, q6=0: max tip x for tip z 0.039")
# forearm vertical means fdir = (0,0,-1). Solve q2,q3 with q5=0 for fdir vertical and tip z = 0.039
def solve_vertical(z_tip, elev_deg=90.0):
    def res(v):
        q = [0, v[0], v[1], 0, 0, 0]
        tip, rod, fa, elbow, wrist, fdir = summary(q)
        elev = math.atan2(-fdir[2], math.hypot(fdir[0], fdir[1]))
        return [(tip[2] - z_tip) * 100, (elev - math.radians(elev_deg)) * 10]
    best = None
    for start in ([-0.6, -0.9], [-1.5, 0.5], [0.3, -1.3], [-1.0, 0.0]):
        sol = least_squares(res, start, bounds=([math.radians(LIMITS_DEG[1][0]), math.radians(LIMITS_DEG[2][0])], [math.radians(LIMITS_DEG[1][1]), math.radians(LIMITS_DEG[2][1])]))
        if best is None or sol.cost < best.cost: best = sol
    q = [0, best.x[0], best.x[1], 0, 0, 0]
    return q, best.cost

for elev in (90, 100, 110, 120, 80, 70, 60, 45):
    q, cost = solve_vertical(0.039, elev)
    tip, rod, fa, elbow, wrist, fdir = summary(q)
    print(f"  forearm elevation {elev:3d} deg (>90 = sloping back toward base): tip x={tip[0]:.3f} z={tip[2]:.3f}, elbow x={elbow[0]:.3f} z={elbow[2]:.3f}, q2={math.degrees(q[1]):.1f} q3={math.degrees(q[2]):.1f}, rod axis z={rod[2]:.2f}, cost={cost:.1e}, limits_ok={within(q)}")

print("\nPerpendicular offset of the tool axis from the elbow when the forearm is vertical:")
q, cost = solve_vertical(0.039, 90)
tip, rod, fa, elbow, wrist, fdir = summary(q)
print(f"  elbow x {elbow[0]:.4f}, wrist x {wrist[0]:.4f}, tip x {tip[0]:.4f}  -> tool axis is {1000*(tip[0]-elbow[0]):.1f} mm forward of the elbow axis")

print("\nCheck: planar reach at elbow height needed for vertical forearm; stack below elbow:")
print(f"  elbow z {elbow[2]:.3f} minus tip z {tip[2]:.3f} = {elbow[2]-tip[2]:.4f} m (C6 says 0.17+0.065+0.0197+0.0285 = {0.17+0.065+0.0197+0.0285:.4f} to rod end)")
