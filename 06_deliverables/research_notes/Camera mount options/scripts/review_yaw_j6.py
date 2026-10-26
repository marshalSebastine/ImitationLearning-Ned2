"""Pose yaw <-> joint 6 for a vertical rod on the Ned2 (URDF FK, ned2_fk.py)."""
import math, itertools
from ned2_fk import fk, rpy, mm, rz, ry, rx, col, tool_pose, in_limits, norm, sub, PI
from ned2_yaw import gn

def R_eq(A, B): return max(abs(A[i][j]-B[i][j]) for i in range(3) for j in range(3))
# 1. gimbal lock identity at pitch = +pi/2
for r, y in [(0.3, 1.0), (-1.2, 0.4), (2.0, -2.5)]:
    A = rpy(r, PI/2, y); B = mm(rz(y - r), ry(PI/2)); C = mm(rz(y + r), ry(PI/2))
    print(f"roll {r:+.1f} yaw {y:+.1f}: Rz(y-r)Ry(90) err {R_eq(A,B):.1e}   Rz(y+r)Ry(90) err {R_eq(A,C):.1e}")
# at pitch 1.57 (what the scripts send) the orientation differs from pi/2 by:
A = rpy(0, 1.57, 0); B = rpy(0, PI/2, 0)
print("pitch 1.57 vs pi/2: max matrix diff", f"{R_eq(A,B):.2e}", "=", f"{math.degrees(PI/2-1.57):.3f} deg")

# 2. IK for a vertical rod at (x, y, z) with tool x pointing down, then vary j6 and read the tool yaw
def ik_vertical(target, yaw_world, seed_j6=0.0):
    """Solve joints 1..5 for the tool at target with rod vertical (tool x down) and the requested tool yaw (Rz(yaw)Ry(pi/2))."""
    Rt = mm(rz(yaw_world), ry(PI/2))
    best = None
    for s2, s3, s5, s6 in itertools.product([-0.9, -0.5], [-0.9, -0.3, 0.3], [-0.6, 0.0, 0.6], [seed_j6]):
        def resid(v):
            q = [v[0], v[1], v[2], v[3], v[4], v[5]]
            p, R = fk(q)["tool"]; W = 1000.0
            return [W*(p[k]-target[k]) for k in range(3)] + [R[i][j]-Rt[i][j] for i in range(3) for j in range(3)]
        v, r = gn(resid, [math.atan2(target[1], target[0]), s2, s3, 0.0, s5, s6], iters=120)
        p, R = fk(v)["tool"]
        if norm(sub(p, target)) < 1e-4 and R_eq(R, Rt) < 1e-4:
            if best is None or abs(v[3]) < abs(best[3]): best = v
    return best

for x in [0.16, 0.20, 0.26, 0.35, 0.40]:
    y = -0.024; z = 0.036
    j1 = math.atan2(y, x)
    print(f"\n--- tip ({x:.2f}, {y:.3f}, {z:.3f}); j1 (base azimuth) = {math.degrees(j1):+.1f} deg")
    for yaw_deg in [0, 20, 45, 90, 120, 135, 140, 150, 180, -90, -140, -150]:
        q = ik_vertical([x, y, z], math.radians(yaw_deg))
        if q is None:
            print(f"  yaw {yaw_deg:+4d}: no solution"); continue
        p, (r, pch, yw) = tool_pose(q)
        bad = in_limits(q)
        print(f"  yaw {yaw_deg:+4d}: joints deg {[round(math.degrees(v),1) for v in q]}  j6 - yaw = {math.degrees(q[5])-yaw_deg:+.1f}  j6 + j1 = {math.degrees(q[5]+q[0]):+.1f}  FK rpy = ({r:+.2f},{pch:+.3f},{yw:+.2f})  {'LIMIT ' + str(bad) if bad else 'ok'}")
