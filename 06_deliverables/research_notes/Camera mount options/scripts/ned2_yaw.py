import math, itertools
from ned2_fk import fk, col, sub, norm, in_limits, rz, PI
def solve(A, b):
    n = len(A); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-14: raise ZeroDivisionError
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                for c in range(i, n + 1): M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]
def gn(resid, x0, iters=80, lam=1e-3):
    x = x0[:]
    for _ in range(iters):
        r = resid(x); n, m = len(x), len(r); J = []; h = 1e-6
        for j in range(n):
            xp = x[:]; xp[j] += h; rp = resid(xp); J.append([(rp[i]-r[i])/h for i in range(m)])
        A = [[sum(J[a][i]*J[b][i] for i in range(m)) + (lam if a == b else 0) for b in range(n)] for a in range(n)]
        g = [-sum(J[a][i]*r[i] for i in range(m)) for a in range(n)]
        try: dx = solve(A, g)
        except ZeroDivisionError: break
        x = [x[k]+dx[k] for k in range(n)]
        if max(abs(d) for d in dx) < 1e-10: break
    return x, resid(x)
def tool_pos(q): return fk(q)["tool"][0]
def hand_z(q): return col(fk(q)["joint_6"][1], 2)
def cam_axis(q):
    p, R = fk(q)["joint_5"]; Rc = rz(-PI/2 + math.radians(10))
    return [sum(R[i][k]*Rc[k][0] for k in range(3)) for i in range(3)]
Y = -0.02; Z = 0.037
print("Wrist-mounted camera (URDF) with joint_4 used for yaw, tip held on the line, rod tilt minimised:")
for x in [0.20, 0.25, 0.30, 0.35]:
    target = [x, Y, Z]
    for q4d in [45, 90]:
        q4 = math.radians(q4d); best = None
        def resid(v):
            q = [v[0], v[1], v[2], q4, v[3], 0.0]; p = tool_pos(q); hz = hand_z(q); W = 1000.0
            return [W*(p[0]-target[0]), W*(p[1]-target[1]), W*(p[2]-target[2]), hz[0], hz[1], hz[2]+1]
        for s2, s3, s5 in itertools.product([-0.8,-1.0], [-0.8,-0.3,0.3], [-1.0,-0.4,0.2,0.8]):
            v, r = gn(resid, [math.atan2(Y, x), s2, s3, s5]); q = [v[0], v[1], v[2], q4, v[3], 0.0]
            if norm(sub(tool_pos(q), target)) < 1e-4 and not in_limits(q) and hand_z(q)[2] < 0:
                tilt = math.degrees(math.acos(max(-1, min(1, -hand_z(q)[2]))))
                if best is None or tilt < best[0]: best = (tilt, q)
        tilt, q = best; c = cam_axis(q); q1 = q[0]; rh = [math.cos(q1), math.sin(q1)]
        az = math.degrees(math.atan2(c[1]*rh[0]-c[0]*rh[1], c[0]*rh[0]+c[1]*rh[1]))
        hz = hand_z(q); rod_az = math.degrees(math.atan2(hz[1]*rh[0]-hz[0]*rh[1], hz[0]*rh[0]+hz[1]*rh[1]))
        print(f"  x={x:.2f} q4={q4d:>3}: rod tilt {tilt:4.1f} deg (leans toward azimuth {rod_az:+6.1f}); camera axis {math.degrees(math.acos(-c[2])):4.1f} deg from straight down, azimuth {az:+6.1f} deg from radial-out; joints(deg)={[round(math.degrees(v),1) for v in q]}")
