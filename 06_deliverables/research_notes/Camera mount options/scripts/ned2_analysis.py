import math, itertools
from ned2_fk import fk, col, sub, norm, unit, in_limits, LIM_DEG, PI

Y_LINE = -0.02
Z_TCP = 0.037

# ---------- tiny linear algebra ----------
def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-14:
            raise ZeroDivisionError
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]

def gauss_newton(resid, x0, iters=60, lam=1e-3):
    x = x0[:]
    for _ in range(iters):
        r = resid(x)
        n, m = len(x), len(r)
        J = []
        h = 1e-6
        for j in range(n):
            xp = x[:]; xp[j] += h
            rp = resid(xp)
            J.append([(rp[i] - r[i]) / h for i in range(m)])  # J[j][i] = d r_i / d x_j
        # normal equations (J^T J + lam I) dx = -J^T r
        A = [[sum(J[a][i] * J[b][i] for i in range(m)) + (lam if a == b else 0) for b in range(n)] for a in range(n)]
        g = [-sum(J[a][i] * r[i] for i in range(m)) for a in range(n)]
        try:
            dx = solve(A, g)
        except ZeroDivisionError:
            break
        x = [x[k] + dx[k] for k in range(n)]
        if max(abs(d) for d in dx) < 1e-10:
            break
    return x, resid(x)

def tool_pos(q): return fk(q)["tool"][0]
def hand_z(q): return col(fk(q)["joint_6"][1], 2)      # rod axis (= tool x)
def forearm_dir(q): return col(fk(q)["joint_4"][1], 2)  # joint_4 z = forearm axis, elbow -> wrist

def describe(q, label=""):
    f = fk(q)
    fd = forearm_dir(q)
    hz = hand_z(q)
    q1 = q[0]
    rhat = [math.cos(q1), math.sin(q1), 0]
    radial = fd[0]*rhat[0] + fd[1]*rhat[1]
    elev = math.degrees(math.asin(max(-1, min(1, fd[2]))))
    tilt = math.degrees(math.acos(max(-1, min(1, -hz[2]))))
    E = f["joint_3"][0]; P4 = f["joint_4"][0]; W = f["joint_5"][0]
    print(f"   {label} joints(deg)={[round(math.degrees(v),1) for v in q]}")
    print(f"      forearm axis={[round(v,3) for v in fd]}  elevation={elev:+.1f} deg (-90 = straight down)  horizontal part {'OUTWARD (away from base)' if radial > 1e-3 else ('INWARD (toward base)' if radial < -1e-3 else 'zero')} radial={radial:+.3f}")
    print(f"      elbow(joint_3) at r={math.hypot(E[0],E[1]):.3f} z={E[2]:.3f}; joint_4 origin z={P4[2]:.3f}; wrist(joint_5) r={math.hypot(W[0],W[1]):.3f} z={W[2]:.3f}; rod tilt from vertical={tilt:.2f} deg")
    bad = in_limits(q)
    print(f"      joint-limit violations: {bad if bad else 'none'}")
    return radial, elev, bad

# =====================================================================
print("=" * 90)
print("A. ROD STRAIGHT DOWN, joint_4 = 0, tip on the line (y=%.2f, TCP z=%.3f): all planar solutions" % (Y_LINE, Z_TCP))
print("=" * 90)
summary = {}
for x in [0.15, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40]:
    target = [x, Y_LINE, Z_TCP]
    print(f"\n tip x={x:.2f}")
    sols = []
    for q1 in (math.atan2(Y_LINE, x), math.atan2(Y_LINE, x) + PI):
        def resid(v):
            q = [q1, v[0], v[1], 0.0, v[2], 0.0]
            p = tool_pos(q); hz = hand_z(q)
            return [p[0]-target[0], p[1]-target[1], p[2]-target[2], hz[0], hz[1]]   # hz[2] = -1 follows from unit norm
        for s2, s3, s5 in itertools.product([-1.5,-0.8,0,0.5], [-1.2,-0.5,0.3,1.2], [-1.8,-0.9,0,0.9,1.8]):
            v, r = gauss_newton(resid, [s2, s3, s5])
            if norm(r) < 1e-7:
                q = [q1, v[0], v[1], 0.0, v[2], 0.0]
                # wrap angles
                q = [math.atan2(math.sin(a), math.cos(a)) for a in q]
                if hand_z(q)[2] > 0:      # rod pointing up, not down
                    continue
                if not any(max(abs(a-b) for a, b in zip(q, s)) < 1e-4 for s in sols):
                    sols.append(q)
    for i, q in enumerate(sols):
        branch = "q1 toward tip" if abs(math.atan2(math.sin(q[0]-math.atan2(Y_LINE,x)), math.cos(q[0]-math.atan2(Y_LINE,x)))) < 0.1 else "q1 turned 180 (reach over backward)"
        radial, elev, bad = describe(q, f"sol{i} [{branch}]")
    summary[x] = sols

# =====================================================================
print("\n" + "=" * 90)
print("B. TOP-DOWN VIEW: forearm vertical (q5 = 0), rod down. Max tip radius vs joint_4 and TCP z")
print("=" * 90)
def topdown_reach(z_tcp, q4, step=0.001):
    best = None
    x = 0.10
    while x < 0.45:
        target = [x, Y_LINE, z_tcp]
        found = None
        def resid(v):
            q = [v[0], v[1], v[2], q4, 0.0, 0.0]
            p = tool_pos(q)
            return [p[0]-target[0], p[1]-target[1], p[2]-target[2]]
        for s1, s2, s3 in itertools.product([-0.2, 0.3, -0.7], [-1.2, -0.6, 0.0], [-1.2, -0.6, 0.0, 0.6]):
            v, r = gauss_newton(resid, [s1, s2, s3])
            if norm(r) < 1e-7:
                q = [math.atan2(math.sin(a), math.cos(a)) for a in [v[0], v[1], v[2], q4, 0.0, 0.0]]
                if hand_z(q)[2] < -0.999 and not in_limits(q) and fk(q)["joint_3"][0][2] > 0:
                    found = q; break
        if found: best = (x, found)
        else:
            if best: break
        x += step
    return best

for z_tcp, note in [(Z_TCP, "actual pushing TCP z from steps.json"), (Z_TCP + 0.0245, "if the TCP were 2.45 cm above the rod end (claim C6 stack)"), (0.0, "TCP on the table")]:
    for q4 in [0.0, math.radians(119.75), math.radians(-119.75)]:
        b = topdown_reach(z_tcp, q4)
        if b:
            x, q = b
            E = fk(q)["joint_3"][0]
            print(f"  TCP z={z_tcp:.4f} ({note}), q4={math.degrees(q4):+.0f}: max tip x = {x:.3f} m  joints(deg)={[round(math.degrees(v),1) for v in q]}  elbow z={E[2]:.3f} r={math.hypot(E[0],E[1]):.3f}  limit-binding: {[n for n,(lo,hi) in LIM_DEG.items() if abs(math.degrees(q[int(n[-1])-1])-lo)<0.6 or abs(math.degrees(q[int(n[-1])-1])-hi)<0.6]}")
        else:
            print(f"  TCP z={z_tcp:.4f}, q4={math.degrees(q4):+.0f}: no solution")

# also: the pure geometric number without joint limits
zE = Z_TCP + 0.17 + 0.0197 + 0.0285 + 0.065
rise = zE - 0.1663
L1 = math.hypot(0.221, 0.012)
rE = math.sqrt(L1**2 - rise**2)
print(f"\n  Geometric check: elbow(joint_3) height needed = {zE:.4f}; rise above shoulder = {rise:.4f}; upper-arm horizontal reach = {rE:.4f};")
print(f"  + 0.0325 (joint_4 origin is outboard of the elbow when forearm hangs down) + 0.00925 (joint_6 x offset, at q4=0) = {rE+0.0325+0.00925:.4f} m")
print(f"  Claim C6 used stack 0.334 -> elbow 0.34 -> rise {0.34-0.1663:.4f} -> reach {math.sqrt(L1**2-(0.34-0.1663)**2):.4f}")

# =====================================================================
print("\n" + "=" * 90)
print("C. NON-ZERO joint_4 with the tip held on the line: minimum achievable rod tilt (least squares, tip position exact)")
print("=" * 90)
for x in [0.20, 0.25, 0.30, 0.35]:
    target = [x, Y_LINE, Z_TCP]
    line = f"  tip x={x:.2f}: "
    for q4d in [0, 10, 20, 30, 45, 60, 90, 119.75]:
        q4 = math.radians(q4d)
        best = None
        def resid(v):
            q = [v[0], v[1], v[2], q4, v[3], 0.0]
            p = tool_pos(q); hz = hand_z(q)
            W = 1000.0
            return [W*(p[0]-target[0]), W*(p[1]-target[1]), W*(p[2]-target[2]), hz[0], hz[1], (hz[2]+1)]
        for s2, s3, s5 in itertools.product([-1.0,-0.5], [-0.8,-0.2,0.4], [-1.5,-0.8,0,0.8,1.5]):
            v, r = gauss_newton(resid, [math.atan2(Y_LINE,x), s2, s3, s5], iters=80)
            q = [v[0], v[1], v[2], q4, v[3], 0.0]
            p = tool_pos(q)
            if norm(sub(p, target)) < 1e-4 and not in_limits(q) and hand_z(q)[2] < 0:
                tilt = math.degrees(math.acos(max(-1,min(1,-hand_z(q)[2]))))
                if best is None or tilt < best[0]:
                    best = (tilt, q)
        if best:
            fd = forearm_dir(best[1])
            line += f"q4={q4d:>6}: tilt={best[0]:5.1f} deg (forearm elev {math.degrees(math.asin(fd[2])):+.0f}) | "
        else:
            line += f"q4={q4d:>6}: none | "
    print(line)
print("\n  Analytic: with forearm axis at angle a from vertical, rod tilt >= acos(sqrt(cos^2 a + cos^2 q4 * sin^2 a)); zero only if q4=0 or a=0.")
