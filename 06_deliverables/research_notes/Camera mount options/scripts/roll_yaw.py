import json, math, sys
import numpy as np
sys.argv = ["x", "fk_model.json"]
src = open("geometry.py").read()
exec(src.split("# sanity")[0])
exec("def ik" + src.split("def ik")[1].split("print(\"\nPushing poses along the line")[0])

print("Joint-4 roll at the pushing pose (rod vertical solution, then only q4 changed; q1,q2,q3,q5 kept):")
print("  x_tip  theta=|q5| (rod-vs-forearm-axis)  | for q4 = 30,45,60,90,120 deg: rod tilt from vertical (deg), camera yaw change (deg), rod-end horizontal shift (cm)")
for x in [0.14, 0.18, 0.20, 0.22, 0.26, 0.30, 0.34, 0.38, 0.40]:
    q, cost = ik(np.array([x, -0.02, 0.039]))
    tip0, rod0, fa0, elbow0, wrist0, fdir0 = summary(q)
    # camera direction proxy: a unit vector fixed in the forearm frame that at q4=0 points along the forearm (elbow->wrist) direction
    F0 = frames(q); Rf0 = F0[3][:3, :3]
    cam_local = Rf0.T @ (fdir0 / np.linalg.norm(fdir0))
    out = []
    for q4 in (30, 45, 60, 90, 120):
        qq = list(q); qq[3] = math.radians(q4)
        tip, rod, fa, elbow, wrist, fdir = summary(qq)
        tilt = math.degrees(math.acos(max(-1, min(1, -rod[2]))))
        cam = frames(qq)[3][:3, :3] @ cam_local
        yaw = math.degrees(math.atan2(cam[1], cam[0]) - math.atan2(fdir0[1], fdir0[0]))
        yaw = (yaw + 180) % 360 - 180
        shift = 100 * math.hypot(tip[0] - tip0[0], tip[1] - tip0[1])
        out.append(f"q4={q4:3d}: tilt {tilt:4.1f} yaw {yaw:6.1f} shift {shift:3.1f}")
    print(f"  {x:.2f}  theta={abs(math.degrees(q[4])):4.1f}  | " + " | ".join(out))

print("\nMax |q4| (deg) that keeps the rod within 10 / 15 / 20 deg of vertical, per x:")
for x in [0.14, 0.18, 0.20, 0.22, 0.26, 0.30, 0.34, 0.38, 0.40]:
    q, cost = ik(np.array([x, -0.02, 0.039]))
    res = []
    for budget in (10, 15, 20):
        ok = 0
        for q4 in range(0, 121):
            qq = list(q); qq[3] = math.radians(q4)
            rod = summary(qq)[1]
            if math.degrees(math.acos(-rod[2])) <= budget: ok = q4
            else: break
        res.append(ok)
    print(f"  x={x:.2f}: q4 max = {res[0]:3d} / {res[1]:3d} / {res[2]:3d} deg")

print("\nHand-camera (joint 6) yaw coverage with rod vertical: q6 in +-144.96 deg -> dead zone of", 360 - 2 * 144.96, "deg centred opposite the mount's zero direction")

print("\nVertical forearm reach vs tip height (q5=0): max tip x")
def solve_vertical2(z_tip):
    def res(v):
        q = [0, v[0], v[1], 0, 0, 0]
        tip, rod, fa, elbow, wrist, fdir = summary(q)
        return [(tip[2] - z_tip) * 100, (fa[0]) * 10, (fa[1]) * 10]   # forearm roll axis vertical
    best = None
    for start in ([-0.6, -0.9], [-1.5, 0.5], [0.3, -1.3], [-1.0, 0.0], [-0.2, -1.2]):
        sol = least_squares(res, start, bounds=([math.radians(LIMITS_DEG[1][0]), math.radians(LIMITS_DEG[2][0])], [math.radians(LIMITS_DEG[1][1]), math.radians(LIMITS_DEG[2][1])]))
        if best is None or sol.cost < best.cost: best = sol
    q = [0, best.x[0], best.x[1], 0, 0, 0]
    return q, best.cost
for z in (0.039, 0.07, 0.10, 0.12):
    q, cost = solve_vertical2(z)
    tip, rod, fa, elbow, wrist, fdir = summary(q)
    print(f"  tip z {z:.3f}: tip x {tip[0]:.3f}, elbow z {elbow[2]:.3f}, q2 {math.degrees(q[1]):.1f} q3 {math.degrees(q[2]):.1f}, roll-axis {np.round(fa,3)}, cost {cost:.1e} {'OK' if cost<1e-6 else 'NO SOLUTION'}")

print("\nOblique overview pose: camera (forearm end) high, forearm at 45-60 deg: elbow and wrist heights when tip z = 0.15:")
for elev in (45, 60):
    def res(v):
        q = [0, v[0], v[1], 0, 0, 0]
        tip, rod, fa, elbow, wrist, fdir = summary(q)
        el = math.atan2(-fdir[2], math.hypot(fdir[0], fdir[1]))
        return [(tip[2] - 0.15) * 100, (el - math.radians(elev)) * 10]
    sol = min((least_squares(res, s, bounds=([math.radians(LIMITS_DEG[1][0]), math.radians(LIMITS_DEG[2][0])], [math.radians(LIMITS_DEG[1][1]), math.radians(LIMITS_DEG[2][1])])) for s in ([-0.6, -0.9], [-1.0, 0.0], [0.3, -1.3])), key=lambda s: s.cost)
    q = [0, sol.x[0], sol.x[1], 0, 0, 0]
    tip, rod, fa, elbow, wrist, fdir = summary(q)
    print(f"  forearm elev {elev}: tip ({tip[0]:.3f}, z {tip[2]:.3f}), wrist ({wrist[0]:.3f}, z {wrist[2]:.3f}), elbow ({elbow[0]:.3f}, z {elbow[2]:.3f}), q2 {math.degrees(q[1]):.0f} q3 {math.degrees(q[2]):.0f}, cost {sol.cost:.1e}")
