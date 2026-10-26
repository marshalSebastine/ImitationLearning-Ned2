import math, json, itertools
from ned2_fk import fk, col, sub, norm, in_limits, LIM_DEG, PI
from ned2_analysis import gauss_newton, tool_pos, hand_z, forearm_dir

Y_LINE = -0.02

def topdown_strict(z_tcp, q4, step=0.0005):
    best = None; x = 0.12
    while x < 0.30:
        target = [x, Y_LINE, z_tcp]
        def resid(v):
            q = [v[0], v[1], v[2], q4, 0.0, 0.0]; p = tool_pos(q)
            return [p[0]-target[0], p[1]-target[1], p[2]-target[2]]
        found = None
        for s1, s2, s3 in itertools.product([-0.1, 0.3, -0.5], [-1.0, -0.6], [-1.0, -0.6, 0.0]):
            v, r = gauss_newton(resid, [s1, s2, s3])
            if norm(r) < 1e-8:
                q = [v[0], v[1], v[2], q4, 0.0, 0.0]
                if hand_z(q)[2] < -0.99999 and not in_limits(q) and fk(q)["joint_3"][0][2] > 0:
                    found = q; break
        if found: best = (x, found)
        elif best: break
        x += step
    return best

print("B (strict, rod within 0.26 deg of vertical): top-down max tip x")
for z in (0.037, 0.0615, 0.0):
    for q4d in (0, 119.75, -119.75):
        b = topdown_strict(z, math.radians(q4d))
        if b:
            x, q = b
            print(f"  TCP z={z:.4f} q4={q4d:+7.2f}: max x={x:.4f}  joints(deg)={[round(math.degrees(v),1) for v in q]}")

print("\nForearm pitch at each recorded step of 2026-10-25_1430 (rod down, q4=0, feasible branch):")
steps = json.load(open(r"D:\AI_Project\03_data\raw\workspace\push_straight\2026-10-25_1430_steps.json"))
for s in steps:
    target = [s["x"], s["y"], s["z"]]
    q1 = math.atan2(s["y"], s["x"])
    def resid(v):
        q = [q1, v[0], v[1], 0.0, v[2], 0.0]; p = tool_pos(q); hz = hand_z(q)
        return [p[0]-target[0], p[1]-target[1], p[2]-target[2], hz[0], hz[1]]
    sol = None
    for s2, s3, s5 in itertools.product([-0.8,-1.2], [-1.0,-0.4,0.3], [-0.8,0,0.3]):
        v, r = gauss_newton(resid, [s2, s3, s5])
        q = [q1, v[0], v[1], 0.0, v[2], 0.0]
        if norm(r) < 1e-8 and not in_limits(q) and hand_z(q)[2] < 0:
            sol = q; break
    fd = forearm_dir(sol)
    el = math.degrees(math.asin(fd[2]))
    rad = fd[0]*math.cos(q1) + fd[1]*math.sin(q1)
    print(f"  step {s['step']:2d} tip x={s['x']:.4f}: forearm {90+el:5.1f} deg from vertical, leaning {'OUT' if rad>0 else 'IN '}; joints(deg)={[round(math.degrees(v),1) for v in sol]}")

print("\nJoint-4 'camera yaw' band: forearm angle from vertical for x = 0.16..0.26")
for x in [0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26]:
    target = [x, Y_LINE, 0.037]; q1 = math.atan2(Y_LINE, x)
    def resid(v):
        q = [q1, v[0], v[1], 0.0, v[2], 0.0]; p = tool_pos(q); hz = hand_z(q)
        return [p[0]-target[0], p[1]-target[1], p[2]-target[2], hz[0], hz[1]]
    for s2, s3, s5 in itertools.product([-0.8], [-1.0,-0.6], [0.0]):
        v, r = gauss_newton(resid, [s2, s3, s5])
        q = [q1, v[0], v[1], 0.0, v[2], 0.0]
        if norm(r) < 1e-8 and not in_limits(q) and hand_z(q)[2] < 0: break
    fd = forearm_dir(q); a = 90 + math.degrees(math.asin(fd[2]))
    tilt45 = math.degrees(math.acos(math.sqrt(math.cos(math.radians(a))**2 + math.cos(math.radians(45))**2*math.sin(math.radians(a))**2)))
    tilt90 = math.degrees(math.acos(math.sqrt(math.cos(math.radians(a))**2)))
    print(f"  x={x:.2f}: forearm {a:4.1f} deg from vertical -> min rod tilt for q4=45: {tilt45:4.1f} deg, q4=90: {tilt90:4.1f} deg; joints(deg)={[round(math.degrees(v),1) for v in q]} (rad {[round(v,3) for v in q]})")
