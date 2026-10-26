"""Pure-Python forward kinematics of the Niryo Ned2 from the URDF (ned_ros master).

Chain (parent->child, origin xyz, origin rpy, axis in the child's pre-rotation frame):
  joint_1  base->shoulder   (0,0,0.1013)        (0,0,0)        z
  joint_2  shoulder->arm    (0,0,0.065)         (pi/2,0,0)     z
  joint_3  arm->elbow       (0.012,0.221,0)     (0,0,pi/2)     z
  joint_4  elbow->forearm   (0.0325,-0.065,0)   (pi/2,0,0)     z
  joint_5  forearm->wrist   (0,0,0.17)          (-pi/2,0,0)    z
  joint_6  wrist->hand      (0.00925,-0.0197,0) (pi/2,0,0)     z
  hand_tool hand->tool      (0,0,0.0285)        (pi,-pi/2,0)   fixed
"""
import math, json, sys

PI = math.pi

def rx(a):
    c, s = math.cos(a), math.sin(a)
    return [[1,0,0],[0,c,-s],[0,s,c]]
def ry(a):
    c, s = math.cos(a), math.sin(a)
    return [[c,0,s],[0,1,0],[-s,0,c]]
def rz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c,-s,0],[s,c,0],[0,0,1]]
def mm(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def mv(a, v):
    return [sum(a[i][k]*v[k] for k in range(3)) for i in range(3)]
def add(u, v): return [u[i]+v[i] for i in range(3)]
def sub(u, v): return [u[i]-v[i] for i in range(3)]
def scale(u, s): return [x*s for x in u]
def norm(u): return math.sqrt(sum(x*x for x in u))
def unit(u): n = norm(u); return [x/n for x in u]
def rpy(r, p, y):  # URDF: R = Rz(y) Ry(p) Rx(r)
    return mm(rz(y), mm(ry(p), rx(r)))
def col(R, j): return [R[0][j], R[1][j], R[2][j]]

CHAIN = [
    ("joint_1", (0,0,0.1013),        (0,0,0),       True),
    ("joint_2", (0,0,0.065),         (PI/2,0,0),    True),
    ("joint_3", (0.012,0.221,0),     (0,0,PI/2),    True),
    ("joint_4", (0.0325,-0.065,0),   (PI/2,0,0),    True),
    ("joint_5", (0,0,0.17),          (-PI/2,0,0),   True),
    ("joint_6", (0.00925,-0.0197,0), (PI/2,0,0),    True),
    ("tool",    (0,0,0.0285),        (PI,-PI/2,0),  False),
]
LIM_DEG = {"joint_1": (-171.88,171.88), "joint_2": (-105.0,34.96), "joint_3": (-76.78,85.5),
           "joint_4": (-119.75,119.75), "joint_5": (-110.01,110.17), "joint_6": (-144.96,144.96)}

def fk(q):
    """Return dict name -> (origin position, rotation matrix) for every frame after its joint rotation."""
    R = [[1,0,0],[0,1,0],[0,0,1]]
    p = [0,0,0]
    out = {}
    qi = 0
    for name, xyz, rpy_, movable in CHAIN:
        p = add(p, mv(R, list(xyz)))
        R = mm(R, rpy(*rpy_))
        if movable:
            R = mm(R, rz(q[qi])); qi += 1
        out[name] = (p, R)
    return out

def rot_to_rpy(R):
    # URDF/ROS convention: R = Rz(y) Ry(p) Rx(r)
    p = math.atan2(-R[2][0], math.sqrt(R[0][0]**2 + R[1][0]**2))
    r = math.atan2(R[2][1], R[2][2])
    y = math.atan2(R[1][0], R[0][0])
    return r, p, y

def tool_pose(q):
    p, R = fk(q)["tool"]
    return p, rot_to_rpy(R)

def in_limits(q):
    names = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
    bad = []
    for n, v in zip(names, q):
        lo, hi = LIM_DEG[n]
        d = math.degrees(v)
        if d < lo - 1e-6 or d > hi + 1e-6:
            bad.append((n, round(d,1), (lo,hi)))
    return bad

if __name__ == "__main__":
    # --- 1. zero pose
    f = fk([0]*6)
    print("ZERO POSE frames (origin, forward dir):")
    for n in ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","tool"]:
        p, R = f[n]
        print(f"  {n:8s} origin={[round(x,4) for x in p]}  x={[round(x,3) for x in col(R,0)]} y={[round(x,3) for x in col(R,1)]} z={[round(x,3) for x in col(R,2)]}")
    print("  tool pose rpy:", [round(x,3) for x in tool_pose([0]*6)[1]])

    # --- 2. validate against recorded lean20 data
    d = json.load(open(r"D:\AI_Project\03_data\raw\workspace\2026-10-25_lean20_orientation.json"))
    print("\nVALIDATION against 2026-10-25_lean20_orientation.json:")
    for k in ("high","low"):
        q = d[k]["joints"]; rec = d[k]["pose"]
        p, (r, pch, y) = tool_pose(q)
        print(f"  {k}: recorded pos={[round(x,4) for x in rec[:3]]} rpy={[round(x,3) for x in rec[3:]]}")
        print(f"        FK       pos={[round(x,4) for x in p]} rpy={[round(r,3), round(pch,3), round(y,3)]}  pos err mm={round(1000*norm(sub(p, rec[:3])),2)}")
        fr = fk(q)
        fz = col(fr["joint_4"][1], 2)   # forearm axis (joint_4 z) direction
        tx = col(fr["tool"][1], 0)      # tool x = approach (rod) axis
        hz = col(fr["joint_6"][1], 2)
        print(f"        forearm axis dir={[round(x,3) for x in fz]}  elevation(deg from horizontal)={round(math.degrees(math.asin(fz[2])),1)}  rod axis (tool x)={[round(x,3) for x in tx]} == hand z {[round(x,3) for x in hz]}")
        print(f"        rod tilt from straight-down = {round(math.degrees(math.acos(-tx[2])),1)} deg; joints deg={[round(math.degrees(v),1) for v in q]}; limit violations={in_limits(q)}")
