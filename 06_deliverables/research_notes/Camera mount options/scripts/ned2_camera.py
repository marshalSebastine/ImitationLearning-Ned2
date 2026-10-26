import math, json
from ned2_fk import fk, col, mv, rz, PI
# Niryo camera URDF: camera_joint parent=wrist_link, origin xyz (camH/2+0.02, 0, 0), rpy (0,0,-PI/2+10deg)
def cam(q):
    p, R = fk(q)["joint_5"]          # wrist_link frame (after joint_5 rotation)
    Rc = [[sum(R[i][k]*rz(-PI/2+math.radians(10))[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return p, R, Rc
def show(label, q):
    p, Rw, Rc = cam(q)
    q1 = q[0]; rhat = [math.cos(q1), math.sin(q1), 0]
    wx, wy = col(Rw,0), col(Rw,1)
    cx = col(Rc,0)
    az = math.degrees(math.atan2(cx[1]*rhat[0]-cx[0]*rhat[1], cx[0]*rhat[0]+cx[1]*rhat[1]))
    print(f"{label}: wrist x={[round(v,3) for v in wx]} (radial {wx[0]*rhat[0]+wx[1]*rhat[1]:+.3f}, up {wx[2]:+.3f}); wrist y={[round(v,3) for v in wy]}")
    print(f"     camera x-axis={[round(v,3) for v in cx]} -> {math.degrees(math.acos(-cx[2])):.1f} deg from straight down, azimuth {az:+.1f} deg from radial-outward; wrist axis z={p[2]:.3f}")
show("x=0.20 rod down q4=0   ", [-0.1, -0.748, -0.822, 0.0, -0.001, 0.0])
show("x=0.20 rod down q4=+90 ", [-0.1, -0.748, -0.822, math.radians(90), -0.001, 0.0])
show("x=0.35 rod down q4=0   ", [math.radians(-3.3), math.radians(-58.5), math.radians(-1.2), 0.0, math.radians(-30.3), 0.0])
show("x=0.153 step1 q4=0     ", [math.radians(-8.9), math.radians(-42.1), math.radians(-58.6), 0.0, math.radians(10.7), 0.0])
d = json.load(open(r"D:\AI_Project\03_data\raw\workspace\2026-10-25_lean20_orientation.json"))
show("lean20 high (recorded) ", d["high"]["joints"]); show("lean20 low  (recorded) ", d["low"]["joints"])
