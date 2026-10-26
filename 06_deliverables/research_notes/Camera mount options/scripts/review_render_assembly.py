"""Render forearm, wrist and hand meshes (from review_mesh_*_mm.npy, URDF frames) assembled at joint 5 = 0, joint 6 = 0."""
import math, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def rx(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
M2L = rz(-math.pi/2)
forearm = np.load("review_mesh_forearm_mm.npy") @ M2L.T
wrist_l = np.load("review_mesh_wrist_mm.npy") @ M2L.T
hand_l = np.load("review_mesh_hand_mm.npy") @ M2L.T
R5 = rx(-math.pi/2); t5 = np.array([0,0,170.0])
R6 = rx(math.pi/2); t6 = np.array([9.25,-19.7,0.0])
wrist_f = wrist_l @ R5.T + t5
hand_f = (hand_l @ R6.T + t6) @ R5.T + t5
hand_origin_f = (t6 @ R5.T) + t5
print("hand origin in forearm frame", hand_origin_f.round(2), "rod dir", (R5 @ R6 @ np.array([0,0,1.0])).round(3))
print("wrist in forearm frame bbox", wrist_f.reshape(-1,3).min(0).round(1), wrist_f.reshape(-1,3).max(0).round(1))
print("hand in forearm frame bbox", hand_f.reshape(-1,3).min(0).round(1), hand_f.reshape(-1,3).max(0).round(1))
fig, axes = plt.subplots(1, 3, figsize=(24, 9))
views = [((0, 2), "forearm x (= wrist x, radial) vs z (along forearm; hand points +z)"),
         ((1, 2), "forearm y (= joint-5 axis = sideways) vs z"), ((0, 1), "x vs y, looking along the forearm from the elbow")]
for ax, ((i, j), title) in zip(axes, views):
    for tri, colour, label in ((forearm, "tab:blue", "forearm"), (wrist_f, "tab:orange", "wrist"), (hand_f, "tab:green", "hand")):
        sel = tri if label != "forearm" else tri[(tri[:, :, 2] > 95).any(axis=1)]
        for t in sel:
            ax.plot(list(t[:, i]) + [t[0, i]], list(t[:, j]) + [t[0, j]], color=colour, linewidth=0.35, alpha=0.6)
    ax.plot([hand_origin_f[i]], [hand_origin_f[j]], "k+", markersize=14, markeredgewidth=2)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10); ax.grid(True, linewidth=0.3)
    ax.set_xlabel("xyz"[i] + " mm"); ax.set_ylabel("xyz"[j] + " mm")
fig.tight_layout(); fig.savefig("review_assembly_views.png", dpi=100)
print("written")
