"""
Sceptical re-check of option E1 with the CAMERA BODY (not the lens point).

Hand frame of the URDF (joint_6 after rotation): x outward at pan 0, y sideways, z DOWN the rod; origin = hand face
(joint-6 origin, 59.5 mm above the paper while pushing). Heights are reported as -z (up).

Fixed bodies (do not pan): forearm mesh (joint_4 frame), wrist mesh (joint_5 frame); placed per tip x with the
validated FK. Panning bodies (fixed to the hand): camera box, cable boot, collar ring + arm, magnet block.
Puck: cylinder r 12, h 29 on the paper, centre anywhere 17 to 33 mm from the rod axis at ANY azimuth (the push
heading is arbitrary relative to the pan), so the body is checked against the swept annulus.
"""
import math, sys, json
import numpy as np
from scipy import ndimage
import camgeo, cam_options as co

MM = 1000.0
RES = 1.0                                   # voxel mm
XR, YR, ZR = (-95, 95), (-95, 95), (-80, 75)   # hand frame, z DOWN (mm): z = -80 is 80 mm above the hand face
HAND_FACE_ABOVE_PAPER = 59.5                # 5 + 26 + 28.5 (team: tip 5 mm up, block 26 mm, rod 28.5 mm (STEP))
PUCK_TOP_Z = HAND_FACE_ABOVE_PAPER - 29.0   # z (down) of the puck top in the hand frame = +30.5
PUCK_R_MIN, PUCK_R_MAX = 17.0 - 1.5, 33.0 + 12.0        # puck annulus from the rod axis (centre 18.5 to 33 fitted/contact), mm
BLOCK_TOP_Z, BLOCK_R = 28.5, 27.6           # block top at the rod end; circumscribed radius of the 41 x 32 footprint (STEP bbox)

def rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
M2L = rz(-math.pi / 2)

def load_meshes():
    return {"forearm": np.load("review_mesh_forearm_mm.npy") @ M2L.T,
            "wrist": np.load("review_mesh_wrist_mm.npy") @ M2L.T,
            "hand": np.load("review_mesh_hand_mm.npy") @ M2L.T}

def surface_points(tri, spacing=0.7):
    out = []
    for t in tri:
        a, b, c = t
        n = max(1, int(math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing)))
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        m = (i + j) <= n
        u, v = i[m] / n, j[m] / n
        out.append(a + u[:, None] * (b - a) + v[:, None] * (c - a))
    return np.vstack(out)

def to_hand_frame(points_base, frame6):
    o, R = frame6
    return (points_base - o * MM) @ R          # R columns are hand axes in base; p_h = R^T (p - o)

def signed_distance_grid(points_mm):
    """Voxel grid (hand frame, mm) with signed distance: positive outside, negative inside the filled solids."""
    shape = (int((XR[1] - XR[0]) / RES) + 1, int((YR[1] - YR[0]) / RES) + 1, int((ZR[1] - ZR[0]) / RES) + 1)
    occ = np.zeros(shape, dtype=bool)
    idx = np.round((points_mm - np.array([XR[0], YR[0], ZR[0]])) / RES).astype(int)
    ok = (idx >= 0).all(1) & (idx[:, 0] < shape[0]) & (idx[:, 1] < shape[1]) & (idx[:, 2] < shape[2])
    idx = idx[ok]
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ = ndimage.binary_fill_holes(occ)
    outside = ndimage.distance_transform_edt(~occ) * RES
    inside = ndimage.distance_transform_edt(occ) * RES
    return outside - inside

def sample_grid(grid, pts):
    coords = ((pts - np.array([XR[0], YR[0], ZR[0]])) / RES).T
    return ndimage.map_coordinates(grid, coords, order=1, mode="nearest")

# ----------------------------------------------------------------------------- the camera body (hand frame, pan 0)
def camera_points(L, h, tilt_deg, depth, face=37.0, cable=True, spacing=2.5, model="box"):
    """
    Points on the camera envelope. Lens at (L, 0, +h) (h = mm BELOW the hand face, z down). Optical axis tilted
    tilt_deg from straight down towards +x. Body = face x face x depth box behind the lens (lens centred on the front
    face); with model="board": a 14 x 14 x 12 lens holder then a 38 x 38 x 8 board slab (bare ELP board).
    Cable boot: 20 mm long, 8 mm wide, leaving the body's top edge at mid-depth, along +u.
    """
    t = math.radians(tilt_deg)
    f = np.array([math.sin(t), 0.0, math.cos(t)])      # optical axis (z down): forward-and-down
    u = np.array([math.cos(t), 0.0, -math.sin(t)])     # face up direction (towards +x and up)
    r = np.array([0.0, 1.0, 0.0])
    lens = np.array([L, 0.0, h])
    pts = []
    def box(centre_front, half_u, half_r, dep):
        us = np.arange(-half_u, half_u + 1e-6, spacing); rs = np.arange(-half_r, half_r + 1e-6, spacing); ds = np.arange(0, dep + 1e-6, spacing)
        U, Rr = np.meshgrid(us, rs, indexing="ij")
        for dd in (0.0, dep):                                    # front and back faces
            pts.append(centre_front + U.ravel()[:, None] * u + Rr.ravel()[:, None] * r - dd * f)
        U, D = np.meshgrid(us, ds, indexing="ij")
        for rr in (-half_r, half_r):                             # side faces
            pts.append(centre_front + U.ravel()[:, None] * u + rr * r - D.ravel()[:, None] * f)
        Rr, D = np.meshgrid(rs, ds, indexing="ij")
        for uu in (-half_u, half_u):                             # top and bottom faces
            pts.append(centre_front + uu * u + Rr.ravel()[:, None] * r - D.ravel()[:, None] * f)
    if model == "box":
        box(lens, face / 2, face / 2, depth)
        boot_base = lens + (face / 2) * u - (depth / 2) * f
    else:
        box(lens, 7.0, 7.0, 12.0)
        box(lens - 12.0 * f, 19.0, 19.0, 8.0)
        boot_base = lens - 16.0 * f + 19.0 * u
    if cable:
        ls = np.arange(0, 20 + 1e-6, 2.0); ws = np.arange(-4, 4 + 1e-6, 2.0)
        Ls, Ws = np.meshgrid(ls, ws, indexing="ij")
        pts.append(boot_base + Ls.ravel()[:, None] * u + Ws.ravel()[:, None] * r)
        pts.append(boot_base + Ls.ravel()[:, None] * u + Ws.ravel()[:, None] * f)
    return np.vstack(pts)

def collar_points(z_top=10.0, z_bottom=24.0, r_out=14.5, L=30.0, h=0.0):
    """Split-ring collar on the 21 mm rod and a 10 x 8 mm arm to the camera (hand frame, pan 0)."""
    ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    zs = np.arange(z_top, z_bottom + 1e-6, 2.0)
    A, Zs = np.meshgrid(ang, zs, indexing="ij")
    ring = np.stack([r_out * np.cos(A.ravel()), r_out * np.sin(A.ravel()), Zs.ravel()], 1)
    xs = np.arange(r_out, max(r_out, L) + 1e-6, 2.0)
    ys = np.array([-5.0, 5.0]); zz = np.array([z_top + 2, z_top + 10])
    X, Y, Z = np.meshgrid(xs, ys, zz, indexing="ij")
    arm = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    return np.vstack([ring, arm])

def pan_points(pts, pan_deg):
    return pts @ rz(math.radians(pan_deg)).T

# ----------------------------------------------------------------------------- analytic checks (pan independent)
def puck_clearance(pts):
    """Smallest vertical gap between body points over the puck annulus and the puck top (mm); the body sweeps all azimuths."""
    rad = np.hypot(pts[:, 0], pts[:, 1])
    over = (rad >= PUCK_R_MIN) & (rad <= PUCK_R_MAX)
    if not over.any():
        return float("inf")
    return float(PUCK_TOP_Z - pts[over, 2].max())

def block_clearance(pts):
    rad = np.hypot(pts[:, 0], pts[:, 1])
    over = rad <= BLOCK_R + 2.0
    if not over.any():
        return float("inf")
    return float(BLOCK_TOP_Z - pts[over, 2].max())

def paper_clearance(pts):
    return float(HAND_FACE_ABOVE_PAPER - pts[:, 2].max())

# ----------------------------------------------------------------------------- fixed-body grids per tip x
def build_grids(xs, meshes):
    grids = {}
    for x in xs:
        pose = co.pose_vertical(x)
        joints = pose["joints"]
        fr = camgeo.frames(joints)
        parts = []
        for name, key in (("forearm", "joint_4"), ("wrist", "joint_5")):
            o, R = fr[key]
            tri_base = meshes[name] @ R.T + o * MM               # mm in base frame
            keep = (np.linalg.norm(tri_base - fr["joint_6"][0] * MM, axis=2) < 140).any(axis=1)
            parts.append(to_hand_frame(surface_points(tri_base[keep]).reshape(-1, 3), fr["joint_6"]))
        pts = np.vstack(parts)
        grid = signed_distance_grid(pts)
        radial = np.array([x, 0, 0]) / x
        wx = fr["joint_5"][1][:, 0]
        grids[x] = {"grid": grid, "joints": joints, "forearm_lean_deg": round(math.degrees(math.asin(fr["joint_4"][1][2, 2])), 1),
                    "j5_deg": round(math.degrees(joints[4]), 1), "wrist_x_dot_radial": round(float(wx @ radial), 3),
                    "lowest_fixed_point_below_face_mm": round(float(pts[:, 2].max()), 1),
                    "lowest_point_xy": pts[np.argmax(pts[:, 2]), :2].round(1).tolist()}
        print(f"x={x}: j5={grids[x]['j5_deg']} forearm lean={grids[x]['forearm_lean_deg']} wrist x.radial={grids[x]['wrist_x_dot_radial']} "
              f"lowest fixed point {grids[x]['lowest_fixed_point_below_face_mm']} mm below hand face at xy {grids[x]['lowest_point_xy']}", flush=True)
    return grids

def sweep(grids, body, pans):
    """Min signed distance of the panning body to the fixed bodies per (x, pan)."""
    table = {}
    for x, g in grids.items():
        row = []
        for p in pans:
            row.append(float(sample_grid(g["grid"], pan_points(body, p)).min()))
        table[x] = np.array(row)
    return table

def free_range(dists, pans, margin):
    """Largest symmetric-about-zero contiguous pan range with clearance >= margin at every pan inside it."""
    ok = dists >= margin
    i0 = int(np.argmin(np.abs(pans)))
    if not ok[i0]:
        return None
    lo = i0
    while lo - 1 >= 0 and ok[lo - 1]: lo -= 1
    hi = i0
    while hi + 1 < len(pans) and ok[hi + 1]: hi += 1
    return (float(pans[lo]), float(pans[hi]))

if __name__ == "__main__":
    meshes = load_meshes()
    xs = [0.14, 0.20, 0.26, 0.32, 0.38, 0.42]
    grids = build_grids(xs, meshes)
    pans = np.arange(-145, 145.01, 2.5)
    np.save("review_pans.npy", pans)
    results = {}
    cases = []
    for model in ("box", "board"):
        for d in ((15, 20, 25) if model == "box" else (20,)):
            for tilt in (30, 20, 10, 0):
                for L in (30, 35, 40, 45, 50, 55):
                    for h in range(0, 42, 2):
                        cases.append((model, d, tilt, L, h))
    print(len(cases), "cases", flush=True)
    for model, d, tilt, L, h in cases:
        cam = camera_points(L, h, tilt, d, cable=True, model=model)
        cam_nocable = camera_points(L, h, tilt, d, cable=False, model=model)
        col = collar_points(L=L)
        tab = sweep(grids, cam_nocable, pans)
        tab_c = sweep(grids, cam, pans)
        worst = min(float(v.min()) for v in tab.values())
        worst_c = min(float(v.min()) for v in tab_c.values())
        worst120 = min(float(v[np.abs(pans) <= 120].min()) for v in tab.values())
        worst0 = min(float(v[np.abs(pans) < 1e-6].min()) for v in tab.values())
        ranges = {x: free_range(v, pans, 3.0) for x, v in tab.items()}
        rng = None
        if all(r is not None for r in ranges.values()):
            rng = (max(r[0] for r in ranges.values()), min(r[1] for r in ranges.values()))
        key = f"{model}_d{d}_t{tilt}_L{L}_h{h}"
        results[key] = {"model": model, "depth": d, "tilt": tilt, "L": L, "h": h,
                        "lens_above_paper_mm": HAND_FACE_ABOVE_PAPER - h,
                        "worst_arm_housing_clearance_mm_pm145": round(worst, 1), "with_cable_boot": round(worst_c, 1),
                        "worst_pm120": round(worst120, 1), "at_pan0": round(worst0, 1),
                        "free_pan_range_3mm_all_x": rng,
                        "per_x_pan_profile": {str(x): v.round(1).tolist() for x, v in tab.items()} if (h % 10 == 0 and L in (30, 40)) else None,
                        "puck_clearance_mm": round(puck_clearance(cam_nocable), 1),
                        "block_clearance_mm": round(block_clearance(np.vstack([cam_nocable, col])), 1),
                        "paper_clearance_mm": round(paper_clearance(cam_nocable), 1),
                        "body_top_above_face_mm": round(float(-cam_nocable[:, 2].min()), 1),
                        "body_bottom_below_face_mm": round(float(cam_nocable[:, 2].max()), 1),
                        "body_inner_radius_mm": round(float(np.hypot(cam_nocable[:, 0], cam_nocable[:, 1]).min()), 1),
                        "body_outer_radius_mm": round(float(np.hypot(cam_nocable[:, 0], cam_nocable[:, 1]).max()), 1)}
        if h % 10 == 0 and L in (30, 40):
            print(key, {k: v for k, v in results[key].items() if k != "per_x_pan_profile"}, flush=True)
    json.dump({"grids": {str(x): {k: v for k, v in g.items() if k != "grid"} for x, g in grids.items()}, "results": results},
              open("review_e1_body_results.json", "w"), indent=1)
    print("done")
