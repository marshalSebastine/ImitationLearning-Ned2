"""
Side check: option F (wrist-mounted stalk, no pan) with the camera BODY against the forearm fork and the wrist, per tip x.
Camera fixed to the wrist link: lens 60 mm ahead of the wrist origin (51 mm ahead of the rod axis), 60 or 80 mm above the
joint-5 axis, tilt -10 deg (towards the base). Body 37 x 37 x d behind the lens; stalk 10 x 10 mm from the wrist front
(x = 18 mm in the wrist frame, at the joint-5 axis height) up to the camera.
"""
import math, numpy as np
import camgeo, cam_options as co
import review_e1_body as rb

rb.XR, rb.YR, rb.ZR = (-95, 95), (-95, 95), (-130, 75)
meshes = rb.load_meshes()
xs = [0.14, 0.16, 0.18, 0.20, 0.26, 0.32, 0.38, 0.42]
for up in (60.0, 80.0):
    for d in (15.0, 25.0):
        print(f"\n=== F body: lens 51 mm ahead of rod axis, {up:.0f} mm above joint-5 axis, tilt -10, body depth {d:.0f} ===")
        for x in xs:
            pose = co.pose_vertical(x); joints = pose["joints"]; fr = camgeo.frames(joints)
            parts = []
            for name, key in (("forearm", "joint_4"), ("wrist", "joint_5")):
                o, R = fr[key]
                tri_base = meshes[name] @ R.T + o * rb.MM
                keep = (np.linalg.norm(tri_base - fr["joint_6"][0] * rb.MM, axis=2) < 170).any(axis=1)
                parts.append(rb.to_hand_frame(rb.surface_points(tri_base[keep]).reshape(-1, 3), fr["joint_6"]))
            grid = rb.signed_distance_grid(np.vstack(parts))
            # camera in the hand frame (z down): lens at X = 51, z_down = -(19.7 + up); tilt -10 (towards the base = -x)
            cam = rb.camera_points(51.0, -(19.7 + up), -10.0, d, cable=True)
            # stalk: from the wrist front face (X = 9 mm ahead of rod axis... wrist x = 18.25 -> X = 9) at z_down -19.7 up to the lens
            zs = np.arange(-(19.7 + up), -19.7 + 1e-6, 2.0)
            X = np.linspace(14.0, 51.0, len(zs))
            stalk = np.stack([np.repeat(X, 4), np.tile([-5, 5, -5, 5], len(zs)), np.repeat(zs, 4)], 1)
            dc = float(rb.sample_grid(grid, cam).min()); ds = float(rb.sample_grid(grid, stalk).min())
            print(f"x={x:.2f} (forearm {rb.math.degrees(rb.math.asin(fr['joint_4'][1][2, 2])):6.1f} deg elevation): camera body clearance {dc:6.1f} mm, stalk {ds:6.1f} mm")
