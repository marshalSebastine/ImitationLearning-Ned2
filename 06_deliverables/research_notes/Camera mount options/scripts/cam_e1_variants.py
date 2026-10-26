"""E1 hand-collar variants: lens further out or lower, coverage and clearance at r = 0.26 (and 0.14 for clearance)."""
import math
import camgeo, cam_options as co, cam_clearance as cl

pv = co.pose_vertical(0.26); p14 = co.pose_vertical(0.14); p42 = co.pose_vertical(0.42)
for label, forward, up in (("E1 as computed (3.0 out, hand-origin height)", 0.030, 0.0), ("E1b (4.0 out, hand-origin height)", 0.040, 0.0),
                           ("E1c (4.5 out, hand-origin height)", 0.045, 0.0), ("E1d (3.5 out, 1.2 cm below hand origin)", 0.035, -0.012)):
    par = {"hfov_deg": 52.0, "mount": "hand", "offset_m": (forward, up, 0.0), "tilt_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0}
    tilt = co.hand_mount_tilt(par, pv["joints"]); par["tilt_deg"] = tilt; cam = camgeo.Camera(par)
    scan = co.heading_scan(cam, pv["joints"], pans_deg = co.CONFIG["pan_search_offsets_deg"])
    look = co.lookahead_table(scan)
    worst = {}
    for r, pose in ((0.14, p14), (0.26, pv), (0.42, p42)):
        for pan in (0, 90, -90, 145, -145):
            j = list(pose["joints"]); j[5] = math.radians(pan)
            c = cl.clearances(cam, j, puck_heading_rad = math.radians(-pan))
            key = (r,)
            if key not in worst or c["forearm_surface_cm"] < worst[key]["forearm_surface_cm"]:
                worst[key] = {**c, "pan": pan}
    print(f"{label}: tilt {tilt} deg, window {co.window_report(cam, pv['joints'])}")
    print(f"   covered {co.intervals(scan['covered'])} ({co.coverage_width_deg(scan['covered'])} deg); look-ahead " + " / ".join(str(look[h]['furthest_cm']) for h in ('0','45','90','135','180')) + f"; puck edge row at 0: {look['0']['puck_edge_row']}")
    for key, c in worst.items():
        print(f"   r {key[0]}: worst forearm clearance {c['forearm_surface_cm']} cm at pan {c['pan']}, housing {c['housing_surface_cm']} cm, lens {c['lens_height_cm']} cm up, {c['lens_above_puck_top_cm']} cm above puck top")
