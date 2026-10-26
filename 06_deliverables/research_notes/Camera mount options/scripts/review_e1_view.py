"""
View consequences of the collision-free E1 variants: window, puck row, look-ahead, heading coverage (pan limited to
the collision-free range) and sheet observable fractions. Uses camgeo / cam_options (52 and 64.4 deg lenses).
Usage: py -3.13 review_e1_view.py [quick]
"""
import json, math, sys, time
import numpy
import camgeo, cam_options as co

QUICK = "quick" in sys.argv[1:]
RADII = [0.26] if QUICK else [0.14, 0.26, 0.42]
poses = {r: co.pose_vertical(r) for r in RADII}
sheets = None if QUICK else co.load_sheets()

# (label, L mm, h mm below hand face, tilt deg or "auto", symmetric pan limit deg)
CASES = [
    ("E1 as proposed (lens 30 out, hand-face height, tilt auto, pan +-145)", 30, 0, "auto", 144.96),
    ("E1 proposed geometry but pan limited to the collision-free +-5 deg", 30, 0, "auto", 5.0),
    ("E1 lens 45 out, hand-face height, pan limited to +-17 deg", 45, 0, "auto", 17.5),
    ("E1 lens 50 out, hand-face height, pan limited to +-30 deg", 50, 0, "auto", 30.0),
    ("E1 lens 55 out, hand-face height, pan limited to +-50 deg", 55, 0, "auto", 50.0),
    ("Lowered: lens 40 out, 22 below face (37.5 mm up), tilt 10, pan +-145", 40, 22, 10.0, 144.96),
    ("Lowered: lens 55 out, 22 below face (37.5 mm up), tilt 10, pan +-145", 55, 22, 10.0, 144.96),
    ("Lowered: lens 40 out, 24 below face (35.5 mm up), tilt 0, pan +-145", 40, 24, 0.0, 144.96),
    ("Lowered: lens 30 out, 24 below face (35.5 mm up), tilt 0, pan +-145", 30, 24, 0.0, 144.96),
    ("Lowered bare board: lens 55 out, 28 below face (31.5 mm up), tilt 10, pan +-145", 55, 28, 10.0, 144.96),
]

out = {}
for fov in (52.0, 64.4):
    for label, L, h, tilt, pan_limit in CASES:
        t0 = time.time()
        par = {"hfov_deg": fov, "mount": "hand", "offset_m": (L / 1000.0, -h / 1000.0, 0.0), "tilt_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0}
        if tilt == "auto":
            par["tilt_deg"] = co.hand_mount_tilt(par, poses[0.26]["joints"] if 0.26 in poses else list(poses.values())[0]["joints"])
        else:
            par["tilt_deg"] = float(tilt)
        cam = camgeo.Camera(par)
        co.CONFIG["pan_limit_deg"] = pan_limit
        rec = {"fov": fov, "L_mm": L, "h_mm_below_face": h, "tilt_deg": par["tilt_deg"], "pan_limit_deg": pan_limit, "per_radius": {}}
        coverage = {}; pans = {}
        for r, pose in poses.items():
            joints = pose["joints"]
            win = co.window_report(cam, joints)
            view0 = camgeo.view_at_heading(cam, joints, 0.0)
            scan = co.heading_scan(cam, joints, pans_deg=co.CONFIG["pan_search_offsets_deg"])
            coverage[r] = scan["covered"]; pans[r] = scan["best_pan_deg"]
            rec["per_radius"][str(r)] = {"window": win, "pan0_view": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in view0.items()},
                                         "covered": co.intervals(scan["covered"]), "covered_width_deg": co.coverage_width_deg(scan["covered"]),
                                         "lookahead": co.lookahead_table(scan)}
        if sheets is not None:
            frac = co.observable_fraction(coverage, sheets, pans)
            rec["sheets"] = {k: v["best"]["observable_fraction"] for k, v in frac.items()}
        out[f"{fov}|{label}"] = rec
        pr = rec["per_radius"][str(RADII[len(RADII)//2])]
        print(f"\n[{fov} deg] {label}\n   tilt {par['tilt_deg']} deg; window from rod axis {pr['window']['near_from_rod_axis_cm']} / {pr['window']['far_from_rod_axis_cm']} cm, "
              f"from puck centre {pr['window']['near_from_puck_centre_cm']} / {pr['window']['far_from_puck_centre_cm']} cm, widths {pr['window']['near_width_cm']} / {pr['window']['far_width_cm']} cm; lens {pr['window']['lens_height_above_table_cm']} cm up")
        print(f"   pan 0, heading 0: line visible {100*pr['pan0_view']['line_visible_m']:.1f} cm, furthest {100*pr['pan0_view']['line_furthest_m']:.1f} cm, puck edge visible {pr['pan0_view']['puck_edge_visible']} at row {pr['pan0_view']['puck_edge_row']:.0f}, puck top fraction {pr['pan0_view']['puck_top_fraction']:.2f}")
        print(f"   covered headings {pr['covered']} ({pr['covered_width_deg']} deg); look-ahead 0/45/90/135/180: " + " / ".join(str(pr['lookahead'][k]['furthest_cm']) for k in ('0','45','90','135','180')))
        if sheets is not None:
            print("   sheets:", rec["sheets"])
        print(f"   ({time.time()-t0:.0f} s)", flush=True)
json.dump(out, open("review_e1_view_results.json", "w"), indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
print("done")
