"""
Compare the camera options A to G for the puck-pushing task with the calibrated camera model.

For every option, field of view and tip radius this script computes the table window, the set of
push headings (relative to radially-away-from-the-base) with at least 2 cm of line ahead visible
and with the puck visible, the look-ahead at five headings, the rod tilt the option needs, and
the fraction of each path sheet that is observable (headings from cam_sheets.py, best sheet
orientation and start end, the option's camera freedom used at every point).

Inputs next to this script: cam_calibration.json (from cam_calibrate.py), sheet_headings.npz
(from cam_sheets.py). Outputs: cam_options_results.json and cam_options_tables.md.

Usage: py -3.13 cam_options.py [quick] [--fov 52 104 64.4]   (results are merged into cam_options_results.json)
"""

import json
import math
import os
import sys
import time

import numpy

import camgeo


HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "fovs_deg": [52.0, 104.0],
    "radii_m": [0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42],
    "heading_step_deg": 3,
    "lookahead_headings_deg": [0, 45, 90, 135, 180],
    "line_visible_min_m": 0.02,
    "puck_visible_min_fraction": 0.25,          # of the top-rim points (reported only)
    "puck_criterion": "leading edge in the picture with a 12 px margin and not hidden",
    "lean_deg": [10, 20, 30],
    "roll_tilt_limit_deg": 8.0,
    "roll_step_deg": 5,
    "pan_limit_deg": 144.96,
    "pan_search_offsets_deg": list(range(-60, 61, 10)),
    "pan_move_cost_s_per_deg": 0.6 / 20.0,      # team: a small joint move costs about 0.6 s
    "step_cost_s_per_cm": 0.77,                 # team: 1 cm step, 25.10.26
    "puck_offset_m": 0.029,                     # fitted from the pictures (cam_calibrate.py combined fit); geometric contact would be 2.1 to 2.5 cm
    "hand_mount_designs": {
        "E1_collar": {"forward_m": 0.030, "up_m": 0.000, "note": "collar on the rod above the block, lens 3 cm ahead of the rod axis at the hand origin height"},
        "E2_arm": {"forward_m": 0.035, "up_m": 0.040, "note": "collar with an arm up beside the wrist housing, lens 3.5 cm ahead, 4 cm above the hand origin"}
    },
    "hand_mount_puck_row": 380,                 # the puck's far top-rim point is put at this row (lower fifth)
    "wrist_mount_grid": {
        "forward_m": [0.02, 0.04, 0.06, 0.08],
        "up_m": [0.00, 0.02, 0.04, 0.06, 0.08],
        "tilt_deg": [-10, 0, 10, 20, 30, 40, 50],
        "right_m": 0.0
    },
    "sheet_weights": {"01_straight_line": 1, "02_arc": 1, "03_two_curve": 1, "04_meander": 1, "05_alpha_loop": 1, "06_alpha_loop_dashed": 1}
}

QUICK = "quick" in sys.argv[1:]
if "--fov" in sys.argv:
    CONFIG["fovs_deg"] = [float(v) for v in sys.argv[sys.argv.index("--fov") + 1:] if v.replace(".", "", 1).isdigit()]
if QUICK:
    CONFIG["radii_m"] = [0.18, 0.26, 0.34]
    CONFIG["heading_step_deg"] = 6
    CONFIG["wrist_mount_grid"] = {"forward_m": [0.04, 0.08], "up_m": [0.0, 0.04, 0.08], "tilt_deg": [0, 20, 40], "right_m": 0.0}

HEADINGS = numpy.arange(-180, 181, CONFIG["heading_step_deg"])
camgeo.GEOMETRY["puck_offset_from_rod_axis_m"] = CONFIG["puck_offset_m"]   # one source for the puck position



# ----------------------------------------------------------------------------- cameras

def calibrated_cameras() -> dict:
    """The calibrated wrist-camera parameters per field of view (from cam_calibrate.py)."""

    data = json.load(open(os.path.join(HERE, "cam_calibration.json")))
    cameras = {}
    for fov in CONFIG["fovs_deg"]:
        entry = data[f"hfov_{int(fov)}"]
        if fov < 80:
            parameters = dict(entry["combined_fit"]["camera_parameters"])
            source = "combined fit (pixels of run 1430 + summary window + puck-offset prior)"
        else:
            fit = entry["window_fit"]
            parameters = {"hfov_deg": fov, "mount": "wrist", "offset_m": [fit["forward_cm"] / 100, fit["up_cm"] / 100, 0.0],
                          "tilt_deg": fit["tilt_deg"], "yaw_deg": 0.0, "roll_deg": 0.0}
            source = "window fit only (the pixel data cannot be fitted with this field of view, see cam_calibration.json)"
        cameras[fov] = {"parameters": parameters, "source": source}
    return cameras



# ----------------------------------------------------------------------------- poses

def tip_at(radius: float) -> numpy.ndarray:
    """Rod end on the line at this radius (y = 0 so radially-outward is +x)."""

    return numpy.array([radius, 0.0, camgeo.rod_end_height(radius, 0.0)])



def pose_vertical(radius: float) -> dict:
    """Rod vertical, joints 4 and 6 at zero (the baseline pushing pose)."""

    joints, tip_error, direction_error = camgeo.solve_joints(tip_at(radius), (0, 0, -1), fixed = {3: 0.0, 5: 0.0})
    return {"joints": joints, "tip_error_mm": tip_error, "direction_error_deg": direction_error,
            "rod_tilt_deg": camgeo.rod_tilt_deg(joints) if joints else None}



def pose_leaned(radius: float, lean_deg: float, roll_rad: float = 0.0) -> dict:
    """Rod leaned towards the base by lean_deg (tip nearer the base than the wrist), joint 4 fixed."""

    lean = math.radians(lean_deg)
    joints, tip_error, direction_error = camgeo.solve_joints(tip_at(radius), (-math.sin(lean), 0.0, -math.cos(lean)),
                                                             fixed = {3: roll_rad, 5: 0.0})
    return {"joints": joints, "tip_error_mm": tip_error, "direction_error_deg": direction_error,
            "rod_tilt_deg": camgeo.rod_tilt_deg(joints) if joints else None}



def pose_rolled(radius: float, roll_deg: float) -> dict:
    """Joint 4 at roll_deg, tip exact, rod as vertical as the arm allows (least squares on direction)."""

    joints, tip_error, direction_error = camgeo.solve_joints(tip_at(radius), (0, 0, -1), fixed = {3: math.radians(roll_deg), 5: 0.0},
                                                             position_weight = 10_000.0, direction_weight = 10.0)
    return {"joints": joints, "tip_error_mm": tip_error, "direction_error_deg": direction_error,
            "rod_tilt_deg": camgeo.rod_tilt_deg(joints) if joints else None}



def camera_azimuth_deg(camera: camgeo.Camera, joints) -> float:
    """Horizontal direction of the optical axis, degrees from +x (radially outward at y = 0)."""

    forward = camera.pose(joints)[3]
    return math.degrees(math.atan2(forward[1], forward[0]))



# ----------------------------------------------------------------------------- scans

def heading_scan(
        camera: camgeo.Camera,
        joints,
        pans_deg = None,
        include_arm: bool = True
) -> dict:
    """
    Line-ahead visibility and puck visibility for every push heading.

    With ``pans_deg`` given (hand mount), joint 6 is set to each candidate pan near the heading
    and the pan with the longest visible line (puck visible) is kept.
    """

    line_visible = numpy.zeros(len(HEADINGS)); furthest = numpy.zeros(len(HEADINGS))
    puck = numpy.zeros(len(HEADINGS)); best_pan = numpy.full(len(HEADINGS), numpy.nan)
    edge = numpy.zeros(len(HEADINGS), dtype = bool); edge_row = numpy.zeros(len(HEADINGS))
    for index, heading in enumerate(HEADINGS):
        heading_rad = math.radians(float(heading))
        if pans_deg is None:
            view = camgeo.view_at_heading(camera, joints, heading_rad, include_arm = include_arm)
            line_visible[index], furthest[index], puck[index] = view["line_visible_m"], view["line_furthest_m"], view["puck_top_fraction"]
            edge[index], edge_row[index] = view["puck_edge_visible"], view["puck_edge_row"]
            continue
        best = None
        for offset in pans_deg:
            pan = max(-CONFIG["pan_limit_deg"], min(CONFIG["pan_limit_deg"], -float(heading) + offset))
            trial = list(joints); trial[5] = math.radians(pan)
            view = camgeo.view_at_heading(camera, trial, heading_rad, include_arm = include_arm)
            ok = view["puck_edge_visible"]
            score = (ok, view["line_visible_m"], -abs(offset))
            if best is None or score > best[0]:
                best = (score, view, pan)
        view, pan = best[1], best[2]
        line_visible[index], furthest[index], puck[index], best_pan[index] = view["line_visible_m"], view["line_furthest_m"], view["puck_top_fraction"], pan
        edge[index], edge_row[index] = view["puck_edge_visible"], view["puck_edge_row"]
    covered = (line_visible >= CONFIG["line_visible_min_m"]) & edge
    return {"line_visible_m": line_visible, "line_furthest_m": furthest, "puck_top_fraction": puck,
            "puck_edge_visible": edge, "puck_edge_row": edge_row, "best_pan_deg": best_pan, "covered": covered}



def intervals(covered: numpy.ndarray) -> list:
    """Heading intervals (deg) of a boolean coverage array over HEADINGS."""

    out = []
    start = None
    for index, flag in enumerate(covered):
        if flag and start is None:
            start = index
        if (not flag or index == len(covered) - 1) and start is not None:
            end = index if flag else index - 1
            out.append([int(HEADINGS[start]), int(HEADINGS[end])])
            start = None
    return out



def coverage_width_deg(covered: numpy.ndarray) -> float:
    """Total width of the covered headings in degrees."""

    return float(covered.sum() * CONFIG["heading_step_deg"])



def lookahead_table(scan: dict) -> dict:
    """Furthest visible line point and visible length at the five reporting headings (cm)."""

    out = {}
    for heading in CONFIG["lookahead_headings_deg"]:
        index = int(numpy.argmin(numpy.abs(HEADINGS - heading)))
        out[str(heading)] = {"furthest_cm": round(100 * scan["line_furthest_m"][index], 1),
                             "visible_cm": round(100 * scan["line_visible_m"][index], 1),
                             "puck_edge_visible": bool(scan["puck_edge_visible"][index]),
                             "puck_edge_row": round(float(scan["puck_edge_row"][index]), 0),
                             "puck_top_fraction": round(float(scan["puck_top_fraction"][index]), 2)}
        if not numpy.isnan(scan["best_pan_deg"][index]):
            out[str(heading)]["pan_deg"] = round(float(scan["best_pan_deg"][index]), 0)
    return out



def window_report(camera: camgeo.Camera, joints, azimuth_deg: float = 0.0) -> dict:
    """Table window along ``azimuth_deg`` (0 = radially outward), relative to the rod axis and to the puck centre (cm)."""

    frame = camgeo.frames(joints)
    tip = frame["tool"][0]
    table_z = camgeo.table_height(tip[0], tip[1])
    azimuth = math.radians(azimuth_deg)
    window = camera.table_window(camera.pose(joints), table_z, tip[:2], azimuth)
    lens = camera.pose(joints)[0]

    def cm(value):
        return round(100 * value, 1) if math.isfinite(value) else "horizon"

    puck = 100 * CONFIG["puck_offset_m"]
    return {"near_from_rod_axis_cm": cm(window["near_m"]), "far_from_rod_axis_cm": cm(window["far_m"]),
            "near_from_puck_centre_cm": (round(100 * window["near_m"] - puck, 1) if math.isfinite(window["near_m"]) else "horizon"),
            "far_from_puck_centre_cm": (round(100 * window["far_m"] - puck, 1) if math.isfinite(window["far_m"]) else "horizon"),
            "near_width_cm": cm(window["near_width_m"]), "far_width_cm": cm(window["far_width_m"]),
            "near_sideways_cm": cm(window["near_sideways_m"]), "window_measured_along_deg": azimuth_deg,
            "camera_azimuth_deg": round(camera_azimuth_deg(camera, joints), 1),
            "camera_axis_from_down_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, -camera.pose(joints)[3][2])))), 1),
            "lens_height_above_table_cm": round(100 * (lens[2] - table_z), 1),
            "lens_ahead_of_rod_axis_cm": round(100 * float((lens[:2] - tip[:2]) @ numpy.array([math.cos(azimuth), math.sin(azimuth)])), 1)}



# ----------------------------------------------------------------------------- sheets

def load_sheets() -> dict:
    """Per-sheet placement arrays from cam_sheets.py."""

    data = numpy.load(os.path.join(HERE, "sheet_headings.npz"), allow_pickle = True)
    sheets = {}
    for key in data.files:
        name, field = key.split("/")
        sheets.setdefault(name, {})[field] = data[key]
    return sheets



def observable_fraction(
        coverage_by_radius: dict,
        sheets: dict,
        pan_by_radius: dict = None
) -> dict:
    """
    Observable fraction of each sheet's length for the best placement.

    ``coverage_by_radius`` maps radius (m) to a boolean array over HEADINGS; the nearest radius
    is used at each path point. Points the arm cannot reach count as not observable. With
    ``pan_by_radius`` the pan travel along the sheet is summed for the time cost.
    """

    radii = numpy.array(sorted(coverage_by_radius))
    out = {}
    for name, sheet in sheets.items():
        best = None
        current = None
        for index in range(len(sheet["rotation_deg"])):
            psi = sheet["psi_deg"][index]; radius = sheet["radius_m"][index]
            segment = sheet["segment_m"][index]; reachable = sheet["reachable"][index]
            nearest = radii[numpy.argmin(numpy.abs(radius[:, None] - radii[None, :]), axis = 1)]
            heading_index = numpy.clip(numpy.round((psi + 180.0) / CONFIG["heading_step_deg"]).astype(int), 0, len(HEADINGS) - 1)
            visible = numpy.array([coverage_by_radius[r][h] for r, h in zip(nearest, heading_index)]) & reachable
            fraction = float((visible * segment).sum() / segment.sum())
            reach_fraction = float((reachable * segment).sum() / segment.sum())
            pan_travel = 0.0
            if pan_by_radius is not None:
                pans = numpy.array([pan_by_radius[r][h] for r, h in zip(nearest, heading_index)])
                pans = pans[~numpy.isnan(pans)]
                pan_travel = float(numpy.abs(numpy.diff(pans)).sum()) if len(pans) > 1 else 0.0
            record = {"rotation_deg": int(sheet["rotation_deg"][index]), "reversed": bool(sheet["reversed"][index]),
                      "observable_fraction": round(fraction, 3), "reachable_fraction": round(reach_fraction, 3),
                      "length_cm": round(100 * float(segment.sum()), 1), "pan_travel_deg": round(pan_travel, 0)}
            if sheet["rotation_deg"][index] == 90 and not sheet["reversed"][index]:
                current = record
            if best is None or (fraction, reach_fraction) > (best["observable_fraction"], best["reachable_fraction"]):
                best = record
        out[name] = {"best": best, "longest_axis_along_x_forward": current}
    return out



# ----------------------------------------------------------------------------- options

def evaluate_fixed_camera(camera, poses: dict, label: str, include_arm: bool = True, pans = None) -> dict:
    """Windows, coverage and look-ahead per radius for a camera that does not depend on the policy's choice."""

    per_radius = {}
    coverage = {}
    pans_by_radius = {}
    for radius, pose in poses.items():
        if pose["joints"] is None:
            per_radius[radius] = {"infeasible": True}
            coverage[radius] = numpy.zeros(len(HEADINGS), dtype = bool)
            continue
        scan = heading_scan(camera, pose["joints"], pans_deg = pans, include_arm = include_arm)
        coverage[radius] = scan["covered"]
        pans_by_radius[radius] = scan["best_pan_deg"]
        per_radius[radius] = {"window": window_report(camera, pose["joints"]),
                              "covered_headings_deg": intervals(scan["covered"]),
                              "covered_width_deg": coverage_width_deg(scan["covered"]),
                              "puck_visible_headings_deg": intervals(scan["puck_edge_visible"]),
                              "line_2cm_headings_deg": intervals(scan["line_visible_m"] >= CONFIG["line_visible_min_m"]),
                              "lookahead": lookahead_table(scan),
                              "rod_tilt_deg": round(pose["rod_tilt_deg"], 1), "tip_error_mm": round(pose["tip_error_mm"], 2),
                              "direction_error_deg": round(pose["direction_error_deg"], 1),
                              "joints_deg": [round(math.degrees(q), 1) for q in pose["joints"]]}
    return {"label": label, "per_radius": per_radius, "coverage": coverage, "pans": pans_by_radius if pans is not None else None}



def contact_geometry(lean_deg: float) -> dict:
    """
    Where the leaned pusher meets the puck, for pushing outward (0), sideways (90) and back (180).

    The rod end stays put; the magnet tip moves towards the base and rises. The pusher's surface
    at each height above the paper is the magnet (r 0.65 cm, lowest 1 cm), the block body
    (r 0.9 cm) and, above the rod end, the rod and housing. The puck (2.9 cm tall) is touched at
    the height where the pusher surface reaches furthest towards it.
    """

    lean = math.radians(lean_deg)
    tip_shift_cm = 2.6 * math.sin(lean)
    tip_rise_cm = 2.6 * (1 - math.cos(lean))
    lowest_rim_drop_cm = 0.65 * math.sin(lean)
    out = {"magnet_tip_shift_towards_base_cm": round(tip_shift_cm, 2), "magnet_tip_rise_cm": round(tip_rise_cm, 2),
           "lowest_magnet_rim_below_tip_centre_cm": round(lowest_rim_drop_cm, 2),
           "clearance_change_at_outward_rim_cm": round(tip_rise_cm - lowest_rim_drop_cm, 2)}
    heights = numpy.linspace(0.5, 2.9, 49)       # above the paper; the magnet tip is at 0.5 cm
    for heading in (0, 90, 180):
        towards_puck = -math.cos(math.radians(heading))   # the lean displaces the pusher towards the base (-x)
        reach = []
        for h in heights:
            above_tip = h - 0.5
            radius = 0.65 if above_tip < 1.0 else (0.9 if above_tip < 2.6 else (0.9 if above_tip < 5.1 else 2.3))
            reach.append(radius + above_tip * math.tan(lean) * towards_puck)
        reach = numpy.array(reach)
        best = int(numpy.argmax(reach))
        out[f"contact_height_cm_heading_{heading}"] = round(float(heights[best]), 1)
        out[f"overhang_at_puck_top_cm_heading_{heading}"] = round(float(reach[-1] - reach[0]), 2)
    housing_bottom_above_paper = 0.5 + 2.6 + 2.5
    out["housing_shift_towards_base_at_its_bottom_cm"] = round((housing_bottom_above_paper - 0.5) * math.tan(lean), 2)
    return out



def hand_mount_tilt(camera_parameters: dict, joints) -> float:
    """Tilt that puts the puck's far top-rim point at the configured row when looking along the push."""

    best = None
    for tilt in numpy.arange(0.0, 80.0, 0.5):
        parameters = dict(camera_parameters); parameters["tilt_deg"] = float(tilt)
        camera = camgeo.Camera(parameters)
        frame = camgeo.frames(joints)
        tip = frame["tool"][0]
        table_z = camgeo.table_height(tip[0], tip[1])
        puck_centre = tip[:2] + CONFIG["puck_offset_m"] * numpy.array([1.0, 0.0])
        far_rim = numpy.array([[puck_centre[0] + camgeo.GEOMETRY["puck_radius_m"], puck_centre[1], table_z + camgeo.GEOMETRY["puck_height_m"]]])
        columns, rows, depth = camera.project(camera.pose(joints), far_rim)
        if depth[0] <= 0:
            continue
        error = abs(rows[0] - CONFIG["hand_mount_puck_row"])
        if best is None or error < best[0]:
            best = (error, float(tilt))
    return best[1]



def serialise(value):
    """Make numpy types JSON-friendly."""

    if isinstance(value, dict):
        return {str(k): serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialise(v) for v in value]
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    if isinstance(value, (numpy.floating, numpy.integer)):
        return value.item()
    if isinstance(value, (numpy.bool_,)):
        return bool(value)
    return value



def main() -> None:

    started = time.time()
    cameras = calibrated_cameras()
    sheets = load_sheets()
    results = {"config": serialise(CONFIG), "cameras": cameras, "options": {}}
    radii = CONFIG["radii_m"]

    print("poses ...")
    vertical = {r: pose_vertical(r) for r in radii}
    leaned = {lean: {r: pose_leaned(r, lean) for r in radii} for lean in CONFIG["lean_deg"]}
    for r in radii:
        print(f"  r={r}: vertical tilt {vertical[r]['rod_tilt_deg']:.2f} tip err {vertical[r]['tip_error_mm']:.2f} mm; "
              + "; ".join(f"lean {lean}: tilt {leaned[lean][r]['rod_tilt_deg']:.1f} dir err {leaned[lean][r]['direction_error_deg']:.1f} tip err {leaned[lean][r]['tip_error_mm']:.2f}" for lean in CONFIG["lean_deg"]))

    # --- option C: joint-4 roll band
    print("option C roll scan ...")
    roll_band = {}
    for r in radii:
        best_roll = 0.0; tilts = {}
        for roll in range(0, 120, CONFIG["roll_step_deg"]):
            pose = pose_rolled(r, roll)
            tilts[roll] = (pose["rod_tilt_deg"], pose["tip_error_mm"], pose["joints"])
            if pose["joints"] is not None and pose["rod_tilt_deg"] <= CONFIG["roll_tilt_limit_deg"] and pose["tip_error_mm"] < 1.0:
                best_roll = float(roll)
            else:
                break
        roll_band[r] = {"max_roll_deg": best_roll, "tilt_by_roll": {k: (round(v[0], 1), round(v[1], 2)) for k, v in tilts.items()}}
        print(f"  r={r}: max joint-4 roll with rod tilt <= {CONFIG['roll_tilt_limit_deg']} deg: {best_roll}")

    for fov in CONFIG["fovs_deg"]:
        print(f"=== field of view {fov} deg ===")
        base_parameters = cameras[fov]["parameters"]
        camera = camgeo.Camera(base_parameters)
        options = {}

        # A
        print("option A ...")
        A = evaluate_fixed_camera(camera, vertical, "A: current bracket, rod vertical")
        A["sheets"] = observable_fraction(A["coverage"], sheets)
        options["A"] = A

        # B
        for lean in CONFIG["lean_deg"]:
            print(f"option B lean {lean} ...")
            B = evaluate_fixed_camera(camera, leaned[lean], f"B: current bracket, rod leaned {lean} deg towards the base")
            B["sheets"] = observable_fraction(B["coverage"], sheets)
            B["contact"] = contact_geometry(lean)
            options[f"B{lean}"] = B

        # C: yaw by joint 4 where the rod tilt stays under the limit; the policy may pick any roll in the band
        print("option C ...")
        C_cov = {}; C_rad = {}
        for r in radii:
            max_roll = roll_band[r]["max_roll_deg"]
            rolls = sorted(set([0.0, max_roll / 2, max_roll, -max_roll / 2, -max_roll]))
            union = numpy.zeros(len(HEADINGS), dtype = bool)
            yaws = {}
            for roll in rolls:
                pose = pose_rolled(r, roll) if roll != 0 else vertical[r]
                if pose["joints"] is None:
                    continue
                scan = heading_scan(camera, pose["joints"])
                union |= scan["covered"]
                yaws[roll] = {"camera_azimuth_deg": round(camera_azimuth_deg(camera, pose["joints"]), 1),
                              "window_along_camera_azimuth": window_report(camera, pose["joints"], camera_azimuth_deg(camera, pose["joints"])),
                              "rod_tilt_deg": round(pose["rod_tilt_deg"], 1),
                              "covered_headings_deg": intervals(scan["covered"]),
                              "lookahead": lookahead_table(scan)}
            C_cov[r] = union
            C_rad[r] = {"max_roll_deg": max_roll, "by_roll": yaws, "union_headings_deg": intervals(union),
                        "union_width_deg": coverage_width_deg(union), "added_over_A_deg": coverage_width_deg(union & ~A["coverage"][r])}
        options["C"] = {"label": "C: current bracket, joint-4 roll as camera yaw (rod tilt <= 8 deg)", "per_radius": C_rad,
                        "coverage": C_cov, "roll_band": roll_band, "sheets": observable_fraction(C_cov, sheets)}

        # D: B and C combined (lean towards the base with joint 4 rolled)
        print("option D ...")
        D_all = {}
        for lean in (10, 20):
            D_cov = {}; D_rad = {}
            for r in radii:
                max_roll = roll_band[r]["max_roll_deg"]
                union = numpy.zeros(len(HEADINGS), dtype = bool)
                details = {}
                for roll in sorted(set([0.0, max_roll, -max_roll])):
                    pose = pose_leaned(r, lean, math.radians(roll))
                    if pose["joints"] is None or pose["direction_error_deg"] > 5.0 or pose["tip_error_mm"] > 1.0:
                        details[roll] = {"feasible": False, "direction_error_deg": round(pose["direction_error_deg"], 1) if pose["joints"] else None}
                        continue
                    scan = heading_scan(camera, pose["joints"])
                    union |= scan["covered"]
                    details[roll] = {"feasible": True, "rod_tilt_deg": round(pose["rod_tilt_deg"], 1),
                                     "camera_azimuth_deg": round(camera_azimuth_deg(camera, pose["joints"]), 1),
                                     "covered_headings_deg": intervals(scan["covered"]), "lookahead": lookahead_table(scan)}
                D_cov[r] = union
                D_rad[r] = {"by_roll": details, "union_headings_deg": intervals(union), "union_width_deg": coverage_width_deg(union)}
            D_all[f"D{lean}"] = {"label": f"D: lean {lean} deg towards the base plus joint-4 roll", "per_radius": D_rad,
                                 "coverage": D_cov, "sheets": observable_fraction(D_cov, sheets)}
        options.update(D_all)

        # E: hand mount with joint-6 pan
        for name, design in CONFIG["hand_mount_designs"].items():
            print(f"option {name} ...")
            parameters = {"hfov_deg": fov, "mount": "hand", "offset_m": (design["forward_m"], design["up_m"], 0.0),
                          "tilt_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0}
            tilt = hand_mount_tilt(parameters, vertical[radii[len(radii) // 2]]["joints"])
            parameters["tilt_deg"] = tilt
            hand_camera = camgeo.Camera(parameters)
            E = evaluate_fixed_camera(hand_camera, vertical, f"{name}: hand mount, pan by joint 6", pans = CONFIG["pan_search_offsets_deg"])
            E_no_arm = evaluate_fixed_camera(hand_camera, {r: vertical[r] for r in radii[::max(1, len(radii) // 3)]},
                                             "without arm occlusion", include_arm = False, pans = CONFIG["pan_search_offsets_deg"])
            E["sheets"] = observable_fraction(E["coverage"], sheets, pan_by_radius = E["pans"])
            E["design"] = {**design, "tilt_deg": tilt, "lens_height_above_table_cm": E["per_radius"][radii[0]]["window"]["lens_height_above_table_cm"]}
            E["arm_occlusion_check"] = {r: {"covered_with_arm_deg": E["per_radius"][r]["covered_width_deg"],
                                            "covered_without_arm_deg": E_no_arm["per_radius"][r]["covered_width_deg"],
                                            "lookahead_180_with_arm_cm": E["per_radius"][r]["lookahead"]["180"]["furthest_cm"],
                                            "lookahead_180_without_arm_cm": E_no_arm["per_radius"][r]["lookahead"]["180"]["furthest_cm"]}
                                        for r in E_no_arm["per_radius"]}
            for sheet_name, record in E["sheets"].items():
                best = record["best"]
                best["pan_time_if_separate_moves_s"] = round(best["pan_travel_deg"] * CONFIG["pan_move_cost_s_per_deg"], 1)
                best["push_time_s"] = round(best["length_cm"] * CONFIG["step_cost_s_per_cm"], 1)
            options[name] = E

        # F: wrist mount grid, no pan
        print("option F grid ...")
        grid = CONFIG["wrist_mount_grid"]
        F_trials = []
        reference_pose = vertical[radii[len(radii) // 2]]
        for forward in grid["forward_m"]:
            for up in grid["up_m"]:
                for tilt in grid["tilt_deg"]:
                    parameters = {"hfov_deg": fov, "mount": "wrist", "offset_m": (forward, up, grid["right_m"]),
                                  "tilt_deg": tilt, "yaw_deg": 0.0, "roll_deg": 0.0}
                    trial_camera = camgeo.Camera(parameters)
                    scan = heading_scan(trial_camera, reference_pose["joints"])
                    cov = {r: scan["covered"] for r in radii}
                    fractions = observable_fraction(cov, sheets)
                    mean_fraction = float(numpy.mean([fractions[s]["best"]["observable_fraction"] * CONFIG["sheet_weights"][s] for s in fractions]))
                    puck_at_0 = float(scan["puck_edge_visible"][int(numpy.argmin(numpy.abs(HEADINGS)))])
                    F_trials.append({"forward_m": forward, "up_m": up, "tilt_deg": tilt, "mean_fraction": round(mean_fraction, 3),
                                     "covered_width_deg": coverage_width_deg(scan["covered"]), "puck_top_fraction_at_0": round(puck_at_0, 2),
                                     "covered_headings_deg": intervals(scan["covered"]),
                                     "parameters": parameters})
        feasible = [t for t in F_trials if t["puck_top_fraction_at_0"] >= 0.5]
        best_F = max(feasible, key = lambda t: (t["mean_fraction"], t["covered_width_deg"]))
        best_camera = camgeo.Camera(best_F["parameters"])
        F = evaluate_fixed_camera(best_camera, vertical, "F: printed wrist mount, higher and tilted, no pan")
        F["sheets"] = observable_fraction(F["coverage"], sheets)
        F["best_design"] = best_F
        F["grid_top_10"] = sorted(feasible, key = lambda t: -t["mean_fraction"])[:10]
        options["F"] = F

        # G1: E2 plus a lean towards the base (joint 5), pan by joint 6
        print("option G ...")
        design = CONFIG["hand_mount_designs"]["E2_arm"]
        parameters = dict(options["E2_arm"]["design"])
        hand_parameters = {"hfov_deg": fov, "mount": "hand", "offset_m": (design["forward_m"], design["up_m"], 0.0),
                           "tilt_deg": parameters["tilt_deg"], "yaw_deg": 0.0, "roll_deg": 0.0}
        hand_camera = camgeo.Camera(hand_parameters)
        for lean in (10, 20):
            G = evaluate_fixed_camera(hand_camera, leaned[lean], f"G1: E2 hand mount plus {lean} deg lean towards the base", pans = CONFIG["pan_search_offsets_deg"])
            G["sheets"] = observable_fraction(G["coverage"], sheets, pan_by_radius = G["pans"])
            G["contact"] = contact_geometry(lean)
            options[f"G1_E2_lean{lean}"] = G
        # G2: best F mount plus a 20 degree lean
        G2 = evaluate_fixed_camera(best_camera, leaned[20], "G2: F wrist mount plus 20 deg lean towards the base")
        G2["sheets"] = observable_fraction(G2["coverage"], sheets)
        options["G2_F_lean20"] = G2

        for key, option in options.items():
            option.pop("coverage", None); option.pop("pans", None)
        results["options"][str(int(fov))] = serialise(options)
        print(f"   done fov {fov} after {time.time() - started:.0f} s")

    path = os.path.join(HERE, "cam_options_results.json")
    if os.path.exists(path):
        previous = json.load(open(path))
        previous["cameras"].update({str(k): v for k, v in results["cameras"].items()})
        previous["options"].update(results["options"])
        previous["config"] = results["config"]
        results = previous
    with open(path, "w") as handle:
        json.dump(results, handle, indent = 1)
    print("written cam_options_results.json after", round(time.time() - started), "s")



if __name__ == "__main__":
    main()
