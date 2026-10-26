"""
Calibrate the wrist-camera offset and tilt from the scripted-push pictures of 25.10.26.

Two fits are made for each candidate field of view (52 and 104 degrees):

1. the four summary numbers of the measured window with the rod vertical (near edge at the
   puck centre 1.85 cm ahead of the rod axis, far edge 7 cm beyond the puck centre, 8.5 cm wide
   at the bottom, 10 cm wide at the top), fitting forward offset, up offset and tilt;
2. the pixel measurements of run 2026-10-25_1430 (line edges at rows 5 and 300, line edges
   just above the puck, puck top row, puck extent at the bottom row) at eight tip positions,
   fitting forward, up and right offsets, tilt, yaw and roll.

The pixel fit is then checked against run 1419 and against the 20-degree-lean picture of the
morning (recorded joints in 2026-10-25_lean20_orientation.json). Results go to
cam_calibration.json next to this script.

Usage: py -3.13 cam_calibrate.py [hfov_deg ...]   (default 52 104; results are merged into cam_calibration.json)
"""

import json
import math
import os

import cv2
import numpy
from scipy.optimize import least_squares

import camgeo
import ned2_fk


HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = "D:/AI_Project/03_data/raw/workspace/"

MEASURED_WINDOW = {                   # team, 25.10.26, rod vertical, magnet block on the rod
    "near_from_rod_axis_m": 0.0185,   # bottom edge at the puck centre
    "far_from_rod_axis_m": 0.0885,    # 7 cm beyond the puck centre
    "near_width_m": 0.085,
    "far_width_m": 0.100
}

LINE_START = numpy.array([0.156, -0.024])    # taught 25.10.26
LINE_END = numpy.array([0.398, -0.018])
LINE_HALF_WIDTH = 0.005
PUCK_OFFSET = camgeo.GEOMETRY["puck_offset_from_rod_axis_m"]

CALIBRATION_ROWS = (5, 300)



def measure_picture(path: str) -> dict:
    """Line edges at the calibration rows and just above the puck, puck top row and bottom extent."""

    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 60)
    counts = red.sum(axis = 1)
    rows_with_puck = numpy.where(counts >= 15)[0]
    top_row = int(rows_with_puck.min()) if len(rows_with_puck) else None
    bottom = numpy.nonzero(red[479])[0]

    def line_edges(row: int):
        pixels = image[row].astype(int)
        dark = (pixels.max(axis = 1) < 120) & (hsv[row, :, 1] < 120)
        indices = numpy.nonzero(dark)[0]
        if len(indices) < 20:
            return None
        runs = numpy.split(indices, numpy.where(numpy.diff(indices) > 1)[0] + 1)
        best = max(runs, key = len)
        if not 40 <= len(best) <= 120:
            return None
        return (int(best.min()), int(best.max()))

    edges = {row: line_edges(row) for row in CALIBRATION_ROWS}
    if top_row and top_row > 60:
        edges[top_row - 15] = line_edges(top_row - 15)
    return {"edges": edges, "puck_top_row": top_row,
            "puck_bottom_columns": (int(bottom.min()), int(bottom.max())) if len(bottom) else None}



def load_run(tag: str, steps: tuple) -> list:
    """Tip positions (reported pose) and picture measurements for the chosen steps of a run."""

    folder = WORKSPACE + "push_straight/"
    poses = {s["step"]: s for s in json.load(open(folder + f"2026-10-25_{tag}_steps.json"))}
    out = []
    for step in steps:
        pose = poses[step]
        picture = measure_picture(folder + f"2026-10-25_{tag}_step_{step:02d}.jpg")
        out.append({"step": step, "tip": (pose["x"], pose["y"], pose["z"]), "picture": picture})
    return out



def vertical_rod_joints(tip) -> list:
    """Joint angles for the rod vertical at the reported tip position (joint 4 and 6 at zero)."""

    joints, tip_error, direction_error = camgeo.solve_joints(tip, (0, 0, -1), fixed = {3: 0.0, 5: 0.0})
    assert tip_error < 1.0 and direction_error < 0.2, (tip_error, direction_error)
    return joints



def line_edge_columns(camera, pose, table_z: float, rows) -> dict:
    """Predicted left and right line-edge columns at the given picture rows."""

    direction = (LINE_END - LINE_START) / numpy.linalg.norm(LINE_END - LINE_START)
    normal = numpy.array([-direction[1], direction[0]])
    distances = numpy.linspace(-0.05, 0.35, 801)
    out = {}
    for row in rows:
        columns = []
        for side in (+1, -1):
            points = numpy.stack([LINE_START[0] + distances * direction[0] + side * LINE_HALF_WIDTH * normal[0],
                                  LINE_START[1] + distances * direction[1] + side * LINE_HALF_WIDTH * normal[1],
                                  numpy.full_like(distances, table_z)], axis = 1)
            cols, prow, depth = camera.project(pose, points)
            order = numpy.argsort(prow)
            columns.append(float(numpy.interp(row, prow[order], cols[order])))
        out[row] = (min(columns), max(columns))
    return out



def puck_surface_points(centre_xy, table_z: float) -> numpy.ndarray:
    """Dense points on the puck's top rim and side wall."""

    angles = numpy.linspace(0, 2 * math.pi, 180, endpoint = False)
    heights = numpy.linspace(0, camgeo.GEOMETRY["puck_height_m"], 30)
    ring_x = centre_xy[0] + camgeo.GEOMETRY["puck_radius_m"] * numpy.cos(angles)
    ring_y = centre_xy[1] + camgeo.GEOMETRY["puck_radius_m"] * numpy.sin(angles)
    xs = numpy.repeat(ring_x, len(heights)); ys = numpy.repeat(ring_y, len(heights))
    zs = table_z + numpy.tile(heights, len(angles))
    return numpy.stack([xs, ys, zs], axis = 1)



def predict_step(camera, joints, tip, puck_offset = PUCK_OFFSET) -> dict:
    """Predicted line-edge columns, puck top row and puck bottom-row extent for one picture."""

    pose = camera.pose(joints)
    table_z = camgeo.table_height(tip[0], tip[1])
    direction = (LINE_END - LINE_START) / numpy.linalg.norm(LINE_END - LINE_START)
    puck_centre = numpy.array(tip[:2]) + puck_offset * direction
    surface = puck_surface_points(puck_centre, table_z)
    cols, rows, depth = camera.project(pose, surface)
    top_rows = rows[surface[:, 2] > table_z + camgeo.GEOMETRY["puck_height_m"] - 1e-6]
    bottom_band = (rows >= 476) & (rows <= 482) & (depth > 0)
    bottom_columns = (float(cols[bottom_band].min()), float(cols[bottom_band].max())) if bottom_band.any() else None
    return {"pose": pose, "table_z": table_z, "puck_top_row": float(top_rows.min()),
            "puck_bottom_columns": bottom_columns}



FAR_EDGE_FROM_ROD_AXIS = {"target_m": 0.0955, "sigma_m": 0.005}   # wood enters the top row between tip x 0.3263 and 0.3351 (run 1419) and 0.3175 and 0.3324 (run 1430); paper edge at x about 0.425
PIXEL_SIGMA = {"line_edge_px": 3.0, "puck_top_px": 4.0}



def pixel_residuals(parameters, data, hfov_deg: float, fit_roll: bool) -> numpy.ndarray:
    """
    Weighted residuals (in standard deviations) of the pixel fit over all steps.

    Parameters are forward, up and right offsets (m), tilt, yaw and roll (deg) and the puck
    centre's distance from the rod axis while pushing (m). The puck's extent at the bottom row is
    not used because the puck drifts sideways along a push (team: about 1 cm per 19 cm).
    """

    ox, oy, oz, tilt, yaw = parameters[:5]
    roll = parameters[5] if fit_roll else 0.0
    puck_offset = parameters[6]
    camera = camgeo.Camera({"hfov_deg": hfov_deg, "mount": "wrist", "offset_m": (ox, oy, oz),
                            "tilt_deg": tilt, "yaw_deg": yaw, "roll_deg": roll})
    residuals = []
    for item in data:
        prediction = predict_step(camera, item["joints"], item["tip"], puck_offset)
        measured = item["picture"]
        edges = line_edge_columns(camera, prediction["pose"], prediction["table_z"],
                                  [row for row, value in measured["edges"].items() if value])
        for row, value in measured["edges"].items():
            if value:
                residuals.extend([(edges[row][0] - value[0]) / PIXEL_SIGMA["line_edge_px"],
                                  (edges[row][1] - value[1]) / PIXEL_SIGMA["line_edge_px"]])
        if measured["puck_top_row"] and measured["puck_top_row"] > 60:
            residuals.append((prediction["puck_top_row"] - measured["puck_top_row"]) / PIXEL_SIGMA["puck_top_px"])
    # far edge of the window from the rod axis (paper's far edge entering the top row)
    reference = data[len(data) // 2]
    window = camera.table_window(camera.pose(reference["joints"]), camgeo.table_height(*reference["tip"][:2]),
                                 reference["tip"][:2], math.atan2(reference["tip"][1], reference["tip"][0]))
    far = window["far_m"] if math.isfinite(window["far_m"]) else 1.0
    residuals.append((far - FAR_EDGE_FROM_ROD_AXIS["target_m"]) / FAR_EDGE_FROM_ROD_AXIS["sigma_m"])
    return numpy.array(residuals)



WINDOW_SIGMA = {"near_m": 0.010, "far_m": 0.005, "near_width_m": 0.004, "far_width_m": 0.005}
PUCK_OFFSET_PRIOR = {"mean_m": 0.021, "sigma_m": 0.003}   # block body (radius 0.9 cm) touching the puck (radius 1.2 cm)



def combined_residuals(parameters, data, hfov_deg: float) -> numpy.ndarray:
    """
    Residuals of the combined fit: pixel data of run 1430, the team's summary window numbers as
    soft constraints, and a prior on the puck centre's distance from the rod axis.

    Parameters are forward, up and right offsets (m), tilt and yaw (deg) and the puck offset (m).
    """

    ox, oy, oz, tilt, yaw, puck_offset = parameters
    pixel = pixel_residuals([ox, oy, oz, tilt, yaw, 0.0, puck_offset], data, hfov_deg, False)[:-1]
    camera = camgeo.Camera({"hfov_deg": hfov_deg, "mount": "wrist", "offset_m": (ox, oy, oz), "tilt_deg": tilt, "yaw_deg": yaw})
    reference = data[len(data) // 2]
    tip = reference["tip"]
    window = camera.table_window(camera.pose(reference["joints"]), camgeo.table_height(tip[0], tip[1]), tip[:2],
                                 math.atan2(tip[1], tip[0]))
    far = window["far_m"] if math.isfinite(window["far_m"]) else 1.0
    far_width = window["far_width_m"] if math.isfinite(window["far_width_m"]) else 1.0
    soft = [(window["near_m"] - puck_offset) / WINDOW_SIGMA["near_m"],
            (far - FAR_EDGE_FROM_ROD_AXIS["target_m"]) / WINDOW_SIGMA["far_m"],
            (window["near_width_m"] - MEASURED_WINDOW["near_width_m"]) / WINDOW_SIGMA["near_width_m"],
            (far_width - MEASURED_WINDOW["far_width_m"]) / WINDOW_SIGMA["far_width_m"],
            (puck_offset - PUCK_OFFSET_PRIOR["mean_m"]) / PUCK_OFFSET_PRIOR["sigma_m"]]
    return numpy.concatenate([pixel, soft])



def window_residuals(parameters, hfov_deg: float, joints, tip) -> numpy.ndarray:
    """Residuals (metres) of the summary-window fit: forward offset, up offset, tilt."""

    ox, oy, tilt = parameters
    camera = camgeo.Camera({"hfov_deg": hfov_deg, "mount": "wrist", "offset_m": (ox, oy, 0.0), "tilt_deg": tilt})
    pose = camera.pose(joints)
    table_z = camgeo.table_height(tip[0], tip[1])
    radial = math.atan2(tip[1], tip[0])
    window = camera.table_window(pose, table_z, tip[:2], radial)
    far = window["far_m"] if math.isfinite(window["far_m"]) else 1.0
    far_width = window["far_width_m"] if math.isfinite(window["far_width_m"]) else 1.0
    return numpy.array([window["near_m"] - MEASURED_WINDOW["near_from_rod_axis_m"],
                        far - MEASURED_WINDOW["far_from_rod_axis_m"],
                        window["near_width_m"] - MEASURED_WINDOW["near_width_m"],
                        far_width - MEASURED_WINDOW["far_width_m"]])



def window_summary(camera, joints, tip) -> dict:
    """The table window of a camera at a vertical-rod pose, relative to the rod axis, in cm."""

    pose = camera.pose(joints)
    table_z = camgeo.table_height(tip[0], tip[1])
    window = camera.table_window(pose, table_z, tip[:2], math.atan2(tip[1], tip[0]))
    return {key: round(100 * value, 2) if math.isfinite(value) else "horizon" for key, value in window.items()}



def main() -> None:

    results = {}
    reference_tip = (0.2556, -0.0216, 0.0378)                  # step 6 of run 1430 (reported)
    reference_joints = vertical_rod_joints(reference_tip)
    print("reference pose joints (deg):", [round(math.degrees(q), 1) for q in reference_joints],
          "rod tilt", round(camgeo.rod_tilt_deg(reference_joints), 2))

    fit_data = load_run("1430", tuple(range(1, 9)))
    check_data = load_run("1419", tuple(range(2, 17)))
    for item in fit_data + check_data:
        item["joints"] = vertical_rod_joints(item["tip"])

    import sys
    fovs = [float(v) for v in sys.argv[1:]] or [52.0, 104.0]
    if os.path.exists(os.path.join(HERE, "cam_calibration.json")):
        results = json.load(open(os.path.join(HERE, "cam_calibration.json")))
    for hfov in fovs:
        entry = {}
        # --- fit 1: the four summary numbers
        start = [0.0325, 0.0, 15.0] if hfov < 80 else [0.02, -0.05, 15.0]
        solution = least_squares(window_residuals, start, args = (hfov, reference_joints, reference_tip),
                                 bounds = ([-0.05, -0.09, -30.0], [0.10, 0.10, 70.0]))
        ox, oy, tilt = solution.x
        camera = camgeo.Camera({"hfov_deg": hfov, "offset_m": (ox, oy, 0.0), "tilt_deg": tilt})
        entry["window_fit"] = {"forward_cm": round(100 * ox, 2), "up_cm": round(100 * oy, 2), "right_cm": 0.0,
                               "tilt_deg": round(tilt, 1), "yaw_deg": 0.0, "roll_deg": 0.0,
                               "residual_cm": [round(100 * r, 2) for r in solution.fun],
                               "rms_cm": round(100 * math.sqrt(numpy.mean(solution.fun ** 2)), 2),
                               "window_cm": window_summary(camera, reference_joints, reference_tip),
                               "lens_height_above_table_cm": round(100 * (camera.pose(reference_joints)[0][2]
                                                                          - camgeo.table_height(*reference_tip[:2])), 2)}

        # --- fit 2: the pixel data of run 1430 (several starts; roll free, then roll fixed at zero)
        lower = [-0.05, -0.09, -0.05, -30.0, -40.0, -15.0, 0.0185]
        upper = [0.10, 0.10, 0.05, 70.0, 40.0, 15.0, 0.026]
        for label, fit_roll in (("pixel_fit", True), ("pixel_fit_roll_zero", False)):
            best = None
            for start_tilt in (tilt, tilt + 10.0, tilt - 5.0):
                for start_yaw in (0.0, 6.0):
                    start = [ox, oy, 0.0, start_tilt, start_yaw, 0.0, 0.021]
                    solution = least_squares(pixel_residuals, start, args = (fit_data, hfov, fit_roll),
                                             bounds = (lower, upper))
                    if best is None or solution.cost < best.cost:
                        best = solution
            solution = best
            ox2, oy2, oz2, tilt2, yaw2, roll2, puck_offset = solution.x
            roll2 = roll2 if fit_roll else 0.0
            camera = camgeo.Camera({"hfov_deg": hfov, "offset_m": (ox2, oy2, oz2), "tilt_deg": tilt2,
                                    "yaw_deg": yaw2, "roll_deg": roll2})
            check = pixel_residuals(solution.x, check_data, hfov, fit_roll)
            pixel_part = solution.fun[:-1] * PIXEL_SIGMA["line_edge_px"]
            lens = camera.pose(reference_joints)[0]
            entry[label] = {"forward_cm": round(100 * ox2, 2), "up_cm": round(100 * oy2, 2), "right_cm": round(100 * oz2, 2),
                            "forward_of_rod_axis_cm": round(100 * (ox2 - 0.00925), 2),
                            "tilt_deg": round(tilt2, 1), "yaw_deg": round(yaw2, 1), "roll_deg": round(roll2, 1),
                            "puck_offset_cm": round(100 * puck_offset, 2),
                            "rms_px_fit_run_1430": round(float(numpy.sqrt(numpy.mean(pixel_part ** 2))), 1),
                            "max_px_fit_run_1430": round(float(numpy.abs(pixel_part).max()), 1),
                            "rms_px_check_run_1419": round(float(numpy.sqrt(numpy.mean((check[:-1] * PIXEL_SIGMA["line_edge_px"]) ** 2))), 1),
                            "observations": int(len(solution.fun)),
                            "window_cm": window_summary(camera, reference_joints, reference_tip),
                            "lens_height_above_table_cm": round(100 * (lens[2] - camgeo.table_height(*reference_tip[:2])), 2),
                            "lens_above_wrist_axis_cm": round(100 * oy2, 2),
                            "vfov_deg": round(camera.vfov_deg(), 1)}
        ox, oy, oz, tilt, yaw, roll = (entry["pixel_fit"]["forward_cm"] / 100, entry["pixel_fit"]["up_cm"] / 100,
                                       entry["pixel_fit"]["right_cm"] / 100, entry["pixel_fit"]["tilt_deg"],
                                       entry["pixel_fit"]["yaw_deg"], entry["pixel_fit"]["roll_deg"])
        camera = camgeo.Camera({"hfov_deg": hfov, "offset_m": (ox, oy, oz), "tilt_deg": tilt, "yaw_deg": yaw, "roll_deg": roll})

        # --- fit 3: combined (pixels + summary window + puck-offset prior); this is the model used later
        lower = [-0.05, -0.09, -0.05, -30.0, -40.0, 0.0185]
        upper = [0.10, 0.10, 0.05, 70.0, 40.0, 0.035]
        best = None
        for tilt0 in (12.0, 20.0, 28.0):
            start = [ox + 0.00925 if False else entry["window_fit"]["forward_cm"] / 100, entry["window_fit"]["up_cm"] / 100,
                     0.007, tilt0, 1.0, 0.021]
            solution = least_squares(combined_residuals, start, args = (fit_data, hfov), bounds = (lower, upper))
            if best is None or solution.cost < best.cost:
                best = solution
        ox, oy, oz, tilt, yaw, puck_offset = best.x
        roll = 0.0
        camera = camgeo.Camera({"hfov_deg": hfov, "offset_m": (ox, oy, oz), "tilt_deg": tilt, "yaw_deg": yaw})
        pixel_part = best.fun[:-5] * PIXEL_SIGMA["line_edge_px"]
        soft_part = best.fun[-5:]
        check = pixel_residuals([ox, oy, oz, tilt, yaw, 0.0, puck_offset], check_data, hfov, False)[:-1] * PIXEL_SIGMA["line_edge_px"]
        lens = camera.pose(reference_joints)[0]
        entry["combined_fit"] = {"forward_cm": round(100 * ox, 2), "up_cm": round(100 * oy, 2), "right_cm": round(100 * oz, 2),
                                 "forward_of_rod_axis_cm": round(100 * (ox - 0.00925), 2),
                                 "tilt_deg": round(tilt, 1), "yaw_deg": round(yaw, 1), "roll_deg": 0.0,
                                 "puck_offset_cm": round(100 * puck_offset, 2),
                                 "rms_px_fit_run_1430": round(float(numpy.sqrt(numpy.mean(pixel_part ** 2))), 1),
                                 "max_px_fit_run_1430": round(float(numpy.abs(pixel_part).max()), 1),
                                 "rms_px_check_run_1419": round(float(numpy.sqrt(numpy.mean(check ** 2))), 1),
                                 "soft_residuals_sigma_units": {"near_edge": round(float(soft_part[0]), 2), "far_edge": round(float(soft_part[1]), 2),
                                                                "near_width": round(float(soft_part[2]), 2), "far_width": round(float(soft_part[3]), 2),
                                                                "puck_offset": round(float(soft_part[4]), 2)},
                                 "window_cm": window_summary(camera, reference_joints, reference_tip),
                                 "lens_height_above_table_cm": round(100 * (lens[2] - camgeo.table_height(*reference_tip[:2])), 2),
                                 "lens_above_wrist_axis_cm": round(100 * oy, 2),
                                 "vfov_deg": round(camera.vfov_deg(), 1),
                                 "camera_parameters": {"hfov_deg": hfov, "mount": "wrist", "offset_m": [round(ox, 5), round(oy, 5), round(oz, 5)],
                                                       "tilt_deg": round(tilt, 2), "yaw_deg": round(yaw, 2), "roll_deg": 0.0}}

        # --- check against the 20-degree lean picture (recorded joints, bare rod, x = 0.135)
        lean = json.load(open(WORKSPACE + "2026-10-25_lean20_orientation.json"))
        lean_checks = {}
        for key in ("high", "low"):
            joints = lean[key]["joints"]
            tip = lean[key]["pose"][:3]
            pose = camera.pose(joints)
            table_z = camgeo.table_height(tip[0], tip[1])
            # puck at the taught line start, 2.1 cm ahead of the rod (bare rod, start gap)
            puck_centre = LINE_START
            surface = puck_surface_points(puck_centre, table_z)
            cols, rows, depth = camera.project(pose, surface)
            top_rows = rows[surface[:, 2] > table_z + camgeo.GEOMETRY["puck_height_m"] - 1e-6]
            edges = line_edge_columns(camera, pose, table_z, [5, 160, 340])
            lean_checks[key] = {"tip_z_m": round(tip[2], 4), "rod_tilt_deg": round(camgeo.rod_tilt_deg(joints), 1),
                                "camera_axis_from_down_deg": round(math.degrees(math.acos(-pose[3][2])), 1),
                                "lens_height_above_table_cm": round(100 * (pose[0][2] - table_z), 2),
                                "puck_top_row": round(float(top_rows.min()), 0),
                                "line_width_px_rows_5_160_340": [round(edges[r][1] - edges[r][0], 0) for r in (5, 160, 340)]}
        entry["lean20_check"] = lean_checks
        entry["lean20_measured"] = {"lean_90_width_px_rows_5_160_340": [61, 65, 71], "lean_90_puck_top_row": 353,
                                    "lean_110_width_px_rows_5_160_200": [83, 81, 81], "lean_110_puck_top_row": "245 to 270"}
        results[f"hfov_{int(hfov)}"] = entry
        print(json.dumps(entry, indent = 1))

    with open(os.path.join(HERE, "cam_calibration.json"), "w") as handle:
        json.dump(results, handle, indent = 1)
    print("written", os.path.join(HERE, "cam_calibration.json"))



if __name__ == "__main__":
    main()
