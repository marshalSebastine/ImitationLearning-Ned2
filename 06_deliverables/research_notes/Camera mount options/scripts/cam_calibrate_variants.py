"""
Variants of the pixel calibration: puck offset free or fixed, field of view free.

Usage: py -3.13 cam_calibrate_variants.py
"""

import json
import math

import numpy
from scipy.optimize import least_squares

import camgeo
import cam_calibrate


def residuals_general(parameters, data, puck_offset_fixed, hfov_fixed) -> numpy.ndarray:
    ox, oy, oz, tilt, yaw = parameters[:5]
    index = 5
    if puck_offset_fixed is None:
        puck_offset = parameters[index]; index += 1
    else:
        puck_offset = puck_offset_fixed
    hfov = parameters[index] if hfov_fixed is None else hfov_fixed
    full = [ox, oy, oz, tilt, yaw, 0.0, puck_offset]
    return cam_calibrate.pixel_residuals(full, data, hfov, False)


def main() -> None:
    fit_data = cam_calibrate.load_run("1430", tuple(range(1, 9)))
    check_data = cam_calibrate.load_run("1419", tuple(range(2, 17)))
    for item in fit_data + check_data:
        item["joints"] = cam_calibrate.vertical_rod_joints(item["tip"])
    reference_tip = (0.2556, -0.0216, 0.0378)
    reference_joints = cam_calibrate.vertical_rod_joints(reference_tip)
    variants = [
        ("hfov 52, puck offset free up to 3.5 cm", 0.0185, 0.035, None, 52.0),
        ("hfov 52, puck offset fixed 2.1 cm", None, None, 0.021, 52.0),
        ("hfov free, puck offset fixed 2.1 cm", None, None, 0.021, None),
        ("hfov free, puck offset free", 0.0185, 0.035, None, None),
        ("hfov 104, puck offset fixed 2.1 cm", None, None, 0.021, 104.0),
    ]
    for label, puck_low, puck_high, puck_fixed, hfov_fixed in variants:
        lower = [-0.05, -0.09, -0.05, -30.0, -40.0]; upper = [0.10, 0.10, 0.05, 70.0, 40.0]
        start = [0.034, 0.008, 0.007, 22.0, 1.0]
        if puck_fixed is None:
            lower.append(puck_low); upper.append(puck_high); start.append(0.024)
        if hfov_fixed is None:
            lower.append(30.0); upper.append(120.0); start.append(55.0)
        best = None
        for tilt0 in (12.0, 22.0, 32.0):
            s = list(start); s[3] = tilt0
            if hfov_fixed is None:
                for h0 in (45.0, 60.0, 90.0):
                    s[-1] = h0
                    sol = least_squares(residuals_general, s, args = (fit_data, puck_fixed, hfov_fixed), bounds = (lower, upper))
                    if best is None or sol.cost < best.cost: best = sol
            else:
                sol = least_squares(residuals_general, s, args = (fit_data, puck_fixed, hfov_fixed), bounds = (lower, upper))
                if best is None or sol.cost < best.cost: best = sol
        p = list(best.x)
        ox, oy, oz, tilt, yaw = p[:5]; i = 5
        puck = p[i] if puck_fixed is None else puck_fixed
        if puck_fixed is None: i += 1
        hfov = p[i] if hfov_fixed is None else hfov_fixed
        camera = camgeo.Camera({"hfov_deg": hfov, "offset_m": (ox, oy, oz), "tilt_deg": tilt, "yaw_deg": yaw})
        pixel_part = best.fun[:-1] * 3.0
        check = residuals_general(best.x, check_data, puck_fixed, hfov_fixed)[:-1] * 3.0
        window = cam_calibrate.window_summary(camera, reference_joints, reference_tip)
        lens = camera.pose(reference_joints)[0]
        print(f"{label}: hfov {hfov:.1f} | lens fwd of rod axis {100*(ox-0.00925):.2f} cm, up {100*oy:.2f} cm, right {100*oz:.2f} cm, "
              f"tilt {tilt:.1f} yaw {yaw:.1f} | puck offset {100*puck:.2f} cm | rms fit {numpy.sqrt(numpy.mean(pixel_part**2)):.1f} px, max {numpy.abs(pixel_part).max():.1f}, "
              f"check {numpy.sqrt(numpy.mean(check**2)):.1f} px | far-edge resid {best.fun[-1]*0.5:.2f} cm | window {window} | lens height {100*(lens[2]-camgeo.table_height(*reference_tip[:2])):.2f} cm")


if __name__ == "__main__":
    main()
