"""
Trace the six path sheets and tabulate the push heading relative to "radially away from the base"
along each path, for every sheet orientation and start end that fits on the table.

Builds on trace_sheets.py (same thresholding and thinning; the walk through the skeleton is
extended to step across the crossing of the alpha loops and the dashes of sheet 06 are bridged
by a morphological closing). The 1 cm line width gives the pixel scale of each photo.

Outputs next to this script:
    sheet_paths.json      path points in cm and tangent headings for each sheet
    sheet_headings.npz    per sheet and placement: heading relative to radial (deg), radius (m),
                          segment length (m), reachable flag

Usage: py -3.13 cam_sheets.py
"""

import json
import math
import os

import cv2
import numpy
from scipy import ndimage

# Reference: the thinning and the first version of the skeleton walk are from trace_sheets.py
# (team scratchpad, 25.10.26), which implements Zhang and Suen (1984), "A fast parallel algorithm
# for thinning digital patterns", Communications of the ACM 27(3).

HERE = os.path.dirname(os.path.abspath(__file__))

SHEETS = {
    "01_straight_line": {"file": "path_01_straight_line.jpeg", "close_px": 9, "top_cut_rows": 0},
    "02_arc": {"file": "path_02_arc.jpeg", "close_px": 9, "top_cut_rows": 0},
    "03_two_curve": {"file": "path_03_two_curve.jpeg", "close_px": 9, "top_cut_rows": 0},
    "04_meander": {"file": "path_04_meander.jpeg", "close_px": 9, "top_cut_rows": 0},
    "05_alpha_loop": {"file": "path_05_alpha_loop.jpeg", "close_px": 9, "top_cut_rows": 0},
    "06_alpha_loop_dashed": {"file": "path_06_alpha_loop_dashed.jpeg", "close_px": 75, "top_cut_rows": 300}
}

PLACEMENT = {
    "sheet_folder": "D:/AI_Project/01_course/path_sheets/",
    "line_width_cm": 1.0,
    "resample_step_m": 0.002,
    "smoothing_window_points": 9,
    "centre_xy_m": (0.28, -0.02),          # centre of the path's bounding box on the table
    "rotation_step_deg": 15,
    "reach_radius_m": (0.13, 0.42),        # team: reach confirmed to 0.41 m along the middle
    "reach_min_x_m": 0.10,
    "tangent_half_baseline_points": 3          # heading from points 6 mm behind and ahead (1.2 cm baseline)
}



def tangent_along(
        points: numpy.ndarray,
        half_baseline: int
) -> numpy.ndarray:
    """Tangent vectors from a central difference over a baseline of 2 * half_baseline points."""

    count = len(points)
    ahead = numpy.clip(numpy.arange(count) + half_baseline, 0, count - 1)
    behind = numpy.clip(numpy.arange(count) - half_baseline, 0, count - 1)
    return points[ahead] - points[behind]



def zhang_suen(image: numpy.ndarray) -> numpy.ndarray:
    """Thin a binary image to a one-pixel skeleton."""

    image = image.copy().astype(numpy.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = numpy.pad(image, 1)
            P2 = padded[:-2, 1:-1]; P3 = padded[:-2, 2:]; P4 = padded[1:-1, 2:]; P5 = padded[2:, 2:]
            P6 = padded[2:, 1:-1]; P7 = padded[2:, :-2]; P8 = padded[1:-1, :-2]; P9 = padded[:-2, :-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            sequence = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((sequence[i] == 0) & (sequence[i + 1] == 1)).astype(numpy.uint8) for i in range(8))
            if step == 0:
                condition = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                condition = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            mask = (image == 1) & (B >= 2) & (B <= 6) & (A == 1) & condition
            if mask.any():
                image[mask] = 0
                changed = True
    return image



def walk_skeleton(skeleton: numpy.ndarray) -> numpy.ndarray:
    """
    Walk the skeleton from an endpoint, continuing straight across junctions and small gaps.

    At each step the unvisited skeleton pixel within four pixels that best continues the current
    direction is chosen, so a crossing (alpha loop) is passed straight through instead of ending
    the walk. The longest walk over all endpoints is returned as an array of (row, column).
    """

    rows, columns = numpy.nonzero(skeleton)
    pixels = set(zip(rows.tolist(), columns.tolist()))
    pixel_array = numpy.array(list(pixels), dtype = float)

    def neighbours(pixel):
        r, c = pixel
        return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (r + dr, c + dc) in pixels]

    endpoints = [pixel for pixel in pixels if len(neighbours(pixel)) == 1]
    if not endpoints:
        return None
    best = None
    for endpoint in endpoints:
        path = [endpoint]
        seen = {endpoint}
        direction = None
        current = endpoint
        while True:
            candidates = [n for n in neighbours(current) if n not in seen]
            if not candidates:
                # jump across a junction or a one-pixel gap: nearest unvisited pixels within 4 px
                offsets = pixel_array - numpy.array(current, dtype = float)
                distance = numpy.hypot(offsets[:, 0], offsets[:, 1])
                near = numpy.nonzero((distance <= 4.0) & (distance > 0))[0]
                candidates = [tuple(int(v) for v in pixel_array[i]) for i in near
                              if tuple(int(v) for v in pixel_array[i]) not in seen]
            if not candidates:
                break
            if direction is not None:
                def forwardness(candidate):
                    step = numpy.array(candidate, dtype = float) - numpy.array(current, dtype = float)
                    return float(step @ direction) / max(numpy.linalg.norm(step), 1e-9)
                candidates.sort(key = lambda candidate: -forwardness(candidate))
                if forwardness(candidates[0]) < -0.2:
                    break
            following = candidates[0]
            step = numpy.array(following, dtype = float) - numpy.array(current, dtype = float)
            step = step / max(numpy.linalg.norm(step), 1e-9)
            direction = step if direction is None else 0.6 * direction + 0.4 * step
            direction = direction / max(numpy.linalg.norm(direction), 1e-9)
            seen.add(following)
            path.append(following)
            current = following
        if best is None or len(path) > len(best):
            best = path
    return numpy.array(best, dtype = float)



def trace_sheet(name: str, settings: dict) -> dict:
    """Line mask, pixel scale and resampled path in centimetres for one sheet photo."""

    image = cv2.imread(PLACEMENT["sheet_folder"] + settings["file"])
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (grey < 90).astype(numpy.uint8)
    dark[:settings["top_cut_rows"], :] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    # the line width in pixels, from the distance transform of the ink before any closing
    labels, count = ndimage.label(dark)
    sizes = ndimage.sum(dark, labels, range(1, count + 1))
    ink = numpy.isin(labels, 1 + numpy.nonzero(sizes > 2_000)[0])
    distance = ndimage.distance_transform_edt(ink)
    width_px = 2.0 * numpy.percentile(distance[ink], 97)
    pixels_per_cm = width_px / PLACEMENT["line_width_cm"]
    kernel = settings["close_px"]
    closed = cv2.morphologyEx(ink.astype(numpy.uint8), cv2.MORPH_CLOSE, numpy.ones((kernel, kernel), numpy.uint8))
    labels, count = ndimage.label(closed)
    sizes = ndimage.sum(closed, labels, range(1, count + 1))
    keep = (labels == (1 + int(numpy.argmax(sizes)))).astype(numpy.uint8)
    skeleton = zhang_suen(keep)
    path_px = walk_skeleton(skeleton)
    skeleton_rows, skeleton_cols = numpy.nonzero(skeleton)
    print(f"   {name}: skeleton rows {skeleton_rows.min()}..{skeleton_rows.max()} cols {skeleton_cols.min()}..{skeleton_cols.max()}; "
          f"walk rows {path_px[:, 0].min():.0f}..{path_px[:, 0].max():.0f} cols {path_px[:, 1].min():.0f}..{path_px[:, 1].max():.0f}; "
          f"start {path_px[0]} end {path_px[-1]}")
    # to centimetres, y upwards
    points_cm = numpy.stack([path_px[:, 1] / pixels_per_cm, -path_px[:, 0] / pixels_per_cm], axis = 1)
    # resample at a constant step
    steps = numpy.linalg.norm(numpy.diff(points_cm, axis = 0), axis = 1)
    cumulative = numpy.concatenate([[0.0], numpy.cumsum(steps)])
    step_cm = 100.0 * PLACEMENT["resample_step_m"]
    samples = numpy.arange(0.0, cumulative[-1], step_cm)
    resampled = numpy.stack([numpy.interp(samples, cumulative, points_cm[:, 0]),
                             numpy.interp(samples, cumulative, points_cm[:, 1])], axis = 1)
    window = PLACEMENT["smoothing_window_points"]
    kernel_1d = numpy.ones(window) / window
    smooth = numpy.stack([numpy.convolve(numpy.pad(resampled[:, k], (window // 2, window // 2), mode = "edge"),
                                         kernel_1d, mode = "valid") for k in range(2)], axis = 1)
    tangent = tangent_along(smooth, PLACEMENT["tangent_half_baseline_points"])
    heading_deg = numpy.degrees(numpy.arctan2(tangent[:, 1], tangent[:, 0]))
    turn = numpy.degrees(numpy.abs(numpy.angle(numpy.exp(1j * numpy.radians(numpy.diff(heading_deg))))))
    sorted_headings = numpy.sort(numpy.mod(heading_deg, 360.0))
    gaps = numpy.diff(numpy.concatenate([sorted_headings, [sorted_headings[0] + 360.0]]))
    span = 360.0 - gaps.max()
    return {"name": name, "pixels_per_cm": round(float(pixels_per_cm), 2), "skeleton_px": int(skeleton.sum()),
            "traced_px": int(len(path_px)), "length_cm": round(float(cumulative[-1]), 1),
            "points_cm": smooth.round(3).tolist(), "heading_deg": heading_deg.round(2).tolist(),
            "direction_span_deg": round(float(span), 1), "max_turn_from_best_direction_deg": round(float(span / 2), 1),
            "total_turning_deg": round(float(turn.sum()), 1),
            "sharpest_turn_deg_per_cm": round(float(turn.max() / step_cm), 1)}



def placements(sheet: dict) -> dict:
    """
    Heading relative to radial, radius and reach flag along the path for every placement.

    The path's bounding-box centre goes to ``centre_xy_m``; the sheet is rotated in steps of
    ``rotation_step_deg`` and traversed from either end. The rod stands 2.1 cm behind the puck,
    the puck on the path, so the radius is that of the rod's ground point.
    """

    points = numpy.array(sheet["points_cm"]) / 100.0
    centre = 0.5 * (points.min(axis = 0) + points.max(axis = 0))
    local = points - centre
    out = {"rotation_deg": [], "reversed": [], "psi_deg": [], "radius_m": [], "segment_m": [], "reachable": [],
           "heading_world_deg": []}
    for rotation in range(0, 360, PLACEMENT["rotation_step_deg"]):
        angle = math.radians(rotation)
        rotated = local @ numpy.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        placed = rotated + numpy.array(PLACEMENT["centre_xy_m"])
        for reverse in (False, True):
            path = placed[::-1] if reverse else placed
            tangent = tangent_along(path, PLACEMENT["tangent_half_baseline_points"])
            heading = numpy.arctan2(tangent[:, 1], tangent[:, 0])
            puck = path
            rod = puck - 0.021 * numpy.stack([numpy.cos(heading), numpy.sin(heading)], axis = 1)
            radial = numpy.arctan2(rod[:, 1], rod[:, 0])
            psi = numpy.degrees(numpy.angle(numpy.exp(1j * (heading - radial))))
            radius = numpy.hypot(rod[:, 0], rod[:, 1])
            segment = numpy.full(len(path), PLACEMENT["resample_step_m"])
            reachable = ((radius >= PLACEMENT["reach_radius_m"][0]) & (radius <= PLACEMENT["reach_radius_m"][1])
                         & (rod[:, 0] >= PLACEMENT["reach_min_x_m"]))
            out["rotation_deg"].append(rotation); out["reversed"].append(reverse)
            out["psi_deg"].append(psi.round(2)); out["radius_m"].append(radius.round(4))
            out["segment_m"].append(segment); out["reachable"].append(reachable)
            out["heading_world_deg"].append(numpy.degrees(heading).round(2))
    return out



def main() -> None:

    sheets = {}
    headings = {}
    for name, settings in SHEETS.items():
        sheet = trace_sheet(name, settings)
        sheets[name] = sheet
        print(f"{name}: {sheet['pixels_per_cm']} px/cm, skeleton {sheet['skeleton_px']} px, traced {sheet['traced_px']} px, "
              f"length {sheet['length_cm']} cm, direction span {sheet['direction_span_deg']} deg "
              f"(max turn from best fixed direction {sheet['max_turn_from_best_direction_deg']}), "
              f"total turning {sheet['total_turning_deg']} deg, sharpest {sheet['sharpest_turn_deg_per_cm']} deg/cm")
        headings[name] = placements(sheet)
    # sheet 06 is the alpha loop printed as dashes; the walk stops at the dashed crossing, so its
    # placements are taken from sheet 05 (same printed shape) and the partial trace is kept for the record
    sheets["06_alpha_loop_dashed"]["placements_from"] = "05_alpha_loop"
    headings["06_alpha_loop_dashed"] = placements(sheets["05_alpha_loop"])
    print("06_alpha_loop_dashed: placements taken from 05_alpha_loop (same shape; the dashed crossing stops the walk)")
    with open(os.path.join(HERE, "sheet_paths.json"), "w") as handle:
        json.dump(sheets, handle)
    arrays = {}
    for name, item in headings.items():
        for key in ("psi_deg", "radius_m", "segment_m", "reachable", "heading_world_deg"):
            arrays[f"{name}/{key}"] = numpy.array(item[key])
        arrays[f"{name}/rotation_deg"] = numpy.array(item["rotation_deg"])
        arrays[f"{name}/reversed"] = numpy.array(item["reversed"])
    numpy.savez_compressed(os.path.join(HERE, "sheet_headings.npz"), **arrays)
    print("written sheet_paths.json and sheet_headings.npz")



if __name__ == "__main__":
    main()
