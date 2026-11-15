"""
Verifier's centring measure (11.11.26): the sideways offset between the puck's foot and the line's middle from ONE
picture, with the puck and the line both taken from that same picture. An error of the camera model then moves the
puck and the line together, so it nearly cancels; the price is that the line under and just beyond the puck is hidden
(the puck is 29 mm tall), so the line's middle has to be followed back to the puck from where it is seen.

    nearest_ink_m       how far the puck's centre is from the nearest line pixel seen in the picture, on the table;
    offset              the distance from the puck's centre to a curve fitted to the line's middle near the puck;
    extrapolated_m      how far beyond the seen line that curve had to be followed (0: the line is seen beside the
                        puck; then the offset is as good as the puck fit).

Nothing connects to the robot.
"""

from typing import Dict, Optional, Sequence

import numpy

import end_tools
import my_measure

LOOK_M = 0.06
BIN_M = 0.002


def same_picture_offset(picture_bgr: numpy.ndarray, joints: Sequence[float], camera: Optional[Dict], travel_xy: Optional[numpy.ndarray] = None) -> Optional[Dict]:
    """
    The puck's sideways offset from the line's middle, both from this one picture.

    Returns
    -------
    dict or None
        None when no puck or no line near it is found. Otherwise "puck_xy", "offset_mm" (positive: the puck is left of
        the line looking along the travel; unsigned without a travel), "extrapolated_mm", "nearest_ink_mm" (distance to
        the nearest seen line pixel), "seen_mm" (the length of line used), "bend_per_m" (the fitted curve's bend) and
        "spread_mm" (how well the line's middle lies on one curve).
    """

    seen = end_tools.line_points_on_table(picture_bgr, joints, camera)
    puck = seen["puck"]
    if puck is None or len(seen["points"]) == 0:
        return None
    points, components = seen["points"], seen["components"]
    is_finite = numpy.all(numpy.isfinite(points), axis = 1)
    points, components = points[is_finite], components[is_finite]
    distance = numpy.hypot(points[:, 0] - puck["xy"][0], points[:, 1] - puck["xy"][1])
    is_near = distance <= LOOK_M
    if is_near.sum() < 150:
        return None
    # the component with the pixel nearest to the puck
    nearest_label = components[is_near][numpy.argmin(distance[is_near])]
    own = points[is_near & (components == nearest_label)]
    if len(own) < 150:
        return None
    nearest_ink = float(numpy.hypot(own[:, 0] - puck["xy"][0], own[:, 1] - puck["xy"][1]).min())
    centre = own.mean(axis = 0)
    _, _, axes = numpy.linalg.svd(own - centre[None, :], full_matrices = False)
    along_axis = axes[0]
    if travel_xy is not None and float(along_axis @ travel_xy) < 0.0:
        along_axis = -along_axis
    left_axis = numpy.array([-along_axis[1], along_axis[0]])
    along = (own - centre[None, :]) @ along_axis
    across = (own - centre[None, :]) @ left_axis
    bins = numpy.floor((along - along.min()) / BIN_M).astype(int)
    middle_along, middle_across, weights = [], [], []
    for index in numpy.unique(bins):
        is_bin = bins == index
        width = float(across[is_bin].max() - across[is_bin].min()) if is_bin.sum() > 3 else 0.0
        # a slice across the line counts when it shows the line's width (about 9 mm in my mask), not a cut-off part
        if 0.0075 <= width <= 0.0125:
            middle_along.append(float(along[is_bin].mean()))
            middle_across.append(float((across[is_bin].max() + across[is_bin].min()) / 2.0))
            weights.append(1.0)
    if len(middle_along) < 5:
        return None
    middle_along, middle_across = numpy.array(middle_along), numpy.array(middle_across)
    span = float(middle_along.max() - middle_along.min())
    degree = 2 if span >= 0.025 else 1
    curve = numpy.poly1d(numpy.polyfit(middle_along, middle_across, degree))
    spread = float(numpy.sqrt(numpy.mean((curve(middle_along) - middle_across) ** 2)))
    samples = numpy.arange(middle_along.min() - 0.06, middle_along.max() + 0.06, 0.00025)
    curve_points = numpy.column_stack([samples, curve(samples)])
    puck_local = numpy.array([float((puck["xy"] - centre) @ along_axis), float((puck["xy"] - centre) @ left_axis)])
    gaps = numpy.hypot(curve_points[:, 0] - puck_local[0], curve_points[:, 1] - puck_local[1])
    foot = int(numpy.argmin(gaps))
    slope = float(curve.deriv()(samples[foot]))
    tangent = numpy.array([1.0, slope]) / numpy.hypot(1.0, slope)
    relative = puck_local - curve_points[foot]
    signed = float(tangent[0] * relative[1] - tangent[1] * relative[0])
    extrapolated = float(max(0.0, middle_along.min() - samples[foot], samples[foot] - middle_along.max()))
    return {"puck_xy": puck["xy"], "offset_mm": 1000.0 * (signed if travel_xy is not None else abs(signed)), "extrapolated_mm": 1000.0 * extrapolated,
            "nearest_ink_mm": 1000.0 * nearest_ink, "seen_mm": 1000.0 * span, "bend_per_m": float(abs(curve.deriv().deriv()(samples[foot]))) if degree == 2 else 0.0,
            "spread_mm": 1000.0 * spread, "residual_px": puck["residual_px"]}
