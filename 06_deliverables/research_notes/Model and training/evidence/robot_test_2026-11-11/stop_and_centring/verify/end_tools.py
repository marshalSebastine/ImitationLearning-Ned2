"""
Verifier's end measure (11.11.26): where the puck stands at the stop against the line's round end, with my own tools.

The line's end is taken from ONE earlier picture at a time (a "reference picture", some steps before the stop, in which
the end still lies in the open beyond the puck); the puck's foot from the stop picture. Every usable reference picture
gives its own answer, so the spread over the reference pictures shows how well the camera model carries a table point
from one view to the next. Nothing connects to the robot.
"""

from typing import Dict, List, Optional, Sequence

import cv2
import numpy

import my_measure

LINE_HALF_WIDTH_M = 0.005


def line_points_on_table(picture_bgr: numpy.ndarray, joints: Sequence[float], camera: Optional[Dict], lift_m: float = 0.0) -> Dict:
    """The line's pixels of one picture (components of 120 pixels or more), their component and their places on the table."""

    found = my_measure.puck_foot(picture_bgr, joints, camera, lift_m)
    mask = my_measure.line_mask(picture_bgr, joints, camera, puck = None if found is None else found["mask"])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(numpy.uint8), connectivity = 8)
    kept = numpy.zeros(mask.shape, dtype = bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 120:
            kept |= labels == label
    rows, columns = numpy.nonzero(kept)
    pixels = numpy.column_stack([columns, rows]).astype(float)
    points = my_measure.to_table(pixels, joints, camera, 0.0, lift_m) if len(pixels) else numpy.zeros((0, 2))
    return {"pixels": pixels, "points": points, "components": labels[rows, columns], "puck": found, "mask": kept}


def end_of_line(seen: Dict, near_xy: numpy.ndarray, travel: numpy.ndarray, look_m: float = 0.07, fit_m: float = 0.018, touch_m: float = 0.04) -> Optional[Dict]:
    """
    The round end of the line near ``near_xy``: of the line components that come within ``touch_m`` of it the largest
    is taken; its far tip along the line's own direction (from the last ``fit_m`` of line before the tip, short enough
    for a curved end) is the end's tip. The centre of the round end is taken from the cap itself: the middle of the
    line pixels of the last 5 mm, moved back by 2 mm (a half disc's middle lies 0.42 radii beyond its centre), so that a
    curved line before the end does not pull it sideways.
    """

    points, components = seen["points"], seen["components"]
    is_finite = numpy.all(numpy.isfinite(points), axis = 1)
    points, components = points[is_finite], components[is_finite]
    if len(points) == 0:
        return None
    distance = numpy.hypot(points[:, 0] - near_xy[0], points[:, 1] - near_xy[1])
    best, best_size = None, 0
    for label in numpy.unique(components):
        is_own = components == label
        if float(distance[is_own].min()) <= touch_m and int(is_own.sum()) > best_size:
            best, best_size = label, int(is_own.sum())
    if best is None:
        return None
    is_kept = (components == best) & (distance <= look_m)
    points = points[is_kept]
    if len(points) < 150:
        return None
    direction = travel / numpy.linalg.norm(travel)
    if seen.get("puck") is not None:
        ahead = points.mean(axis = 0) - seen["puck"]["xy"]
        if float(numpy.linalg.norm(ahead)) > 0.01:
            direction = ahead / numpy.linalg.norm(ahead)
    for _ in range(4):
        along = (points - near_xy[None, :]) @ direction
        tip_along = float(numpy.percentile(along, 99.5))
        last = points[along >= tip_along - fit_m]
        if len(last) < 80:
            return None
        centre = last.mean(axis = 0)
        _, _, axes = numpy.linalg.svd(last - centre[None, :], full_matrices = False)
        direction = axes[0] if float(axes[0] @ direction) >= 0.0 else -axes[0]
    left = numpy.array([-direction[1], direction[0]])
    along = (points - near_xy[None, :]) @ direction
    across = (points - near_xy[None, :]) @ left
    tip_along = float(numpy.percentile(along, 99.5))
    is_before = (along >= tip_along - 0.016) & (along <= tip_along - 0.006)
    is_cap = along >= tip_along - LINE_HALF_WIDTH_M
    if is_before.sum() < 40 or is_cap.sum() < 30:
        return None
    width = float(numpy.percentile(across[is_before], 97.5) - numpy.percentile(across[is_before], 2.5))
    end_centre = points[is_cap].mean(axis = 0) - 0.002 * direction
    tip_xy = near_xy + tip_along * direction + float(numpy.median(across[is_cap])) * left
    return {"end_centre": end_centre, "direction": direction, "width_m": width, "seen_length_m": float(tip_along - along.min()), "points": int(len(points)),
            "tip_xy": tip_xy}


def stop_against_end(puck_xy: numpy.ndarray, end: Dict) -> Dict:
    """Short of the end's centre along the line (negative: beyond) and left of the line (negative: right), in mm."""

    gap = end["end_centre"] - puck_xy
    left = numpy.array([-end["direction"][1], end["direction"][0]])
    return {"short_mm": 1000.0 * float(gap @ end["direction"]), "aside_mm": -1000.0 * float(gap @ left), "gap_mm": 1000.0 * float(numpy.linalg.norm(gap))}


def measure_stop(pictures: List[numpy.ndarray], joints: List[Sequence[float]], tips_xy: numpy.ndarray, camera: Optional[Dict],
                 reference_offsets: Sequence[int] = range(3, 13), lift_m: float = 0.0) -> Dict:
    """
    The stop of one run: the puck's foot in the last picture against the line's end as each reference picture shows it.

    Parameters
    ----------
    pictures, joints : lists
        The pictures (BGR) and joint sets of the run's last steps in step order; the last entry is the stop.
    tips_xy : numpy.ndarray
        The magnet tip of the same steps, for the direction of travel.
    reference_offsets : sequence of int
        How many steps before the stop each reference picture lies.

    Returns
    -------
    dict
        "puck" (the fit at the stop) and "references": one entry per usable reference picture with "steps_before",
        "short_mm", "aside_mm", "gap_mm", "width_mm", "end_centre", "direction".
    """

    stop_index = len(pictures) - 1
    puck = my_measure.puck_foot(pictures[stop_index], joints[stop_index], camera, lift_m)
    if puck is None:
        return {"puck": None, "references": []}
    back = max(0, stop_index - 4)
    travel = tips_xy[stop_index] - tips_xy[back]
    if float(numpy.linalg.norm(travel)) < 1e-4:
        travel = numpy.array([1.0, 0.0])
    references = []
    for offset in reference_offsets:
        index = stop_index - offset
        if index < 0:
            continue
        seen = line_points_on_table(pictures[index], joints[index], camera, lift_m)
        end = end_of_line(seen, puck["xy"], travel)
        if end is None:
            continue
        tip_px = my_measure.to_picture((end["tip_xy"] + 0.004 * end["direction"])[None, :], joints[index], camera, 0.0, lift_m)[0]
        height, width = pictures[index].shape[:2]
        if not (numpy.all(numpy.isfinite(tip_px)) and 12 <= tip_px[0] <= width - 13 and 12 <= tip_px[1] <= height - 13):
            continue
        if my_measure.hand_mask((height, width), joints[index], camera)[int(round(tip_px[1])), int(round(tip_px[0]))]:
            continue
        entry = stop_against_end(puck["xy"], end)
        entry.update({"steps_before": offset, "width_mm": 1000.0 * end["width_m"], "end_centre": end["end_centre"], "direction": end["direction"],
                      "seen_length_mm": 1000.0 * end["seen_length_m"]})
        references.append(entry)
    return {"puck": puck, "references": references}


def summary_of(references: List[Dict], width_range_mm: Sequence[float] = (8.0, 11.0), seen_least_mm: float = 15.0) -> Optional[Dict]:
    """The median answer over the reference pictures whose line looks like the line (its width and a seen stretch)."""

    good = [entry for entry in references if width_range_mm[0] <= entry["width_mm"] <= width_range_mm[1] and entry["seen_length_mm"] >= seen_least_mm]
    if not good:
        return None
    short = numpy.array([entry["short_mm"] for entry in good])
    aside = numpy.array([entry["aside_mm"] for entry in good])
    return {"references": len(good), "short_mm": float(numpy.median(short)), "aside_mm": float(numpy.median(aside)),
            "short_range_mm": [float(short.min()), float(short.max())], "aside_range_mm": [float(aside.min()), float(aside.max())],
            "gap_mm": float(numpy.hypot(numpy.median(short), numpy.median(aside)))}
