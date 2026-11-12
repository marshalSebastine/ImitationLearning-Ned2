"""
Make printable test tracks for the robot: new lines the models have never seen, drawn by a fixed random rule.

Every track is an A4 SVG at true size (print at 100 %, "actual size", no fitting to the page), the line 1 cm thick like
the course sheets, solid with round ends or dashed like sheet 06. The rule and its limits are in TEST_TRACK_PARAMETERS.
Each track gets a plan of features before it is drawn (dashed or solid, one crossing or none, the far end near or far
from the paper's edge), so the sets cover them evenly. tracks.json lists every track's length, features and start.

Run from 02_code:
    python make_test_tracks.py
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy

from config import TEST_TRACK_PARAMETERS


def smallest_radius_mm(curve: numpy.ndarray) -> float:
    """The tightest bend of ``curve``: the smallest radius of the circle through three points 5 mm apart."""

    step = 5
    first, middle, last = curve[:-2 * step], curve[step:-step], curve[2 * step:]
    sides = [numpy.linalg.norm(middle - first, axis = 1), numpy.linalg.norm(last - middle, axis = 1), numpy.linalg.norm(last - first, axis = 1)]
    twice_area = numpy.abs(numpy.cross(middle - first, last - first))
    radii = sides[0] * sides[1] * sides[2] / numpy.maximum(2 * twice_area, 1e-12)
    return float(radii.min()) if len(radii) else math.inf



def crossings(curve: numpy.ndarray) -> List[Dict]:
    """
    Where ``curve`` crosses itself.

    Returns
    -------
    list of dict
        One per crossing: "positions" (the two sample indices) and "angle_deg" (0 to 90, 90 = at right angles).
    """

    found = []
    starts, ends = curve[:-1], curve[1:]
    directions = ends - starts
    for index in range(0, len(starts) - 20):
        later = numpy.arange(index + 20, len(starts))
        offset = starts[later] - starts[index]
        denominator = numpy.cross(directions[index], directions[later])
        with numpy.errstate(divide = "ignore", invalid = "ignore"):
            along_first = numpy.cross(offset, directions[later]) / denominator
            along_second = numpy.cross(offset, directions[index]) / denominator
        hits = later[(numpy.abs(denominator) > 1e-12) & (along_first >= 0) & (along_first <= 1) & (along_second >= 0) & (along_second <= 1)]
        for other in hits:
            cosine = abs(directions[index] @ directions[other]) / (numpy.linalg.norm(directions[index]) * numpy.linalg.norm(directions[other]))
            found.append({"positions": (int(index), int(other)), "angle_deg": math.degrees(math.acos(min(1.0, cosine)))})
    return found



def close_approaches(
        curve: numpy.ndarray,
        smallest_gap_mm: float,
        crossing_positions: List[tuple]
) -> int:
    """
    How many sample pairs of ``curve`` come closer than ``smallest_gap_mm`` although they lie more than twice that apart
    along the line, leaving out the stretches around the crossings.
    """

    near_crossing = numpy.zeros(len(curve), dtype = bool)
    reach = int(2.5 * smallest_gap_mm)
    for first, second in crossing_positions:
        for position in (first, second):
            near_crossing[max(0, position - reach):position + reach] = True
    sampled = numpy.arange(0, len(curve), 3)
    points = curve[sampled]
    distances = numpy.linalg.norm(points[:, None, :] - points[None, :, :], axis = 2)
    apart_along = numpy.abs(sampled[:, None] - sampled[None, :]) > 2 * smallest_gap_mm
    is_free = ~near_crossing[sampled][:, None] & ~near_crossing[sampled][None, :]
    return int(numpy.sum((distances < smallest_gap_mm) & apart_along & is_free) // 2)



def drawn_track(
        plan: Dict,
        randomness_generator: numpy.random.Generator,
        parameters: Dict = TEST_TRACK_PARAMETERS
) -> Optional[Dict]:
    """
    Draw random curves until one keeps every rule and matches ``plan`` ("is_dashed", "has_crossing", "is_end_near").

    A curve is driven like a car: from a start near the bottom edge, heading roughly up the page, it runs through pieces
    of constant bend (straight, or a circle no tighter than "smallest_radius_mm"), each "piece_mm" long, until it
    reaches a length drawn within "length_mm".

    Returns
    -------
    dict or None
        "curve" (samples in mm, x right, y down, the start first), "length_mm", "smallest_radius_mm", "crossing_deg"
        and "end_from_edge_mm"; None when no attempt succeeded.
    """

    width, height = parameters["page_mm"]
    margin = parameters["edge_margin_mm"]
    tightest = 1.0 / parameters["smallest_radius_mm"]
    for _ in range(parameters["attempts_per_track"]):
        position = numpy.array([randomness_generator.uniform(margin + 30, width - margin - 30),
                                randomness_generator.uniform(height - parameters["start_from_bottom_mm"], height - margin - 5)])
        heading = math.radians(-90.0 + randomness_generator.uniform(-35.0, 35.0))
        target_length = randomness_generator.uniform(*parameters["length_mm"])
        points = [position.copy()]
        is_inside = True
        last_side = randomness_generator.choice([-1.0, 1.0])
        while len(points) - 1 < target_length and is_inside:
            piece = randomness_generator.uniform(*parameters["piece_mm"])
            # bends that change side make S-curves and meanders, bends to the same side make loops
            side = -last_side if randomness_generator.random() < parameters["side_change_share"] else last_side
            bend = 0.0 if randomness_generator.random() < parameters["straight_share"] else side * randomness_generator.uniform(0.25 * tightest, tightest)
            last_side = side if bend else last_side
            for _ in range(int(piece)):
                heading += bend
                position = position + numpy.array([math.cos(heading), math.sin(heading)])
                points.append(position.copy())
                if not (margin <= position[0] <= width - margin and margin <= position[1] <= height - margin):
                    is_inside = False
                    break
        if not is_inside:
            continue
        curve = numpy.array(points)
        length = len(curve) - 1.0
        found = crossings(curve)
        if len(found) != int(plan["has_crossing"]):
            continue
        if found and found[0]["angle_deg"] < parameters["smallest_crossing_deg"]:
            continue
        if close_approaches(curve, parameters["smallest_gap_mm"], [crossing["positions"] for crossing in found]):
            continue
        end = curve[-1]
        end_from_edge = float(min(end[0], width - end[0], end[1], height - end[1]))
        is_near = end_from_edge < parameters["near_end_mm"]
        is_far = end_from_edge > parameters["far_end_mm"]
        if (plan["is_end_near"] and not is_near) or (not plan["is_end_near"] and not is_far):
            continue
        return {"curve": curve, "length_mm": round(length, 1), "smallest_radius_mm": round(smallest_radius_mm(curve), 1),
                "crossing_deg": round(found[0]["angle_deg"], 1) if found else None, "end_from_edge_mm": round(end_from_edge, 1)}
    return None



def track_svg(
        curve: numpy.ndarray,
        is_dashed: bool,
        name: str,
        parameters: Dict = TEST_TRACK_PARAMETERS
) -> str:
    """An A4 SVG at true size: the line, and the track's name small and light grey in the top corner."""

    width, height = parameters["page_mm"]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in curve[::2]) + f" L {curve[-1][0]:.1f} {curve[-1][1]:.1f}"
    stroke = (f'stroke-dasharray="{parameters["dash_mm"]} {parameters["gap_mm"]}" stroke-linecap="butt"' if is_dashed
              else 'stroke-linecap="round"')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">\n'
            f'  <rect width="{width}" height="{height}" fill="white"/>\n'
            f'  <path d="{path}" fill="none" stroke="black" stroke-width="{parameters["line_width_mm"]}" stroke-linejoin="round" {stroke}/>\n'
            f'  <text x="4" y="6" font-family="Arial" font-size="3" fill="#bbbbbb">{name}: print at 100 %, the line must measure 10 mm</text>\n'
            f'</svg>\n')



def feature_plans(count: int) -> List[Dict]:
    """``count`` feature plans that cover dashed or solid, a crossing or none, and a near or far end evenly."""

    combinations = [{"is_dashed": dashed, "has_crossing": crossing, "is_end_near": near}
                    for crossing in (False, True) for near in (False, True) for dashed in (False, True)]
    return [dict(combinations[index % len(combinations)]) for index in range(count)]



def main() -> None:

    parameters = TEST_TRACK_PARAMETERS
    randomness_generator = numpy.random.default_rng(parameters["seed"])
    folder = Path(parameters["output_folder"])
    listing = []
    for set_name, count in parameters["sets"].items():
        (folder / set_name).mkdir(parents = True, exist_ok = True)
        plans = feature_plans(count)
        randomness_generator.shuffle(plans)
        for number, plan in enumerate(plans, start = 1):
            name = f"{set_name}_{number:02d}"
            track = drawn_track(plan, randomness_generator, parameters)
            if track is None:
                print(f"{name}: no track kept the rules for {plan}; skipped")
                continue
            (folder / set_name / f"{name}.svg").write_text(track_svg(track["curve"], plan["is_dashed"], name, parameters), encoding = "utf-8")
            entry = {"name": name, "set": set_name, **plan, **{key: value for key, value in track.items() if key != "curve"},
                     "start_mm": [round(float(value), 1) for value in track["curve"][0]], "end_mm": [round(float(value), 1) for value in track["curve"][-1]]}
            listing.append(entry)
            print(f"{name}: {entry['length_mm'] / 10:.1f} cm, {'dashed' if plan['is_dashed'] else 'solid'}, "
                  f"{'one crossing' if plan['has_crossing'] else 'no crossing'}, end {entry['end_from_edge_mm'] / 10:.1f} cm from the edge, "
                  f"tightest bend {entry['smallest_radius_mm'] / 10:.1f} cm")
    (folder / "tracks.json").write_text(json.dumps({"rule": {key: (str(value) if isinstance(value, Path) else value) for key, value in parameters.items()},
                                                    "tracks": listing}, indent = 1), encoding = "utf-8")



if __name__ == "__main__":
    main()
