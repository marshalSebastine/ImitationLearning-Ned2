"""
Robot-session critic: a concrete change to the track rule that does draw bends as tight as sheets 03 and 04. The
generator's pieces of constant bend are 30 to 140 mm long, so at a radius of 10 mm one piece turns 172 to 802 degrees and
the track runs into itself (track_rule_with_tight_bends.out.txt: 0 of 6 kept). Here a second kind of piece is added, a
"corner": radius 8 to 13 mm, turning 60 to 120 degrees (so only 8 to 27 mm long), and every track must hold at least
two corners. Everything else is the rule of TEST_TRACK_PARAMETERS (gap 25 mm, margin 22 mm, length 24 to 77 cm), and
the checks are make_test_tracks.py's own functions. This is a copy of the drawing loop for the test only; nothing is
written into the project's test_tracks folder and make_test_tracks.py is not changed. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe track_rule_with_corner_pieces.py > track_rule_with_corner_pieces.out.txt
"""

import math
import sys
import time
from pathlib import Path

import numpy

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))

from config import TEST_TRACK_PARAMETERS  # noqa: E402
from make_test_tracks import close_approaches, crossings, smallest_radius_mm  # noqa: E402

CORNER = {"radius_mm": [8.0, 13.0], "turn_deg": [60.0, 120.0], "share_of_pieces": 0.3, "least_per_track": 2, "gentle_radius_mm": 35.0}


def track_with_corners(generator, parameters, corner, attempts):
    width, height = parameters["page_mm"]
    margin = parameters["edge_margin_mm"]
    gentle = 1.0 / corner["gentle_radius_mm"]
    for attempt in range(attempts):
        position = numpy.array([generator.uniform(margin + 30, width - margin - 30), generator.uniform(height - parameters["start_from_bottom_mm"], height - margin - 5)])
        heading = math.radians(-90.0 + generator.uniform(-35.0, 35.0))
        target_length = generator.uniform(*parameters["length_mm"])
        points = [position.copy()]
        is_inside = True
        last_side = generator.choice([-1.0, 1.0])
        corners = 0
        # the first piece is straight, as the six sheets begin (the puck needs a few centimetres before the first bend)
        pieces = 0
        while len(points) - 1 < target_length and is_inside:
            side = -last_side if generator.random() < parameters["side_change_share"] else last_side
            if pieces == 0:
                bend, piece = 0.0, generator.uniform(30.0, 60.0)
            elif generator.random() < corner["share_of_pieces"]:
                radius = generator.uniform(*corner["radius_mm"])
                bend = side / radius
                piece = radius * math.radians(generator.uniform(*corner["turn_deg"]))
                corners += 1
                last_side = side
            else:
                piece = generator.uniform(*parameters["piece_mm"])
                bend = 0.0 if generator.random() < parameters["straight_share"] else side * generator.uniform(0.25 * gentle, gentle)
                last_side = side if bend else last_side
            pieces += 1
            for _ in range(max(1, int(round(piece)))):
                heading += bend
                position = position + numpy.array([math.cos(heading), math.sin(heading)])
                points.append(position.copy())
                if not (margin <= position[0] <= width - margin and margin <= position[1] <= height - margin):
                    is_inside = False
                    break
        if not is_inside or corners < corner["least_per_track"]:
            continue
        curve = numpy.array(points)
        found = crossings(curve)
        if found:
            continue
        if close_approaches(curve, parameters["smallest_gap_mm"], []):
            continue
        return {"curve": curve, "length_mm": len(curve) - 1.0, "smallest_radius_mm": smallest_radius_mm(curve), "corners": corners, "attempts": attempt + 1}
    return None


generator = numpy.random.default_rng(TEST_TRACK_PARAMETERS["seed"])
started = time.perf_counter()
kept = []
for number in range(6):
    track = track_with_corners(generator, TEST_TRACK_PARAMETERS, CORNER, 20_000)
    if track is None:
        print(f"track {number + 1}: none kept in 20,000 attempts")
        continue
    kept.append(track)
    print(f"track {number + 1}: kept after {track['attempts']} attempts; length {track['length_mm'] / 10:.1f} cm; {track['corners']} corners; tightest bend {track['smallest_radius_mm']:.1f} mm")
print(f"\n{len(kept)} of 6 tracks kept in {time.perf_counter() - started:.0f} s with corner pieces (radius {CORNER['radius_mm'][0]:.0f} to {CORNER['radius_mm'][1]:.0f} mm, "
      f"turn {CORNER['turn_deg'][0]:.0f} to {CORNER['turn_deg'][1]:.0f} degrees), the gap rule unchanged at {TEST_TRACK_PARAMETERS['smallest_gap_mm']:.0f} mm, no crossing")
print("for comparison, the six course sheets by the same 3-point measure (course_sheets_under_the_track_rule.out.txt): 6 mm (03), 6 mm (04), 20 mm (05), 10 mm (06)")
