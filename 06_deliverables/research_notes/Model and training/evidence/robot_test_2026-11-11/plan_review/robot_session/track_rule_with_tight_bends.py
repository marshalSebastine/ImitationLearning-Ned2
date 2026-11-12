"""
Robot-session critic: can make_test_tracks.py draw tracks "with bends as tight as the course sheets" (0.8 to 1.3 cm
radius) by changing a setting only? The rule is tried with "smallest_radius_mm" 35 (today), 13, 10 and 8, everything
else as in TEST_TRACK_PARAMETERS, and with the gap rule relaxed. Nothing is written to the project's test_tracks
folder; nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe track_rule_with_tight_bends.py > track_rule_with_tight_bends.out.txt
"""

import math
import sys
import time
from pathlib import Path

import numpy

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))

from config import TEST_TRACK_PARAMETERS  # noqa: E402
from make_test_tracks import close_approaches, crossings, drawn_track, smallest_radius_mm  # noqa: E402

PLAN = {"is_dashed": False, "has_crossing": False, "is_end_near": False}


def turning_per_piece(radius_mm: float, piece_mm) -> str:
    """How far one piece turns at the tightest bend: piece length over radius, in degrees."""

    return f"{math.degrees(piece_mm[0] / radius_mm):.0f} to {math.degrees(piece_mm[1] / radius_mm):.0f} degrees"


for radius, gap in ((35.0, 25.0), (13.0, 25.0), (10.0, 25.0), (8.0, 25.0), (10.0, 12.0), (8.0, 12.0)):
    parameters = dict(TEST_TRACK_PARAMETERS, smallest_radius_mm = radius, smallest_gap_mm = gap, attempts_per_track = 3000)
    generator = numpy.random.default_rng(TEST_TRACK_PARAMETERS["seed"])
    started = time.perf_counter()
    kept = []
    for _ in range(6):
        track = drawn_track(dict(PLAN), generator, parameters)
        if track is not None:
            kept.append(track)
    radii = [track["smallest_radius_mm"] for track in kept]
    lengths = [track["length_mm"] for track in kept]
    print(f"smallest radius {radius:4.0f} mm, smallest gap {gap:3.0f} mm: one tightest piece turns {turning_per_piece(radius, TEST_TRACK_PARAMETERS['piece_mm'])}; "
          f"kept {len(kept)} of 6 tracks within 3000 attempts each ({time.perf_counter() - started:.0f} s); "
          f"tightest bend of the kept tracks (mm): {[round(value, 1) for value in radii]}; lengths (cm): {[round(value / 10, 1) for value in lengths]}")
