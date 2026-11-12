"""
Verifier's spot check of the claim that averaging the joint angles of a hand-guided path over 5 guided points moves the
magnet tip by up to 27 mm sideways and 7.5 mm down.

Every guided_path.csv of the main data is read (never written), the six joints are averaged over a centred window of 5
points, and the arm model gives the tip before and after. Reported: how far the smoothed tip lies from the hand's path
(distance to the nearest piece of the recorded tip path, on the table) and how much its height changes, and WHERE the
largest values occur (how far apart the guided points are there).

Read-only. Prints to the screen and to v10_smoothing_spot_check.out.txt.
"""

import sys
from pathlib import Path
from typing import List

import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(VERIFY_FOLDER.parents[6] / "02_code"))

from side_view_test import magnet_tip_and_lean

SMOOTHING_PARAMETERS = {
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "window_points": 5
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


def distance_to_path(points: numpy.ndarray, path: numpy.ndarray) -> numpy.ndarray:
    """Return the distance of every point to the nearest piece of a polyline, in the units of the input."""

    starts, ends = path[:-1], path[1:]
    pieces = ends - starts
    lengths = numpy.maximum((pieces ** 2).sum(axis = 1), 1e-18)
    result = numpy.zeros(len(points))
    for index, point in enumerate(points):
        along = numpy.clip(((point - starts) * pieces).sum(axis = 1) / lengths, 0.0, 1.0)
        nearest = starts + along[:, None] * pieces
        result[index] = numpy.sqrt(((nearest - point) ** 2).sum(axis = 1)).min()
    return result


if __name__ == "__main__":
    settings = dict(SMOOTHING_PARAMETERS)
    files = sorted(settings["raw_folder"].glob("*/*/guided_path.csv"))
    say(f"guided_path.csv files found: {len(files)}")
    off_path, height_change, point_gap, where = [], [], [], []
    half = settings["window_points"] // 2
    for file in files:
        path = pandas.read_csv(file)
        if len(path) < settings["window_points"] + 2:
            continue
        joints = path[[f"joint_{number}" for number in range(1, 7)]].to_numpy()
        tips = numpy.array([magnet_tip_and_lean(list(row))[0] for row in joints])
        smoothed = joints.copy()
        for index in range(len(joints)):
            low, high = max(0, index - half), min(len(joints), index + half + 1)
            smoothed[index] = joints[low:high].mean(axis = 0)
        smoothed_tips = numpy.array([magnet_tip_and_lean(list(row))[0] for row in smoothed])
        off = 1000.0 * distance_to_path(smoothed_tips[:, :2], tips[:, :2])
        gap = 1000.0 * numpy.linalg.norm(numpy.diff(tips[:, :2], axis = 0), axis = 1)
        largest_gap_near = numpy.array([gap[max(0, index - half):min(len(gap), index + half)].max() for index in range(len(joints))])
        off_path.append(off)
        height_change.append(1000.0 * (smoothed_tips[:, 2] - tips[:, 2]))
        point_gap.append(largest_gap_near)
        where += [file.parent.name] * len(joints)
    off_path = numpy.concatenate(off_path)
    height_change = numpy.concatenate(height_change)
    point_gap = numpy.concatenate(point_gap)
    say(f"guided points: {len(off_path)}")
    say(f"smoothed tip off the hand's path: median {numpy.median(off_path):.2f} mm, 95th percentile {numpy.percentile(off_path, 95):.2f} mm, 99th "
        f"{numpy.percentile(off_path, 99):.2f} mm, largest {off_path.max():.1f} mm")
    say(f"tip height change: lowest {height_change.min():+.2f} mm, 1st percentile {numpy.percentile(height_change, 1):+.2f} mm, 99th percentile "
        f"{numpy.percentile(height_change, 99):+.2f} mm, highest {height_change.max():+.2f} mm")
    say(f"points more than 5 mm off the path: {int((off_path > 5.0).sum())} ({100 * (off_path > 5.0).mean():.2f} %); more than 10 mm: {int((off_path > 10.0).sum())}")
    worst = numpy.argsort(off_path)[-5:][::-1]
    for index in worst:
        say(f"   {off_path[index]:.1f} mm off in {where[index]}: the largest gap between neighbouring guided points there is {point_gap[index]:.0f} mm "
            f"(the tool samples a point every few mm)")
    say(f"median gap between guided points near the points more than 5 mm off: {numpy.median(point_gap[off_path > 5.0]):.0f} mm; near all points: {numpy.median(point_gap):.0f} mm")
    (VERIFY_FOLDER / "v10_smoothing_spot_check.out.txt").write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
