"""
Proofreader sitting_a, check 7b: the travel back to "5 cm above the start" on paper.

After a run the puck must stay where it stopped (photo and ruler), so the arm has to travel back above it. Here the
lifted end pose of a run (5 cm up, wrist and hand held, forearm as it ended) is joined to the lifted start joints by
one joint move (every joint changing evenly, which is how a joint move is planned; the robot's own planner may differ),
and the magnet tip's lowest height above the taught table plane on the way is printed, for the ten runs of 11.11 and
for the last pictures of the 209 demonstrations. A second variant turns the forearm straight first, in place, as the
recording tool does (__turn_back_lifted), and then travels.

Arm model only; nothing moves. Run:
    D:/AI_Project/.venv/Scripts/python.exe -B p07b_travel_back_on_paper.py > p07b_travel_back_on_paper.out.txt
"""

import math
import sys
from pathlib import Path

import numpy
import pandas

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
sys.path.insert(0, str(CODE))

from config import MODEL_MODE_PARAMETERS, PUCK_PARAMETERS, table_height     # noqa: E402
from model_mode import held_joints_at                                       # noqa: E402
from side_view_test import magnet_tip_and_lean                              # noqa: E402

START = [math.radians(value) for value in (-6.06, -41.87, -61.93, 0.0, 13.53, -6.15)]
START_LIFTED = held_joints_at(START, 0.05)


def lowest_on_the_way(joints_from, joints_to, samples = 60):
    lowest, widest_reach = math.inf, 0.0
    for share in numpy.linspace(0.0, 1.0, samples):
        joints = [a + share * (b - a) for a, b in zip(joints_from, joints_to)]
        tip = magnet_tip_and_lean(joints)[0]
        lowest = min(lowest, tip[2] - table_height(tip[0], tip[1]))
        widest_reach = max(widest_reach, math.hypot(tip[0], tip[1]))
    return 1000.0 * lowest, 100.0 * widest_reach


def both_ways(end_joints):
    lifted = held_joints_at(end_joints, 0.05)
    if lifted is None:
        return None
    one_move = lowest_on_the_way(lifted, START_LIFTED)
    straightened = held_joints_at([lifted[0], lifted[1], lifted[2], 0.0, lifted[4], lifted[5]], 0.0)
    if straightened is None:
        return one_move, None
    tip_lifted = magnet_tip_and_lean(lifted)[0]
    # forearm turned straight over the same tip point, still 5 cm up
    from side_view_test import solve_with_wrist_fixed
    target = numpy.array([tip_lifted[0], tip_lifted[1], tip_lifted[2]])
    straight = solve_with_wrist_fixed(target, 0.0, math.radians(MODEL_MODE_PARAMETERS["wrist_held_deg"]), [lifted[0], lifted[1], lifted[2]])
    if straight is None:
        return one_move, None
    straight[5] = math.radians(MODEL_MODE_PARAMETERS["hand_held_deg"])
    turn = lowest_on_the_way(lifted, straight)
    travel = lowest_on_the_way(straight, START_LIFTED)
    return one_move, (min(turn[0], travel[0]), max(turn[1], travel[1]))


print(f"puck height {1000 * PUCK_PARAMETERS['height_m']:.0f} mm; lifted start joints {[round(math.degrees(value), 1) for value in START_LIFTED]} degrees")
print("\nthe ten runs of 11.11 (tip's lowest height on the way back, mm above the table): one joint move | forearm straightened first")
for folder in sorted(Path("D:/AI_Project/03_data/model_runs").glob("2026-11-11_17*")):
    last = pandas.read_csv(folder / "steps.csv").iloc[-1]
    result = both_ways([float(last[f"joint_{index}"]) for index in range(1, 7)])
    one, two = result
    print(f"   {folder.name:42s} forearm {math.degrees(last['joint_4']):+6.1f} deg: {one[0]:5.0f} mm | {('%5.0f mm' % two[0]) if two else 'not solvable'}")

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", usecols = ["episode_id", "row_in_episode"] + [f"joint_{index}" for index in range(1, 7)], low_memory = False)
last_rows = frames.sort_values(["episode_id", "row_in_episode"]).groupby("episode_id").last()
ones, twos, unsolved = [], [], 0
for _, row in last_rows.iterrows():
    result = both_ways([float(row[f"joint_{index}"]) for index in range(1, 7)])
    if result is None:
        unsolved += 1
        continue
    ones.append(result[0][0])
    if result[1] is not None:
        twos.append(result[1][0])
ones, twos = numpy.array(ones), numpy.array(twos)
print(f"\nthe last pictures of {len(last_rows)} demonstrations as end poses ({unsolved} could not be lifted 5 cm with the wrist held):")
print(f"   one joint move: lowest tip height median {numpy.median(ones):.0f} mm, smallest {ones.min():.0f} mm; under 40 mm in {int((ones < 40).sum())}, under the puck's 29 mm in {int((ones < 29).sum())} of {len(ones)}")
print(f"   forearm straightened first: median {numpy.median(twos):.0f} mm, smallest {twos.min():.0f} mm; under 40 mm in {int((twos < 40).sum())}, under 29 mm in {int((twos < 29).sum())} of {len(twos)}")
