"""
Robot-session critic: block 0 of the plan asks "do steps of 1 to 3 mm come out as commanded?" and wants to read it
"from the log". What did the logs of 11.11.26 show for the 1 cm steps: how far was the tip (computed from the joint
angles the robot reported) from the planned point after each move? That error is what a 1 to 3 mm step has to be
compared with. Reads steps.csv of the ten runs. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe step_accuracy_from_the_logs.py > step_accuracy_from_the_logs.out.txt
"""

from pathlib import Path

import numpy
import pandas

RUN_ROOT = Path("D:/AI_Project/03_data/model_runs")
every_miss, every_length, every_along, every_across, every_twist_miss = [], [], [], [], []
print("run | moves | reached minus planned tip, mm: median, 90th percentile, largest | length really moved for a commanded 10 mm: 5th to 95th percentile | joint 4: reached minus planned, degrees, largest")
for folder in sorted(RUN_ROOT.glob("2026-11-11_17*")):
    steps = pandas.read_csv(folder / "steps.csv")
    moved = steps[steps["action"] == "model_step"].index
    moved = [index for index in moved if index + 1 in steps.index]
    now = steps.loc[moved, ["tip_x", "tip_y"]].to_numpy() * 1000.0
    planned = steps.loc[moved, ["tip_next_x", "tip_next_y"]].to_numpy() * 1000.0
    reached = steps.loc[[index + 1 for index in moved], ["tip_x", "tip_y"]].to_numpy() * 1000.0
    miss = numpy.linalg.norm(reached - planned, axis = 1)
    command = planned - now
    unit = command / numpy.linalg.norm(command, axis = 1)[:, None]
    done = reached - now
    along = numpy.einsum("ij,ij->i", done, unit) - numpy.linalg.norm(command, axis = 1)
    across = numpy.abs(unit[:, 0] * done[:, 1] - unit[:, 1] * done[:, 0])
    length = numpy.linalg.norm(done, axis = 1)
    twist_planned = numpy.degrees(steps.loc[moved, "joint_4"].to_numpy()) + steps.loc[moved, "twist_used_deg"].to_numpy()
    twist_miss = numpy.abs(numpy.degrees(steps.loc[[index + 1 for index in moved], "joint_4"].to_numpy()) - twist_planned)
    every_miss.extend(miss); every_length.extend(length); every_along.extend(along); every_across.extend(across); every_twist_miss.extend(twist_miss)
    print(f"{folder.name[18:]:28s} | {len(moved):3d} | {numpy.median(miss):.2f}, {numpy.percentile(miss, 90):.2f}, {miss.max():.2f} | "
          f"{numpy.percentile(length, 5):.2f} to {numpy.percentile(length, 95):.2f} | {twist_miss.max():.2f}")
every_miss, every_length, every_along, every_across = map(numpy.array, (every_miss, every_length, every_along, every_across))
print(f"\nall {len(every_miss)} moves [measured from the joint angles the robot reported, so the arm's own sensors, not a ruler]:")
print(f"   tip reached minus planned: median {numpy.median(every_miss):.2f} mm, 90th percentile {numpy.percentile(every_miss, 90):.2f} mm, 99th {numpy.percentile(every_miss, 99):.2f} mm, largest {every_miss.max():.2f} mm")
print(f"   along the commanded direction (too far +, too short -): median {numpy.median(every_along):+.2f} mm, 5th to 95th percentile {numpy.percentile(every_along, 5):+.2f} to {numpy.percentile(every_along, 95):+.2f} mm")
print(f"   across it: median {numpy.median(every_across):.2f} mm, 95th percentile {numpy.percentile(every_across, 95):.2f} mm")
print(f"   joint 4 reached minus planned: largest {max(every_twist_miss):.2f} degrees")
for step_mm in (1.0, 2.0, 3.0, 6.5):
    print(f"   if the same error held for a commanded {step_mm:.1f} mm step [calculated]: the 90th-percentile miss is {100 * numpy.percentile(every_miss, 90) / step_mm:.0f} % of the step, "
          f"the median miss {100 * numpy.median(every_miss) / step_mm:.0f} %")
print("\nwhat the log cannot show: play in the joints, the arm sagging, the rod bending against the puck, or the puck sticking. The tip's place in the log is")
print("calculated from the reported joint angles; a ruler, a dial gauge or the camera's own pictures are needed for that.")
