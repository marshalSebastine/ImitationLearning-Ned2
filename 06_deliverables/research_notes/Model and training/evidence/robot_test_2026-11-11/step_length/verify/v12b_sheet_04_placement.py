"""
Verifier: the placement of run 6's sheet (sheet 04) against the demonstrations again, with more of the line in view:
ink points of several pictures per episode (run 6: decisions 0, 2, 4, 6, 8; demonstrations: rows 0, 3, 6, 9, 12) are
pooled before the turn-and-shift search of v12_sheet_placement.py, in both directions.

Run from this folder:  python v12b_sheet_04_placement.py > v12b_sheet_04_placement.out.txt
"""

import contextlib
import io
import os

import numpy
import pandas

with contextlib.redirect_stdout(io.StringIO()):
    import v12_sheet_placement as base

frames = base.frames
steps = pandas.read_csv(os.path.join(base.RUNS, base.RUN_FOLDERS[6], "steps.csv"))
run_points = numpy.vstack([base.ink_points_on_table(os.path.join(base.RUNS, base.RUN_FOLDERS[6], steps["frame"].iloc[index]), steps.loc[index, base.JOINTS].to_numpy(dtype = float))
                           for index in (0, 2, 4, 6, 8)])
run_distance = base.distance_map(run_points)
print(f"run 6: {len(run_points)} ink points from 5 pictures")
clouds = {}
for plan in ["04-01", "04-02", "04-05", "04-13", "04-16", "04-17"]:
    group = frames[frames["plan_number"] == plan]
    clouds[plan] = numpy.vstack([base.ink_points_on_table(os.path.join(base.RAW, group["frame_path"].iloc[index]), group[base.JOINTS].iloc[index].to_numpy(dtype = float))
                                 for index in (0, 3, 6, 9, 12)])
    forward = base.register(clouds[plan], run_distance)
    backward = base.register(run_points, base.distance_map(clouds[plan]))
    print(f"   demonstration {plan} ({len(clouds[plan])} ink points) onto run 6: turn {forward[1]:+.1f} degrees (mean capped distance {1000 * forward[0]:.1f} mm); "
          f"run 6 onto the demonstration: turn {backward[1]:+.1f} degrees ({1000 * backward[0]:.1f} mm)")
for first, second in [("04-01", "04-02"), ("04-01", "04-05"), ("04-01", "04-13"), ("04-01", "04-17"), ("04-13", "04-16"), ("04-13", "04-17")]:
    value, turn, shift_x, shift_y = base.register(clouds[second], base.distance_map(clouds[first]))
    print(f"   {second} onto {first}: turn {turn:+.1f} degrees, mean capped distance {1000 * value:.1f} mm")
