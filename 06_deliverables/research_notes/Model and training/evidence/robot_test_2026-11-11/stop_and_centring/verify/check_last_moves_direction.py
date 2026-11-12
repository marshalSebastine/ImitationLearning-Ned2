"""
Verifier's check 17 (11.11.26): in run 2 the model's last moves pointed LEFT while the puck already stood LEFT of the
line. Is that what the demonstrations show? For the demonstrations whose puck stands more than 7 mm beside the line
three pictures before the stop, the sideways move of the magnet tip (arm model, frames.csv) and of the puck (the
analyst's ../demonstration_last_steps_*.csv) over the last three pictures are compared with the side the puck was on.
The line's direction at the end is worked back from the analyst's own numbers (end centre, puck place, short, aside).
"""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")

frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False)
ends = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_ends_01_03.csv", "demonstration_ends_04_06.csv")], ignore_index = True)
ends["is_reliable"] = (ends["is_end_found"] & (ends["is_end_seen"] == True) & ends["line_width_mm"].between(8.0, 12.5) & (ends["fit_quality"] >= 0.8))
last_steps = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_last_steps_01_03.csv", "demonstration_last_steps_04_06.csv")], ignore_index = True)
rows = []
for _, end in ends[ends["is_reliable"]].iterrows():
    gap = numpy.array([end["end_x"] - end["puck_x"], end["end_y"] - end["puck_y"]]) * 1000.0
    if float(numpy.linalg.norm(gap)) < 3.0:
        continue
    # gap = short * direction - aside * left, with left = (-direction_y, direction_x)
    matrix = numpy.array([[end["short_mm"], end["aside_mm"]], [-end["aside_mm"], end["short_mm"]]])
    direction = numpy.linalg.solve(matrix, gap)
    if abs(float(numpy.linalg.norm(direction)) - 1.0) > 0.05:
        continue
    direction = direction / numpy.linalg.norm(direction)
    left = numpy.array([-direction[1], direction[0]])
    own = last_steps[last_steps["episode_id"] == end["episode_id"]].set_index("steps_before_stop")
    if 3 not in own.index or 0 not in own.index:
        continue
    steps = frames[frames["episode_id"] == end["episode_id"]].sort_values("row_in_episode")
    tips = steps[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float) * 1000.0
    tip_move = tips[-1] - tips[-4]
    rows.append({"episode_id": end["episode_id"], "aside_3": own.loc[3, "aside_mm"], "aside_0": own.loc[0, "aside_mm"],
                 "tip_left": float(tip_move @ left), "tip_along": float(tip_move @ direction)})
table = pandas.DataFrame(rows)
aside = table[table["aside_3"].abs() > 7.0].copy()
side = numpy.sign(aside["aside_3"])
aside["tip_towards_puck_side"] = side * aside["tip_left"]
aside["puck_towards_line"] = -side * (aside["aside_0"] - aside["aside_3"])
print(f"demonstrations with a usable line direction: {len(table)}; puck more than 7 mm beside the line three pictures before the stop: {len(aside)}")
print(f"  over the last three pictures the puck moves towards the line by a median {aside['puck_towards_line'].median():+.1f} mm (towards it in {int((aside['puck_towards_line'] > 1).sum())}, away in {int((aside['puck_towards_line'] < -1).sum())})")
print(f"  over the same pictures the magnet tip moves sideways towards the side the puck is on by a median {aside['tip_towards_puck_side'].median():+.1f} mm"
      f" (towards the puck's side by over 2 mm in {int((aside['tip_towards_puck_side'] > 2).sum())}, towards the other side by over 2 mm in {int((aside['tip_towards_puck_side'] < -2).sum())},"
      f" within 2 mm in {int((aside['tip_towards_puck_side'].abs() <= 2).sum())})")
print(f"  the tip's move along the line over these pictures: median {aside['tip_along'].median():.1f} mm")
print("run 2 of the robot for comparison (verify_robot_last_steps.out.txt): the puck stood 5.7 mm left three pictures before the stop; the tip then moved 8.3 mm to the LEFT"
      " (the puck's side) and the puck went 8.0 mm further from the line.")
