"""
Plan review, key "evidence": would the recorded demonstrations of sheet 03 themselves pass the plan's block A rule
("the puck stays within 12 mm of the line's middle to 10 cm past the bend")? Uses the stop-and-centring analyst's
per-picture offsets of the six sampled sheet-03 episodes (demonstration_centring_03.csv); nothing is changed.
The first bend of sheet 03 ends after about 6.7 cm of tip travel (60 degrees of forearm turn; step-length verdict 4),
so "10 cm past the bend" is taken here as the first 17 cm of tip travel, and also as the first 20 and 25 cm.
"""

import numpy
import pandas

folder = "D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/stop_and_centring/"
table = pandas.read_csv(folder + "demonstration_centring_03.csv")
print("episodes:", table["episode_id"].nunique(), "pictures:", len(table))

for episode_id, rows in table.groupby("episode_id", sort = False):
    rows = rows.sort_values("step").reset_index(drop = True)
    travel_cm = 100.0 * numpy.concatenate([[0.0], numpy.cumsum(numpy.hypot(numpy.diff(rows["tip_x"]), numpy.diff(rows["tip_y"])))])
    size_mm = rows["offset_mm"].abs()
    name = episode_id.split("/")[-1]
    line = f"{name:55s} {rows['start_kind'].iloc[0]:6s} corrections {int(rows['hand_corrections'].iloc[0])}"
    for limit_cm in (17.0, 20.0, 25.0):
        part = (travel_cm <= limit_cm) & size_mm.notna().to_numpy()
        sizes = size_mm.to_numpy()[part]
        over_12 = sizes > 12.0
        two_in_a_row_12 = bool(numpy.any(over_12[1:] & over_12[:-1])) if len(sizes) > 1 else False
        over_17 = sizes > 17.0
        two_in_a_row_17 = bool(numpy.any(over_17[1:] & over_17[:-1])) if len(sizes) > 1 else False
        line += (f" | first {limit_cm:.0f} cm: {int(part.sum()):3d} pictures, largest {numpy.max(sizes) if len(sizes) else float('nan'):5.1f} mm, "
                 f"over 12 mm {int(over_12.sum()):2d} (two in a row: {two_in_a_row_12}), over 17 mm {int(over_17.sum()):2d} (two in a row: {two_in_a_row_17})")
    print(line)
