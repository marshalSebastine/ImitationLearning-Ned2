"""
Verifier, check 14: how much of the lean label is plain kinematics? For a random sample of demonstrated steps the next
row's tip and twist are reached with the wrist HELD at its present value (project's solve_with_wrist_fixed), and the
forward lean change that results (record_episodes.rod_lean_forward_deg, the label's own function) is compared with
the lean label. Calculated; 2,500 steps.
    D:/AI_Project/.venv/Scripts/python.exe v14_lean_kinematic_part.py
"""
import math
import sys
import numpy
import pandas
sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
from record_episodes import forearm_camera_heading_rad, rod_lean_forward_deg
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed

JOINTS = [f"joint_{index}" for index in range(1, 7)]
frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "action", "lean_change_deg", "picture_heading_deg", "corrected_end" if False else "sheet"] + JOINTS)
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
has_next = (frames["episode_id"].shift(-1) == frames["episode_id"]).to_numpy() & (frames["action"] != "stop").to_numpy()
candidates = numpy.flatnonzero(has_next)
chosen = numpy.sort(numpy.random.default_rng(14).choice(candidates, 2500, replace = False))
values = frames[JOINTS].to_numpy(dtype = float)
rows = []
for position in chosen:
    now, after = values[position].tolist(), values[position + 1].tolist()
    heading = forearm_camera_heading_rad(now)
    lean_now = rod_lean_forward_deg(now, heading)
    recomputed = rod_lean_forward_deg(after, heading) - lean_now
    held = solve_with_wrist_fixed(magnet_tip_and_lean(after)[0], after[3], now[4], now[:3])
    if held is None:
        continue
    held[5] = now[5]
    rows.append({"label": frames.loc[position, "lean_change_deg"], "recomputed": recomputed, "kinematic": rod_lean_forward_deg(held, heading) - lean_now,
                 "wrist_change": math.degrees(after[4] - now[4]), "heading_gap": math.degrees(heading) - frames.loc[position, "picture_heading_deg"]})
table = pandas.DataFrame(rows)
print("steps:", len(table), "; heading recomputed against the logged one: largest gap %.3f degrees" % table["heading_gap"].abs().max())
print("label against the lean change recomputed from the next row's joints: median |gap| %.3f, 99th %.3f degrees" % (
    (table["label"] - table["recomputed"]).abs().median(), (table["label"] - table["recomputed"]).abs().quantile(0.99)))
for target in ("recomputed", "label"):
    truth = table[target]
    residual = truth - table["kinematic"]
    leaning = truth.abs() >= 0.5
    print("target %-10s: sd %.3f; kinematic part sd %.3f; explains %.1f %% of the variance (1 - residual variance / variance); sign of the kinematic part right on %.1f %% of the %d steps of 0.5 degrees or more" % (
        target, truth.std(), table["kinematic"].std(), 100 * (1 - residual.var() / truth.var()), 100 * (numpy.sign(table.loc[leaning, "kinematic"]) == numpy.sign(truth[leaning])).mean(), int(leaning.sum())))
wrist_part = table["recomputed"] - table["kinematic"]
moved = table["wrist_change"].abs() > 1.0
print("the part that needs the wrist: correlation with the wrist's own change %.3f; slope %.3f degrees of lean per degree of wrist (steps with a wrist change over 1 degree: %d)" % (
    numpy.corrcoef(wrist_part, table["wrist_change"])[0, 1], numpy.polyfit(table.loc[moved, "wrist_change"], wrist_part[moved], 1)[0] if moved.sum() > 5 else float("nan"), int(moved.sum())))
