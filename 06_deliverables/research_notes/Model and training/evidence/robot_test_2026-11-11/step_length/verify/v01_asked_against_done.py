"""
Verifier's own recomputation of "asked against done" for the ten robot runs of 11.11.26.

Reads only the steps.csv files of 03_data/model_runs. Nothing connects to the robot.
Run from this folder:  python v01_asked_against_done.py > v01_asked_against_done.out.txt
"""

import glob
import os

import numpy
import pandas

RUNS_FOLDER = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = [
    "2026-11-11_172021_check_no_puck", "2026-11-11_172421_E_sheet01_run1", "2026-11-11_172743_E_sheet02_run1",
    "2026-11-11_173036_E_sheet02_run2", "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1",
    "2026-11-11_174448_E_sheet05_run1", "2026-11-11_174936_E_sheet06_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie",
    "2026-11-11_175544_B_team5_sheet03"
]

pandas.set_option("display.width", 250)
rows = []
all_asked = []
all_done = []
for number, folder in enumerate(RUN_FOLDERS, start = 1):
    steps = pandas.read_csv(os.path.join(RUNS_FOLDER, folder, "steps.csv"))
    is_move = (steps["action"] == "model_step").to_numpy()
    asked = numpy.hypot(steps["model_tip_forward_cm"].to_numpy(), steps["model_tip_left_cm"].to_numpy())
    tip = steps[["tip_x", "tip_y"]].to_numpy()
    done = 100.0 * numpy.linalg.norm(numpy.diff(tip, axis = 0), axis = 1)
    joint_4_deg = numpy.degrees(steps["joint_4"].to_numpy())
    twist_done = numpy.diff(joint_4_deg)
    asked_moves = asked[is_move]
    all_asked.append(asked_moves)
    all_done.append(done)
    rows.append({
        "run": number, "pictures": len(steps), "moves": int(is_move.sum()), "last_action": steps["action"].iloc[-1],
        "asked_median": numpy.median(asked_moves), "asked_under_0.3": (asked_moves < 0.3).mean(), "asked_under_0.5": (asked_moves < 0.5).mean(),
        "asked_path": asked_moves.sum(), "done_path": done.sum(), "done_median": numpy.median(done),
        "done_min": done.min(), "done_max": done.max(),
        "twist_done_per_cm_done": numpy.abs(twist_done).sum() / done.sum(),
        "twist_asked_per_cm_asked": numpy.abs(steps["model_twist_deg"].to_numpy()[is_move]).sum() / asked_moves.sum(),
        "net_joint4_turn": joint_4_deg[-1] - joint_4_deg[0],
        "kept_directions": int(steps["is_direction_kept"].fillna(0).sum()),
        "turn_clamped": int(steps["is_turn_clamped"].fillna(0).sum())
    })
table = pandas.DataFrame(rows)
print(table.round(3).to_string(index = False))
asked = numpy.concatenate(all_asked)
done = numpy.concatenate(all_done)
print()
print(f"all moves: {len(asked)}; asked median {numpy.median(asked):.3f}; under 0.5 cm {100 * (asked < 0.5).mean():.1f} %; over 1.0 cm {100 * (asked > 1.0).mean():.1f} %; "
      f"over 1.2 cm {100 * (asked > 1.2).mean():.2f} %; under 0.05 cm {100 * (asked < 0.05).mean():.2f} %; largest {asked.max():.3f}")
print(f"done: n {len(done)}; median {numpy.median(done):.3f}; 5th to 95th percentile {numpy.percentile(done, 5):.3f} to {numpy.percentile(done, 95):.3f}")
