"""Verifier 01: twist per cm, demonstrations per sheet and the ten robot runs; clamps; commanded against measured twist."""
import numpy
import pandas
from vcommon import demos, run_log, RUNS

frames = demos()
print("cleaned rows", len(frames), "episodes", frames["episode_id"].nunique(), "non-stop rows", int((frames["action"] != "stop").sum()))
moving = frames[frames["action"] != "stop"]
print("\nA. demonstrations: sum|twist| / sum(step length), per episode, median over episodes; and by tip-position path length")
sheet_median = {}
for sheet, rows in moving.groupby("sheet"):
    by_label, by_tip = [], []
    for episode, e in rows.groupby("episode_id"):
        if e["len_cm"].sum() < 1.0:
            continue
        by_label.append(e["twist"].abs().sum() / e["len_cm"].sum())
        whole = frames[frames["episode_id"] == episode]
        path = 100.0 * numpy.hypot(numpy.diff(whole["tip_x_m"]), numpy.diff(whole["tip_y_m"])).sum()
        j4_path = numpy.abs(numpy.diff(whole["j4"])).sum()
        by_tip.append(j4_path / path)
    sheet_median[sheet] = float(numpy.median(by_label))
    print(f"  {sheet}: episodes {len(by_label)}  median {numpy.median(by_label):.2f}  pooled {rows['twist'].abs().sum() / rows['len_cm'].sum():.2f}"
          f"  (from joint-4 and tip positions of consecutive kept rows: median {numpy.median(by_tip):.2f})")

print("\nB. robot runs: measured |joint-4 change| per cm of measured tip path; clamps; commanded against measured")
for number in RUNS:
    s = run_log(number)
    done = s[(s["action"] == "model_step") & (s["moved_cm"] > 0.2)]
    path = done["moved_cm"].sum()
    per_cm = done["twist_done"].abs().sum() / path
    gap = (done["twist_done"] - done["twist_used_deg"]).abs()
    print(f"  run {number:2d} {RUNS[number][1]} {RUNS[number][2]}: moves {len(done)}  path {path:.1f} cm  median step {done['moved_cm'].median():.2f} (min {done['moved_cm'].min():.2f}, max {done['moved_cm'].max():.2f})"
          f"  twist/cm {per_cm:.2f}  demo {sheet_median[RUNS[number][2]]:.2f}  ratio {per_cm / sheet_median[RUNS[number][2]]:.2f}"
          f"  clamped {int(s['is_turn_clamped'].sum())}  largest |asked twist| {s['model_twist_deg'].abs().max():.1f}"
          f"  |used - asked| max {(s['twist_used_deg'] - s['model_twist_deg']).abs().max():.3f}"
          f"  |measured - used| median {gap.median():.2f} max {gap.max():.2f}  joint 4 {s['j4'].min():.1f}..{s['j4'].max():.1f}")
