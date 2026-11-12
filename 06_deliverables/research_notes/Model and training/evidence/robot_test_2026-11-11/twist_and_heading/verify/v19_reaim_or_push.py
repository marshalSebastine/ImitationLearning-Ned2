"""
Verifier 19: what the fold models ask on demonstrated re-aim steps and push steps of held-out sheets (the analyst's
fold_steps files; 87 rows of E checked against my own inference in v05), and how well the answers tell the two apart.
Also the same question at the first bends only (the first 15 pictures of standard starts), where the robot failed.
"""
from pathlib import Path
import numpy
import pandas
from vcommon import demos

EVIDENCE = Path(__file__).resolve().parent.parent
frames = demos()
frames = frames[frames["action"] != "stop"].copy()


def auc(score, is_positive):
    order = numpy.argsort(score)
    ranks = numpy.empty(len(score)); ranks[order] = numpy.arange(1, len(score) + 1)
    positives = is_positive.sum(); negatives = len(score) - positives
    return float((ranks[is_positive].sum() - positives * (positives + 1) / 2) / (positives * negatives))


for model in ("E", "A"):
    saved = numpy.load(EVIDENCE / f"fold_steps_{model}.npz")
    t = frames.merge(pandas.DataFrame({"frame_index": saved["frame_index"], "asked_cm": numpy.hypot(saved["forward_cm"], saved["left_cm"]), "asked_twist": saved["twist_deg"]}), on = "frame_index")
    reaim = (t["len_cm"] < 0.7) & (t["twist"].abs() >= 3)
    push = t["len_cm"] >= 0.7
    both = t[reaim | push]
    per_cm = both["asked_twist"].abs() / numpy.maximum(both["asked_cm"], 0.05)
    print(f"MODEL {model}: re-aim steps {int(reaim.sum())}: demonstrated {t.loc[reaim, 'len_cm'].mean():.2f} cm and {t.loc[reaim, 'twist'].abs().mean():.2f} degrees; asked {t.loc[reaim, 'asked_cm'].mean():.2f} cm and {t.loc[reaim, 'asked_twist'].abs().mean():.2f} degrees")
    print(f"         push steps {int(push.sum())}: demonstrated {t.loc[push, 'len_cm'].mean():.2f} cm and {t.loc[push, 'twist'].abs().mean():.2f} degrees; asked {t.loc[push, 'asked_cm'].mean():.2f} cm and {t.loc[push, 'asked_twist'].abs().mean():.2f} degrees")
    print(f"         told apart (area under the curve): by asked length {auc(-both['asked_cm'].to_numpy(), (both['len_cm'] < 0.7).to_numpy()):.2f}, by asked |twist| {auc(both['asked_twist'].abs().to_numpy(), (both['len_cm'] < 0.7).to_numpy()):.2f}, by asked twist per cm {auc(per_cm.to_numpy(), (both['len_cm'] < 0.7).to_numpy()):.2f}")
    start = t[(t["start_kind"] == "STD")].sort_values(["episode_id", "row_in_episode"])
    start = start.groupby("episode_id").head(15)
    line = []
    for sheet, rows in start.groupby("sheet"):
        line.append(f"{sheet[-2:]}: demonstrated {rows['len_cm'].mean():.2f} cm, {rows['twist'].abs().mean():.1f} deg; asked {rows['asked_cm'].mean():.2f} cm, {rows['asked_twist'].abs().mean():.1f} deg; net twist asked / demonstrated {rows['asked_twist'].sum() / rows['twist'].sum():.2f}")
    print("         first 15 pictures of standard starts, per held-out sheet:\n            " + "\n            ".join(line))
