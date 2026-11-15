"""
Verifier 21: twist per 5 cm of path, NET (what is still there at the end of the stretch) and absolute.
Demonstrations: every episode cut into 5-cm stretches of labelled path (a last stretch under 3 cm dropped).
Robot: the same on the measured tip path and joint 4.
"""
import numpy
from vcommon import demos, run_log, RUNS


def stretches(length, twist, window = 5.0, least = 3.0):
    arc = numpy.r_[0.0, numpy.cumsum(length)[:-1]]
    index = numpy.floor(arc / window).astype(int)
    out = []
    for k in numpy.unique(index):
        chosen = index == k
        travel = length[chosen].sum()
        if travel >= least:
            out.append((abs(twist[chosen].sum()) / travel, numpy.abs(twist[chosen]).sum() / travel, abs(twist[chosen].sum()), numpy.abs(twist[chosen]).sum()))
    return numpy.array(out)


frames = demos()
moving = frames[frames["action"] != "stop"]
net_median, kept = {}, {}
for sheet, rows in moving.groupby("sheet"):
    parts = [stretches(e["len_cm"].to_numpy(), e["twist"].to_numpy()) for _, e in rows.groupby("episode_id") if e["len_cm"].sum() >= 1.0]
    parts = numpy.vstack([p for p in parts if len(p)])
    net_median[sheet] = float(numpy.median(parts[:, 0]))
    kept[sheet] = float(parts[:, 2].sum() / parts[:, 3].sum())
    print(f"{sheet}: net per cm over 5-cm stretches, median {net_median[sheet]:.2f}; share of the twisting still there at the end of the stretch {kept[sheet]:.2f}")
for number in RUNS:
    s = run_log(number)
    done = s[(s["action"] == "model_step") & (s["moved_cm"] > 0.2)]
    parts = stretches(done["moved_cm"].to_numpy(), done["twist_done"].to_numpy())
    net = float(numpy.median(parts[:, 0]))
    print(f"run {number:2d} {RUNS[number][2]}: net per cm, median over stretches {net:.2f}; against the demonstrations {net / net_median[RUNS[number][2]]:.2f}")
