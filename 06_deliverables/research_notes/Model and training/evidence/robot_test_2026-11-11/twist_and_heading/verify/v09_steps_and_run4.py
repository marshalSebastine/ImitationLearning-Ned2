"""
Verifier 09: (a) the two kinds of demonstrated step and the recording rule (1 cm of tip travel OR 5 degrees on a joint);
(b) run 4 on sheet 02 against the standard-start demonstrations from the same end: net turn over the arc and by path length;
(c) run 8 on sheet 06 and run 6 on sheet 04: joint-4 change by path length against the demonstrations.
"""
import numpy
import pandas
from vcommon import demos, run_log

frames = demos()
same = (frames["episode_id"].shift(-1) == frames["episode_id"]) & (frames["row_in_episode"].shift(-1) == frames["row_in_episode"] + 1) & (frames["action"] != "stop")
joint_change = numpy.column_stack([numpy.degrees(frames[f"joint_{k}"].shift(-1) - frames[f"joint_{k}"]).abs() for k in range(1, 7)])
steps = frames[same].copy()
changes = joint_change[same.to_numpy()]
largest = changes.max(axis = 1)
size = numpy.maximum(steps["len_cm"].to_numpy() / 1.0, largest / 5.0)
print("(a) consecutive demonstrated steps: %d" % len(steps))
print("    size = max(length / 1 cm, largest joint change / 5 degrees): 5th percentile %.2f, median %.2f, share under 0.9: %.3f" % (numpy.quantile(size, 0.05), numpy.median(size), (size < 0.9).mean()))
short = steps["len_cm"].to_numpy() < 0.7
twist = steps["twist"].abs().to_numpy()
print("    steps shorter than 0.7 cm: %.3f of the steps, %.3f of the path, %.3f of all |twist|; joint 4 is the joint that moved most in %.3f of them; median length %.2f cm, median |twist| %.1f; pooled %.1f degrees per cm against %.1f on longer steps" % (
    short.mean(), steps["len_cm"].to_numpy()[short].sum() / steps["len_cm"].sum(), twist[short].sum() / twist.sum(), (changes[short].argmax(axis = 1) == 3).mean(),
    numpy.median(steps["len_cm"].to_numpy()[short]), numpy.median(twist[short]), twist[short].sum() / steps["len_cm"].to_numpy()[short].sum(), twist[~short].sum() / steps["len_cm"].to_numpy()[~short].sum()))
moving = frames[frames["action"] != "stop"]
print("    all non-stop rows: median step %.2f cm, mean %.2f; under 0.5 cm: %.3f; under 0.2 cm: %.3f; median length of steps with |twist| >= 6: %.2f" % (
    moving["len_cm"].median(), moving["len_cm"].mean(), (moving["len_cm"] < 0.5).mean(), (moving["len_cm"] < 0.2).mean(), moving.loc[moving["twist"].abs() >= 6, "len_cm"].median()))
print("    largest |asked twist| on the robot (all ten runs): %.1f" % max(run_log(n)["model_twist_deg"].abs().max() for n in range(1, 11)))


def demo_turn_at(sheet, end, places):
    reference = frames[(frames["sheet"] == sheet) & (frames["start_kind"] == "STD") & (frames["start_end"] == end)]
    out = {place: [] for place in places}
    whole = []
    for _, e in reference.groupby("episode_id"):
        e = e.reset_index(drop = True)
        arc = numpy.r_[0.0, numpy.cumsum(e["len_cm"].to_numpy())[:-1]]
        whole.append(e["j4"].iloc[-1] - e["j4"].iloc[0])
        for place in places:
            if arc[-1] >= place:
                out[place].append(float(numpy.interp(place, arc, e["j4"].to_numpy() - e["j4"].iloc[0])))
    return out, whole


for label, number, sheet, end, places in (("(b) run 4, sheet 02 from end B", 4, "path_02", "B", (5, 10, 15, 20, 25, 30, 35)), ("(c) run 8, sheet 06 from end A", 8, "path_06", "A", (5, 9, 12, 15, 18, 20)),
                                          ("(c) run 6, sheet 04 from end A", 6, "path_04", "A", (4, 8, 12, 16, 20))):
    out, whole = demo_turn_at(sheet, end, places)
    s = run_log(number)
    arc = numpy.r_[0.0, numpy.cumsum(s["moved_cm"].to_numpy())[:-1]]
    print(f"{label}: robot joint 4 from {s['j4'].iloc[0]:+.1f} to {s['j4'].iloc[-1]:+.1f} (net {s['j4'].iloc[-1] - s['j4'].iloc[0]:+.1f}); demonstrations' net over the whole episode: median {numpy.median(whole):+.1f} (range {min(whole):+.1f} to {max(whole):+.1f}, {len(whole)} episodes)")
    for place in places:
        robot = float(numpy.interp(place, arc, s["j4"].to_numpy() - s["j4"].iloc[0]))
        values = out[place]
        print(f"      after {place:2d} cm of path: robot turned {robot:+6.1f}; demonstrations median {numpy.median(values):+6.1f} (range {min(values):+.1f} to {max(values):+.1f})")
