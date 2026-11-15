"""
Verifier 11: my own walk of a simulated robot along the held-out demonstrations, to check the analyst's rule table.

Independent choices: the picture seen is simply the demonstration picture whose path position is nearest to the robot's
(no joint-4 matching); the share of the first big turn is read by interpolation. Two measures of path length:
    "path"     the sum of the labelled step lengths (what the analyst used; it contains the hand's back-and-forth);
    "smooth"   the length of the tip path after a running mean over plus or minus 2 rows (the hand's jitter removed).
A robot that moves in straight 1-cm steps advances along the SMOOTH path, so the second measure is the fairer one.
The answers are the fold models' (fold_steps_E.npz, fold_steps_A.npz: written by the analyst; I checked 87 rows of E
against my own inference in v05, difference 0.0000). No model is run here. Calculated, not measured on the robot.
"""
from pathlib import Path
import numpy
import pandas
from vcommon import demos

EVIDENCE = Path(__file__).resolve().parent.parent
frames = demos()
frames = frames[frames["action"] != "stop"].copy()


def rules():
    def clamp(twist):
        return max(-10.0, min(10.0, twist))

    def unit(length, twist):
        size = max(length / 1.0, abs(twist) / 5.0, 1e-6)
        factor = min(2.0, 1.0 / size)
        return factor * length, clamp(factor * twist)

    return {
        "now (1 cm)": lambda l, t: (1.0, clamp(t)),
        "gain 1.5": lambda l, t: (1.0, clamp(1.5 * t)),
        "gain 2.0": lambda l, t: (1.0, clamp(2.0 * t)),
        "asked length x 1.0, 0.2 to 1.5 cm": lambda l, t: (max(0.2, min(1.5, l)), clamp(t)),
        "asked length x 1.3, 0.2 to 1.5 cm": lambda l, t: (max(0.2, min(1.5, 1.3 * l)), clamp(t)),
        "driver as coded today: predicted, gain 1.0, at most 1.0 cm, no floor": lambda l, t: (min(1.0, l), clamp(t)),
        "unit step": unit,
        "half step 0.5 cm": lambda l, t: (0.5, clamp(t)),
    }


def walk(arc, asked_cm, asked_twist, total, rule):
    position, joint_4, decisions = 0.0, 0.0, 0
    places, angles = [0.0], [0.0]
    budget = 8 * len(arc) + 50
    while position < total and decisions < budget:
        seen = int(numpy.argmin(numpy.abs(arc - position)))
        travel, twist = rule(float(asked_cm[seen]), float(asked_twist[seen]))
        position += travel
        joint_4 += twist
        decisions += 1
        places.append(position)
        angles.append(joint_4)
    return numpy.array(places), numpy.array(angles), decisions, position >= total


for model in ("E", "A"):
    saved = numpy.load(EVIDENCE / f"fold_steps_{model}.npz")
    answers = pandas.DataFrame({"frame_index": saved["frame_index"], "asked_cm": numpy.hypot(saved["forward_cm"], saved["left_cm"]), "asked_twist": saved["twist_deg"]})
    table = frames.merge(answers, on = "frame_index", how = "inner").sort_values(["episode_id", "row_in_episode"])
    print(f"MODEL {model} fold answers on held-out sheets: {table['episode_id'].nunique()} demonstrations; asked length median {table['asked_cm'].median():.2f} cm (demonstrated {table['len_cm'].median():.2f});"
          f" on demonstrated steps of 0.7 cm or more: asked {table.loc[table['len_cm'] >= 0.7, 'asked_cm'].median():.2f} against {table.loc[table['len_cm'] >= 0.7, 'len_cm'].median():.2f}")
    results = []
    ratio_smooth = []
    for episode, e in table.groupby("episode_id", sort = False):
        e = e.reset_index(drop = True)
        length = e["len_cm"].to_numpy()
        twist = e["twist"].to_numpy()
        if length.sum() < 5.0:
            continue
        # joint 4 before each picture, relative to the first
        joint_4 = numpy.r_[0.0, numpy.cumsum(twist)[:-1]]
        arcs = {"path": numpy.r_[0.0, numpy.cumsum(length)[:-1]]}
        x = pandas.Series(100.0 * e["tip_x_m"].to_numpy()).rolling(5, center = True, min_periods = 1).mean().to_numpy()
        y = pandas.Series(100.0 * e["tip_y_m"].to_numpy()).rolling(5, center = True, min_periods = 1).mean().to_numpy()
        arcs["smooth"] = numpy.r_[0.0, numpy.cumsum(numpy.hypot(numpy.diff(x), numpy.diff(y)))]
        ratio_smooth.append(arcs["smooth"][-1] / arcs["path"][-1])
        for measure, arc in arcs.items():
            total = float(arc[-1] + (length[-1] if measure == "path" else 0.5))
            # the first big turn: within the first 20 cm of the path measure, where joint 4 is furthest from its start
            first = arc <= 20.0
            k = int(numpy.argmax(numpy.abs(joint_4[first])))
            turn = float(joint_4[k])
            if abs(turn) < 30.0:
                continue
            reference = float(numpy.cumsum(e["asked_twist"].to_numpy())[k - 1]) if k > 0 else 0.0
            results.append({"measure": measure, "rule": "one decision per demonstrated picture", "share": reference / turn, "decisions": len(e) / len(e), "finished": 1.0,
                            "sheet": e["sheet"].iloc[0], "kind": e["start_kind"].iloc[0], "end": e["start_end"].iloc[0], "turn": abs(turn), "at": arc[k]})
            for name, rule in rules().items():
                places, angles, decisions, finished = walk(arc, e["asked_cm"].to_numpy(), e["asked_twist"].to_numpy(), total, rule)
                made = float(numpy.interp(arc[k], places, angles))
                results.append({"measure": measure, "rule": name, "share": made / turn, "decisions": decisions / len(e), "finished": float(finished),
                                "sheet": e["sheet"].iloc[0], "kind": e["start_kind"].iloc[0], "end": e["start_end"].iloc[0], "turn": abs(turn), "at": arc[k]})
    results = pandas.DataFrame(results)
    print(f"   smooth path / labelled path, median over demonstrations: {numpy.median(ratio_smooth):.2f}")
    for measure in ("path", "smooth"):
        part = results[results["measure"] == measure]
        some = part[part["rule"] == "now (1 cm)"]
        print(f"   measure '{measure}': {len(some)} first big turns, median {some['turn'].median():.0f} degrees at {some['at'].median():.0f} cm")
        print(f"   {'rule':72s} {'share of the first big turn (median)':>36s} {'sheet 03 STD end A':>18s} {'decisions per picture':>21s} {'walks finished':>14s}")
        for name in ["now (1 cm)", "gain 1.5", "gain 2.0", "asked length x 1.0, 0.2 to 1.5 cm", "asked length x 1.3, 0.2 to 1.5 cm",
                     "driver as coded today: predicted, gain 1.0, at most 1.0 cm, no floor", "unit step", "half step 0.5 cm", "one decision per demonstrated picture"]:
            chosen = part[part["rule"] == name]
            sheet3 = chosen[(chosen["sheet"] == "path_03") & (chosen["kind"] == "STD") & (chosen["end"] == "A")]
            print(f"   {name:72s} {chosen['share'].median():36.2f} {sheet3['share'].median():18.2f} {chosen['decisions'].mean():21.2f} {chosen['finished'].mean():14.2f}")
        print("   per sheet, share of the first big turn (median): " + "; ".join(
            f"{name.split(',')[0]}: " + " ".join(f"{sheet[-2:]} {part[(part['rule'] == name) & (part['sheet'] == sheet)]['share'].median():.2f}" for sheet in sorted(part["sheet"].unique()))
            for name in ("now (1 cm)", "gain 1.5", "asked length x 1.0, 0.2 to 1.5 cm", "unit step")))
    print()
