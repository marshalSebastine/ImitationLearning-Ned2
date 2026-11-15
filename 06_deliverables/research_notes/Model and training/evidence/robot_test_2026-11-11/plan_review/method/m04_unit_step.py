"""
The driver's step rule judged against the recording rule: every recorded step is "1 cm of tip travel or 5 degrees on a
joint", so a recorded step always has the size 1 (m01: 5th percentile 1.01, median 1.22) and its size carries no
information; only the mix of travel and twist does. A "unit step" carries the model's answer out scaled to that size.

Measured on the fold models' answers on the sheets they never saw (the outputs the step-length analyst saved and the
verifier checked against an own forward pass); nothing is trained, nothing connects to the robot. The net-turn measure
is the verifier's (step_length/verify/v08): windows of 10 rows, the twist carried out summed along the demonstrated
sense, per centimetre carried out, as a share of the demonstrated; "sharp" windows have a demonstrated net turn of 30
degrees or more. Reproducing the verifier's 0.42 to 0.43 (fixed 1 cm) and 0.63 to 0.68 (own length) checks this script.

    1. Rules compared: fixed 1.0 cm; the model's own length; fixed 0.65 cm; the unit step (answer scaled until
       max(travel / 1 cm, |twist| / 5 degrees) = "unit_size", by at most "unit_gain_most").
    2. Pace: decisions per centimetre carried out, against the demonstrations' rows per centimetre.
    3. The end: the step each rule carries out at the last picture before the stop row and at the first picture whose
       stop chance is 0.5 or more.
    4. Where the unit step enlarges a doubtful answer: how often, and how wrong the direction is there.

Run from this folder:
    python m04_unit_step.py > m04_unit_step.out.txt
"""

from typing import Callable, Dict, Optional, Tuple

import numpy
import pandas

import method_common

UNIT_STEP_PARAMETERS = {
    "window_rows": 10,
    "sharp_net_turn_deg": 30.0,
    "twist_step_most_deg": 10.0,
    "own_length_most_cm": 1.2,
    "fixed_short_cm": 0.65,
    # the unit step: sizes tried and the largest factor an answer is enlarged by
    "unit_sizes": [1.0, 1.22],
    "unit_gain_most": [2.0, 3.0, 100.0],
    "stop_threshold": 0.5
}



def unit_step_rule(
        unit_size: float,
        gain_most: float,
        parameters: Dict
) -> Callable[[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]:
    """
    Return a rule that scales travel and twist together to one recorded step.

    Parameters
    ----------
    unit_size : float
        The size the carried-out step gets, in recorded steps (1 = 1 cm of travel or 5 degrees of twist).
    gain_most : float
        The largest factor an answer is enlarged by; answers already larger than the unit are made smaller.
    parameters : dict
        "unit_length_cm" and "unit_turn_deg" (the recording rule).

    Returns
    -------
    callable
        (asked length in cm, asked twist in degrees) -> (length carried out, twist carried out).
    """

    def rule(
            asked_cm: numpy.ndarray,
            asked_twist: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:

        size = numpy.maximum(asked_cm / parameters["unit_length_cm"], numpy.abs(asked_twist) / parameters["unit_turn_deg"])
        gain = numpy.minimum(gain_most, unit_size / numpy.maximum(size, 1e-9))
        return gain * asked_cm, gain * asked_twist

    return rule



def unit_step_report(parameters: Optional[Dict] = None) -> None:
    """Print the four measurements named in the module's text."""

    settings = method_common.merged_parameters(UNIT_STEP_PARAMETERS)
    settings.update(parameters or {})
    frames = method_common.cleaned_frames(settings)
    window = settings["window_rows"]
    clip = settings["twist_step_most_deg"]
    rules = {
        "fixed 1.0 cm (11.11.26)": lambda asked_cm, asked_twist: (numpy.ones_like(asked_cm), asked_twist),
        f"own length, at most {settings['own_length_most_cm']} cm": lambda asked_cm, asked_twist: (numpy.minimum(asked_cm, settings["own_length_most_cm"]), asked_twist),
        f"fixed {settings['fixed_short_cm']} cm": lambda asked_cm, asked_twist: (numpy.full_like(asked_cm, settings["fixed_short_cm"]), asked_twist)
    }
    for unit_size in settings["unit_sizes"]:
        for gain_most in settings["unit_gain_most"]:
            name = f"unit step, size {unit_size}, enlarged at most x{gain_most:g}"
            rules[name] = unit_step_rule(unit_size, gain_most, settings)

    for recipe, path in settings["fold_outputs"].items():
        data = numpy.load(path)
        predicted = pandas.DataFrame({"frame_index": data["frame_index"], "asked_forward": data["exact"][:, 0], "asked_left": data["exact"][:, 1],
                                      "asked_twist": data["exact"][:, 2], "stop_chance": data["stop_probability"]})
        table = frames.merge(predicted, on = "frame_index", how = "left")
        table["shown_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"].fillna(0.0), table["tip_left_m"].fillna(0.0))
        table["shown_twist"] = table["forearm_change_deg"].fillna(0.0)
        table["asked_cm"] = numpy.hypot(table["asked_forward"], table["asked_left"])
        table["shown_angle"] = numpy.degrees(numpy.arctan2(table["tip_left_m"].fillna(0.0), table["tip_forward_m"].fillna(0.0)))
        table["asked_angle"] = numpy.degrees(numpy.arctan2(table["asked_left"], table["asked_forward"]))
        table["angle_error"] = numpy.abs(method_common.wrapped_degrees(table["asked_angle"] - table["shown_angle"]))
        steps = table[table["action"] != "stop"].copy()
        print(f"================ recipe {recipe}: {len(steps)} step rows on held-out sheets")
        asked_size = numpy.maximum(steps["asked_cm"] / settings["unit_length_cm"], steps["asked_twist"].abs() / settings["unit_turn_deg"])
        shown_size = numpy.maximum(steps["shown_cm"] / settings["unit_length_cm"], steps["shown_twist"].abs() / settings["unit_turn_deg"])
        print(f"size of a step in recorded steps (max of travel / 1 cm and |twist| / 5 degrees): shown median {shown_size.median():.2f}"
              f" (5th percentile {shown_size.quantile(0.05):.2f}); asked median {asked_size.median():.2f} (5th {asked_size.quantile(0.05):.2f},"
              f" 25th {asked_size.quantile(0.25):.2f}, 75th {asked_size.quantile(0.75):.2f}); asked under 0.5 in {100 * (asked_size < 0.5).mean():.1f} %,"
              f" under 0.33 in {100 * (asked_size < 0.33).mean():.1f} %; the asked size comes from the twist in"
              f" {100 * (steps['asked_twist'].abs() / settings['unit_turn_deg'] > steps['asked_cm'] / settings['unit_length_cm']).mean():.1f} % of rows"
              f" (shown: {100 * (steps['shown_twist'].abs() / settings['unit_turn_deg'] > steps['shown_cm'] / settings['unit_length_cm']).mean():.1f} %)")

        print("1 and 2. Net turn per centimetre carried out as a share of the demonstrated, and the pace")
        totals = {name: numpy.zeros(4) for name in rules}
        shown = numpy.zeros(4)
        counts = numpy.zeros(2)
        for episode_id, group in steps.groupby("episode_id", sort = False):
            shown_twist, shown_cm = group["shown_twist"].to_numpy(), group["shown_cm"].to_numpy()
            asked_twist, asked_cm = group["asked_twist"].to_numpy(), group["asked_cm"].to_numpy()
            for start in range(0, len(group) - window + 1, window):
                part = slice(start, start + window)
                net_shown = abs(shown_twist[part].sum())
                is_sharp = net_shown >= settings["sharp_net_turn_deg"]
                shown += numpy.array([net_shown, shown_cm[part].sum(), net_shown * is_sharp, shown_cm[part].sum() * is_sharp])
                counts += numpy.array([window, window * is_sharp])
                for name, rule in rules.items():
                    done_cm, done_twist = rule(asked_cm[part], asked_twist[part])
                    signed = numpy.clip(done_twist, -clip, clip).sum() * numpy.sign(shown_twist[part].sum())
                    totals[name] += numpy.array([signed, done_cm.sum(), signed * is_sharp, done_cm.sum() * is_sharp])
        print(f"   demonstrated: {shown[0] / shown[1]:.2f} degrees per cm over all windows, {shown[2] / shown[3]:.2f} in sharp windows; rows per cm {counts[0] / shown[1]:.2f}"
              f" (sharp windows {counts[1] / shown[3]:.2f})")
        print("   rule                                             share of the turn per cm: all   sharp   | share of the net turn itself: all   sharp   | decisions per cm: all   sharp   | decisions against the demonstrations' rows")
        for name, total in totals.items():
            print(f"   {name:48s} {(total[0] / total[1]) / (shown[0] / shown[1]):26.2f}   {(total[2] / total[3]) / (shown[2] / shown[3]):5.2f}"
                  f"   | {total[0] / shown[0]:31.2f}   {total[2] / shown[2]:5.2f}   | {counts[0] / total[1]:20.2f}   {counts[1] / total[3]:5.2f}"
                  f"   | {shown[1] / total[1]:.2f} times")

        print("3. The end of the line")
        grouped = table.groupby("episode_id", sort = False)
        is_last_step = (grouped["action"].shift(-1) == "stop") & (table["action"] != "stop")
        last = table[is_last_step]
        is_stop_like = (table["stop_chance"] >= settings["stop_threshold"]) & (table["action"] != "stop")
        first_stop_like = table[is_stop_like].groupby("episode_id", sort = False).head(1)
        for label, rows in ((f"the last picture before the stop row ({len(last)} episodes; shown last move: median {last['shown_cm'].median():.2f} cm)", last),
                            (f"the first picture with a stop chance of {settings['stop_threshold']} or more, the stop row itself left out ({len(first_stop_like)} episodes)", first_stop_like)):
            print(f"   {label}: asked length median {rows['asked_cm'].median():.2f} cm, asked |twist| median {rows['asked_twist'].abs().median():.1f} degrees")
            for name, rule in rules.items():
                done_cm, _ = rule(rows["asked_cm"].to_numpy(), rows["asked_twist"].to_numpy())
                print(f"      {name:48s} carries out a median {numpy.median(done_cm):.2f} cm (10th to 90th percentile {numpy.percentile(done_cm, 10):.2f} to {numpy.percentile(done_cm, 90):.2f})")

        print("4. Where the unit step enlarges a doubtful answer (direction scored on rows whose shown tip move is 0.3 cm or more)")
        is_scored = steps["shown_cm"] >= 0.3
        for low, high in ((0.0, 0.33), (0.33, 0.5), (0.5, 0.8), (0.8, 10.0)):
            band = (asked_size >= low) & (asked_size < high)
            scored = steps[band & is_scored]
            print(f"   asked size {low:.2f} to {high:.2f}: {100 * band.mean():5.1f} % of rows; direction within 45 degrees of the shown step in"
                  f" {100 * (scored['angle_error'] <= 45).mean():5.1f} %, more than 90 degrees off in {100 * (scored['angle_error'] > 90).mean():5.1f} %;"
                  f" twist sense right (shown twist 2 degrees or more) in"
                  f" {100 * (numpy.sign(steps[band & (steps['shown_twist'].abs() >= 2)]['asked_twist']) == numpy.sign(steps[band & (steps['shown_twist'].abs() >= 2)]['shown_twist'])).mean():5.1f} %")
        print()



if __name__ == "__main__":
    unit_step_report()
