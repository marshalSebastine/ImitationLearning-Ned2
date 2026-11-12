"""
Driver rules compared without the robot: how far would the forearm fall behind the demonstration under each rule?

A simulated robot is walked along every held-out demonstration with the fold model's own answers (fold_steps_E.npz and
fold_steps_A.npz from 04_fold_model_steps.py: each sheet predicted by the model trained without it). The robot's state
is its path length along the demonstrated tip path and its joint 4. At each decision it "sees" one demonstration
picture: among the pictures taken within 0.5 cm of path of where it is (the nearest one when there is none), the one
whose joint 4 is nearest to its own. It takes the model's answer to that picture, and the rule turns the answer into
tip travel and twist.

What this can and cannot show: the tip is assumed to stay on the demonstrated path (so the direction is not tested),
and the pictures are the demonstration's (so a camera turned differently from every demonstration picture is not
tested). It shows how the RULE distributes twist over the path, with the model's real answers on unseen sheets.

Rules:
    now            the driver of 11.11.26: tip 1.0 cm, twist as predicted (at most 10 degrees);
    gain 1.5, 2    the same with the twist multiplied (the --twist-gain option that exists already);
    per cm         tip 1.0 cm, twist = predicted twist x 1 cm / predicted length (the predicted twist per cm kept);
    length         tip = the predicted length (0.2 to 1.5 cm), twist as predicted;
    length x 1.3   the same with the length multiplied by 1.3;
    half step      tip 0.5 cm, twist as predicted (twice as many decisions per cm everywhere);
    unit step      tip and twist scaled TOGETHER until the step is as large as a recorded one (1 cm of tip travel or 5
                   degrees of twist, whichever comes first), by a factor of at most "unit_step_most_factor";
    per picture    the reference: one decision per demonstrated picture, tip as demonstrated, twist as predicted.

Output: 07_driver_rules.out.txt, 07_driver_rules_example.png
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 07_driver_rules.py
"""

from typing import Callable, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames

DRIVER_RULE_PARAMETERS = {
    "twist_step_most_deg": 10.0,
    "fixed_step_cm": 1.0,
    "half_step_cm": 0.5,
    "twist_gains": [1.5, 2.0],
    "length_factor": 1.3,
    "length_limits_cm": [0.2, 1.5],
    "unit_step_cm": 1.0,
    "unit_step_deg": 5.0,
    "unit_step_most_factor": 2.0,
    "smallest_length_cm": 0.2,
    "same_place_cm": 0.5,
    "step_budget_factor": 6.0,
    "first_turn_cm": 20.0,
    "first_turn_least_deg": 30.0,
    # the placements of the failed robot runs: sheet 03 and sheet 04 from end A, standard start
    "first_turn_groups": [("path_03", "A"), ("path_04", "A"), ("path_06", "A"), ("path_05", "B"), ("path_02", "B")],
    "window_cm": 5.0,
    "models": ["E", "A"],
    "example_plans": ["03-01", "04-01", "06-01"]
}



def rule_table(settings: Dict) -> Dict[str, Callable[[float, float], Tuple[float, float]]]:
    """Return the rules: each turns (predicted length in cm, predicted twist in degrees) into (tip travel, twist)."""

    most = settings["twist_step_most_deg"]
    low, high = settings["length_limits_cm"]
    step, half = settings["fixed_step_cm"], settings["half_step_cm"]
    gains, factor = settings["twist_gains"], settings["length_factor"]

    def clamped(twist: float) -> float:
        return max(-most, min(most, twist))

    def unit_step(length: float, twist: float) -> Tuple[float, float]:
        size = max(length / settings["unit_step_cm"], abs(twist) / settings["unit_step_deg"], 1e-6)
        factor = min(settings["unit_step_most_factor"], 1.0 / size)
        return factor * length, clamped(factor * twist)

    return {
        "now (1 cm, twist as predicted)": lambda length, twist: (step, clamped(twist)),
        f"gain {gains[0]}": lambda length, twist: (step, clamped(gains[0] * twist)),
        f"gain {gains[1]}": lambda length, twist: (step, clamped(gains[1] * twist)),
        "per cm (twist x 1 cm / length)": lambda length, twist: (step, clamped(twist * step / max(length, settings["smallest_length_cm"]))),
        f"length ({low} to {high} cm)": lambda length, twist: (max(low, min(high, length)), clamped(twist)),
        f"length x {factor}": lambda length, twist: (max(low, min(high, factor * length)), clamped(twist)),
        f"half step ({half} cm, twist as predicted)": lambda length, twist: (half, clamped(twist)),
        "unit step (1 cm or 5 deg)": unit_step
    }



def walked(
        arc_cm: numpy.ndarray,
        joint_4_deg: numpy.ndarray,
        length_cm: numpy.ndarray,
        twist_deg: numpy.ndarray,
        total_cm: float,
        rule: Callable[[float, float], Tuple[float, float]],
        settings: Dict
) -> pandas.DataFrame:
    """
    Walk a simulated robot along one demonstration.

    Parameters
    ----------
    arc_cm, joint_4_deg : numpy.ndarray
        Per demonstration picture: the path length of the tip and joint 4 (relative to the first picture) when it was taken.
    length_cm, twist_deg : numpy.ndarray
        The model's answer to each picture.
    total_cm : float
        The length of the demonstrated path.
    rule : callable
        (predicted length, predicted twist) -> (tip travel, twist).
    settings : dict
        Supplies "same_place_cm", "unit_step_cm", "unit_step_deg" (the measure of nearness) and "step_budget_factor".

    Returns
    -------
    pandas.DataFrame
        One row per decision: "arc_cm" and "joint_4_deg" before the step, "row" (the picture seen), "travel_cm", "twist_deg".
    """

    position, joint_4 = 0.0, 0.0
    rows = []
    budget = int(settings["step_budget_factor"] * len(arc_cm)) + 50
    while position < total_cm and len(rows) < budget:
        # only pictures taken at this place count ("same_place_cm" of path either way; the nearest one when there is
        # none); among them the one whose joint 4 is nearest to the robot's
        along = numpy.abs(arc_cm - position)
        is_here = along <= max(settings["same_place_cm"], float(along.min()) + 1e-9)
        nearness = numpy.where(is_here, numpy.abs(joint_4_deg - joint_4) / settings["unit_step_deg"] + along / settings["unit_step_cm"], numpy.inf)
        seen = int(numpy.argmin(nearness))
        travel, twist = rule(float(length_cm[seen]), float(twist_deg[seen]))
        rows.append({"arc_cm": position, "joint_4_deg": joint_4, "row": seen, "travel_cm": travel, "twist_deg": twist})
        position += travel
        joint_4 += twist
    return pandas.DataFrame(rows)



def main(parameters: Optional[Dict] = None) -> None:
    """Walk a simulated robot along every held-out demonstration under each driver rule and print how far its forearm falls behind."""

    settings = dict(DRIVER_RULE_PARAMETERS)
    settings.update(parameters or {})
    report = Report("07_driver_rules.out.txt")
    frames = demonstration_frames()
    frames = frames[frames["action"] != "stop"].copy()
    rules = rule_table(settings)
    report.say("DRIVER RULES WALKED ALONG THE HELD-OUT DEMONSTRATIONS (calculated: a simulation with the fold models' real answers; not a robot test)")
    report.say("gap = the simulated robot's joint 4 minus the demonstration's joint 4 at the same place on the path, degrees")
    report.say("")

    example_figure, example_axes = pyplot.subplots(len(settings["example_plans"]), 1, figsize = (14, 4.5 * len(settings["example_plans"])))
    for model in settings["models"]:
        saved = numpy.load(EVIDENCE_FOLDER / f"fold_steps_{model}.npz")
        answers = pandas.DataFrame({"frame_index": saved["frame_index"], "asked_cm": numpy.hypot(saved["forward_cm"], saved["left_cm"]), "asked_twist_deg": saved["twist_deg"]})
        table = frames.merge(answers, on = "frame_index", how = "inner", validate = "one_to_one").sort_values(["episode_id", "row_in_episode"])
        report.say(f"MODEL {model} (fold models, each sheet predicted by the model trained without it; {table['episode_id'].nunique()} demonstrations)")
        report.say(f"   asked step length: median {table['asked_cm'].median():.2f} cm against demonstrated {table['step_cm'].median():.2f}; on demonstrated short steps (< 0.7 cm) "
                   f"{table.loc[table['step_cm'] < 0.7, 'asked_cm'].median():.2f} against {table.loc[table['step_cm'] < 0.7, 'step_cm'].median():.2f}; on long steps "
                   f"{table.loc[table['step_cm'] >= 0.7, 'asked_cm'].median():.2f} against {table.loc[table['step_cm'] >= 0.7, 'step_cm'].median():.2f}")
        report.say(f"   asked |twist|: mean {table['asked_twist_deg'].abs().mean():.2f} degrees against demonstrated {table['forearm_change_deg'].abs().mean():.2f}; on short steps "
                   f"{table.loc[table['step_cm'] < 0.7, 'asked_twist_deg'].abs().mean():.2f} against {table.loc[table['step_cm'] < 0.7, 'forearm_change_deg'].abs().mean():.2f}; on long steps "
                   f"{table.loc[table['step_cm'] >= 0.7, 'asked_twist_deg'].abs().mean():.2f} against {table.loc[table['step_cm'] >= 0.7, 'forearm_change_deg'].abs().mean():.2f}")
        size = numpy.maximum(table["asked_cm"] / settings["unit_step_cm"], table["asked_twist_deg"].abs() / settings["unit_step_deg"])
        report.say(f"   size of the asked step in the recording's measure (1 = 1 cm or 5 degrees): median {size.median():.2f}, 10th percentile {size.quantile(0.1):.2f}, share under 0.5: {(size < 0.5).mean():.2f}")
        report.say("")

        results = []
        examples = {}
        for episode_id, episode in table.groupby("episode_id", sort = False):
            step_cm = episode["step_cm"].to_numpy()
            demonstrated_twist = episode["forearm_change_deg"].fillna(0.0).to_numpy()
            arc = numpy.r_[0.0, numpy.cumsum(step_cm)[:-1]]
            joint_4 = numpy.r_[0.0, numpy.cumsum(demonstrated_twist)[:-1]]
            total = float(step_cm.sum())
            if total < 5.0:
                continue
            # the demonstration's joint 4 as a function of the path length: after everything it did up to that place
            arc_after = numpy.cumsum(step_cm)
            joint_4_after = numpy.cumsum(demonstrated_twist)
            grid = numpy.arange(0.0, total, 0.5)
            demonstrated_on_grid = numpy.array([joint_4_after[arc_after <= place + 1e-9][-1] if (arc_after <= place + 1e-9).any() else 0.0 for place in grid])
            # the first big turn: the place within the first "first_turn_cm" of path where the demonstration's joint 4 is
            # furthest from its start value
            first_part = grid <= settings["first_turn_cm"]
            first_turn_position = int(numpy.argmax(numpy.abs(demonstrated_on_grid[first_part])))
            first_turn_demonstrated = float(demonstrated_on_grid[first_turn_position])
            start_kind, start_end = episode["start_kind"].iloc[0], episode["start_end"].iloc[0]
            all_rules = dict(rules)
            for name, rule in all_rules.items():
                walk = walked(arc, joint_4, episode["asked_cm"].to_numpy(), episode["asked_twist_deg"].to_numpy(), total, rule, settings)
                walk_after_arc = (walk["arc_cm"] + walk["travel_cm"]).to_numpy()
                walk_after_joint_4 = (walk["joint_4_deg"] + walk["twist_deg"]).to_numpy()
                robot_on_grid = numpy.array([walk_after_joint_4[walk_after_arc <= place + 1e-9][-1] if (walk_after_arc <= place + 1e-9).any() else 0.0 for place in grid])
                gap = robot_on_grid - demonstrated_on_grid
                window = int(settings["window_cm"] / 0.5)
                results.append({"rule": name, "sheet": episode["sheet"].iloc[0], "episode_id": episode_id, "decisions": len(walk), "pictures": len(episode),
                                "is_finished": float(walk_after_arc[-1] >= total), "median_gap": float(numpy.median(numpy.abs(gap))), "largest_gap": float(numpy.abs(gap).max()),
                                "share_over_20": float((numpy.abs(gap) > 20.0).mean()), "share_over_45": float((numpy.abs(gap) > 45.0).mean()),
                                "robot_turns": robot_on_grid[window:] - robot_on_grid[:-window], "demonstrated_turns": demonstrated_on_grid[window:] - demonstrated_on_grid[:-window],
                                "robot_twisting": float(numpy.abs(walk["twist_deg"]).sum()), "demonstrated_twisting": float(numpy.abs(demonstrated_twist).sum()), "path_cm": total,
                                "start_kind": start_kind, "start_end": start_end, "first_turn_demonstrated": first_turn_demonstrated,
                                "first_turn_robot": float(robot_on_grid[first_turn_position]), "first_turn_at_cm": float(grid[first_turn_position])})
                if model == settings["models"][0] and episode["plan_number"].iloc[0] in settings["example_plans"]:
                    examples[(episode["plan_number"].iloc[0], name)] = (grid, robot_on_grid, demonstrated_on_grid)
            # the reference: one decision per demonstrated picture
            predicted_after = numpy.cumsum(episode["asked_twist_deg"].to_numpy())
            reference_on_grid = numpy.array([predicted_after[arc_after <= place + 1e-9][-1] if (arc_after <= place + 1e-9).any() else 0.0 for place in grid])
            gap = reference_on_grid - demonstrated_on_grid
            window = int(settings["window_cm"] / 0.5)
            results.append({"rule": "per picture (reference)", "sheet": episode["sheet"].iloc[0], "episode_id": episode_id, "decisions": len(episode), "pictures": len(episode),
                            "is_finished": 1.0, "median_gap": float(numpy.median(numpy.abs(gap))), "largest_gap": float(numpy.abs(gap).max()),
                            "share_over_20": float((numpy.abs(gap) > 20.0).mean()), "share_over_45": float((numpy.abs(gap) > 45.0).mean()),
                            "robot_turns": reference_on_grid[window:] - reference_on_grid[:-window], "demonstrated_turns": demonstrated_on_grid[window:] - demonstrated_on_grid[:-window],
                            "robot_twisting": float(numpy.abs(episode["asked_twist_deg"]).sum()), "demonstrated_twisting": float(numpy.abs(demonstrated_twist).sum()), "path_cm": total,
                            "start_kind": start_kind, "start_end": start_end, "first_turn_demonstrated": first_turn_demonstrated,
                            "first_turn_robot": float(reference_on_grid[first_turn_position]), "first_turn_at_cm": float(grid[first_turn_position])})
            if model == settings["models"][0] and episode["plan_number"].iloc[0] in settings["example_plans"]:
                examples[(episode["plan_number"].iloc[0], "per picture (reference)")] = (grid, reference_on_grid, demonstrated_on_grid)
        results = pandas.DataFrame(results)

        report.say(f"   {'rule':40s} {'sheet':8s} {'decisions / pictures':>20s} {'twist per cm / demo':>19s} {'turn per 5 cm: robot on demo':>28s} {'correlation':>11s} {'median |gap|':>12s} "
                   f"{'share |gap|>20':>14s} {'share |gap|>45':>14s} {'median largest |gap|':>20s}")
        for name in list(rules) + ["per picture (reference)"]:
            for sheet in ANALYSIS_PARAMETERS["sheets"] + ["all"]:
                chosen = results[(results["rule"] == name) & ((results["sheet"] == sheet) | (sheet == "all"))]
                robot_turns = numpy.concatenate(chosen["robot_turns"].to_list())
                demonstrated_turns = numpy.concatenate(chosen["demonstrated_turns"].to_list())
                slope = float((robot_turns * demonstrated_turns).sum() / (demonstrated_turns ** 2).sum())
                report.say(f"   {name:40s} {sheet:8s} {chosen['decisions'].sum() / chosen['pictures'].sum():20.2f} {chosen['robot_twisting'].sum() / chosen['demonstrated_twisting'].sum():19.2f} "
                           f"{slope:28.2f} {numpy.corrcoef(robot_turns, demonstrated_turns)[0, 1]:11.2f} {chosen['median_gap'].median():12.1f} {chosen['share_over_20'].mean():14.2f} "
                           f"{chosen['share_over_45'].mean():14.2f} {chosen['largest_gap'].median():20.1f}")
            report.say("")
        report.say("   'decisions / pictures': how many decisions the robot would need, against the number of demonstration pictures (time).")
        report.say("   'twist per cm / demo': all the twisting done (sum of |twist|) against the demonstration's.")
        report.say("   'turn per 5 cm: robot on demo': the share of the demonstration's net joint-4 change over each 5 cm of path that the robot gives back (1 = the same).")
        report.say("")
        report.say(f"   The first big turn of each demonstration (the place within the first {settings['first_turn_cm']:.0f} cm of path where its joint 4 is furthest from the start; only turns of {settings['first_turn_least_deg']:.0f} degrees or more):")
        report.say("   the share of that turn the simulated robot has made when it reaches the same place (median over the demonstrations; 1 = the same)")
        groups = [("all sheets, all starts", results["sheet"].notna())]
        for sheet, end in settings["first_turn_groups"]:
            groups.append((f"{sheet} standard starts from end {end}", (results["sheet"] == sheet) & (results["start_kind"] == "STD") & (results["start_end"] == end)))
        report.say(f"   {'rule':40s} " + " ".join(f"{name:>40s}" for name, _ in groups))
        for name in list(rules) + ["per picture (reference)"]:
            parts = []
            for _, is_group in groups:
                chosen = results[(results["rule"] == name) & is_group & (results["first_turn_demonstrated"].abs() >= settings["first_turn_least_deg"])]
                share = chosen["first_turn_robot"] / chosen["first_turn_demonstrated"]
                parts.append(f"{share.median():.2f} ({len(chosen)} demos; turn {chosen['first_turn_demonstrated'].abs().median():.0f} deg at {chosen['first_turn_at_cm'].median():.0f} cm)")
            report.say(f"   {name:40s} " + " ".join(f"{part:>40s}" for part in parts))
        report.say("")

        if model == settings["models"][0]:
            for axis, plan in zip(example_axes, settings["example_plans"]):
                drawn = False
                for name, colour in (("now (1 cm, twist as predicted)", "tab:red"), ("gain 2.0", "tab:orange"), ("per cm (twist x 1 cm / length)", "tab:purple"),
                                     ("length (0.2 to 1.5 cm)", "tab:blue"), ("unit step (1 cm or 5 deg)", "tab:green"), ("per picture (reference)", "0.5")):
                    if (plan, name) not in examples:
                        continue
                    grid, robot_on_grid, demonstrated_on_grid = examples[(plan, name)]
                    if not drawn:
                        axis.plot(grid, demonstrated_on_grid, color = "black", linewidth = 2.5, label = "demonstration")
                        drawn = True
                    axis.plot(grid, robot_on_grid, color = colour, linewidth = 1.3, label = name)
                axis.set_title(f"plan {plan} (sheet held out, model {model}): joint 4 along the path, degrees from the start", fontsize = 9)
                axis.set_xlabel("path length of the tip, cm")
                axis.legend(fontsize = 7, ncol = 4)
                axis.grid(alpha = 0.3)
    example_figure.tight_layout()
    example_figure.savefig(EVIDENCE_FOLDER / "07_driver_rules_example.png", dpi = 100)
    report.save()



if __name__ == "__main__":
    main()
