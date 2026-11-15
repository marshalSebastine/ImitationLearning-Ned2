"""
The proposed length rule of the driver, written out exactly, with the numbers behind it and what it would have done
on the logged decisions of the robot runs (no robot: the logged model answers are replayed on paper).

The rule (LENGTH_RULE_PARAMETERS):
    step     the tip moves length_gain x the predicted length (hypot of the model's forward and left, cm) in the
             predicted direction, at most step_most_cm; no smallest step. A predicted length under
             smallest_direction_cm has no direction: the tip stays and only the forearm twists.
    twist    as today (the model's twist times twist_gain, at most 10 degrees per step); length_gain and twist_gain
             are kept EQUAL, because only the pair keeps the forearm's turn per centimetre of travel.
    stall    after every decision: if over the last stall_window decisions the tip's net move is under
             stall_net_move_cm AND the forearm's net turn is under stall_net_turn_deg, the run is stalled. If the stop
             chance reached stall_stop_chance on a picture of the window, the run ends ("stalled near the end"). If
             not, one nudge: nudge_cm along the sum of the window's predicted steps on the table (the picture's up
             direction if that sum is under smallest_direction_cm), and the window starts again. The
             (nudges_most + 1)-th stalled window in a row ends the run ("stalled"). A window that is not stalled
             resets the count.

The replay is open loop: every decision after the first was made for a picture the rule would not have produced, so it
shows what the rule does with the same answers, not how the run would have gone.

Run from this folder after fold_model_outputs.py:
    python 04_length_rule.py
Writes 04_length_rule.out.txt, 04_length_rule_first_15.csv and 04_length_rule_first_15.png.
"""

import importlib
import math
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import step_length_common as common

LENGTH_RULE_PARAMETERS = {
    "length_gain": 1.0,
    "twist_gain": 1.0,
    "step_most_cm": 1.2,
    "smallest_direction_cm": 0.05,
    "twist_step_most_deg": 10.0,
    "stall_window": 6,
    "stall_net_move_cm": 1.0,
    "stall_net_turn_deg": 10.0,
    "stall_stop_chance": 0.25,
    "nudge_cm": 0.5,
    "nudges_most": 3,
    "step_budget": 400,
    # the replay: which runs, how many decisions, and the older rules it is compared with
    "replayed_runs": [5, 9, 10],
    "decisions_replayed": 15,
    "sure_direction_cm": 0.7
}



def table_vectors(steps: pandas.DataFrame) -> numpy.ndarray:
    """
    Return the predicted step of every picture of a robot run as a vector on the table, in cm.

    The driver logged the picture's heading on the table for every decision it carried out; the picture's left is 90
    degrees counter-clockwise of its up (RECORDING_PARAMETERS["picture_left_sign"] is 1).
    """

    heading = numpy.radians(steps["picture_heading_deg"].ffill().to_numpy())
    forward, left = steps["model_tip_forward_cm"].to_numpy(), steps["model_tip_left_cm"].to_numpy()
    return numpy.column_stack([forward * numpy.cos(heading) - left * numpy.sin(heading), forward * numpy.sin(heading) + left * numpy.cos(heading)])



def replayed_run(
        steps: pandas.DataFrame,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    Carry out the logged answers of a robot run with the length rule, on paper.

    Parameters
    ----------
    steps : pandas.DataFrame
        From ``step_length_common.load_robot_run``.
    parameters : dict, optional
        Overrides of ``LENGTH_RULE_PARAMETERS``.

    Returns
    -------
    pandas.DataFrame
        One row per decision until the rule ends the run: "step", "asked_cm", "rule_cm", "kind" ("step", "twist only",
        "nudge", "stalled", "stalled near the end" or "stop"), "twist_deg", the tip the rule would have reached
        ("tip_x_cm", "tip_y_cm", before the decision's move), "travel_cm" and "turn_deg" (joint 4 from its start value).
    """

    settings = dict(LENGTH_RULE_PARAMETERS)
    settings.update(parameters or {})
    vectors = table_vectors(steps)
    asked = steps["asked_cm"].to_numpy()
    twists = numpy.clip(settings["twist_gain"] * steps["model_twist_deg"].to_numpy(), -settings["twist_step_most_deg"], settings["twist_step_most_deg"])
    stop_chance = steps["model_stop_probability"].to_numpy()
    tip = numpy.array([100.0 * steps["tip_x"].iloc[0], 100.0 * steps["tip_y"].iloc[0]])
    turn, travel = 0.0, 0.0
    history: List[Dict] = []
    stalled_windows, stop_count = 0, 0
    rows = []
    for index in range(len(steps)):
        stop_count = stop_count + 1 if stop_chance[index] >= 0.5 else 0
        row = {"step": index, "asked_cm": asked[index], "tip_x_cm": tip[0], "tip_y_cm": tip[1], "travel_cm": travel, "turn_deg": turn,
               "twist_deg": twists[index], "stop_chance": stop_chance[index]}
        if stop_count >= 2:
            rows.append({**row, "rule_cm": 0.0, "kind": "stop"})
            break
        history.append({"tip": tip.copy(), "turn": turn, "vector": vectors[index], "stop_chance": stop_chance[index]})
        length = min(settings["length_gain"] * asked[index], settings["step_most_cm"])
        direction = vectors[index] / max(asked[index], 1e-9)
        kind = "step"
        if asked[index] < settings["smallest_direction_cm"]:
            length, kind = 0.0, "twist only"
        if len(history) > settings["stall_window"]:
            window = history[-settings["stall_window"] - 1:]
            net_move = float(numpy.linalg.norm(window[-1]["tip"] - window[0]["tip"]))
            net_turn = abs(window[-1]["turn"] - window[0]["turn"])
            if net_move < settings["stall_net_move_cm"] and net_turn < settings["stall_net_turn_deg"]:
                stalled_windows += 1
                if max(entry["stop_chance"] for entry in window) >= settings["stall_stop_chance"]:
                    rows.append({**row, "rule_cm": 0.0, "kind": "stalled near the end"})
                    break
                if stalled_windows > settings["nudges_most"]:
                    rows.append({**row, "rule_cm": 0.0, "kind": "stalled"})
                    break
                asked_sum = numpy.sum([entry["vector"] for entry in window[1:]], axis = 0)
                heading = math.radians(steps["picture_heading_deg"].ffill().iloc[index])
                direction = asked_sum / numpy.linalg.norm(asked_sum) if numpy.linalg.norm(asked_sum) >= settings["smallest_direction_cm"] else numpy.array([math.cos(heading), math.sin(heading)])
                length, kind = settings["nudge_cm"], "nudge"
                history = history[-1:]
            else:
                stalled_windows = 0
        rows.append({**row, "rule_cm": length, "kind": kind})
        tip = tip + length * direction
        travel += length
        turn += twists[index]
    return pandas.DataFrame(rows)



def nudge_direction_check(recipe_name: str, parameters: Dict) -> pandas.DataFrame:
    """
    Compare, on held-out sheets, two directions the driver could take when the predicted step is short: the model's
    current direction, and the last direction it was sure of (predicted length of "sure_direction_cm" or more).

    Both are compared with the direction the demonstration's tip really took over the next five rows.

    Returns
    -------
    pandas.DataFrame
        One row per band of predicted length: the rows, and the median error and the share within 45 degrees for each
        of the two directions.
    """

    offline = importlib.import_module("03_predicted_length_offline")
    frames = offline.held_out_table(recipe_name)
    records = []
    for _, group in frames.groupby("episode_id", sort = False):
        heading = group["picture_heading_deg"].to_numpy()
        predicted_angle = heading + numpy.degrees(numpy.arctan2(group["predicted_left_cm"].to_numpy(), group["predicted_forward_cm"].to_numpy()))
        predicted = group["predicted_cm"].to_numpy()
        twist = group["predicted_twist"].abs().to_numpy()
        tip = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
        is_stop = (group["kind"] == "stop").to_numpy()
        last_sure = math.nan
        for index in range(len(group) - 1):
            ahead = min(index + 5, len(group) - 1)
            moved = tip[ahead] - tip[index]
            if not is_stop[index] and numpy.hypot(moved[0], moved[1]) >= 1.0:
                path_angle = math.degrees(math.atan2(moved[1], moved[0]))
                records.append({"predicted_cm": predicted[index], "predicted_twist": twist[index],
                                "current_error": abs((predicted_angle[index] - path_angle + 180.0) % 360.0 - 180.0),
                                "last_sure_error": abs((last_sure - path_angle + 180.0) % 360.0 - 180.0) if not math.isnan(last_sure) else math.nan})
            if predicted[index] >= parameters["sure_direction_cm"]:
                last_sure = predicted_angle[index]
    table = pandas.DataFrame(records)
    bands = [("predicted under 0.15 cm", table["predicted_cm"] < 0.15), ("predicted under 0.3 cm", table["predicted_cm"] < 0.3),
             ("predicted 0.3 to 0.5 cm", (table["predicted_cm"] >= 0.3) & (table["predicted_cm"] < 0.5)),
             ("predicted 0.5 to 0.7 cm", (table["predicted_cm"] >= 0.5) & (table["predicted_cm"] < 0.7)),
             ("predicted 0.7 cm or more", table["predicted_cm"] >= 0.7)]
    summary = []
    for name, mask in bands:
        part = table[mask]
        known = part["last_sure_error"].notna()
        summary.append({"recipe": recipe_name, "band": name, "rows": len(part), "current_direction_median_error_deg": part["current_error"].median(),
                        "current_direction_within_45": (part["current_error"] <= 45.0).mean(),
                        "last_sure_direction_median_error_deg": part.loc[known, "last_sure_error"].median(),
                        "last_sure_direction_within_45": (part.loc[known, "last_sure_error"] <= 45.0).mean()})
    return pandas.DataFrame(summary)



def main() -> None:

    settings = dict(LENGTH_RULE_PARAMETERS)
    lines: List[str] = []
    pandas.set_option("display.width", 250)
    runs = {run["number"]: run for run in common.STEP_LENGTH_PARAMETERS["robot_runs"]}
    frames = common.corrected_start_ends(common.load_demonstration_frames())

    common.print_and_keep(lines, "THE RULE (LENGTH_RULE_PARAMETERS)")
    for name, value in settings.items():
        common.print_and_keep(lines, f"   {name}: {value}")
    common.print_and_keep(lines)

    # what the clamp and the budget are based on
    moving = frames[(frames["action"] != "stop") & (frames["step_cm"] <= 2.0)]
    real_moves = moving[moving["step_cm"] >= 0.3]
    common.print_and_keep(lines, "THE NUMBERS BEHIND THE CLAMP AND THE BUDGET [measured: frames.csv and the ten steps.csv]")
    common.print_and_keep(lines, f"   demonstrated steps (frames.csv as recorded, without stop rows and the steps over 2 cm): {100 * (moving['step_cm'] <= settings['step_most_cm']).mean():.1f} % are "
                                 f"{settings['step_most_cm']} cm or shorter; real moves (0.3 cm or more): median {real_moves['step_cm'].median():.2f} cm, 75th percentile {real_moves['step_cm'].quantile(0.75):.2f}, "
                                 f"90th {real_moves['step_cm'].quantile(0.9):.2f}")
    all_asked = numpy.concatenate([common.load_robot_run(run).query("action == 'model_step'")["asked_cm"].to_numpy() for run in runs.values()])
    common.print_and_keep(lines, f"   asked on the robot (545 moves of the ten runs): over {settings['step_most_cm']} cm on {100 * (all_asked > settings['step_most_cm']).mean():.2f} %, "
                                 f"under {settings['smallest_direction_cm']} cm on {100 * (all_asked < settings['smallest_direction_cm']).mean():.2f} %; largest {all_asked.max():.2f} cm")
    episodes = frames.groupby("episode_id").agg(sheet = ("sheet", "first"), rows = ("step_cm", "size"), path_cm = ("travelled_cm", "max"))
    longest = episodes.sort_values("path_cm").iloc[-1]
    common.print_and_keep(lines, f"   longest demonstration: {longest['path_cm']:.0f} cm of tip path in {int(longest['rows'])} rows ({longest['sheet']}); at the mean length the fold models "
                                 f"predict on unseen sheets (0.65 to 0.67 cm, script 03) that path takes about {longest['path_cm'] / 0.66:.0f} decisions, so the budget of 250 is too small: {settings['step_budget']}")
    common.print_and_keep(lines)

    # decisions each logged run would have needed
    common.print_and_keep(lines, "PACE: what the rule does to the number of decisions [calculated from the logged answers]")
    pace = []
    for number, run in runs.items():
        steps = common.load_robot_run(run)
        moves = steps[steps["action"] == "model_step"]
        rule = numpy.minimum(settings["length_gain"] * moves["asked_cm"], settings["step_most_cm"])
        pace.append({"run": number, "sheet": run["sheet"][-2:], "outcome": run["outcome"], "moves_done": len(moves), "path_done_cm": steps["done_cm"].sum(),
                     "mean_rule_step_cm": rule.mean(), "decisions_for_the_same_path": steps["done_cm"].sum() / rule.mean(),
                     "twist_per_cm_done": moves["twist_done_deg"].abs().sum() / steps["done_cm"].sum(), "twist_per_cm_rule": moves["twist_done_deg"].abs().sum() / rule.sum()})
    common.print_and_keep(lines, pandas.DataFrame(pace).round(2).to_string(index = False))
    common.print_and_keep(lines)

    # the direction of a nudge
    common.print_and_keep(lines, "WHICH DIRECTION WHEN THE PREDICTED STEP IS SHORT [measured offline: fold models on held-out sheets, against the demonstrated path over the next 5 rows]")
    for recipe_name in ["E_recipe", "A_recipe"]:
        common.print_and_keep(lines, nudge_direction_check(recipe_name, settings).round(2).to_string(index = False))
    common.print_and_keep(lines, "   so a stale direction is worse than the model's current one: the nudge follows the model's current answers, and the driver's old rule "
                                 "'a direction under 0.05 cm keeps the last direction' has no support")
    common.print_and_keep(lines)

    # the replay of the first decisions of runs 5, 9 and 10
    common.print_and_keep(lines, f"THE FIRST {settings['decisions_replayed']} DECISIONS OF RUNS 5, 9 AND 10 UNDER THE RULE [calculated, open loop: later pictures would differ]")
    reference = frames[(frames["sheet"] == "path_03") & (frames["start_kind"] == "STD") & (frames["start_end"] == "A")]
    figure, axes = pyplot.subplots(2, len(settings["replayed_runs"]), figsize = (6.5 * len(settings["replayed_runs"]), 10))
    kept = []
    for column, number in enumerate(settings["replayed_runs"]):
        steps = common.load_robot_run(runs[number])
        replay = replayed_run(steps, settings)
        count = settings["decisions_replayed"]
        shown = replay.iloc[:count].copy()
        shown["done_cm"] = steps["done_cm"].to_numpy()[:count]
        shown["asked_angle_deg"] = steps["model_angle_deg"].to_numpy()[:count]
        shown["done_travel_cm"] = steps["travelled_cm"].to_numpy()[:count]
        shown["done_turn_deg"] = (steps["joint_4_deg"] - steps["joint_4_deg"].iloc[0]).to_numpy()[:count]
        shown.insert(0, "run", number)
        kept.append(shown)
        common.print_and_keep(lines, f"--- run {number} (model {runs[number]['model']})")
        common.print_and_keep(lines, shown[["step", "asked_cm", "asked_angle_deg", "twist_deg", "rule_cm", "kind", "done_cm", "travel_cm", "done_travel_cm", "turn_deg", "done_turn_deg"]].round(2).to_string(index = False))
        after = replay.iloc[min(count, len(replay) - 1)]
        done_after = steps.iloc[count]
        net_rule = math.hypot(after["tip_x_cm"] - replay["tip_x_cm"].iloc[0], after["tip_y_cm"] - replay["tip_y_cm"].iloc[0])
        net_done = 100.0 * math.hypot(done_after["tip_x"] - steps["tip_x"].iloc[0], done_after["tip_y"] - steps["tip_y"].iloc[0])
        shortest, longest_step = shown["rule_cm"].min(), shown["rule_cm"].max()
        common.print_and_keep(lines, f"after {count} decisions: the rule has moved the tip {after['travel_cm']:.1f} cm along its path (net {net_rule:.1f} cm from the start) and turned the forearm "
                                     f"{after['turn_deg']:+.0f} degrees; the driver of 11.11.26 moved it {done_after['travelled_cm']:.1f} cm (net {net_done:.1f} cm) with the same turn "
                                     f"({done_after['joint_4_deg'] - steps['joint_4_deg'].iloc[0]:+.0f} degrees measured); rule steps {shortest:.2f} to {longest_step:.2f} cm, "
                                     f"{int((shown['kind'] != 'step').sum())} decisions that were not plain steps; turn per cm: rule {abs(after['turn_deg']) / after['travel_cm']:.1f}, "
                                     f"done {abs(done_after['joint_4_deg'] - steps['joint_4_deg'].iloc[0]) / done_after['travelled_cm']:.1f} degrees per cm")
        common.print_and_keep(lines)

        axis = axes[0, column]
        width = 0.4
        axis.bar(shown["step"] - width / 2, shown["done_cm"], width = width, color = "tab:red", label = "done on 11.11.26 (fixed 1 cm)")
        axis.bar(shown["step"] + width / 2, shown["rule_cm"], width = width, color = "tab:orange", label = "the rule (the model's own length)")
        axis.set_xlabel("decision number")
        axis.set_ylabel("tip move, cm")
        axis.set_ylim(0, 1.3)
        axis.set_title(f"run {number} (model {runs[number]['model']}): step per decision", fontsize = 10)
        axis.legend(fontsize = 8)
        axis.grid(True, alpha = 0.3)
        axis = axes[1, column]
        for index, (_, group) in enumerate(reference.groupby("episode_id", sort = False)):
            axis.plot(group["travelled_cm"], (group["joint_4_deg"] - group["joint_4_deg"].iloc[0]).abs(), color = "tab:green", linewidth = 0.9, alpha = 0.8,
                      label = "demonstrations (sheet 03, end A, standard start)" if index == 0 else None)
        axis.plot(shown["done_travel_cm"], shown["done_turn_deg"].abs(), color = "tab:red", marker = ".", linewidth = 1.8, label = "done on 11.11.26")
        axis.plot(shown["travel_cm"], shown["turn_deg"].abs(), color = "tab:orange", marker = ".", linewidth = 1.8, linestyle = "--", label = "the rule, same answers (open loop)")
        axis.set_xlim(0, 16)
        axis.set_ylim(0, 110)
        axis.set_xlabel("tip travel, cm")
        axis.set_ylabel("forearm turn from the start, degrees")
        axis.set_title(f"run {number}: forearm turn against tip travel, first {count} decisions", fontsize = 10)
        axis.legend(fontsize = 8)
        axis.grid(True, alpha = 0.3)
    figure.tight_layout()
    figure.savefig(common.EVIDENCE_FOLDER / "04_length_rule_first_15.png", dpi = 90)
    pandas.concat(kept, ignore_index = True).round(3).to_csv(common.EVIDENCE_FOLDER / "04_length_rule_first_15.csv", index = False)

    # the whole runs: where the stall guard would act
    common.print_and_keep(lines, "THE STALL GUARD ON THE TEN LOGGED RUNS [calculated, open loop]")
    guard = []
    for number, run in runs.items():
        steps = common.load_robot_run(run)
        replay = replayed_run(steps, settings)
        nudges = replay[replay["kind"] == "nudge"]
        guard.append({"run": number, "sheet": run["sheet"][-2:], "outcome": run["outcome"], "pictures_logged": len(steps), "decisions_replayed": len(replay),
                      "ends_as": replay["kind"].iloc[-1] if replay["kind"].iloc[-1] in ("stop", "stalled", "stalled near the end") else "log ends",
                      "nudges": len(nudges), "first_nudge_at": int(nudges["step"].iloc[0]) if len(nudges) else None,
                      "twist_only_decisions": int((replay["kind"] == "twist only").sum()), "rule_travel_cm": replay["rule_cm"].sum(),
                      "done_travel_cm": steps["done_cm"].iloc[:len(replay)].sum()})
    common.print_and_keep(lines, pandas.DataFrame(guard).round(1).to_string(index = False))
    steps = common.load_robot_run(runs[6])
    replay = replayed_run(steps, settings)
    hairpin = replay[(replay["step"] >= 19) & (replay["step"] <= 50)]
    logged = steps[(steps["step"] >= 19) & (steps["step"] <= 50)]
    common.print_and_keep(lines, f"run 6, decisions 19 to 50 (the hairpin): asked median {logged['asked_cm'].median():.2f} cm, largest {logged['asked_cm'].max():.2f}; twist asked median size "
                                 f"{logged['model_twist_deg'].abs().median():.2f} degrees; the driver moved {logged['done_cm'].sum():.1f} cm back and forth, net "
                                 f"{100 * math.hypot(steps['tip_x'].iloc[51] - steps['tip_x'].iloc[19], steps['tip_y'].iloc[51] - steps['tip_y'].iloc[19]):.1f} cm; "
                                 f"under the rule the replay reaches decision {int(replay['step'].iloc[-1])} and ends as '{replay['kind'].iloc[-1]}'")
    common.print_and_keep(lines, replay[(replay["step"] >= 17) & (replay["step"] <= 50)][["step", "asked_cm", "rule_cm", "kind", "twist_deg", "tip_x_cm", "tip_y_cm", "stop_chance"]].round(2).to_string(index = False))
    (common.EVIDENCE_FOLDER / "04_length_rule.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")



if __name__ == "__main__":
    main()
