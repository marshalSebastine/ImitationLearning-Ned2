"""
What the PREDICTED step length is worth, measured offline on sheets the model never saw (no robot, no training).

The six fold models of each recipe were run on their held-out sheets by fold_model_outputs.py. For every held-out
picture this script compares the predicted exact step (its length, direction and twist) with the demonstrated step of
the same row (the cleaned labels the models were trained and scored on):
    1. does the predicted length tell a demonstrated twist in place (tip move under 0.3 cm, twist of 3 degrees or
       more) from a real move (distributions, the area under the curve, thresholds and their two error rates);
    2. what gain maps the predicted onto the demonstrated length for moving rows;
    3. what the demonstrations did where the model predicts a short step (binned), and how good the predicted
       direction is there;
    4. how much the forearm turns per centimetre of tip travel under each way of choosing the length;
    5. how far each length rule lands from the demonstrated next tip position, one step ahead;
    6. the numbers behind the stall guard: how little a demonstration ever moved over a run of steps, and how often the
       model itself would stand still on demonstrated pictures;
    7. the first bend of sheet 03 seen by the fold model that never saw sheet 03: the forearm turn per centimetre it
       asks for on the demonstrations' pictures, with its own length and with a fixed 1 cm.

Run from this folder after fold_model_outputs.py:
    python 03_predicted_length_offline.py
Writes 03_predicted_length_offline.out.txt, 03_predicted_length_bins.csv and 03_predicted_length_offline.png.
"""

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import step_length_common as common

from config import TRAINING_TARGET_PARAMETERS
from training_targets import load_training_frames

OFFLINE_PARAMETERS = {
    "recipes": ["E_recipe", "A_recipe"],
    "thresholds_cm": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
    "length_bins_cm": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 3.0],
    # the length rules compared: name, gain on the predicted length, clamp (cm); "fixed" is the driver of 11.11.26
    "rules": [{"name": "fixed 1.0 cm (driver of 11.11.26)", "fixed_cm": 1.0},
              {"name": "predicted length, gain 1.0, at most 1.2 cm", "gain": 1.0, "least_cm": 0.0, "most_cm": 1.2},
              {"name": "gain 1.5, at most 1.0 cm", "gain": 1.5, "least_cm": 0.0, "most_cm": 1.0},
              {"name": "report 4.4 backup: gain 1.6, clamped 0.5 to 1.5 cm", "gain": 1.6, "least_cm": 0.5, "most_cm": 1.5}],
    # a bend row: the demonstration twisted the forearm by at least this much on the step
    "bend_twist_deg": 5.0,
    # the stall guard's window lengths tried, and its candidate limits
    "window_steps": [4, 6, 8],
    "stall_net_move_cm": 1.0,
    "stall_net_turn_deg": 10.0,
    "no_opinion_cm": 0.15,
    "no_opinion_twist_deg": 1.5,
    # the first bend of sheet 03 (end A, standard start): the rows until the forearm has turned this far from its start
    "first_bend": {"sheet": "path_03", "start_end": "A", "start_kind": "STD", "turn_deg": 60.0}
}



def held_out_table(recipe_name: str) -> pandas.DataFrame:
    """
    Return the cleaned training rows with the fold models' predictions on their held-out sheets joined on.

    Returns
    -------
    pandas.DataFrame
        The rows of ``load_training_frames`` (audit cleaning on) with "step_cm", "twist" (demonstrated), "kind"
        ("move", "twist_in_place", "still" or "stop"), "predicted_cm", "predicted_forward_cm", "predicted_left_cm",
        "predicted_twist" and "direction_error_deg" (between the predicted and the demonstrated step direction).
    """

    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters["is_audit_cleaning"] = True
    frames = load_training_frames(target_parameters)
    outputs = numpy.load(common.EVIDENCE_FOLDER / f"fold_outputs_{recipe_name}.npz")
    position = pandas.Series(numpy.arange(len(outputs["frame_index"])), index = outputs["frame_index"])
    exact = outputs["exact"][position.loc[frames["frame_index"].to_numpy()].to_numpy()]
    frames["forward_cm"] = 100.0 * frames["tip_forward_m"].fillna(0.0)
    frames["left_cm"] = 100.0 * frames["tip_left_m"].fillna(0.0)
    frames["step_cm"] = numpy.hypot(frames["forward_cm"], frames["left_cm"])
    frames["twist"] = frames["forearm_change_deg"].fillna(0.0)
    frames["predicted_forward_cm"] = exact[:, 0]
    frames["predicted_left_cm"] = exact[:, 1]
    frames["predicted_cm"] = numpy.hypot(exact[:, 0], exact[:, 1])
    frames["predicted_twist"] = exact[:, 2]
    is_twist_in_place = (frames["step_cm"] < common.STEP_LENGTH_PARAMETERS["twist_in_place_tip_cm"]) & (frames["twist"].abs() >= common.STEP_LENGTH_PARAMETERS["twist_in_place_twist_deg"])
    frames["kind"] = numpy.where(frames["action"] == "stop", "stop",
                                 numpy.where(frames["step_cm"] >= common.STEP_LENGTH_PARAMETERS["twist_in_place_tip_cm"], "move",
                                             numpy.where(is_twist_in_place, "twist_in_place", "still")))
    difference = numpy.degrees(numpy.arctan2(frames["predicted_left_cm"], frames["predicted_forward_cm"]) - numpy.arctan2(frames["left_cm"], frames["forward_cm"]))
    frames["direction_error_deg"] = numpy.abs((difference + 180.0) % 360.0 - 180.0)
    return frames



def area_under_curve(positives: numpy.ndarray, negatives: numpy.ndarray) -> float:
    """Return the chance that a random positive value is larger than a random negative one (ties count half)."""

    ranks = pandas.Series(numpy.r_[positives, negatives]).rank().to_numpy()
    return float((ranks[:len(positives)].sum() - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives)))



def rule_lengths(predicted_cm: numpy.ndarray, rule: Dict) -> numpy.ndarray:
    """Return the step length a rule gives for each predicted length."""

    if "fixed_cm" in rule:
        return numpy.full(len(predicted_cm), rule["fixed_cm"])
    return numpy.clip(rule["gain"] * predicted_cm, rule["least_cm"], rule["most_cm"])



def window_table(
        rows: pandas.DataFrame,
        window: int
) -> pandas.DataFrame:
    """
    Return, for every run of ``window`` consecutive non-stop rows of one episode, how far the tip really got and how
    far the model's own steps would have carried it (the predicted steps of the same pictures, added up on the table).

    Returns
    -------
    pandas.DataFrame
        "net_move_cm" and "net_turn_deg" (demonstrated, from the first row of the window to the row after its last),
        "predicted_net_move_cm" and "predicted_net_turn_deg".
    """

    results = []
    for _, group in rows.groupby("episode_id", sort = False):
        group = group[group["kind"] != "stop"]
        if len(group) <= window:
            continue
        heading = numpy.radians(group["picture_heading_deg"].to_numpy())
        # the picture's left is 90 degrees counter-clockwise of its up (RECORDING_PARAMETERS["picture_left_sign"] is 1)
        step_x = group["predicted_forward_cm"].to_numpy() * numpy.cos(heading) - group["predicted_left_cm"].to_numpy() * numpy.sin(heading)
        step_y = group["predicted_forward_cm"].to_numpy() * numpy.sin(heading) + group["predicted_left_cm"].to_numpy() * numpy.cos(heading)
        sums = numpy.cumsum(numpy.column_stack([step_x, step_y, group["predicted_twist"].to_numpy()]), axis = 0)
        sums = numpy.vstack([numpy.zeros(3), sums])
        tip = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
        joint_4 = numpy.degrees(group["joint_4"].to_numpy())
        first = numpy.arange(0, len(group) - window)
        predicted = sums[first + window] - sums[first]
        results.append(pandas.DataFrame({
            "sheet": group["sheet"].iloc[0],
            "net_move_cm": numpy.hypot(tip[first + window, 0] - tip[first, 0], tip[first + window, 1] - tip[first, 1]),
            "net_turn_deg": numpy.abs(joint_4[first + window] - joint_4[first]),
            "predicted_net_move_cm": numpy.hypot(predicted[:, 0], predicted[:, 1]),
            "predicted_net_turn_deg": numpy.abs(predicted[:, 2])}))
    return pandas.concat(results, ignore_index = True)



def first_bend_table(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Return, per demonstration of the first-bend group, what the fold model asks for on the demonstration's own
    pictures until the forearm has turned "turn_deg": travel and turn, demonstrated and predicted.

    Returns
    -------
    pandas.DataFrame
        "plan", "rows", "demonstrated_travel_cm", "demonstrated_turn_deg", "predicted_travel_cm", "predicted_turn_deg",
        the three turns per centimetre (demonstrated, predicted twist over predicted length, predicted twist over a
        fixed 1 cm per row) and the medians of the predicted length and twist.
    """

    bend = parameters["first_bend"]
    selected = frames[(frames["sheet"] == bend["sheet"]) & (frames["start_kind"] == bend["start_kind"]) & (frames["start_end"] == bend["start_end"])]
    results = []
    for _, group in selected.groupby("episode_id", sort = False):
        group = group[group["kind"] != "stop"]
        turn = numpy.degrees(group["joint_4"].to_numpy())
        reached = numpy.flatnonzero(numpy.abs(turn - turn[0]) >= bend["turn_deg"])
        stretch = group.iloc[:int(reached[0]) if len(reached) else len(group)]
        results.append({"plan": group["plan_number"].iloc[0], "rows": len(stretch),
                        "demonstrated_travel_cm": stretch["step_cm"].sum(), "demonstrated_turn_deg": stretch["twist"].sum(),
                        "predicted_travel_cm": stretch["predicted_cm"].sum(), "predicted_turn_deg": stretch["predicted_twist"].sum(),
                        "demonstrated_deg_per_cm": abs(stretch["twist"].sum()) / stretch["step_cm"].sum(),
                        "own_length_deg_per_cm": abs(stretch["predicted_twist"].sum()) / stretch["predicted_cm"].sum(),
                        "fixed_1_cm_deg_per_cm": abs(stretch["predicted_twist"].sum()) / len(stretch),
                        "predicted_length_median_cm": stretch["predicted_cm"].median(), "predicted_twist_median_deg": stretch["predicted_twist"].median()})
    return pandas.DataFrame(results)



def main() -> None:

    settings = dict(OFFLINE_PARAMETERS)
    lines: List[str] = []
    pandas.set_option("display.width", 250)
    figure, axes = pyplot.subplots(2, 3, figsize = (20, 11))
    for recipe_number, recipe_name in enumerate(settings["recipes"]):
        frames = held_out_table(recipe_name)
        rows = frames[frames["kind"] != "stop"]
        moves = rows[rows["kind"] == "move"]
        twists = rows[rows["kind"] == "twist_in_place"]
        common.print_and_keep(lines, "=" * 110)
        common.print_and_keep(lines, f"{recipe_name}: six fold models on their held-out sheets, {len(frames)} pictures, {len(rows)} without the stop rows "
                                     "[measured: fold_outputs npz against the cleaned labels]")
        common.print_and_keep(lines, "=" * 110)
        common.print_and_keep(lines, f"demonstrated step (cleaned rows): median {rows['step_cm'].median():.2f} cm, mean {rows['step_cm'].mean():.2f}; under 0.5 cm "
                                     f"{100 * (rows['step_cm'] < 0.5).mean():.1f} %, under 0.3 cm {100 * (rows['step_cm'] < 0.3).mean():.1f} %, under 0.2 cm {100 * (rows['step_cm'] < 0.2).mean():.1f} %; "
                                     f"90th, 95th and 99th percentile {rows['step_cm'].quantile(0.9):.2f}, {rows['step_cm'].quantile(0.95):.2f}, {rows['step_cm'].quantile(0.99):.2f} cm; "
                                     f"steps with a twist of 6 degrees or more: median {rows.loc[rows['twist'].abs() >= 6.0, 'step_cm'].median():.2f} cm")
        common.print_and_keep(lines, f"predicted length: median {rows['predicted_cm'].median():.2f} cm, mean {rows['predicted_cm'].mean():.2f}; under 0.3 cm "
                                     f"{100 * (rows['predicted_cm'] < 0.3).mean():.1f} %, over 1.2 cm {100 * (rows['predicted_cm'] > 1.2).mean():.2f} %, largest {rows['predicted_cm'].max():.2f} cm")
        common.print_and_keep(lines)

        common.print_and_keep(lines, "1. TWISTS IN PLACE AGAINST REAL MOVES")
        kinds = rows.groupby(["sheet", "kind"]).size().unstack(fill_value = 0)
        common.print_and_keep(lines, kinds.to_string())
        for kind, group in rows.groupby("kind"):
            percentiles = numpy.percentile(group["predicted_cm"], [5, 25, 50, 75, 95])
            common.print_and_keep(lines, f"   {kind:15s} rows {len(group):6d}: predicted length 5th to 95th percentile {percentiles[0]:.2f} {percentiles[1]:.2f} {percentiles[2]:.2f} "
                                         f"{percentiles[3]:.2f} {percentiles[4]:.2f} cm; demonstrated median {group['step_cm'].median():.2f} cm; "
                                         f"twist size median predicted {group['predicted_twist'].abs().median():.2f}, demonstrated {group['twist'].abs().median():.2f} degrees")
        auc_all = area_under_curve(moves["predicted_cm"].to_numpy(), twists["predicted_cm"].to_numpy())
        common.print_and_keep(lines, f"   area under the curve, predicted length as the score (1.0 = perfect, 0.5 = chance): {auc_all:.3f}; per held-out sheet: " + ", ".join(
            f"{sheet[-2:]} {area_under_curve(group.loc[group['kind'] == 'move', 'predicted_cm'].to_numpy(), group.loc[group['kind'] == 'twist_in_place', 'predicted_cm'].to_numpy()):.3f}"
            for sheet, group in rows.groupby("sheet")))
        twist_ratio = (rows["predicted_twist"].abs() / numpy.maximum(rows["predicted_cm"], 0.05))
        auc_ratio = area_under_curve(twist_ratio[rows["kind"] == "twist_in_place"].to_numpy(), twist_ratio[rows["kind"] == "move"].to_numpy())
        common.print_and_keep(lines, f"   for comparison, predicted twist size divided by predicted length as the score: {auc_ratio:.3f}")
        best = None
        for threshold in settings["thresholds_cm"]:
            missed = float((twists["predicted_cm"] >= threshold).mean())
            false = float((moves["predicted_cm"] < threshold).mean())
            flagged = rows[rows["predicted_cm"] < threshold]
            common.print_and_keep(lines, f"   threshold {threshold:.2f} cm: twists in place predicted at or above it (missed) {100 * missed:5.1f} %; real moves predicted under it "
                                         f"(would be frozen) {100 * false:5.1f} %; of the {len(flagged)} rows under it, {100 * (flagged['kind'] == 'twist_in_place').mean():.1f} % are twists in place")
            if best is None or abs(missed - false) < best[0]:
                best = (abs(missed - false), threshold, missed, false)
        order = numpy.sort(rows["predicted_cm"].to_numpy())
        candidates = order[::20]
        errors = [(abs(float((twists["predicted_cm"] >= value).mean()) - float((moves["predicted_cm"] < value).mean())), value) for value in candidates]
        balanced = min(errors)[1]
        common.print_and_keep(lines, f"   the threshold with equal error rates is {balanced:.2f} cm: {100 * (twists['predicted_cm'] >= balanced).mean():.1f} % of the twists in place missed, "
                                     f"{100 * (moves['predicted_cm'] < balanced).mean():.1f} % of the real moves frozen")
        common.print_and_keep(lines)

        common.print_and_keep(lines, "2. THE GAIN FROM PREDICTED TO DEMONSTRATED LENGTH, moving rows (tip move of 0.3 cm or more)")
        gains = []
        for sheet, group in moves.groupby("sheet"):
            gains.append({"held_out_sheet": sheet[-2:], "rows": len(group), "median_of_ratios": (group["step_cm"] / group["predicted_cm"]).median(),
                          "ratio_of_medians": group["step_cm"].median() / group["predicted_cm"].median(), "ratio_of_means": group["step_cm"].mean() / group["predicted_cm"].mean(),
                          "predicted_median_cm": group["predicted_cm"].median(), "demonstrated_median_cm": group["step_cm"].median()})
        gains.append({"held_out_sheet": "all", "rows": len(moves), "median_of_ratios": (moves["step_cm"] / moves["predicted_cm"]).median(),
                      "ratio_of_medians": moves["step_cm"].median() / moves["predicted_cm"].median(), "ratio_of_means": moves["step_cm"].mean() / moves["predicted_cm"].mean(),
                      "predicted_median_cm": moves["predicted_cm"].median(), "demonstrated_median_cm": moves["step_cm"].median()})
        common.print_and_keep(lines, pandas.DataFrame(gains).round(3).to_string(index = False))
        common.print_and_keep(lines)

        common.print_and_keep(lines, "3. WHAT THE DEMONSTRATIONS DID WHERE THE MODEL PREDICTS A SHORT STEP (all rows but the stop rows, by predicted length)")
        binned = rows.assign(bin = pandas.cut(rows["predicted_cm"], settings["length_bins_cm"]))
        is_clear_move = binned["step_cm"] >= 0.5
        table = binned.groupby("bin", observed = True).agg(
            rows = ("step_cm", "size"), demonstrated_median_cm = ("step_cm", "median"), demonstrated_mean_cm = ("step_cm", "mean"),
            twist_in_place_share = ("kind", lambda values: (values == "twist_in_place").mean()),
            demonstrated_twist_size_median = ("twist", lambda values: values.abs().median()),
            predicted_twist_size_median = ("predicted_twist", lambda values: values.abs().median()))
        table["direction_error_median_deg"] = binned[is_clear_move].groupby("bin", observed = True)["direction_error_deg"].median()
        table["direction_within_45_share"] = binned[is_clear_move].groupby("bin", observed = True)["direction_error_deg"].apply(lambda values: (values <= 45.0).mean())
        common.print_and_keep(lines, table.round(3).to_string())
        common.print_and_keep(lines, "   (direction error: between the predicted and the demonstrated step direction, rows whose demonstrated tip move is 0.5 cm or more)")
        table.round(4).to_csv(common.EVIDENCE_FOLDER / f"03_predicted_length_bins_{recipe_name}.csv")
        mean_vector = numpy.hypot(rows["forward_cm"].mean(), rows["left_cm"].mean())
        common.print_and_keep(lines, f"   why the predicted length is short: mean demonstrated length {rows['step_cm'].mean():.2f} cm, mean predicted length {rows['predicted_cm'].mean():.2f} cm; "
                                     f"correlation of predicted with demonstrated length {numpy.corrcoef(rows['predicted_cm'], rows['step_cm'])[0, 1]:.2f}, "
                                     f"of predicted length with the direction error (clear moves) {numpy.corrcoef(rows.loc[rows['step_cm'] >= 0.5, 'predicted_cm'], rows.loc[rows['step_cm'] >= 0.5, 'direction_error_deg'])[0, 1]:.2f}")
        common.print_and_keep(lines)

        common.print_and_keep(lines, "4. FOREARM TURN PER CENTIMETRE OF TIP TRAVEL (sum of twist sizes over sum of step lengths), per held-out sheet")
        turning = []
        for sheet, group in list(rows.groupby("sheet")) + [("all", rows)]:
            entry = {"held_out_sheet": sheet[-2:] if sheet != "all" else "all", "demonstrated_deg_per_cm": group["twist"].abs().sum() / group["step_cm"].sum(),
                     "demonstrated_step_mean_cm": group["step_cm"].mean(), "predicted_step_mean_cm": group["predicted_cm"].mean(),
                     "demonstrated_twist_mean_deg": group["twist"].abs().mean(), "predicted_twist_mean_deg": group["predicted_twist"].abs().mean()}
            for rule in settings["rules"]:
                entry[rule["name"]] = (group["predicted_twist"].abs().sum() / rule_lengths(group["predicted_cm"].to_numpy(), rule).sum()) / entry["demonstrated_deg_per_cm"]
            turning.append(entry)
        turning = pandas.DataFrame(turning)
        common.print_and_keep(lines, turning.iloc[:, :6].round(3).to_string(index = False))
        common.print_and_keep(lines, "   predicted twist per centimetre of each rule's travel, as a share of the demonstrated turn per centimetre (1.0 = as demonstrated):")
        common.print_and_keep(lines, turning[["held_out_sheet"] + [rule["name"] for rule in settings["rules"]]].round(3).to_string(index = False))
        common.print_and_keep(lines, f"   shrink of the two outputs against the demonstrations (all sheets): step length {rows['predicted_cm'].mean() / rows['step_cm'].mean():.2f}, "
                                     f"twist size {rows['predicted_twist'].abs().mean() / rows['twist'].abs().mean():.2f}")
        common.print_and_keep(lines)

        common.print_and_keep(lines, "5. ONE STEP AHEAD: distance between where each rule puts the tip and where the demonstration put it (cm)")
        direction = numpy.column_stack([rows["predicted_forward_cm"], rows["predicted_left_cm"]]) / numpy.maximum(rows["predicted_cm"].to_numpy()[:, None], 1e-9)
        demonstrated = numpy.column_stack([rows["forward_cm"], rows["left_cm"]])
        groups = {"all rows": numpy.ones(len(rows), dtype = bool), f"bend rows (demonstrated twist of {settings['bend_twist_deg']:.0f} degrees or more)": (rows["twist"].abs() >= settings["bend_twist_deg"]).to_numpy(),
                  "straight rows (demonstrated twist under 2 degrees)": (rows["twist"].abs() < 2.0).to_numpy(), "twists in place": (rows["kind"] == "twist_in_place").to_numpy()}
        landing = []
        for rule in settings["rules"]:
            error = numpy.linalg.norm(rule_lengths(rows["predicted_cm"].to_numpy(), rule)[:, None] * direction - demonstrated, axis = 1)
            entry = {"rule": rule["name"], "mean_step_cm": rule_lengths(rows["predicted_cm"].to_numpy(), rule).mean()}
            for name, mask in groups.items():
                entry[f"{name}: mean"] = error[mask].mean()
                entry[f"{name}: median"] = numpy.median(error[mask])
            landing.append(entry)
        landing = pandas.DataFrame(landing)
        for name, mask in groups.items():
            common.print_and_keep(lines, f"   {name} ({int(mask.sum())} rows):")
            common.print_and_keep(lines, landing[["rule", "mean_step_cm", f"{name}: mean", f"{name}: median"]].round(3).to_string(index = False))
        common.print_and_keep(lines)

        common.print_and_keep(lines, "6. NUMBERS FOR THE STALL GUARD")
        is_no_opinion = (rows["predicted_cm"] < settings["no_opinion_cm"]) & (rows["predicted_twist"].abs() < settings["no_opinion_twist_deg"])
        common.print_and_keep(lines, f"   rows where the model asks for under {settings['no_opinion_cm']} cm and under {settings['no_opinion_twist_deg']} degrees ('no opinion'): "
                                     f"{int(is_no_opinion.sum())} of {len(rows)} ({100 * is_no_opinion.mean():.2f} %); the demonstrations there moved a median {rows.loc[is_no_opinion, 'step_cm'].median():.2f} cm "
                                     f"and twisted a median {rows.loc[is_no_opinion, 'twist'].abs().median():.2f} degrees")
        for window in settings["window_steps"]:
            windows = window_table(frames, window)
            is_still = (windows["net_move_cm"] < settings["stall_net_move_cm"]) & (windows["net_turn_deg"] < settings["stall_net_turn_deg"])
            is_predicted_still = (windows["predicted_net_move_cm"] < settings["stall_net_move_cm"]) & (windows["predicted_net_turn_deg"] < settings["stall_net_turn_deg"])
            common.print_and_keep(lines, f"   windows of {window} steps ({len(windows)}): demonstrated net tip move 1st, 5th, 50th percentile "
                                         f"{windows['net_move_cm'].quantile(0.01):.2f}, {windows['net_move_cm'].quantile(0.05):.2f}, {windows['net_move_cm'].median():.2f} cm; "
                                         f"net forearm turn 50th, 95th percentile {windows['net_turn_deg'].median():.1f}, {windows['net_turn_deg'].quantile(0.95):.1f} degrees; "
                                         f"under {settings['stall_net_move_cm']} cm AND under {settings['stall_net_turn_deg']:.0f} degrees: demonstrated {100 * is_still.mean():.2f} % of the windows, "
                                         f"the model's own steps on the same pictures {100 * is_predicted_still.mean():.2f} % "
                                         f"(model net move median {windows['predicted_net_move_cm'].median():.2f} cm)")
        common.print_and_keep(lines)

        common.print_and_keep(lines, "7. THE FIRST BEND OF SHEET 03 AS AN UNSEEN SHEET (the fold model without sheet 03 on the demonstrations' pictures, until the forearm has turned 60 degrees)")
        bend = first_bend_table(frames, settings)
        common.print_and_keep(lines, bend.round(2).to_string(index = False))
        middle = bend.drop(columns = "plan").median()
        common.print_and_keep(lines, f"   medians: demonstrated {middle['demonstrated_deg_per_cm']:.2f} degrees per cm; the model's twist over its own length {middle['own_length_deg_per_cm']:.2f} "
                                     f"({middle['own_length_deg_per_cm'] / middle['demonstrated_deg_per_cm']:.2f} of the demonstrated); over a fixed 1 cm {middle['fixed_1_cm_deg_per_cm']:.2f} "
                                     f"({middle['fixed_1_cm_deg_per_cm'] / middle['demonstrated_deg_per_cm']:.2f}); predicted length median {middle['predicted_length_median_cm']:.2f} cm, "
                                     f"predicted twist median {middle['predicted_twist_median_deg']:.2f} degrees")
        common.print_and_keep(lines)

        # figures: the two distributions, what the demonstrations did per predicted length, and the direction error
        axis = axes[recipe_number, 0]
        bins = numpy.arange(0.0, 1.6, 0.05)
        axis.hist(moves["predicted_cm"], bins = bins, density = True, alpha = 0.6, color = "tab:blue", label = f"real moves ({len(moves)} rows)")
        axis.hist(twists["predicted_cm"], bins = bins, density = True, alpha = 0.6, color = "tab:red", label = f"twists in place ({len(twists)} rows)")
        axis.hist(moves["step_cm"], bins = bins, density = True, histtype = "step", color = "black", linewidth = 1.2, label = "demonstrated length of the real moves")
        axis.set_xlabel("predicted step length, cm")
        axis.set_ylabel("density")
        axis.set_title(f"{recipe_name}: predicted length on held-out sheets; area under the curve {auc_all:.2f}", fontsize = 10)
        axis.legend(fontsize = 8)
        axis.grid(True, alpha = 0.3)
        axis = axes[recipe_number, 1]
        centres = [interval.mid for interval in table.index]
        axis.plot(centres, table["demonstrated_median_cm"], marker = "o", color = "black", label = "demonstrated length, median")
        axis.plot(centres, table["demonstrated_mean_cm"], marker = "o", color = "grey", label = "demonstrated length, mean")
        axis.plot([0, 1.3], [0, 1.3], color = "tab:blue", linestyle = ":", label = "equal")
        axis.plot(centres, table["twist_in_place_share"], marker = "s", color = "tab:red", label = "share of twists in place")
        axis.set_xlim(0, 1.3)
        axis.set_ylim(0, 1.4)
        axis.set_xlabel("predicted step length, cm (bins of 0.1 cm)")
        axis.set_ylabel("cm, or share")
        axis.set_title(f"{recipe_name}: what the demonstrations did at each predicted length", fontsize = 10)
        axis.legend(fontsize = 8)
        axis.grid(True, alpha = 0.3)
        axis = axes[recipe_number, 2]
        axis.plot(centres, table["direction_error_median_deg"], marker = "o", color = "tab:purple", label = "direction error, median (degrees)")
        axis.plot(centres, 100.0 * table["direction_within_45_share"], marker = "s", color = "tab:green", label = "direction within 45 degrees (%)")
        axis.set_xlim(0, 1.3)
        axis.set_ylim(0, 100)
        axis.set_xlabel("predicted step length, cm (bins of 0.1 cm)")
        axis.set_title(f"{recipe_name}: a short predicted step means an unsure direction", fontsize = 10)
        axis.legend(fontsize = 8)
        axis.grid(True, alpha = 0.3)

    figure.tight_layout()
    figure.savefig(common.EVIDENCE_FOLDER / "03_predicted_length_offline.png", dpi = 90)
    (common.EVIDENCE_FOLDER / "03_predicted_length_offline.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")



if __name__ == "__main__":
    main()
