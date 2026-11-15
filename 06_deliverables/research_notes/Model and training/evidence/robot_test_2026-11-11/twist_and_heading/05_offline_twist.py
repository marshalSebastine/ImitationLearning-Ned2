"""
The fold models' forearm twist on the sheets they never saw (saved predictions, no model is run here).

Parts:
    1  step by step: size, sign, and the two slopes (demonstrated on predicted = the gain that fits best step by step;
       predicted on demonstrated = the share of the demonstrated twist the model gives back on average);
    2  summed along each held-out demonstration: the drift of joint 4 after 10, 20 and 40 steps when the predicted
       twists are added up over the demonstration's own pictures, against the demonstrated joint-4 change;
    3  how noisy the demonstrated twist is from step to step, and what a target smoothed over plus or minus 2 steps (or
       averaged over the next 5 steps) would change.

Output: 05_offline_twist.out.txt, 05_summed_twist_scatter.png, 05_twist_noise_example.png
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 05_offline_twist.py
"""

from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames, fold_predictions

OFFLINE_TWIST_PARAMETERS = {
    "window_steps": [10, 20, 40],
    "smoothing_half_width": 2,
    "forward_steps": 5,
    "sign_threshold_deg": 2.0,
    "main_runs": ["A recipe (exact)", "E recipe (chunk, stop 25)"]
}



def window_sums(
        values: numpy.ndarray,
        run_id: numpy.ndarray,
        steps: int
) -> numpy.ndarray:
    """
    Return, for every row, the sum of ``values`` over this row and the next ``steps`` - 1 rows, or NaN when those rows
    do not all belong to the same unbroken run of consecutive steps (``run_id``).
    """

    cumulative = numpy.r_[0.0, numpy.cumsum(values)]
    sums = numpy.full(len(values), numpy.nan)
    last = numpy.arange(len(values)) + steps - 1
    is_inside = last < len(values)
    is_same = numpy.zeros(len(values), dtype = bool)
    is_same[is_inside] = run_id[last[is_inside]] == run_id[is_inside]
    sums[is_same] = cumulative[last[is_same] + 1] - cumulative[numpy.flatnonzero(is_same)]
    return sums



def centred_mean(
        values: numpy.ndarray,
        run_id: numpy.ndarray,
        half_width: int
) -> numpy.ndarray:
    """Return the mean of each value with its ``half_width`` neighbours on both sides inside the same run (fewer at the ends)."""

    total = numpy.zeros(len(values))
    count = numpy.zeros(len(values))
    for shift in range(-half_width, half_width + 1):
        moved = numpy.roll(values, -shift)
        moved_id = numpy.roll(run_id, -shift)
        position = numpy.arange(len(values)) + shift
        is_valid = (position >= 0) & (position < len(values)) & (moved_id == run_id)
        total[is_valid] += moved[is_valid]
        count[is_valid] += 1
    return total / count



def slope_through_origin(target: numpy.ndarray, predictor: numpy.ndarray) -> float:
    """Return the least-squares factor f in target = f x predictor."""

    return float((target * predictor).sum() / (predictor ** 2).sum())



def main(parameters: Optional[Dict] = None) -> None:
    """Score the saved twist predictions of the fold models step by step and summed along the demonstrations, and measure the noise of the demonstrated twist."""

    settings = dict(OFFLINE_TWIST_PARAMETERS)
    settings.update(parameters or {})
    report = Report("05_offline_twist.out.txt")
    frames = demonstration_frames()
    report.say("THE FOLD MODELS' TWIST ON HELD-OUT SHEETS (measured from predictions_*.npz in 04_runs/gpu_ki014 and frames.csv with the audit cleaning)")
    report.say("")

    tables = {}
    for label, run_folder in ANALYSIS_PARAMETERS["fold_runs"].items():
        table = fold_predictions(run_folder, frames)
        table = table[table["action"] != "stop"].reset_index(drop = True)
        # an unbroken run of consecutive steps: a new one starts at each episode and after each row dropped by the cleaning
        is_new = (table["episode_id"] != table["episode_id"].shift(1)) | (table["row_in_episode"] != table["row_in_episode"].shift(1) + 1)
        table["run_id"] = numpy.cumsum(is_new.to_numpy())
        table["demonstrated"] = table["forearm_change_deg"].fillna(0.0)
        tables[label] = table

    report.say("1. Step by step, per held-out sheet")
    report.say(f"   {'model':26s} {'sheet':8s} {'rows':>5s} {'mean |demo|':>11s} {'mean |pred|':>11s} {'size ratio':>10s} {'demo on pred':>12s} {'pred on demo':>12s} {'correlation':>11s} {'sign right':>10s}")
    for label, table in tables.items():
        for sheet, rows in list(table.groupby("sheet")) + [("all", table)]:
            demonstrated, predicted = rows["demonstrated"].to_numpy(), rows["predicted_twist_deg"].to_numpy()
            is_twisting = numpy.abs(demonstrated) >= settings["sign_threshold_deg"]
            report.say(f"   {label:26s} {sheet:8s} {len(rows):5d} {numpy.abs(demonstrated).mean():11.2f} {numpy.abs(predicted).mean():11.2f} {numpy.abs(predicted).mean() / numpy.abs(demonstrated).mean():10.2f} "
                       f"{slope_through_origin(demonstrated, predicted):12.2f} {slope_through_origin(predicted, demonstrated):12.2f} {numpy.corrcoef(demonstrated, predicted)[0, 1]:11.2f} "
                       f"{(numpy.sign(predicted[is_twisting]) == numpy.sign(demonstrated[is_twisting])).mean():10.2f}")
    report.say("   'size ratio': mean |predicted| / mean |demonstrated| (the brief's 'about half').")
    report.say("   'demo on pred': the factor that fits demonstrated = factor x predicted best, step by step (above 1: the model twists too little where it twists).")
    report.say("   'pred on demo': predicted = factor x demonstrated: the share of each demonstrated twist the model gives back on average.")
    report.say(f"   'sign right': on steps that twisted {settings['sign_threshold_deg']:.0f} degrees or more.")
    report.say("")

    report.say("2. Summed along the held-out demonstrations: joint 4 after N steps when the PREDICTED twists are added up over the demonstration's own pictures")
    report.say(f"   {'model':26s} {'sheet':8s} {'steps':>5s} {'windows':>7s} {'median |demo turn|':>18s} {'median |drift|':>14s} {'90% |drift|':>11s} {'pred on demo':>12s} {'demo on pred':>12s} {'correlation':>11s} {'median |drift|, pred x best gain':>32s}")
    figure, axes = pyplot.subplots(len(settings["main_runs"]), len(settings["window_steps"]), figsize = (6 * len(settings["window_steps"]), 5.5 * len(settings["main_runs"])))
    for label, table in tables.items():
        demonstrated, predicted, run_id = table["demonstrated"].to_numpy(), table["predicted_twist_deg"].to_numpy(), table["run_id"].to_numpy()
        for steps in settings["window_steps"]:
            demonstrated_sum = window_sums(demonstrated, run_id, steps)
            predicted_sum = window_sums(predicted, run_id, steps)
            is_valid = numpy.isfinite(demonstrated_sum)
            for sheet in ANALYSIS_PARAMETERS["sheets"] + ["all"]:
                chosen = is_valid & ((table["sheet"] == sheet).to_numpy() if sheet != "all" else True)
                if chosen.sum() < 20:
                    continue
                drift = predicted_sum[chosen] - demonstrated_sum[chosen]
                gain = slope_through_origin(demonstrated_sum[chosen], predicted_sum[chosen])
                report.say(f"   {label:26s} {sheet:8s} {steps:5d} {int(chosen.sum()):7d} {numpy.median(numpy.abs(demonstrated_sum[chosen])):18.1f} {numpy.median(numpy.abs(drift)):14.1f} "
                           f"{numpy.quantile(numpy.abs(drift), 0.9):11.1f} {slope_through_origin(predicted_sum[chosen], demonstrated_sum[chosen]):12.2f} {gain:12.2f} "
                           f"{numpy.corrcoef(demonstrated_sum[chosen], predicted_sum[chosen])[0, 1]:11.2f} {numpy.median(numpy.abs(gain * predicted_sum[chosen] - demonstrated_sum[chosen])):32.1f}")
            if label in settings["main_runs"]:
                axis = axes[settings["main_runs"].index(label), settings["window_steps"].index(steps)]
                axis.scatter(demonstrated_sum[is_valid][::5], predicted_sum[is_valid][::5], s = 3, alpha = 0.3, c = pandas.Categorical(table["sheet"][is_valid][::5]).codes, cmap = "tab10")
                limit = numpy.nanquantile(numpy.abs(demonstrated_sum[is_valid]), 0.995)
                axis.plot([-limit, limit], [-limit, limit], color = "black", linewidth = 1)
                axis.set_xlim(-limit, limit)
                axis.set_ylim(-limit, limit)
                axis.set_xlabel(f"demonstrated joint-4 change over {steps} steps, degrees")
                axis.set_ylabel("sum of the predicted twists over the same pictures, degrees")
                axis.set_title(f"{label}, {steps} steps", fontsize = 9)
                axis.grid(alpha = 0.3)
        report.say("")
    figure.suptitle("Fold models on held-out sheets: summed predicted twist against the demonstrated joint-4 change (colour = sheet; black line = equal)")
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "05_summed_twist_scatter.png", dpi = 100)
    pyplot.close(figure)
    report.say("   'drift': predicted sum minus demonstrated sum, degrees. 'pred on demo': the share of the demonstrated turn the summed predictions give back.")
    report.say("   'demo on pred': the gain on the predicted twist that fits the summed turns best; the last column is the drift left with that gain.")
    report.say("   The pictures are the demonstration's own, so this is the model's twist with the camera where the demonstrator had it, one decision per demonstrated picture.")
    report.say("")

    report.say("   Whole episodes (all kept steps of an episode added up; episodes with a gap from the cleaning are included, the gap's step is simply missing on both sides)")
    report.say(f"   {'model':26s} {'sheet':8s} {'episodes':>8s} {'median |demo end turn|':>22s} {'median |drift at end|':>21s} {'pred on demo':>12s} {'correlation':>11s} {'median largest |drift| on the way':>33s}")
    for label in settings["main_runs"]:
        table = tables[label]
        ends = []
        for episode_id, rows in table.groupby("episode_id"):
            demonstrated_path = numpy.cumsum(rows["demonstrated"].to_numpy())
            predicted_path = numpy.cumsum(rows["predicted_twist_deg"].to_numpy())
            ends.append({"sheet": rows["sheet"].iloc[0], "demonstrated": demonstrated_path[-1], "predicted": predicted_path[-1],
                         "largest": float(numpy.abs(predicted_path - demonstrated_path).max())})
        ends = pandas.DataFrame(ends)
        for sheet, rows in list(ends.groupby("sheet")) + [("all", ends)]:
            report.say(f"   {label:26s} {sheet:8s} {len(rows):8d} {rows['demonstrated'].abs().median():22.1f} {(rows['predicted'] - rows['demonstrated']).abs().median():21.1f} "
                       f"{slope_through_origin(rows['predicted'].to_numpy(), rows['demonstrated'].to_numpy()):12.2f} {numpy.corrcoef(rows['demonstrated'], rows['predicted'])[0, 1]:11.2f} {rows['largest'].median():33.1f}")
    report.say("")

    report.say("3. How noisy the demonstrated twist is from step to step, and what smoothing the target would change")
    table = tables[settings["main_runs"][1]]
    demonstrated, run_id = table["demonstrated"].to_numpy(), table["run_id"].to_numpy()
    smoothed = centred_mean(demonstrated, run_id, settings["smoothing_half_width"])
    forward = window_sums(demonstrated, run_id, settings["forward_steps"]) / settings["forward_steps"]
    is_same_as_previous = numpy.r_[False, run_id[1:] == run_id[:-1]]
    change = numpy.diff(demonstrated, prepend = demonstrated[0])[is_same_as_previous]
    smoothed_change = numpy.diff(smoothed, prepend = smoothed[0])[is_same_as_previous]
    lag_1 = numpy.corrcoef(demonstrated[1:][is_same_as_previous[1:]], demonstrated[:-1][is_same_as_previous[1:]])[0, 1]
    report.say(f"   demonstrated twist per step (non-stop rows, all sheets, {len(demonstrated)} rows): standard deviation {demonstrated.std():.2f} degrees, mean |twist| {numpy.abs(demonstrated).mean():.2f}, "
               f"correlation between one step and the next {lag_1:+.2f}")
    report.say(f"   change of the twist from one step to the next: standard deviation {change.std():.2f} degrees (raw) against {smoothed_change.std():.2f} after smoothing over +-{settings['smoothing_half_width']} steps "
               f"(factor {smoothed_change.std() / change.std():.2f})")
    residual = demonstrated - smoothed
    report.say(f"   smoothing over +-{settings['smoothing_half_width']} steps: the smoothed twist keeps {smoothed.var() / demonstrated.var():.2f} of the variance (standard deviation {smoothed.std():.2f}), "
               f"the removed part has standard deviation {residual.std():.2f} degrees ({residual.var() / demonstrated.var():.2f} of the variance); mean |smoothed twist| {numpy.abs(smoothed).mean():.2f}")
    sign_flip = (numpy.sign(demonstrated[1:]) * numpy.sign(demonstrated[:-1]) < 0)[is_same_as_previous[1:] & (numpy.abs(demonstrated[1:]) >= 2) & (numpy.abs(demonstrated[:-1]) >= 2)]
    report.say(f"   of two steps in a row that both twist 2 degrees or more, {sign_flip.mean():.2f} twist in opposite directions")
    report.say("")
    report.say(f"   how the saved predictions fit the raw, the smoothed and the forward-averaged twist (the models were trained on the raw twist)")
    report.say(f"   {'model':26s} {'target':34s} {'mean |target|':>13s} {'mean |pred|':>11s} {'size ratio':>10s} {'correlation':>11s} {'pred on target':>14s} {'target on pred':>14s}")
    for label in ANALYSIS_PARAMETERS["fold_runs"]:
        table = tables[label]
        demonstrated, predicted, run_id = table["demonstrated"].to_numpy(), table["predicted_twist_deg"].to_numpy(), table["run_id"].to_numpy()
        targets = {"raw twist of the step": demonstrated,
                   f"smoothed over +-{settings['smoothing_half_width']} steps": centred_mean(demonstrated, run_id, settings["smoothing_half_width"]),
                   f"mean of the next {settings['forward_steps']} steps": window_sums(demonstrated, run_id, settings["forward_steps"]) / settings["forward_steps"]}
        for name, target in targets.items():
            is_valid = numpy.isfinite(target)
            report.say(f"   {label:26s} {name:34s} {numpy.abs(target[is_valid]).mean():13.2f} {numpy.abs(predicted[is_valid]).mean():11.2f} {numpy.abs(predicted[is_valid]).mean() / numpy.abs(target[is_valid]).mean():10.2f} "
                       f"{numpy.corrcoef(target[is_valid], predicted[is_valid])[0, 1]:11.2f} {slope_through_origin(predicted[is_valid], target[is_valid]):14.2f} {slope_through_origin(target[is_valid], predicted[is_valid]):14.2f}")
    report.say("")

    report.say("   per sheet, for the E recipe: how much of the step-to-step twist is noise by this measure")
    report.say(f"   {'sheet':8s} {'sd raw':>7s} {'sd smoothed':>11s} {'sd removed':>10s} {'removed share of variance':>26s} {'next-step correlation':>21s}")
    table = tables[settings["main_runs"][1]]
    for sheet, rows in table.groupby("sheet"):
        demonstrated, run_id = rows["demonstrated"].to_numpy(), rows["run_id"].to_numpy()
        smoothed = centred_mean(demonstrated, run_id, settings["smoothing_half_width"])
        is_same = numpy.r_[False, run_id[1:] == run_id[:-1]]
        report.say(f"   {sheet:8s} {demonstrated.std():7.2f} {smoothed.std():11.2f} {(demonstrated - smoothed).std():10.2f} {(demonstrated - smoothed).var() / demonstrated.var():26.2f} "
                   f"{numpy.corrcoef(demonstrated[1:][is_same[1:]], demonstrated[:-1][is_same[1:]])[0, 1]:21.2f}")

    # an example episode: raw, smoothed, predicted, and the summed joint 4
    table = tables[settings["main_runs"][1]]
    figure, axes = pyplot.subplots(2, 2, figsize = (18, 9))
    for column, plan in enumerate(["03-01", "04-01"]):
        rows = table[table["plan_number"] == plan]
        demonstrated, run_id = rows["demonstrated"].to_numpy(), rows["run_id"].to_numpy()
        smoothed = centred_mean(demonstrated, run_id, settings["smoothing_half_width"])
        axis = axes[0, column]
        axis.plot(rows["row_in_episode"], demonstrated, color = "0.6", label = "demonstrated twist per step")
        axis.plot(rows["row_in_episode"], smoothed, color = "black", linewidth = 2, label = f"smoothed over +-{settings['smoothing_half_width']} steps")
        axis.plot(rows["row_in_episode"], rows["predicted_twist_deg"], color = "tab:red", label = "predicted by the E-recipe fold model (sheet held out)")
        axis.set_title(f"plan {plan}: twist per step, degrees")
        axis.legend(fontsize = 7)
        axis.grid(alpha = 0.3)
        axis = axes[1, column]
        axis.plot(rows["row_in_episode"], numpy.cumsum(demonstrated), color = "black", linewidth = 2, label = "demonstrated joint 4 (summed labels)")
        axis.plot(rows["row_in_episode"], numpy.cumsum(rows["predicted_twist_deg"].to_numpy()), color = "tab:red", label = "summed predicted twist, E recipe")
        other = tables[settings["main_runs"][0]]
        other_rows = other[other["plan_number"] == plan]
        axis.plot(other_rows["row_in_episode"], numpy.cumsum(other_rows["predicted_twist_deg"].to_numpy()), color = "tab:orange", label = "summed predicted twist, A recipe")
        axis.set_title(f"plan {plan}: joint 4 from the start of the episode, degrees")
        axis.set_xlabel("step of the demonstration")
        axis.legend(fontsize = 7)
        axis.grid(alpha = 0.3)
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "05_twist_noise_example.png", dpi = 100)
    report.save()



if __name__ == "__main__":
    main()
