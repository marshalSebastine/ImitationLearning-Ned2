"""
Did the blue hoodie matter? Where strongly blue things were in the robot's pictures, and what model E answers when
the hoodie is painted out, or painted into pictures that never had one.

Three parts, all on the 160 x 120 pictures the model actually sees:
    where       every robot picture's share of strongly blue pixels outside the robot's own area of the picture, in
                three kinds told apart by colour (measured on sample pixels of the runs): "royal" (the hoodie:
                hue 113 or more, bright), "navy" (the dark thing beyond the table's far edge) and "robot blue" (the
                robot's own parts); the same shares for all training pictures say how usual each kind is;
    paint out   in every robot picture with royal blue, the hoodie (the royal pixels and the darker blue touching
                them) is replaced by a flat patch of the picture's own table colour and model E answers again: the change of its direction, step length, twist and stop chance, and of
                the picture's distance to the training pictures, is what the hoodie did to that picture;
    paint in    the hoodie as it looked in run 6 (top right) and in run 4 (top left) is pasted into training pictures
                and into the hoodie-free robot run 9: how far model E's answers move.

Nothing connects to the robot. Run from this folder, after extract_features.py model_e (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe hoodie_check.py
Output: hoodie_check.out.txt, hoodie_check_per_picture.csv, figures/hoodie_painted_out.png, figures/hoodie_examples.png
"""

import math
from typing import Dict, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import lines
from matplotlib import pyplot

from common import (EVIDENCE_FOLDER, FAILURE_STEPS, PLOT_STYLE, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, load_descriptors, load_policy,
                    picture_descriptors, run_small_pictures, save_montage, styled_axes, trained_frames, training_frames, training_pictures, unit_rows)
from config import PICTURE_CHECK_PARAMETERS
from neighbours import nearest_rows

HOODIE_PARAMETERS = {
    # OpenCV hue runs 0 to 180; the limits were read off sample pixels of runs 4, 5, 6 and 9
    "blue_hue": [100, 125],
    "blue_saturation_at_least": 200,
    "royal_hue_at_least": 113,
    "bright_value_at_least": 110,
    "navy_value_at_least": 40,
    "in_view_share": 0.005,
    "table_hue": [15, 40],
    "table_saturation": [20, 110],
    "table_value": [90, 210],
    "table_colour_default": [129, 128, 107],
    "mask_growth_px": 1,
    "royal_pixels_in_patch": 10,
    # the hoodie shapes pasted into other pictures: (run, step)
    "pasted_shapes": {"top right, as in run 6": (6, 30), "top left, as in run 4": (4, 30)},
    "training_sample": 600,
    "neighbour_count": 5,
    "seed": 11_11_26,
    "chunk_pictures": 1000
}



def blue_masks(
        small_pictures: numpy.ndarray,
        parameters: Dict
) -> Dict[str, numpy.ndarray]:
    """
    Mark the strongly blue pixels of 160 x 120 pictures outside the robot's own area, by kind.

    Parameters
    ----------
    small_pictures : numpy.ndarray
        (pictures, 120, 160, 3), uint8 RGB.
    parameters : dict
        ``HOODIE_PARAMETERS``.

    Returns
    -------
    dict of str to numpy.ndarray
        "royal", "navy" and "robot_blue": boolean masks, (pictures, 120, 160).
    """

    pictures = numpy.asarray(small_pictures)
    hsv = numpy.stack([cv2.cvtColor(picture, cv2.COLOR_RGB2HSV) for picture in pictures])
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    is_strong_blue = (hue >= parameters["blue_hue"][0]) & (hue <= parameters["blue_hue"][1]) & (saturation >= parameters["blue_saturation_at_least"])
    # the robot's hand, block and gripper sit in "robot_area_px" of the 640 x 480 picture (left, top, right, bottom)
    left, top, right, bottom = [int(round(edge / 4)) for edge in PICTURE_CHECK_PARAMETERS["robot_area_px"]]
    is_outside_robot = numpy.ones(pictures.shape[1:3], dtype = bool)
    is_outside_robot[top:bottom, left:right] = False
    is_strong_blue = is_strong_blue & is_outside_robot[None]
    is_bright = value >= parameters["bright_value_at_least"]
    return {"royal": is_strong_blue & (hue >= parameters["royal_hue_at_least"]) & is_bright,
            "navy": is_strong_blue & ~is_bright & (value >= parameters["navy_value_at_least"]),
            "robot_blue": is_strong_blue & (hue < parameters["royal_hue_at_least"]) & is_bright}



def whole_hoodie_masks(
        masks: Dict[str, numpy.ndarray],
        parameters: Dict
) -> numpy.ndarray:
    """
    Join the hoodie's bright royal blue with the darker blue that touches it (its folds and shaded side).

    The dark "navy" pixels alone are not the hoodie: the dark thing beyond the table's far edge has the same colour and
    is in many training pictures. Only a patch of strong blue that holds at least "royal_pixels_in_patch" royal pixels
    counts as the hoodie.

    Parameters
    ----------
    masks : dict of str to numpy.ndarray
        From ``blue_masks``.
    parameters : dict
        ``HOODIE_PARAMETERS``.

    Returns
    -------
    numpy.ndarray
        Boolean masks of the whole hoodie, (pictures, 120, 160).
    """

    result = numpy.zeros_like(masks["royal"])
    for index, (royal, navy) in enumerate(zip(masks["royal"], masks["navy"])):
        if royal.sum() < parameters["royal_pixels_in_patch"]:
            continue
        count, labels = cv2.connectedComponents((royal | navy).astype(numpy.uint8), connectivity = 8)
        for label in range(1, count):
            patch = labels == label
            if (patch & royal).sum() >= parameters["royal_pixels_in_patch"]:
                result[index] |= patch
    return result



def painted_out(
        small_pictures: numpy.ndarray,
        masks: numpy.ndarray,
        parameters: Dict
) -> numpy.ndarray:
    """
    Replace the masked pixels of every picture by that picture's own table colour.

    Parameters
    ----------
    small_pictures : numpy.ndarray
        (pictures, 120, 160, 3), uint8 RGB.
    masks : numpy.ndarray
        (pictures, 120, 160) boolean; grown by "mask_growth_px" before painting, so the blended rim goes too.
    parameters : dict
        ``HOODIE_PARAMETERS`` (the table's colour limits).

    Returns
    -------
    numpy.ndarray
        The painted pictures, uint8 RGB.
    """

    result = numpy.array(small_pictures, copy = True)
    size = 2 * parameters["mask_growth_px"] + 1
    kernel = numpy.ones((size, size), dtype = numpy.uint8)
    for index, (picture, mask) in enumerate(zip(small_pictures, masks)):
        if not mask.any():
            continue
        hsv = cv2.cvtColor(numpy.ascontiguousarray(picture), cv2.COLOR_RGB2HSV)
        is_table = ((hsv[..., 0] >= parameters["table_hue"][0]) & (hsv[..., 0] <= parameters["table_hue"][1])
                    & (hsv[..., 1] >= parameters["table_saturation"][0]) & (hsv[..., 1] <= parameters["table_saturation"][1])
                    & (hsv[..., 2] >= parameters["table_value"][0]) & (hsv[..., 2] <= parameters["table_value"][1]))
        colour = numpy.median(picture[is_table], axis = 0) if is_table.sum() >= 30 else numpy.array(parameters["table_colour_default"])
        grown = cv2.dilate(mask.astype(numpy.uint8), kernel).astype(bool)
        result[index][grown] = colour.astype(numpy.uint8)
    return result



def answers_and_distance(
        policy,
        small_pictures: numpy.ndarray,
        reference_unit: numpy.ndarray,
        count: int
) -> pandas.DataFrame:
    """Return model E's answers to a stack of pictures and each picture's mean distance to its nearest training pictures."""

    descriptors = picture_descriptors(policy, small_pictures)
    distance = nearest_rows(unit_rows(descriptors["grid"]), reference_unit, {"all": None}, {"count": count})["all"]["distance"].mean(axis = 1)
    exact = descriptors["exact"]
    return pandas.DataFrame({"forward_cm": exact[:, 0], "left_cm": exact[:, 1], "length_cm": numpy.hypot(exact[:, 0], exact[:, 1]),
                             "angle_deg": numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0])), "twist_deg": exact[:, 2],
                             "stop_probability": descriptors["stop_probability"], "distance": distance})



def answer_changes(
        before: pandas.DataFrame,
        after: pandas.DataFrame
) -> pandas.DataFrame:
    """Return how far the answers moved: the angle between the two directions, and the changes of length, twist, stop chance and distance."""

    angle = numpy.abs((after["angle_deg"].to_numpy() - before["angle_deg"].to_numpy() + 180.0) % 360.0 - 180.0)
    return pandas.DataFrame({"angle_change_deg": angle,
                             "vector_change_cm": numpy.hypot(after["forward_cm"].to_numpy() - before["forward_cm"].to_numpy(), after["left_cm"].to_numpy() - before["left_cm"].to_numpy()),
                             "length_change_cm": after["length_cm"].to_numpy() - before["length_cm"].to_numpy(),
                             "twist_change_deg": after["twist_deg"].to_numpy() - before["twist_deg"].to_numpy(),
                             "stop_change": after["stop_probability"].to_numpy() - before["stop_probability"].to_numpy(),
                             "distance_change": after["distance"].to_numpy() - before["distance"].to_numpy()})



def say_where(
        robot_shares: pandas.DataFrame,
        parameters: Dict,
        log: OutputLog
) -> None:
    """Say in which pictures of each run strongly blue things were in view, and how usual they are in the training pictures."""

    limit = parameters["in_view_share"]
    cache = training_pictures()
    training_shares = []
    for start in range(0, len(cache), parameters["chunk_pictures"]):
        masks = blue_masks(cache[start:start + parameters["chunk_pictures"]], parameters)
        training_shares.append(pandas.DataFrame({name: mask.mean(axis = (1, 2)) for name, mask in masks.items()}))
    training_shares = pandas.concat(training_shares).reset_index(drop = True)

    log.say("WHERE STRONGLY BLUE THINGS WERE (share of the 160 x 120 picture outside the robot's own area; 'in view' = more than 0.5 % of the picture)")
    log.say("")
    log.say("Training pictures (all 18,522): how often each kind is in view")
    for name in ("royal", "navy", "robot_blue"):
        log.say(f"  {name:10s}: in view in {100 * (training_shares[name] > limit).mean():5.2f} % of the training pictures "
                f"({int((training_shares[name] > limit).sum())} pictures); above 5 % of the picture in {int((training_shares[name] > 0.05).sum())} pictures; largest share {100 * training_shares[name].max():.1f} %")
    log.say("")
    log.say("Robot runs: pictures with each kind in view, the first such step and the largest share; and where the run went wrong")
    log.say("run  pictures | royal (hoodie): pictures, first step, largest % | navy: pictures, first step, largest % | robot blue: pictures, largest % | first wrong decision, puck off line")
    for run_number in ROBOT_RUNS:
        rows = robot_shares[robot_shares["run"] == run_number]
        cells = []
        for name in ("royal", "navy", "robot_blue"):
            seen = rows[rows[name] > limit]
            first = "none" if seen.empty else f"{int(seen['step'].min())}"
            cells.append(f"{len(seen):3d}, {first:>4s}, {100 * rows[name].max():5.1f}")
        failure = FAILURE_STEPS.get(run_number)
        log.say(f"{run_number:3d}  {len(rows):8d} | {cells[0]:>45s} | {cells[1]:>35s} | {cells[2]:>30s} | "
                + (f"{failure['first_wrong_decision']}, {failure['puck_off_line']}" if failure else "reached the end"))



def paint_out_table(
        policy,
        reference_unit: numpy.ndarray,
        parameters: Dict
) -> pandas.DataFrame:
    """Paint the royal blue out of every robot picture that has it and compare model E's answers; returns one row per picture."""

    count = parameters["neighbour_count"]
    tables = []
    for run_number in ROBOT_RUNS:
        pictures = run_small_pictures(run_number)
        masks = blue_masks(pictures, parameters)
        share = masks["royal"].mean(axis = (1, 2))
        before = answers_and_distance(policy, pictures, reference_unit, count)
        after = answers_and_distance(policy, painted_out(pictures, whole_hoodie_masks(masks, parameters), parameters), reference_unit, count)
        table = pandas.concat([pandas.DataFrame({"run": run_number, "step": numpy.arange(len(pictures)), "royal": share,
                                                 "navy": masks["navy"].mean(axis = (1, 2)), "robot_blue": masks["robot_blue"].mean(axis = (1, 2))}),
                               before.add_prefix("with_hoodie_"), after.add_prefix("painted_out_"), answer_changes(before, after)], axis = 1)
        tables.append(table)
    return pandas.concat(tables).reset_index(drop = True)



def say_paint_out(
        table: pandas.DataFrame,
        parameters: Dict,
        log: OutputLog
) -> None:
    """Say what painting the hoodie out changed, for every run that had it in view."""

    limit = parameters["in_view_share"]
    log.say("")
    log.say("PAINT OUT: model E's answer to each robot picture with the hoodie against the same picture with the hoodie replaced by table colour")
    log.say("(only pictures with the hoodie in view; medians; 'direction' = angle between the two asked directions)")
    log.say("run  steps with hoodie   hoodie share median | direction change deg: median, 90 % | asked length cm: with, without | twist change deg (median abs) | stop chance change (largest abs) | distance to training pictures: with, without")
    for run_number in ROBOT_RUNS:
        rows = table[(table["run"] == run_number) & (table["royal"] > limit)]
        if rows.empty:
            continue
        log.say(f"{run_number:3d}  {int(rows['step'].min()):3d} to {int(rows['step'].max()):3d} ({len(rows):2d})   {100 * rows['royal'].median():16.1f} % | "
                f"{rows['angle_change_deg'].median():27.1f}, {rows['angle_change_deg'].quantile(0.9):5.1f} | {rows['with_hoodie_length_cm'].median():21.2f}, {rows['painted_out_length_cm'].median():5.2f} | "
                f"{rows['twist_change_deg'].abs().median():28.2f} | {rows['stop_change'].abs().max():31.3f} | {rows['with_hoodie_distance'].median():34.3f}, {rows['painted_out_distance'].median():.3f}")



def paint_in_test(
        policy,
        reference_unit: numpy.ndarray,
        parameters: Dict,
        log: OutputLog
) -> None:
    """Paste the hoodie as it looked in runs 6 and 4 into training pictures and into the hoodie-free run 9, and compare model E's answers."""

    count = parameters["neighbour_count"]
    randomness_generator = numpy.random.default_rng(parameters["seed"])
    frames = training_frames()
    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"])
    moving = trained[trained["action"] != "stop"]
    chosen = numpy.sort(randomness_generator.choice(moving["frame_index"].to_numpy(), size = parameters["training_sample"], replace = False))
    training_sample = numpy.asarray(training_pictures()[chosen])
    sample_masks = blue_masks(training_sample, parameters)
    is_clean = ~(sample_masks["royal"].mean(axis = (1, 2)) > parameters["in_view_share"])
    training_sample = training_sample[is_clean]
    sets = {f"{len(training_sample)} training pictures (moving rows, all sheets)": training_sample,
            "run 9, all 90 pictures (no hoodie)": run_small_pictures(9),
            "run 9, steps 0 to 5 (before it went wrong)": run_small_pictures(9)[:6],
            "run 6, steps 0 to 14 (before the hoodie came into view)": run_small_pictures(6)[:15]}

    log.say("")
    log.say("PAINT IN: the hoodie pasted into pictures that had none; how far model E's answers move (medians; in brackets the 90 % point)")
    log.say("pasted shape (share of the picture)      into                                                      | direction change deg | vector change cm | asked length: before, after | twist change deg (abs) | stop chance: largest rise | distance: before, after")
    for label, (run_number, step) in parameters["pasted_shapes"].items():
        source = run_small_pictures(run_number)[step]
        shape = whole_hoodie_masks(blue_masks(source[None], parameters), parameters)[0]
        for set_label, pictures in sets.items():
            pasted = numpy.array(pictures, copy = True)
            pasted[:, shape] = source[shape]
            before = answers_and_distance(policy, pictures, reference_unit, count)
            after = answers_and_distance(policy, pasted, reference_unit, count)
            change = answer_changes(before, after)
            log.say(f"{label:24s} ({100 * shape.mean():4.1f} %)  {set_label:56s} | {change['angle_change_deg'].median():11.1f} ({change['angle_change_deg'].quantile(0.9):5.1f}) | "
                    f"{change['vector_change_cm'].median():9.2f} ({change['vector_change_cm'].quantile(0.9):4.2f}) | {before['length_cm'].median():18.2f}, {after['length_cm'].median():5.2f} | "
                    f"{change['twist_change_deg'].abs().median():13.2f} ({change['twist_change_deg'].abs().quantile(0.9):4.2f}) | {change['stop_change'].max():25.3f} | "
                    f"{before['distance'].median():15.3f}, {after['distance'].median():.3f}")



def draw_painted_out(
        table: pandas.DataFrame,
        figure_path
) -> None:
    """Draw, for the two runs with the hoodie in view, its share of the picture and model E's asked step length and distance with and without it."""

    runs = [4, 6]
    figure, grid = pyplot.subplots(3, 2, figsize = (13, 8.6), facecolor = PLOT_STYLE["surface"], sharex = "col")
    panels = [("royal", None, "hoodie's share of the picture (%)"),
              ("with_hoodie_length_cm", "painted_out_length_cm", "step length model E asked for (cm)"),
              ("with_hoodie_distance", "painted_out_distance", "distance to the 5 nearest training pictures")]
    for column, run_number in enumerate(runs):
        rows = table[table["run"] == run_number]
        for row, (first, second, label) in enumerate(panels):
            axes = grid[row, column]
            styled_axes(axes)
            scale = 100.0 if first == "royal" else 1.0
            if run_number in FAILURE_STEPS:
                axes.axvspan(FAILURE_STEPS[run_number]["puck_off_line"] - 0.5, rows["step"].max() + 0.5, color = PLOT_STYLE["went_wrong"], alpha = 0.08, linewidth = 0)
                axes.axvline(FAILURE_STEPS[run_number]["first_wrong_decision"], color = PLOT_STYLE["went_wrong"], linewidth = 1.4)
            axes.plot(rows["step"], scale * rows[first], color = PLOT_STYLE["series"][0], linewidth = 1.8)
            if second is not None:
                axes.plot(rows["step"], rows[second], color = PLOT_STYLE["series"][1], linewidth = 1.8)
            axes.set_ylabel(label, fontsize = 8, color = PLOT_STYLE["ink_secondary"])
            axes.set_ylim(bottom = 0)
            if row == 0:
                outcome = "went wrong" if run_number in FAILURE_STEPS else "reached the end"
                axes.set_title(f"Run {run_number}: sheet {ROBOT_RUNS[run_number]['sheet'][-2:]}, {outcome}", fontsize = 9.5, color = PLOT_STYLE["ink"], loc = "left")
            if row == 2:
                axes.set_xlabel("picture (step) of the run", fontsize = 8, color = PLOT_STYLE["ink_secondary"])
    keys = [lines.Line2D([], [], color = PLOT_STYLE["series"][0], linewidth = 1.8, label = "the picture as the robot saw it"),
            lines.Line2D([], [], color = PLOT_STYLE["series"][1], linewidth = 1.8, label = "the same picture with the hoodie painted out (table colour)"),
            lines.Line2D([], [], color = PLOT_STYLE["went_wrong"], linewidth = 1.4, label = "first wrong decision (shaded: after the puck left the line)")]
    figure.legend(handles = keys, loc = "upper left", bbox_to_anchor = (0.008, 0.955), ncol = 3, fontsize = 8.5, frameon = False, labelcolor = PLOT_STYLE["ink"])
    figure.suptitle("The blue hoodie: when it was in view, and what model E asks for with it and with it painted out", fontsize = 10.5, color = PLOT_STYLE["ink"], x = 0.01, ha = "left")
    figure.tight_layout(rect = (0, 0, 1, 0.93))
    figure.savefig(figure_path, dpi = 110, facecolor = PLOT_STYLE["surface"])
    pyplot.close(figure)



def draw_examples(
        parameters: Dict,
        figure_path
) -> None:
    """Draw example pictures as the model saw them and with the hoodie painted out, enlarged four times."""

    tiles = []
    for run_number, step in ((4, 30), (6, 20), (6, 30)):
        picture = run_small_pictures(run_number)[step][None]
        masks = blue_masks(picture, parameters)
        painted = painted_out(picture, whole_hoodie_masks(masks, parameters), parameters)
        pair = numpy.hstack([picture[0], painted[0]])
        pair = cv2.resize(cv2.cvtColor(pair, cv2.COLOR_RGB2BGR), None, fx = 3, fy = 3, interpolation = cv2.INTER_NEAREST)
        cv2.rectangle(pair, (0, 0), (pair.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(pair, f"run {run_number} step {step}: as seen (left), hoodie painted out (right)", (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(pair)
    save_montage(figure_path, numpy.vstack(tiles))



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(HOODIE_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "hoodie_check.out.txt")
    policy = load_policy(VIEW_NOVELTY_PARAMETERS["model_e_run_name"], VIEW_NOVELTY_PARAMETERS["model_e_folder"])
    frames = training_frames()
    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"])
    reference_index = frames.loc[frames["frame_index"].isin(trained["frame_index"]), "frame_index"].to_numpy()
    reference_unit = unit_rows(load_descriptors("model_e_training")["grid"][reference_index])

    table = paint_out_table(policy, reference_unit, settings)
    say_where(table, settings, log)
    say_paint_out(table, settings, log)
    paint_in_test(policy, reference_unit, settings, log)

    table.round(5).to_csv(EVIDENCE_FOLDER / "hoodie_check_per_picture.csv", index = False)
    figure_folder = EVIDENCE_FOLDER / "figures"
    figure_folder.mkdir(exist_ok = True)
    draw_painted_out(table, figure_folder / "hoodie_painted_out.png")
    draw_examples(settings, figure_folder / "hoodie_examples.png")
    log.say("")
    log.say("written: hoodie_check_per_picture.csv, figures/hoodie_painted_out.png, figures/hoodie_examples.png")
    log.save()



if __name__ == "__main__":
    main()
