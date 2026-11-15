"""
Run 6 (sheet 04, model E): the back-and-forth steps, the demonstration pictures that look most alike, and what the
demonstrations did next.

Model E describes every picture with the 256 numbers of its hidden layer; its answer is computed from those numbers
alone, so demonstration pictures with nearly the same numbers are the ones whose labels shaped the answer. For every
picture of the run the script finds the "neighbours" nearest demonstration pictures (of the 18,032 the model was trained
on) and reads their labels: the tip move, the forearm twist, the move over the next 3 steps, the sheet, the direction of
travel and the kind of place. If the answer is the average of two opposite demonstrated ways, the neighbours must split
into two groups with opposite moves and twists while the model's own answer is short.

Also printed: how far the run's pictures are from any demonstration picture, against how far demonstration pictures are
from the pictures of other episodes (a large distance means a view the demonstrations never showed).

Needs the cache of s02_model_outputs.py (jobs all_sheets_e and robot_runs).
Run from this folder:  python s04_run6_neighbours.py
Output: s04_run6_neighbours.out.txt, s04_run6_neighbours.csv, s04_run6_neighbours.jpg, s04_run6_pictures.jpg,
s04_run6_path.png.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy
import pandas
from matplotlib import pyplot

import common
from config import TRAINING_DATA_PARAMETERS

NEIGHBOUR_PARAMETERS = {
    "run_folder": "2026-11-11_174108_E_sheet04_run1",
    "neighbours": 12,
    "flip_steps": [19, 50],
    "reference_steps": [0, 4],
    "sideways_cm": 0.3,
    "twist_counts_from_deg": 2.0,
    "reference_sample": 600,
    "montage_steps": [3, 16, 19, 20, 22, 24, 30, 40],
    "montage_neighbours": 5,
    "picture_steps": [4, 6, 8, 9, 11, 13, 16, 18, 19, 20, 22, 24],
    "seed": 11_11_26
}



def nearest_rows(
        queries: numpy.ndarray,
        references: numpy.ndarray,
        count: int
) -> tuple:
    """Return the positions and distances of the ``count`` nearest reference rows of every query row (Euclidean)."""

    queries = queries.astype(numpy.float32)
    references = references.astype(numpy.float32)
    squared = (queries ** 2).sum(axis = 1)[:, None] - 2.0 * queries @ references.T + (references ** 2).sum(axis = 1)[None, :]
    order = numpy.argsort(squared, axis = 1)[:, :count]
    return order, numpy.sqrt(numpy.clip(numpy.take_along_axis(squared, order, axis = 1), 0.0, None))



def draw_arrow(
        picture: numpy.ndarray,
        forward_cm: float,
        left_cm: float,
        colour: tuple,
        pixels_per_cm: float = 60.0
) -> None:
    """Draw a move given in the picture's frame as an arrow from the picture's middle (forward = up, left = left)."""

    height, width = picture.shape[:2]
    start = (width // 2, height // 2)
    end = (int(start[0] - pixels_per_cm * left_cm), int(start[1] - pixels_per_cm * forward_cm))
    cv2.arrowedLine(picture, start, end, colour, 3, tipLength = 0.3)



def main(parameters: Optional[Dict] = None) -> None:

    settings = common.merged_parameters(dict(NEIGHBOUR_PARAMETERS, **(parameters or {})))
    report = common.TeeOutput(common.ANALYSIS_FOLDER / "s04_run6_neighbours.out.txt")
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 400)
    frames = common.load_demonstration_frames(settings)
    path = common.path_direction(frames, settings)
    demonstrations = numpy.load(Path(settings["cache_folder"]) / "all_sheets_e_on_demonstrations.npz")
    robot = numpy.load(Path(settings["cache_folder"]) / "all_sheets_e_on_robot_runs.npz")
    run_name = settings["run_folder"]
    steps = pandas.read_csv(Path(settings["model_runs_folder"]) / run_name / "steps.csv")
    hidden = robot[f"{run_name}__hidden"].astype(numpy.float32)
    chunk = robot[f"{run_name}__chunk"].reshape(len(steps), 5, 2)
    exact = robot[f"{run_name}__exact"]
    reference_hidden = demonstrations["hidden"].astype(numpy.float32)

    # how far apart are demonstration pictures of different episodes, as a yardstick
    randomness_generator = numpy.random.default_rng(settings["seed"])
    sample = randomness_generator.choice(len(frames), size = settings["reference_sample"], replace = False)
    order, distance = nearest_rows(reference_hidden[sample], reference_hidden, 400)
    episode_ids = frames["episode_id"].to_numpy()
    other_episode_distance = []
    for position, row in enumerate(sample):
        is_other = episode_ids[order[position]] != episode_ids[row]
        other_episode_distance.append(distance[position][is_other][0] if is_other.any() else numpy.nan)
    other_episode_distance = numpy.array(other_episode_distance)
    report(f"yardstick: distance from a demonstration picture to the nearest picture of ANOTHER episode (hidden layer of model E, {len(sample)} random pictures): "
           f"median {numpy.nanmedian(other_episode_distance):.2f}, 90th percentile {numpy.nanpercentile(other_episode_distance, 90):.2f}, "
           f"99th {numpy.nanpercentile(other_episode_distance, 99):.2f}")

    # the same question for ordinary demonstration pictures: how often do the nearest pictures of OTHER episodes disagree?
    label_forward = frames["step_forward_cm"].to_numpy()
    label_left = frames["step_left_cm"].to_numpy()
    sample_places = frames["place"].to_numpy()[sample]
    yardstick_rows = []
    for position, row in enumerate(sample):
        is_other = episode_ids[order[position]] != episode_ids[row]
        near = order[position][is_other][:settings["neighbours"]]
        if len(near) < settings["neighbours"]:
            continue
        forward, left = label_forward[near], label_left[near]
        length = numpy.hypot(forward, left)
        is_real = length >= settings["short_step_cm"]
        resultant = float(numpy.hypot((forward[is_real] / length[is_real]).sum(), (left[is_real] / length[is_real]).sum()) / max(1, is_real.sum()))
        yardstick_rows.append({"place": sample_places[position], "resultant": resultant,
                               "both_groups": bool((left >= settings["sideways_cm"]).sum() >= 3 and (left <= -settings["sideways_cm"]).sum() >= 3),
                               "length_of_mean": float(numpy.hypot(forward.mean(), left.mean())), "mean_length": float(length.mean())})
    yardstick = pandas.DataFrame(yardstick_rows)
    report(f"yardstick for the neighbours' agreement ({len(yardstick)} random demonstration pictures, their {settings['neighbours']} nearest pictures of other episodes): "
           f"resultant median {yardstick['resultant'].median():.2f} | pictures whose neighbours hold 3 or more of both the left and the right group: {yardstick['both_groups'].mean():.2f} | "
           f"length of the neighbours' mean move {yardstick['length_of_mean'].mean():.2f} cm against their mean length {yardstick['mean_length'].mean():.2f} cm")
    for place, part in yardstick.groupby("place"):
        report(f"    {place:12s} | pictures {len(part):3d} | resultant median {part['resultant'].median():.2f} | both groups {part['both_groups'].mean():.2f} | "
               f"length of the neighbours' mean move {part['length_of_mean'].mean():.2f} cm")

    order, distance = nearest_rows(hidden, reference_hidden, settings["neighbours"])
    label_twist = frames["twist_deg"].to_numpy()
    ahead_forward = path["ahead_3_forward_cm"].to_numpy()
    ahead_left = path["ahead_3_left_cm"].to_numpy()
    sheets = frames["sheet"].to_numpy()
    places = frames["place"].to_numpy()
    from_end = numpy.where(frames["travels_towards_A"].isna(), "?", numpy.where(frames["travels_towards_A"].fillna(False).astype(bool), "B", "A"))
    bend = frames["bend_turn_deg"].to_numpy()
    arcs = frames["puck_arc_cm"].to_numpy()

    rows = []
    for step in range(len(steps)):
        near = order[step]
        forward = label_forward[near]
        left = label_left[near]
        length = numpy.hypot(forward, left)
        is_left_group = left >= settings["sideways_cm"]
        is_right_group = left <= -settings["sideways_cm"]
        is_real = length >= settings["short_step_cm"]
        unit_forward = numpy.where(is_real, forward / numpy.maximum(length, 1e-9), 0.0)
        unit_left = numpy.where(is_real, left / numpy.maximum(length, 1e-9), 0.0)
        resultant = float(numpy.hypot(unit_forward.sum(), unit_left.sum()) / max(1, is_real.sum()))
        rows.append({
            "step": step,
            "model_forward_cm": round(float(exact[step, 0]), 2), "model_left_cm": round(float(exact[step, 1]), 2),
            "model_length_cm": round(float(numpy.hypot(exact[step, 0], exact[step, 1])), 2),
            "model_deg": round(float(numpy.degrees(numpy.arctan2(exact[step, 1], exact[step, 0]))), 0),
            "model_twist_deg": round(float(exact[step, 2]), 1),
            "chunk3_cm": round(float(numpy.linalg.norm(chunk[step, 2])), 2),
            "chunk3_deg": round(float(numpy.degrees(numpy.arctan2(chunk[step, 2, 1], chunk[step, 2, 0]))), 0),
            "nearest_distance": round(float(distance[step, 0]), 2),
            "neighbours_mean_forward_cm": round(float(forward.mean()), 2), "neighbours_mean_left_cm": round(float(left.mean()), 2),
            "neighbours_mean_length_cm": round(float(length.mean()), 2),
            "length_of_neighbours_mean_cm": round(float(numpy.hypot(forward.mean(), left.mean())), 2),
            "neighbours_resultant": round(resultant, 2),
            "neighbours_left_group": int(is_left_group.sum()), "neighbours_right_group": int(is_right_group.sum()),
            "neighbours_short": int((~is_real).sum()),
            "neighbours_twist_right": int((label_twist[near] >= settings["twist_counts_from_deg"]).sum()),
            "neighbours_twist_left": int((label_twist[near] <= -settings["twist_counts_from_deg"]).sum()),
            "neighbours_mean_twist_deg": round(float(label_twist[near].mean()), 1),
            "neighbours_sheet_04": int((sheets[near] == "path_04").sum()),
            "neighbours_from_A": int((from_end[near] == "A").sum()), "neighbours_from_B": int((from_end[near] == "B").sum()),
            "neighbours_bend_right": int((bend[near] <= -60).sum()), "neighbours_bend_left": int((bend[near] >= 60).sum()),
            "neighbours_hairpin_or_sharp": int(numpy.isin(places[near], ["hairpin", "sharp bend"]).sum())
        })
    table = pandas.DataFrame(rows)
    table.to_csv(common.ANALYSIS_FOLDER / "s04_run6_neighbours.csv", index = False)
    report("")
    report(f"== run {run_name}: every picture, the model's answer and the labels of its {settings['neighbours']} nearest demonstration pictures ==")
    report("(left group / right group: neighbours whose labelled tip move goes 0.3 cm or more to the picture's left / right; resultant: 1 when the neighbours' real "
           "moves all point one way, near 0 when they point in opposite ways; twist right = forearm change of +2 degrees or more)")
    report(table.to_string(index = False))

    first, last = settings["flip_steps"]
    flip = table[(table["step"] >= first) & (table["step"] <= last)]
    start_first, start_last = settings["reference_steps"]
    start = table[(table["step"] >= start_first) & (table["step"] <= start_last)]
    report("")
    report("== summary: the back-and-forth steps against the first steps of the same run ==")
    for label, part in ((f"steps {first} to {last} (back and forth)", flip), (f"steps {start_first} to {start_last} (on the line)", start)):
        both = ((part["neighbours_left_group"] >= 3) & (part["neighbours_right_group"] >= 3)).mean()
        both_twists = ((part["neighbours_twist_right"] >= 3) & (part["neighbours_twist_left"] >= 3)).mean()
        report(f"{label}: pictures {len(part)} | model step median {part['model_length_cm'].median():.2f} cm | neighbours' label length mean {part['neighbours_mean_length_cm'].mean():.2f} cm | "
               f"length of the neighbours' mean move {part['length_of_neighbours_mean_cm'].mean():.2f} cm | resultant median {part['neighbours_resultant'].median():.2f} | "
               f"pictures whose neighbours hold 3 or more of BOTH the left and the right group: {both:.2f} | both twist signs (3 or more each): {both_twists:.2f} | "
               f"nearest distance median {part['nearest_distance'].median():.2f} | neighbours from sheet 04: {part['neighbours_sheet_04'].mean():.1f} of {settings['neighbours']}, "
               f"from end A {part['neighbours_from_A'].mean():.1f}, from end B {part['neighbours_from_B'].mean():.1f} | at a bend to the right {part['neighbours_bend_right'].mean():.1f}, "
               f"to the left {part['neighbours_bend_left'].mean():.1f}")

    # does the answer's side follow the side most of its neighbours went to?
    is_decided = flip["neighbours_left_group"] != flip["neighbours_right_group"]
    is_majority_left = flip["neighbours_left_group"] > flip["neighbours_right_group"]
    is_followed = (flip["model_left_cm"] > 0) == is_majority_left
    twist_decided = flip["neighbours_twist_right"] != flip["neighbours_twist_left"]
    twist_followed = (flip["model_twist_deg"] > 0) == (flip["neighbours_twist_right"] > flip["neighbours_twist_left"])
    report(f"steps {first} to {last}: the answer went to the side most of its neighbours went to in {int((is_followed & is_decided).sum())} of the {int(is_decided.sum())} pictures "
           f"with an unequal count; its twist had the sign of most neighbours' twists in {int((twist_followed & twist_decided).sum())} of {int(twist_decided.sum())}")

    # the model's two alternating answers during the back and forth
    flip_steps = steps.iloc[first:last + 1]
    is_right = flip_steps["model_tip_left_cm"] < 0
    report("")
    report(f"== the two answers the model alternated between, steps {first} to {last} (from steps.csv) ==")
    for label, part in (("answers to the picture's right", flip_steps[is_right]), ("answers to the picture's left", flip_steps[~is_right])):
        report(f"{label}: {len(part)} | forward {part['model_tip_forward_cm'].mean():+.2f} cm, left {part['model_tip_left_cm'].mean():+.2f} cm, length median "
               f"{numpy.hypot(part['model_tip_forward_cm'], part['model_tip_left_cm']).median():.2f} cm | twist mean {part['model_twist_deg'].mean():+.2f} deg, "
               f"share of twists to the right (positive) {float((part['model_twist_deg'] > 0).mean()):.2f}")
    tip = flip_steps[["tip_x", "tip_y"]].to_numpy() * 100.0
    report(f"tip positions during the back and forth: x {tip[:, 0].min():.1f} to {tip[:, 0].max():.1f} cm, y {tip[:, 1].min():.1f} to {tip[:, 1].max():.1f} cm; "
           f"net displacement {numpy.linalg.norm(tip[-1] - tip[0]):.1f} cm over {len(tip) - 1} moves of 1 cm; "
           f"forearm (joint 4) {numpy.degrees(flip_steps['joint_4']).min():.1f} to {numpy.degrees(flip_steps['joint_4']).max():.1f} deg; "
           f"sum of the asked step lengths {numpy.hypot(flip_steps['model_tip_forward_cm'], flip_steps['model_tip_left_cm']).sum():.1f} cm")
    change = common.angle_difference_deg(flip_steps["model_angle_deg"].to_numpy()[:-1] + flip_steps["picture_heading_deg"].to_numpy()[:-1],
                                         flip_steps["model_angle_deg"].to_numpy()[1:] + flip_steps["picture_heading_deg"].to_numpy()[1:])
    report(f"direction changes over 90 degrees between consecutive answers: {int((change > 90).sum())} of {len(change)}")

    # ---- montage ----
    raw_folder = Path(TRAINING_DATA_PARAMETERS["raw_folder"])
    tiles_rows = []
    for step in settings["montage_steps"]:
        picture = cv2.imread(str(Path(settings["model_runs_folder"]) / run_name / "frames" / f"{step:03d}.jpg"))
        picture = cv2.resize(picture, (320, 240), interpolation = cv2.INTER_AREA)
        length = float(numpy.hypot(exact[step, 0], exact[step, 1]))
        draw_arrow(picture, exact[step, 0] / max(length, 1e-6), exact[step, 1] / max(length, 1e-6), (0, 255, 0))
        cv2.putText(picture, f"run 6 step {step}: {length:.2f} cm, twist {exact[step, 2]:+.1f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles = [picture]
        for near in order[step][:settings["montage_neighbours"]]:
            neighbour = cv2.imread(str(raw_folder / frames["frame_path"].iloc[near]))
            neighbour = cv2.resize(neighbour, (320, 240), interpolation = cv2.INTER_AREA)
            draw_arrow(neighbour, ahead_forward[near] / 3.0, ahead_left[near] / 3.0, (0, 200, 255))
            draw_arrow(neighbour, label_forward[near], label_left[near], (0, 255, 0))
            cv2.putText(neighbour, f"{sheets[near][-2:]} from {from_end[near]} arc {arcs[near]:.0f} {places[near]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(neighbour, f"move {label_forward[near]:+.1f},{label_left[near]:+.1f} tw {label_twist[near]:+.0f}", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(neighbour)
        tiles_rows.append(numpy.hstack(tiles))
    cv2.imwrite(str(common.ANALYSIS_FOLDER / "s04_run6_neighbours.jpg"), numpy.vstack(tiles_rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

    # ---- the run's own pictures: where the puck left the line, and the back and forth ----
    tiles = []
    for step in settings["picture_steps"]:
        picture = cv2.imread(str(Path(settings["model_runs_folder"]) / run_name / "frames" / f"{step:03d}.jpg"))
        picture = cv2.resize(picture, (480, 360), interpolation = cv2.INTER_AREA)
        length = float(numpy.hypot(exact[step, 0], exact[step, 1]))
        draw_arrow(picture, exact[step, 0] / max(length, 1e-6), exact[step, 1] / max(length, 1e-6), (0, 255, 0), pixels_per_cm = 90.0)
        cv2.putText(picture, f"step {step}: asked {length:.2f} cm at {numpy.degrees(numpy.arctan2(exact[step, 1], exact[step, 0])):+.0f} deg, twist {exact[step, 2]:+.1f}",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(picture)
    columns = 4
    while len(tiles) % columns:
        tiles.append(numpy.zeros_like(tiles[0]))
    cv2.imwrite(str(common.ANALYSIS_FOLDER / "s04_run6_pictures.jpg"), numpy.vstack([numpy.hstack(tiles[start:start + columns]) for start in range(0, len(tiles), columns)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])

    # ---- the tip's path on the table, with what the model asked for at every step ----
    figure, axes = pyplot.subplots(1, 2, figsize = (15, 6.5))
    tip_all = steps[["tip_x", "tip_y"]].to_numpy() * 100.0
    heading_all = numpy.radians(steps["picture_heading_deg"].to_numpy())
    for axis, (lower, upper, title) in zip(axes, ((0, len(steps) - 1, "the whole run"), (first - 6, last + 3, f"steps {first - 6} to {last + 3}: the back and forth"))):
        part = slice(lower, upper + 1)
        axis.plot(tip_all[part, 0], tip_all[part, 1], "-", color = "0.6", linewidth = 1.0, zorder = 1)
        for step in range(lower, upper + 1):
            asked_table = numpy.array([exact[step, 0] * numpy.cos(heading_all[step]) - exact[step, 1] * numpy.sin(heading_all[step]),
                                       exact[step, 0] * numpy.sin(heading_all[step]) + exact[step, 1] * numpy.cos(heading_all[step])])
            colour = "tab:red" if numpy.hypot(*asked_table) < settings["short_step_cm"] else "tab:blue"
            axis.annotate("", xy = tip_all[step] + asked_table, xytext = tip_all[step], arrowprops = {"arrowstyle": "->", "color": colour, "linewidth": 1.3}, zorder = 3)
            if step % 5 == 0 or lower > 0:
                axis.text(tip_all[step, 0], tip_all[step, 1], str(step), fontsize = 6, zorder = 4)
        axis.set_aspect("equal")
        axis.grid(alpha = 0.3)
        axis.set_xlabel("x from the robot's base (cm)")
        axis.set_ylabel("y (cm), the robot's left is up")
        axis.set_title(f"run 6, {title}\ngrey: the tip's path (1 cm per move); arrows: the step the model asked for, at its own length\n(red: under 0.3 cm)", fontsize = 9)
    figure.tight_layout()
    figure.savefig(common.ANALYSIS_FOLDER / "s04_run6_path.png", dpi = 110)
    pyplot.close(figure)
    report("")
    report("figures: s04_run6_pictures.jpg (the run's own pictures at the first bend and at the back and forth, with the asked direction), "
           "s04_run6_path.png (the tip's path on the table with the asked steps at their own length)")
    report("figure: s04_run6_neighbours.jpg (left column: the run's picture with the direction the model asked for; to its right its nearest demonstration pictures, "
           "green arrow = the demonstrated move at 60 px per cm, orange arrow = the mean move over the next 3 steps; the text gives the sheet, the end the episode started from, "
           "the puck's place along the line, the kind of place, the labelled move (forward, left in cm) and the twist)")



if __name__ == "__main__":
    main()
