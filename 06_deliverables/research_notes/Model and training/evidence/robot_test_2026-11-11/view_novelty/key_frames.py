"""
The nearest demonstration pictures of the robot's key pictures, and what the demonstration did there against what the
model asked.

For every robot picture the 5 nearest training pictures in model E's descriptors are taken (from novelty_model_e.py).
Their labels, as model E was trained on them, say what the demonstrators did in the most similar views: the tip's step
in the picture's frame (forward and left, cm) and the forearm twist. Two questions are answered:
    - do the neighbours show the same place on the same sheet (was the view recognised)?
    - did the model ask for what the demonstrations did there (then the fault is not in reading the picture)?

Montages are drawn for the start of every run and for the pictures around each failure: the robot picture on the
left with the model's answer, then its 5 nearest demonstration pictures with their labels.

Run from this folder, after novelty_model_e.py:
    D:/AI_Project/.venv/Scripts/python.exe key_frames.py
Output: key_frames.out.txt, key_frames_neighbour_labels.csv, figures/key_frames_run_NN.png, figures/key_frames_starts.png
"""

import math
from typing import Dict, List, Optional

import cv2
import numpy
import pandas

from common import (EVIDENCE_FOLDER, FAILURE_STEPS, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, cache_folder, load_descriptors,
                    run_picture_paths, run_steps, save_montage, trained_frames)

KEY_FRAME_PARAMETERS = {
    "neighbour_count": 5,
    # run number -> the steps drawn; the start of every run is drawn in its own figure
    "key_steps": {3: [0, 1, 2, 3, 4], 5: [0, 1, 2, 3, 4], 6: [0, 8, 11, 13, 14, 15, 19], 8: [0, 4, 6, 7, 9, 10],
                  9: [0, 1, 2, 3, 4], 10: [0, 1, 2, 3, 4], 1: [0, 10, 20], 4: [0, 22, 27, 33]},
    "tile_size": [320, 240],
    "caption_height": 46,
    "arrow_length_px": 70
}



def tile_with_caption(
        picture_bgr: numpy.ndarray,
        lines: List[str],
        angle_deg: Optional[float],
        parameters: Dict
) -> numpy.ndarray:
    """
    Shrink a picture to a tile, draw a step direction on it and write a caption beneath.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The camera picture.
    lines : list of str
        Up to three caption lines.
    angle_deg : float or None
        The step's direction in the picture (0 = towards its top, positive = to its left); no arrow when None.
    parameters : dict
        ``KEY_FRAME_PARAMETERS``.

    Returns
    -------
    numpy.ndarray
        The tile with its caption band, BGR.
    """

    width, height = parameters["tile_size"]
    tile = cv2.resize(picture_bgr, (width, height), interpolation = cv2.INTER_AREA)
    if angle_deg is not None and numpy.isfinite(angle_deg):
        angle = math.radians(angle_deg)
        start = (width // 2, height - 8)
        end = (int(start[0] - parameters["arrow_length_px"] * math.sin(angle)), int(start[1] - parameters["arrow_length_px"] * math.cos(angle)))
        cv2.arrowedLine(tile, start, end, (0, 220, 0), 2, tipLength = 0.25)
    band = numpy.full((parameters["caption_height"], width, 3), 255, dtype = numpy.uint8)
    for number, line in enumerate(lines[:3]):
        cv2.putText(band, line, (4, 13 + 14 * number), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    return numpy.vstack([tile, band])



def neighbour_label_table(log: OutputLog) -> Dict:
    """
    Join every robot picture with the labels of its nearest training pictures.

    Returns
    -------
    dict
        "table" (one row per robot picture: the model's answer, the neighbours' mean step and twist, how alike the
        two are), "neighbour_rows" (per robot picture, the trained rows of its neighbours) and "trained" (the
        training rows with model E's own answers to them).
    """

    settings = KEY_FRAME_PARAMETERS
    count = settings["neighbour_count"]
    with numpy.load(cache_folder() / "model_e_robot_neighbours.npz") as saved:
        reference_frame_index = saved["reference_frame_index"]
        neighbour_index = saved["grid_index"]
        neighbour_distance = saved["grid_distance"]
    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"]).set_index("frame_index")
    training = load_descriptors("model_e_training")
    robot = load_descriptors("model_e_robot")

    # how closely model E gives back its own training labels: the size of "the model asks what the demonstration did"
    ordered = trained.loc[reference_frame_index]
    label = numpy.stack([100 * ordered["tip_forward_m"].fillna(0.0), 100 * ordered["tip_left_m"].fillna(0.0), ordered["forearm_change_deg"].fillna(0.0)], axis = 1)
    answer = training["exact"][reference_frame_index][:, :3]
    is_moving = (ordered["action"] != "stop").to_numpy()
    log.say("Model E on its own training pictures (the moving rows): how far its answer is from the label")
    log.say(f"  forward: median absolute difference {numpy.median(numpy.abs(answer[is_moving, 0] - label[is_moving, 0])):.2f} cm; "
            f"left: {numpy.median(numpy.abs(answer[is_moving, 1] - label[is_moving, 1])):.2f} cm; twist: {numpy.median(numpy.abs(answer[is_moving, 2] - label[is_moving, 2])):.2f} degrees")
    log.say(f"  step length: label median {numpy.median(numpy.hypot(label[is_moving, 0], label[is_moving, 1])):.2f} cm, "
            f"answer median {numpy.median(numpy.hypot(answer[is_moving, 0], answer[is_moving, 1])):.2f} cm")

    rows = []
    for position in range(len(robot["run"])):
        frame_index = reference_frame_index[neighbour_index[position, :count]]
        neighbours = trained.loc[frame_index]
        forward = 100 * neighbours["tip_forward_m"].fillna(0.0).to_numpy()
        left = 100 * neighbours["tip_left_m"].fillna(0.0).to_numpy()
        twist = neighbours["forearm_change_deg"].fillna(0.0).to_numpy()
        model_forward, model_left, model_twist = robot["exact"][position, :3]
        mean_forward, mean_left = forward.mean(), left.mean()
        difference = math.degrees(math.atan2(model_left, model_forward)) - math.degrees(math.atan2(mean_left, mean_forward))
        rows.append({
            "run": int(robot["run"][position]), "step": int(robot["step"][position]),
            "model_forward_cm": model_forward, "model_left_cm": model_left, "model_length_cm": math.hypot(model_forward, model_left),
            "model_angle_deg": math.degrees(math.atan2(model_left, model_forward)), "model_twist_deg": model_twist,
            "neighbour_forward_cm": mean_forward, "neighbour_left_cm": mean_left,
            "neighbour_mean_step_length_cm": float(numpy.hypot(forward, left).mean()),
            "neighbour_vector_length_cm": math.hypot(mean_forward, mean_left),
            "neighbour_angle_deg": math.degrees(math.atan2(mean_left, mean_forward)),
            "neighbour_twist_deg": twist.mean(),
            "angle_difference_deg": abs((difference + 180.0) % 360.0 - 180.0),
            "vector_difference_cm": math.hypot(model_forward - mean_forward, model_left - mean_left),
            "neighbour_stop_share": float((neighbours["action"] == "stop").mean()),
            "neighbour_same_sheet_share": float((neighbours["sheet"] == ROBOT_RUNS[int(robot["run"][position])]["sheet"]).mean()),
            "neighbour_distance": float(neighbour_distance[position, :count].mean()),
            "neighbour_joint_4_deg": float(numpy.degrees(neighbours["joint_4"]).mean()),
            "neighbour_episodes": int(neighbours["episode_id"].nunique()),
            "neighbour_start_kinds": "/".join(sorted(set(neighbours["start_kind"].fillna("?").astype(str)))),
            "neighbour_plans": "/".join(sorted(set(neighbours["plan_number"].fillna("?").astype(str)))),
            "neighbour_frame_index": " ".join(str(value) for value in frame_index)
        })
    return {"table": pandas.DataFrame(rows), "trained": trained, "reference_frame_index": reference_frame_index,
            "neighbour_index": neighbour_index, "neighbour_distance": neighbour_distance}



def draw_key_frames(
        steps_by_run: Dict[int, List[int]],
        joined: Dict,
        figure_path
) -> None:
    """Draw one row per key picture: the robot picture with the model's answer, then its nearest demonstration pictures."""

    settings = KEY_FRAME_PARAMETERS
    count = settings["neighbour_count"]
    robot = load_descriptors("model_e_robot")
    raw_folder = VIEW_NOVELTY_PARAMETERS["raw_folder"]
    trained = joined["trained"]
    strips = []
    for run_number, key_steps in steps_by_run.items():
        logged = run_steps(run_number)
        paths = run_picture_paths(run_number)
        for step in key_steps:
            if step >= len(paths):
                continue
            position = int(numpy.flatnonzero((robot["run"] == run_number) & (robot["step"] == step))[0])
            forward, left, twist = robot["exact"][position, :3]
            angle_logged = logged.iloc[step]["model_angle_deg"]
            lines = [f"RUN {run_number} step {step} (model {ROBOT_RUNS[run_number]['model']} drove; model E's answer below)",
                     f"model E: {math.hypot(forward, left):.2f} cm at {math.degrees(math.atan2(left, forward)):+.0f} deg, twist {twist:+.1f}",
                     f"joint 4 {math.degrees(logged.iloc[step]['joint_4']):+.0f} deg; arrow = the direction driven"]
            tiles = [tile_with_caption(cv2.imread(str(paths[step]), cv2.IMREAD_COLOR), lines, angle_logged, settings)]
            for neighbour in range(count):
                frame_index = joined["reference_frame_index"][joined["neighbour_index"][position, neighbour]]
                row = trained.loc[frame_index]
                picture = cv2.imread(str(raw_folder / row["frame_path"]), cv2.IMREAD_COLOR)
                step_forward, step_left = 100 * (row["tip_forward_m"] or 0.0), 100 * (row["tip_left_m"] or 0.0)
                is_stop = row["action"] == "stop"
                demonstrated = "STOP" if is_stop else f"{math.hypot(step_forward, step_left):.2f} cm at {math.degrees(math.atan2(step_left, step_forward)):+.0f} deg, twist {row['forearm_change_deg']:+.1f}"
                lines = [f"{row['sheet'][-2:]} plan {row['plan_number']} {row['start_kind']} end {row['start_end']}, row {int(row['row_in_episode'])} (+{int(row['steps_to_end'])} to end)",
                         f"demo: {demonstrated}",
                         f"joint 4 {math.degrees(row['joint_4']):+.0f} deg; distance {joined['neighbour_distance'][position, neighbour]:.3f}"]
                angle = None if is_stop else math.degrees(math.atan2(step_left, step_forward))
                tiles.append(tile_with_caption(picture, lines, angle, settings))
            strips.append(numpy.hstack(tiles))
    save_montage(figure_path, numpy.vstack(strips))



def main() -> None:

    log = OutputLog(EVIDENCE_FOLDER / "key_frames.out.txt")
    joined = neighbour_label_table(log)
    table = joined["table"]
    table.round(4).to_csv(EVIDENCE_FOLDER / "key_frames_neighbour_labels.csv", index = False)

    log.say("")
    log.say("KEY PICTURES: what model E asked against what the demonstrations did in the 5 most similar training pictures")
    log.say("(vector = the mean of the 5 demonstrated steps; step = their mean length; a short vector with a long step means the neighbours disagree on the direction)")
    log.say("run step | model E: cm at deg, twist | neighbours: vector cm at deg, mean step cm, twist | angle apart | same sheet | plans (kinds of start) | their joint 4 | robot joint 4")
    for run_number, key_steps in KEY_FRAME_PARAMETERS["key_steps"].items():
        logged = run_steps(run_number)
        for step in key_steps:
            row = table[(table["run"] == run_number) & (table["step"] == step)]
            if row.empty:
                continue
            row = row.iloc[0]
            log.say(f"{run_number:3d} {step:4d} | {row['model_length_cm']:.2f} at {row['model_angle_deg']:+5.0f}, {row['model_twist_deg']:+5.1f} | "
                    f"{row['neighbour_vector_length_cm']:.2f} at {row['neighbour_angle_deg']:+5.0f}, {row['neighbour_mean_step_length_cm']:.2f}, {row['neighbour_twist_deg']:+5.1f} | "
                    f"{row['angle_difference_deg']:5.0f} | {100 * row['neighbour_same_sheet_share']:3.0f} % | {row['neighbour_plans']} ({row['neighbour_start_kinds']}) | "
                    f"{row['neighbour_joint_4_deg']:+5.0f} | {math.degrees(logged.iloc[step]['joint_4']):+5.0f}")

    log.say("")
    log.say("ALL PICTURES BY PHASE: model E's answer against the demonstrated steps of the 5 nearest training pictures")
    log.say("run  phase                         pictures  model step cm  neighbours' step cm  vector apart cm (median)  twist: model, neighbours (median absolute)  same sheet")
    for run_number in ROBOT_RUNS:
        rows = table[table["run"] == run_number]
        if run_number in FAILURE_STEPS:
            wrong, off_line = FAILURE_STEPS[run_number]["first_wrong_decision"], FAILURE_STEPS[run_number]["puck_off_line"]
            phases = {"seen before it went wrong": rows["step"] <= wrong, "between": (rows["step"] > wrong) & (rows["step"] < off_line),
                      "after the puck left the line": rows["step"] >= off_line}
        else:
            phases = {"whole run": rows["step"] >= 0}
        for phase, is_phase in phases.items():
            part = rows[is_phase]
            if part.empty:
                continue
            log.say(f"{run_number:3d}  {phase:28s}  {len(part):8d}  {part['model_length_cm'].median():13.2f}  {part['neighbour_mean_step_length_cm'].median():19.2f}  "
                    f"{part['vector_difference_cm'].median():24.2f}  {part['model_twist_deg'].abs().median():18.1f}, {part['neighbour_twist_deg'].abs().median():.1f}"
                    f"  {100 * part['neighbour_same_sheet_share'].mean():16.0f} %")

    figure_folder = EVIDENCE_FOLDER / "figures"
    figure_folder.mkdir(exist_ok = True)
    draw_key_frames({run_number: [0] for run_number in ROBOT_RUNS}, joined, figure_folder / "key_frames_starts.png")
    for run_number, key_steps in KEY_FRAME_PARAMETERS["key_steps"].items():
        draw_key_frames({run_number: key_steps}, joined, figure_folder / f"key_frames_run_{run_number:02d}.png")
    log.say("")
    log.say("written: key_frames_neighbour_labels.csv, figures/key_frames_starts.png, figures/key_frames_run_NN.png")
    log.save()



if __name__ == "__main__":
    main()
