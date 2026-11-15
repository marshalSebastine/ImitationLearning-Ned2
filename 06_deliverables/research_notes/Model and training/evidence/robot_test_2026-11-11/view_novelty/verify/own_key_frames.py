"""
What did the demonstrations do at the places where the robot runs went wrong, and what did the model ask there?

The verifier's own check of the analyst's key-frame claims. For every picture of the first part of each failed run the
5 nearest training pictures are found in model E's own descriptors (all 18,032 pictures it trained on, any sheet) and
their labels are read from the training table as model E saw them (after the audit's cleaning):
    the step of the tip in the picture's frame (length in cm, direction in degrees, 0 = towards the picture's top,
    positive = to its left) and the forearm twist in degrees.
Beside them: what model E asked on the robot picture (from steps.csv), and what the robot did (always 1.00 cm; the
twist it used).

A second table follows the demonstrations themselves through the first bend of sheet 03 (the place where runs 5, 9
and 10 failed): how far the tip travelled and how far the forearm turned while it turned, against the robot.

Run from this folder, after extract_own.py model_e (about half a minute):
    D:/AI_Project/.venv/Scripts/python.exe own_key_frames.py
Output: own_key_frames.out.txt, own_key_frames_neighbours.csv, figures/own_neighbours_run_NN_step_SSS.png
"""

import cv2
import numpy
import pandas

from own_novelty import load_table, nearest, trained_rows
from verify_common import (RUN_SHEETS, VERIFY_FOLDER, VERIFY_PARAMETERS, Report, labelled_montage, read_picture, run_log, run_picture_path,
                           training_frames)

KEY_FRAME_PARAMETERS = {
    "neighbour_count": 5,
    "steps": {3: range(0, 8), 5: range(0, 20), 6: range(3, 17), 8: range(4, 22), 9: range(0, 19), 10: range(0, 14)},
    "montages": [(5, 3), (9, 3), (10, 3), (6, 8), (6, 9), (8, 9), (8, 14), (3, 0)],
    "bend_twist_deg": 4.0
}



def mean_direction(forward: numpy.ndarray, left: numpy.ndarray) -> tuple:
    """Return the length (cm) and direction (degrees) of the mean of several steps."""

    return float(numpy.hypot(forward.mean(), left.mean())), float(numpy.degrees(numpy.arctan2(left.mean(), forward.mean())))



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_key_frames.out.txt")
    settings = KEY_FRAME_PARAMETERS
    count = settings["neighbour_count"]
    frames = training_frames()
    trained = trained_rows().set_index("frame_index")
    reference = frames[frames["frame_index"].isin(trained.index)].reset_index(drop = True)
    labels = trained.loc[reference["frame_index"], ["tip_forward_m", "tip_left_m", "forearm_change_deg", "action"]].reset_index(drop = True)
    reference = pandas.concat([reference[["frame_index", "episode_id", "sheet", "plan_number", "start_kind", "start_end", "row_in_episode", "steps_to_end",
                                          "joint_4", "frame_path"]], labels], axis = 1)
    reference["forward_cm"], reference["left_cm"] = 100 * reference["tip_forward_m"].fillna(0.0), 100 * reference["tip_left_m"].fillna(0.0)
    training, robot = load_table("model_e_training"), load_table("model_e_robot")
    reference_grid = training["grid"][reference["frame_index"].to_numpy()]

    # how well model E reproduces its own training labels (the claim "model E asks for shorter steps")
    is_moving = (reference["action"] != "stop").to_numpy()
    asked = training["exact"][reference["frame_index"].to_numpy()]
    label_length = numpy.hypot(reference["forward_cm"], reference["left_cm"]).to_numpy()
    asked_length = numpy.hypot(asked[:, 0], asked[:, 1])
    report.say(f"MODEL E ON ITS OWN {len(reference)} TRAINING PICTURES ({int(is_moving.sum())} moving rows)")
    report.say(f"   demonstrated step length: median {numpy.median(label_length[is_moving]):.2f} cm, mean {label_length[is_moving].mean():.2f}; "
               f"model E's answer: median {numpy.median(asked_length[is_moving]):.2f} cm, mean {asked_length[is_moving].mean():.2f}")
    report.say(f"   median absolute difference: forward {numpy.median(numpy.abs(asked[is_moving, 0] - reference['forward_cm'].to_numpy()[is_moving])):.2f} cm, "
               f"left {numpy.median(numpy.abs(asked[is_moving, 1] - reference['left_cm'].to_numpy()[is_moving])):.2f} cm, "
               f"twist {numpy.median(numpy.abs(asked[is_moving, 2] - reference['forearm_change_deg'].fillna(0.0).to_numpy()[is_moving])):.2f} degrees")
    is_twisting = is_moving & (reference["forearm_change_deg"].abs().fillna(0.0).to_numpy() >= 5.0)
    report.say(f"   rows with a demonstrated twist of 5 degrees or more ({int(is_twisting.sum())}): demonstrated length median "
               f"{numpy.median(label_length[is_twisting]):.2f} cm, model E's {numpy.median(asked_length[is_twisting]):.2f} cm; demonstrated twist median "
               f"{numpy.median(numpy.abs(reference['forearm_change_deg'].to_numpy()[is_twisting])):.1f}, model E's {numpy.median(numpy.abs(asked[is_twisting, 2])):.1f} degrees")
    short = is_moving & (label_length < 0.5)
    report.say(f"   share of demonstrated steps under 0.5 cm: {100 * short.sum() / is_moving.sum():.1f} %; under 0.2 cm: {100 * (is_moving & (label_length < 0.2)).sum() / is_moving.sum():.1f} %")

    rows = []
    for run_number, steps in settings["steps"].items():
        log = run_log(run_number)
        is_run = robot["run"] == run_number
        distance, index = nearest(robot["grid"][is_run], reference_grid, count = count)
        report.say("")
        report.say(f"RUN {run_number} ({RUN_SHEETS[run_number]}): the 5 nearest training pictures of each robot picture (model E's descriptors)")
        report.say("step | model asked: cm  deg  twist | robot did: cm  twist | joint 4 | neighbours: same sheet  plans (row)                                   joint 4      | they did: mean step cm  deg | median length  median twist | each: cm@deg")
        for step in steps:
            if step >= len(log):
                continue
            neighbours = reference.iloc[index[step]]
            row = log.iloc[step]
            asked_cm = float(numpy.hypot(row["model_tip_forward_cm"], row["model_tip_left_cm"]))
            done_cm = 100 * float(numpy.hypot(row["tip_next_x"] - row["tip_x"], row["tip_next_y"] - row["tip_y"])) if numpy.isfinite(row["tip_next_x"]) else numpy.nan
            mean_cm, mean_deg = mean_direction(neighbours["forward_cm"].to_numpy(), neighbours["left_cm"].to_numpy())
            lengths = numpy.hypot(neighbours["forward_cm"], neighbours["left_cm"]).to_numpy()
            angles = numpy.degrees(numpy.arctan2(neighbours["left_cm"], neighbours["forward_cm"])).to_numpy()
            same_sheet = float((neighbours["sheet"] == RUN_SHEETS[run_number]).mean())
            places = " ".join(f"{plan}({int(row_number)})" for plan, row_number in zip(neighbours["plan_number"], neighbours["row_in_episode"]))
            joint_4 = numpy.degrees(neighbours["joint_4"].to_numpy())
            report.say(f"{step:4d} | {asked_cm:15.2f} {row['model_angle_deg']:+5.0f} {row['model_twist_deg']:+6.1f} | {done_cm:13.2f} {row['twist_used_deg']:+6.1f} | {numpy.degrees(row['joint_4']):+7.1f} | "
                       f"{100 * same_sheet:20.0f} %  {places:45s} {joint_4.min():+5.0f} to {joint_4.max():+4.0f} | {mean_cm:21.2f} {mean_deg:+5.0f} | {numpy.median(lengths):13.2f} "
                       f"{neighbours['forearm_change_deg'].median():+13.1f} | " + " ".join(f"{length:.2f}@{angle:+.0f}" for length, angle in zip(lengths, angles)))
            rows.append({"run": run_number, "step": step, "asked_cm": asked_cm, "asked_deg": row["model_angle_deg"], "asked_twist_deg": row["model_twist_deg"],
                         "done_cm": done_cm, "twist_used_deg": row["twist_used_deg"], "same_sheet_share": same_sheet, "neighbour_mean_cm": mean_cm,
                         "neighbour_mean_deg": mean_deg, "neighbour_median_cm": float(numpy.median(lengths)),
                         "neighbour_median_twist_deg": float(neighbours["forearm_change_deg"].median()), "nearest_distance": float(distance[step, 0]),
                         "neighbour_plans": places})
    pandas.DataFrame(rows).round(3).to_csv(VERIFY_FOLDER / "own_key_frames_neighbours.csv", index = False)

    # montages: the robot picture and its 5 nearest training pictures, for the eye
    for run_number, step in settings["montages"]:
        is_run = robot["run"] == run_number
        _, index = nearest(robot["grid"][is_run][step:step + 1], reference_grid, count = count)
        pictures = [read_picture(run_picture_path(run_number, step))]
        names = [f"ROBOT run {run_number} step {step}"]
        for _, neighbour in reference.iloc[index[0]].iterrows():
            pictures.append(read_picture(VERIFY_PARAMETERS["raw_folder"] / neighbour["frame_path"]))
            length = float(numpy.hypot(neighbour["forward_cm"], neighbour["left_cm"]))
            angle = float(numpy.degrees(numpy.arctan2(neighbour["left_cm"], neighbour["forward_cm"])))
            names.append(f"{neighbour['plan_number']} row {int(neighbour['row_in_episode'])}: {length:.2f} cm at {angle:+.0f}, twist {neighbour['forearm_change_deg']:+.1f}")
        montage = labelled_montage(pictures, names, columns = 3, cell_width = 480)
        cv2.imencode(".png", montage)[1].tofile(str(VERIFY_FOLDER / "figures" / f"own_neighbours_run_{run_number:02d}_step_{step:03d}.png"))

    # the demonstrations through the first bend of sheet 03 (standard starts from end A): tip travel and forearm turn
    report.say("")
    report.say("THE FIRST BEND OF SHEET 03 IN THE DEMONSTRATIONS (standard start, end A; all rows of frames.csv, cleaned labels where trained)")
    report.say("The 'turn' is the stretch of rows, within the first 25, from the first to the last row whose forearm twist is 4 degrees or more in size.")
    report.say("plan    rows of the turn  forearm turned deg  tip travelled cm  cm per 10 degrees  steps under 0.5 cm in the turn")
    sheet = frames[(frames["sheet"] == "path_03") & (frames["start_kind"] == "STD") & (frames["start_end"] == "A")]
    summary = []
    for plan, episode in sheet.groupby("plan_number"):
        episode = episode.sort_values("row_in_episode").head(25)
        twist = episode["forearm_change_deg"].fillna(0.0).to_numpy()
        length = 100 * numpy.hypot(episode["tip_forward_m"].fillna(0.0), episode["tip_left_m"].fillna(0.0)).to_numpy()
        turning = numpy.flatnonzero(numpy.abs(twist) >= settings["bend_twist_deg"])
        if len(turning) == 0:
            continue
        first, last = turning[0], turning[-1]
        turned, travelled = float(twist[first:last + 1].sum()), float(length[first:last + 1].sum())
        summary.append((turned, travelled, last - first + 1))
        report.say(f"{plan}   {int(episode['row_in_episode'].iloc[first]):3d} to {int(episode['row_in_episode'].iloc[last]):3d}        {turned:+18.1f}  {travelled:16.2f}  "
                   f"{10 * travelled / max(abs(turned), 1e-9):17.2f}  {int((length[first:last + 1] < 0.5).sum()):3d} of {last - first + 1}")
    if summary:
        turned, travelled, steps = numpy.array(summary).T
        report.say(f"median over {len(summary)} demonstrations: forearm turned {numpy.median(turned):+.1f} degrees over {numpy.median(steps):.0f} rows while the tip travelled "
                   f"{numpy.median(travelled):.2f} cm ({numpy.median(10 * travelled / numpy.abs(turned)):.2f} cm per 10 degrees)")
    report.say("the robot at the same bend (from the first to the last step whose twist used was 4 degrees or more in size, within the first 25 steps):")
    for run_number in (5, 9, 10):
        log = run_log(run_number).head(25)
        twist = log["twist_used_deg"].fillna(0.0).to_numpy()
        length = 100 * numpy.hypot(log["tip_next_x"] - log["tip_x"], log["tip_next_y"] - log["tip_y"]).fillna(0.0).to_numpy()
        asked = numpy.hypot(log["model_tip_forward_cm"], log["model_tip_left_cm"]).to_numpy()
        turning = numpy.flatnonzero(numpy.abs(twist) >= settings["bend_twist_deg"])
        first, last = turning[0], turning[-1]
        report.say(f"run {run_number:2d}: steps {first} to {last}: forearm turned {twist[first:last + 1].sum():+.1f} degrees, tip travelled {length[first:last + 1].sum():.2f} cm "
                   f"({10 * length[first:last + 1].sum() / abs(twist[first:last + 1].sum()):.2f} cm per 10 degrees); the model had asked for {asked[first:last + 1].sum():.2f} cm "
                   f"({10 * asked[first:last + 1].sum() / abs(twist[first:last + 1].sum()):.2f} cm per 10 degrees)")
    report.say("")
    report.say("written: own_key_frames_neighbours.csv, figures/own_neighbours_run_NN_step_SSS.png")
    report.save()



if __name__ == "__main__":
    main()
