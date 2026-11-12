"""
Smaller claims of the view-novelty answer, recomputed by the verifier:

    A  the forearm twist, wrist and hand of the robot runs against the training pictures (joint ranges);
    B  the proposed stop rule "distance above 0.18 on 4 pictures in a row" (model E, distance to all training pictures),
       against when the puck really left the line (puck_on_line.py);
    C  the first pictures of plans 01-17 (labelled OL1.5) and 01-18 (labelled OR1.5), for the eye.

Run from this folder, after extract_own.py model_e (about half a minute):
    D:/AI_Project/.venv/Scripts/python.exe own_misc_checks.py
Output: own_misc_checks.out.txt, figures/own_sheet_01_offset_labels.png
"""

import cv2
import numpy

from own_novelty import OWN_NOVELTY_PARAMETERS, load_table, nearest, trained_rows
from verify_common import RUN_FOLDERS, VERIFY_FOLDER, VERIFY_PARAMETERS, Report, labelled_montage, read_picture, run_log, training_frames

MISC_PARAMETERS = {
    "rule_distance": 0.18,
    "rule_pictures_in_a_row": 4
}



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_misc_checks.out.txt")
    frames = training_frames()
    trained = trained_rows()
    reference = frames[frames["frame_index"].isin(trained["frame_index"])].reset_index(drop = True)

    report.say("A. JOINTS: the robot runs against the training pictures (degrees)")
    for name, column in (("forearm twist (joint 4)", "joint_4"), ("wrist (joint 5)", "joint_5"), ("hand (joint 6)", "joint_6")):
        training = numpy.degrees(reference[column].to_numpy())
        robot = numpy.concatenate([numpy.degrees(run_log(run_number)[column].to_numpy()) for run_number in RUN_FOLDERS])
        outside = int(((robot < numpy.quantile(training, 0.01)) | (robot > numpy.quantile(training, 0.99))).sum())
        report.say(f"{name:24s} training: least {training.min():+7.1f}, 1 % {numpy.quantile(training, 0.01):+7.1f}, median {numpy.median(training):+6.1f}, 99 % {numpy.quantile(training, 0.99):+7.1f}, "
                   f"most {training.max():+7.1f} | robot: least {robot.min():+7.1f}, most {robot.max():+7.1f}; outside the training pictures' 1 % to 99 %: {outside} of {len(robot)}")
    wrist, hand = numpy.degrees(reference["joint_5"]), numpy.degrees(reference["joint_6"])
    near = ((wrist - 13.4).abs() <= 3.0) & ((hand + 6.1).abs() <= 3.0)
    report.say(f"training pictures with wrist and hand within 3 degrees of the held 13.4 and -6.1: {100 * near.mean():.1f} % of {len(reference)}")
    report.say(f"wrist of the training pictures: 5 % {wrist.quantile(0.05):+.1f}, 95 % {wrist.quantile(0.95):+.1f}; hand: 5 % {hand.quantile(0.05):+.1f}, 95 % {hand.quantile(0.95):+.1f}")

    report.say("")
    settings = MISC_PARAMETERS
    report.say(f"B. THE PROPOSED STOP RULE: model E's distance to all its training pictures above {settings['rule_distance']} on {settings['rule_pictures_in_a_row']} pictures in a row")
    training, robot = load_table("model_e_training"), load_table("model_e_robot")
    distance, _ = nearest(robot["grid"], training["grid"][reference["frame_index"].to_numpy()], count = 5)
    mean_distance = distance.mean(axis = 1)
    report.say("run  outcome   rule fires at step  puck left the line at step (verifier)  steps between  largest distance of the run")
    for run_number in RUN_FOLDERS:
        values = mean_distance[robot["run"] == run_number]
        above = values > settings["rule_distance"]
        fired = [step + settings["rule_pictures_in_a_row"] - 1 for step in range(len(above) - settings["rule_pictures_in_a_row"] + 1)
                 if above[step:step + settings["rule_pictures_in_a_row"]].all()]
        steps = OWN_NOVELTY_PARAMETERS["verifier_steps"].get(run_number)
        fired_text = str(fired[0]) if fired else "never"
        off_text = str(steps[1]) if steps else "-"
        between = str(fired[0] - steps[1]) if (fired and steps) else "-"
        report.say(f"{run_number:3d}  {'failed' if steps else 'reached':8s}  {fired_text:>18s}  {off_text:>37s}  {between:>13s}  {values.max():27.3f}")

    pictures, names = [], []
    for plan in ("01-17", "01-18"):
        episode = frames[frames["plan_number"] == plan].sort_values("row_in_episode")
        for row_number in (0, 3):
            row = episode.iloc[row_number]
            pictures.append(read_picture(VERIFY_PARAMETERS["raw_folder"] / row["frame_path"]))
            names.append(f"plan {plan}, labelled {row['start_kind']}, offset {row['offset_lateral_cm']:+.1f} cm, row {int(row['row_in_episode'])}")
    cv2.imencode(".png", labelled_montage(pictures, names, columns = 2, cell_width = 560))[1].tofile(str(VERIFY_FOLDER / "figures" / "own_sheet_01_offset_labels.png"))
    report.say("")
    report.say("C. written: figures/own_sheet_01_offset_labels.png (first pictures of plans 01-17 and 01-18)")
    report.save()



if __name__ == "__main__":
    main()
