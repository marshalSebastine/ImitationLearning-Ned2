"""
Two smaller recomputations of the verifier:

    A  the camera's shift undone on run 1 (model A drove it): every picture slid 17 px to the left, model A's answer as
       seen and slid; the analyst gives a median turn of +2.5 degrees and builds a 0.96 cm drift on it;
    B  the shoulder and elbow (joints 2 and 3) and the tip's reach in the robot runs against the training pictures: the
       camera sits on the forearm, so its height and tilt follow from joints 2 and 3; with the wrist held these two
       joints must do what the wrist did in the demonstrations;
    C  how much royal blue the training pictures show (large blue objects at the border, as in own_setup_checks.py),
       from a sample of the raw pictures.

Run from this folder (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe own_model_a_and_joints.py
Output: own_model_a_and_joints.out.txt
"""

import cv2
import numpy
import torch

from own_novelty import trained_rows
from own_setup_checks import decide_on_picture, large_blue_object
from verify_common import RUN_FOLDERS, VERIFY_FOLDER, VERIFY_PARAMETERS, Report, read_picture, run_log, run_picture_path, training_frames

SLIDE_PX = 17



def main() -> None:

    from model_mode import ModelPolicy

    report = Report(VERIFY_FOLDER / "own_model_a_and_joints.out.txt")
    folder = VERIFY_PARAMETERS["trained_models_folder"] / "2026-11-10_resnet18_model_1_all_sheets"
    policy = ModelPolicy("resnet18_model_1_all_sheets", folder / "checkpoints" / "all_sheets.pt", folder / "metrics.json")
    torch.set_num_threads(VERIFY_PARAMETERS["threads"])
    log = run_log(1)
    report.say("A. RUN 1 (model A): the answer as seen and with the picture slid 17 px to the left")
    turns, checks = [], []
    for step in range(len(log)):
        picture = read_picture(run_picture_path(1, step))
        seen = policy.decide(run_picture_path(1, step).read_bytes())
        checks.append(abs(seen["angle_deg"] - log["model_angle_deg"].iloc[step]))
        slid = decide_on_picture(policy, numpy.concatenate([picture[:, SLIDE_PX:], numpy.repeat(picture[:, -1:], SLIDE_PX, axis = 1)], axis = 1))
        if numpy.hypot(seen["tip_forward_cm"], seen["tip_left_cm"]) >= 0.3:
            turns.append((slid["angle_deg"] - seen["angle_deg"] + 180) % 360 - 180)
    turns = numpy.array(turns)
    report.say(f"model A's answers recomputed here match the log of run 1 within {max(checks):.4f} degrees of direction on {len(checks)} pictures")
    report.say(f"turn of the asked direction when the shift is undone ({len(turns)} pictures): median {numpy.median(turns):+.1f} degrees, middle half {numpy.quantile(turns, 0.25):+.1f} to "
               f"{numpy.quantile(turns, 0.75):+.1f}; positive = to the left")
    report.say(f"calculated from it: {22 * numpy.tan(numpy.radians(abs(numpy.median(turns)))):.2f} cm sideways over 22 cm if nothing corrects it (the tip's own y went from "
               f"{100 * log['tip_y'].iloc[0]:+.2f} to {100 * log['tip_y'].iloc[22]:+.2f} cm between steps 0 and 22)")

    report.say("")
    report.say("B. SHOULDER, ELBOW AND REACH: the robot runs against the training pictures model E trained on")
    frames = training_frames()
    reference = frames[frames["frame_index"].isin(trained_rows()["frame_index"])]
    logs = {run_number: run_log(run_number) for run_number in RUN_FOLDERS}
    for name, column, scale in (("shoulder (joint 2), degrees", "joint_2", numpy.degrees(1.0)), ("elbow (joint 3), degrees", "joint_3", numpy.degrees(1.0))):
        training = scale * reference[column].to_numpy()
        robot = numpy.concatenate([scale * run[column].to_numpy() for run in logs.values()])
        low, high = numpy.quantile(training, 0.01), numpy.quantile(training, 0.99)
        report.say(f"{name:30s} training 1 % {low:+7.1f}, median {numpy.median(training):+7.1f}, 99 % {high:+7.1f} | robot least {robot.min():+7.1f}, median {numpy.median(robot):+7.1f}, "
                   f"most {robot.max():+7.1f}; outside 1 % to 99 %: {int(((robot < low) | (robot > high)).sum())} of {len(robot)}")
    training = 100 * numpy.hypot(reference["tip_x_m"], reference["tip_y_m"]).to_numpy()
    robot = numpy.concatenate([100 * numpy.hypot(run["tip_x"], run["tip_y"]).to_numpy() for run in logs.values()])
    low, high = numpy.quantile(training, 0.01), numpy.quantile(training, 0.99)
    report.say(f"{'tip reach from the base, cm':30s} training 1 % {low:+7.1f}, median {numpy.median(training):+7.1f}, 99 % {high:+7.1f} | robot least {robot.min():+7.1f}, median {numpy.median(robot):+7.1f}, "
               f"most {robot.max():+7.1f}; outside 1 % to 99 %: {int(((robot < low) | (robot > high)).sum())} of {len(robot)}")
    # the same joints at the same reach: does the held wrist change the elbow for a given reach?
    report.say("elbow (joint 3) by reach band, training median against robot median (degrees):")
    training_reach = 100 * numpy.hypot(reference["tip_x_m"], reference["tip_y_m"]).to_numpy()
    training_elbow, training_shoulder = numpy.degrees(reference["joint_3"].to_numpy()), numpy.degrees(reference["joint_2"].to_numpy())
    robot_elbow = numpy.concatenate([numpy.degrees(run["joint_3"].to_numpy()) for run in logs.values()])
    robot_shoulder = numpy.concatenate([numpy.degrees(run["joint_2"].to_numpy()) for run in logs.values()])
    for low, high in ((14, 18), (18, 22), (22, 26), (26, 30), (30, 36)):
        in_training, in_robot = (training_reach >= low) & (training_reach < high), (robot >= low) & (robot < high)
        if in_training.sum() < 20 or in_robot.sum() < 5:
            continue
        report.say(f"   reach {low} to {high} cm: elbow training {numpy.median(training_elbow[in_training]):+6.1f} (5 % {numpy.quantile(training_elbow[in_training], 0.05):+6.1f}, 95 % "
                   f"{numpy.quantile(training_elbow[in_training], 0.95):+6.1f}), robot {numpy.median(robot_elbow[in_robot]):+6.1f} | shoulder training "
                   f"{numpy.median(training_shoulder[in_training]):+6.1f} (5 % {numpy.quantile(training_shoulder[in_training], 0.05):+6.1f}, 95 % {numpy.quantile(training_shoulder[in_training], 0.95):+6.1f}), "
                   f"robot {numpy.median(robot_shoulder[in_robot]):+6.1f}   ({int(in_training.sum())} training, {int(in_robot.sum())} robot pictures)")

    report.say("")
    report.say("C. ROYAL BLUE IN THE TRAINING PICTURES (every 12th raw picture; large royal-blue objects at the border as in own_setup_checks.py)")
    sample = frames.iloc[::12]
    shares = numpy.array([large_blue_object(read_picture(VERIFY_PARAMETERS["raw_folder"] / frame_path)).mean() for frame_path in sample["frame_path"]])
    report.say(f"{len(shares)} pictures: {100 * (shares > 0.005).mean():.1f} % show more than 0.5 % of such blue, {100 * (shares > 0.026).mean():.1f} % more than 2.6 %, "
               f"{100 * (shares > 0.05).mean():.1f} % more than 5 %; largest share {100 * shares.max():.1f} %")
    twist = numpy.degrees(sample["joint_4"].to_numpy())
    report.say(f"of the pictures with more than 2.6 %: forearm twist median {numpy.median(twist[shares > 0.026]) if (shares > 0.026).any() else float('nan'):+.0f} degrees "
               f"(the robot's own base comes into view when the forearm is twisted far to the negative side)")
    report.save()



if __name__ == "__main__":
    main()
