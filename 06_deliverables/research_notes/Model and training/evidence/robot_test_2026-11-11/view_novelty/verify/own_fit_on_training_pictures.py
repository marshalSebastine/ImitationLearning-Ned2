"""
How well does model E reproduce the demonstrated step on the pictures it was trained on, by kind of step?

It follows own_answers_on_neighbours.py: where a robot run went wrong, model E gave the same answer on the nearest
training pictures as on the robot picture, and that answer was far from those pictures' labels. This script asks how
common that is over all 18,032 training pictures: the direction error of model E's answer against the label, by the
kind of demonstrated step.

Kinds (from the label as model E saw it):
    straight push        length 0.8 cm or more, direction within 20 degrees of the picture's top
    sideways push        length 0.8 cm or more, direction 40 degrees or more off the picture's top (the swing round
                         the puck and the pushes out of a bend)
    turn in place        length under 0.5 cm with a forearm twist of 4 degrees or more
    short without twist  length under 0.5 cm with less twist

Run from this folder, after extract_own.py model_e (a few seconds):
    D:/AI_Project/.venv/Scripts/python.exe own_fit_on_training_pictures.py
Output: own_fit_on_training_pictures.out.txt
"""

import numpy

from own_novelty import load_table, trained_rows
from verify_common import VERIFY_FOLDER, Report



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_fit_on_training_pictures.out.txt")
    trained = trained_rows()
    trained = trained[trained["action"] != "stop"]
    answers = load_table("model_e_training")["exact"][trained["frame_index"].to_numpy()]
    forward, left = 100 * trained["tip_forward_m"].fillna(0.0).to_numpy(), 100 * trained["tip_left_m"].fillna(0.0).to_numpy()
    twist = trained["forearm_change_deg"].fillna(0.0).to_numpy()
    label_length, label_angle = numpy.hypot(forward, left), numpy.degrees(numpy.arctan2(left, forward))
    answer_length, answer_angle = numpy.hypot(answers[:, 0], answers[:, 1]), numpy.degrees(numpy.arctan2(answers[:, 1], answers[:, 0]))
    error = numpy.abs((answer_angle - label_angle + 180) % 360 - 180)
    kinds = {
        "straight push": (label_length >= 0.8) & (numpy.abs(label_angle) <= 20),
        "sideways push": (label_length >= 0.8) & (numpy.abs(label_angle) >= 40),
        "turn in place": (label_length < 0.5) & (numpy.abs(twist) >= 4),
        "short without twist": (label_length < 0.5) & (numpy.abs(twist) < 4),
        "all moving rows": numpy.ones(len(trained), bool)
    }
    report.say("MODEL E ON ITS OWN TRAINING PICTURES, by kind of demonstrated step (moving rows only)")
    report.say("kind                  pictures  share   label length cm  answer length cm  direction error deg: median  75 %   share over 45 deg   label twist  answer twist (median size)")
    for name, is_kind in kinds.items():
        report.say(f"{name:20s}  {int(is_kind.sum()):8d}  {100 * is_kind.mean():4.0f} %  {numpy.median(label_length[is_kind]):15.2f}  {numpy.median(answer_length[is_kind]):16.2f}  "
                   f"{numpy.median(error[is_kind]):27.0f}  {numpy.quantile(error[is_kind], 0.75):4.0f}  {100 * (error[is_kind] > 45).mean():15.0f} %  "
                   f"{numpy.median(numpy.abs(twist[is_kind])):11.1f}  {numpy.median(numpy.abs(answers[is_kind, 2])):12.1f}")
    report.say("")
    report.say("the same by sheet, sideways pushes only (direction error in degrees: median, share over 45)")
    for sheet in sorted(trained["sheet"].unique()):
        is_kind = kinds["sideways push"] & (trained["sheet"] == sheet).to_numpy()
        if is_kind.sum() >= 20:
            report.say(f"   {sheet}: {int(is_kind.sum()):5d} pictures, median {numpy.median(error[is_kind]):3.0f}, over 45 degrees {100 * (error[is_kind] > 45).mean():3.0f} %, "
                       f"answer length median {numpy.median(answer_length[is_kind]):.2f} cm against label {numpy.median(label_length[is_kind]):.2f} cm")
    report.save()



if __name__ == "__main__":
    main()
