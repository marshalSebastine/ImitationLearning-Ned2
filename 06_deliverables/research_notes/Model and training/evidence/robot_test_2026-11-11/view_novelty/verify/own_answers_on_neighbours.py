"""
Does model E answer on the nearest DEMONSTRATION pictures as it answered on the robot picture? A sharper form of the
"view or decision" question for the steps where the failed runs first left what the demonstrations did.

For a robot picture: model E's answer on it (from the log), the labels of its 5 nearest training pictures (model E's
descriptors) and model E's own answers on those 5 training pictures. If model E reproduces the labels on the
demonstration pictures but answers differently on the robot picture, the robot picture differs from them in a way that
matters to the model although the descriptor distance is ordinary.

Run from this folder, after extract_own.py model_e (a few seconds):
    D:/AI_Project/.venv/Scripts/python.exe own_answers_on_neighbours.py
Output: own_answers_on_neighbours.out.txt
"""

import numpy
import pandas

from own_novelty import load_table, nearest, trained_rows
from verify_common import VERIFY_FOLDER, Report, run_log, training_frames

KEY_PICTURES = [(3, 0), (3, 3), (5, 2), (5, 3), (5, 4), (9, 3), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 12), (8, 9), (8, 14), (8, 17)]



def step_text(forward: float, left: float, twist: float) -> str:
    """Return 'length cm @ direction deg, twist' of a step."""

    return f"{numpy.hypot(forward, left):.2f}@{numpy.degrees(numpy.arctan2(left, forward)):+4.0f} t{twist:+.1f}"



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_answers_on_neighbours.out.txt")
    frames = training_frames()
    trained = trained_rows().set_index("frame_index")
    reference = frames[frames["frame_index"].isin(trained.index)].reset_index(drop = True)
    labels = trained.loc[reference["frame_index"], ["tip_forward_m", "tip_left_m", "forearm_change_deg"]].fillna(0.0).reset_index(drop = True)
    training, robot = load_table("model_e_training"), load_table("model_e_robot")
    reference_grid = training["grid"][reference["frame_index"].to_numpy()]
    answers = training["exact"][reference["frame_index"].to_numpy()]
    report.say("run step | model E on the robot picture | mean of the 5 neighbours' LABELS | mean of model E's ANSWERS on those 5 pictures | each neighbour: plan(row) label -> model E's answer")
    for run_number, step in KEY_PICTURES:
        is_run = robot["run"] == run_number
        _, index = nearest(robot["grid"][is_run][step:step + 1], reference_grid, count = 5)
        row = run_log(run_number).iloc[step]
        label = labels.iloc[index[0]]
        answer = answers[index[0]]
        parts = []
        for position, reference_row in enumerate(index[0]):
            neighbour = reference.iloc[reference_row]
            parts.append(f"{neighbour['plan_number']}({int(neighbour['row_in_episode'])}) {step_text(100 * label['tip_forward_m'].iloc[position], 100 * label['tip_left_m'].iloc[position], label['forearm_change_deg'].iloc[position])}"
                         f" -> {step_text(answer[position, 0], answer[position, 1], answer[position, 2])}")
        report.say(f"{run_number:3d} {step:4d} | {step_text(row['model_tip_forward_cm'], row['model_tip_left_cm'], row['model_twist_deg']):28s} | "
                   f"{step_text(100 * label['tip_forward_m'].mean(), 100 * label['tip_left_m'].mean(), label['forearm_change_deg'].mean()):32s} | "
                   f"{step_text(answer[:, 0].mean(), answer[:, 1].mean(), answer[:, 2].mean()):45s} | " + "; ".join(parts))
    report.save()



if __name__ == "__main__":
    main()
