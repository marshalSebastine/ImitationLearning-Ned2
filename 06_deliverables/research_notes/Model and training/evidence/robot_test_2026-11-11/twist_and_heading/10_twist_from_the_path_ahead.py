"""
Could a hand-written rule make the twist from the model's OTHER outputs (the direction of travel, or the chunk: the
tip's next five places in the picture's frame)? Scored against the demonstrated twist, with the true values of those
outputs (the best such a rule could do) and with the E-recipe fold models' predicted values on held-out sheets.

Candidates, all in degrees in the current picture's frame (0 = picture top, + = left):
    travel direction      the direction of this step;
    chunk end direction   the direction from the tip to its place five steps ahead;
    chunk bend            the direction of the chunk's last two steps minus the direction of its first two steps.
Targets: the twist of this step, and the mean twist of the next five steps.

Output: 10_twist_from_the_path_ahead.out.txt
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 10_twist_from_the_path_ahead.py
"""

from typing import Dict, Optional

import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames, wrapped_deg

from training_targets import chunk_targets

PATH_AHEAD_PARAMETERS = {
    "chunk_steps": 5,
    "smallest_reach_cm": 0.5,
    "largest_angle_deg": 90.0
}



def chunk_features(chunk_cm: numpy.ndarray, smallest_reach_cm: float) -> Dict[str, numpy.ndarray]:
    """
    Return the chunk's end direction and bend.

    Parameters
    ----------
    chunk_cm : numpy.ndarray
        (rows, 5, 2): forward and left of the current tip after each of the next five steps.
    smallest_reach_cm : float
        A direction over a shorter distance is not trusted (NaN).

    Returns
    -------
    dict
        "end" and "bend", degrees, NaN where a direction is not defined.
    """

    end = numpy.degrees(numpy.arctan2(chunk_cm[:, -1, 1], chunk_cm[:, -1, 0]))
    end[numpy.hypot(chunk_cm[:, -1, 0], chunk_cm[:, -1, 1]) < smallest_reach_cm] = numpy.nan
    first = chunk_cm[:, 1, :]
    last = chunk_cm[:, -1, :] - chunk_cm[:, -3, :]
    bend = wrapped_deg(numpy.degrees(numpy.arctan2(last[:, 1], last[:, 0])) - numpy.degrees(numpy.arctan2(first[:, 1], first[:, 0])))
    bend[(numpy.hypot(first[:, 0], first[:, 1]) < smallest_reach_cm) | (numpy.hypot(last[:, 0], last[:, 1]) < smallest_reach_cm)] = numpy.nan
    return {"end": end, "bend": bend}



def main(parameters: Optional[Dict] = None) -> None:
    """Score the direction of travel and the chunk as sources of a hand-written twist rule against the demonstrated twist."""

    settings = dict(PATH_AHEAD_PARAMETERS)
    settings.update(parameters or {})
    report = Report("10_twist_from_the_path_ahead.out.txt")
    frames = demonstration_frames()
    true_chunk = chunk_targets(frames, settings["chunk_steps"])
    features = chunk_features(true_chunk, settings["smallest_reach_cm"])
    frames["true_end_deg"], frames["true_bend_deg"] = features["end"], features["bend"]
    saved = numpy.load(EVIDENCE_FOLDER / "fold_steps_E.npz")
    predicted_features = chunk_features(saved["chunk_cm"], settings["smallest_reach_cm"])
    answers = pandas.DataFrame({"frame_index": saved["frame_index"], "predicted_end_deg": predicted_features["end"], "predicted_bend_deg": predicted_features["bend"],
                                "predicted_travel_deg": numpy.degrees(numpy.arctan2(saved["left_cm"], saved["forward_cm"])), "predicted_twist_deg": saved["twist_deg"]})
    table = frames.merge(answers, on = "frame_index").sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    table = table[table["action"] != "stop"].reset_index(drop = True)
    table["twist"] = table["forearm_change_deg"].fillna(0.0)
    table["twist_next_5"] = table.groupby("episode_id")["twist"].transform(lambda values: values[::-1].rolling(5, min_periods = 5).mean()[::-1])

    report.say("THE TWIST FROM THE PATH AHEAD: could a rule make the twist out of the direction of travel or the chunk? (measured on the demonstrations; fold E on held-out sheets)")
    report.say("correlation with the demonstrated twist, and the least-squares twist per degree; a rule 'turn the camera TOWARDS that direction' needs a NEGATIVE sign")
    report.say("(direction + = to the picture's left; twist + = the camera turns to the right)")
    report.say("")
    report.say(f"   {'what the rule would use':52s} {'rows':>6s} {'twist of this step':>30s} {'mean twist of the next 5 steps':>32s}")
    candidates = [("true travel direction of this step", "move_angle_deg"), ("true chunk end direction (5 steps ahead)", "true_end_deg"), ("true chunk bend", "true_bend_deg"),
                  ("predicted travel direction (fold E)", "predicted_travel_deg"), ("predicted chunk end direction (fold E)", "predicted_end_deg"), ("predicted chunk bend (fold E)", "predicted_bend_deg"),
                  ("predicted twist itself (fold E), for comparison", "predicted_twist_deg")]
    for name, column in candidates:
        parts = []
        for target in ("twist", "twist_next_5"):
            is_used = table[column].notna() & table[target].notna()
            if column != "predicted_twist_deg":
                is_used &= table[column].abs() <= settings["largest_angle_deg"]
            values, targets = table.loc[is_used, column].to_numpy(), table.loc[is_used, target].to_numpy()
            parts.append(f"r {numpy.corrcoef(values, targets)[0, 1]:+.2f}, slope {float((values * targets).sum() / (values ** 2).sum()):+.3f}")
            rows = int(is_used.sum())
        report.say(f"   {name:52s} {rows:6d} {parts[0]:>30s} {parts[1]:>32s}")
    report.say("")
    report.say("   per sheet: correlation of the demonstrated twist of this step with the true travel direction, the true chunk end direction and the true chunk bend")
    for sheet, rows in table.groupby("sheet"):
        parts = []
        for column in ("move_angle_deg", "true_end_deg", "true_bend_deg"):
            is_used = rows[column].notna() & (rows[column].abs() <= settings["largest_angle_deg"])
            parts.append(f"{numpy.corrcoef(rows.loc[is_used, column], rows.loc[is_used, 'twist'])[0, 1]:+.2f}")
        report.say(f"   {sheet}: " + "  ".join(parts))
    report.save()



if __name__ == "__main__":
    main()
