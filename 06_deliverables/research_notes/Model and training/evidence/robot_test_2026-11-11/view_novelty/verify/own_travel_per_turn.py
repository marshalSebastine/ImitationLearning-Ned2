"""
How far did the tip travel while the forearm turned, at the first bend of sheet 03 and at the square start of sheet 02:
the demonstrations against the robot, and against what the model had asked for.

For each episode or run the count starts at the first row whose forearm twist is 4 degrees or more in size and ends
at the row where the summed twist first reaches 30 degrees in size (the same sign). Reported: rows needed, tip path
travelled (sum of the step lengths) and, for the robot, the path the model had asked for over the same pictures.

Run from this folder (a few seconds):
    D:/AI_Project/.venv/Scripts/python.exe own_travel_per_turn.py
Output: own_travel_per_turn.out.txt
"""

import numpy

from own_novelty import trained_rows
from verify_common import VERIFY_FOLDER, Report, run_log

TRAVEL_PARAMETERS = {
    "twist_start_deg": 4.0,
    "turn_deg": 30.0,
    "rows_considered": 40
}



def travel_for_turn(twist: numpy.ndarray, length: numpy.ndarray, asked = None):
    """Return (rows, path cm, asked path cm) from the first twisting row until the summed twist reaches the set turn."""

    settings = TRAVEL_PARAMETERS
    starts = numpy.flatnonzero(numpy.abs(twist) >= settings["twist_start_deg"])
    if len(starts) == 0:
        return None
    first = starts[0]
    sign = numpy.sign(twist[first])
    summed = numpy.cumsum(twist[first:]) * sign
    reached = numpy.flatnonzero(summed >= settings["turn_deg"])
    if len(reached) == 0:
        return None
    last = first + reached[0]
    return last - first + 1, float(length[first:last + 1].sum()), (float(asked[first:last + 1].sum()) if asked is not None else numpy.nan), int(first)



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_travel_per_turn.out.txt")
    settings = TRAVEL_PARAMETERS
    trained = trained_rows()
    report.say(f"TIP PATH WHILE THE FOREARM TURNS ITS FIRST {settings['turn_deg']:.0f} DEGREES (from the first row with a twist of {settings['twist_start_deg']:.0f} degrees or more)")
    for title, is_chosen, runs in (
            ("sheet 03, standard starts from end A (the first bend)", (trained["sheet"] == "path_03") & (trained["start_kind"] == "STD") & (trained["start_end"] == "A"), [5, 9, 10]),
            ("sheet 02, square portrait starts (SQP)", (trained["sheet"] == "path_02") & (trained["start_kind"] == "SQP"), [3])):
        report.say("")
        report.say(title)
        report.say("demonstration  first twisting row  rows  tip path cm  cm per 10 degrees")
        collected = []
        for plan, episode in trained[is_chosen].groupby("plan_number"):
            episode = episode.sort_values("row_in_episode").head(settings["rows_considered"])
            twist = episode["forearm_change_deg"].fillna(0.0).to_numpy()
            length = 100 * numpy.hypot(episode["tip_forward_m"].fillna(0.0), episode["tip_left_m"].fillna(0.0)).to_numpy()
            result = travel_for_turn(twist, length)
            if result is None:
                report.say(f"{plan:13s}  the forearm never turned {settings['turn_deg']:.0f} degrees one way within {settings['rows_considered']} rows")
                continue
            rows, path, _, first = result
            collected.append((rows, path))
            report.say(f"{plan:13s}  {int(episode['row_in_episode'].iloc[first]):18d}  {rows:4d}  {path:11.2f}  {10 * path / settings['turn_deg']:17.2f}")
        if collected:
            rows, path = numpy.array(collected).T
            report.say(f"median of {len(collected)} demonstrations: {numpy.median(rows):.0f} rows, {numpy.median(path):.2f} cm ({10 * numpy.median(path) / settings['turn_deg']:.2f} cm per 10 degrees); "
                       f"range {path.min():.2f} to {path.max():.2f} cm")
        report.say("robot run  first twisting step  steps  tip path done cm  path the model asked cm  done / asked")
        for run_number in runs:
            log = run_log(run_number).head(settings["rows_considered"])
            twist = log["twist_used_deg"].fillna(0.0).to_numpy()
            done = 100 * numpy.hypot(log["tip_next_x"] - log["tip_x"], log["tip_next_y"] - log["tip_y"]).fillna(0.0).to_numpy()
            asked = numpy.hypot(log["model_tip_forward_cm"], log["model_tip_left_cm"]).to_numpy()
            result = travel_for_turn(twist, done, asked)
            if result is None:
                report.say(f"{run_number:9d}  the forearm never turned {settings['turn_deg']:.0f} degrees one way within {settings['rows_considered']} steps")
                continue
            rows, path, asked_path, first = result
            report.say(f"{run_number:9d}  {first:19d}  {rows:5d}  {path:16.2f}  {asked_path:23.2f}  {path / asked_path:12.2f}")
    report.save()



if __name__ == "__main__":
    main()
