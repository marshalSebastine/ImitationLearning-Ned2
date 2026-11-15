"""
Robot-session critic: which wording of block A's rule gives back what the check found on 11.11.26 (runs 5 and 9 lost
the puck at the first bend of sheet 03, run 10 passed it)? Windows of 4, 6, 8 and 10 cm past the bend and limits of
12 and 17 mm from the line's middle, on stop_and_centring/robot_runs_steps.csv; a reading counts only when the
measure's own flags are good (is_clean, is_aligned, the line followed back less than 5 mm) and the limit must be passed
on two such pictures in a row. Also the five sheet-03 demonstrations the check measured. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe block_a_rule_windows.py > block_a_rule_windows.out.txt
"""

from pathlib import Path

import numpy
import pandas

FOLDER = Path(__file__).resolve().parents[2] / "stop_and_centring"
runs = pandas.read_csv(FOLDER / "robot_runs_steps.csv")
demonstrations = pandas.read_csv(FOLDER / "demonstration_centring_03.csv")


def verdict(rows: pandas.DataFrame, past_bend_cm: float, limit_mm: float) -> str:
    rows = rows.sort_values("step").reset_index(drop = True)
    puck = pandas.DataFrame(rows[["puck_x", "puck_y"]].to_numpy() * 100.0).ffill().bfill().to_numpy()
    path = numpy.r_[0.0, numpy.cumsum(numpy.hypot(numpy.diff(puck[:, 0]), numpy.diff(puck[:, 1])))]
    if path[-1] < 14.0:
        return "short"
    first = puck[numpy.searchsorted(path, 2.0)] - puck[0]
    first = first / numpy.linalg.norm(first)
    bend = None
    for index in range(3, len(puck)):
        base = int(numpy.searchsorted(path, max(0.0, path[index] - 1.5)))
        move = puck[index] - puck[base]
        if numpy.linalg.norm(move) > 0.8 and numpy.degrees(numpy.arccos(numpy.clip(move @ first / numpy.linalg.norm(move), -1.0, 1.0))) > 45.0:
            bend = index
            break
    end = int(numpy.searchsorted(path, path[bend] + past_bend_cm, side = "right"))
    window = rows.iloc[:end]
    is_sound = window["is_clean"].astype(bool) & window["is_aligned"].astype(bool) & (window["extrapolated_mm"] < 5.0)
    over = ((window["offset_mm"].abs() > limit_mm) & is_sound).to_numpy()
    fails = [index for index in range(1, len(over)) if over[index] and over[index - 1]]
    return "pass" if not fails else f"FAIL@{int(window['step'].iloc[fails[0]])}"


print("wording (window past the bend, limit from the line's middle) | run 5 | run 9 | run 10 | gives back the check's reading (fail, fail, pass)? | demonstrations 03-25, 03-02, 03-13, 03-18, 03-07")
for past_bend_cm in (4.0, 6.0, 8.0, 10.0):
    for limit_mm in (12.0, 17.0):
        results = [verdict(runs[runs["run"] == run], past_bend_cm, limit_mm) for run in (5, 9, 10)]
        is_right = results[0].startswith("FAIL") and results[1].startswith("FAIL") and results[2] == "pass"
        demonstration_results = [verdict(group, past_bend_cm, limit_mm) for _, group in demonstrations.groupby("episode_id", sort = False)]
        demonstration_results = [value for value in demonstration_results if value != "short"]
        print(f"   {past_bend_cm:4.0f} cm, {limit_mm:2.0f} mm | {results[0]:8s} | {results[1]:8s} | {results[2]:8s} | {'yes' if is_right else 'NO'} | {demonstration_results}")
print("\nnote: run 5's first counted failure comes from readings of 49 and 44 mm at pictures 8 and 9 that the measure's own flags call good; the check's other tool")
print("(step_length/line_and_puck.py) puts the puck 0.8 to 1.4 cm outside the ink there. The verdict is right, the number is not.")
