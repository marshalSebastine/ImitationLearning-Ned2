"""
Robot-session critic: the plan's pass rule of block A ("the puck stays within 12 mm of the line's middle to 10 cm past
the bend") applied to the three sheet-03 runs of 11.11.26 with the measure that exists
(stop_and_centring/robot_runs_steps.csv, written by robot_runs.py with run_measures.measure_steps).

The check's own reading of those runs is "the first bend was passed in 1 of 3 runs" (run 10; runs 5 and 9 lost the
puck there). Here the same runs are scored by the rule as the plan words it, in four readings of the wording, to see
whether the rule gives that 1 of 3 back. Reads one CSV. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe block_a_rule_on_the_three_runs.py > block_a_rule_on_the_three_runs.out.txt
"""

from pathlib import Path

import numpy
import pandas

STEPS = Path(__file__).resolve().parents[2] / "stop_and_centring" / "robot_runs_steps.csv"
RUNS = {5: "run 5 (model E, lost the puck at the first bend)", 9: "run 9 (model E, lost the puck at the first bend)",
        10: "run 10 (team B, 'passed the first bend', lost at the next turn)"}

table = pandas.read_csv(STEPS)
print("the puck's centre against the line's middle, mm (offset_mm of run_measures.measure_steps); 'bend' = the first picture at which the puck's own path has turned")
print("45 degrees from its first direction; the window ends when the puck has travelled 10 cm more\n")
for run, name in RUNS.items():
    rows = table[table["run"] == run].reset_index(drop = True)
    puck = rows[["puck_x", "puck_y"]].to_numpy() * 100.0
    path = numpy.r_[0.0, numpy.nancumsum(numpy.hypot(numpy.diff(puck[:, 0]), numpy.diff(puck[:, 1])))]
    # the puck's direction over 3 pictures, against its direction over its first 2 cm
    first = puck[numpy.searchsorted(path, 2.0)] - puck[0]
    first = first / numpy.linalg.norm(first)
    bend = None
    for index in range(3, len(puck)):
        move = puck[index] - puck[index - 3]
        if numpy.linalg.norm(move) > 0.5 and numpy.degrees(numpy.arccos(numpy.clip(move @ first / numpy.linalg.norm(move), -1.0, 1.0))) > 45.0:
            bend = index
            break
    end = int(numpy.searchsorted(path, path[bend] + 10.0, side = "right"))
    window = rows.iloc[:end]
    offset = window["offset_mm"].abs()
    is_sound = window["is_clean"].astype(bool) & window["is_aligned"].astype(bool) & (window["extrapolated_mm"] < 5.0)
    print(name)
    print(f"   bend at picture {bend} (puck path {path[bend]:.1f} cm), window = pictures 0 to {end - 1} (the puck travelled {path[end - 1] - path[bend]:.1f} cm past the bend "
          f"by picture {end - 1}); the run itself had {len(rows)} pictures")
    print(f"   offsets in the window (mm): {[None if numpy.isnan(value) else round(float(value), 1) for value in window['offset_mm']]}")
    print(f"   readings the measure itself marks as doubtful (the line followed back more than 5 mm under the puck, or its flag is_clean or is_aligned is False): pictures "
          f"{[int(step) for step in window['step'][~is_sound & offset.notna()]]}")
    for label, limit, sound_only, in_a_row in (("as worded: any picture over 12 mm fails", 12.0, False, 1),
                                               ("doubtful readings left out", 12.0, True, 1),
                                               ("doubtful readings left out, two pictures in a row", 12.0, True, 2),
                                               ("17 mm (the puck no longer touches the ink), doubtful left out, two in a row", 17.0, True, 2)):
        over = (offset > limit) & (is_sound if sound_only else True)
        over = over.to_numpy()
        fails = [int(window["step"].iloc[index]) for index in range(len(over)) if all(over[max(0, index - in_a_row + 1):index + 1]) and index >= in_a_row - 1]
        largest = float(offset[is_sound].max()) if sound_only else float(offset.max())
        print(f"      {label:78s}: {'PASS' if not fails else 'FAIL at picture ' + str(fails[0])} (largest counted offset {largest:.1f} mm)")
    # how close to the limit is the passing run at the end of the window?
    last_three = [round(float(value), 1) for value in window["offset_mm"].abs().iloc[-3:]]
    print(f"   the last three pictures of the window read {last_three} mm; one picture later: "
          f"{round(float(abs(rows['offset_mm'].iloc[end])), 1) if end < len(rows) else 'none'} mm\n")
print("the measure's own error [from the check, verdict 12 of 'stop and centring']: two independent measures differ by a median of 1 to 2 mm on runs 1, 2 and 4;")
print("views of one place differ by 1.8 to 3.3 mm; not checked against a ruler; unreliable where two stretches of line are near each other")
