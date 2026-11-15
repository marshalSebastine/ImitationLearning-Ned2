"""
Proofread (fidelity): how long the three test sittings of version 2 take with the robot-session critic's own unit times
(robot_session/session_time_proposed.py: 0.94 s per picture; gap between runs 112 s as in the recordings or 181 s as the
median of 11.11.26; extra seconds per full run for the ruler reading). Calculated; the set-up minutes are guesses taken
from the same script (switching on 15 min, 5 min of tape per lay-out, 10 min of checklists for new movements, 15 min of
repeated runs for a full-run sitting). Version 2 adds a photo with a ruler at every stop and one uncounted warm-up run;
the photo is counted as 30 s on short runs and 60 s on full runs, as the critic counted the ruler reading.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe sitting_times_of_version_2.py > sitting_times_of_version_2.out.txt
"""

SECONDS_PER_PICTURE = 0.94

# name, runs, pictures per run, extra seconds per run
SITTING_A = [
    ("warm-up run, not counted (sheet 01)", 1, 33, 30),
    ("A1: fixed list of answers without the puck (30 small steps and 10 twists)", 1, 40, 0),
    ("A1: the same with the puck on sheet 01", 1, 40, 0),
    ("A2: the measure at 20 places, two bare pictures and one with the puck (30 min, the checker's figure)", 0, 0, 30 * 60),
    ("A3: model E, sheet 01, driver of 11.11, 3 runs", 3, 33, 30),
    ("A4: model E, fixed 1.0 cm, 5 short runs (27 cm budget)", 5, 27, 30),
    ("A4: model E, fixed 0.65 cm, 5 short runs", 5, 42, 30),
    ("A4: model E, own length, 4 short runs", 4, 44, 30),
    ("A4: team of five, fixed 1.0 cm, 4 short runs", 4, 27, 30),
    ("A4: model without sheet 03, fixed 1.0 and 0.65 cm, 3 short runs each", 6, 35, 30),
    ("A5: sheet 04, model E, driver of 11.11, 3 short runs past the first hairpin (about 30 pictures)", 3, 30, 30),
    ("A5: one full run of sheet 04 if the bends are passed (153 cm at 1.0 cm)", 1, 153, 60),
]
SITTING_B = [
    ("warm-up run, not counted", 1, 33, 30),
    ("B1: model E, sheet 01, chosen step (0.65 cm), 3 runs", 3, 51, 60),
    ("B2: sheet 05, model E, fixed 1.0 cm, 3 runs (105 cm)", 3, 105, 60),
    ("B2: sheet 05, model E, chosen step, 3 runs", 3, 162, 60),
    ("B2: sheet 05, model without sheet 05, 3 runs (step not stated in the plan; 1.0 cm taken)", 3, 105, 60),
    ("B3: sheet 03, model without it, fixed 1.0 cm, 3 runs (114 pictures)", 3, 114, 60),
    ("B3: sheet 03, model without it, chosen step, 3 runs", 3, 175, 60),
    ("B3: new track 1, model E, fixed 1.0 cm, 3 runs (about 80 cm)", 3, 80, 60),
    ("B3: new track 1, model E, chosen step, 3 runs", 3, 123, 60),
    ("B3: new track 2, model E, chosen step, 3 runs", 3, 123, 60),
    ("B4: our own print of sheet 01, model E, 2 runs", 2, 33, 60),
]
SITTING_C = [
    ("warm-up run, not counted", 1, 33, 30),
    ("C1: near-end starts on sheets 02, 05, 06 and 03, 3 runs each (about 12 pictures)", 12, 12, 60),
    ("C2: shapes 05 and 06, solid and dashed, 2 runs each (97 cm at 1.0 cm)", 8, 97, 60),
    ("C3: sheet 01 at about 45 degrees, 2 starts, 2 runs each", 4, 33, 60),
]

for gap_name, gap in (("gap as the recordings' own (112 s)", 112.0), ("gap as the median of 11.11 (181 s)", 181.0)):
    print(gap_name)
    for sitting_name, blocks, extra_minutes, what, stated in (
            ("sitting A", SITTING_A, 15 + 15 + 10, "switching on and camera check 15, tape for sheets 01, 03 and 04: 15, checklists for the new movements 10", 2.5),
            ("sitting B", SITTING_B, 15 + 30 + 15, "switching on 15, tape for six lay-outs 30, repeated runs 15", 2.5),
            ("sitting C", SITTING_C, 15 + 45 + 10, "switching on 15, tape for nine lay-outs 45, repeated runs 10", 2.0)):
        minutes = 0.0
        for name, runs, pictures, extra in blocks:
            block_minutes = (runs * (pictures * SECONDS_PER_PICTURE + gap + extra) if runs else extra) / 60.0
            minutes += block_minutes
            print(f"      {name}: {block_minutes:.0f} min")
        runs_total = sum(block[1] for block in blocks)
        print(f"   {sitting_name}: {runs_total} runs, {minutes:.0f} min of runs + {extra_minutes} min ({what}) [guess] "
              f"= {(minutes + extra_minutes) / 60:.1f} h; the plan says about {stated} h")
    print()
