"""
Robot-session critic: the time of the leaner two-visit order proposed in the review, with the same measured pieces as
session_time.py (0.94 s per picture; the gap between runs of 11.11.26: median 181 s; the recordings' own set-up between
episodes, 112 s = 1.86 minutes, from report section 7.4). Calculated; the set-up minutes are guesses.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe session_time_proposed.py > session_time_proposed.out.txt
"""

SECONDS_PER_PICTURE = 0.94
# name, runs, pictures per run, extra seconds per run (ruler reading, note)
DAY_1 = [
    ("block 0a: scripted steps without the puck (30 small steps and 10 twists)", 1, 40, 0),
    ("block 0b: the same with the puck on sheet 01", 1, 40, 0),
    ("measure at known places: 20 placements, one picture each (45 s each, no run)", 0, 0, 20 * 45),
    ("control: model E on sheet 01, 11.11 driver, 3 runs (33 cm of tip travel)", 3, 33, 30),
    ("block A: model E, fixed 1.0 cm, 5 short runs (22 cm)", 5, 22, 0),
    ("block A: model E, fixed 0.65 cm, 5 short runs", 5, 34, 0),
    ("block A: model E, own length, 4 short runs", 4, 36, 0),
    ("block A: model without sheet 03, fixed 1.0 and 0.65 cm, 3 short runs each", 6, 28, 0),
]
DAY_2 = [
    ("stop: model E on sheet 01 with the chosen step (0.65 cm), 3 runs", 3, 51, 60),
    ("sheet 05: model E 3 runs and the model without sheet 05 3 runs (105 cm at 0.65 cm)", 6, 162, 60),
    ("two new tracks: model E, 3 runs each (about 80 cm of tip travel at 0.65 cm)", 6, 123, 60),
    ("sheet 04 repeat: model E, 11.11 driver, 2 runs (153 cm at 1.0 cm)", 2, 153, 60),
    ("block D: two start poses, 2 short runs each (12 cm)", 4, 15, 0),
]
BLOCK_C = [("block C as 2 runs per cell of shape x line style (97 cm at 1.0 cm)", 8, 97, 60)]

for gap_name, gap in (("gap as the recordings' own (112 s)", 112.0), ("gap as the median of 11.11 (181 s)", 181.0)):
    print(gap_name)
    for day_name, blocks, extra_minutes, what in (("day 1", DAY_1, 15 + 10 + 10, "switching on and tests 15, tape for sheets 01 and 03: 10, checklists for the new movements 10"),
                                                  ("day 2 without block C", DAY_2, 15 + 25 + 15, "switching on 15, tape for five lay-outs 25, repeated runs 15"),
                                                  ("block C on top", BLOCK_C, 20, "tape for four prints 20")):
        minutes = 0.0
        for name, runs, pictures, extra in blocks:
            block_minutes = (runs * (pictures * SECONDS_PER_PICTURE + gap + extra) if runs else extra) / 60.0
            minutes += block_minutes
            print(f"      {name}: {block_minutes:.0f} min")
        runs_total = sum(block[1] for block in blocks)
        print(f"   {day_name}: {runs_total} runs, {minutes:.0f} min of runs + {extra_minutes} min ({what}) [guess] = {(minutes + extra_minutes) / 60:.1f} h")
    print()
