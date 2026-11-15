# Robot test of 15.11.26: results (Emranul at the robot, 15:40 to 17:36)

Run list `2026-11-15_robot_test_runs.csv`, marks given on the recording page's new test part
(`2026-11-15_robot_test_marks.csv`, Emranul's words), run logs in `D:\AI_Project\03_data\model_runs\2026-11-15_1*`.
Paths and counts from the logs: `evidence/robot_test_2026-11-15/run_paths.py` (`run_paths.csv`, `run_paths.png`).

Note on the marks file: its fifth row (16:09, "A4", "better than model E") is a second run of model E on the loop
(`2026-11-15_160557_A4`), started by mistake with "Run it again"; the remark was written believing it was model 15.
Test U9 (`2026-11-15_172126_U9`) was started from a terminal, after the page twice ended it at once (cause not found).

## Finished runs (puck covers the line end and stayed on the line, Emranul's rule)

| Model | Runs | Finished | Which |
|---|---|---|---|
| E (round 1) | 11 | 2 | sheet 02 landscape, recovery on sheet 01 |
| 13 (hand corrections, no mirror, 8 rounds) | 2 | 2 | sheet 03, recovery on sheet 01 |
| 14 (13 with two pictures) | 1 | 0 | |
| 15 (mirror, after h x3, 16 rounds) | 14 | 2 | sheet 02 landscape, new track D01 |

On the six new printed tracks model 15 kept the puck on the line in 4 of 6 runs (Emranul's marks), model E in 0 of 6;
model E left the paper or stopped early on every one of them. On sheet 03 model E got lost, model 15 stopped 1 to 2 mm
short of covering the end, model 13 finished.

## How model 15 fails (Emranul's marks and the tip paths)

1. Loops: on sheet 05 laid as on 11.11 it does not curl into the loop (A5, 0.65 cm) or curls in the wrong place (A6,
   1 cm); on the new loop tracks D03 and D09 it goes round the loop again and again (U9: 45 of 173 steps on ground it
   had passed before). Emranul: "it has no sense of direction". Measured: model E's two loop runs turned sharply (95th
   percentile of the turn per step 42 to 44 degrees), model 15's failed loop run A5 only 21 degrees: model 15 steers too
   gently to make the tight curl.
2. The end: on three new tracks (D02, D04, D06) it drove well to near the end, then turned and went back along the
   line instead of stopping (13 to 15 steps on ground passed before); on sheet 01 it stopped past the end three times
   (A10, R2, R4), there also with the end effector hiding the puck.
3. Recovery: it brings a puck set 1.5 cm beside the line back (R2, R4) but then misses the stop; model E and model 13
   recovered and finished.

The robot never stepped backwards (no turn of more than 120 degrees from one step to the next in any run): going back
along the line happens by a gradual U-turn.

## Ninth queue (state when the VPN went off for the robot, about 15:40)

On model 15's recipe (0.856 path within 45, 17 of 55 right stops, after an h press 0.693): the approach to the goal
counted twice 0.847, 20 of 55 right stops (the most so far, 10 early), after an h press 0.671; after h five times
0.835, 16 of 55, 0.680 (worse than three times). Two pictures, 24 rounds and the two seeds were still running; the
rest needs a fetch login.
