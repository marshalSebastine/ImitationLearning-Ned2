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

## Two driver fixes checked on the logs and the recorded pictures before building them (15.11.26, 18:00)

Asked for by Emranul ("let's try idea 1 and 2"); both checked first, neither built.

1. Stop when the tip goes back over old ground. On today's 28 runs the longest stretch on old ground (tip within 1.0
   to 1.5 cm of a place passed more than 10 steps before) was 4 to 6 steps in model E's two good loop runs (the crossing)
   and also only 4 to 6 steps in model 15's runs that turned back near the end (D02, D06): no limit tells them apart.
   A limit of 8 steps would end the long circling (U9, U7, A6), but only after the puck had been pushed back about
   5 cm, so no run would have finished. The model's own stop chance stayed near 0 at the true line end of D02, D04 and
   D06 and rose only one or two steps before its stop, so a lower stop threshold would not help either: the model does
   not see these line ends (the paper-edge cue of section 1 of the search notes).
2. Steer towards a point 2, 3 or 5 steps ahead instead of the next step. Measured on the held-out pictures of model 15's
   recipe against where the recorded tip went over the next 2 steps (within 30 degrees): next step 0.689, 2 ahead 0.695,
   3 ahead 0.686, 5 ahead 0.662; on bends of 25 to 45 degrees 0.690, 0.687, 0.654, 0.620. The gain of looking 2 steps
   ahead is under 1 point and looking further costs on bends, where the loop needs sharper steering, not smoother. The
   earlier "+1 to 2.5 points" was measured against the path direction 3 steps ahead, which favours a look-ahead.
