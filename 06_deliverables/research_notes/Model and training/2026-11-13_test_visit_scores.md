# Test visit of 12.11.26: the runs scored from their pictures

Written 13.11.26 by a scoring session for the planning session and for Emranul. The rules are those written before the
visit (`2026-11-12_test_visit_checklist.md`, "The rules, written before the visit"). Nothing here touched the robot or
the run folders. Scripts, their outputs and the figures are in `evidence/test_visit_2026-11-12/`.

## What this does and does not show

- Every run was measured from its own camera pictures. For each picture I found where the puck stands on the table and
  how far its centre is from the middle of the printed line. Emranul's eye at the robot is the second observer.
- **The step question is not settled.** With model E on sheet 03, 0.65 cm steps kept the line in 3 of 5 runs and
  1.0 cm steps in 1 of 5 (main reading). The rule needs 3 more kept runs, not 2. So, by the rule, **0.65 cm does not go
  on**. Four of the ten runs are doubtful; only if all four fell in favour of 0.65 cm would it pass, by exactly 3.
- **By the rule, the model's own step length goes on**, but only just. Its four runs all kept the line, and 1.0 cm kept
  it in at most 1 of 5. That "at most 1" holds only if R05 and R18 count as lost. In R18 the puck was 12.4 mm off in one
  picture, against a limit of 12 mm. That is a coin toss, not a measurement.
- What the pictures do show clearly: in almost every run the puck swings 8 to 18 mm to the left at the first bend of
  sheet 03, whichever step is used. Whether it crosses 12 mm decides the verdict. The four own-length runs swung least
  (8 to 9 mm).
- The model trained without sheet 03 kept the line with 0.65 cm in all 3 runs and lost it with 1.0 cm in all 3.
- The team of five lost the line in all 4 runs, by 19 to 32 mm.
- Sheet 04: all 3 runs lost the line, at the first sharp right bend (63 to 72 mm along the line). In R30 and R31 the
  sheet lay on its faint copy within 1 to 4 mm in the picture, and they still failed. So the failure is not only the
  placement.
- Sheet 01: in all 7 runs the robot stopped with the puck 21 to 25 mm left of the line's end. None is "finished" by the
  17 mm rule. Starting as on 11.11 (X01 to X03) made no difference: those stops were also 23 to 25 mm left. Emranul's
  ruler said about 2 cm for the new start (this agrees) and 1 to 1.5 cm for the old start (this does not).
- Sheet 01 itself lay about 16 mm to one side of its recorded place in the picture, from R00 on.
- How sure: the measuring is good to about 2 to 3 mm (typical) and 4 to 7 mm (worst tenth). Runs that miss or meet the
  12 mm limit by more than that are clear. The doubtful ones are named below. No ruler photo has been compared yet.
  Emranul's phone photos could settle sheet 01's stops and the doubtful runs.

## How it was measured

- **The puck.** A cylinder of the puck's size (24 mm across, 29 mm tall) is fitted to the red blob through the forearm
  camera model. This gives the centre of the puck's foot on the table. The method is the one of 11.11
  (`robot_test_2026-11-11/stop_and_centring/picture_measures.py`, reused unchanged). The camera correction of 11.11 (picture
  shifted 9.9 px right, 2.7 px up, rolled 1.26 degrees) was tested again on the visit's own pictures and was still the
  best of 54 tried (`s01_camera_check.out.txt`) [measured].
- **The line: three measures of the offset** (signed, positive = left looking from end A to end B):
  - A: against the line voted by the last 12 pictures (the measure of 11.11, reused unchanged);
  - B: against the printed drawing, laid by a turn and a shift onto the line all of the run's pictures voted for;
  - C: against the printed drawing, laid onto the ink of that one picture alone. **C is the main measure.** An error of
    the camera model that the puck and the ink beside it share cancels out, and the drawing carries the line under the
    puck, where the camera cannot see it.
- **The geometry scored against: the traced redrawings** (`tracks_2026-11-12/course_sheets/C01, C03, C04_course.svg`),
  because those were printed for 12.11. As a check, the original drawings (Niryo1 to 6.svg) were laid the same way.
  On sheet 03 the traced drawing fits the voted ink better in all 24 runs (median 1.18 to 1.39 mm against 1.29 to
  1.54 mm), which supports the record. On sheets 01 and 04 the two fit equally well over the part the runs saw. No
  verdict changes with the original drawing [measured, `s03_verdicts.csv`, column `kept_b_original_drawing`].
- **How far along the line:** the puck's nearest point on the laid drawing, counted from end A's cap centre.
- **The rule's travel:** each picture stands for the tip travel from halfway since the last picture to halfway to the
  next (the middle reading). The shortest and longest readings are also computed. With 1 cm steps, one picture over 12 mm
  counts as about 10 mm of travel, so it sits right on the rule's own limit.

### The check of the measuring (point 3)

| Check | Result | Kind |
|---|---|---|
| C against A, inside the scoring windows (649 pictures) | median 3.0 mm apart, 90th percentile 6.8 mm, mean -0.2 mm; over 12 mm by one but not the other in 44 | measured |
| C against B (676 pictures) | median 2.3 mm, 90th percentile 4.2 mm, largest 7.2 mm; C reads 1.35 mm further left on average | measured |
| B against A (650 pictures) | median 1.9 mm, 90th percentile 5.4 mm | measured |
| Along the line, C against B | median 1.4 mm, 90th percentile 3.9 mm | measured |
| By eye, 36 pictures with the measured points drawn in (`s04_visual_check_tiles_1` to `_6.jpg`, values in `s04_visual_check_values.csv`; 24 drawn at random, 12 that decide verdicts) | The fitted foot circle sat on the puck in all 36. The drawn centre lines lay on the printed ink. B (green) often ran along the inner edge of the ink at the first bend of sheet 03, 3 to 5 mm off; C (magenta) lay nearer the middle of the ink. I could not read the offsets to better than about 3 to 5 mm by eye. Where the line comes near itself (sheet 03 after 200 mm, sheet 04's loops), A sometimes measured to the wrong stretch, with the wrong sign (for example R28 picture 47: A -13 mm, B +13 mm). Inside the scoring windows this did not happen. | looked at |
| Sheet 01 near the end, 4 pictures by eye (R02 pictures 20, 24, 26; X01 picture 22) | The line ahead is visible to the right of the puck, and the drawn line lies on its middle. The puck clearly drifts left as A and B say (5, 10, 15 and 12 mm). | looked at |
| Scale at the puck (millimetres per pixel) | 2.28 to 2.36 px per mm (0.43 mm per pixel), from the camera model and the taught table plane | calculated |
| How wrong the scale can be | With a free scale, the drawing fits the voted ink best at 0.97 to 0.98 of its size (sheet 03), 0.98 to 0.99 (sheet 04) and 0.996 to 1.02 (sheet 01). On 11.11 the puck measured 2.47 cm at half height against 2.4 cm. So the scale is good to about 3 percent: at most 0.4 mm on a 12 mm offset, and about 4 mm on the 119 mm mark. Whether the 2 to 3 percent comes from the printer or from the camera model cannot be told from here. The 100 mm bar page of the printer check was not measured into the records. | measured, cause guessed |
| Mark moved 10 mm earlier and later (119 → 109 / 129; 160 → 150 / 170) | No verdict changes. Every failure happens at a bend well before the mark (20 to 100 mm on sheet 03, 63 to 72 mm on sheet 04). | measured |

So the progress along the line is known to a few millimetres, much better than 1 cm, and the marks do not matter for
any verdict. The sideways offset is good to about 2 to 3 mm typically and 4 to 7 mm at worst. A verdict is called
**doubtful** when A, B, C, C's shortest or longest travel reading, or C with the limit at 10 or 14 mm (±2 mm) disagree.

## Sheet 03: all runs (R05 to R28)

Mark: 119 mm. All runs ended by their step budget; every one passed the mark. Offsets: largest within the window, C
first (A, B). "Left the line" is where the first stretch over 12 mm for 10 mm of travel began (C), over the whole run.

| Run | Model | Step | Verdict | Largest offset C (A, B), mm | Stretch over 12 mm (C) | Left the line at | Emranul's eye | Notes |
|---|---|---|---|---|---|---|---|---|
| R05 | E | 1.0 | **doubtful** (main: not kept) | +16.9 (12.9, 15.0) | 20 mm, pictures 5 to 6 | 24 mm | barely touching | A and B: kept, 9.9 mm of travel |
| R06 | E | 0.65 | **doubtful** (main: not kept) | +18.4 (12.4, 14.7) | 20 mm, pictures 9 to 11 | 27 mm | on or close | A: kept |
| R07 | without 03 | 1.0 | not kept | -22.1 (24.4, 24.7) | 40 mm | 97 mm | lost after 4 to 5 steps | |
| R08 | without 03 | 0.65 | kept | -6.7 (9.2, 9.3) | none | | best run, slow, on | |
| R09 | E | 0.65 | **doubtful** (main: kept) | +11.6 (6.8, 7.7) | none | | on or close | fails only with the limit at 10 mm |
| R10 | E | 1.0 | kept | +10.2 (8.8, 11.5) | none | | almost good | |
| R11 | E | 1.0 | not kept | -26.7 (23.6, 29.6) | 71 mm | 69 mm, to the right at the second bend | lost (cut across) | |
| R12 | E | 0.65 | not kept | +17.2 (13.2, 14.7) | 20 mm, pictures 8 to 10 | 23 mm | on or close | A and B 13 to 15 mm; C clearly over |
| R13 | without 03 | 1.0 | not kept | +25.7 (27.4, 26.4) | 60 mm | 29 mm | off early, then barely touching | |
| R14 | without 03 | 0.65 | kept | -8.1 (8.6, 10.8) | none | | mostly on | |
| R15 | E | 1.0 | not kept | +17.9 (26.6, 15.7) | 20 mm, pictures 7 to 8 | 19 mm | "a bit faster", line not reported | |
| R16 | E | 0.65 | kept | +9.1 (7.1, 7.0) | none | | on or close | |
| R17 | E | 0.65 | kept | +8.7 (6.6, 5.2) | none | | on or close | |
| R18 | E | 1.0 | **doubtful** (main: not kept) | +12.4 (12.5, 12.7) | 10.1 mm, picture 6 only | 33 mm | mostly on | one picture 0.4 mm over the limit |
| R19 | without 03 | 0.65 | kept | -6.2 (10.3, 8.3) | none | | on | |
| R20 | without 03 | 1.0 | not kept | +15.7 (20.1, 19.1) | 40 mm | 55 mm | about as its partner | disagrees with the eye |
| R21 | team of five | 1.0 | not kept | +23.8 (24.7, 23.9) | 110 mm | 71 mm | bad | |
| R22 | E | own | kept | +9.3 (5.6, 6.4) | none | 292 mm | almost whole track? | see below |
| R23 | team of five | 1.0 | not kept | +32.2 (33.5, 31.8) | 90 mm | 84 mm | bad | |
| R24 | E | own | kept | +8.9 (6.7, 5.8) | none | 278 mm | see below | |
| R25 | team of five | 1.0 | not kept | +18.8 (18.9, 17.0) | 90 mm | 33 mm | bad | |
| R26 | E | own | kept | +8.6 (5.7, 6.2) | none | 267 mm | see below | |
| R27 | team of five | 1.0 | not kept | +20.7 (20.2, 19.0) | 80 mm | 91 mm | bad | |
| R28 | E | own | kept | +8.0 (5.4, 7.0) | none | 252 mm | see below | |

Figure: `evidence/test_visit_2026-11-12/s08_pairs_offsets.png` shows model E's five pairs, offset against tip travel,
with all three measures.

**The own-length runs, R22 to R28 (90 steps, median step 0.55 cm at the first bend, 0.70 to 0.77 cm over the run).** All
four kept the line well past the mark: within 9 mm up to about 220 mm. All four then cut the right-hand corner at
226 to 264 mm. Right after it the puck stood 8 to 31 mm left of the line (R28 the worst: over 20 mm from 252 to 330 mm).
At the sharp left bend near 350 mm it then stood 17 to 22 mm to the right. The budget ended all four. R22, R24 and R26
had the puck at 450 to 466 mm of the 507 mm line (about 4 to 6 cm before the end), 11 to 19 mm left of the last straight
in the last picture. R28 ended at about 401 mm, back on the line. Emranul saw "once almost the whole track (3 to 4 cm
before the end), twice ended early on the left, once mostly off the line". The pictures say three of them got about as
far, and the one mostly off the line is most likely R28. All four count as "kept" for the rule up to 119 mm [measured;
the corner pictures 41 to 53 have no measure C, A or B, because the puck and the block hide the corner].

## The decisions, worded as the rules word them

**0.65 cm against 1.0 cm, model E's five pairs** (R05/R06, R09/R10, R11/R12, R15/R16, R17/R18):

| | Kept surely | Doubtful | Kept, main reading |
|---|---|---|---|
| 1.0 cm (R05, R10, R11, R15, R18) | 1 (R10) | 2 (R05, R18) | 1 |
| 0.65 cm (R06, R09, R12, R16, R17) | 2 (R16, R17) | 2 (R06, R09) | 3 (R09, R16, R17) |

"0.65 cm goes on only if it keeps the line in at least 3 more runs than 1.0 cm": the main reading gives 3 against 1,
2 more. **0.65 cm has not passed.** If every doubtful run fell for 0.65 cm the difference would be 4 against 1, exactly
3. If they all fell against it, the difference would be -1. Whichever way, the pairs are close: in four of the five
pairs both runs swung 9 to 18 mm left at the same bend.

**The own length** (rule: "If 0.65 cm has not passed, the own length goes on only if it keeps the line in all 4 runs
while 1.0 cm keeps it in at most 1 of 5"): the own length kept the line in 4 of 4 (all four sure). 1.0 cm kept it in 1
of 5 by the main reading (R10), or in up to 3 if R05 and R18 count as kept. **By the main reading the own length goes
on; this rests on R05 and R18 counting as lost.** R18 was over the limit in one picture by 0.4 mm, which is inside the
measuring error. A and B would count R05 and R18 as kept, and then the own length would not go on. I would not act on
this decision without a look at the phone photos or a repeat of the 1.0 cm runs.

**The model that never saw sheet 03** (decides nothing): 1.0 cm lost the line in 3 of 3 (R07, R13, R20); 0.65 cm kept it
in 3 of 3 (R08, R14, R19). All six are sure.

**The team of five** (decides nothing): lost the line in 4 of 4 (R21, R23, R25, R27), all sure, by 19 to 32 mm.

**Sheet 04** (mark 160 mm; "3 of 3: mostly start and placement; 0 of 3: not the placement"): **0 of 3 kept the line**,
all sure. R29 and R30 passed the mark; R31's budget ended before it ("window not completed"), but it had already lost
the line by then. All three left the line at the first sharp right bend, 63 to 72 mm along the line (pictures 12 to 13),
by 25 to 64 mm. Emranul saw "lost after 4 to 5 steps". The pictures put the clear departure at steps 12 to 13, at the
bend. Because R29 was misplaced, the count that speaks to placement is R30 and R31: both well placed, both lost.

| Run | Verdict | Largest offset C (A, B), mm | Left the line at | Sheet against its faint copy (in the picture) |
|---|---|---|---|---|
| R29 | not kept | -53.9 (32.8, 52.0) | 72 mm | 29 to 32 mm apart near the puck, turned (about 10 degrees by eye) |
| R30 | not kept | -28.2 (27.7, 27.0) | 63 mm | 1 to 2 mm |
| R31 | not kept, window not completed | -64.3 (25.6, 63.7) | 64 mm | about 4 mm; but the arm stood 2.04 cm from the recorded start |

## Sheet 01: where the puck stopped (R00, R02 to R04, X01 to X03)

The robot stopped by itself in all seven (28 to 29 pictures with the new start, 26 as on 11.11). Along: positive = the
puck's centre short of the end's centre, negative = beyond it. Left: of the line, looking from end A to end B. Gap: the
distance between the two centres. B is the drawing laid on the whole run, A is the end found in the voted map (11.11's
end_cover). In the stop picture the end itself is hidden by the puck and the block; it is placed from the earlier
pictures.

| Run | Start | Left, B (A), mm | Along, B (A), mm | Gap, B (A), mm | 17 mm | 12 mm | 7 mm |
|---|---|---|---|---|---|---|---|
| R00 (warm-up, not scored) | new | 23.1 (23.0) | -5.1 (-8.7) | 23.7 (24.6) | no | no | no |
| R02 | new | 21.3 (20.4) | +1.8 (-9.9) | 21.4 (22.7) | no | no | no |
| R03 | new | 24.6 (24.5) | -9.1 (-10.2) | 26.2 (26.5) | no | no | no |
| R04 | new | 22.4 (22.0) | -10.4 (-12.8) | 24.7 (25.4) | no | no | no |
| X01 | as on 11.11 | 23.1 (21.0) | -7.2 (-14.2) | 24.2 (25.4) | no | no | no |
| X02 | as on 11.11 | 24.7 (23.7) | -10.6 (-11.7) | 26.9 (26.4) | no | no | no |
| X03 | as on 11.11 | 24.7 (24.0) | -10.2 (-11.4) | 26.8 (26.6) | no | no | no |

- **Finished in Part 1 (R02 to R04): 0 of 3** by the 17 mm rule, by both methods; also 0 of 3 by eye at the robot.
  X01 to X03: 0 of 3.
- The two methods agree on "left" within 2 mm. "Along" is weaker: they differ by up to 12 mm, and both put most stops
  past the end's centre. So the robot did not stop short; it stopped beside the end.
- The puck drifts left steadily from about 130 mm along the line (0 to 3 mm) to 12 mm at about 205 to 215 mm and 21 to
  25 mm at the stop. I also looked at it in four pictures (see the check of the measuring).
- Emranul's ruler: about 2 cm left for the new start (agrees, 21 to 25 mm). For the old start, 1 to 1.5 cm (does not
  agree: 23 to 25 mm). If he measured from the puck's edge, or to the line's edge, the numbers would differ by 5 to
  12 mm. The phone photos would settle it.
- The 11.11 runs on sheet 01 had stopped 13.2 and 21.7 mm from the end's centre (run 2 and run 1, check_results_complete.md).

## Was the sheet laid as in the recording? (point 4)

Compared: the run's first picture against the reference picture of the run list, each with its own joints and the
camera correction of its day. The traced drawing was laid onto the line each picture shows, and the gap was measured
square to the line at 0, 20, 40 and 60 mm from end A (`s05_lay_out_against_ghost.csv`, overlays
`s05_lay_out_<run>.jpg`: the live line in green, the faint one in magenta).

| Run | Arm from recorded start | Gap in the picture, 20 / 40 / 60 mm | In mm at the puck (2.3 px per mm) | What the overlay shows |
|---|---|---|---|---|
| R00 (sheet 01) | 0.10 cm | 39 / 36 / 37 px | about 16 mm | the two lines side by side and parallel, not overlapping; the paper edges doubled as well |
| R05 (sheet 03) | 0.61 cm | 18 / 17 / 9 px | 4 to 8 mm | overlapping near the puck, apart further out |
| R29 (sheet 04) | 0.41 cm | 74 / 38 / 10 px | up to about 32 mm | clearly off and turned (Claude saw about 10 degrees at the visit) |
| R30 (sheet 04) | 0.87 cm | 3 / 3 / 2 px | 1 to 2 mm | on its faint copy |
| R31 (sheet 04) | 2.04 cm | 9 / 9 / 9 px | about 4 mm | on its faint copy, but the arm was 2 cm from the recorded start |

Between the runs (the drawing laid on each run, `s03_lay_out.csv`): sheet 03 stayed put through all 24 runs. End A
moved by at most about 4 mm, and the turn changed by less than 2 degrees, alternating with the arm's pose, not
drifting. Sheet 01 stayed within about 4 mm. Sheet 04 moved 37 mm and 11 degrees between R29 and R30 (Emranul turned
it), and again 19 mm and 3 degrees between R30 and R31.

The same comparison on the table (instead of in the picture) gave gaps of 15 to 38 mm, even where the pictures matched
within 2 mm. It mixes the arm's offset with the difference between the camera corrections of 8.11 and 12.11, so I do
not use it. The table turns it gave (R05 11 degrees, R29 -8, R30 4) are not trusted either.

## R01 and the loop demo

- **R01** (no puck, own length, 40 steps at most): 19 moves of 0.30 to 0.55 cm, then refused at picture 20 ("Command
  MOVE KO ... Error Code: -3"). The tip went 7.6 cm to the robot's right and 2 cm back towards the base (x from 0.141 to
  0.122 m). That is close to the driver's smallest reach of 0.12 m, which is most likely why the robot refused [guess].
  Without a puck nothing was scored. Emranul saw the small steps as smooth.
- **Demo** (model E, sheet 05, start as on 11.11, 1.0 cm): it ran from the traced file's other end (the drawing fits
  only that way round). It stopped by itself after 88 pictures, with the puck at 563 mm of the 566 mm line, 3 to 5 mm to
  the right [measured, B]. On the way the puck stood 12 to 15 mm left of the line from about 158 to 230 mm (C; 10 to
  12 mm by A). By the 12 mm rule that stretch would count as off the line. No rule applies to the demo. Emranul saw "whole
  loop, stopped by itself"; that agrees.

## Doubts, all in one place

1. R05, R06, R09 and R18 are doubtful. The main reading is C. A and B count R05 and R18 as kept and R06 near the limit.
   Both decisions of Part 2 move with these runs.
2. The rule's "10 mm of tip travel" cannot be resolved with 1 cm steps. One picture over 12 mm is about 10 mm by the
   middle reading, 0 by the shortest and 20 by the longest. R05 (C: two pictures) and R18 (one picture) sit on this edge.
3. Choosing C as the main measure is reasoned, and supported by the overlays (C lies nearer the middle of the ink at the
   first bend than B). It is not proven against a ruler. C reads 1 to 2 mm further left than A and B on average. At the
   first bend, where the puck swings left, that makes C stricter.
4. The scale is uncertain by about 3 percent (printer or camera model); this changes no verdict.
5. No measure exists for the corner pictures of the own-length runs (226 to 264 mm). They lie after the mark.
6. Sheet 01's stop is placed from pictures before the stop; at the stop the end is hidden. "Along" differs by up to 12 mm
   between the methods; "left" agrees within 2 mm.
7. The table-frame comparison of the lay-out is unreliable (see above); only the picture comparison is used.
8. R31 started 2.04 cm from the recorded start (over the 1 cm warning), and its sheet had moved 19 mm since R30.
9. Emranul's phone photos with the ruler were not available to me. They are the independent check that is still
   missing, above all for sheet 01's stops and for R05, R06, R12 and R18.

## Files

In `evidence/test_visit_2026-11-12/`: `visit_common.py` (shared pieces), `s01_camera_check.py` (camera correction), `s02_measure_runs.py`
(puck and measure A), `s03_lay_drawing_and_score.py` (drawing laid, measure B, first verdicts), `s04_visual_check.py`
(pictures with the points drawn in), `s05_lay_out_against_ghost.py` (sheet against the faint picture),
`s06_sheet01_ends.py` (sheet 01 stops), `s07_single_picture_lay.py` (measure C), `s08_final_verdicts.py` (verdicts,
agreement, decisions, figure). Each has its `.out.txt`. The CSV files hold the numbers per picture (`s03_steps.csv`,
`s07_steps_method_c.csv`) and per run (`s08_verdicts.csv`, `s06_sheet01_ends.csv`, `s05_lay_out_against_ghost.csv`,
`s03_lay_out.csv`). The scripts import the 11.11 measures from `evidence/robot_test_2026-11-11/stop_and_centring/`
unchanged. The large vote maps are in the scratch folder only, and s02 makes them again in about half a minute.
