---
name: project-first-robot-test
description: First robot test with the model mode on 11.11.26 - 10 runs, 4 reached the end and stopped (2 on the end by the 12 mm rule), 6 lost the line; one confirmed mechanism (fixed 1 cm step) and four weaknesses; sheet 04 and the dashed loop still open; the exam case (unseen line) never driven; plan version 2 with 11 decisions
metadata:
  type: project
---

11.11.26, Emranul at the robot, model mode (`02_code/model_mode.py` in the worktree `.claude/worktrees/training-analysis`,
not committed), logs in `03_data/model_runs/2026-11-11_17*` (pictures and `steps.csv` per run). Every run used a model
trained on the sheet it drove.

| Run | Model, sheet | Result (read from the pictures) |
|---|---|---|
| 1 | A (model 1), 01 | end reached, stopped by itself 21 mm right of the end; the drift began at about picture 12 |
| 2 | E (model 4: chunk, stop weighted 25), 01 | on the line, stopped by itself 13 mm left of the end |
| 3 | E, 02 lying square | lost within 5 pictures, ended by hand |
| 4 | E, 02 as in plan 02-01 | finished; puck 18 to 21 mm off at pictures 21 to 29, then back |
| 5, 9 | E, 03 | tip 2 to 3.5 cm too wide at the first bend, puck lost at picture 17 to 18 (9 = without the hoodie) |
| 10 | B (team of 5), 03 | passed the first bend, lost at the next turn |
| 6 | E, 04 | wrong direction from picture 8, about 30 steps back and forth, then across the lines |
| 7 | E, 05 | whole loop with the crossing, 88 steps, stopped by itself; puck over 1 cm off the line in 24 of 89 pictures |
| 8 | E, 06 | followed the dashes into the loop, left the tight bend at pictures 23 to 30, false stop on blank paper |

By the 12 mm rule two stops were on the end (runs 4 and 7). Nine of ten runs were ended by the model's own stop; in
five of the six lost runs it fired at a wrong place.

**What the check found** (eight analyses, each recomputed by an independent checker;
`06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/check_results_complete.md`), as
corrected by the review of the plan (`.../plan_review/`):

1. Confirmed mechanism: the driver moved a fixed 1.0 cm where the model asked a median 0.61 cm (2.1 to 2.4 times the
   asked distance at sheet 03's first bend). Likely cause of runs 5 and 9, not proven. On unseen sheets the model's
   own length is almost a plain 0.67 cm; it would make the stop come about 0.8 cm earlier and costs 1.3 to 2 times
   the pictures.
2. The model does not turn too little as a habit: where it asks for a sharp turn it asks the right amount; on an
   unseen sheet it is unsure where sharp turns come (correlation 0.71 over ten steps). No twist factor.
3. The start is shown as a cause for run 3 only (block touching the puck, line leaving 53 degrees to the side).
   149 of 210 recordings begin from one arm pose, puck a median 5.5 cm ahead; the runs began 2.3 to 3.6 cm from the
   puck with the forearm turned up to 10 degrees.
4. Sheet 04 (run 6) is open: the sheet lay 15 to 25 degrees off its recorded place; on the recorded pictures the same
   model asks for the full recorded turn. A repeat at the recorded place decides.
5. Demonstrations differ at the same place (four ways at the square start, two through sheet 04's first S-bend) and
   are not centred (within 7 mm in 68 % of pushed pictures, ends covered in 52 to 60 %). In the loops all recordings
   walk round the puck. The camera cannot look along about 22 % of the recorded way (line running back to the robot).
6. The exam case was never driven. Models trained without a sheet exist (`without_path_0X.pt`) and run through the
   driver. Offline on unseen sheets the stop is right in 127, early in 27, never in 47 of 209 episodes (last picture
   counted); weakest on sheets 02, 05, 06. Run 7's loop is probably learnt by heart.

Ruled out for the six failures: late pictures, the hoodie, the camera's shift (it moved once, on 8.11.26, about 16 px
of 640), unfamiliar pictures before the first wrong move. Open: dashes against shape, the held wrist for the missing
walk in run 8. Also: the puck is 2.4 cm across (not 4); the practice tracks of 10.11 are too easy and are redrawn.

Emranul's requirements: [[project-stop-must-cover-line-end]]; the exam rules he reported: [[project-exam-rules-reported]].

**The plan** (version 2, after four critics and their checkers):
`06_deliverables/research_notes/Model and training/2026-11-11_plan_after_first_robot_test.md`. Three short test
sittings (A: small steps, the measure, the step at sheet 03's bend, sheet 04 again; B: the loop, lines never seen, the
driver decision; C: the stop, dashes, orientation), version 1 frozen early as the fallback, then a pilot and a
recording round on 18 to 24 new shapes in one written style, models judged by a panel fixed in advance, a paired test,
corrections, a final test in exam form. About 35 to 45 robot hours until 20.12.26. Eleven decisions go to Emranul one
at a time; the first three: a safety copy and committing, the exam lay-out rule, go for sitting A.

State on 12.11.26 (night): two proofreaders found about fifty corrections to version 2; they are saved in
`.../plan_review/proofread_complete.md` and not yet worked into the plan (a note at its top says so). Emranul allowed a
local commit: three commits on `emranul/training-analysis` in the worktree (b0a12df code, e19415b notes and plan,
164ee79 pictures), not pushed; the safety copy to a second disk is still open. Emranul's dates: at the robot on
Thursday 12.11 for half a day, recording with Marshal on Saturday 14.11 for at least six hours. So Thursday is a
first test visit with the driver as it is plus small switches (fixed step length from the command line, stall rule
logging only, the start's offset from the recorded start, a placing aid, a faint reference picture; checklist
`2026-11-12_test_visit_checklist.md` with its run list), and Saturday is the hand trial of the demonstration rules,
a pilot (4 new prints, each recorded by both) and a first batch on new shapes. Still to build for Saturday: the new
tracks as true-size prints, their entries in the recording program (which runs from the main checkout, so the
branch has to be brought over there), the one-page rules.
