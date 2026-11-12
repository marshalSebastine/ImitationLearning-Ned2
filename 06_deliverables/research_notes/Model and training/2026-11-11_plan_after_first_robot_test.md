# Plan after the first robot test (11.11.26)

Version 2, written late on 11.11.26 by the Model and training session for Emranul and Marshal. Version 1 (20:47) was
attacked by four independent critics; a second agent then tested each of their 59 objections against the files. None
fell and about half were narrowed. This version takes the corrected fixes and the eight exam rules Emranul reported the
same evening. Nothing in it is decided: the decisions are in section 7 and go to Emranul one at a time. Nothing here
runs on the robot without Emranul's go and a written checklist.

**State on 12.11.26, read this first.** Two readers proofread this version in the night and found about fifty
corrections; they are not yet worked into the text below (all of them, with the exact old and new wording, are in
`evidence/robot_test_2026-11-11/plan_review/proofread_complete.md`). The three that matter most: the tools for the
full sitting A take three to five days, not one to two; sitting A is too much for one visit and is cut in two; and
"the puck ahead of the tip as in the recordings" differs by sheet (about 5.5 cm on sheets 01 and 04, about 3 cm on
sheet 03). The dates also changed, on Emranul's word of 12.11.26: he is at the robot on Thursday 12.11 for half a day
and records with Marshal on Saturday 14.11 for at least six hours. So Thursday is a first test visit with the driver
as it is plus two small switches (its checklist: `2026-11-12_test_visit_checklist.md`), and Saturday is the hand
trial, the pilot and a first batch of new shapes. Sittings B and C and the tools of the full sitting A follow in the
week after; they do not block the recording. This session's work was committed on Emranul's laptop on 12.11.26
(three commits on `emranul/training-analysis`, not pushed).

Words in square brackets say how sure a statement is: [measured] from logs, pictures or data; [calculated] on paper
from logged or recorded pictures; [guess]; [judgement].

Where the evidence is, all under `evidence/robot_test_2026-11-11/`: `check_results_complete.md` (the check of the ten
runs: eight analyses, each recomputed by an independent checker) and `plan_review/<lens>/` (the four critics' scripts
and outputs, the checkers' in `check/`).

## This week

| What | Who | Needs |
|---|---|---|
| A safety copy of the recordings, the ten runs, the models and this session's code; whether to commit | Emranul decides (decision 1) | a second disk |
| The rule for how a sheet is laid at the exam | Emranul and Marshal (decision 2) | ten minutes together |
| Tools and checklist for test sitting A | this session, after Emranul's go (decision 3) | one to two days, no robot |
| Test sitting A at the robot, about 2.5 hours | Emranul, with the checklist | sheets 01, 03, 04 and one print of our own |

## 1. In short

1. On 11.11.26 the robot drove ten runs. Four brought the puck to the end of the line and stopped by themselves
   (sheet 01 twice, sheet 02, sheet 05); by the 12 mm rule two of them stopped on the end (runs 4 and 7). Six lost the
   line (sheet 02 lying square, sheet 03 three times, sheet 04, sheet 06); five of those six also stopped by
   themselves, at a wrong place. Every run used a model that had been trained on the sheet it drove. [measured]
2. The check confirmed one mechanism (the driver moved 1.0 cm where the model asked a median 0.61 cm) and four more
   weaknesses: how the model follows a turn on a line it has not seen, starts unlike the recorded ones,
   demonstrations that differ at the same place and are not centred, and an exam case that was never driven. It found
   the cause of one failure, by calculation: run 3, the touching start. For runs 5 and 9 the fixed step is the likely
   cause, not a proven one; for run 6 the start, the placement and the model's fit are open; for run 8 dashes, shape
   and the held wrist are open. The order of the weaknesses is this session's judgement.
3. Ruled out for all six failures [measured]: late pictures, the hoodie, the camera's shift, unfamiliar pictures before
   the first wrong move. Not ruled out: the held wrist for the missing walk in run 8, where the sheet lay in run 6,
   and shadows (not measured).
4. The exam, as Emranul reported it on 11.11.26: own printed tracks may be trained on; an unseen track can be
   anything; properly finished first, then time, ranked against the other teams; the robot must stop by itself; one
   try per track; the team lays every sheet; unseen tracks are seen only on the day (section 2).
5. The plan: three short test sittings that each answer named questions against a control (the driver's step and the
   measure; the loop, lines never seen and the driver decision; the stop, dashes and orientation). Then a first
   version is frozen as the fallback. Then the bet with the best support for an unseen line: more different tracks,
   recorded in one written style, starting with a small pilot. Models are judged by a small panel fixed in advance, on
   shapes they were not trained on. Then one comparison of the new version against the frozen one, corrections, and a
   final test in exam form: tracks nobody has seen, one run each.
6. Nothing is judged on the six course sheets alone, and no rule is taken from the ten runs of 11.11 without a test
   on a line the model has not trained on.

## 2. The exam, as reported

Emranul's answers of 11.11.26; he confirmed that the first four are the professor's own words. They are not yet in
`05_meetings/requests_to_professor.md`, which still lists the first as open (the orchestrator's file).

| Reported | What it changes here |
|---|---|
| Own printed tracks may be used for training | the recording round on new shapes is open; the print set serves tests and training |
| An unseen track can be anything | dashes, loops, crossings and hairpins on a line never seen are certain, not a risk |
| Properly finished first, then time; ranked against the other teams | finishing first; seconds are logged from now on and tuned last (Emranul's own priority, 11.11.26) |
| What the model lacks may be trained | correction rounds and recoveries are allowed |
| The robot must stop by itself | the stop is part of finishing; a run that never stops, or stops at a wrong place, fails |
| The team lays every sheet and sets the start | a written lay-out rule can remove most of the orientation risk at the exam (decision 2) |
| One try per track | reliability over many first runs counts, not the best run |
| Unseen tracks are seen only at the exam | the final test must be tracks nobody has seen, one run each, laid by the rule against a clock |

Still not known: how close to the line's end the puck must be; a time limit; which of the six sheets and how many
unseen tracks are run, where and under which light; whether the driver may turn the model's answer into moves with
fixed rules; whether the start may be set by hand or by one fixed move (section 8).

## 3. What the check found

### 3.1 Run by run

| Run | Model and sheet | What the pictures show [measured] | Cause, as far as it is verified |
|---|---|---|---|
| 1 | A, sheet 01 | reached the end and stopped by itself, 21 mm to the right of the end's centre, clear of the ink; drifting from about picture 12 | not settled; the camera's shift explains at most half [calculated]; sheet 01's own recordings end sideways too (5 of 18 ends covered) |
| 2 | E, sheet 01 | on the line all the way (median 1 mm), stopped 13 mm left of the end | the last three answers pointed 5, 16 and 27 degrees away from the line, and the fixed step made them 1.1, 1.7 and 4 times as long |
| 3 | E, sheet 02 lying square | lost within five pictures; the run was ended by hand | the start, by calculation: the block touched the puck while the line left 53 degrees to the left. The six recordings of this lay-out start about 6 cm back and go round the puck in at least four ways. Not the step length |
| 4 | E, sheet 02 as recorded | finished; more than 12 mm off the line's middle from about picture 21 to 29 (18 to 21 mm at the peak), then back | the stop at the end may have leaned on the hoodie in view (stop chance 0.90; 0.22 to 0.65 with it painted over) |
| 5 and 9 | E, sheet 03 | tip 2.0 to 3.5 cm too wide at the first bend; puck off the ink from picture 17 or 18; false stop | the driver moved 2.1 to 2.4 times the distance asked for. Likely, not proven: only a robot test decides |
| 10 | B (team of five), sheet 03 | passed the first bend with the same driver, lost at the next turn (picture 30 to 37); false stop | on the same pictures the team points 15 to 17 degrees further into the bend than E and asks 19 to 25 % more twist [calculated] |
| 6 | E, sheet 04 | wrong direction from picture 8, corner cut at 10 to 12, back and forth from 19 to 50, then across the lines; false stop | open. The sheet lay 15 to 25 degrees off its recorded place and the start 5 cm off. In that state the model asked for 6.8 degrees of twist; on the recorded pictures of the same bend it asks for the full recorded turn (73.5 of 75.4 degrees) [calculated]. The back and forth was the fixed step jumping over a point the model steered to (asked 0.15 cm, moved 1 cm). Only a repeat decides |
| 7 | E, sheet 05 | finished the loop with the crossing; puck more than 1 cm off the line in 24 of 89 pictures | made the recorded walk round the puck, carried by the stretched steps (asked 2.3 cm over nine steps, driven 9 cm). A model that never saw sheet 05 answers differently on the same pictures (correlation 0.11): probably learnt by heart [calculated, never driven] |
| 8 | E, sheet 06 (dashed) | followed the dashes through the crossing into the loop; left the tight bend about 1 cm wide at pictures 23 to 30; no walk round the puck; stopped on blank paper | open between dashes, shape and the held wrist (for the missing walk). E pushed outwards there (models that never saw sheet 06 do not); none of four models asks for the walk on these pictures |

The stops [measured, no ruler check yet]: by the 12 mm rule in `config.py` two of the four stops are on the end (runs
4 and 7). By Emranul's rule (the end completely covered) only run 7 passes the plain 7 mm rule; by the puck's rim runs
4 and 7 both do, because a puck straight beyond the end hides it up to 11 mm. No stop was short: all four stood 1 to
9 mm past the end's centre; the misses were sideways. Nine of the ten runs were ended by the model's own stop; in five
of the six lost runs it fired with the puck off the line, on blank paper or beside a line end. "Stopped by itself"
counts only together with the place.

### 3.2 The weaknesses

1. **The driver's fixed step** [measured mechanism]. The model asked for a median of 0.61 cm per picture and the
   driver always moved 1.0 cm. Limits of the fix: on sheets a model has not seen its asked length is nearly the same
   at bends (0.63 cm) as on straight parts (0.70 cm), so "the model's own length" behaves almost like a plain fixed
   0.67 cm [measured offline]. It makes the stop come about 0.8 cm earlier, about 5 mm short at the median
   [calculated]. The one loop that worked (run 7) used the stretched steps. With model E's own lengths the tip may cut
   sheet 03's first bend on the inside (3.8 to 5.3 cm of advance against 6.0 to 8.9 cm recorded) [calculated]. And
   it costs time: the model's own length needs 1.2 to 1.3 times the pictures on easy sheets and about twice on
   sheet 03, a fixed 0.65 cm 1.5 times. So the step rule is tested against controls, not adopted.
2. **How the model follows a turn.** On a sheet it has not seen, its twist follows the recordings only loosely (over
   ten steps a correlation of 0.71) [measured offline]. Where it asks for a sharp turn it asks about the right amount
   (the recordings show 0.94 to 1.00 of it); what it lacks is knowing where the sharp turns come. On the recorded
   pictures of sheets it trained on it asks for about the full recorded turn. So nothing is to be scaled up. Where
   recordings differ at the same place it answers with a short step; offline such a step points in no clear
   direction, in run 6 it pointed back to one balance point. Even on pictures it trained on, its direction is more
   than 45 degrees off in 23 % of the moving steps. Which model: at sheet 04's first bend and in the loop of sheet 06
   all four models fail alike on the logged pictures; at sheet 03's first bend they differ (row of run 10). Whether
   the team's recipe is better at sharp bends is open and is tested in sitting A.
3. **The start.** Shown as a cause for run 3 only. Elsewhere the offsets are measured but not shown to matter (runs 4
   and 7 started 3.8 and 4.0 cm off and finished; run 9 started 1.3 cm off and failed). The facts: 149 of the 210
   recordings begin from one arm pose with the forearm straight (148 of the 209 trained on); the puck lies a median
   5.5 cm ahead of the tip. The ten runs began with the block touching or nearly touching the puck (2.3 to 3.6 cm),
   the forearm turned by up to 10 degrees (the clearest difference), and where it can be measured the sheet about
   2 cm off its recorded place. The checklist for 11.11 (written by this session) said to put the tip on the line's
   start; that did not match the recordings.
4. **The demonstrations.** Different ways at the same place: four ways round the puck at the square start, two ways
   through sheet 04's first S-bend (8 of 11 recordings turn 60 degrees within 9 to 11.5 cm, 3 do not). In the loops
   all recordings walk round the puck, but often with the puck beside the line. Not centred: while the puck is being
   pushed its centre is within 7 mm of the line's middle in 68 % of the pictures and more than 12 mm off in 16 %; a
   bit over half of the line ends are covered (52 to 60 %); in the loops the puck's centre is on the ink in 45 to
   58 % of the pictures [measured on samples]. A model that copies can average out a shaky hand (run 2 was better
   centred than its recordings), but it cannot remove a lean that all recordings share. One limit is built into the
   robot: the camera turns with the forearm, about 110 degrees to each side, so where the line runs back towards the
   robot it cannot look along the line. That is 22 % of the recorded way, and there the models follow the recordings
   far worse (correlation 0.37 against 0.75) [measured offline].
5. **The exam case was never driven, and the stop is weakest there.** Models trained without a sheet exist for every
   sheet (`04_runs/gpu_ki014/2026-11-10_resnet18_chunk_stop_weight_25/checkpoints/without_path_0X.pt`); they load and
   run through the driver on the simulated robot [measured]. Offline, as unseen sheets, the stop comes at the right
   place, never, or too early in: sheet 01 16, 1, 3; sheet 03 34, 8, 3; sheet 04 30, 6, 9; sheet 02 12, 29, 4;
   sheet 05 6, 12, 11; sheet 06 8, 15, 2 episodes [measured]. Part of "never" is the scoring: 24 of those 71
   episodes fire on the recording's last picture, where no second picture exists to confirm; counted as stops the
   totals are 127 right, 27 early, 8 in between and 47 never of 209. The stop is the weak point on sheets 02, 05 and
   06 as unseen sheets.

Open: **the dashes.** On demonstration pictures dashes cost a few points. On the robot's own pictures they change
model E's answers a lot: dashes cut into run 7's pictures take away 62 to 70 % of the twist in the walk (a control
repaint: 12 %) [calculated on edited pictures]. Sheets 05 and 06 differ in shape, size and turning side as well.

### 3.3 Ruled out, and not

- Ruled out for all six failures [measured]: late or repeated pictures (none in 544 pairs); the blue hoodie (it came
  into view after the first wrong moves; but see run 4's stop); the camera's shift (it moved once on its tape on
  8.11.26, between plans 04-24 and 04-30, by about 15 to 17 px of 640; undoing it changes the direction by a median
  of 1 to 4 degrees and leaves the twist asked at the first bends unchanged; do not touch the camera); unfamiliar
  pictures before the first wrong move (the pictures became unfamiliar because the tip had swung wide).
- Inside the range, without being the cause: all day the robot's pictures sat in the upper half of the recordings'
  range, in good and failed runs alike; 60 of 554 pictures were brighter or darker than 98 % of the training
  pictures, most of them in runs that finished.
- The held wrist: ruled out for the first bends of sheets 03 and 04 and for the place where run 8 left the line; not
  for the missing walk in run 8 (the recordings have the wrist at 22 to 28 degrees there, the driver holds 13.5).
- Not ruled out: where the sheet lay in run 6; shadows.

### 3.4 Corrections to what this session said earlier

- The puck is 2.4 cm across and 2.9 cm tall, not 4 cm. Run 1 did not start beside the line. Run 6 went wrong at
  picture 8, not after 15 steps. Run 10 did not fail at the first bend.
- Version 1 of this plan said "170 of 210" recordings start from one pose; the files say 149.
- Told to Emranul on 11.11 and changed by the critics' checks: sheet 04's failure is open between placement and the
  model's fit (not shown to come from two ways through the bend); the stop on unseen sheets is "never" in 47 of 209
  episodes when the last picture is counted, not 71; the model does not turn too little as a habit, it is unsure
  where the sharp turns come; the first recording batch of new shapes needs 9 to 10 robot hours, not 6.
- The practice tracks drawn on 10.11 are too easy (tightest bend 3.5 cm radius against 0.8 cm on sheets 03 and 04,
  1.3 cm on sheet 06, 2.2 cm on sheet 05; dashes of 35 and 15 mm against sheet 06's 20 and 10 mm), and `tracks.json`
  lists 7 of the 20 tracks. They are drawn again before anything is printed.

## 4. What we aim for, and how it is measured

| Aim | Measure |
|---|---|
| The puck reaches the end and the robot stops there by itself, on the first run | how the run ended (stopped, ended by hand, stalled, step budget, refused) and where: centres of puck and line end at most 12 mm apart (`config.py`) |
| The puck covers the end completely (Emranul) | the same centres at most 7 mm apart when the puck is short of or beside the end, up to 11 mm when it is straight beyond it |
| The puck stays in the middle of the line (Emranul) | share of pictures with the puck's centre within 7 mm of the centre line; three states: on the line, left and came back (how many pictures), lost |
| The six provided sheets, graded as they are | the same numbers with the model trained on all sheets, each sheet laid by the exam rule from the end we will use |
| An unseen line (the exam) | the same numbers on printed tracks no model was trained on, counted on the first run on each track |
| However the sheet lies (Emranul) | the same numbers on lay-outs that were never recorded; for the exam the lay-out rule of decision 2 comes first |
| Time (second criterion) | seconds per finished run, logged from now on; among settings that finish equally often the faster wins; tuned last |

The measure today [measured]: two independent picture measures agree to a median of 1 to 2 mm on sheets 01 and 02, at
the four stops and on run 8's puck path, but they share the fitted camera. On sheets 03 to 06 the measure's own two
variants differ by a median of 3 to 8 mm and disagree about the 12 mm limit in 15 to 46 % of the pictures; it fails
inside loops and on dashes. Until it has passed the checks of sitting A it is a help, not a judge. Two limits are
hard: the driver ends a run when the rod would lean more than 58 degrees (about 47.5 cm of reach with a straight
forearm), and the camera cannot look along a line that runs back to the base. No target percentages are promised:
the sittings measure where we stand first.

## 5. The plan

### Step 0. Now, without the robot

- A safety copy and the question of committing (decision 1): the driver, the analysis, the plan and the ten runs
  exist in one folder on one laptop; the recordings also lie on ki-014. Nothing has been committed since 10.11, on
  Emranul's instruction, so Marshal sees none of it and "rules fixed in advance" cannot be shown. The repository is
  public, so what a push would publish is looked at first (section 11).
- The exam lay-out rule (decision 2).
- The eight reported answers go into `05_meetings/requests_to_professor.md` (the orchestrator's file), with the day
  Emranul got them.

### Step 1. Tools before sitting A (this session, one to two days, no robot)

1. **A scoring command**, run on a run's folder afterwards and kept out of the driver. It measures the puck's
   distance to the sheet's traced centre line laid onto the run's own pictures (the traced lines exist:
   `Recording tool/2026-11-06_sheet_measurements.json`), so that "the bend" and "8 cm past it" are places on the
   line and dashed prints of our own can be scored. It reports sizes only, pushed pictures apart from standing ones,
   the three states of section 4, how and where the run ended, the seconds, and a sheet of small pictures. A run
   whose ink lies more than 2.5 mm (median) from the laid-on line is not scored by number. Acceptance test before it
   judges anything: on the logs of 11.11 it must say that runs 5 and 9 fail the first bend of sheet 03, run 10
   passes it (the method does [measured by the checker]) and runs 2 and 4 are not lost.
2. **Driver switches**, each with a test without the robot: the fixed step's length from the command line; a
   tip-travel budget that ends short runs by itself; a start move to the recorded start joints, made as the recording
   tool makes it (lift, travel 5 cm above, straight down); after a run the arm lifts, waits 5 cm above the start and
   comes down only on Enter; a snapshot that saves a picture and moves nothing; a fixed list of answers through the
   normal loop for the small-step check; the offset from the recorded start printed and logged; a camera check
   against a new reference picture saved on the first day (warn above 4 px, stop above 10 px); an Esc key that also
   works on Marshal's Mac; the driver's own check extended to steps of 1 to 5 mm.
3. **The stall rule counts path, not pictures:** a run is stalled when the pictures holding the last 15 cm of the
   tip's own path (at least 15) stay within 4 cm, or the last 40 pictures within 1.5 cm, each with under 15 degrees
   of forearm turn. On paper it ends run 6 at picture 33, no other run of 11.11 and none of the 209 recordings; the
   rule of version 1 (15 pictures) would have ended 16 to 132 of 209 healthy walks once the steps are short
   [calculated]. In the test sittings it only logs and a person ends the run; at the exam it stays loose, because it
   can only save time on a run that is already lost.
4. **A run list** fixed before each sitting (number, block, sheet, lay-out, start, model, step setting, budget). A
   small wrapper starts run N from the list, shows model, step and sheet before Enter and writes them with the code
   version into the run's summary. The scoring command refuses a run that does not match the list.
5. **Prints.** For sitting A: sheet 03's shape with thin grey circles at about 20 places, and a page with a 100 mm
   bar that checks the printer first. For sittings B and C: our own print of sheet 01; two new solid tracks (one
   loop with a crossing, one bend of about 0.8 cm radius); the shapes of sheets 05 and 06, each solid and dashed.
   The track generator needs real work for this: corners as tight as the course sheets, a given centre line drawn
   solid and dashed, dash and gap lengths drawn per track (dash 15 to 35 mm, gap 8 to 15 mm), true-size PDF,
   `tracks.json` rebuilt. Every traced print is laid over the real sheet against a window (lines within 3 mm).
6. **A reach check for every new lay-out:** a script walks the track's centre line through the driver's guards and
   must find no refused step and the paper clear of the base; the table's edge is measured before taping.
7. **On ki-014 and the laptop, no robot:** the stop scored with the last picture counted and reported per sheet; one
   training without the 42 near-end-start episodes; the two stop targets that exist in the code and were never run
   end to end; a cleaning rule that drops the rows just before a hand correction; sampling per shape; the judging
   panel of step 5 as a script.

### Step 2. Three test sittings

Rules for every sitting. The sheet lies on tape marks; the start is the recorded start pose with the forearm straight
and the puck ahead of the tip as in the recordings. Nobody and nothing coloured is in the camera's view. One warm-up
run is not counted. Short runs end by the travel budget; a full run is ended by hand only when the puck has left the
paper, has been pushed more than 3 cm past the line's end, or for safety. Every run gets one label: finished, lost,
stopped early, never stopped, stalled, refused, void. Void, and repeated, only for a lost connection, no fresh
picture, a set-up mistake or something in the camera's view. At every stop a phone photo from straight above with a
ruler in view is taken while the arm waits above the start. Sheets of small pictures are judged by someone who does
not see the run list. Rules are written into the checklist before the sitting; three or four runs per setting can
only show large differences, and anything closer is run again.

**Sitting A: the small steps, the measure, the driver's step (about 2.5 hours; sheets 01, 03, 04 and the circle print)**

| Part | Question | Runs | Rule fixed in advance |
|---|---|---|---|
| A1 | Do steps of 1 to 3 mm come out, and does the puck follow them? | a fixed list of answers (ten steps each of 1, 2 and 3 mm, five twists in place of 5 and of 10 degrees), first without the puck, then with the puck on sheet 01 | no refused step; the tip moves 10, 20 and 30 mm, each within 10 %; the puck moves the same within 2 mm and less than 2 mm under the twists. If the 1 mm group fails the driver gets a smallest step of 2 mm |
| A2 | Is the measure right where it has to judge? | the puck set by hand into the grey circles (on the line, 7, 12 and 17 mm beside it, at the bend, at the end, far out, forearm turned), the arm held still | within 3 mm at 18 of 20 places and never more than 5 mm off; if not, nothing in these sittings is decided by number, and the sheets of small pictures decide |
| A3 | Does what worked on 11.11 still work from the recorded start? | model E, sheet 01, the driver of 11.11, 3 runs | fewer than 2 finished: stop and repair the start before anything else |
| A4 | Does a shorter step keep the puck on the line at the first bend of sheet 03? | short runs. Model E: fixed 1.0 cm against fixed 0.65 cm, 5 pairs, the order inside each pair by coin. Model E with its own length, 4 runs. The team of five with fixed 1.0 cm, 4 runs. The model trained without sheet 03 with fixed 1.0 and 0.65 cm, 3 runs each | a run keeps the line when the puck's centre is never more than 12 mm from the centre line on two pictures in a row, until the puck has passed 119 mm of line (8 cm past the bend). 0.65 cm goes on only if it keeps the line in at least 3 more runs than 1.0 cm (a setting that is no better passes this by luck in at most 1 of 18 sittings [calculated]). The own length replaces 0.65 cm only if it keeps the line in all 4 runs while 0.65 cm loses it in 2 or more of 5. If 0.65 cm keeps the line in fewer runs than 1.0 cm with the model that never saw the sheet, both steps go to sitting B |
| A5 | Was sheet 04's failure the placement? | model E, the driver of 11.11, sheet 04 on tape marks at its main recorded place (plans 04-01, 04-02, 04-05), 3 short runs to past the first hairpin; one full run only if the bends are passed | through the first S-bend in at least 2 of 3 runs: run 6 was mostly start and placement. If not: the same three runs with the step A4 chose, before anything is recorded for sheet 04 |

**Sitting B: the loop, lines never seen, the driver decision (about 2.5 hours; needs our own print of sheet 01 and the two new tracks)**

| Part | Question | Runs |
|---|---|---|
| B1 | Where does the puck stop with the chosen step? | model E, sheet 01, 3 runs; stop places beside those of A3. If the median is more than 3 mm shorter, the last move before the stop is looked at before the next sitting; nothing is decided on six runs |
| B2 | Does the step break the loop that works? | sheet 05 on tape marks: model E with fixed 1.0 cm and with the chosen step, 3 runs each, taking turns; then the model trained without sheet 05, 3 runs |
| B3 | A line never seen | sheet 03 with the model trained without it, fixed 1.0 cm and the chosen step, 3 runs each; new track 1 with model E, both steps, 3 runs each; new track 2 with the chosen step, 3 runs |
| B4 | Is our own print like the course's to the camera? | our own print of sheet 01, model E, 2 runs |

The driver's step is decided after sitting B by this rule, written before sitting A: a new step replaces fixed 1.0 cm
only if (1) it loses the line in at least 2 fewer of the 6 runs on a line never seen (sheet 03 with the model trained
without it, and new track 1), (2) it finishes sheet 05's loop at least as often, and (3) of the runs that reach the
end it covers the end at least as often. Otherwise fixed 1.0 cm stays and the question is run again with more runs.
Reading B2: if the model that never saw sheet 05 fails where model E finishes at least twice, "learnt by heart" is
seen on the robot and new loop shapes are recorded first.

**Sitting C: the stop, dashes, orientation (about 2 hours; needs the four prints of the shapes of sheets 05 and 06)**

| Part | Question | Runs |
|---|---|---|
| C1 | Does the robot stop at the end of a line it has not seen? | starts 6 to 8 cm before the line's end on sheets 02, 05 and 06, each with the model trained without it, sheet 03 as the control, 3 runs each: stopped on the end, early or never |
| C2 | Dashes, shape, or learnt by heart? | shape 05 solid and dashed, shape 06 solid and dashed, all four printed by us, 2 runs each in mixed order, the driver of 11.11. Three readings of 4 runs against 4: solid against dashed, shape 05 against 06, the two prints model E knows against the two it does not. A reading counts at 3 or more of 4 against 1 or fewer of 4 (by luck 1 in 10 [calculated]) |
| C3 | A lay-out never recorded | sheet 01 taped at about 45 degrees: 2 runs from the straight start, 2 from a start with the forearm turned along the line. A first look, no decision |

If time runs short, C2 is cut first, then C3; never A1 to A3, the 1.0 cm runs on sheet 05 or the runs on lines never
seen.

### Step 3. Version 1 is frozen

Right after sitting B: the model trained on all six sheets (recipe E), the step the rule chose, the lay-out rule and
the start routine. Its code state and model file are named in `STATUS.md` and copied to a second disk, and it is
started once on a second machine. Version 1 goes to the exam unless a later version beats it in the paired test of
step 6. Corrections may be recorded from here on, because the driver no longer changes.

### Step 4. Demonstration rules, a pilot, recording round 2

More different tracks is the best-supported lever for an unseen line, and still a bet: about +4 points of direction
by the earlier report's fit, carried beyond its range (one to five sheets, frozen features), and more shapes were not
shown to help the stop [calculated].

**The rules** are questions for a hand trial of about 45 minutes at the robot, then written down and used by both
operators for every new episode:

1. Start: the sheet laid by the lay-out rule, the arm in the recorded start pose with the forearm straight, the puck
   ahead of the tip as recorded (a band is written down and the starts are spread over it on purpose: gap 4 to 6 cm,
   up to 1 cm to the side, forearm within 10 degrees). Tests, recordings and the exam use this same start.
2. Centring: watch the replay. When the puck's centre leaves the ink, pause, press `h`, guide it back and go on to
   the line's end. Record again only when the puck was lost. (In this tool the pictures come from the replay and the
   moves from the earlier guiding pass, so a drift in the replay is where a recovery can be recorded.)
3. Sharp bends: keep the recorded way. The camera leads into the bend while pushing; the tip goes round the outside
   while the forearm turns; turn in place only where the bend is tighter than the puck can follow. A new way is
   recorded only where the old recordings disagree: the square start and, if sitting A says so, sheet 04's first
   S-bend.
4. Loops: keep the walk round the puck as the one way, with its visible sign written down from what the 31 loop
   recordings did. For the stretch the camera cannot follow, two written ways are tried three times each on one loop
   print with the forearm's angle shown live; the way whose repeats agree best is kept.
5. The end: be on the middle of the line 3 cm before the end, then push straight until the end is covered. Every
   recovery and every near-end start is guided on to the line's end, so that no stop is ever recorded in the middle
   of a line.
6. The wrist: the trial shows whether a hand can keep it within 3 degrees of 13.5 with the angle shown live. If not,
   the replay holds it (a change of the recording tool, tried on 5 test episodes; kept only if the puck still
   follows). If neither works in the loops, the wrist stays free there, as in the old recordings.
7. A crossing is driven straight across. A dashed line is driven exactly like a solid one.
8. Software limits as Emranul set them on 11.11.26: 2 to 3 degrees inside the hardware limits.

**A pilot before the hours are spent** (about 1.5 hours): 8 episodes on 4 new prints, each print recorded once by
Emranul and once by Marshal. Read from it, with limits written before: can the rules be kept; do two recordings of
the same print agree (forearm angle and tip path at the same place: "one style" measured directly); the real minutes
per episode and per lay-out. If the rules cannot be kept, the rules change, not the amount.

**Recording round 2** [hours calculated from 5.8 minutes per episode and 10 minutes per new lay-out]:

| What | How much | Robot hours |
|---|---|---|
| New shapes: 18 to 24, 2 or 3 episodes each (decided in the pilot), each shape printed solid and dashed; 4 of them repeated in two lay-outs; at least 4 loop shapes in both turning senses; some lines ending far from the paper's edge | about 48 episodes | 9 to 10 |
| Line ends: a first batch of 10 on new prints; how many more follows from the two stop checks on ki-014 (step 1) | 10, then as found | 1 to 3 |
| Recoveries recorded on purpose: the puck 1 to 2 cm beside the line with the line still in view | about 30 short episodes, after the driver decision (a 1 cm step cannot carry out a walk round a standing puck) | about 3 |
| A provided sheet that still fails after the sittings: extra recordings of that sheet in its exam lay-out, in the agreed style | as found | open |

From the first recording day the newest model drives each fresh print once before it is demonstrated (all the day's
prints first, then the driver is closed and the recording tool opened, because only one program may be connected):
free first runs on lines never seen, labelled as looks. The model is retrained every week.

**Three kinds of printed tracks.** Training shapes are recorded on; corrections are driven on them and on the six
sheets. Development shapes (at least ten) are never recorded on and carry every choice between versions; a fresh set
is drawn once one has been used for a choice. Sealed shapes are used once, in the final test: Marshal draws and
prints them with a seed he keeps, after the drawing rule is fixed, and their files stay out of git; two or three are
drawn by hand, and the professor's extra tracks are held out if they come. The drawing rule is matched to measured
features of the six sheets (tightest bend, length, loop, crossing, dashes, the end's distance from the paper's edge).

**The old 209 episodes** will still be about four fifths of the pictures. What stays is decided by whole episodes and
written before any result: two variants are trained (all old episodes; all but the groups that break a written rule
as whole episodes: the six square starts, and the loop episodes only if the hand trial changes the way through a
loop), and the panel of step 5 decides. Sampling is per shape, so that a shape with two episodes is not drowned by
sheet 04.

### Step 5. Training and how a model is judged

Training stays on ki-014 (minutes per model). The earlier rule (5 points of "path within 45 degrees") is replaced: it
reads 0.96 at the first bend of sheet 03, where the robot lost the puck, and cannot see the twist or the stop
[measured]. The panel, with today's values, is fixed before the first run; every candidate is trained twice and
scored on shapes it was not trained on:

| Line of the panel | Today | A candidate is kept at |
|---|---|---|
| How closely the asked twist, summed over ten steps, follows the recorded one | correlation 0.71 | +0.05 with both trainings, and only while the recordings still show 0.85 to 1.1 of what the model asks (bigger twists alone must not pass) |
| The stop over whole episodes, the last picture counted | 127 right, 27 early, 47 never of 209; reported per sheet | 15 more right stops with both trainings and no more early ones |
| Guard: path within 45 degrees | 0.79 to 0.82 | not down by more than 2 points |
| States off the line: direction in the first five steps after a hand correction | 33.5 % against 69.5 % elsewhere | reported; one in four new recoveries is held back as a test |
| Robustness (dashes, sideways shift, coloured patches) | on dashed copies the model loses 62 to 70 % of its twist in the walk (control repaint 12 %) | its own stress test, and under 2 points lost on clean pictures |

The headline number is each course sheet left out with all new shapes in training: it shows directly whether our
shapes help on the professor's shapes. A table kept from sitting A on puts the same model's panel numbers beside its
robot result on the same shape; until it shows which line follows the robot, no single line may veto a candidate. The
pictures of the ten runs stay a fixed test that is never trained on, reported as "k of n expectations met" and never
used as a pass rule.

Candidates, in this order, the list closed before version 2 is frozen: the present recipe on the new data with
sampling per shape; the two stop targets; the two variants of the old episodes; dashes cut into solid-line pictures
with a wider sideways shift and coloured patches (each edited picture checked by a number, a sample looked at). A
target that separates "which way" from "how far" only if a one-hour probe shows a gain. Not now: a second picture as
input, direction as classes, carrying out several predicted steps.

### Step 6. Version 2 against version 1, corrections, speed

- The paired test: version 2 and version 1 from the same start on at least ten development shapes and the six
  sheets, one pair each, scored from pictures and photos. Version 2 replaces version 1 only if it finishes more first
  runs; at most one more such comparison, because each costs 2 to 5 robot hours.
- Corrections. The take-over does not exist yet: the driver has no take-over key and the recording tool asks no
  model. It is built by the session that owns the recording tool, through the orchestrator (2 to 4 code days
  [guess]), with a written rule (take over as soon as the puck's centre leaves the ink, also for an early stop and
  for running past the end; the model's stop waits for Enter) and a pilot of 5 episodes. Then about 50 episodes in
  two rounds of about 25, a week apart, each on the newest model, on fresh prints as well as the six sheets; training
  from scratch on everything after each; only the human's part is trained on.
- Speed: one block only once version 2 finishes most development shapes.

### Step 7. Final test, freeze, exam day

- The final test in exam form by 20.12.26: the six sheets three times and twenty sealed tracks once each, the first
  run counts, every sheet laid by the rule against a clock; the light block; one timed rehearsal.
- A readiness check before the freeze of 8.1.27 if the room can be used (camera check, one run on each provided
  sheet, one on a fresh print; about an hour): otherwise the first run after three weeks is the graded one.
- An exam-day sheet: who runs the program and who lays the sheet, nothing and nobody in the camera's view, the tape
  marks, the start routine, what to do if the camera check fails.
- 21.12.26 to 8.1.27: documentation and slides. Hand-in 10.1.27, presentation 11.1.27.

## 6. Calendar and robot hours

| Week | Without the robot | With the robot |
|---|---|---|
| 12.11 to 15.11 | decisions 1 to 3; tools and checklist for sitting A | sitting A |
| 16.11 to 22.11 | scoring; prints for B and C; the tracks drawn again; the rules drafted | sittings B and C; version 1 frozen; the hand trial and the pilot |
| 23.11 (meeting 3) to 6.12 | weekly retraining with the panel; the stop checks | recording round 2, about 13 to 16 hours; the paired test at its end |
| 7.12 (meeting 4) to 13.12 | retraining after corrections | corrections if the tool exists, else recoveries on purpose; extra recordings for provided sheets that still fail; the speed block |
| 14.12 to 20.12 | | final test in exam form, light block, timed rehearsal. PRIMUSS registration 9 to 17.12. Robot work ends 20.12 |
| 21.12 to 11.1.27 | documentation, slides; freeze 8.1; hand-in 10.1 | the readiness check; the demo on 11.1 runs the frozen version |

All of it together is about 35 to 45 robot hours [calculated from measured times; the lay-out minutes are a guess];
the 210 recordings took 28.5. If hours run short they are cut in this order: the second round of corrections, the
first round with its tool, sitting C's parts C2 and C3. The light block and the rehearsal stay.

## 7. Decisions, one at a time

Each is put to Emranul as one card: the question, the choices, what each costs, this session's advice.

| # | Decision | Who |
|---|---|---|
| 1 | A safety copy on a second disk; commit this session's work on this laptop; push it or not (the repository is public) | Emranul |
| 2 | The exam lay-out rule: every sheet on one taped start mark, the line leaving straight away from the robot, started from the end that keeps the whole line within about 42 cm, turned by up to 15 degrees if neither end does. It replaces the note of 7.11.26 (sheets lying square "for a tidy look"), the lay-out that failed in run 3 | Emranul and Marshal |
| 3 | Go for the tools and the checklist of sitting A, and the day at the robot; the new start move of the arm is part of that checklist | Emranul |
| 4 | Robot hours per week until 20.12, who is at the robot, Marshal's share | Emranul and Marshal |
| 5 | A short message to the professor (section 8) now, or the questions kept for the meeting of 23.11 | Emranul |
| 6 | Freeze version 1 after sitting B (with one line confirming that the backup model "names plus exact" is dropped, as the rule of the earlier report says) | Emranul and Marshal |
| 7 | The demonstration rules, after the hand trial; what "any orientation" should mean with this camera, and how many hours it gets | Emranul and Marshal |
| 8 | Size and order of recording round 2, after the pilot | Emranul and Marshal |
| 9 | Extra recordings for a provided sheet that still fails | Emranul |
| 10 | Version 2 or version 1, after the paired test | Emranul and Marshal |
| 11 | The correction rounds, after the pilot of 5 episodes | Emranul |

## 8. A short message to the professor (draft content)

Part A, "we understood this, please correct us", one line each: the eight statements of section 2.

Part B, open questions, in the order of how much the answer changes the plan:

1. Our program turns the model's answer (a tip step and a forearm twist) into joint moves, holds the wrist still,
   limits the step, and ends a run that stands still. Is that within the rules? May it move a fixed distance in the
   model's direction, and add one fixed move after the first stop picture?
2. Before a run, may we put the pusher behind the puck by hand, or by one fixed start move?
3. The model drives, we take over by hand where it goes wrong and record that. Is this our own demonstration?
4. When is the puck "at the end": touching it or covering it? Does pushing past it count as failed?
5. Which of the six sheets and how many unseen tracks are run, where and under which light? May we do one test run
   there? Is there a time limit?

Last line: the extra track files promised on 26.10. Kept for the meeting of 23.11: what to hand in, where the model
runs, whether the model may see the previous picture.

## 9. What we will not do, and why

- No new hand-written steering: no twist rule made from the model's direction of travel (wrong sign against the
  recordings, checked) or from the puck's place (weaker than the model's own twist, and not imitation); no "keep the
  last direction" rule (it would have broken run 7); no nudges; no wrist that follows the model's lean output (three
  to four times worse than holding it).
- No extra push beyond the present centimetre and no higher stop threshold. Whether the move after the first stop
  picture should be shorter is looked at together with the step rule.
- No larger twist factor as the fix: the model asks the right amount where it asks, and with a factor the summed turn
  drifts more. As a diagnosis only, and only if sheet 04 fails again from its recorded place.
- No switch of model for sheet 04 or the dashed loop. Whether the other recipe is better at sharp bends is tested in
  sitting A and weighed against the stop, where recipe E is better offline.
- No plain rotation of pictures; no touching the camera; no rented computers (a training takes minutes).
- Nothing judged on the six sheets alone.

## 10. Risks

| The sign | What we do | What it costs |
|---|---|---|
| The robot does not stop, or stops early, on a line never seen (C1) | the stop targets first; near-end starts on every new print; tracks whose ends lie far from the paper's edge; the lay-out rule starts from the end that stopped better | GPU minutes; 1 to 3 robot hours |
| A provided sheet still fails after retraining | extra recordings of that sheet in its exam lay-out, kept only if the score on unseen shapes does not fall | a few robot hours per sheet |
| A step is refused at the reach or lean limit (the run ends there) | the reach line in the lay-out rule; every new lay-out walked through the driver's guards first; one dry run to a sheet's far end in sitting A | none on the robot |
| The chosen step stops short of the end | no extra move against one 5 mm move after the stop rule fires, on unseen sheets, scored by the photos | one block of a sitting |
| Loops stay unreliable on unseen lines | loop shapes in both turning senses in round 2; start from the end from which loops went better in the tests | inside round 2 |
| The measure fails its check in A2 | sheets of small pictures and photos decide; no automatic gate on recordings | time at the desk |
| The camera moves again | the camera check at every sitting; if it fails, stop; never touch the camera | a sitting |
| Three runs disagree | a decision only at 3 against 0, otherwise three more runs | minutes |
| The pool computer is closed | about 2 to 2.5 hours per six-sheet training on the laptop | laptop hours |
| A new version is worse | version 1 stays | none |
| Robot hours run short | the cut order of section 6 | |
| The professor forbids a driver rule | every such rule (step rule, stall rule, last move) sits behind one switch | none |

## 11. Who does what

- This session (Model and training): the driver `model_mode.py`, training, the scoring command as a separate script,
  the track generator, the checklists' technical content.
- The recording tool (the wrist held by the replay, the take-over for corrections, any number shown after an episode)
  belongs to the Recording and data session: these go through the orchestrator, or are made here with the
  orchestrator's written yes; never two sessions in one file.
- The orchestrator: `05_meetings/requests_to_professor.md` with the eight reported answers, and the record of meeting 2.
- Before a push: the repository is public and the result is ranked against the other teams. A push would publish the
  plan, the analysis of the failures, the driver, the track generator and its seed, and the memory note on the pool
  computers (host names and account names). Emranul and Marshal decide whether it stays public or is made private
  until 11.1.27. Sealed tracks and their seed never go in.
- Marshal's share is the two students' decision (decision 4); candidates: drawing and keeping the sealed tracks, half
  of recording round 2 under the new rules, the photo check of the measure, driving one block of a sitting.
