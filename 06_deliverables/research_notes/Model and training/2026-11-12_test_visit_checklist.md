# Test visit of Thursday 12.11.26: checklist

Written 12.11.26 by the Model and training session for Emranul, who drives the visit alone at the robot. The run list
is `2026-11-12_test_visit_runs.csv` in this folder: 32 runs, R00 to R31, and 7 extra runs, X01 to X07, that are driven
only when a rule below asks for them. You never read or change that file; the command reads it for you.

Nothing new on this list has been tried at the real robot. The placing line, the faint picture and the two new step
lengths have run on the simulated robot only, and no person has pressed their keys yet [measured there by the session
that built them]. Tick each box. Stop and ask when anything looks different from what is written.

Print this list, or keep paper and a pen beside the laptop: one label per run is written by hand. R00 to R31 are the
numbers of this visit's runs. They are not the new track prints T01 to T16; none of those is used in this visit. If
the visit is driven on another day than 12.11, nothing in this list changes.

## What this visit is for

On 11.11.26 the robot drove ten runs: four brought the puck to the end of the line, six lost the line, and the check
found that the driver pushed 1.0 cm at every step where the model had asked for about 0.6 cm. Today you test that one
thing against a control: does a shorter step (a fixed 0.65 cm, or the length the model itself asks for) keep the puck on
the line at the first bend of sheet 03, where model E lost it with 1.0 cm steps on 11.11? Before that, three runs on
sheet 01 show whether what worked on 11.11 still works when the start is placed with the two new aids. At the end, three
short runs on sheet 04, laid properly this time, show whether its failure on 11.11 was the placement. You drive, take
one photo and write one label per run; the scoring is done afterwards at the desk by the planning session (the Claude
session that planned this visit), by rules that were written before the visit and stand below.

**Time: 3 to 4 hours for R00 to R31, about 4.5 hours with the extra runs and one or two repeats.** Three hours is
the least: it is the pace of 11.11.26, when a run had no photo, no placing line and no faint picture.

| Part | Runs | Minutes | You may end the visit here |
|---|---|---|---|
| 0. Before switching on, and set-up | none | 30 | |
| 1. R00, the warm-up; sheet 01 is laid and taped | 1 | 12 to 20 | |
| 2. R01, a tool test without the puck | 1 | 4 to 6 | |
| 3. Part 1, the control: R02 to R04 | 3 | 10 to 16 | if the control fails: then X01 to X03 (10 to 16 minutes more) and the end; or if too little time is left for Part 2 (a) and (b) |
| 4. Part 2 (a) and (b): R05 to R20; sheet 03 is laid and taped | 16 | 65 to 97 | after R20, never before |
| 4. Part 2 (c): R21 to R28 | 8 | 29 to 47 | after R22, R24, R26 or R28 |
| 5. Part 3: R29 to R31; sheet 04 is laid and taped | 3 | 19 to 25 | after R31 |
| 5. Extra runs of Part 3: X07, or X04 to X06 | 1 or 3 | 7 to 9, or 11 to 17 | |
| 6. After the visit | none | 10 | |

Where the minutes come from. On 11.11.26 a picture took 0.93 s, and the time from the last picture of one run to the
first picture of the next was 3 minutes (the median; shortest 1.8, longest 4.6) [measured]. The first number of each
range is counted with these [calculated]. It is the least, because today every run also has a photo with the ruler,
the placing line and the faint picture. The second number counts 5 minutes between runs [guess: nobody has timed
the new hand work]. Set-up (30), laying and taping a sheet (8 each, counted into parts 1, 4 and 5), the 20 for a
first run with the new aids, and the end (10) are guesses. A repeated void run adds 4 to 6 minutes. Your own pace
is measured in section 3: there you write down how long R02 to R04 took.

Where you may end early:

- At any moment, for safety or when something looks different from what is written.
- After Part 1, if fewer than 2 of its 3 runs finished: drive X01 to X03, then end the visit and report.
- In Part 2 after R20 at the earliest: (a) and (b), R05 to R20, are never cut. They are 16 runs and one change of
  sheet: 65 to 97 minutes [calculated; the 97 rests on the guess of 5 minutes between runs]. Do not begin Part 2
  with less than 1 hour 45 minutes left; section 3 counts it from your own pace. After R20 you may end after any
  own-length run: R22, R24, R26, R28.
- After R28, Part 3 may be left out. After R31, the extra runs may be left out.
- A pause is possible between two runs, never inside one. In R05 to R20 pause only after the second run of a pair
  (after R06, R08, R10, R12, R14, R16, R18 or R20), so that the two runs of a pair stay together. For a pause: one
  hand ready at the arm, motors off (FreeMotion), take the last run's photo, park the arm by hand; the robot stays on.
  Afterwards go on with the next label.
- Going on with the rest on another day is possible only after an answer from the planning session: the set-up and
  the warm-up are then done again, and it says how. The clean places to stop for the day are after Part 1 (before
  R05) and after R28 (before R29). Inside Part 2 a day ends only after the second run of a pair; the pairs driven so
  far then count (see "Decided by the planning session" under the rules).
- Whenever you end: section 6.

## 0. Before switching on, and set-up (about 30 minutes)

### At the laptop (no robot needed)

- [ ] Open **PowerShell** (not Git Bash, not an editor's output pane) and **maximise the window**. The placing line is
      about 110 characters long and has to fit into one row.
- [ ] `cd D:\AI_Project\.claude\worktrees\training-analysis\02_code`
- [ ] Type this once in every new PowerShell window. It gives the run list a short name:
      `$list = "..\06_deliverables\research_notes\Model and training\2026-11-12_test_visit_runs.csv"`
- [ ] `..\..\..\..\.venv\Scripts\python.exe test_model_mode.py` ends with `all tests passed` (about half a minute).
      If it prints nothing new for two minutes: Ctrl+C, run it again, and write down where it stood.
- [ ] `..\..\..\..\.venv\Scripts\python.exe test_visit_runs.py` ends with `all tests passed`.
- [ ] The whole run list is in order. This starts nothing and takes about half a minute. It prints 39 lines that
      begin with `Run`, and none that begins with `not started`:
      `foreach ($row in Import-Csv $list) { ..\..\..\..\.venv\Scripts\python.exe visit_runs.py $list $row.label --dry | Select-String "^Run |^not started" }`
- [ ] Look at one run in full before any run is started. `--dry` starts nothing:
      `..\..\..\..\.venv\Scripts\python.exe visit_runs.py $list R05 --dry`
      It prints the sheet, the model and its files, the step, the step budget, the reference picture (the faint
      picture) and, last, the full command the run would use.
- [ ] Rehearse the keys twice on the simulated robot (about 4 minutes; no robot is connected, nothing real moves):
      `..\..\..\..\.venv\Scripts\python.exe model_mode.py rehearsal --place-help --fresh-keys --step-budget 10 --ghost "D:\AI_Project\03_data\raw\main_data\2026-11-06_session_01_emranul\path_01_straight_line_episode_004_plan_01-02_try_1_score_02\frames\000.jpg"`
      The questions are those of the robot from Enter 2 on (Enter 1 belongs to `visit_runs.py`). The first time:
      wait for the question `Put the puck at the line's start. Press Enter to switch the motors OFF ...`, then Enter
      (the motors go "off"). The placing line starts at `too high` and comes closer by itself. Wait about 6 seconds,
      until it reads `: close   Enter = placed`, then Enter. The picture window opens with the faint picture: click
      on it, then Enter there. The simulated run is over within seconds; scroll up to `run rehearsal ended: ...`.
      The second time (Up arrow, Enter): the same, but press Esc in the picture window in place of the last Enter;
      it prints `nothing moved by the model; ending`.
      An Enter pressed while the line still said `too high` gives `not started: the tip is ... from pushing height`,
      as it would at the robot: run it once more. The simulated picture looks nothing like the real one; only the
      keys are practised.
- [ ] Look once at the three set-up pictures, so that you know roughly how each sheet lies. Each shows the table
      from above, the robot's base on the left:
      `Invoke-Item "..\06_deliverables\research_notes\Recording tool\figures\setup_per_episode\plan_01-02.png"`
      then the same line with `plan_03-04.png` and with `plan_04-05.png`. Close the picture viewer afterwards. The
      faint picture gives the exact place at the robot; it may lie a few centimetres from the drawn one.
- [ ] The run list is not opened in Excel and not changed. Nothing in it needs changing at the robot.

### At the robot

- [ ] Only one program will be connected to the robot: the recording page, Niryo Studio, every camera viewer and
      every other script are closed.
- [ ] The robot stands where it stood on 11.11.26, and away from the table edge. Good to have from the middle of the
      base: 55 cm of table ahead and 35 cm to each side [a guess; the driver lets the tip go 49.5 cm out and 30 cm to
      the side]. Where there is less, write down how much and watch the arm on that side. If the robot or the table
      was moved since 11.11: stop and tell the planning session, because the table height the driver uses was taught
      for that place.
- [ ] The pusher is the magnet block on the rod, as on 11.11.
- [ ] The camera and its focus ring are not touched, now or at any point of this list. Look only: if the camera's tape
      looks moved, stop the visit.
- [ ] Robot switched on. Laptop connected to the robot by the Ethernet cable, as on 11.11, and on its charger.
- [ ] Only the automatic calibration, never the manual one. Table clear, puck away, nobody touching the arm, stop
      button in reach. Then, from 02_code: `..\..\..\..\.venv\Scripts\python.exe calibrate_robot.py`
      It answers `no calibration needed` (nothing moves), or `automatic calibration running; the arm moves by
      itself` and then `calibration done`. Anything else (`calibration FAILED`, or lines that begin with
      `Traceback`): stop and ask.
- [ ] Ceiling light on, as on 11.11. Time now, and is it daylight or dark outside? ____________________
- [ ] The laptop stands outside the camera's view, where one hand reaches its keyboard while the other carries the
      arm, and where you can read its screen while your hands are at the sheet. Stop button in reach from there.
- [ ] On the table, outside the camera's view: sheets 01, 03 and 04, the red puck, a ruler with millimetres, the
      phone, clear tape (the kind used in the recordings, nothing coloured), a pen and this checklist.
- [ ] One phone photo of the laptop's clock with seconds (type `Get-Date` and photograph the answer), so that photos
      and run folders can be matched by time afterwards.
- [ ] Nobody and nothing coloured in the camera's view during a run. You stand behind or beside the robot, hands and
      sleeves out of the picture.
- [ ] The two rules for the arm, read aloud once: **motors off (FreeMotion) before anyone touches the arm; never push
      on a held arm.** In today's runs the driver switches the motors off for you, at the second Enter. Until then
      a hand only waits at the arm, without pushing, to carry it when it goes soft.
- [ ] My go for the two things that are new today in how the arm moves: steps of the model's own length, which can be
      shorter than 1 cm (first in R01), and fixed steps of 0.65 cm (first in R06). Nothing else is new in the arm's
      movements: the settling, the pushing and the lift are those of 11.11.

## How one run goes (the same for every run; you learn it in R00)

**The command.** Click on the title bar of the PowerShell window, so that it has the keyboard. Not inside its text: a
click inside the text can freeze the window until a key is pressed [known of the classic Windows console; not tried on
this laptop]. Then type, with the run's label at the end:

`..\..\..\..\.venv\Scripts\python.exe visit_runs.py $list R00`

For the next run press the Up arrow and change the label. A label is a capital letter and two digits (R07, not
r07). To see a run without starting it, add ` --dry` at the end. To drive a label a second time (a void run), add
` --again` at the end. Written out without the short name, the same command is:

`..\..\..\..\.venv\Scripts\python.exe visit_runs.py "..\06_deliverables\research_notes\Model and training\2026-11-12_test_visit_runs.csv" R00`

**Enter is pressed four times in a run. Each time it does something else:**

| Enter | Where | What it does |
|---|---|---|
| 1 | terminal | starts the program; nothing moves; the model loads for some seconds |
| 2 | terminal | switches the motors OFF; the arm goes soft and sags, so a hand is ready at it first |
| 3 | terminal | "placed": the motors come ON, the arm lifts 3 cm and comes down again |
| 4 | picture window, after a click on it | the model pushes |

**Esc ends a run at once** (within one step) from the moment the picture window is on the screen, in the picture
window or in the terminal. The arm then lifts 5 cm, so take your hands away from it before you press Esc. While the
placing line is on the screen (step 4), Esc does nothing. **Never Ctrl+C after Enter 1.** The stop button is for
anything unsafe.

**After Enter 3, fingers off the Enter key until you mean Enter 4.** The picture window may take the keyboard by
itself when it opens [not tried on this laptop]. An Enter pressed then lets the model push at once, wherever your
hands are.

**Every run, in short.** The seven steps below say each line in full; after a few runs this is enough:

1. The command with the run's label. Read sheet, model, step and step budget. Enter 1.
2. One hand ready at the arm. Enter 2. The arm is soft: rest it on its rod, away from the puck.
3. The photo of the last run's puck with the ruler. Puck and ruler off the sheet.
4. Guide the tip until the placing line says `close`, as near 0 as you can. Let go. Enter 3. Hands off the arm.
5. In the picture window. First run on a sheet: slide the sheet onto its faint copies and tape it. Later runs: the
   line and its faint copy overlap; if not, Esc and place again. Then the puck on the start of the line, on its
   faint copy. Nothing else in the picture. Hands off. Click the window. Enter 4.
6. Watch the puck and the arm. Esc only by the rules.
7. Read how the run ended. Write the label. Puck and arm stay as they are.

**Step 1. Start.**

- [ ] Type the command with the run's label. Read what it prints: `Run R00: THE REAL ROBOT`, then `sheet`, `model`,
      `step` and `step budget`. They are those of this run's line in the table of all runs, and that sheet lies on
      the table (in R05 and R29 it lies ready: the sheet is changed inside the run). The step reads `fixed, 1 cm`,
      `fixed, 0.65 cm` or `the length the model asks for`; a full run has a step budget of 250 steps.
- [ ] If they do not fit: Ctrl+C at this question leaves without starting anything. It is the only place for
      Ctrl+C in a run.
- [ ] At `Press Enter to start this run`: **Enter 1**. Then press no key until the next question is on the screen.

**Step 2. Motors off.**

- [ ] The question reads `Put the puck at the line's start. Press Enter to switch the motors OFF (hold the arm: it
      will sag)`. Its words about the puck are the driver's old ones: today the puck comes later, in step 5.
- [ ] Put one hand at the arm where you hold it for guiding, never at the camera, its tape or its cable, ready to
      carry it. Do not push, pull or lift yet: the motors still hold it. Then **Enter 2**, one short press with the
      other hand. The arm goes soft and lies in your hand.
- [ ] Do not let the arm drop onto the puck: the puck stays where the run left it until its photo is taken. Lift
      the block clear of the puck, guide the arm back towards the start and let it rest on its rod, away from the
      puck. Now both hands are free.

**Step 3. The photo of the last run, then the puck off.** (No photo in R00: there is no last run. None in R02: R01
had no puck. None when the photo was already taken, as after a start that did not push.)

- [ ] Do not move the puck yet. Lay the ruler flat beside it, not touching it and not shifting the sheet: along the
      line at a line's end, across the line anywhere else.
- [ ] One photo from straight above: the phone level, the puck in the middle, no side wall of the puck showing, the
      ruler's numbers readable. At a line's end the photo has to show whether white paper lies between the puck and
      the black end, and how much black of the end still shows.
- [ ] Puck and ruler off the sheet.

**Step 4. Place the tip with the placing line.** After Enter 2 the driver prints nine lines that begin `Guide the
magnet tip just behind the puck, facing along the line.` Those first words are the driver's old ones as well: today
no puck is on the sheet, and the tip goes to the recorded start, which is what the line under them measures. That
line renews itself twice a second, for example:

`tip  0.9 cm off: 0.1 too far out, 0.9 too far left, forearm   +2.1 deg, height  +0.3 cm: close   Enter = placed`

It says where the tip IS against the recorded start, so move it the other way: "too far left" means move it to the
right. Left and right are the robot's own, as if you stood behind the robot and looked out over the sheet. "forearm" is
the twist of the forearm. The word after the last colon is the verdict: `close` means the tip within 1 cm and the
forearm within 3 degrees; `not close` means not yet; `too high` or `too low` means the run would not start from
there: let the arm rest on its rod. For the first half second the line ends in `(wait)`: Enter does not count yet.

- [ ] Guide the arm by its body, never by the camera or its tape, until the verdict is `close`, not `not close`.
      `close` is only the least: go on until the tip is as near 0.0 cm and the forearm as near 0.0 deg as you can
      bring them. Lift the block a little to move it; do not drag it across the paper.
- [ ] In the first run on a sheet (R00, R05, R29) take extra care here: the sheet is laid by this start and stays
      there for every later run on it.
- [ ] Let go of the arm; it rests on its rod. Read the line once more: it still reads `: close   Enter = placed`.
- [ ] Both hands off the arm. **Enter 3**, one short press. The arm lifts 3 cm and comes down again. From now on
      its motors are on: do not touch it, and do not press Enter again before step 5 says so.
- [ ] The driver prints `start: tip ... from the recorded start`. If a second line follows that says the start is
      further off than 1 cm or 3 degrees, the arm moved after the last reading: press Esc in the picture window
      (step 5) and start the same label again. If the picture window covers these lines, drag it aside by its title
      bar.

**Step 5. The picture window: the sheet (first run on a sheet only), the puck, the go.** The window is named
`record_episodes`. It shows the live camera picture with a faint second picture laid over it: the first picture of one
recorded demonstration. Its dark top band reads `Enter = let the model push, Esc = do not start`. The terminal says
`slide the sheet and set the puck` each time; the sheet is slid in the first run on a sheet only. The arm stands
still with its motors on: your hands work beside it and under it, never against it.

- [ ] First run on this sheet only: slide the sheet until its line and its paper edge lie on their faint copies. Tape
      it down with clear tape at its corners, away from the line, and look again: still on the copies. If the sheet
      catches on the block, do not force it: hands away, Esc (the arm lifts), free the sheet, start the same label
      again.
- [ ] Every later run on this sheet: the sheet stays where it is taped. Look at the start of the line. The real line
      and its faint copy lie on each other or overlap; a small shift is normal, because the arm never stands in
      exactly the same place. If they lie side by side without overlapping (more than the line's width, about 1 cm,
      apart), the arm does not stand as it stood in the sheet's first run: hands away, Esc, start the same label
      again and place the arm nearer to 0. Never move the taped sheet for this.
- [ ] Set the puck on the start of the line, on its faint copy: it covers the line's rounded end, and the line
      comes out from under the middle of the puck, as in the faint picture. If the faint copy of the puck lies a
      little beside the real line, the line wins: the puck sits on the line. (No puck in R01.) Do not touch the
      block.
- [ ] Do not match the block at the bottom of the picture. It sits about 15 px to the right of its faint copy (about
      half the line's width), because the reference pictures were taken before the camera slipped on 8.11.26. Only
      the line, the paper edge and the puck are matched.
- [ ] Look at the live picture: nothing in it but the sheet, the puck, the table and the robot's own block.
- [ ] Hands off the table. Say "hands off" aloud. Click on the picture window. **Enter 4** there. The model pushes.

**Step 6. Watch.**

- [ ] Watch the puck and the arm, not the screen. Nothing and nobody in the camera's view.
- [ ] The run ends by itself (the model's own stop, the step budget, or a refused step) or with your Esc. The arm
      lifts 5 cm and stays there, held by its motors.

**Step 7. The label.**

- [ ] The last line on the screen reads `the run's folder: ...`: the run is logged. Scroll up with the mouse wheel,
      two or three screens, past a long block in curly brackets, to the line `run R00 ended: ...`. Its word after
      "ended:" says how the run ended: `stopped` is the model's own stop, `step_budget_reached` the budget,
      `refused` a refused step, `quit` your Esc. The third line of the block says the same, for example
      `"ending": "stopped"`.
- [ ] Write one label for the run into the table of all runs. Leave the puck and the arm as they are: the photo is
      taken in step 3 of the next run.

**The eight labels.** Exactly one per run. If two fit, write the one that happened first.

| Label | When |
|---|---|
| finished | the puck reached the line's end, the robot stopped by itself, and the puck still touches the end's ink |
| stopped off the end | the puck reached the line's end and the robot stopped by itself, but the puck no longer touches the end's ink (pushed past it or beside it) |
| lost | the puck left the line (white paper between the puck and the line), or the block lost the puck |
| stopped early | the robot stopped by itself before the puck reached the line's end; the puck was still on the line |
| never stopped | the puck came to the line's end and the robot pushed on; you ended the run with Esc |
| refused | the driver printed `refused: ...` and ended the run |
| ended by the budget | short runs only: the step budget ended the run and the puck was still on the line |
| void | only for the four reasons below; the run is repeated at once |

If none fits, write what you saw in a few words; for a run you ended with Esc after the stall line, write "stalled,
ended by hand".

**Void** only for these four: the wrong line of the run list was started; the puck stood clearly off its faint copy at
the start; something or somebody was in the camera's view; the connection was lost. Decide it from what went wrong,
never from how the run turned out. Write "void" and the reason, then drive the same label again at once: the same
command with ` --again` at the end.

**When you may press Esc.** For safety: always. In a full run: when the puck has left the paper, or has been pushed
more than 3 cm past the line's end. In any run: when the driver has printed `the stall rule fired` and the puck no
longer moves (today that rule only writes its line into the terminal and ends nothing; look there when the arm has
stayed on one spot for about half a minute). Otherwise let the run end by itself: a short run ends at its budget
within a minute and a half, a full run after 250 pictures, about 4 minutes, at the latest.

**Where the puck stands at the start** [measured from the recordings' pictures, no ruler check]. The faint picture is
the guide, not these numbers.

| Sheet | Faint picture of | The puck at the start |
|---|---|---|
| 01 | plan 01-02 | about 5.5 to 5.9 cm from the tip, almost straight ahead; in the middle of the picture, on the start of the line |
| 03, from end A | plan 03-04 | about 3 cm from the tip, ahead and to the left; the block nearly touches it |
| 04, from end A | plan 04-05 | about 5.5 to 5.8 cm from the tip, ahead and a little to the left |

## 1. R00: the warm-up, never scored (12 to 20 minutes)

Model E, sheet 01 from end A, fixed 1.0 cm, a full run. It is also the first time the placing line and the faint
picture meet the real robot.

- [ ] Sheet 01 lies roughly as in its recordings: straight in front of the base, the line pointing away from the robot
      (set-up picture: `06_deliverables\research_notes\Recording tool\figures\setup_per_episode\plan_01-02.png`).
      The puck is off the sheet.
- [ ] Start R00 and go through steps 1 to 7. Take your time.
- [ ] In step 4 the numbers of the placing line change while you move the arm.
- [ ] After Enter 3 the block ends just above the table, with a gap of about 5 mm that you can see from the side. If
      it presses on the table or the paper, or hangs more than about 1 cm above it [by eye; the 1 cm is a guess]:
      hands away, Esc in the picture window, end the visit and report.
- [ ] In step 5 the picture is live: it changes when you slide the sheet.
- [ ] In step 5 the sheet is slid onto its faint copies and taped. If the line and the paper edge cannot both be
      brought onto their copies, the sheet lies the wrong way round.
- [ ] In step 5 look where the block sits against its faint copy (a little to the right). It has to look the same in
      every run today. If that changes during the visit, the camera has moved: end the visit, do not touch the camera.
- [ ] The run ends when the model stops by itself. Write its label for practice.

## 2. R01: a tool test without the puck (4 to 6 minutes)

Model E with its own step length, sheet 01, no puck, at most 40 steps. It shows whether steps shorter than 1 cm look
smooth on the real arm. No run without a puck has been driven with a model before: the run of 11.11 that was named
"check_no_puck" had a puck [measured: its first picture].

- [ ] I know: **R01 is the first time the real arm makes steps shorter than 1 cm with this driver.** The stop button is
      in reach. I say go.
- [ ] Sheet 01 stays as taped. **No puck on the sheet**: in step 5 the faint copy of the puck stays empty.
- [ ] Start R01 and go through steps 1 to 7. Step 3 takes the photo of R00's puck.
- [ ] Watch every step: one small, smooth move; no jerk, no jump, no shaking or humming on the spot; the block stays
      just above the paper. The model may also twist the forearm on the spot without going forward: that is expected.
      Without a puck the model has nothing to push: it may leave the line or stop after a few steps. That is
      expected too and says nothing about the model; only the smoothness is watched.
- [ ] Smooth: yes or no? ________ What I saw: ________________________________________
- [ ] If any move looked wrong: Esc, and write down what you saw and at which step. Part 1 can still be driven (it
      uses 1.0 cm steps, as on 11.11). Ask the planning session before Part 2.

## 3. Part 1, the control: R02 to R04 (10 to 16 minutes)

Model E, sheet 01 from end A, fixed 1.0 cm, three full runs with the puck.

- [ ] Time now, before R02: ____________
- [ ] R02, R03 and R04, each by steps 1 to 7. Sheet 01 stays as taped.
- [ ] After each run look at the puck from above, as straight as the arm allows, before anything is touched, and
      write the label. "finished" needs all three: the puck reached the line's end; the robot stopped by itself; the
      puck still touches the end's ink (no white paper between the puck and the black end). If you cannot tell,
      write "finished?" and count it as finished; the photo decides at the desk.
- [ ] After R04: runs labelled "finished": ____ of 3.
- [ ] Time now, after R04: ____________ Minutes for the three runs: ______ This is your own pace today. Part 2 (a)
      and (b), which are never cut, take about five and a half times as long, plus 10 minutes for laying the sheet
      [calculated: 16 runs against 3]: ______ minutes.
- [ ] 2 or 3 finished, and that much time is left: go on to Part 2.
- [ ] 2 or 3 finished, but less time is left: end the visit here (section 6) and tell the planning session.
- [ ] 0 or 1 finished: drive X01, X02 and X03 as written below. Then end the visit (section 6) and report. Part 2 is
      not started before the planning session has answered.

**X01 to X03, only if 0 or 1 finished: the start as on 11.11, without the two aids.**

- [ ] Steps 1 to 3 as always: start, Enter 1, a hand ready at the arm, Enter 2, arm away, photo of the last run,
      puck off.
- [ ] Put the puck at the start of the line, as on 11.11.
- [ ] The question now reads `Guide the magnet tip just behind the puck, facing along the line. LET GO of the arm,
      then press Enter`. Do that; no placing line is shown. Then **Enter 3**.
- [ ] The driver prints `start: tip ... from the recorded start`, and under it a warning that the start is further
      off than 1 cm or 3 degrees and should be placed again. The warning is certain here: just behind the puck is
      2.5 to 4 cm from the recorded start [calculated]. In X01 to X03 it is ignored: no Esc.
- [ ] The picture window shows one still picture, without a faint copy. While it waits, Esc counts in the picture
      window only, not in the terminal. Hands off the table, click on it, **Enter 4**.
- [ ] Label each run as in R02 to R04.

## 4. Part 2, the step at the first bend of sheet 03: R05 to R28 (94 to 144 minutes)

24 short runs on sheet 03 from end A. The sheet is laid ONCE, in R05, and stays taped for all 24.

**Changing the sheet, inside steps 2 to 5 of R05.**

- [ ] Step 2 of R05: let the arm rest on the table beside the sheet, not on it.
- [ ] Step 3 of R05: the photo of R04's puck, the puck off. Then sheet 01 and its tape off the table.
- [ ] Lay sheet 03 roughly as in its recordings of plan 03-04: end A nearest the robot, the sheet turned so that its
      far end swings to the left (set-up picture: `...\setup_per_episode\plan_03-04.png`). End A is the end from which
      the line bends to the LEFT within its first 6 cm; from end B it first runs straight for about 12 cm.
- [ ] Step 4 as always. Step 5 of R05: slide and turn the sheet until the line and the paper edge lie on their faint
      copies, then tape it.
- [ ] The puck stands about 3 cm from the tip, ahead and to the left; the block nearly touches it. Do not touch the
      block when you set the puck.
- [ ] Time now, and is it daylight or dark outside? ____________________

**The runs.** All with the stall rule only logging. A short run is ended by its step budget (27 steps at 1.0 cm, 42 at
0.65 cm, 90 with the model's own length), usually before the model stops. That is the normal ending:
`ended: step_budget_reached`, and the label is "ended by the budget" if the puck was still on the line.

- [ ] (a) and (b), never cut: R05, R06, R07, R08, R09, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, in
      this order. The table of all runs says which model and which step each one has; the command sets them by itself.
- [ ] R06 is the first run with 0.65 cm steps on the real arm: watch as in R01.
- [ ] (c), taking turns: R21, R22, R23, R24, R25, R26, R27, R28. With the model's own length (R22, R24, R26, R28) the
      arm makes many small steps and may twist on the spot.
- [ ] At the robot you write one label per run and count nothing. Part 2 is scored at the desk.
- [ ] If time runs out: end after R20 at the earliest, and after that only after R22, R24, R26 or R28.

## 5. Part 3, sheet 04 again: R29 to R31 (19 to 25 minutes, then 7 to 17 for the extra runs)

Model E, sheet 04 from end A, fixed 1.0 cm, three short runs of 42 steps.

**Changing the sheet, inside steps 2 to 5 of R29**, as in R05:

- [ ] The arm rests beside the sheet; the photo of the last run's puck; the puck off; sheet 03 and its tape off the
      table.
- [ ] Lay sheet 04 roughly as in its recordings of plan 04-05: end A nearest the robot, the sheet lying a little to
      the right (set-up picture: `...\setup_per_episode\plan_04-05.png`). End A is the end from which the line bends
      to the RIGHT after about 3.5 cm and then sharply back to the LEFT.
- [ ] Step 5 of R29: slide the sheet onto its faint copies and tape it.
- [ ] The puck stands about 5.5 to 5.8 cm from the tip, ahead and a little to the left.
- [ ] Time now, and is it daylight or dark outside? ____________________

**The runs.**

- [ ] R29, R30 and R31, each by steps 1 to 7. In each run watch for one thing more: does the puck stay on the line
      through both turns of the first S-bend (first to the right, then sharply to the left) and along the short
      straight piece after them, up to where the next bend begins? The budget lets the run go on some way further;
      what happens there does not count here. Write "S-bend: yes" or "S-bend: no" beside the label.
- [ ] After R31: runs with "S-bend: yes": ____ of 3. This is your count by eye; the scoring at the desk decides for
      good.
- [ ] 3 of 3: drive X07, one full run. It ends when the model stops; Esc by the rule above.
- [ ] 1 or 2 of 3: drive X04, X05 and X06, three more short runs.
- [ ] 0 of 3: no more runs.

## 6. After the visit (about 10 minutes)

- [ ] The last run's photo: one hand ready at the arm, motors off (FreeMotion), guide the arm away from the puck,
      take the photo with the ruler as in step 3.
- [ ] Motors off before the arm is parked by hand. Park the arm and switch the robot off as on every robot day.
- [ ] The run folders stay where they are, in `D:\AI_Project\03_data\model_runs`, one per run, named by date, time
      and label. Nothing in them is renamed, moved or deleted.
- [ ] All photos are kept.
- [ ] Tell the planning session "visit done" and give it: your labels, your notes, the times and the light you wrote
      down, the photos, and everything that looked different from this list.

## All runs

The models. **E**: model E of 11.11, trained on all six sheets. **E without 03**: the same recipe trained without
sheet 03, so it has never seen that sheet; the screen calls it `E recipe trained without sheet 03`. **Team of five**:
five networks of model 1 whose answers are averaged; the screen calls it `team of five (B)`.
"Full run" means up to 250 steps: the run ends when the model stops by itself. A number alone is the most steps of a
short run.

| Done | Run | Sheet | Model | Step, and most steps | After it | Your label |
|---|---|---|---|---|---|---|
| [ ] | R00 | 01 | E | 1.0 cm, full run | Warm-up, never scored. The sheet was laid and taped in this run. | |
| [ ] | R01 | 01 | E | own length, 40, NO PUCK | Write "smooth: yes or no" in section 2. If no: ask before Part 2. | |
| [ ] | R02 | 01 | E | 1.0 cm, full run | Label. | |
| [ ] | R03 | 01 | E | 1.0 cm, full run | Label. | |
| [ ] | R04 | 01 | E | 1.0 cm, full run | Label. Count "finished" in R02 to R04: with 2 or 3 go to R05, with 0 or 1 drive X01 to X03. | |
| [ ] | X01 | 01 | E | 1.0 cm, full run, start as on 11.11 | Only if 0 or 1 finished. Label. | |
| [ ] | X02 | 01 | E | 1.0 cm, full run, start as on 11.11 | Label. | |
| [ ] | X03 | 01 | E | 1.0 cm, full run, start as on 11.11 | Label. Then end the visit and report. | |
| [ ] | R05 | 03 | E | 1.0 cm, 27 | (a) pair 1. The sheet was laid ONCE and taped in this run. Label. | |
| [ ] | R06 | 03 | E | 0.65 cm, 42 | (a) pair 1. Label. | |
| [ ] | R07 | 03 | E without 03 | 1.0 cm, 27 | (b) pair 1. Label. | |
| [ ] | R08 | 03 | E without 03 | 0.65 cm, 42 | (b) pair 1. Label. | |
| [ ] | R09 | 03 | E | 0.65 cm, 42 | (a) pair 2. Label. | |
| [ ] | R10 | 03 | E | 1.0 cm, 27 | (a) pair 2. Label. | |
| [ ] | R11 | 03 | E | 1.0 cm, 27 | (a) pair 3. Label. | |
| [ ] | R12 | 03 | E | 0.65 cm, 42 | (a) pair 3. Label. | |
| [ ] | R13 | 03 | E without 03 | 1.0 cm, 27 | (b) pair 2. Label. | |
| [ ] | R14 | 03 | E without 03 | 0.65 cm, 42 | (b) pair 2. Label. | |
| [ ] | R15 | 03 | E | 1.0 cm, 27 | (a) pair 4. Label. | |
| [ ] | R16 | 03 | E | 0.65 cm, 42 | (a) pair 4. Label. | |
| [ ] | R17 | 03 | E | 0.65 cm, 42 | (a) pair 5. Label. | |
| [ ] | R18 | 03 | E | 1.0 cm, 27 | (a) pair 5. Label. | |
| [ ] | R19 | 03 | E without 03 | 0.65 cm, 42 | (b) pair 3. Label. | |
| [ ] | R20 | 03 | E without 03 | 1.0 cm, 27 | (b) pair 3. Label. (a) and (b) are complete: the earliest point to end Part 2. | |
| [ ] | R21 | 03 | Team of five | 1.0 cm, 27 | (c). Label. | |
| [ ] | R22 | 03 | E | own length, 90 | (c). Label. You may end Part 2 here. | |
| [ ] | R23 | 03 | Team of five | 1.0 cm, 27 | (c). Label. | |
| [ ] | R24 | 03 | E | own length, 90 | (c). Label. You may end Part 2 here. | |
| [ ] | R25 | 03 | Team of five | 1.0 cm, 27 | (c). Label. | |
| [ ] | R26 | 03 | E | own length, 90 | (c). Label. You may end Part 2 here. | |
| [ ] | R27 | 03 | Team of five | 1.0 cm, 27 | (c). Label. | |
| [ ] | R28 | 03 | E | own length, 90 | (c). Label. Part 2 is complete. The next run is on sheet 04. | |
| [ ] | R29 | 04 | E | 1.0 cm, 42 | The sheet was laid and taped in this run. Label, and "S-bend: yes" or "no". | |
| [ ] | R30 | 04 | E | 1.0 cm, 42 | Label, and "S-bend: yes" or "no". | |
| [ ] | R31 | 04 | E | 1.0 cm, 42 | Label. Count "S-bend: yes" in R29 to R31: with 3 drive X07, with 1 or 2 drive X04 to X06, with 0 go to section 6. | |
| [ ] | X04 | 04 | E | 1.0 cm, 42 | Only if 1 or 2. Label, and "S-bend: yes" or "no". | |
| [ ] | X05 | 04 | E | 1.0 cm, 42 | Label, and "S-bend: yes" or "no". | |
| [ ] | X06 | 04 | E | 1.0 cm, 42 | Label, and "S-bend: yes" or "no". | |
| [ ] | X07 | 04 | E | 1.0 cm, full run | Only if 3. Label. | |

Every run except X01 to X03 uses the placing line and the faint picture. In every run the stall rule only logs.

## The rules, written before the visit

Nothing here is decided at the robot. At the robot you write one label per run; the planning session scores the runs
afterwards at the desk, from the logged pictures and your photos.

**Decided by the planning session on 12.11.26, before the visit** (points the two readers of this list left open):

- The run labels are R00 to R31 ("R" for run), so that they cannot be mixed up with the track prints T01 to T16.
- In later runs on a taped sheet the puck stands on the start of the real line; where the faint copy differs a
  little, the line wins. When the line and its faint copy do not overlap, the arm is placed again.
- The counts made by eye stand as written: "finished?" counts as finished in Part 1, and "S-bend: yes" is counted
  in Part 3. The scoring at the desk decides for good.
- The coin's order stands as written: one toss per pair in the order the pairs are driven.
- X01 to X03 are driven without the placing line and without the faint picture.
- A start that is still "not close" after about three minutes is driven with its numbers written beside the run. If
  R01 does not look smooth, Part 1 is still driven and Part 2 waits for an answer. If the placing line fails in two
  runs in a row, the visit ends.
- If the visit has to stop for the day: stop after the second run of a pair. The pairs driven so far count. On the
  next day the set-up and the warm-up (R00 with ` --again`) are done again, the sheet is laid again by its faint
  picture, and the remaining pairs are driven; the planning session says so again when asked.

**Part 1, the control.** A run counts as finished when the puck reached the line's end, the robot stopped by itself,
and the puck still touches the end's ink: the centres of the puck and of the line's end are at most 17 mm apart, read
from the photo with the ruler. The 12 mm and the 7 mm reading are written beside it (at 12 mm the puck covers the
centre of the line's end, at 7 mm it covers the whole rounded end) [calculated from the puck's 24 mm and the line's
10 mm]. Fewer than 2 of 3 finished: three more runs with the tip placed just behind the puck as on 11.11, without the
placing line, to tell a fault of the new start from anything else; then stop and report before Part 2.

**Part 2, the step at the first bend of sheet 03.** Short runs, 24 in all, all with the stall rule only logging.
(a) Model E, fixed 1.0 cm (27 steps) against fixed 0.65 cm (42 steps), five pairs. (b) The model trained without
sheet 03, fixed 1.0 and 0.65 cm, three pairs, placed after pairs 1, 3 and 5 of (a). (c) Then, taking turns, the team of
five with fixed 1.0 cm (27 steps), four runs, and model E with its own length (90 steps), four runs.

- A run keeps the line when the puck's centre is never more than 12 mm from the centre line over 10 mm or more of tip
  travel, from the start until the puck has passed 119 mm of line, counted from the cap centre of end A. That mark is
  8 cm past the first bend's apex, which lies at 39 mm.
- A run that its own stop or a refused step ends before that mark counts as not kept and keeps its label.
- A run that the budget ends before the mark is scored on the part it covered and marked "window not completed".
- A run that you ended with Esc before the mark is scored on the part it covered, as a run that the budget ended,
  and marked "ended by hand". Write beside it why you ended it [decided by the planning session on 12.11.26,
  before the visit].
- 0.65 cm goes on only if it keeps the line in at least 3 more runs than 1.0 cm, counted on model E's five pairs. A
  step that is no better passes this by luck in at most 1 visit of 18 [calculated by the proofreader].
- The own length replaces 0.65 cm only if 0.65 cm has itself passed and the own length keeps the line in all 4 runs
  while 0.65 cm loses it in 2 or more of 5.
- If 0.65 cm has not passed, the own length goes on only if it keeps the line in all 4 runs while 1.0 cm keeps it in
  at most 1 of 5.
- If no step passes, the step stays 1.0 cm, and 0.65 cm is driven again in the next sitting as the challenger.
- The team's four runs and the six runs of the model that never saw the sheet decide nothing today; they are written
  down.
- If time runs out: stop after (b) at the earliest; (a) and (b) are never cut.

**The coin for Part 2.** Which step is driven first in each of the eight pairs was fixed when this list was written,
not at the robot: Python's `random.Random(20261112)`, so the seed is 20261112, and one `choice(["1.0", "0.65"])` per
pair in the order the pairs are driven. It gave 1.0, 1.0, 0.65, 1.0, 1.0, 1.0, 0.65, 0.65, which is the order in the
table of all runs. To see it again, from 02_code:
`..\..\..\..\.venv\Scripts\python.exe -c "import random; coin = random.Random(20261112); print([coin.choice(['1.0', '0.65']) for pair in range(8)])"`

**Part 3, sheet 04 again.** Model E, fixed 1.0 cm, three short runs of 42 steps with the stall rule only logging, the
sheet laid with the faint picture of plan 04-05 and taped. A run is through the first S-bend when it keeps the line, by
the rule of Part 2, until the puck has passed 160 mm of line from the cap centre of end A. 3 of 3: the failure of 11.11
was mostly start and placement. 0 of 3: it was not the placement. 1 or 2 of 3: three more runs before anything is
said. One full run (250 steps) only if all three kept the line. At the robot the count is made by eye: did the puck
stay on the line through both turns of the first S-bend and along the straight piece after them? The mark at 160 mm
lies at the end of that straight piece, which runs from 128 to 163 mm [measured on the traced line by the
proofreader]. The desk scoring decides.

**Why the budgets are what they are** [measured on the recordings by the proofreader]. On sheet 03 the eight recorded
standard starts needed 16.2 to 21.3 cm of tip travel to bring the puck past the 119 mm mark: 27 steps of 1.0 cm are
27 cm, 42 steps of 0.65 cm are 27.3 cm. On sheet 04 the eleven recorded standard starts needed 27.9 to 36.2 cm to
bring it past 160 mm: 42 steps of 1.0 cm are 42 cm. The 90 steps of the own length are a guess.

## If something goes wrong

**The driver prints that the arm is still low.**

- After a run: `THE ARM IS STILL LOW: the lift after the run failed (...)`. The arm stands low, held by its motors,
  perhaps against the puck. Do not pull it up. Start the next run as usual, with a hand ready at the arm: at Enter 2
  the driver switches the motors off, and then you lift the arm by hand. Write "lift failed" beside the run. If the
  next run does not start: a hand ready at the arm, motors off (FreeMotion), lift it by hand, end the visit and
  report.
- Before a run: `not started: the live window stopped (...). THE ARM IS STILL LOW, held by its motors`. Nothing was
  lifted. Start the same label again, with a hand ready at the arm before Enter 2.
- `THE MOTORS ARE STILL OFF. Hold the arm, set it down safely, LET GO, then press Enter to switch them on`: do
  exactly that. The arm is then held where it rests. Start the same label again, with a hand ready at the arm before
  Enter 2.

**A run does not start.** Read the line that begins `not started:`. Except in the first case below, the label has no
run folder yet, so the same command works again without ` --again`.

- `the label R07 already has a run folder`: it was driven before. Take the next label; or, for a void run, add
  ` --again`.
- `the run list has no line with the label ...`: a typing mistake. A label is a capital letter and two digits: R07.
- Red lines that say `... is not recognized as the name of a cmdlet ...`: this window is not in the folder 02_code.
  Type the `cd` line and the `$list` line of section 0 again.
- `usage: visit_runs.py ...`, or `the run list ... cannot be read`: the short name is not set in this window. Type
  the line `$list = ...` of section 0 again.
- A checkpoint file, a metrics file or a reference picture is `missing` or `not the one the list means`: a file on the
  laptop has changed since the list was checked. Do not drive that line; end the visit and report.
- After Enter 3, `the tip is ... from pushing height; place it again just above the sheet` (the placing line ended in
  `too high` or `too low`): the arm is held by its motors where it was. Start the same label again, with a hand
  ready at the arm before Enter 2, and let the arm rest on its rod when you place it.
- After Enter 3, `outside the allowed reach`, `outside the workspace`, `the forearm is turned ...`, `the rod would
  lean too much` or `cannot be reached with the wrist and hand held`: the arm did not stand at the recorded start.
  Start the same label again and place it until the line says `close`.
- Lines that begin with `Traceback` after Enter 1, before the first question, and at the end `model_mode.py ended
  with exit code 1` and `no run folder: the run was not started`: the driver did not get as far as the motors. Read
  the last line above `model_mode.py ended`. If it says `the robot needs its automatic calibration first`: run the
  calibration line of section 0, then start the same label again. Anything else: the robot was not reached. Is it
  on, is the cable in, is every other program closed? Start the same label again; if it comes a second time, stop
  and ask.
- `nothing moved by the model; ending`: you pressed Esc in the picture window. Start the same label again.

**The picture window does not react.**

- Enter does nothing: click on the picture first. Enter 4 counts in the picture window only, not in the terminal.
- The dark band reads `NO CAMERA PICTURE: Esc = do not start`: no camera picture arrives, and Enter does not count.
  Wait ten seconds: if the live picture comes back by itself, go on. If not: hands away, Esc; the arm lifts 5 cm.
  Start the same label again. If it comes again: end the visit; do not touch the camera or its cable.
- No window is to be seen: it is named `record_episodes` and stays on top of other windows; look in the taskbar. If
  none has come 20 seconds after Enter 3, press Esc in the terminal: the arm lifts 5 cm and nothing is logged. (In
  X01 to X03 Esc in the terminal does not count at this point: if no window comes, close the PowerShell window as
  the last line of this block says.)
- Esc does nothing during a run: click on the picture and press Esc again, or press it in the terminal. If the arm
  does something unsafe: the stop button.
- The picture stands still, nothing new is printed for more than 10 seconds and Esc does not help. (While a run
  pushes, the terminal prints nothing: that alone is normal.) Look first at the title of the PowerShell window. If
  it begins with "Select", a click inside its text has frozen it: click on its title bar and press Esc once; the
  run goes on. If not, the link to the robot is stuck: close the PowerShell window. The arm stays as it is: held by
  its motors, or soft if this happened while you were placing it. Open a new PowerShell, `cd` to 02_code, type the
  `$list` line, start the run again (with ` --again` if it had begun to push), with a hand ready at the arm before
  Enter 2.

**The placing line does not come within 1 cm.**

- Check the direction. The line says where the tip IS: "too far left" means move it to the right, as seen from behind
  the robot. Move it a little and watch whether the number shrinks.
- One thing at a time: first out or close, then left or right, then the twist of the forearm.
- The arm sags a little when you let go. Let go, read the line, correct once more.
- The line ends in `too high` or `too low`: the tip is more than 2 cm from pushing height. Let the arm rest on its
  rod.
- Still `not close` after about three minutes: write the line's numbers beside the run (for example "tip 1.4,
  forearm +2.0"), press Enter 3 and drive the run; the driver's warning about the start comes and is ignored this
  once. The driver writes the same numbers into the run's folder; the planning session decides what the run counts
  for.
- The numbers do not change while you move the arm. If the window's title begins with "Select", a click inside the
  text has frozen it: press Esc once (only now, while the placing line is on the screen). If the driver prints
  `the placing aid stopped (...)`, it asks its plain question instead: set the arm down at the start as well as you
  can, let go, Enter 3, and read the line `start: tip ... from the recorded start` that follows. If a warning comes
  with it (further off than 1 cm or 3 degrees), press Esc in the picture window and start the same label again. If
  the placing line fails in two runs in a row: end the visit and report.
- `(this terminal is ... columns wide and cuts the end of the line below off ...)`: maximise the window.
- `the placing aid reads Enter from a console window ... and this is not one`: this is not PowerShell. Go on to the
  picture window, press Esc there, and start the label again from PowerShell.

**Other things.**

- Something or somebody came into the camera's view during a run, or the connection broke (`error: ...`, or
  `model_mode.py ended with exit code ...`): the run is void. Repeat it at once with ` --again`.
- The wrong label was started. Noticed before Enter 4: go on to the picture window and press Esc there; nothing is
  logged. Noticed later: that run is void; go on with the label that was due, and drive the mistaken label with
  ` --again` when its turn comes.
- The taped sheet has moved: do not push it back by eye. In step 5 of the next run slide it onto its faint copies
  again, tape it, and write "sheet laid again before T.." beside that run.
- The stop button was pressed: the visit pauses. Restart the robot as on every robot day, with the automatic
  calibration only, and begin again at "At the robot" in section 0.
- `keys pressed before this question were ignored` or `keys pressed before this prompt were ignored`: a key was
  pressed too early and thrown away. Nothing happened; go on.
- `the picture window had gone; opened it again`: the window was closed or hidden and came back by itself. Go on.
- Anything else that looks different from this list: Esc, write it down, ask.

## What is measured, and what is not

- Measured without the robot on 12.11.26: every line of the run list passes the dry check (39 of 39), and both test
  files end with "all tests passed". The three models are the files of 11.11: their checksums stand in the run list
  and are checked before every run, and model E and the team give back the answers logged on 11.11 [measured by the
  proofreader]. A run without the new switches does what the driver did before them [measured by the builder on a
  stand-in for the robot, 23 cases].
- Measured on the simulated robot only: the placing line, the faint picture, the 0.65 cm step, the stall rule only
  logging, and the keys. No person has pressed the keys yet. The questions and lines of the driver that this list
  quotes were compared on 12.11.26 with what the driver prints in a run on a stand-in for the robot [measured by a
  second reader]; the rehearsal line of section 0 ran there to its end, with a stand-in for the keyboard.
- Never tried at the real robot: all of these, and steps shorter than 1 cm with this driver. Never tried by a
  person: the order of the hand's work in this list (the photo at the start of the next run, the arm resting on its
  rod while both hands are free, the sheet slid under the held arm).
- Measured from the recordings' pictures, without a ruler: where the puck stands at the start.
- Calculated: the minutes of the runs; the marks at 119 mm and 160 mm, from the traced lines.
- Guessed: the minutes of set-up, of laying a sheet and of the end, and the 5 minutes between runs; the table
  margins of 55 and 35 cm; that the tip can be placed within 1 cm by hand in a minute or so; that the picture
  window takes the keyboard by itself; the 90 steps of the own-length runs.
