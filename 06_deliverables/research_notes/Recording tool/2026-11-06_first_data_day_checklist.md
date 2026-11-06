# First data day, 6.11.26: checklist at the robot

For Emranul and whoever helps (two people are much safer than one). Work through it from the top and tick each box.
Claude (the Recording and data session) does the laptop parts marked "Claude": tell it in the chat when you reach them.
Times are guesses. Rules that always hold: only one program talks to the robot at a time; motors off before anyone
touches the arm; never push on an arm while its motors are on; the robot stays away from the table's edges; the camera
and its focus ring are not touched; only the automatic calibration after a restart.

## A. Before anything moves (laptop, 5 minutes)

- [ ] A1. Claude: fetch the repository, merge `main`, run the simulator tests (`test_record_episodes.py`,
      `test_puck_tracking.py`). Nothing starts until they pass.
- [ ] A2. Close NiryoStudio and every camera viewer.

## B. Moving the robot to the first-floor table (two people, 15 minutes)

- [ ] B1. If a tool is still running: Esc at the "Put the puck at the start" prompt; the robot parks. Switch the robot
      off.
- [ ] B2. Unplug power and Ethernet at the base. Leave the camera's USB cable plugged into the robot; tape any loose loop so
      nothing can catch it (the camera is held only by three strips of tape).
- [ ] B3. Leave the magnet block on the rod.
- [ ] B4. Carry: one person lifts the robot by its base with both hands; the other supports the blue upper arm so the arm
      cannot swing. Nobody holds the black forearm cover, the camera, its cable, the hand or the block. Upright, slowly.

## C. Setting up on the new table (20 minutes)

- [ ] C1. The robot well away from every table edge (course rule), with free table in front of it. Mark the base's
      outline on the table with tape, so that a shift later is easy to see.
- [ ] C2. Clear about 1 m around the robot: no white things (they look like paper) and no red things (they look like the
      puck: pens, mugs, the red-wheeled mouse). The camera turns with the forearm and sees far beyond the sheet.
- [ ] C3. Write down the light: which lamps, blinds, daylight, time. Keep it the same all day if you can.
- [ ] C4. The START MARK: a small tape cross on the table 8 cm in front of the middle of the base's front edge (that is
      about 17 cm from the robot's centre). Every sheet's line will start on this cross. A long strip of tape from the
      base straight forward through the cross helps to lay the sheets straight.
- [ ] C5. Ethernet to the laptop, power on, wait for the start-up. Then (nothing moves):
      `D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/check_hardware.py`
      It must say "Calibration needed: False". If it says True: table clear, puck away, nobody touching, Stop button in
      reach, then `D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/calibrate_robot.py` (the arm moves by
      itself; watch the camera cable).

## D. Teaching the new table (Claude helps, 15 minutes)

Everything taught on the old desk (table height, sheet corners, start) is wrong on the new table until this is done. A
wrong table height can press the block into the paper.

- [ ] D1. Lay sheet 01 with its line's start (the round end cap) on the START MARK and the line pointing straight away
      from the robot along the tape strip. Tape all four corners.
- [ ] D2. In a terminal: `D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/teach_positions.py`
      Hold the arm at the elbow, press Enter (motors off). With the MAGNET TIP touching the paper and the rod upright,
      touch in this order, Enter after each: line start, line end, near-left corner, near-right corner, far-right corner,
      far-left corner. For the line start touch where the black ink BEGINS (the outer edge of the round end, nearest the
      robot), not the middle of the round end: the robot later comes down 1.85 cm behind the taught point, which only
      just touches a puck standing on it; with the puck centred on the round end there is then about half a centimetre
      to spare, as on 4 and 5.11. Keep holding the arm until it says the motors are on.
- [ ] D3. Tell Claude at once (a second teach run the same day overwrites the file). Claude turns the touches into the
      new table height, start, sheet corners and table-edge guard in `config.py`, runs the tests and commits. Restart any
      open tool afterwards.
- [ ] D4. Camera check (did carrying move the camera?). The arm goes where it went on 4 and 5.11; nothing needs to lie
      under it:
      `D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/record_episodes.py --operator emranul --mode keyboard --path-id camera_check --start-x 0.1707 --start-y 0.0374 --orientation "camera check after the move" --lighting "<your light note>"`
      Click the picture window, Enter. The terminal prints "camera check: ... N px from the reference". Under about 3 px:
      fine. 3 to 4 px: tell Claude. Over 4 px the tool asks: press x and tell Claude (the camera is re-taped to its
      reference picture, or recorded under a new mount name; the puck map then needs a refit). Then Esc.
- [ ] D5. Dry run without the puck (checks the new table height along the line): start the keyboard tool on sheet 01 with
      the start Claude gives you, Enter, then w five to ten times, one a, one d. The tip must stay just clear of the paper
      (about 5 mm) and never drag. Then x, m, Esc.

## E. Trial episodes: one per sheet, then Claude checks (about 2 hours)

The trials test the new set-up; they are not the training data yet. Order: 01, 04, 02, 03, 05, 06.

- [ ] E1. Lay the sheet with its start cap on the START MARK and the line's first centimetres pointing straight away from
      the robot, as in `figures/2026-11-06_rule_a_placements.png` (the sheet is turned on the table so that this works).
      Start ends: any end for 01 to 04 (no sheet marks a start; 04 as on 4 and 5.11), the end near the bottom edge for 05,
      and for 06 the end halfway up the sheet near its right edge in the course photo (not the one near the bottom edge).
      Tape the corners and take a phone photo from above.
- [ ] E2. Claude gives you one ready command per sheet (the right `--path-id`, the start from D3, an `--orientation` note
      ending in "trial").
- [ ] E3. One `hand_guided_exact` episode (see "One episode" below).
- [ ] E4. On sheet 01 only: one more trial with `--mode hand_guided` (the 1 cm replay), to compare the two hand-guided
      modes (see "Decide before the real data").
- [ ] E5. Esc, then tell Claude the episode names. Claude runs the episode checker and looks at the pictures: complete,
      clean start picture, no hands, puck in view, camera steady, time per episode.

## One episode in `hand_guided_exact`

Keys go to the PICTURE window (click it first).

1. "Put the puck at the start of the line": puck centred on the start cap, HANDS OUT OF THE PICTURE, then Enter. The
   first picture is taken right away and cannot be taken again.
2. The robot comes down behind the puck and checks the camera.
3. "Hold the arm at the elbow": hold it first, then Enter: motors off.
4. Guide the block so it pushes the puck along the line to the end cap. Smooth, steady, the same way every time (rules in
   the episode plan). Space when the puck is in the goal; x if it went wrong (then answer why).
5. Let go of the arm completely, then Enter: motors on. The arm lifts.
6. Put the puck back exactly where it started (on the start cap), hands away, Enter.
7. The robot replays the guided path. Watch the first few waypoints with Enter, then g to let it run. Any key pauses. h =
   correct by hand when the puck drifts; u = the last step was a mistake; l = the line is not in this picture.
8. After the last waypoint: Enter. "Puck in the goal?" Answer y or n. Do not press Esc here (the answer would be lost).

## Decide before the real data (Emranul and Marshal, with Claude)

The planned model (Model and training in STATUS.md) picks one of the 1 cm step classes from one picture, and the robot
then makes that step with the rod upright. In the exact replay the robot copies every joint of the hand: the rod leans
16 to 22 degrees (median), the forearm twists a little at almost every step (never labelled as a turn), 13 to 41 percent of
the rows are smaller than 3 mm, and the label is only the nearest class (measured on 039, 042, 043). A model trained on
those pictures would later see different pictures from its own upright steps. Two clean ways out:

1. Keep the 1 cm classes and record with a mode whose robot moves ARE the labels: `hand_guided` (you guide by hand as now,
   the robot replays in 1 cm steps with the rod upright; Enter for each step) or keyboard or gamepad.
2. Keep `hand_guided_exact` and change the model to predict the next motion itself (forward, left, forearm turn) as
   numbers or fine classes; the robot then makes that motion with the gamepad mode's solver.

Claude's lean: option 1 for the first model, because it keeps what the robot does at the exam identical to the training
data, and it is the method already chosen in STATUS.md. Today's trials are useful either way. Please do not record many
more than the trials before the two of you have chosen.

## F. End of the day (15 minutes)

- [ ] F1. Backup: nothing in `03_data` is in git. Copy the day's session to a USB stick or drive (E: here), in PowerShell:
      `robocopy D:\AI_Project\03_data E:\AI_Project_backup\03_data /E /XO /R:1 /W:1`
      (copies new and changed files, never deletes anything in the backup).
- [ ] F2. For Marshal: zip the day's session folder, `03_data\raw\camera_check\camera_reference` and
      `03_data\raw\workspace`, and put the zip in a cloud folder you both can open.
- [ ] F3. Claude: session notes, STATUS.md, memory, commit; push only after you say yes.
