# First robot test of the recording tool (planned 30.10.26)

Written 29.10.26 by Marshal's session. Goal: check that `02_code/record_episodes.py` works on the real Ned2, with
sheet 01 (straight line), the puck, and the camera as it is now. This is a test, not real data collection: the episodes
are useful practice, but real training data waits for the new camera mount.

What is already checked: both modes pass the automatic tests on a simulated robot (29.10.26). What is NOT checked yet
and is the point of tomorrow: whether "left" in the picture is really left (`picture_left_sign` in `config.py`, a guess),
whether every logged photo is fresh on the real camera, whether the lowering and lifting heights are right, and whether
Python 3.13 on Marshal's Mac talks to the robot (earlier robot scripts ran on Emranul's Python 3.11).

Two people make it easier: one at the keyboard, one beside the robot with the Stop button in reach (and, in
hand-guided mode, holding the arm).

## Stop rules (read before starting)

- Anything unexpected (the tip touching or dragging on the paper, the arm heading for the table edge, a strange noise):
  press the robot's **Stop button**.
- **Ctrl-C** in the terminal stops the script, but the tool then sends the robot **home**. Use it only when going home
  is safe; otherwise use the Stop button.
- Never touch the arm while the motors are on. Never push on a held arm. Hands on the arm only after the tool says
  "motors OFF".
- Never touch the camera or its focus ring. No manual calibration.
- Only one program connected to the robot at a time: close `view_camera.py` before starting the recording tool.

## 0. The evening before or at home (Mac)

- [ ] `git pull` on main, so the latest tool is on the laptop.
- [ ] From `02_code`: `../.venv/bin/python test_record_episodes.py` prints "all tests passed".
- [ ] Pack the Ethernet cable (and a USB-C adapter if the Mac needs one).
- [ ] Check the robot's address in `02_code/config.py` (`ROBOT_PARAMETERS["ip_address"]`): `169.254.200.200` over the
      Ethernet cable, `10.10.10.10` over the robot's Wi-Fi hotspot. Change it if you connect the other way.

## 1. Set up the desk (robot off)

- [ ] Sheet 01 taped down as before: 5 cm from the front of the base, the line pointing straight away from the robot
      (photo `06_deliverables/team_m_project6_documentation/figures/setup_2026-10-25_overhead.jpg`).
- [ ] The robot stands where it stood on 25.10, away from the table edge. If anything moved, the taught positions are
      wrong: run `teach_positions.py` (six touches, five minutes) before anything else.
- [ ] Magnet block snapped onto the rod. Camera untouched.
- [ ] Puck beside the sheet, not on it yet.

## 2. Power on and connect (nothing moves)

- [ ] Switch the robot on and wait until its start-up and **automatic** calibration have finished.
- [ ] Connect the cable or the hotspot.
- [ ] From `02_code`: `../.venv/bin/python check_hardware.py`.
      Pass: it prints the pose and the joints with no error. If it fails, stop here and write down the error.
- [ ] `../.venv/bin/python view_camera.py`. Pass: a live picture. Press `q` to close it (it must be closed before step 3).

## 3. Direction test, no puck (the most important check)

- [ ] Clear the sheet; keep the puck off it.
- [ ] Start: `../.venv/bin/python record_episodes.py --operator marshal --mode keyboard`
      (use your own name after `--operator`).
- [ ] The robot moves to its home pose. Press **Enter** in the terminal: the robot goes to 5 cm above the line start,
      then comes down. Pass: the tip stops about **5 mm above the paper** and does not touch it.
- [ ] A window shows the camera picture. **Click the window** so it gets the key presses.
- [ ] Press `w` once (forward). Pass: the line in the picture moves **down** (towards the bottom of the picture), and
      the picture visibly changes.
- [ ] Press `a` once (left). Pass: the scene moves **to the right** in the picture.
      If it moves to the LEFT instead: press `x`, then `Esc` (the robot goes home), set `"picture_left_sign": -1` in
      `config.py` and repeat this step. Write down which it was.
- [ ] Press `d` once (right): the scene moves back.
- [ ] Watch the terminal: no line saying "warning: step N has the same picture as the step before".
- [ ] Press `w` twice quickly while the robot is still moving: the terminal must report that a key press made during
      the move was ignored, and only one step must follow (key drain, added 31.10).
- [ ] Press `x` (discard: the episode stays on disk, marked discarded).

## 4. Keyboard episodes with the puck (three)

- [ ] Robot lifted (it lifts after every episode). Put the puck on the **start of the line**, centred on it.
- [ ] Press **Enter**. The robot comes down behind the puck. Pass: the tip lands behind the puck, not on it.
- [ ] Push the puck along the line, one key per step: `w` forward, `q` forward-left, `e` forward-right
      (`a`, `d`, `s` if needed). Expect to correct with `q` or `e` now and then: on this sheet the line runs about
      10 degrees off the picture's "forward" at the start.
- [ ] At the goal, press **Space** (stop). Pass: the terminal says "completed", and the puck is within about 1.2 cm of the
      line's end.
- [ ] Repeat for three episodes. Note anything odd (the puck spinning away, the tip catching the puck's top).
- [ ] Each time you wish you had back-left, back-right or a hop over the puck, note what the picture showed. That note
      decides whether those actions are added (decision of 29.10.26, `STATUS.md`).

## 5. One hand-guided episode

- [ ] Press `Esc` to end the keyboard session (the robot goes home), then start
      `../.venv/bin/python record_episodes.py --operator marshal --mode hand_guided`.
- [ ] Puck on the line start, press **Enter**: the robot comes down behind the puck.
- [ ] The person beside the robot **holds the arm at the elbow first**. Then press **Enter**: the motors go OFF and the
      arm sags into the holding hand.
- [ ] Guide the pusher so it pushes the puck smoothly along the line to the goal. Keep the tip low, just above the paper.
- [ ] At the goal, press **Space**. The tool says "MOTORS OFF. Let go of the arm completely, then press Enter": let go
      first, then press **Enter**, and the motors come back ON (changed 31.10; the same prompt appears after Ctrl-C or
      any error while guiding, so the arm is never powered against a hand).
- [ ] The robot lifts. Put the puck back on the line start and press **Enter**.
- [ ] The robot replays your path in 1 cm steps, showing its planned step each time and waiting: press **Enter** to run
      the shown step, or a direction key to do a different one instead; `x` discards (changed 31.10: no step runs
      without a key). Pass: "completed", the puck at the goal.

## 6. Check the files (Mac, robot can stay on)

- [ ] Press `Esc` to finish: the robot goes home and the tool disconnects.
- [ ] Open `03_data/raw/<today>_session_01_<operator>/`. One folder per episode: `frames/` (one photo per step), `steps.csv`,
      `episode.json`, plus `guided_path.csv` for the hand-guided one. `session_notes.md` lists the episodes.
- [ ] Open a few photos: sharp, the puck at the bottom, the line ahead, no hand in view (replay photos).
- [ ] Copy the session folder to a second disk the same day (README rule 3).

## 7. Afterwards

- [ ] Write the results into the session's `session_notes.md` (append only): direction test result, anything odd.
- [ ] Tell Claude the results: it updates `config.py` (the `picture_left_sign` value, marked tested), `STATUS.md` and the
      shared memory, on a branch, so Emranul sees them.
