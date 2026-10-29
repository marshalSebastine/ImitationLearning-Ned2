# STATUS: where the project stands

The one live state file. Every Claude session reads it at the start and updates its own workstream section at the end
("update STATUS.md and memory with what changed"). Keep it under 150 lines; details go into the notes it points to.
Full background for a newcomer: `HANDOFF_2026-10-26.md`. How the sessions are organised: `HOW_WE_WORK.md`.

Last updated: 29.10.26 by Marshal's session (recording tool merged into main, not yet tested on the robot; action list kept at seven).

## Dates that cannot move

- Meeting 2: 9 Nov 2026, 09:50, BigBlueButton. Meeting 3: 23 Nov. Meeting 4: 7 Dec. Attendance mandatory.
- Exam registration in PRIMUSS: 9 to 17 Dec 2026. Our freeze: 8 Jan 2027. Submission: 10 Jan 2027 23:59. Presentation: 11 Jan.

## Orchestrator (planning, meetings, this file)

- Done: meeting 1 held and recorded; questions for meeting 2 listed in `05_meetings/requests_to_professor.md` (7, 10, 13 to 16);
  handoff bundle for Marshal built on 26.10 (`D:\AI_Project_handoff_2026-10-26.zip`, lines marked "[Emranul: fill in]" still open).
- Done 27.10: orchestrator session running in the sidebar group "AI Project (Team M)"; briefs written for three
  workstream sessions (Robot and hardware: measurement-hour sheet and mock-up checklist; Recording and data: recording-tool
  design for approval before code; Questions: answers only, never edits). All three started on 27.10 and sit in the same sidebar group.
- Done 27.10 (git set-up, Emranul's laptop): Git and the GitHub CLI installed, signed in as `emrantawsif`; the shared repo
  cloned to `D:\AI_Project` (the pre-git folder kept as `D:\AI_Project_pre_git`); `.venv` and `03_data` copied over;
  `setup_claude.ps1` run, so Claude's memory and the code-style skill now live in the repo (`claude_memory/`,
  `claude_skills/`); the laptop's work from 27.10 merged into `main` by pull request 1. All sessions now follow
  `COLLABORATION.md`: branch per topic, pull request into `main`, commits under the students' names.
- Next: read the three sessions' reports and their `STATUS.md` sections; fill in the remaining "[Emranul: fill in]"
  lines of `HANDOFF_2026-10-26.md` (Emranul gives the facts; only the git-hosting line is filled); agree who does what
  (provisional table in the meeting-1 note); start the contribution log.
- Blocked on: nothing.

## Robot and hardware (camera mount, measurements, robot tests)

- Done: workspace taught, pushes work on sheet 01 with the magnet block; camera facts settled (wrist link, 9 x 7 cm window,
  looks away from the base); collar mount withdrawn; tall wrist bracket is the default via the staged path.
- Done 27.10 (Robot and hardware session, files only): `measurement_hour_sheet.md` (checklist step 3; shortened on Emranul's
  request: Part 1 = the ten numbers the print needs, on the holder, its screws, the brace holes, the cover opening and the
  camera case, about 20 min, mostly at a desk with the holder unscrewed; Part 2 = optional confirmations of Niryo's STEP
  values) and `mockup_session_checklist.md` (checklist step 5: safety, four tip positions, gap and view tables, photo list,
  acceptance rule), both in `06_deliverables/research_notes/Camera mount options/`. Part 1 covers exactly what Niryo does
  not publish (`niryo_data.md`, section 6); nothing is measured yet.
- Done 27.10 (later): Maker Space answered all seven questions (filed under "Answers" in
  `05_meetings/2026-10-26_maker_space_questions.md`, written there by this session on Emranul's instruction): PLA or PETG,
  0.4 mm nozzle, 0.2 mm layers; several bodies per file allowed; corrective reprint possible, keep the part small; screws are
  ours to buy; print starts 1 to 2 working days after the file; files in mm; calipers lent 8:00 to 13:00, same-day return.
  Checklist step 1 is done; step 2 moved after step 3 and reduced to screws only (no clamp prototype needed, the mock-up uses
  a cardboard block); steps 3, 7, 9 updated with the answers.
- Next: book a morning robot session (calipers only 8:00 to 13:00); then in order step 3 (measurement hour, print the
  sheet), step 4 (field of view and focus sweep, `view_camera.py` only), step 5 (mock-up with a cardboard block of the
  measured camera size), step 6 at the desk; then buy the screws of the measured thread (step 2). Also check the camera
  cable's sleeve near the case (row D10 of the sheet).
- Also 27.10: e-mail to Niryo support drafted (`06_deliverables/research_notes/Camera mount options/email_niryo_support_2026-10-27.md`,
  to support@niryo.com, asks for the holder drawing, screw size, brace hole positions, camera case size). Sent by Emranul
  27.10 09:16; only an automatic reply so far, which says requests must go through https://niryo.com/contact-product-support/ ,
  so the same text goes through that form once; chase two working days after that. An answer would shrink the measurement hour to a check.
- Blocked on: the morning robot session (date open).

## Recording and data (recording tool, demonstrations, datasets)

- Done: design chosen (SOTA report section 2): keyboard stepping, capture after the move returns, stale-frame guard, six
  joints and the action logged per frame, explicit stop step, LeRobot-style layout. Timings and camera rate measured.
- Decided 28.10 (Emranul and Marshal): step directions are named in the PICTURE's frame (forward = top of the picture),
  not the robot base's; the tool supports both keyboard and hand-guided demonstrations.
- Done 28.10 (Marshal, branch `marshal/recording-tool`): `02_code/record_episodes.py` with both modes. Hand-guided =
  guide the puck to the goal with motors off while the tip's path is logged, then the robot replays that path in 1 cm
  steps (snapped back onto it each step), with a key override per step and a discard key; only the replay is training
  data, so its pictures match keyboard and exam pictures. `robot_link.py` (real robot or simulated sheet, puck and
  camera) and `test_record_episodes.py` (both modes, headless) pass on the simulator. Settings in
  `config.RECORDING_PARAMETERS`. Nothing tested on the robot yet.
- Next, first robot test (sheet 01, current camera): confirm one "left" step moves the scene RIGHT in the picture
  (`picture_left_sign` in config, marked UNTESTED); a few keyboard episodes; one hand-guided episode; check the logged
  pictures are fresh. Then the keyboard-versus-hand-guiding trial (three pushes each) and a method decision record.
- Decided 29.10 (Marshal): keep the seven actions (forward, forward-left, forward-right, left, right, back, stop) for the
  first robot test; back-left, back-right and a hop over the puck are deferred. While recording, note each time an
  operator wishes for one of them and in what situation; add one only if it recurs and the camera can see its effect
  (a hop to the puck's far side would leave the puck out of the picture). Reasons in `claude_memory/project-recording-tool.md`.
- Open for the team: Emranul's view on the action list above; the stepping
  rule both follow; varied starts; how raw recordings are shared between the two laptops (episode numbers count per
  laptop, so two laptops can produce the same episode name).
- Blocked on: the camera mount for real data; a robot session for the first test.

## Model and training

- Done: method chosen (SOTA report section 1): single-picture classifier over 1 cm steps plus stop, ResNet-18 or a 5-layer
  CNN, random-crop augmentation, trained on the laptop; ACT only as an optional comparison on a GPU.
- Next: ask the professor to activate GPU-pool SSH for both accounts and test the login (pattern in
  `01_course/moodle_snapshot.md`, VPN needed); write the dataset builder and the training script once the first 20 episodes exist.
- Blocked on: data.

## Evaluation and write-up

- Done: evaluation protocol chosen (SOTA report section 4): pre-registered success rule, 20 trials per sheet, counts with
  intervals, per-path-type reporting; the documentation rule is question 7 for meeting 2 (7 pages per the course catalogue).
- Next: nothing until a trained model exists; keep the contribution log and the figures folder current.
- Blocked on: a model.

## Open questions for the professor (meeting 2)

7 (what to hand in), 10 (offline or local policy), 13 (reversible clamp mount for recording), 14 (rod clamp, low priority),
15 (print our own training tracks), 16 (second camera). Full wording in `05_meetings/requests_to_professor.md`.

## Risks being watched

- The camera's narrow forward view on hairpins and loops: the mount is the answer; the one print must copy a geometry
  photographed working.
- Robot time: everything that needs the robot should be finished by 20 Dec; January is writing and rehearsing.
- Lighting: record and test in the same room with the same lamps; add brightness jitter in training.
