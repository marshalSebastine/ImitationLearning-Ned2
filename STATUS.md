# STATUS: where the project stands

The one live state file. Every Claude session reads it at the start and updates its own workstream section at the end
("update STATUS.md and memory with what changed"). Keep it under 150 lines; details go into the notes it points to.
Full background for a newcomer: `HANDOFF_2026-10-26.md`. How the sessions are organised: `HOW_WE_WORK.md`.

Last updated: 30.10.26 by the Robot and hardware session (Niryo CAD checked against the print file) and by Marshal's
session (recording tool merged into main, not yet tested on the robot; action list kept at seven).

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
- Done 28.10 (Emranul at the robot, guided from this session): checklist step 3, Part 1 complete, values in
  `measurement_hour_sheet.md`, photos `03_data/raw/camera_check/2026-10-28_measure_photo01..11`. Key numbers: holder screws
  M2.5 x 6, 2 mm hex key; holder holes 26 mm apart (the 40 to 45 guess was wrong); brace is a 34 x 22 mm plate with four
  corner holes (top two: screws, bottom two: locating pins of the holder); camera 37 wide x 53 tall x 31 deep, 58 g with
  cable; cover window 37.3 x 77.5 mm; rod 21 mm (STEP confirmed), 27 mm visible below the cover; cable 5.3 mm, sleeve ok.
  Camera screwed back on. Calipers to be returned to the Maker Space the same day. Niryo: e-mail 27.10 got only an
  automatic reply; the same text was submitted through https://niryo.com/contact-product-support/ on 28.10; no real answer yet.
- Done 29.10 (Emranul's decision, to use a Friday print slot): checklist step 7 done before steps 4 to 6. The print file
  `bracket_design/TeamM_Project6_ned2_camera_bracket_F.3mf` (+ STEP, 1:1 mm, 41 x 34 x 86 mm, 30 cm3; foot open in the middle for the raised boss found on the plate 29.10) was built with CadQuery
  from the measured numbers (`bracket_design/bracket_f_design.py`, parametric) and the camera pose fitted on 26.10 (lens 25 mm
  ahead of the rod axis, 6 above joint 5, 22 degrees forward). Target: option F (lens 51 ahead, 60 up, 10 degrees back);
  the file gives 58 ahead, 60 up, -10. Paper pattern B fits the holder (hole rows 16 mm apart confirmed). Assumed, not measured: boss under 19 mm wide, pin 2 mm, ear 3 mm, cover rim
  under 12 mm (`README_bracket_F.md` lists all). Paper test PDF and the Maker Space e-mail text are in `bracket_design/`.
  Cardboard block 58 x 42 x 36 made. Calipers returned.
- Done 29.10 evening: paper test done (B fits the holder; A showed the raised boss, foot redesigned), holder back on the robot, file sent to the Maker Space (3MF + STEP, Friday print asked for). Next:
  two extra M2.5 screws 6 to 10 mm from the Maker Space or the lab. Then, before the bracket goes on the robot: hold the
  printed part with the holder against the wrist (step 11 pre-check), step 4 (focus sweep) and the cardboard check of the
  forearm gap at tip x 0.14 to 0.18 (the reviewers predict a touch at 0.14; keep the tip at x >= 0.16 if so).
- Done 30.10: Niryo support answered with CAD files of the camera holder (filed in `niryo_files/ned2_camera_assembly_from_niryo_support_2026-10-30/`,
  compared in `bracket_design/niryo_support_cad_check_2026-10-30.md`). The CAD agrees with the print file on width 34, rows 16 apart, pin 2 mm and the
  19 mm gap for the boss; it disagrees on two measured numbers: hole spacing 27 (we measured 26) and screw M2 x 8 (we measured M2.5 x 6).
  Two two-minute checks with the printed part and the holder on Monday decide (checks A and B in the note); a corrected file is one parameter away.
  The lens sits about 29 mm in front of the plate face per CAD (we assumed 16): it ends about 7 mm lower than aimed, no change to the part.
  Bracket hand-over by the Maker Space: Monday 2.11.
- Blocked on: the print (Monday 2.11); the next robot day (date open).

## Recording and data (recording tool, demonstrations, datasets)

- Done: design chosen (SOTA report section 2): keyboard stepping, capture after the move returns, stale-frame guard, six
  joints and the action logged per frame, explicit stop step, LeRobot-style layout. Timings and camera rate measured.
- Decided 28.10 (Emranul and Marshal): step directions are named in the PICTURE's frame (forward = top of the picture);
  the tool supports both keyboard and hand-guided demonstrations.
- Done 28.10 (Marshal): `02_code/record_episodes.py` with both modes (hand-guided = guide with motors off, then the robot
  replays the tip path in 1 cm steps with key override and discard); `robot_link.py`, `test_record_episodes.py` pass on
  the simulator. Settings in `config.RECORDING_PARAMETERS`. Nothing tested on the robot yet.
- Done 30.10 (Emranul's session, branch `emranul/weekend-recording-plan`): the data-collection analysis
  `06_deliverables/research_notes/Recording tool/2026-10-30_data_collection_analysis.md`: 13 collection approaches
  criticised, 100 sources checked (80 sound, 20 with caveats; 190 more cited only as unverified), state-of-the-art table,
  episode budget per sheet and method (recommended about 410 episodes and 30 of 36 robot hours until 20 Dec, with the
  traced sheet lengths 24/27/55/77/57/57 cm), an extended first-robot-test checklist with pass thresholds, 14 questions
  for the professor, and corrections to the SOTA report. Recommendations: keyboard stepping primary; guide-then-replay
  only as a measured pilot with "Enter accepts, key overrides" instead of the 400 ms timeout; keep the seven classes and
  add tool keys (mistake, lost) instead; train a first model after 30 to 40 episodes and start human-gated correction
  rounds early; 30 percent offset starts and 25 percent near-end starts; real data only with bracket F.
- Decided 31.10 (Emranul and Marshal, `06_deliverables/research_notes/Recording tool/2026-10-30_data_collection_decisions.md`):
  decision 1: keyboard stepping is the main method, hand-guided replay only as a measured pilot with Enter-to-confirm (kept if
  overrides under 15 percent, discards under 20 percent, time under 1.5 x keyboard; dropped if the Ctrl-C safety test fails);
  decision 2: the seven classes stay, tool keys `u` (mistake) and `l` (lost) are added instead. Still open: the written stepping
  rule; how raw recordings are shared between the two laptops.
- Done 31.10 (Emranul's session, branch `emranul/recording-tool-fixes`, simulator tests pass): the tool fixes from the
  decisions: keys pressed during a move are thrown away; `u` flags the previous step as a mistake, `l` flags "line lost"
  (columns is_mistake, is_lost); a replay step runs only on Enter (no 400 ms timeout); motors stay off after any
  interruption of guiding until the person confirms the arm is released; the workspace guard checks the predicted puck
  centre against the box and the sheet plus 2 cm; step budget per sheet (`config.step_budget_for`, meander 116); three
  timestamps per step; camera mount, mapping version, orientation, start index, offset, lighting and laptop in
  `episode.json` (CLI flags); session folders carry the operator's name.
- Next: the first robot test with Marshal's checklist plus the analysis' section 6 (the 10-versus-10 pilot decides the
  hand-guided mode); set `camera_mount` and `mapping_version` in config when bracket F is fitted; the dataset builder.
- Blocked on: the camera mount (bracket F, 2 Nov) for real data; a robot session for the first test.

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
