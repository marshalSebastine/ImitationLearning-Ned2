# STATUS: where the project stands

The one live state file. Every Claude session reads it at the start and updates its own workstream section at the end
("update STATUS.md and memory with what changed"). Keep it under 150 lines; details go into the notes it points to.
Full background for a newcomer: `HANDOFF_2026-10-26.md`. How the sessions are organised: `HOW_WE_WORK.md`.

Last updated: 29.10.26 by the Robot and hardware session (bracket print file made, going to the Maker Space).

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
- Next: Emranul prints the paper test and checks patterns A and B tonight, sends the 3MF to the Maker Space (print 30.10);
  two extra M2.5 screws 6 to 10 mm from the Maker Space or the lab. Then, before the bracket goes on the robot: hold the
  printed part with the holder against the wrist (step 11 pre-check), step 4 (focus sweep) and the cardboard check of the
  forearm gap at tip x 0.14 to 0.18 (the reviewers predict a touch at 0.14; keep the tip at x >= 0.16 if so).
- Blocked on: the print; the next robot day (date open).

## Recording and data (recording tool, demonstrations, datasets)

- Done: design chosen (SOTA report section 2): keyboard stepping, capture after the move returns, stale-frame guard, six
  joints and the action logged per frame, explicit stop step, LeRobot-style layout. Timings and camera rate measured.
- Next: build the recording tool in `02_code/` and test it with the current camera on sheet 01; real recording waits for the
  final camera mount. Then the keyboard-versus-hand-guiding trial (three pushes each) and a method decision record.
- Blocked on: the camera mount for real data; nothing for building the tool.

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
