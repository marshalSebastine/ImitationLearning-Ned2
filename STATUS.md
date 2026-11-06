# STATUS: where the project stands

The one live state file. Every Claude session reads it at the start and updates its own workstream section at the end
("update STATUS.md and memory with what changed"). Keep it under 150 lines; details go into the notes it points to.
Full background for a newcomer: `HANDOFF_2026-10-26.md`. How the sessions are organised: `HOW_WE_WORK.md`.

Last updated: 6.11.26 by Emranul's Recording and data session (puck tracking, episode checker, data-day plan, review fixes).

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
- Done 5.11: question 17 (may part of the training data be synthetic?) added to the meeting-2 list and withdrawn the same
  day, when Emranul cancelled the synthetic-data idea (see Model and training).
- Done 6.11, on Emranul's request: every session's work written down, pushed and merged into `main` by pull requests:
  recording tool and camera work 31.10 to 5.11 (11), paper publication assessment (12), this update. Unmerged on purpose:
  `emranul/synthetic-pictures`. Not yet committed: the recording session's unfinished puck-tracking change.
- Next: prepare meeting 2 on 9.11: questions 7, 10, 13, 14 and 15 (check whether 13 and 14 still matter now that the camera
  is taped to the forearm cover; copy them into the meeting note first); fill in the remaining "[Emranul: fill in]"
  lines of `HANDOFF_2026-10-26.md`; agree who does what (meeting-1 note); start the contribution log.
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

- Done to 31.10: tool design (capture after the move, stale-frame guard, joints and action per frame, LeRobot-style
  layout); actions named in the PICTURE's frame (Emranul and Marshal, 28.10); data-collection analysis and decisions
  (`06_deliverables/research_notes/Recording tool/2026-10-30_data_collection_*.md`: about 410 episodes, offset and near-end
  starts, keyboard as the main method, keys `u` mistake and `l` lost). 4.11 (Marshal): eleven classes, outcome, discard reason.
- Robot days 31.10 to 5.11 (Emranul, sheet 04, old desk; merged by pull request 11): the BARE camera taped on the forearm
  cover, turning with joint 4 (`claude_memory/project-forearm-camera-mount.md`); forearm turns `v`/`b`, `hand_guided_exact`
  (`g` runs on, `h` corrects by hand), gamepad mode, camera shift check, orientation window, `02_code/HOW_TO_RUN.md`. All
  episodes up to 045 are tool tests, not training data (Emranul, 5.11).
- Done 6.11 (night; branch `emranul/recording-tool-fixes`, not yet merged): the orientation window tracks the red puck
  (`puck_detector.py`: found in 1390 of the 1391 recorded pictures that show it, no false detection; `forearm_camera_model.py`
  fitted without the robot to episodes 037 to 045, map error about 2 mm median, 5 mm at the 90th percentile;
  `puck_tracking.py`; display only), and `check_episodes.py` checks recorded episodes. Settings in `config.py`, tests in
  `test_puck_tracking.py`, evidence in `06_deliverables/research_notes/Recording tool/2026-11-06_forearm_camera_calibration/`.
  An independent review (four reviewers, each finding re-checked) found only low-severity issues; the confirmed ones were
  fixed and tested on 6.11 morning (map texts, the tool's puck guess shown where turns use it, a guard so a map fault
  never stops an episode, checker wording, skipped tests counted, a new `test_forearm_camera_model.py`).
- Written 6.11 for Emranul (same folder, `2026-11-06_*.md`): exam-start opinion (one taped start mark, a taught start as
  fallback; six questions for meeting 2, still to be copied into `05_meetings/requests_to_professor.md`), first-data-day
  checklist, a 200-episode plan per sheet (20, 35, 45, 45, 30, 25; offset, near-end and crossing starts).
- Found 6.11: the start pose leaves no gap between block and puck (1.85 cm); it worked because the start was taught where the
  ink begins, so keep teaching there. Exact-replay data (rod leaning 16 to 22 degrees, unlabelled forearm twists, 13 to
  41 % of rows under 3 mm) do not fit the planned 1 cm class model.
- Next (6.11, Emranul): move to the first-floor table, re-teach (Claude updates config), camera check, one trial per sheet.
- Blocked on: Emranul and Marshal choosing the recording mode before mass collection: 1 cm classes with `hand_guided`,
  keyboard or gamepad (Claude's lean, as decided 31.10), or `hand_guided_exact` with a model that predicts the motion.
  Open: raw-data backup and sharing (`03_data` is not in git).

## Model and training

- Done: method chosen (SOTA report section 1): single-picture classifier over 1 cm steps plus stop, ResNet-18 or a 5-layer
  CNN, random-crop augmentation, trained on the laptop; ACT only as an optional comparison on a GPU.
- Next: ask the professor to activate GPU-pool SSH for both accounts and test the login (pattern in
  `01_course/moodle_snapshot.md`, VPN needed); write the dataset builder and the training script once the first 20 episodes exist.
- Cancelled 5.11 (Emranul): synthetic pictures from a Webots twin; code only on branch `emranul/synthetic-pictures`, not
  merged; findings in `claude_memory/project-synthetic-pictures.md`. GPU pool: VPN not connected on 3.11, GPUs still unknown.
- Blocked on: data.

## Evaluation and write-up

- Done: evaluation protocol chosen (SOTA report section 4): pre-registered success rule, 20 trials per sheet, counts with
  intervals, per-path-type reporting; the documentation rule is question 7 for meeting 2 (7 pages per the course catalogue).
- Next: nothing until a trained model exists; keep the contribution log and the figures folder current.
- Blocked on: a model.

## Paper publication (side topic, asked for by Emranul on 3.11.26)

- Done 4.11.26 (Emranul's paper-publication session, branch `claude/paper-publication-81bd4c`): assessment in
  `06_deliverables/research_notes/Paper publication/2026-11-04_publication_assessment.md`. Verdict (a recommendation, not
  yet decided): plan the paper for after the course; best candidate is a 4-page workshop paper (ICRA 2027 or RSS 2027
  workshop, deadlines likely March to June 2027) comparing keyboard stepping with hand-guided replay for the one-picture
  classifier (Emranul's own idea), occlusion and step-over as a section. Seven more ideas added on Emranul's request;
  the cheapest is an education paper on the course project for Robotics in Education 2027 (Prague, deadline ~late Jan).
- Next: no extra robot work before 10 Jan; pre-register the keyboard-against-replay comparison and the final evaluation,
  keep every number, take figure photos; ask Prof. Nierhoff section 4's eleven questions; go or no-go in late January.
- Blocked on: Prof. Nierhoff's view (co-authorship, publishing data and code).

## Open questions for the professor (meeting 2)

7 (what to hand in), 10 (offline or local policy), 13 (reversible clamp mount for recording), 14 (rod clamp, low priority),
15 (print our own training tracks). 16 answered in effect on 4.11 (the camera may go anywhere on the robot, still one
camera); 17 (synthetic training pictures) withdrawn on 5.11. Full wording in `05_meetings/requests_to_professor.md`.

## Risks being watched

- The camera's narrow forward view on hairpins and loops: the mount is the answer; the one print must copy a geometry
  photographed working.
- Robot time: everything that needs the robot should be finished by 20 Dec; January is writing and rehearsing.
- Lighting: record and test in the same room with the same lamps; add brightness jitter in training.
