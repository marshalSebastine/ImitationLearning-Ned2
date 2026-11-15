# STATUS: where the project stands

The one live state file. Every Claude session reads it at the start and updates its own workstream section at the end
("update STATUS.md and memory with what changed"). Keep it under 150 lines; details go into the notes it points to.
Full background for a newcomer: `HANDOFF_2026-10-26.md`. How the sessions are organised: `HOW_WE_WORK.md`.

Last updated: 6.11.26 by Marshal's session (recording page built; gamepad and hand-guided episodes chosen; hand-guided mode and model output still open), 9.11.26 by the Model and training session (its section only), and 10 and 11.11.26 by the Model and training session (its section only: GPU pool; first robot test, its check and the plan after it). 14.11.26 by the Model and training session (its section only: round 2 recorded, the search for the final model started). Also: 15.11.26 by Marshal's session (round 2 data in v02; batch on two GPUs; tuning running on ki-016).

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
  An independent review found only low-severity issues; the confirmed ones were fixed and tested on 6.11 morning.
- Written 6.11 for Emranul (same folder, `2026-11-06_*.md`): exam-start opinion (one taped start mark, a taught start as
  fallback; six questions for meeting 2, still to be copied into `05_meetings/requests_to_professor.md`), first-data-day
  checklist, a 200-episode plan per sheet (20, 35, 45, 45, 30, 25; offset, near-end and crossing starts).
- Found 6.11: the start pose leaves no gap between block and puck (1.85 cm); it worked because the start was taught where the
  ink begins, so keep teaching there. Exact-replay data (rod leaning 16 to 22 degrees, unlabelled forearm twists, 13 to
  41 % of rows under 3 mm) do not fit the planned 1 cm class model.
- Done 6.11: first-floor table taught (config), tool tests 046 to 048 on sheet 04 (047 and 048 have no ending).
- Done 6.11 (Marshal; extended by Claude for Emranul the same evening): `02_code/recording_page.py`, the form that runs each episode, now used on Windows with the real robot: set-up picture per plan number, teach-start button, blur popup, folder names `..._episode_NNN_plan_NN-NN_try_N_score_NN`, lean and hand labels beside the move name. Sheet 01: 01-01 to 01-10 recorded (6.11); lens cleaned (sharpness 37 -> 184).
- Decided 6.11 (reported by Marshal): record with both the gamepad and hand guiding, not the keyboard.
- Decided 6.11 (Emranul, after Marshal's tests on sheet 04): hand guiding uses `hand_guided_exact`; every row keeps the
  move name, the exact step, the pose and the joints; train first on the move names, exact numbers later.
- Main data from 6.11 in `03_data/raw/main_data/` (from 001 per sheet); the sessions directly in `03_data/raw` are tool tests.
- Blocked on: one model output for both methods, and which gamepad controls are used (the `action` column is only the
  nearest fixed move); asking the professor on 9.11 whether a gamepad is allowed (exam start opinion, question 6).
  Open: raw-data backup and sharing (`03_data` is not in git).

## Model and training

### Emranul's session (picture-to-step models, robot driver `model_mode.py`)

- Done 9.11.26 (branch `emranul/training-analysis`): the training analysis, with a plain summary at the top of
  `06_deliverables/research_notes/Model and training/2026-11-09_training_set_up_analysis.md`; one table of all main-data
  pictures (211 episodes, 18,522 pictures), a twice-checked data audit (drop 469 null rows, relabel 50 as stop, 30 start
  ends corrected, keep score 1), the literature checked against its sources, and runs scored by holding out whole sheets.
  Installed 9.11 into `.venv`: `02_code/requirements_training_emranul.txt` (torch 2.15, processor build); Marshal needs it too.
- Done 10.11.26 (same session; nothing committed, as Emranul said): training moved to the OTH GPU pool computer ki-014
  (OTH VPN; `ssh e.hoque@oth-aw.de@ki-014.oth-aw.de` with Emranul's own OTH password; Adel allowed our code, data and
  Python packages in the home folder): about 10 times faster (one held-out sheet in about 2 minutes, laptop about 19).
  `train_policy_emranul.py` got a `device` setting (not committed). How to: `02_code/HOW_TO_RUN.md`; report section 11.4;
  results in `04_runs/gpu_ki014/`. Laptop queue stopped at about 11:05 on Emranul's request.
- Found 9 to 10.11 (to be confirmed by Emranul and Marshal): first set-up `TRAINING_RUNS["resnet18_final_candidate"]`
  (ImageNet ResNet-18 fine-tuned on one 160 x 120 picture; exact step, forearm twist, its own stop; 8 epochs): on
  held-out sheets 77 to 79 % of its steps point within 45 degrees of where the tip went next (five seeds on ki-014);
  the backup "names plus exact" is 6 to 8 points lower, so the report's rule (2.5) says drop it; the stop is the weak
  point (right place in 92 to 105 of 209 episodes). Augmentation stays (report 6.2). Compare runs within one computer.
- Done 10 to 11.11.26 (not committed): more recipes on ki-014 (chunk outputs, stop weighted 25, ResNet-34, a team of
  five; `04_runs/gpu_ki014/`, table in `claude_memory/project-gpu-pool-access.md`), models trained on all six sheets
  (`.../2026-11-10_resnet18_model_*_all_sheets*/checkpoints/all_sheets.pt`), and the model mode `02_code/model_mode.py`
  (the driver: one picture, one step; `test_model_mode.py` needs no robot).
- Done 11.11.26: first robot test, ten runs (`03_data/model_runs/2026-11-11_17*`): four reached the end and stopped
  (sheet 01 twice, 02, 05; two of them on the end by the 12 mm rule), six lost the line (02 lying square, 03 three
  times, 04, 06). Checked the same evening by eight analyses, each recomputed by an independent checker
  (`06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/check_results_complete.md`).
  Confirmed: the driver moved a fixed 1 cm where the model asked a median 0.61 cm. Open: sheet 04 (the sheet lay 15 to
  25 degrees off) and the dashed loop. Never driven: a line the model has not seen. Short form with all findings:
  `claude_memory/project-first-robot-test.md`.
- Plan from the check, version 2 (reviewed by four critics, each objection tested by a second agent; all in
  `.../evidence/robot_test_2026-11-11/plan_review/`): `.../Model and training/2026-11-11_plan_after_first_robot_test.md`.
  Three short test sittings (A: small steps, the measure, the step at sheet 03's bend, sheet 04 again; B: the loop,
  lines never seen, the driver decision; C: the stop, dashes, orientation); version 1 frozen early as the fallback; a
  pilot and a recording round on 18 to 24 new shapes in one written style; models judged by a panel fixed in advance;
  a final test in exam form. About 35 to 45 robot hours until 20.12.
- Reported by Emranul on 11.11 (the first four are the professor's words; `claude_memory/project-exam-rules-reported.md`):
  own printed tracks may be trained on; an unseen track can be anything; finishing first, then time, ranked; what the
  model lacks may be trained; the robot must stop by itself; the team lays every sheet; one try per track; unseen
  tracks only on the day. Not yet in `05_meetings/requests_to_professor.md` (for the orchestrator).
- 12.11.26: committed on Emranul's laptop with his yes (three commits on `emranul/training-analysis`, not pushed).
  Two proofreaders found about fifty corrections to the plan (`.../plan_review/proofread_complete.md`), not yet worked
  in. Emranul's dates: Thursday 12.11 at the robot (a first test visit with the driver as it is plus small switches,
  checklist `.../Model and training/2026-11-12_test_visit_checklist.md`), Saturday 14.11 recording with Marshal (hand
  trial of the demonstration rules, a pilot, a first batch on new shapes).
- Done 12 to 13.11.26: the test visit (36 runs) and its scoring (`claude_memory/project-test-visit-2026-11-12.md`).
- Done 14.11.26 (Emranul at the robot alone; branch pushed once in the afternoon, later commits local): round 2
  recorded to the end, 55 scored episodes on the 16 new tracks T01 to T16 (4,748 pictures), with a lay-out line in the
  recording tool, quicker lifts and a looser lean limit at the taught start. A two-picture model (steers better, stops
  worse, not kept; `.../2026-11-14_two_picture_model.md`). A mid-way check (`.../2026-11-14_midway_check.md`) and the
  stop check (`.../2026-11-14_stop_check.md`): the model stops near the paper's edge, not at the line's end (every
  round 1 line ends 2.4 to 4.3 cm from the edge).
- Running since 14.11.26, 22:21, on ki-014: the search for the final model on round 1 plus all of round 2 (index
  `03_data/processed/v03_training_index`, 264 episodes, 22,646 pictures). Yardstick and rule fixed before the results:
  each of the 16 new tracks held out once (55 test episodes). First numbers: steering 0.83 path within 45 (round 1
  only: 0.81); the stop right on about 1 episode in 4 and on none of the mid-paper ends. Under test: three changes
  against the edge cue (table repainted as paper, made-up line ends, heavier mid-paper stops), two pictures, ResNet-34,
  16 rounds, score 1 without weight, fewer tracks (would more help?), 320-px pictures. Model 9 (reference recipe on all
  episodes) is on the laptop, not yet driven. All in `.../Model and training/2026-11-14_final_model_search.md`.
- Next: read the search's results by its rule, choose the recipe, train it on every episode and test it on the robot
  with the comparison run list (`.../2026-11-14_comparison_runs.csv`); Marshal's model into the same list.
- Open for Emranul: leave episode 14-02 out (skipped hairpin)? May made-up line ends be used (not demonstrated stops)?
  Make today's camera view the camera check's reference? Pilot feedback on the recording rules.
- Night of 14 to 15.11.26, robot look with model 10 (reference recipe on all data): sheet 01 stopped 1 cm left of the
  end; sheet 02 fails in the standard lay-out where the rod tip is 43 cm and more from the base and goes better lying
  landscape; sheet 04 better in the standard lay-out; on sheet 05 (laid as on 11.11) the round 1 model E works and
  model 10 does not, so E stays the model for the six course sheets. Both models follow the puck and use the line
  about half as much (`.../evidence/night_look_2026-11-14/puck_test.py`); the claim written here that only 280 of
  23,270 pictures show a puck being brought back was wrong (it counted only the 35 starts beside the line): 304
  corrections by hand (h key) in 169 episodes give 2,094 training pictures in the steps after a correction, and 2,369
  in the 8 steps before one, where the replay goes on while the puck drifts (Emranul's objection, checked 15.11.26,
  section 11 of the search notes). A third queue (zoomed-in view, turned and mirrored pictures, larger networks, a
  second yardstick on the six course sheets, robot candidates models 11 and 12) is ready for ki-014.
- 15.11.26, 02:00 to 02:30 (Emranul at the robot, Marshal's handoff): first robot runs of Marshal's three round 2
  policies, sheet 01 only (`06_deliverables/research_notes/Policy runs/2026-11-15_robot_test_round2_results.md`): rank
  1 went to and fro from step 8; ranks 2 and 3 drove 20 cm straight and then got stuck at about 33 cm reach, where the
  end effector hides the puck (the runner keeps the rod upright; the recordings lean it). The robot had to be restarted
  once (its command service hung after run 1) and was calibrated automatically.
- 15.11.26, 03:05 (Emranul: "yes build it", "put more weight on the frames after h is pressed"): hand-correction
  runs built, tested and queued on ki-014 at 03:09 behind the third queue (20 jobs, about 2.5 hours once they start): the steps before an h press
  left out of training, the steps after it counted three times, robot candidates models 13 and 14; results come back through
  the open login window, or later with `.../evidence/final_2026-11-14/ki014_scripts/final5_fetch.cmd` (one password).
  New measure on recorded pictures (`score_hand_corrections.py`): the reference agrees with the demonstration on 83 %
  of all pictures but on 58 % of the pictures after an h press. Section 12 of the search notes has the rule.
- 15.11.26 evening (Emranul and Marshal at the robot, 38 runs; `.../Model and training/2026-11-15_robot_test_results.md`,
  marks given on the recording page's new test part): model 15 (mirror, after-h weight, 16 rounds) keeps new printed
  lines far better than model E, but every new model fails the revolving move: in loops the forearm must swing back
  from about -70 to +60 degrees, model E does it on sheet 05, the others wind on to the joint limit of -117
  (`2026-11-15_revolving_move.md`). Built and tested on the laptop: the direction memory (models 19 and 20; its layer
  was not trained in the first try, fixed), the recurrent model (12 camera pictures through a GRU, models 21 and 22),
  swing weighting (model 23). Queued on ki-016 (free; ki-014 was filled by another person's training): queues 11 and
  13. To read: one fetch login to ki-016 with a NEW sent list (model 19 was retrained under the same name). No-mirror
  scored worse than mirror on recorded pictures (0.828 against 0.856). Emranul keeps the memory runs although the
  memory reads the arm's joints, not the camera (the course rule question is open).
- State on 15.11.26, 02:30: second and third queue on ki-014 (the third ends about 11:30); results come back with a
  fetch-only login (`.../evidence/final_2026-11-14/ki014_scripts/final_fetch.cmd`). Stop fixes scored: none passes.
  Next: score both queues on both yardsticks, build the look-ahead steering into the driver and rehearse it, drive
  models 11 and 12 and the round 1 model E on the six course sheets in their good lay-outs.
- Merged into `main` on 15.11.26 (asked for by Emranul): his training program is now `02_code/train_policy_emranul.py`
  (packages: `requirements_training_emranul.txt`), renamed so that it lives beside Marshal's `train_policy.py`; every
  script and the robot driver follow. The copy on ki-014 (`~/ai_project/02_code/train_policy.py`) keeps the old name
  until the queues there have finished; the next upload has to send the new name. Open: the safety copy to a second
  disk.
- Cancelled 5.11 (Emranul): synthetic pictures from a Webots twin; code only on branch `emranul/synthetic-pictures`, not
  merged; findings in `claude_memory/project-synthetic-pictures.md`.

### Marshal's session (five policies, `train_policy.py`, policy mode of the recording tool)

- Done: method chosen (SOTA report section 1): single-picture classifier over 1 cm steps plus stop, ResNet-18 or a 5-layer
  CNN, random-crop augmentation, trained on the laptop; ACT only as an optional comparison on a GPU.
- Done 13.11 (Marshal): data collection finished (main data of 6.11 to 9.11 in `03_data/raw/main_data`, also on
  Marshal's Mac). `02_code/build_dataset.py` writes `03_data/processed/v01_main_data/`: 15,618 pictures from 205
  episodes, split by whole episode within each sheet (train 10,964, validation 2,188, test 2,466). Left out: unfinished
  episodes, `small_motion` rows (2,196), rows marked a mistake (21), and the hand pictures and five episodes named in the
  operators' own review notes (list in `config.py` `DATASET_PARAMETERS`; the tool's automatic picture flags were not used,
  Marshal 13.11: mostly false alarms). Score 1 episodes stay in, the score is a column.
- Emranul trained five models of his own (results reported by Marshal 13.11: sheets 01, 02 and 05 completed, 03, 04 and
  06 not). Marshal's session builds its own models without looking at them. Candidate order (Claude's proposal 13.11, not
  decided): (1) one picture, move name plus forearm-turn and lean outputs; (2) one picture, exact step as numbers;
  (3) model 1 with the last 2 to 3 pictures and the previous move; (4) small ACT, next 5 to 10 steps; (5) small diffusion
  policy (GPU). Optuna only to tune one family, scored on held-out episodes of sheets 03, 04 and 06.
- Done 13.11 (Marshal): training code for all five candidates (`train_policy.py`, `tune_policy.py`, `policy_data.py`,
  `policy_models.py`, `run_overnight.sh`; guide `02_code/GPU_TRAINING.md`). Tested on made-up pictures (every policy, a
  held-out sheet, a 2-trial Optuna search) and checked on the real data (every policy trains one step and predicts);
  NOT yet trained for real: Marshal's Mac (8 GB) runs out of memory, so the first real training is the GPU-pool night.
- Done 13 to 14.11 (Marshal, GPU pool ki-014, batch `2026-11-13_overnight`, 75 minutes): all five policies with three
  seeds, 30 Optuna trials and the tuned classifier; results in `04_runs/2026-11-13_overnight_summary.md` and
  `..._optuna.md` (weights on Marshal's Mac and ki-014 only). Validation hard-sheet score (sheets 03, 04, 06), mean of 3
  seeds: history classifier 0.618, factorised classifier 0.609, exact-step regressor 0.598, chunk transformer 0.579,
  diffusion policy 0.476. Optuna's best trial 0.632 (one seed), but the full retrain of its settings gave 0.619: no gain
  beyond seed noise. Move balanced accuracy is only about 0.5 and every classifier is best after 5 to 13 epochs, so
  the models overfit fast. Offline numbers only; nothing has run on the robot.
- Done 14.11 (Marshal): the robot runner, as the recording tool's `--mode policy --checkpoint <best.pt>`
  (`02_code/run_policy.py`): the policy's move and forearm label become a gamepad command (1 cm step, 6 degree turn),
  so the gamepad mode's joint solving and table-edge check apply; every step waits for Enter unless `g`; runs saved in
  `03_data/raw/policy_runs/`. Tested with the simulated robot and on raw validation JPEGs (same agreement as in
  training, 9 to 19 ms per decision on a CPU). Not yet run on the robot.
- Done 14 to 15.11 (Marshal; round 2 recorded by Emranul 14.11, 55 episodes on 32 new tracks): dataset `v02_main_data`
  (19,599 pictures, 257 episodes; three round-2 episodes left out from the review notes; track shapes T01, T08, T11, T15
  held out as unseen tracks). Batch `2026-11-14_round2` on ki-015 and ki-016 (`04_runs/2026-11-14_round2_summary.md`).
  Unseen-track score, first batch's models -> round 2: history classifier 0.473 -> 0.505 (1 seed so far), classifier
  0.468 -> 0.476, tuned classifier 0.451 -> 0.483, regressor 0.417 -> 0.466, chunk transformer 0.392 -> 0.448, diffusion
  0.332 -> 0.361: the new tracks help every model on unseen shapes. Course-sheet scores fell by 0.01 to 0.02, most likely
  because every one of the 30 sheets weighed the same; `sheet_balance_power` is now a setting and is being tuned.
- Running 15.11 on ki-016 (ki-015's GPU is taken by another student's job): the two missing history runs, then Optuna
  over the history classifier, then over the classifier (`04_runs/2026-11-15_tuning_*`), each retraining its best
  settings with three seeds.
- Next (Emranul, Windows laptop with Ethernet; decided by Marshal 14.11): first robot runs with policies A (history
  classifier) and B (tuned factorised classifier), checklist `06_deliverables/research_notes/Policy runs/first_policy_run_checklist.md`.
  Needs: PyTorch in `.venv` and the two weight files from Marshal (not in git).
- Blocked on: nothing.

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
