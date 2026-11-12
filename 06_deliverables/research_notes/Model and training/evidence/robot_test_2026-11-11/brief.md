# Brief: the first robot test of 11.11.26 and what is to be found out

Written 11.11.26 by the Model and training session, for the agents that analyse the test and for anyone who reads the
evidence later. Facts marked [scout] were computed quickly by the session and must be re-checked, not trusted.

## The project in ten lines

- Team M (Emranul Hoque, a self-described beginner, and Marshal Sebastine), course "AI Project", Project 6: a Niryo
  Ned2 arm pushes a red puck (4 cm) along a 1-cm black line printed on A4 and stops at the line's end. The exam uses an
  unseen line. Rule: imitation learning only; the robot's one camera is the model's only input.
- The camera is taped on the forearm cover and turns with joint 4 (the forearm twist). Joints 5 and 6 do not move it.
- 211 demonstrations on six course sheets (01 straight, 02 arc, 03 two curves, 04 meander, 05 alpha loop, 06 alpha loop
  dashed), recorded 6 to 9.11.26 in the mode `hand_guided_exact`: a person guides the arm by hand with the motors off,
  then the robot replays the guided path waypoint by waypoint and takes one picture per waypoint. A waypoint was cut
  after 1 cm of tip travel OR 5 degrees on any joint, so some steps are "twists in place" with almost no tip travel.
- Labels per picture ("exact step"): the magnet tip's move in the PICTURE's frame (forward = towards the picture's top,
  left = towards its left; `tip_forward_m`, `tip_left_m`), the forearm twist `forearm_change_deg` (joint 4), the rod's
  lean change, the hand change, and a stop row at the end of each episode.
- Dates: today 11.11.26; mandatory meetings with the professor 23.11.26 and 7.12.26; the team's freeze 8.1.27;
  submission 10.1.27; presentation 11.1.27.

## Where everything is (the git worktree `D:/AI_Project/.claude/worktrees/training-analysis`, called ROOT)

- Code: `ROOT/02_code/` (`config.py` holds every setting; `train_policy.py` the model `PictureToMoveModel` and
  `merged_run_settings`; `training_targets.py` `load_training_frames` (audit cleaning: drops 469 rows, relabels stops,
  clips steps at 2 cm and twists at 15 degrees), `chunk_targets`, `leave_one_sheet_out_folds`; `training_probes.py`
  `design_outputs`, `predictions_from_outputs`; `model_mode.py` the robot driver; `record_episodes.py` the recording
  tool (read only); `forearm_camera_model.py` a fitted camera model with `project_points`; `puck_detector.py`,
  `puck_tracking.py`, `check_pictures.py` picture tools; `side_view_test.py` `magnet_tip_and_lean`,
  `solve_with_wrist_fixed`, `ned2_fk`).
- Python: `D:/AI_Project/.venv/Scripts/python.exe`. Run scripts with `ROOT/02_code` as the working directory or put it
  on `sys.path`. No GPU here: inference with saved models is fast, training is slow (about 20 minutes per held-out
  sheet), so do not train.
- Data index: `D:/AI_Project/03_data/processed/v01_training_index/frames.csv` (18,522 rows, one per picture; columns
  include `episode_id`, `row_in_episode`, `sheet`, `plan_number`, `start_end`, `start_kind`, `action`, `tip_forward_m`,
  `tip_left_m`, `forearm_change_deg`, `lean_change_deg`, `hand_change_deg`, `joint_1` to `joint_6` (radians),
  `tip_x_m`, `tip_y_m`, `tip_z_m`, `rod_from_vertical_deg`, `picture_heading_deg`, `move_length_m`, `move_angle_deg`,
  `goal_distance_cm`, `is_at_goal`, `steps_to_end`, `score`, `lighting`, `frame_path`, `frame_index`),
  `episodes.csv` (211 rows; `sheet_orientation`, `start_kind`, `offset_lateral_cm`, `offset_angle_deg`,
  `start_push_direction_deg`, `start_turn_deg`), `images_wide_160.npy` (all pictures at 160 x 120, RGB, in
  `frame_index` order). Raw pictures (640 x 480 JPEG): `D:/AI_Project/03_data/raw/main_data/<frame_path>`. NEVER write
  into `03_data/raw` or `01_course`.
- Start kinds in the data: STD (standard start on the taped start mark), NE (taught start, near end), OL1, OL1.5, OR1,
  OR1.5 (puck 1 or 1.5 cm left or right), TL10, TL15, TR10, TR15 (start turned 10 or 15 degrees), MID (start in the
  middle of the line), X1, X2 (starts at the crossing), and on sheet 02 also SQP and SQL (16 episodes; check what they
  are in `config.py` / the set-up pictures).
- Set-up pictures, one per plan (top view: where the sheet lay, the start, the first push):
  `D:/AI_Project/06_deliverables/research_notes/Recording tool/figures/setup_per_episode/plan_XX-YY.png`.
- Trained models, `ROOT/04_runs/gpu_ki014/<run folder>/` with `metrics.json`, `fold_<name>.json`,
  `predictions_<fold>.npz` (`frame_index`, `angle_deg`, `twist_deg`, `lean_deg`, `hand_deg`, `stop_probability`; chunk
  runs also `chunk_cm` (rows, 5, 2) and `angle_vector_deg`), `checkpoints/<fold>.pt`:
  - fold models (each trained without one sheet, fold names `without_path_01` to `without_path_06`):
    `2026-11-09_resnet18_final_candidate` (design "exact", run name `resnet18_final_candidate`),
    `2026-11-10_resnet18_final_candidate_chunk` (design "chunk"), `2026-11-10_resnet18_chunk_stop_weight_25`
    (the recipe of model E, run name `resnet18_chunk_stop_weight_25`), each also with `_seed_2`;
    `2026-11-09_resnet18_backup_names_plus_exact` (design "mixed": 8 move names plus the exact step);
  - all-sheets models used on the robot (`checkpoints/all_sheets.pt`): `2026-11-10_resnet18_model_1_all_sheets`
    (A, run name `resnet18_model_1_all_sheets`, also seeds 2 to 5: the team of five is B),
    `2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets` (E, run name `resnet18_model_4_chunk_stop_25_all_sheets`).
  - Loading a model for inference: `model_mode.ModelPolicy(run_name, checkpoint_path, metrics_path)`, then
    `.decide(jpeg_bytes)` gives `tip_forward_cm`, `tip_left_cm`, `twist_deg`, `lean_deg`, `hand_deg`, `angle_deg`,
    `stop_probability` (it does not return the chunk; for the chunk call `policy.models[0](policy.picture_input(bytes))`
    and read `outputs["chunk"]`, 10 numbers = 5 future tip positions (forward cm, left cm) in the current picture frame).
- The robot driver `model_mode.py` (`model_step`, settings `MODEL_MODE_PARAMETERS` in `config.py`): every fresh picture
  gives one decision; the tip moves a FIXED 1.0 cm in the predicted direction (the predicted LENGTH is ignored; a
  direction shorter than 0.05 cm keeps the last direction); the forearm twist follows the model, at most 10 degrees per
  step, joint 4 within plus or minus 117 degrees; the wrist (joint 5) is HELD at 13.5 degrees and the hand (joint 6) at
  -6.1 degrees (the medians of the demonstrations' first rows) and joints 1 to 3 put the tip on its target
  (`solve_with_wrist_fixed`); the lean and hand outputs are ignored; the run stops when the stop chance is 0.5 or more
  on two pictures in a row, after 250 steps, at the first refused step, or on Esc.
- The earlier report, `ROOT/06_deliverables/research_notes/Model and training/2026-11-09_training_set_up_analysis.md`
  (about 2,300 lines; sections 4.2 to 4.6 on small motions, twist, step length, stop and the executor; 7.4 the robot test
  protocol; 8 improvement and correction rounds). Its section 4.4 chose the fixed 1-cm step and named a backup "only if
  fixed 1-cm steps lose the puck at sharp bends on the robot: the predicted length rescaled ... clamped to 0.5 to 1.5 cm".
- Offline scores (six held-out sheets, measured on ki-014): path within 45 degrees 0.787 (A recipe), 0.807 (team of 5),
  0.808 / 0.800 (E recipe); right stops of 209 episodes 92, 95, 103 / 105.

## The robot test of 11.11.26 (10 runs, about 17:20 to 17:57, logs in `D:/AI_Project/03_data/model_runs/`)

Each run folder has `frames/NNN.jpg` (the picture each decision used), `steps.csv` (per picture: `model_tip_forward_cm`,
`model_tip_left_cm`, `model_twist_deg`, `model_lean_deg`, `model_hand_deg`, `model_angle_deg`,
`model_stop_probability`, the pose `x` to `yaw`, `joint_1` to `joint_6` in radians, the magnet tip `tip_x`, `tip_y` in
metres in the robot's base frame, `action`, `refusal`, `picture_heading_deg`, `twist_used_deg`, `is_turn_clamped`,
`is_direction_kept`, `lean_deg`, `tip_next_x`, `tip_next_y`, `decision_ms`) and `summary.json`.

| # | Folder (prefix `2026-11-11_`) | Model | Sheet and placement | What Emranul saw / what the log shows |
|---|---|---|---|---|
| 1 | `172021_check_no_puck` | A | 01, with the puck despite the label; tip about 1.5 cm left of the line at the start | reached the end and stopped by itself (25 moves); the puck drifted about 1 to 2 cm right of the line in the second half |
| 2 | `172421_E_sheet01_run1` | E | 01, puck centred | 24 moves, on the line, stopped by itself at the end; "a bit of the line's end still visible" |
| 3 | `172743_E_sheet02_run1` | E | 02 lying straight (not as in plan 02-01) | steered right at the start while the arc went up-left; lost the line; Esc after 37 moves |
| 4 | `173036_E_sheet02_run2` | E | 02 as in plan 02-01 | whole arc, 41 moves, stopped by itself; "just before the middle the puck was not in the middle of the line" |
| 5 | `173549_E_sheet03_run1` | E | 03 as in plan 03-01 (a blue hoodie may be in view) | overshot the first sharp bend (tip went about 10 cm away before turning), wandered, stopped somewhere after 59 moves |
| 6 | `174108_E_sheet04_run1` | E | 04 as in plan 04-01 (blue hoodie in view at the top right) | first 15 moves fine, then about 30 moves back and forth at the hairpin, then straight across the meander ("skipping through lines"); stopped after 68 moves |
| 7 | `174448_E_sheet05_run1` | E | 05 as in plan 05-01 | whole alpha loop with the crossing, 88 moves, stopped by itself; "a little issue with keeping the puck in the middle of the line" |
| 8 | `174936_E_sheet06_run1` | E | 06 as in plan 06-01, nothing blue in view | left the dashed loop early (puck outside the loop by step 15), ended at the paper's edge, stopped after 40 moves |
| 9 | `175203_E_sheet03_run2_no_hoodie` | E | 03 again, nothing blue in view | failed the same way as run 5 (89 moves) |
| 10 | `175544_B_team5_sheet03` | B | 03 | "2 to 3 % better, then failed"; same overshoot at the first bend (73 moves) |

What Emranul wants (his words, 11.11.26): the puck must cover the line's end COMPLETELY at the stop; the puck must stay
in the MIDDLE of the line the whole way; "I want the robot robust, I want it to somewhat work well in any orientation";
software limits only 2 to 3 degrees inside the hardware's; "the best of the best result"; no overfitting to the six
sheets; many options to test on the robot. He has the robot at hand regularly; a robot session is about 2 to 3 hours.

[scout] numbers of the session (re-check them):

- Demonstrated step length: median 1.05 cm, mean 0.92; 20.5 % of steps under 0.5 cm, 8 % under 0.2 cm; steps with a
  twist of 6 degrees or more have a median length of 0.65 cm.
- What the model ASKED for against the 1 cm the robot DID (median asked length, share asked under 0.3 cm, asked path,
  path done): run 2: 1.14 cm, 4 %, 25.6 cm asked, 24 done; run 4: 0.87, 2 %, 32.9, 41; run 7: 0.83, 9 %, 68.9, 88;
  run 5: 0.55, 12 %, 31.9, 59; run 9: 0.49, 15 %, 44.1, 89; run 10: 0.55, 16 %, 37.5, 73; run 6: 0.31, 50 %, 26.0, 68;
  run 8: 0.82, 2 %, 32.3, 40. The runs that failed at sharp bends are the ones where the model asked for short steps.
- Twist per cm done on the robot against the demonstrations' median twist per cm on the same sheet: sheet 03: 3.1 to 3.2
  against 4.64; sheet 04: 1.37 against 5.65; sheet 05: 2.74 against 3.44; sheet 06: 2.58 against 4.43; sheet 02: 1.9
  to 2.4 against 2.57. Offline the models' twist was already about half the demonstrated size (right sign in 68 to 84 %).
- In one sheet-03 demonstration (plan 03-04) the first ten steps twist the forearm by -5 to -11 degrees per step while
  the step directions jump (101, 17, -11, -114, 12, -82, ... degrees): a turn in place with jitter.

## Rules for every agent

- NOTHING may connect to the robot: never run `model_mode.py robot`, never construct `RobotLink` or `NiryoRobot`, never
  run the recording tool. The robot may be switched on and cabled to this laptop.
- No git command that changes anything (no add, commit, stash, checkout, reset, push).
- Do not edit anything in `ROOT/02_code`, `03_data`, `01_course` or `04_runs`. Write only into your own folder
  `ROOT/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/<your key>/`: your scripts,
  their printed output saved as text, and figures (PNG). Scripts must run again from that folder.
- Keep computations light: `torch.set_num_threads(2)`, a few minutes at most per script; several agents share six cores.
- Say for every number whether it is measured (and from which file), calculated or a guess. British spelling, plain
  words, no em-dashes. Do not write "the user": name Emranul or Marshal.
