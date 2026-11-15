# Robot test of Marshal's round 2 policies: results (15.11.26, night, Emranul at the robot)

Asked for by Marshal's handoff `2026-11-15_handoff_robot_test_round2.md`. Code: main at 86091cf (after pull request 26,
so the recording tool also carries Emranul's changes of 14.11.26), run from the working copy
`D:\AI_Project\.claude\worktrees\training-analysis`. The three policy files were downloaded from the release
`policies-2026-11-14-round2`; their sha256 match the handoff. Runs are in
`D:\AI_Project\03_data\raw\policy_runs\2026-11-15_session_01_emranul\`. One step is 1 cm, a forearm turn 6 degrees.

What the operator saw is marked (Emranul); the rest is read from each run's steps.csv.

| Run | Sheet, end | Policy (rank) | Puck in goal? | How far along the line (cm) | Steps | What went wrong first | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 01 straight line, A, standard start | history classifier (1) | no ("went wrong", Emranul) | the rod tip got 11 cm forward at the most (x 0.140 to 0.254 m); the line needs about 27 cm | 48, the step budget | step 8: after seven moves forward it went back-left and then back four times | moves in all: 20 forward, 10 back, 7 forward-left, 6 right, 4 back-left, 1 left; it went to and fro until the budget ran out; forearm between 0 and 18 degrees |
| 2 | 01 straight line, A, standard start | classifier with tuned settings (2) | no (Emranul: "the end effector was blocking the red puck and it couldn't finish it") | 20 steps straight forward, the rod tip from x 14.0 to 33.6 cm, about 8 cm short of where the recordings stop (x about 41 cm) | 33, stopped by hand (Space) | step 20: back, then forward, back, forward, 13 times between x 32.6 and 33.6 cm | sideways it stayed within 1 cm (y -1.0 to -1.8 cm); forearm not turned; the camera check warned at the start (5.7 px from the reference of run 1, after the robot's restart and automatic calibration), recorded anyway |
| 3 | 01 straight line, A, standard start | classifier (3) | no | 20 steps forward, the rod tip from x 14.0 to 32.7 cm (one step to the left at step 4, forearm turned up to 12 degrees and back), about 8 cm short of where the recordings stop | 37, ended with Esc | step 20: back and forward in turn between x 31.9 and 33.9 cm, 17 steps | the same place as run 2 (about 33 cm from the base); sideways within 1 cm of the line; camera check 5.3 px, recorded anyway |

## What the pictures of run 2 show (`2026-11-15_rank_2_sheet_01_frames.jpg`)

From about step 15 the end effector (the black block above the rod) comes up in the picture and covers the puck; from
step 19 the puck is almost hidden behind it, and from step 20 the policy goes back and forward in turn. In a recording
at the same stage (plan 01-02, step 20, last tile) the puck lies in clear view above the end effector. Read from the
pictures and Emranul's remark, not measured: the rod stands more upright in the policy mode than in the recordings
(the handoff's known limit, "the rod keeps the least lean"), so the farther the arm reaches, the more the end effector
hides the puck, and the policy then sees pictures it was never trained on. Run 1 was not checked for the same thing.

## Between the runs

After run 1 the robot's command service no longer answered (a new program connected and then hung at its first
question, as did `check_hardware.py`). How run 1 was ended is not known here. The robot was switched off and on
(Emranul) and calibrated automatically (`calibrate_robot.py`, "calibration done"); after that it answered again.
Run 2 was left with Esc after the picture window had been closed with its X twice (the tool opened it again); run 3
was ended with Esc and the robot answered normally afterwards.

Ranks 2 and 3 both drive the first 20 cm of the straight line well and both get stuck at the same reach, about 33 cm
from the base. Sheets 03 and 04 and the unseen tracks of the handoff were not driven: with the puck hidden at that
reach they would not show more.

Known limits, from the handoff: the rod's lean label is not acted on; a forearm turn is 6 degrees per step; one step
is 1 cm.
