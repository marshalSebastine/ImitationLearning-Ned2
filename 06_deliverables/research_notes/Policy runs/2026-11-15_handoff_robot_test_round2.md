# Handoff to Emranul's Claude session: robot test of Marshal's round 2 policies (15.11.26)

Marshal gives the prompt below to Emranul, who pastes it into his own Claude Code session on the Windows laptop at the
robot. Written by Marshal's session on 15.11.26.

---

You are helping Emranul (team M, AI Project 6, Niryo Ned2 imitation learning) at the real robot, on his Windows laptop
(project folder `D:/AI_Project`, Ethernet to the robot). Emranul is a beginner and is at the robot: keep every reply
very short, one step at a time, plain words. Ask before anything that makes the robot move. Before giving a command,
read the terminal; never stop a running tool without checking it is idle. Follow the project's CLAUDE.md.

**Goal of this session:** the first real-robot runs of three policies that Marshal's Claude session trained (Marshal's
models, not Emranul's own). Do not train anything and do not run the simulated robot; Marshal asked for the real
robot only. Nothing has driven the robot with these policies before.

**The three policies** (GitHub release https://github.com/marshalSebastine/ImitationLearning-Ned2/releases/tag/policies-2026-11-14-round2,
trained on dataset v02 = the six course sheets plus Emranul's round 2 recordings on 28 new tracks; track shapes T01,
T08, T11 and T15 were never trained on). Offline scores only: the validation score is on course sheets 03, 04 and 06,
the unseen score on the never-trained track shapes.

| Rank | Policy | Validation | Unseen tracks | Stop recall | File |
|---|---|---|---|---|---|
| 1 | history classifier (this picture, the two before, the previous move) | 0.608 | 0.508 | 0.77 | `marshal_round2_history_classifier_seed2_best.pt` |
| 2 | classifier with tuned settings (one picture) | 0.607 | 0.483 | 0.87 | `marshal_round2_factorised_classifier_tuned_seed1_best.pt` |
| 3 | classifier (one picture) | 0.607 | 0.473 | 0.61 | `marshal_round2_factorised_classifier_seed1_best.pt` |

Offline, they agree with the operators' move on about 55 % of pictures and with the forearm turn on about 75 %.
How that turns into pushing the puck is what this session finds out.

## Step 1: the right code (important)

The robot runner exists only on **main** (`record_episodes.py --mode policy`, `02_code/run_policy.py`, merged in
pull request #24). Emranul's branch `emranul/training-analysis` does not contain main and has its own, different
`train_policy.py`, so the policy mode is missing there. Do NOT merge the two branches now. Instead:

1. Check `git status` on Emranul's branch. If anything is uncommitted, commit it on his branch first (ask him for the
   message; no Claude attribution in commits).
2. `git switch main` and `git pull`.
3. Check that `02_code/run_policy.py` exists. After the session, `git switch emranul/training-analysis` brings his
   branch back unchanged.

## Step 2: PyTorch (once)

    D:/AI_Project/.venv/Scripts/python.exe -m pip install torch torchvision

The CPU build is enough (one decision takes about 10 to 20 ms on a laptop CPU).

## Step 3: download the three policies (PowerShell)

The `marshal_` folders keep them apart from Emranul's own runs in `04_runs`.

    New-Item -ItemType Directory -Force D:/AI_Project/04_runs/marshal_2026-11-14_round2_history_classifier_seed2/checkpoints
    Invoke-WebRequest https://github.com/marshalSebastine/ImitationLearning-Ned2/releases/download/policies-2026-11-14-round2/marshal_round2_history_classifier_seed2_best.pt -OutFile D:/AI_Project/04_runs/marshal_2026-11-14_round2_history_classifier_seed2/checkpoints/best.pt
    New-Item -ItemType Directory -Force D:/AI_Project/04_runs/marshal_2026-11-14_round2_factorised_classifier_tuned_seed1/checkpoints
    Invoke-WebRequest https://github.com/marshalSebastine/ImitationLearning-Ned2/releases/download/policies-2026-11-14-round2/marshal_round2_factorised_classifier_tuned_seed1_best.pt -OutFile D:/AI_Project/04_runs/marshal_2026-11-14_round2_factorised_classifier_tuned_seed1/checkpoints/best.pt
    New-Item -ItemType Directory -Force D:/AI_Project/04_runs/marshal_2026-11-14_round2_factorised_classifier_seed1/checkpoints
    Invoke-WebRequest https://github.com/marshalSebastine/ImitationLearning-Ned2/releases/download/policies-2026-11-14-round2/marshal_round2_factorised_classifier_seed1_best.pt -OutFile D:/AI_Project/04_runs/marshal_2026-11-14_round2_factorised_classifier_seed1/checkpoints/best.pt

Check each with `Get-FileHash <file> -Algorithm SHA256` against the release notes:
- rank 1: AADA4839C90E09BAE13B2C937C0F8378B22BDC98D40F59DD6442DB2278AA7489
- rank 2: 0C10044DB5966B36C5C4CBE79BC02DFBFF846272A18AC52810E50CC6A1D8DCB3
- rank 3: 8B63FA3D5985E0B10D70A73072A9267845547B09E7D76F78D604171E97504962

Rank 2 was trained as "factorised_classifier_tuned" and is stored as "factorised_classifier"; main's code loads all
three (checked by Marshal's session with main's code on 15.11.26).

## Step 4: robot ready (Emranul does these; confirm each)

- Ethernet connected; `check_hardware.py` says "Calibration needed: False" (automatic calibration only, never manual).
- Only one program connected to the robot: NiryoStudio, the recording page's tools and any viewer closed.
- Same table, camera taping and light as the recordings; do not touch the camera or its focus ring.
- Robot's Stop button in reach. Motors off (FreeMotion) before anyone touches the arm; never push on a held arm. The
  robot stays away from the table edge.
- Sheet taped as in its set-up picture for a STD episode, puck centred on the start, hands out of the picture.

## Step 5: one run

The start position is the one used when recording that sheet: the first line of a log in
`D:/AI_Project/03_data/recording_page_logs/` for that sheet holds the full command with `--start-x --start-y` (and
`--start-push-deg --start-turn-deg` if the start was taught turned). Copy those numbers.

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/record_episodes.py --operator emranul --mode policy --checkpoint D:/AI_Project/04_runs/marshal_2026-11-14_round2_history_classifier_seed2/checkpoints/best.pt --path-id path_03_two_curve --start-x 0.XXX --start-y 0.XXX --orientation "Marshal rank 1, end A, STD" --lighting "ceiling lights"

It prints `policy history_classifier ... on cpu`. In the picture window: Enter at "Put the puck at the start". Then
each step shows the policy's proposal under the picture:
- **Enter** = do that step. Do the first 10 steps one by one and look before each.
- **g** = let it run on by itself; **any key** pauses. It pauses by itself if the robot refuses a step and before
  the policy's own stop (Enter accepts the stop).
- **Space** = stop by hand when the puck is more than about 2 cm off the line and not coming back, or the same
  wrong push repeats for about 5 steps.
- **x** = discard only for safety (puck tipped, robot stopped, hand in the picture). **Esc** = quit, the robot goes home.
- Afterwards "Puck in the goal?": `y` only with the puck's centre within about 1 cm of the end cap.

Runs are saved in `03_data/raw/policy_runs/`, never in the main data. Each run ends by itself at the sheet's step budget.

## Step 6: order (stop early if all three fail at the same place every time)

| # | Sheet (`--path-id`) | Policies | Runs each |
|---|---|---|---|
| 1 | 01 straight line (`path_01_straight_line`), end A | rank 1 | 1 (sanity check) |
| 2 | 03 two curves (`path_03_two_curve`), end A | ranks 1, 2, 3 | 2 |
| 3 | 04 meander (`path_04_meander`), end A | ranks 1, 2, 3 | 2 |
| 4 | if time: an unseen track, e.g. T01 solid (`path_11_T01_solid`) or T08 solid (`path_18_T08_solid`) | the best so far | 2 |

Row 4 is the closest test of the exam's unseen tracks: those shapes were never trained on.

## Step 7: write down every run

Into `06_deliverables/research_notes/Policy runs/2026-11-15_robot_test_round2_results.md`:

| Run | Sheet, end | Policy (rank) | Puck in goal? | How far along the line (cm) | Steps | What went wrong first | Notes |
|---|---|---|---|---|---|---|---|

Known limits to note if seen: the rod's lean label is not acted on (the rod keeps the least lean); a forearm turn is
6 degrees per step; one step is 1 cm.

## Step 8: afterwards

- Commit the results file on a new branch from main, `emranul/robot-test-round2`, push it, and tell Marshal.
- Copy `D:/AI_Project/03_data/raw/policy_runs/` to Marshal (not in git): the pictures of where it failed show what to
  fix next.
- `git switch emranul/training-analysis` to go back to Emranul's own work.
