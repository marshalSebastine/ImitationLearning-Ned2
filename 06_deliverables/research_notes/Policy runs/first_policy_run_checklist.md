# First robot runs with a trained policy (checklist for Emranul, written 14.11.26)

What this session answers: does a policy trained on our demonstrations push the puck along the line on the real
robot, and which of two policies does it better? Until now every number is offline (validation pictures); nothing has
driven the robot yet. The tool is the recording tool's new **policy mode**: the policy proposes every step, shown on the
picture, and the step happens only when you press Enter (or after `g`, until you press any key).

**The two policies** (from the GPU batch of 13 to 14.11, `04_runs/2026-11-13_overnight_summary.md`):

| Name here | Run folder | Why |
|---|---|---|
| **A** | `2026-11-13_overnight_history_classifier_seed1` | best validation score (0.618 over 3 seeds); sees the last 3 pictures |
| **B** | `2026-11-13_overnight_factorised_classifier_tuned_seed1` | one picture only; best at stop (94 % of stop pictures) and at back-left / back-right |

Measured: offline agreement with our labels is about 50 % (balanced) on sheets 03, 04 and 06; the runner gives the same
on raw validation pictures, 9 to 19 ms per decision on a CPU. Guessed: how that translates into pushing the puck.
Calculated: a forearm turn label becomes a 6 degree turn (the median of the labelled turns), a move one 1 cm step.

## Stop rules (the same as always)

- Robot's Stop button in reach. Press it for any contact you did not expect, a cable pulling, a strange noise, the
  arm heading for the table edge or the base.
- Never touch the arm while the motors are on; motors off (FreeMotion) before anyone touches it. Never push on a held arm.
- Do not touch the camera or its focus ring. Calibration: only the automatic one after a restart.
- Only one program connected to the robot: close NiryoStudio, the recording page's tools and any viewer first.
- The robot stays away from the table edge. The tool refuses steps outside the workspace box, but watch anyway.

## Before the session (on the Windows laptop, no robot needed)

- [ ] `git pull` on main.
- [ ] Install PyTorch into the robot laptop's environment (once, about 10 minutes):
      `D:/AI_Project/.venv/Scripts/python.exe -m pip install torch torchvision`
- [ ] The two weight files (not in git, about 45 MB each), from the GitHub release
      https://github.com/marshalSebastine/ImitationLearning-Ned2/releases/tag/policies-2026-11-13-overnight
      (SHA-256 sums in the release notes). In PowerShell, these four lines put them at exactly the right places:

      New-Item -ItemType Directory -Force D:/AI_Project/04_runs/2026-11-13_overnight_history_classifier_seed1/checkpoints
      Invoke-WebRequest https://github.com/marshalSebastine/ImitationLearning-Ned2/releases/download/policies-2026-11-13-overnight/history_classifier_seed1_best.pt -OutFile D:/AI_Project/04_runs/2026-11-13_overnight_history_classifier_seed1/checkpoints/best.pt
      New-Item -ItemType Directory -Force D:/AI_Project/04_runs/2026-11-13_overnight_factorised_classifier_tuned_seed1/checkpoints
      Invoke-WebRequest https://github.com/marshalSebastine/ImitationLearning-Ned2/releases/download/policies-2026-11-13-overnight/factorised_classifier_tuned_seed1_best.pt -OutFile D:/AI_Project/04_runs/2026-11-13_overnight_factorised_classifier_tuned_seed1/checkpoints/best.pt
- [ ] Tests, in `D:/AI_Project/02_code`: `..\.venv\Scripts\python.exe test_run_policy.py` says "all tests passed"
      (about a minute), and `..\.venv\Scripts\python.exe test_record_episodes.py` too.
- [ ] A dry run with the simulated robot, to see the windows and keys (nothing moves):

      D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/record_episodes.py --operator emranul --mode policy --simulated --checkpoint D:/AI_Project/04_runs/2026-11-13_overnight_history_classifier_seed1/checkpoints/best.pt

      It prints `policy history_classifier ... on cpu`. Enter a few steps, try `g` and a key to pause, then Esc.

## At the robot

- [ ] Ethernet connected; `check_hardware.py` says "Calibration needed: False" (automatic calibration only).
- [ ] Same table, light and camera taping as the main data (ceiling lights on for the first runs; write the light
      into `--lighting`). Camera check at the start of the episode under 4 px (the tool checks).
- [ ] Sheet taped as in its set-up picture for a STD episode (`06_deliverables/research_notes/Recording tool/figures/setup_per_episode/`).
- [ ] The start point for that sheet and end, as taught for the main data: the same `--start-x --start-y` (and
      `--start-push-deg --start-turn-deg` if it was taught turned) that the recording page used for its STD episodes.
      The first line of each log in `D:/AI_Project/03_data/recording_page_logs/` is that episode's full command.

The command (one line; change the checkpoint, sheet, start and notes):

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/record_episodes.py --operator emranul --mode policy --checkpoint D:/AI_Project/04_runs/2026-11-13_overnight_history_classifier_seed1/checkpoints/best.pt --path-id path_03_two_curve --start-x 0.XXX --start-y 0.XXX --orientation "policy A, end A, STD" --lighting "ceiling lights"

Policy runs go to `03_data/raw/policy_runs/`, never into the main data. The step budget is the sheet's own (1.5 times
the line length in 1 cm steps); after that the episode ends by itself.

## How to run each episode

1. Puck placed and centred as for a STD demonstration; hands out of the picture; Enter at "Put the puck at the start".
2. **The first 10 steps one by one with Enter.** Read the proposal under the picture ("policy: forward_left, forearm
   none ...") before pressing. If the steps look sensible, press `g` to let it run; press any key to pause again.
3. It pauses by itself when the robot refuses a step and before the policy's own stop: Enter accepts.
4. **Space** (stop by hand) when the puck is more than about 2 cm off the line and the policy is not bringing it back,
   or it pushes the same way for more than about 5 steps with the puck clearly lost. **x** (discard) only for a safety
   reason (puck tipped, robot stopped, hand in the picture).
5. "Puck in the goal?" afterwards: `y` only with the puck's centre within about 1 cm of the end cap.
6. Write one line into the results table below: how far along the line the puck got (the plan's distance marks, or
   measured with a ruler, in cm), and what went wrong first.

## Order (about 1.5 to 2 hours; stop earlier if A and B both fail at the same place every time)

| # | Sheet, end | Policy | Runs | Why |
|---|---|---|---|---|
| 1 | 01, end A | A | 1 | a sanity check on the easiest sheet |
| 2 | 03, end A | A, then B | 3 each | the sharp turn |
| 3 | 04, end A | A, then B | 3 each | hairpins, the longest |
| 4 | 06, end A | A, then B | 3 each | dashes and the crossing |
| 5 | 02 or 05 | the better one | 2 | sheets Emranul's models finished, for comparison |

## Results (fill in at the robot)

| Run | Sheet, end | Policy | Puck in goal? | How far (cm of the line) | Steps | What went wrong first | Notes (pauses, refused steps, light) |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |
| 9 | | | | | | | |
| 10 | | | | | | | |
| 11 | | | | | | | |
| 12 | | | | | | | |
| 13 | | | | | | | |
| 14 | | | | | | | |
| 15 | | | | | | | |
| 16 | | | | | | | |
| 17 | | | | | | | |
| 18 | | | | | | | |
| 19 | | | | | | | |
| 20 | | | | | | | |

## Afterwards

- Push this file with the table filled in, and copy `03_data/raw/policy_runs/` to Marshal (not in git): the pictures
  of where it failed show what the next round of demonstrations or training must fix.
- What is not acted on yet: the lean label (the rod keeps the least lean, as in the gamepad mode). If the rod's lean
  matters on the far end of sheet 04, note it.
