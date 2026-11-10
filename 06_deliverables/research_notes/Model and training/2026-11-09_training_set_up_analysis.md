# Training set-up analysis: what to build first, the backup, and the plan to the deadline

**Date:** 9.11.26, written in the evening from the results on disk at 22:36; revised after midnight (10.11, about 00:50)
after three independent critiques, with the runs that had finished by then.

**Written by:** the Model and training session (Claude) for Emranul Hoque and Marshal Sebastine (Team M, Project 6,
imitation learning on the Niryo Ned2).

**Status**

| Part | State |
|---|---|
| Data audit and cleaning rules (section 3) | final: 165 findings, each re-checked by two independent checker agents |
| Comparison code and settings (`02_code`, the `TRAINING_*` dictionaries in `config.py`) | final for this round; results in `04_runs` |
| First set-up, the "final candidate" (section 2.1) | six test copies trained, each on five sheets and scored on the sixth, one training seed (finished 22:02). Not built yet: the exam copy trained on all six sheets, and the recording tool's model mode that lets a model drive |
| The earlier exact run (old rules) on all six sheets | finished 23:01; used in sections 2.1, 2.4 and A.4 |
| Stop rule, executor rules, robot test plan (sections 4, 7, 9) | proposals; nothing has run on the robot |
| Backup set-up and a second seed of the first set-up | the backup running since 01:09 on 10.11 (expected by about 03:30), then the second seed (by about 05:30) |
| Other runs on the laptop | section 11.3: the mirror and chunk reruns finished at 00:04 and 01:08 with the same numbers as the first runs; the no-augmentation run follows the second seed (about 06:30); no number of a pending run is used or guessed here |

Nothing in this analysis ran on the robot and nothing in `03_data/raw` was changed. The model's only input is the one
forearm camera, and its motion is learned from our own demonstrations only. Every choice below is a proposal: Emranul
and Marshal decide, and every robot step waits for that decision and a written checklist.

**Markers.** [measured]: read from a run file or counted in the data. [calculated]: arithmetic on measured numbers.
[literature]: a published result that an independent checker confirmed against its source (in its corrected form where
the checker corrected it). [read in the code]: read in the project's code, not run. [judgement]: reasoning on the
evidence. [guess]: an assumption nobody has tested. A "point" is one percentage point. The auditors, checkers, critics
and judges named below are Claude agents, not people. Technical words are explained in the glossary (section 12.3).

---

## 1. Summary for Emranul and Marshal

*To act, read this section, 2.1, 2.5 and 9. Sections 3 to 8 are the evidence, 10 the open questions, 12 how to run the
code and the glossary.*

- **The data** [measured]. 210 demonstrations, 18,522 rows (a row is one picture and the move recorded after it). The
  bookkeeping is right, but the hand-guided labels are noisy, 449 rows are pauses and episodes from one sitting are
  near-copies, so only a sheet left out of training (a *held-out sheet*) tests an unseen track. Training uses 18,032
  cleaned rows, 259 of them stops.
- **Use first: the "final candidate"** (the run's name, not a final model) [judgement of all three judges]. A ResNet-18
  network, pretrained on everyday photos and trained on our pictures, predicts from one camera picture the next step as
  numbers (tip forward, tip sideways, forearm twist) and the chance that this is the stop. On the robot it would step 1
  cm in that direction, twist within limits, keep wrist and hand at fixed values, and stop at a chance of 0.5 or more on
  two pictures in a row. Six test copies exist, each trained on five sheets; the exam copy (all six sheets) and the
  recording tool's *model mode*, which lets a model drive, are not built yet (4.6). Emranul's plan of 6.11 was the move
  names first; switching is his and Marshal's decision (1.1).
- **Offline scores** (recorded pictures of unseen sheets, not robot runs; one training seed, that is one random start)
  [measured]. The push points within 45 degrees of where the demonstrator's tip went in the next three steps in 74 to 86
  of 100 pictures (always straight ahead: 43 to 80). Of 209 held-out episodes it would stop within 2 cm of the
  demonstrator's stop in 102, over 3 cm early in 25, 2 to 3 cm short in 13 and never in 69 (21 of 25 on the dashed
  loop). In natural light its lead roughly halves (3.7). Only robot runs show how often it finishes [judgement].
- **Backup: names plus exact** (results expected by about 03:30 on 10.11) [judgement of two judges]. The same network plus an output
  for the 8 move names, steering by their average direction, aimed at the first set-up's weak spot, pushing backwards: on
  the backward-labelled moves of sheets 03, 05 and 06 an earlier names model pointed within 45 degrees on 0.66, the
  first set-up on 0.43 [measured; one run each]. Whether that helps win back a puck that slid is not shown offline (2.2).
- **Next steps** (owners are yours to agree). (0) Note what meeting 2 answered. (1) 10.11, once the laptop queue is done (about 06:30): run
  `compare_runs.py`, which now also scores every run with the fixed two-in-a-row stop rule, and decide on the backup
  with 2.5. (2) Build the model mode (no estimate exists), check
  that it reproduces the saved predictions, try it in the tool's `--simulated` mode (the robot does not move) and write
  the session-1 checklist (7.4). (3) Before meeting 3 on 23.11, send the professor questions 1, 3 and 5 of 10.1. Plan
  with about 31 to 46 robot hours to 20.12.26 [calculated, partly from guesses; 8.4].

### 1.1 Decisions for Emranul and Marshal

"In the code" means Claude already applied it in the final candidate; each can be undone with a retrain.

| # | Decision | Proposal | Evidence | In the code? |
|---|---|---|---|---|
| 1 | Exact step instead of the move names first (changes Emranul's plan of 6.11) | exact step | 2.4, 4.1 | yes, in the final candidate |
| 2 | Stop rule | fixed 0.5, two pictures in a row, the same for every model | 4.5 | no: the fold files and `compare_runs.py` report one picture at 0.5 and the tuned rules |
| 3 | Robot rules for the first tests | 1-cm step, twist clamped, joint 4 within 115 degrees, wrist and hand at fixed start values, step budget on | 2.1, 4.3 to 4.6 | no: the model mode does not exist |
| 4 | What the robot does after a refused step | none proposed yet (stop, hand over to the operator, or a shorter step) | 2.1, 4.6 | no |
| 5 | Keep, promote or drop the backup | the rules of 2.5 | 2.5 | run pending |
| 6 | The audit cleaning (449 null rows dropped, 50 stops relabelled, score 1 kept) | as applied | 3.9, 10.3 | yes |
| 7 | Who builds the model mode, by when, and how the weights reach the builder | none yet | 4.6, 9 | no |
| 8 | Which professor questions go now, and who sends them | questions 1, 3 and 5 of 10.1 | 10.1 | no |
| 9 | The plan of robot session 1 (phases 0a, 0b, 1a) | 7.4 | 7.4 | needs a checklist first |
| 10 | The seven facts only you can confirm | answer them | 10.2 | no |

---

## 2. What to build first, and the backup

The design panel (Claude agents: five drafted sections, ten critics, three judges) ranked whole training set-ups. All
three judges chose the same first set-up. Two chose "names plus exact" as the backup and one (judge 3) the chunk target.
Below rank 2 the judges' orders differ; section 2.3 shows judge 1's order and where the others differ.

Two measures are used throughout. *Step within 45*: the share of moving pictures whose predicted push direction is
within 45 degrees of the demonstrated next step. *Path within 45*: the same against the direction to where the
demonstrator's tip is three steps later, which is less noisy than one hand-guided step and hardly inflated by copying
the last move. Both are offline scores on recorded pictures of a held-out sheet.

### 2.1 The first set-up: the final candidate

Configured as `resnet18_final_candidate` in `config.py` (lines 1797 to 1804). "Final candidate" is only the run's name:
it is the first set-up, not a final model. Its six fold copies (one per held-out sheet) are for testing; the *exam
model* would be a seventh copy, trained later on all six sheets with the same recipe.

| Part | What it is |
|---|---|
| Input | one colour picture from the forearm camera, shrunk to 160 x 120 (the camera's 4:3 shape), with ImageNet normalisation; nothing else (no joint angles, no previous move) |
| Network | ResNet-18 pretrained on ImageNet (`timm resnet18.tv_in1k`), trained end to end on our pictures; the head keeps a 3 x 4 grid of the last feature map, then one hidden layer of 256 with dropout 0.2; 12.75 million parameters, a 51 MB weights file |
| Outputs | the exact next step in the current picture's frame: tip forward and tip left (cm) and forearm twist (degrees); rod lean and hand change are trained as extra targets but not executed; a separate stop probability trained on the 259 stop pictures |
| Training | the audit-cleaned rows (18,053 kept by the cleaning; 18,032 rows in 209 episodes after also leaving out the 21 rows marked as mistakes); scores 1 to 3 at equal weight; Huber loss (a squared error gentler on outliers), with no extra weight for rare directions; stop output with positive weight 10 (a stop picture counts ten times); augmentation: shift of up to 4 px, brightness and contrast changes, saturation, gamma, a per-channel colour gain, light blur and noise, no mirroring; AdamW (head learning rate 0.001, backbone 0.3 times that, weight decay 0.05, batch 64), one warm-up epoch then a cosine decay to zero; 8 epochs, each on a random half of the rows (about 4 passes); the last weights are kept. The optimiser words are settings in `config.py` that need no change (glossary) |
| Executor (how the robot carries it out) | direction: the angle of the predicted (forward, left) vector; length: a fixed 1.0 cm; twist: as predicted (gain 1), clamped to 10 degrees per step; forearm (joint 4) kept within 115 degrees either way, and at the clamp the robot holds the twist, keeps stepping and logs it; wrist (joint 5) and hand (joint 6): before each run set to the median of their start values over the demonstrations (to be computed from `frames.csv` before session 1) and held there; joints 1 to 3 solved for the tip with `solve_with_wrist_fixed`; one move per picture at the recorded 20 % speed cap; guards: a workspace taken from the sheet in use, a height guard, and a refusal counter, because after a refused step the loop asks again from an almost identical picture and the model could repeat the refused step for ever (audit code_actions-11); what the robot does after repeated refusals is an open decision (1.1, row 4); the step budget switched on (`is_step_budget_enforced` in `config.py`; about 4 steps per cm of line, or about 250 steps when the length is unknown) |
| Stop rule | stop when the stop probability is 0.5 or more on two pictures in a row (the second picture comes after the next normal step); fixed in advance, never tuned per model; one picture at 0.5 is the alternative for a robot A/B test |
| Validation | leave one sheet out (six folds), each sheet beside its own always-forward rule: path within 45 (mean and worst sheet) first, then step within 45, the stop as counts (right, early, 2 to 3 cm short, never) under the fixed rule, the twist sign, backward moves and the pictures after hand corrections; then, after the team decides, robot runs of each fold model on its own held-out sheet |

**What a builder needs** [measured or read in the code]:

| What | Where or how |
|---|---|
| Train it | from `D:\AI_Project\02_code` once this branch is there: `..\.venv\Scripts\python.exe train_policy.py resnet18_final_candidate` (all six held-out sheets; section 12.1) |
| The six test copies | `04_runs/2026-11-09_resnet18_final_candidate/checkpoints/without_path_01.pt` to `_06.pt`, 51 MB each, each the copy that never saw that sheet; out of git and only on Emranul's laptop, so they must reach Marshal another way (for example a shared drive) |
| The exam model | not trained: `train_policy.py` needs an "all sheets" option first (its arguments today: the run name, `--folds`, `--epochs`, `--train-row-limit`); about 1 to 1.5 hours per seed |
| Loading a saved copy | no code in `02_code` does it yet; the training-settings exam-risk critic's `stress_test_final_candidate.py` (now in `evidence/design_review/training_settings/critic_exam_risk/`) loads the copies and reproduces their saved predictions exactly, a starting point for the model mode |
| Preparing a camera picture | the 640 x 480 picture in RGB colour order, shrunk to 160 x 120 with `cv2.INTER_AREA` exactly as `training_dataset.py` line 436 does, then the ImageNet normalisation of `TRAINING_RUN_PARAMETERS["normalisation"]` |
| Not built yet | the model mode with its nine missing pieces (4.6), and the hand-over to the exact-replay decider after a "reguide" |
| Do not change | the stop weight (10) without re-checking the stop: with a fixed threshold it sets where 0.5 falls (6.3) |

**Why it comes first**

- It is the only set-up trained in its final form and scored on all six held-out sheets: path within 45 0.735 to 0.856
  (mean 0.786) and step within 45 0.632 to 0.779 (mean 0.678), above always forward on every sheet [measured; final
  candidate fold files]; on the same pictures the path margin is 5.5 to 38.0 points [calculated by judge 1].
- Its two main design choices are the firmest results of the comparison: training the whole network beat frozen
  features by 7.6 to 19.2 points of step within 45, and the exact step beat the move names on path within 45 and on the
  twist (section 2.4). [measured]
- Its stop is at least as good as the other end-to-end runs', but the lead is inside the seed noise (under the 7-point
  bar of section 6.1). On sheets 03, 05 and 06 it stopped right in 47 of 99 episodes at 0.5 with one picture (46 of 99,
  with 12 early and 36 never, under the proposed two in a row), against 42 of 100 for the same design without the
  cleaning and 18 to 41 for the class heads [measured; judge 1; one seed each]. On all six sheets the earlier exact run
  (old rules) now gives the like-for-like comparison: step within 45 0.678 for both, path within 45 0.786 against 0.781
  (worst sheet 0.735 against 0.726), and at 0.5 with one picture 101 right, 41 early and 50 never of 209 against 90, 47
  and 59 of 208, a gain of 5.0 points of right stops [calculated from the fold files; one seed each, and the cleaning
  changes the test rows slightly]. So the cleaning and the last weights cost nothing measurable and may help the stop
  [judgement].
- It fits the laptop and the robot loop: 23.4 to 55.0 minutes per held-out sheet to train, 37.4 ms per picture (busy
  laptop) against a 1.22 s robot step. [measured]

**Main risks.** The stop (102 of 209 right; 21 of 25 never on the dashed loop). Backing up: 647 of 1,894 (34 %) of the
backward-labelled moves within 45 degrees, against 70 % of the other moves; about half of these moves are ordinary
pushing seen through a turned camera, so this is not a direct measure of recovery after a drift [measured]. Light: in
natural light its lead over always forward is about half that under the ceiling light, and the exam light is not known
(3.7) [measured; calculated]. And everything closed loop, which no offline number shows [judgement].

### 2.2 The backup: names plus exact

Configured as `resnet18_backup_names_plus_exact` (`config.py` lines 1806 to 1813); its six-fold run started at 01:09
on 10.11 and is expected by about 03:30.

| Part | Difference from the first set-up |
|---|---|
| Outputs | a second head for the 8 move names (forward, forward-left, left, back-left, back, back-right, right, forward-right) with square-root class weights, beside the exact-step head and the same separate stop output |
| What the robot executes | the direction is the names head's *mean direction*: the average of the 8 move directions, each weighted by how likely the model finds it; the twist comes from the exact head; length, clamps, wrist, hand and stop rule as in 2.1 |
| Everything else | identical: picture, network, cleaning, augmentation, optimiser, 8 epochs, last weights, validation |

**Why this backup**

- It aims at the first set-up's clearest measured weakness after the stop: pushing backwards. Backward-labelled moves
  within 45 degrees (this table is the reference for every backing-up number in the report):

  | Model | Sheets | Backward moves within 45 degrees |
  |---|---|---|
  | Names model read as a mean direction (earlier run, old rules) | 03, 05, 06 | 0.663 of 1,054 |
  | Exact model (earlier run, old rules) | 03, 05, 06 | 0.392 of 1,054 |
  | Final candidate | 03, 05, 06 | 0.426 of 1,054 |
  | Final candidate | all six | 647 of 1,894 (0.342) |

  [measured; judge 2; the 0.426 calculated by judge 2 from per-sheet shares]. About half of these moves are ordinary
  pushing seen through a sideways-turned camera, and only 15.6 % lie in the first six rows of a hand correction (audit
  class_balance-10), so this is not a measure of recovery after a drift.
- In the first five rows after a hand correction the names model read as a mean direction pointed 216 of 500 moving
  pictures within 45 degrees, the earlier exact model 169 (both on the old rules, like for like), the final candidate
  151 and always forward 138 [measured; judge 1].
- It keeps the exact twist and the same stop output, so it should keep the camera turning on loops, where names-only
  twist heads got the sign right on only 0.37 to 0.47 of the twisting rows [measured; the backup's own twist is
  untested, so judgement].
- In the frozen probes the combined design matched the exact step at the same seed (step within 45 0.602 against
  0.591, path 0.677 against 0.673; the exact step's four-seed means are 0.603 and 0.691) and backed up better (0.363
  against 0.270). [measured]
- It needs one configuration entry and, as far as the code was read, no new training code. [judgement; read in
  `config.py` and `train_policy.py`]

**Main risks.** It has never been trained end to end. The separate names model, read the same way, gave up 4.6 points
of path within 45 (0.744 against 0.790) [calculated by judge 1 from the two runs]. Over all 2,921 pictures after a hand
correction on sheets 03, 05 and 06 the names model read as its most likely class and the exact model tied (0.605 and
0.605), so better backing up has not yet shown as better recovery offline [measured; validation draft V4]. Counting all
three back classes, the names model predicts backward moves on 22.1 % of the pictures against 13.6 % demonstrated, a
risk of stepping back and forth (dithering) [measured].

### 2.3 All options, ranked (judge 1's order)

| Rank | Set-up | State on 9.11 | Why this place | Main risk |
|---|---|---|---|---|
| 1 | Final candidate: exact step plus own stop, ResNet-18 trained end to end (2.1) | six held-out sheets, one seed | complete; at least as good a stop as any other end-to-end run on the shared sheets; best path and twist measured end to end | the stop; backing up; no closed-loop evidence |
| 2 | Names plus exact on the same recipe (2.2) | six-fold run queued | aims at rank 1's weakest measured skill, backing up | never trained end to end; may lose path within 45 |
| 3 | Chunk of 5 future tip places, first step executed (judge 3's backup) | sheets 03, 05, 06 on the old rules | best path within 45 on 03 and 05 (0.837 and 0.795); on 06 0.770 against the final candidate's 0.810, so about +1.2 points over the three sheets (different test rows) | its stop: 31 of 100 right at 0.5 against 42 for the exact run, and on sheet 06 it never stopped in 24 of 25 episodes; needs a rerun with the cleaning and last weights |
| 4 | Small network trained from scratch, exact step plus stop | only its names version was run | ties the ResNet-18 when both names heads are read as a mean direction; tiny and fast | its exact version never ran; pretrained weights are allowed anyway (question 10, answered 26.10.26) |
| 5 | Names head with square-root weights and mirrored pictures | sheets 03, 05, 06 on the old rules | best after a hand correction (221 of 500 within 45 degrees); about level with the plain names head on backing up (0.68 against 0.71 with a straight forearm); best stop of the names heads (41 of 100) | twist sign 0.37 to 0.47 on the loops; mirrored pictures cannot occur with this camera; lost 4.1 points on sheet 06 |
| 6 | One picture plus the change since the previous picture | frozen probe only | +2.6 points of step within 45 on the six-sheet mean | its worst sheet fell; the gain may come from copying the last move |
| 7 | Frozen backbone plus a trained head | six sheets, four seeds | cheap bench for quick stop experiments | 7.6 to 19.2 points below end-to-end training |
| not ranked | Small network on hand-made line-and-puck maps | sheets 03, 05, 06 on the old rules | no offline gain over colour pictures (-1.0 to 0.0 points of step within 45) | its aim, new light and tables, is hardly tested by six sheets |
| not ranked | Fine-tuned MobileNetV3 or EfficientNet-B0 | configured, not run | frozen, neither beat ResNet-18, and speed is not a constraint (5.2) | untested end to end |
| not now | ACT, diffusion policies, fine-tuned DINOv2 or CLIP | literature only | not run: too costly for the laptop now (section 5.3) | no evidence that they handle unseen tracks better |

[judge 1's order. All three judges put the final candidate first. Judge 2: names plus exact 2nd, the small network 3rd,
the chunk 4th, the final candidate with the tuned stop 5th, names only 6th, mirrored names 7th, frozen 8th. Judge 3: the
chunk 2nd (his backup), the final candidate with the tuned stop 3rd, the small network 4th, names plus exact 5th, the
names head with or without mirroring 6th, frozen 7th. The numbers are measured and appear in sections 4 and 5.]

### 2.4 The evidence for each choice

| Choice | Evidence | Kind |
|---|---|---|
| Train the whole network, not frozen features | On the same pictures, end-to-end exact beat the frozen exact probe by 7.6, 10.4 and 19.2 points of step within 45 on sheets 03, 05 and 06 (episode-bootstrap intervals 6.0 to 9.1, 7.7 to 13.1, 15.2 to 22.6) | measured |
| | Frozen pretrained backbones were only 0.7 to 3.9 points above always forward on the six-sheet mean | measured |
| | robomimic Square with 200 demonstrations: ResNet-18 0.94 trained from scratch, 0.92 fine-tuned, 0.58 frozen (Chi et al. 2023) | literature |
| Exact step, not the 8 move names | Same network and sheets: path within 45 +9.8, +6.7 and +8.4 points over the names read as the most likely class, +5.1, +2.4 and +6.3 over the names read as a mean direction; twist sign 0.845, 0.669, 0.802 against 0.703, 0.372, 0.422 | measured, one seed each |
| | On step within 45 the lead is smaller: +2.6 points over the most likely class, +1.3 over the mean direction (per sheet -1.1 to +3.8) | measured |
| | The literature is mixed: from pictures with a ResNet-18, regression was not worse than bins (Pearce et al. 2023: 17.8 against 14.7); in planar pushing from pictures, plain regression scored 35.0, 55.0, 6.7 and 19.6 % against 85.0, 88.3, 83.3 and 48.3 % for an implicit model (Implicit BC, Table 6) | literature |
| ImageNet ResNet-18 rather than the small network | Names heads, most likely class: +2.9, +2.5 and +1.7 points of step within 45 on 03, 05, 06; read as a mean direction: +1.9, +1.9 and -4.4 points of path within 45, so the data do not settle it | measured |
| | An ImageNet start narrowed the seed-to-seed spread of closed-loop success in driving (Codevilla et al. 2019) | literature |
| | Kept for steadiness and because it is ready | judgement |
| One picture, no picture history | The change since the previous picture raised step within 45 from 0.577 to 0.603 on the six-sheet mean, but the worst sheet fell from 0.467 to 0.454; repeating the previous move scores 0.736, so a history input can earn offline points by copying | measured (frozen probes) |
| | History policies copy the previous action when consecutive actions are alike (Wen et al. 2020) | literature |
| The audit cleaning | Offline, every cleaning variant sat inside the seed noise of the frozen probes; end to end on six sheets the cleaned recipe with last weights matched the old rules on direction and stopped right 5.0 points more often, inside the 7-point bar (2.1); the case rests on the dropped rows having wrong labels for their pictures | measured; calculated; judgement |
| | Training with idle actions contributed to single-step policies getting stuck on a real robot (real Push-T: 8 of 20 runs; one contributing cause; Chi et al. 2023) | literature |
| A fixed 8 epochs and the last weights | In robomimic the lowest-validation-loss checkpoint was 57 to 100 % below the best, the last checkpoint 12 to 35 % below (Mandlekar et al. 2021, Table 29) | literature |
| | No held-out-sheet score per epoch exists on our data | judgement |
| A fixed 1-cm step | The predicted step shrinks to a median 0.60 cm on curved unseen sheets (earlier run), 0.68 against 1.11 cm on sheet 02 (final candidate) and 0.22 to 0.36 cm within 2 cm of the goal; with the predicted length the robot would stand almost still on 2.5 % of the pictures more than 3 cm from the goal (earlier run) | measured |
| Twist at gain 1 | Least-squares gains of the demonstrated on the predicted twist are 0.55 to 1.24 per sheet (below 1 on five sheets), so no single gain fits all sheets | measured; gain 1 is a judgement |
| Twist clamp of 10 degrees per step | Below the demonstrated 99th percentile of 11.6 degrees per step, so somewhat more than 1 % of demonstrated steps twist further; never tested | measured; the value is a judgement |
| Wrist and hand held at fixed start values | Solving joints 1 to 3 with wrist and hand kept reproduces the next demonstrated view within 0.5 to 0.7 px per step (90th percentile, with the demonstrated wrist at each step); the keyboard and gamepad paths would move the picture content a median 48 to 56 px (views never seen in training); no executor for a predicted wrist exists. Against it: held at one value over a whole run, the picture centre moves more than 20 px in 23.3 % of the rows, so a wider shift augmentation is proposed for the exam model (6.2) | calculated by the audit; judgement |
| 160 x 120 colour pictures | No size comparison on our data; papers used 84 x 84 to 320 x 240 (6.5) | literature; judgement (untested) |
| The head: a 3 x 4 grid, one hidden layer of 256, dropout 0.2 | Frozen probes: a linear head lost 7.8 points of step within 45, a 1,024-wide layer 1.9 (6.8); dropout in the head helped on real robots (Dasari et al. 2023) | measured (frozen); literature |
| Huber loss | No comparison with another loss on our data | judgement (untested) |
| Stop weight 10 | No other weight tested; with a fixed threshold it sets where 0.5 falls (6.3) | judgement (untested) |
| The photometric augmentation | Ranges set from the measured light and colour of our pictures (6.2); its effect end to end is the pending no-augmentation run (11.3) | measured ranges; judgement |
| Stop at a fixed 0.5, two in a row | Over all 209 held-out episodes: 102 right, 25 early, 69 never; the threshold tuned on validation episodes (two in a row) gave 98 right but 58 early; no rule tried stopped right in more than 105 | measured |
| Backup: names plus exact | Backward moves 0.663 against 0.392 for the earlier exact model and 0.426 for the final candidate (03, 05, 06); first five rows after a correction 216 against 169 of 500 like for like, 151 for the final candidate; a tie (0.605) over all pictures after a correction | measured |

### 2.5 When to switch to the backup

**Offline, when the backup's six folds are in (expected by about 03:30 on 10.11)** [judgement; thresholds from judges 1 and 2]. The
two-in-a-row stop, the backing-up counts and the first rows after a hand correction come from the judges' scripts
(`judge_rescoring.py`, `judge2_checks.py`, `judge_recovery_checks.py`; appendix A.6), pointed at the new run names;
`compare_runs.py` does not compute them yet.

- Keep it as the backup (judge 1's rule) if its path within 45 is at most about 3 points below the first set-up's
  0.786, its first rows after a hand correction are clearly better than the first set-up's (415 of 1,124 within 45
  degrees on the step, 573 of 1,062 on the path, over six sheets), and its stop under the fixed rule (0.5, two in a
  row) is not worse than about 95 right and 30 early of 209 (first set-up: 102 and 25).
- Consider it as the first set-up if, in addition, its path within 45 stays within about 2 points (judge 1), or its
  backward moves within 45 degrees reach about 0.45 or more (judge 2's condition; first set-up 0.342); the robot then
  decides between the two.
- Drop it if it loses more than about 5 points of path within 45 or its stop is clearly worse. The chunk target, rerun
  with the cleaning and the last weights on six sheets, then becomes the backup.
- Between these (3 to 5 points lower on path within 45, or a stop between about 95 right and clearly worse): keep it as
  the backup, mark it weak, and let the robot decide.
- If the first set-up's second seed moves the six-sheet numbers by more than the gap to the backup, the two are tied
  offline and only the robot can choose.
- Before any robot time, the backup passes the same offline gate as the first set-up (section 7.2).

**On the robot, after the first runs of the fold models on their own held-out sheets** [judgement; judges 1 and 2]:

- Failures mostly lost pucks that need a backward move, or no recovery after a drift: run the backup's fold models the
  same way, then a paired comparison if both finish most runs, including starts with the puck laid beside the block
  (judge 1; targets exam-risk critic).
- Failures mostly at the stop (early or never): keep the first set-up; the fix is data (line ends, section 8), not the
  set-up. Two stop variants can go into robot A/B tests: the combined rule (stop only when the stop probability passes
  and the predicted step is short), which needs no retraining, and a stop output trained on every picture within 2 cm
  of the goal, used at a high threshold, which needs one training run (4.5).
- Stalls, back-and-forth moves or joint-limit trouble: fix the executor; that applies to every set-up.

**Rules.** If the professor does not allow the executor to use the joint angles or fixed rules (section 10.1, question
1), every set-up needs a new executor, so the order stays but the plan changes. Pretrained vision models were already
allowed on 26.10.26 (question 10 on `05_meetings/requests_to_professor.md`), so the small network is not needed as a
rules hedge.

---

## 3. The data: what the audit found

Ten auditors each checked one side of the data, and two more agents re-checked every finding: one recomputed the
numbers, often from the raw files, and one attacked the conclusions. Of 165 findings, 61 were confirmed as they stood,
102 were corrected in a number or a wording, 1 was refuted and 1 could not be tested. The full, checked text is
`evidence/audit_synthesis.md`; the numbers below are from it unless marked otherwise.

### 3.1 The bookkeeping is sound

- Each label belongs to the right picture: the label of a row is the move from its picture to the next one (0.09 mm
  median error). [measured]
- Every picture file still has the MD5 checksum it was logged with, and the recording tool's direction rule is reproduced on more
  than 99.9 % of the moving rows (`test_training.py`). [measured]
- Two episodes have their score only in review.json (01-09 try 1 and 02-41 try 1, both score 2), and the start end is
  wrong for 30 episodes in the index (19 on sheets 05 and 06, 11 on sheet 02). The fixed table
  `evidence/corrected_ends.csv` replaces it; 18 of its rows were decided by hand, so it must never be regenerated.
  [measured]

### 3.2 Class balance per sheet

Share of each sheet's rows, in %, before the cleaning (all 18,522 rows) [measured; audit table 1.2]:

| Sheet | Rows | Forward | Fwd-left | Left | Back-left | Back | Back-right | Right | Fwd-right | Small motion | Stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 straight | 671 | 59.2 | 8.0 | 4.6 | 1.8 | 2.7 | 1.0 | 4.5 | 4.6 | 10.6 | 3.0 |
| 02 arc | 2,775 | 37.7 | 14.4 | 6.1 | 3.5 | 3.1 | 3.5 | 5.2 | 14.0 | 10.9 | 1.6 |
| 03 two curves | 4,072 | 33.2 | 13.9 | 6.1 | 1.6 | 2.0 | 4.9 | 9.1 | 14.5 | 13.5 | 1.1 |
| 04 meander | 6,350 | 32.3 | 12.5 | 7.8 | 2.0 | 2.1 | 4.1 | 10.6 | 11.7 | 16.2 | 0.7 |
| 05 loop | 2,672 | 39.7 | 10.0 | 10.7 | 5.7 | 5.0 | 4.9 | 7.4 | 9.2 | 6.3 | 1.1 |
| 06 dashed loop | 1,982 | 28.4 | 14.8 | 17.3 | 5.7 | 4.7 | 4.7 | 8.1 | 7.7 | 7.4 | 1.3 |

- After the cleaning the direction shares hardly change: it drops 449 small-motion rows and only 20 others, and
  relabels 50 small-motion rows as stop, so small motion falls from 2,265 to 1,765 rows and stop rises from 209 to 259.
  No per-sheet recount after the cleaning was made. [calculated from the counts in 3.9]
- Forward is 40.3 % of the 16,034 moving rows and straight back 3.4 %. [measured]
- The mix follows the camera turn: forward is 57.0 % of the moving rows with the forearm within 15 degrees of straight,
  but 28.4 to 31.6 % when it is turned more than 40 degrees. A model that chooses its own twist will meet a different
  mix. [measured]
- Sheet 04 holds 34.3 % of the pictures from 21.4 % of the episodes (a median 168 rows per episode). With
  picture-uniform sampling it is 35.8 to 44.2 % of the training rows in the folds that train on it. [measured]
- After the cleaning the three back classes are 11.8 % of the moving rows; the square-root class weights run from 0.59
  (forward) to 2.06 (back) and 2.86 (stop). [measured]
- Every direction class keeps at least 410 training rows in every fold; only the proportions shift. On held-out sheet 01
  the back classes have only 10 to 24 rows, so per-class results there are rough. [measured]

### 3.3 Label noise and consistency

The labels are noisy, and the noise is in the hand guiding, not in the code [measured]:

- The 8 move names cut a continuous direction into uneven bins (forward covers 60 degrees, left and right 37.5). 18.8 %
  of the moves lie within 5 degrees of a bin edge, 38.1 % within 10.
- From one 1-cm step to the next the direction changes by a median 21 degrees, and consecutive move names differ 55 % of
  the time. The changes follow real turns and drift of the path, not flicker. That smoothing the outputs over time
  would lag behind real turns is a judgement: offline, averaging the predicted direction over 2 or 3 pictures changed
  path within 45 by +1.4 and +1.8 points for the frozen ResNet-18 probe and by +0.3 and -0.5 for the final candidate,
  and in the literature averaging outputs did poorly in closed loop (section 5.4).
- Near-identical pictures from different episodes of the same sheet carry different move names in 26 % of the moving
  pairs (8 % differ by more than 45 degrees). Even a perfect names model could not match every label: for near-identical
  pictures the best achievable agreement is about 59 to 77 %.
- The forearm twist that comes with most steps is not in the move name: 56 % of the non-stop rows twist by 3 degrees or
  more, and the names keep only 16 % of the twist's variance (against 81 to 85 % of the tip move's).
- Smaller sources: the executed move differs from the commanded one enough to flip about 1 % of the labels; at forearm
  turns beyond 80 degrees the label frame is off by several degrees (correcting it would change 403 of 16,034 moving
  labels, 2.5 %; not done, section 10.3); and after a hand correction the puck was put back more than 3 px (of 160) from
  its reference in 24 of 179 measurable corrections.
- Emranul and Marshal are not shown to label differently, but Marshal's 17 episodes (6.9 % of the rows) are too few to
  show that they label alike.

### 3.4 Near-zero moves

"small_motion" is three different things (2,265 rows, 12 %) [measured]:

| Part | Rows | What it is | What to do |
|---|---:|---|---|
| Row 0 of every episode, and the first row after every hand correction | 449 | The robot did not move; the picture is almost the next one, sometimes with a hand in it | Drop |
| The point just before the stop | 51 | A duplicated final waypoint (`record_episodes.py` line 1120); its picture is the stop picture | Relabel as stop |
| The rest | 1,765 | A forearm twist with the tip almost still, in both directions | Keep; the twist output learns it |

The 13 rows labelled turn_left or turn_right are large twists (4 of them from recording stalls); after the clip they are
ordinary 15-degree twists. [measured]

### 3.5 Hands, blur and the picture flags

- The picture check flagged 1,029 of 18,522 pictures (5.56 %), almost all "no puck seen" (524) or "dark object in view"
  (520); 17 were flagged as blurred and 14 as a sudden change. [measured]
- By eye: the sudden-change flags are real (6 of 6 show hands, a person or a puck not yet placed); the dark-object flags
  are mostly shadows, the printed line and clutter; the puck is visible in 35 of 39 "no puck" pictures checked (today's
  detector finds it in 377 of the 524); none of the 17 blur-flagged pictures is blurred. [measured]
- Hands were found in 15 of 210 first pictures, 36 of 266 first pictures after a pause, and 1 of 140 random mid-episode
  pictures. The cleaning rule that drops row 0 and the first row after each correction (both logged as waypoint 0)
  removes 53 of the 55 pictures in the challenge checker's hand-census file; the other two and a few named in review
  notes are on the look-by-eye list (section 3.9). [measured]
- Sharpness: the measured spread fits at most about 0.5 px of blur at 640 x 480; natural light lowers it (median 163.2
  against 219.4 under the ceiling light). Blur augmentation is low priority. [measured]

### 3.6 Near-copies between episodes (leakage)

**Leakage decides how to validate.** 61.5 % of the pictures have a near-twin in another episode of the same sheet, but
only 2.8 % on another sheet, and the twins belong to the recording block: episodes recorded within 15 minutes of each
other share many views, episodes recorded hours apart almost none. A "model" that only copies the label of the most
similar training picture, which cannot generalise at all, errs by a median 27 degrees when whole episodes are held out
and by 53 degrees when a whole sheet is held out, worse than always pushing forward on five of the six sheets. So a
score on held-out episodes measures memory of the recording block, and is optimistic even for a known sheet laid out
again in a new session, as at the exam. Only holding out a whole sheet measures following an unseen track. The
independent units are few: about 1,200 to 1,700 independent views of picture content, about 3,100 independent labels,
210 episodes and 6 sheets. [measured]

### 3.7 Light, camera view and things that are tied together

- Sheet, recording day, light, operator and camera period go together, so no split separates them. Natural light occurs
  only on sheets 02 (10 episodes) and 04 (15). [measured]
- Measured brightness, not the lighting note, marks 24 dim episodes (2,817 pictures), 23 of them in natural light; they
  are the only daylight examples and stay at full weight. Picture brightness runs from 0.74 to 1.12 of the median (1st
  to 99th percentile); the natural-light pictures reach down to 0.674. [measured]
- The taped camera's view moved once, during the break on 8.11.26 between 18:07 and 19:33 (about 15 pixels, mostly
  sideways, at 640 x 480; the cause is not known: the camera on its tape, a re-mounted block or rod, or a joint zero
  after a restart would look the same). Sheets 05 and 06 and the last 16 episodes of sheet 04 are in the later view (71
  episodes, 6,435 pictures). On sheet 04 every end-B episode is in the later view and every natural-light one in the
  earlier. [measured]
- **How the final candidate does in each light and camera view** (its six held-out sheets pooled; always forward on the
  same pictures in brackets; the analysis counts slightly more pictures than the fold files, for example 633 against
  578 on sheet 01) [measured; `training_error_analysis.py`, in `report_parts/results_so_far.md`; margins calculated]:

  | Pictures | Count | Path within 45 | Margin over always forward (points) |
  |---|---:|---|---:|
  | Ceiling light | 15,026 | 0.785 (0.585) | +20.0 |
  | Natural light (sheets 02 and 04 only) | 2,938 | 0.702 (0.605) | +9.7 |
  | Camera view A (before 8.11, 19:33) | 11,693 | 0.769 (0.629) | +14.0 |
  | Camera view B (from 8.11, 19:33) | 6,271 | 0.777 (0.518) | +25.9 |

  In natural light the lead roughly halves, but light is tied to sheets 02 and 04, so the cause cannot be separated. In
  the darkest third of the pictures the earlier exact model's stop probability passed 0.3 far from the goal on 0.041 of
  2,508 pictures against 0.018 of 2,573 in the brightest third, and the final candidate's direction (sheets 01 and 02)
  was 0.596 against 0.710, again tied to sheet and place [measured by the targets exam-risk critic]. Half brightness,
  darker than any training picture, raised sheet 02's early stops from 2 to 10 of 45 (4.5). The later camera view shows
  no loss. The exam light is not known (`requests_to_professor.md` question 19, open; question 5 of 10.1), so new
  recordings should vary the light (step A of 8.4) and the robot test keeps its light block (7.4). [judgement]
- All of this assumes the camera stays taped on the forearm cover as during the recordings; moving it would change every
  picture and need new demonstrations. [judgement]

### 3.8 How scores 1, 2 and 3 differ

| Score | Episodes | Pictures | Hand corrections per episode | Backward share of the rows |
|---|---:|---:|---:|---:|
| 1 | 24 | 2,345 | 1.63 | 11.4 % |
| 2 | 52 (two read from review.json) | 5,285 | 1.80 | 12.5 % |
| 3 | 134 | 10,892 | 0.82 | 9.0 % |

[measured; corrections per episode and backward shares use the 50 score-2 episodes named by folder]

- Score 1 cannot be told apart from score 2 by its motion; with 24 episodes that means "not distinguishable", not "shown
  equal". Scores 1 and 2 differ from score 3 mainly by more hand corrections. [measured]
- Score 1 holds the hardest starts: 55.0 % of the mid-track (MID) and 53.4 % of the 15-degree turned (TR15) start
  pictures. Dropping it would thin TR15 from 436 to 203 pictures and MID from 1,024 to 461. [measured]
- Label noise is about the same in every score (consecutive names differ 56.1, 57.8 and 53.5 % of the time), so dropping
  score 1 would not clean the labels. [measured]
- Frozen probes re-scored on the same pictures: score 3 only lost 4.3 points of step within 45 (lower on all six
  sheets); scores 2 and 3 lost 1.0; weights 0.4 / 0.7 / 1.0 lost 0.3. The stop differences stayed inside the seed
  noise. [measured]
- The final candidate on its six held-out sheets, by score, path within 45 (always forward in brackets): score 1 0.735
  (0.582), score 2 0.723 (0.551), score 3 0.804 (0.608). Pictures of scores 1 and 2 are harder for always forward too;
  the model's margin is 15 to 20 points for all three scores. [measured; margins calculated]
- So: keep scores 1 to 3 at equal weight (section 6.4). [judgement]

### 3.9 The cleaning, step by step

Implemented as `training_targets.audit_cleaned_frames` and switched on in the final candidate (`is_audit_cleaning`),
checked to give exactly these counts [measured]:

| Step | Rows | Why |
|---|---:|---|
| Drop row 0 of every episode | 210 | null move (median 0.11 mm); hands in 15 of these pictures |
| Drop the first row after every hand correction | 239 | null move (median 0.14 mm); hands in many of these pictures |
| Drop one more sudden-change picture | 1 | the 6 viewed were all real (hands or a person) |
| Drop the half run 05-16 (beyond its row 0) | 12 | its stop label sits in the middle of the track |
| Drop 04-01 try 6, rows 169 to 175 | 7 | the puck was pulled back by pressing its top |
| **Dropped in all** | **469** | **18,053 rows in 209 episodes kept** |
| Relabel the duplicated final points as stop | 50 (51, one in 05-16) | their picture is the stop picture; stop rows become 259 (209 stops plus 50) |
| Clip steps longer than 2 cm | 25 steps | only 4 are recording stalls; the rest are fast guiding with matching pictures |
| Clip twists at 15 degrees | 13 rows | the four largest come from recording stalls |
| Read the two scores from review.json | 71 pictures | 01-09 try 1 and 02-41 try 1, both score 2 |
| Use the corrected start ends | 30 episodes | from the fixed table `evidence/corrected_ends.csv` |
| Drop the 21 mistake rows (the worktree's default) | 21 | ordinary moves before a correction; they lie inside the 1,186 rows just before corrections, which are otherwise kept; little effect either way; leaves 18,032 training rows |
| Still to look at by eye | about 50 to 70 rows (0.3 to 0.4 %), plus the audit's lists D1, D2 and D5 | named hand pictures (02-45 frame 043, 03-18 frames 051 and 053, 06-14 frame 071, 04-22 frame 134, 06-20 frame 065, the last pictures of 02-02); 06-20 from the tip-over to the end (22 to 37 rows); 01-18 frames 039 to 043; 01-09 (12 pictures); the 7 relabelled stops whose picture changed by more than 3 grey levels; and (audit 11.1 D) the 27 first rows after a pause over 10 s that the waypoint marker keeps (21 stops, 6 moves), the 2 pictures where a hand persisted into the next one (frame_index 6459 and 12603), and the 863 rows whose fresh-picture wait exceeded 0.2 s, where stale pictures cluster |

Offline, none of the cleaning variants moved a frozen probe by more than the seed noise; the case for the cleaning is
that the dropped rows carry wrong labels for their pictures. [measured; judgement]

### 3.10 What should not be filtered

The 520 "dark object" flags (mostly shadows and the printed line), the 524 "no puck" flags (today's detector finds the
puck in 377 of them), the 17 blur flags (none is blurred), score 1 (not distinguishable from score 2 in motion, and it
holds many of the hardest starts), the 24 dim natural-light episodes (2,817 pictures, the only daylight examples), and
the hand-correction replays (37.9 % of the rows, recovery data). [measured]

### 3.11 What the camera sees at the stop

Every line end lies 2.35 to 8.03 cm from a paper edge, and the stop pictures show no line ahead: 90.0 % of the stop
pictures show no line in their top half, but so do 27.0 % of the pictures more than 3 cm from the goal (sheets 01 to 06:
41.4, 20.1, 22.4, 20.9, 37.9 and 46.4 %). A model may learn "stop near the paper edge" instead of "stop at the line's
end". An unseen track whose line ends far from the edge would expose this, so at least one printed test track should
have such an end. [measured; judgement]

---

## 4. What the model should predict, and how the robot carries it out

### 4.1 Move names, exact step, both, or a chunk

Step and path within 45 are defined at the top of section 2. A *frozen probe* trains only a small head on fixed
ResNet-18 features; *end to end* trains the whole network.

**The 11 classes of the plan against the 12 labels in the data.** The notes speak of 7, 9 or 11 classes, but the data
hold 12 labels: 8 directions, stop, small_motion, turn_left and turn_right. The 2,278 small_motion and turn rows (12.3 %)
lie outside the 11-class set of the 6.11 memory; the step-over classes never occur, because the replay's labeller can
output only these 12; and the diagonal classes sit at 60 degrees (the back diagonals at 135) since 4.11, not at the 45
the analysis of 30.10 assumed [measured; audit notes_context-2]. In the first set-up every row becomes an exact step
(small motions and turns are a short tip move with a twist) and the stop has its own output. In the backup the names
head has the 8 directions (a small motion takes the next move's name, the default rule in `config.py`), the exact head
gives the twist, and the stop keeps its own output [read in the code].

| Target, and how it is read | Held-out sheets | Step within 45 | Path within 45 | Twist sign |
|---|---|---|---|---|
| Always forward (no picture) | 6 | 0.538 | 0.604 | |
| Names, most likely class (frozen probe) | 6 | 0.577 (4 seeds: 0.546 to 0.577) | 0.633 | 0.408 |
| Names, mean direction (frozen probe) | 6 | 0.605 | 0.682 | 0.408 |
| Exact step (frozen probe) | 6 | 0.591 (4 seeds: 0.591 to 0.616) | 0.673 (0.673 to 0.708) | 0.700 |
| Names plus exact, mean direction (frozen probe) | 6 | 0.602 | 0.677 | 0.699 |
| Chunk of 5 steps, first step (frozen probe) | 6 | 0.607 | 0.697 (its place 3 steps ahead: 0.722) | 0.699 |
| Names, most likely class (end to end, old rules) | 03, 05, 06 | 0.654 | 0.707 | 0.499 |
| Names, mean direction (end to end, old rules) | 03, 05, 06 | 0.667 | 0.744 | 0.499 |
| Exact step (end to end, old rules) | 03, 05, 06 | 0.681 | 0.790 | 0.772 |
| Chunk of 5, first step (end to end, old rules) | 03, 05, 06 | 0.670 | 0.801 | 0.760 |
| Exact step, final candidate (cleaned rows, last weights) | 03, 05, 06 | 0.671 | 0.788 | 0.770 |
| Exact step, final candidate | all 6 | 0.678 | 0.786 (worst sheet 0.735) | 0.760 |

[measured; frozen probes in `04_runs/2026-11-09_frozen_feature_probes/designs.json` and `seeds.json`; end-to-end runs in
their fold files `fold_without_path_XX.json` (a run trained in parts keeps only its last part in `metrics.json`); "old
rules" means no audit cleaning and the best-validation weights (the network as it was at the epoch with the lowest
validation loss); twist sign means are the means of the per-sheet values]

- **Against the names as they would run (most likely class)** the exact step leads by 4.3 points of step within 45 and
  7.8 of path within 45 over four probe seeds, and end to end by 2.6 and 8.3 points; the seed range alone is about 3
  points. [calculated]
- **Against the names read as a mean direction** the lead shrinks: 0.9 and 2.2 points in the probes as four-seed means
  (exact 0.603 and 0.691 against 0.594 and 0.669; at the default seed of the table the mean direction even leads), which
  is inside the seed range (mean-direction seeds: step 0.578 to 0.605, path 0.654 to 0.682), and 1.3 and 4.6 points end
  to end. The clear gains of the exact step are the path direction end to end and the twist (sign right on 0.77 against
  0.50 of the twisting rows). [calculated; measured]
- **Where the exact step loses: backing up.** It predicts a straight-back move on 0.1 % of the pictures against 3.9 %
  demonstrated (earlier exact run). Weighting its loss by move name (frozen probe, two seeds) gained 11 to 15 points on
  backward moves but lost 6 on all other moves, 3.6 points of step within 45 and 4.5 of path within 45. So the exact loss
  stays unweighted and the backup covers backing up. Whether that trade is bad on the robot is a judgement, because both
  sides are scored on the demonstrators' pictures. [measured; judgement]
- **Both at once** adds nothing when the names part is read as the most likely class (step within 45 0.565), but read as
  a mean direction it matches the better of the two in the probes. That is the backup. [measured]
- **The chunk** (where the tip is after each of the next 5 steps) is a near-tie on direction with a worse stop: end to
  end, path within 45 +3.5, +1.8 and -2.1 points against the exact run on sheets 03, 05 and 06 (mean +1.1), while it
  stopped right in 31 of 100 episodes at 0.5 against 42, and in none of the 25 on sheet 06. Against the final candidate
  on the same pictures its path lead is 4.0 and 3.9 points on sheets 03 and 05 [measured; judge 3], but on sheet 06 the
  final candidate leads (0.810 against 0.770), so over the three sheets the chunk leads by about 1.2 points (0.801
  against 0.788; different test rows, not paired) [calculated from the fold files]. Executing chunks without looking
  lost under noise (Liu et al. 2025), so a chunk model would still execute one step per picture. [literature]

### 4.2 Small motions and turns

- The 1,765 small_motion rows left after the cleaning are mostly twists in place: 1,687 twist by 3 degrees or more
  (median 5.98 degrees). In the exact target they are simply steps with a short tip move and a twist, so no special rule
  is needed. [measured; judgement]
- For a names model the three ways of handling them (the next move, dropped, a class of their own) scored 0.577, 0.560
  and 0.560 in the probes, all inside the seed range. The next-move rule copies a move measured in a later picture's
  frame (another class in its own frame for 16.0 % of the relabelled rows), and would push 1 cm where the demonstrator
  twisted in place. [measured]

### 4.3 Twist, lean and hand

- **Twist (joint 4, the forearm, and with it the camera).** The forearm twists on most steps: 66.1 % of the non-stop
  rows twist by 2 degrees or more (median 4.04 degrees per step, 99th percentile 11.6), and the camera is within 5
  degrees of straight in only 8.2 % of the pictures. So the twist is predicted, not set by a rule. [measured; judgement]
- **Twist size.** The typical predicted twist is smaller than the demonstrated one (median 1.28 against 4.92 degrees on
  sheet 01, 3.43 against 5.27 on sheet 02), mostly where the model is unsure which way to turn [measured; targets
  draft]. But where it does twist, the demonstrations twisted 0.65, 0.55, 0.94, 0.96 and 0.66 times the predicted amount
  on sheets 01 to 05 and 1.24 times on sheet 06 (least-squares gains of the demonstrated on the predicted twist); where
  it predicts 3 degrees or more, the sign is right on 0.72 to 0.96 of the rows [measured; judge 2]. So scaling the twist
  up has no support (judge 2 reads these gains as a risk of twisting too far), the gain of 0.97 quoted from the earlier
  run does not hold per sheet, and gain 1 is the simplest choice, not a fitted one. [judgement]
- **Twist adds up.** Summed open loop over an episode, the final candidate's twist would pass 115 degrees in 27 of 65
  episodes of sheets 01 and 02, against 3 for the demonstrated path. On the robot the pictures feed back, but the joint-4
  clamp will bind, so its behaviour must be written down and rehearsed on sheets 04 and 06. [measured; judgement]
- **Rod lean (joint 5, the wrist).** It is trained (sign right on 0.79 to 0.91 of the rows that leaned 1 degree or
  more) but not executed: no executor for it exists, and executing it would need a new joint-5 search, untested.
  [measured; judgement]
- **Hand (joint 6).** It changes by 3 degrees or more on only 26 of 17,773 non-stop rows, so it is not executed. Its
  angle differs visibly between start kinds, a shortcut the model could learn. [measured]

### 4.4 Step length

| | Moving rows | Twists in place | Stop rows |
|---|---|---|---|
| Predicted step, median (earlier exact run, sheets 03, 05, 06) | 0.60 cm (0.57, 0.61, 0.65) | 0.48 cm | 0.30 cm |
| Demonstrated step, median | 1.09 cm | 0.20 cm | 0 |

[measured; the saved models re-run on their own test pictures]

- The step length is a recording rule: a waypoint was cut after 1 cm of tip travel or 5 degrees on any joint. [read in
  the recording tool's code and notes by the audit; motion_statistics-3]
- The predicted length shrinks most where the direction is unsure (0.70 cm when the direction is within 15 degrees, 0.48
  cm beyond 90 degrees); the average of moves in scattered directions is a short vector. The final candidate shrinks less
  on the straight sheet 01 (1.00 against 1.12 cm) but as much on the arc, sheet 02 (0.68 against 1.11 cm), and within 2
  cm of the goal it predicts 0.22 and 0.36 cm. Only the angle of the prediction is saved in the run files, so the offline
  numbers say nothing else about step length. [measured; judgement]
- **So the robot steps a fixed 1.0 cm**, close to the demonstrated median of 1.09 cm, which keeps about the
  demonstrated pace and cannot stall [judgement]. A published wrist-camera pushing set-up also executed a predicted
  direction with fixed 1-cm moves (DMD) [literature; the checker's corrected form].
- Time per step (plain steps, measured): a 1-cm tip move without twist 0.95 s; with a twist of 3 to 5 degrees 1.19 s; 5
  to 8 degrees 1.35 s; a twist in place of 5 to 7 degrees 1.12 s. A 1.5-cm step would save about 20 % per cm on straight
  stretches but jumps further between pictures than most training steps; in simulation, executing policies faster than
  demonstrated cost success (SAIL: Square 0.83 to 0.55). [measured; calculated; literature]
- **Backup for the length**, only if fixed 1-cm steps lose the puck at sharp bends on the robot: the predicted length
  rescaled so that its median matches the demonstrated 1.09 cm, clamped to 0.5 to 1.5 cm. The gain must be refitted on
  the final candidate (1.12 on sheet 01 and 1.64 on sheet 02; the 1.81 in the targets draft belongs to the earlier run).
  [calculated; judgement]

### 4.5 The stop

**Designs tried** (fixed threshold 0.5 with one picture; shares are means over the held-out sheets; "best" is the best
threshold chosen with hindsight on each held-out sheet, an optimistic ceiling) [measured]:

| Stop design | Right / early / never at 0.5 | Best threshold, one picture: right / early |
|---|---|---|
| Frozen, names, stop as a ninth class | 0.258 / 0.112 / 0.599 | 0.389 / 0.215 |
| Frozen, names, own stop output on the stop picture | 0.321 / 0.209 / 0.436 | 0.405 / 0.233 |
| Frozen, names, own stop output on every picture within 2 cm | 0.138 / 0.658 / 0.090 | 0.340 / 0.276 |
| Frozen, exact, own stop on the stop picture (4 seeds) | right 0.289 to 0.332, early 0.243 to 0.390 | 0.354 to 0.401 / 0.210 to 0.306 |
| Frozen, exact, predicted distance to the goal | 0.110 / 0.013 / 0.853 | 0.340 / 0.192 |
| End to end, exact, old rules (03, 05, 06) | 0.363 / 0.254 / 0.368 | 0.429 / 0.225 |
| End to end, names, stop as a class (03, 05, 06) | 0.154 / 0.106 / 0.733 | 0.371 / 0.292 |
| Final candidate (six sheets) | 0.460 / 0.199 / 0.262 (pooled over the 209 episodes: 101 / 41 / 50) | one threshold for all six sheets, chosen with hindsight, pooled: 104 / 32 of 209 (0.498 / 0.153; threshold 0.6) |

**Stop rules on the final candidate**, all 209 held-out episodes ("right": the first stop within 2 cm of where the
demonstration stopped; "early": more than 3 cm before it; "2 to 3 cm": between) [measured; judge 2]:

| Rule | Right | Early | 2 to 3 cm | Never |
|---|---:|---:|---:|---:|
| Fixed 0.5, one picture | 101 | 41 | 17 | 50 |
| **Fixed 0.5, two pictures in a row (proposed)** | **102** | **25** | **13** | **69** |
| Tuned on validation episodes, one picture (thresholds 0.3 to 0.8) | 105 | 33 | 16 | 55 |
| Tuned on validation episodes, two in a row (the recipe's former rule; thresholds 0.05 to 0.4) | 98 | 58 | 20 | 33 |
| Chosen on the other five held-out sheets (right minus early) | 86 | 24 | 13 | 86 |

- **No rule changes the right count much:** the rules in the table stop right in 86 to 105 of 209 recorded episodes,
  most of them in 98 to 105; they differ mainly in how many episodes end early against never. The Wilson interval of
  102 of 209 alone is 0.42 to 0.56. [measured; calculated]
- **Reject tuning on validation episodes.** Its thresholds fell to 0.1 or lower on four of six folds and it gave the most
  early stops (58). Validation episodes are near-twins of training episodes (section 3.6). [measured; judgement]
- **Fix 0.5.** It needs no tuning set, so the same rule serves the fold models and the exam model trained on all six
  sheets, which has no held-out sheet to tune on. The right count is flat from 0.4 to 0.7 with one picture and from 0.25
  to 0.5 with two, so a small shift of the exam model's probability scale costs little. But the scale moved between
  probe seeds (median stop probability on the stop pictures 0.30 to 0.65), so check the exam model on pictures it never
  trained on, such as new printed validation tracks. [measured; judgement]
- **One picture or two.** Pooled, two in a row keeps the right count (102 against 101) and removes 16 early stops, which
  become never-stops. Per sheet the change is mixed (sheet 02 lost 5 right stops, sheet 01 gained 4), and offline the
  second picture is the next demonstrated one, a median 1.05 cm further on, not a deliberate second look. So two in a
  row is the default and one picture the robot A/B alternative. [measured; judgement]
- **Where it fails.** On the dashed loop (sheet 06) it stopped right in 3, early in 1 and never in 21 of 25 episodes
  (two in a row); its stop output caught only 0.129 of that sheet's stop pictures. At (0.5, one picture), 34 of the 89
  episodes whose line end lies 5 cm or more from the paper edge never stopped, against 16 of 120 nearer ends; sheet and
  edge distance are tied in these data, so the cause cannot be decided. At (0.5, one picture) the early stops come a
  median 3.7 to 4.7 cm before the goal on sheets 01 to 04 but 14.6 cm before it on the loop, sheet 05, where critics who
  viewed the pictures saw the line turning along the paper edge. [measured; judgement]
- **Tried, did not help.** Extra loss weight on far pictures without a line ahead lowered the stop probability on real
  stop pictures about as much (frozen probe; one-picture stop worse). A predicted distance to the goal had the fewest
  early stops but few right stops (tuned, two in a row: right 0.148). Earlier pictures as extra input gave 0.254, 0.278
  and 0.186 right against 0.258 for one picture. [measured]
- **Stop options not yet run end to end** (each a frozen-probe or three-sheet hint, not a result) [measured; targets
  draft 2.6 and its evidence critic]:
  - *A stop output trained on every picture within 2 cm of the goal* (the near-goal window). At 0.5 it fires far too
    early (exact probe: 0.176 right, 0.512 early); with thresholds tuned on validation episodes, two in a row, the exact
    probe gave 0.318 right and 0.276 early against 0.221 and 0.414 for the stop-picture target. It needs thresholds of
    0.7 to 0.9 and is then about as good as the stop picture.
  - *The combined rule*: stop only when the stop probability passes and the predicted step is short (0.2 to 0.8 cm). On
    the earlier exact run it raised the best-threshold ceiling from 0.439 right and 0.288 early to 0.494 and 0.177 (two
    in a row), and with thresholds from the other sheets from 0.324 and 0.422 to 0.350 and 0.329, a gain that came from
    sheet 06 alone. It needs no retraining, only a re-run of the saved models to read the step length.
  - *Extra weight on the far pictures where a first model wrongly fired* (mined hard negatives): started, not finished.

  The combined rule and the near-goal output at a high threshold are robot A/B candidates (2.5); an end-to-end
  near-goal run is an optional laptop item (9.1). [judgement]
- **Stop picture or at-goal.** The recordings also carry an at-goal label (`is_at_goal`, added on 7.11: the tip within
  2 cm of where the demonstration stopped) on 808 rows (4.36 %, 3.85 times the stop rows). It is not the stop target
  because it is noisy: in 30 of 210 episodes the tip enters the 2-cm zone and leaves it again (124 at-goal rows come
  before that, 99 of them inside correction replays), 111 of the 808 lie more than 10 rows before the stop, and inside
  the zone the demonstrators keep moving (523 non-stop rows within 2 cm of the goal after the cleaning). The near-goal
  window above is the cleaned form of this idea. [measured; audit class_balance-11, notes_context-4, 11.2 point 13]
- **Fragile under changed pictures.** The training-settings exam-risk critic re-ran the saved fold models of sheets 01
  and 02 on nine changed versions of their pictures (appendix A.7). The direction stayed within 2.0 points of the clean
  pictures under every change, but the stop moved: on sheet 02 at 0.5 with one picture, clean 24 right, 2 early and 18
  never of 45; half brightness, darker than any training picture, 21, 10 and 11; 8-px shifts 25, 4 and 14 or 13; sheet
  01 kept 2 to 4 early stops of 20 under every change. No change was scored under the proposed rule (0.5, two in a row),
  and four fold models were not tested. [measured by that critic]
- **What "right" means.** Offline, "right" uses the tip's distance to where the demonstrator stopped; the exam judges
  the puck at the line's end. The robot test must score the puck (section 7.4). [judgement]

### 4.6 How the robot executes the output

The model still sees only the picture. The executor turns its output into joint moves using the robot's joint angles,
as every move command does; whether that is allowed, together with the fixed rules below, is question 1 for the
professor (section 10.1).

- **The model mode does not exist yet.** The code-reading audit lists nine missing pieces in the recording tool: a model
  mode; a *decider* (the part of the tool that picks the next step from the latest picture) that never hands the
  robot's state to the model; a fixed-wrist "model step" action; clamps on every output; guards for the table in use; a
  refusal counter with a fallback; the step budget switched on; a separate raw folder; columns for the model's raw and
  clamped outputs; and checking for the Esc key while the model decides. One more piece: after a hand take-over
  ("reguide"), only the exact-replay decider reads the guided waypoints today, so a model decider must hand over to it.
  None of the ranking matters until these exist and pass the checks below. No time estimate exists. [read in the code by
  the audit and a critic]
- **Checks before the robot** [judgement; improvement exam-risk critic]. The tool's `--simulated` mode draws a plain grey
  page, a dark line and a red disc (`robot_link.py` lines 428 to 444), so it checks the software path, the signs of the
  outputs, the clamps and the guards, not the model's skill [read in the code]. Two more checks need no robot. A replay
  check: run the new decider on the logged raw pictures of one held-out sheet; its angles and stop probabilities must
  equal the saved predictions, which catches a decoding or resizing mismatch with `training_dataset.py` (a first
  sub-step: load `checkpoints/without_path_01.pt` and reproduce the sheet-01 fold file). A fresh-picture check: each
  decision must use a picture taken after the last move; the audit found 863 rows with a fresh-picture wait over 0.2 s,
  where stale pictures cluster, and lists camera latency as a robot test (audit 11.1 D5, 11.3 item 23).
- **Wrist and hand.** Keeping joints 5 and 6 and solving joints 1 to 3 for the tip reproduced the next demonstrated view
  within 0.5 to 0.7 px (90th percentile) when the wrist took its demonstrated value; held at one value for a whole run,
  the picture centre moved more than 20 px in 23.3 % of the rows. So before each run set joints 5 and 6 to the median of
  their start values over the demonstrations (to be computed from `frames.csv` before session 1), hold them there, and
  consider a wider shift augmentation for the exam model (section 6.2). [calculated by the audit; judgement]
- **Not the keyboard or gamepad paths.** They re-solve towards the least rod lean and would move the picture content a
  median 48 to 56 px, views the model never saw. [calculated by the audit]
- **Joint 4 limit.** The demonstrations reached -119.35 and +118.92 degrees against the arm model's limit of 119.75; 27
  of 45 sheet-04 demonstrations passed 110 degrees. Clamp at 115 degrees; at the clamp hold the twist, keep stepping in
  the predicted direction, and log it. [measured; judgement]
- **Workspace.** The configured box would refuse 7.7 % of the demonstrated tip positions, so take the workspace from the
  sheet in use. [measured]
- **Step budget.** It is switched off today (`is_step_budget_enforced` is False). The configured 2 steps per cm would
  have cut off more than half of the standard-start demonstrations, and sheets not listed in `config.py` fall back to 80
  steps, which means every unseen track. Use about 4 steps per cm of line (no demonstration needed more than 3.32), or a
  cap of about 250 steps when the length is unknown (the longest demonstration had 226). [measured; judgement]
- **Speed.** A step takes 1.22 s median (the move 1.03 s, the fresh picture 0.11 s, the decision 0.06 s). Keep the
  recorded 20 % speed cap until a paired robot test shows that a faster cap is safe for the puck. [measured; judgement]

---

## 5. Model options compared on our data

### 5.1 Frozen networks

Step within 45 per held-out sheet; a frozen backbone with a head of one hidden layer of 256 on a 4 x 4 grid of its last
feature map, 224 x 224 pictures, names head [measured; `backbones.json`, `baselines.json`]:

| Frozen backbone | 01 | 02 | 03 | 04 | 05 | 06 | Mean | Beats always forward on | Path within 45, mean (worst) |
|---|---|---|---|---|---|---|---|---|---|
| Always forward (no picture) | 0.747 | 0.565 | 0.522 | 0.485 | 0.509 | 0.402 | 0.538 | | 0.604 (0.433) |
| ResNet-18, ImageNet | 0.756 | 0.530 | 0.565 | 0.560 | 0.583 | 0.467 | 0.577 | 5 of 6 | 0.633 (0.526) |
| MobileNetV3-Large, ImageNet | 0.708 | 0.535 | 0.587 | 0.526 | 0.567 | 0.535 | 0.576 | 4 of 6 | 0.640 (0.556) |
| EfficientNet-B0, ImageNet | 0.685 | 0.513 | 0.514 | 0.532 | 0.588 | 0.447 | 0.546 | 3 of 6 | 0.606 (0.478) |
| DINOv2-small | 0.700 | 0.493 | 0.530 | 0.557 | 0.574 | 0.417 | 0.545 | 4 of 6 | 0.598 (0.459) |
| CLIP ViT-B/32 | 0.714 | 0.477 | 0.587 | 0.552 | 0.513 | 0.438 | 0.547 | 4 of 6 | 0.606 (0.480) |
| ResNet-18, random weights (control) | 0.647 | 0.439 | 0.521 | 0.495 | 0.469 | 0.372 | 0.490 | 1 of 6 | 0.522 (0.409) |

- Frozen features are not enough: the pretrained backbones are only 0.7 to 3.9 points above the picture-free rule on the
  mean and lose to it on one to three sheets, mostly the straight line and the arc. [measured]
- Pretraining does help a frozen network: ImageNet ResNet-18 beat its random-weight copy on all six sheets, by 4.4 to
  11.4 points. [measured]
- No clear winner among the pretrained ones: ResNet-18 and MobileNetV3 lead the others by about the seed range (3.1
  points); MobileNetV3 has the best worst sheet. [measured]

### 5.2 Networks trained end to end

Held-out sheets 03, 05 and 06 (the runs before the final candidate were trained on these three first); 160 x 120
pictures with the augmentation of section 6.2 [measured; the runs' fold files; frozen reference from `backbones.json`
and `designs.json`]:

| Model (rules) | Output | Step within 45: 03 / 05 / 06 (mean) | Path within 45, mean | Twist sign: 03 / 05 / 06 | Stop at 0.5: right / early / never of 100 |
|---|---|---|---|---|---|
| Always forward | | 0.522 / 0.509 / 0.402 (0.478) | 0.527 | | |
| Frozen ResNet-18 probe | names | 0.565 / 0.583 / 0.467 (0.538) | 0.573 | 0.672 / 0.442 / 0.379 | |
| Small network from scratch (old) | names | 0.635 / 0.635 / 0.623 (0.631) | 0.683 (mean direction 0.746) | 0.718 / 0.405 / 0.371 | 19 / 8 / 71 |
| Small network, line-and-puck maps (old) | names | 0.630 / 0.625 / 0.623 (0.626) | 0.683 | 0.659 / 0.399 / 0.398 | 18 / 7 / 74 |
| ResNet-18 (old) | names | 0.663 / 0.659 / 0.640 (0.654) | 0.707 (mean direction 0.744) | 0.703 / 0.372 / 0.422 | 18 / 11 / 70 |
| ResNet-18, half the pictures mirrored (old) | names | 0.670 / 0.659 / 0.599 (0.643) | 0.711 (mean direction 0.754) | 0.709 / 0.370 / 0.470 | 41 / 12 / 44 |
| ResNet-18 (old) | exact step plus stop | 0.700 / 0.667 / 0.675 (0.681) | 0.790 | 0.845 / 0.669 / 0.802 | 42 / 25 / 31 |
| ResNet-18 (old) | chunk of 5 plus stop | 0.692 / 0.681 / 0.638 (0.670) | 0.801 | 0.839 / 0.662 / 0.778 | 31 / 24 / 43 |
| ResNet-18, final candidate (cleaned, last weights) | exact step plus stop | 0.680 / 0.649 / 0.684 (0.671) | 0.788 | 0.848 / 0.677 / 0.786 | 47 / 22 / 27 of 99 |

- **Training on our pictures is the largest gain measured.** But the end-to-end runs also changed the augmentation, the
  picture size and the grid, so the gain is not the fine-tuning alone; the pending run without augmentation (section
  11.3) separates part of it. [measured; judgement]
- **ImageNet ResNet-18 against the small network** (same names head): +2.9, +2.5 and +1.7 points of step within 45 when
  the most likely class is executed, but +1.9, +1.9 and -4.4 points of path within 45 when both are read as a mean
  direction. The two runs also differed in learning rate, epochs (8 against 16) and which weights were kept. So the
  backbone is not settled by our data. [measured; judgement]
- **The hand-made line-and-puck maps gave no offline gain** (-1.0 to 0.0 points of step within 45 on sheets 03, 05 and
  06). Their aim, coping with new light and tables, is hardly tested by these six sheets. [measured; judgement]
- **Fine-tuned MobileNetV3 and EfficientNet-B0** are configured (`mobilenetv3_finetune_classes`,
  `efficientnet_b0_finetune_classes`) but were not run: frozen, no backbone did better than ResNet-18, and speed is not a
  constraint (section 5.5). [judgement]

### 5.3 ACT, diffusion policies and large vision transformers (literature only, not run)

- **ACT** (action chunking with transformers), as published: about 80 million parameters, four cameras plus the joint
  positions, 50 decisions per second, about 5 hours of training per task on one GPU (Zhao et al. 2023). Its gain from
  chunking (1 % to 44 % success) was at 50 decisions per second in simulation; a track of ours takes about 89 decisions.
  The original code always feeds joint positions, which our rule forbids; LeRobot's version can run on pictures only.
  LeRobot's 52-million-parameter ACT took 182 ms per decision on a 2021 MacBook M1 processor, and LeRobot's own guide
  advises against training on a processor only. Its useful idea for us, predicting several steps, was tested inside our
  ResNet-18 (the chunk target, section 4.1). [literature; calculated for the 89 decisions]
- **Diffusion policy.** Strong on the closest published task: real Push-T reached 95 % success with 136 demonstrations,
  an end-to-end ResNet-18, two fixed cameras and 10 decisions per second (Chi et al. 2023), but those policies also got
  the end-effector position. With the same network and chunks, plain regression matched flow policies on most of 28
  benchmarks (Pan et al. 2026), and a diffusion policy trained on 30 demonstrations replayed memorised trajectories even
  for unrelated pictures (He et al. 2026). Training recipes use 75 to 3,000 passes over the data (about 500 in LeRobot's
  Push-T recipe); at 4 to 8 minutes per pass of our ResNet-18 encoder, 500 passes would take about 33 to 67 hours per
  model and held-out sheet before the denoising network. Published processor times are over 5,000 ms per decision for
  the variants measured; a right-sized head might be much faster, but that is unchecked. [literature; calculated]
- **DINOv2 or CLIP fine-tuned.** Fully fine-tuning a large vision transformer on few demonstrations collapsed in one
  study (VC-1, Adroit 59.3 to 15.9) and hurt in another (MVP 44.0 frozen against 31.1 fine-tuned), while CLIP ViT-B/16
  fine-tuned with a ten times lower learning rate was the best encoder on Square (0.98). A ViT-B/16 needs 17.56 G
  multiply-adds per 224 x 224 picture against 1.81 G for ResNet-18. Here, frozen, both scored below ResNet-18 and were
  3.2 to 3.5 times slower at 224 x 224. [literature; measured]
- **Judgement:** none of these fits the laptop now, and nothing shows that they handle unseen tracks better. A fair
  comparison would need the GPU pool, the same encoder and a held-out sheet.

### 5.4 One picture or a short picture history

Frozen ResNet-18 grid probe, names head, step within 45 per held-out sheet [measured; `history.json`, `baselines.json`]:

| Input | 01 | 02 | 03 | 04 | 05 | 06 | Mean | Worst sheet | Path within 45, mean | Stopped right at 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| One picture | 0.756 | 0.530 | 0.565 | 0.560 | 0.583 | 0.467 | 0.577 | 0.467 | 0.633 | 0.258 |
| Plus the previous picture | 0.758 | 0.532 | 0.548 | 0.583 | 0.594 | 0.444 | 0.576 | 0.444 | 0.623 | 0.254 |
| Plus the change since the previous picture | 0.770 | 0.558 | 0.597 | 0.604 | 0.636 | 0.454 | 0.603 | 0.454 | 0.640 | 0.278 |
| Plus pictures 2 and 4 back | 0.772 | 0.517 | 0.612 | 0.573 | 0.554 | 0.488 | 0.586 | 0.488 | 0.625 | 0.186 |
| Rule "repeat the previous move" (reference only) | 0.796 | 0.648 | 0.711 | 0.692 | 0.785 | 0.783 | 0.736 | 0.648 | 0.648 | |

- Only the change since the previous picture gained (+2.6 points on the mean), and its worst sheet fell; the worst path
  sheet fell from 0.526 to 0.498. In our stop-and-go loop both pictures are taken at rest, so their difference mostly
  shows the last 1-cm step, and the last move alone predicts the next one well (0.736). A history input can therefore earn
  offline points by copying and then fail where the line bends. [measured; judgement]
- In the literature, offline gains from history have hidden closed-loop losses (Wen et al. 2020: lower test loss, less
  reward), but with images and 200 demonstrations a recurrent policy also beat single-step cloning (robomimic). Those
  ran at 10 to 20 decisions per second, where two frames show motion; ours are taken at rest. [literature; judgement]
- **Averaging the output over the last pictures** gains nothing worth its risk. Offline, averaging the predicted
  direction over 1, 2, 3 and 5 pictures gave path within 45 0.772, 0.775, 0.767 and 0.731 for the final candidate
  (14,294 pictures pooled) and 0.617, 0.631, 0.635 and 0.620 for the frozen ResNet-18 probe [measured]. There is no
  back-and-forth to smooth offline: consecutive predictions flip the twist sign on 0.5 to 1.2 % of picture pairs against
  3.8 to 16.7 % in the demonstrations, and reverse the direction on 0 to 0.7 % against 4.8 to 10.0 % [measured by the
  targets exam-risk critic; earlier exact model and final candidate]. In the literature, averaging the outputs of
  several pictures did poorly and oscillated on a real robot (Black et al. 2025) and lost under noise in closed loop
  (BID: 18.4 against 29.5) [literature]. So no averaging; count back-and-forth moves on the robot instead [judgement].
- **Keep one picture.** Test a history input only after a copy test (give it the same picture twice and count how often
  it repeats the last move), and then only on the robot. The previous action as an input stays ruled out until the
  professor allows it (audit 11.2 point 16). One picture may not be enough at a crossing (which branch) or in a dashed
  gap (no line in view); the offline numbers cannot show it, so phase 1a on sheets 05 and 06 watches for wrong
  branches and lost lines (7.4). [judgement]

### 5.5 Size and speed: the laptop and the robot loop

Laptop: AMD Ryzen AI 5 PRO 340, 6 cores, no NVIDIA GPU, PyTorch 2.15 on the processor. All timings were taken while other
jobs shared the processor, so they are upper bounds [measured].

| Model | Parameters | Multiply-adds per 160 x 120 picture | Time per picture | Share of a 1.22 s robot step | Training per held-out sheet | Weights file |
|---|---|---|---|---|---|---|
| Small network from scratch | 1.61 M | 0.19 G | 2.5 to 13.5 ms | 0.2 to 1.1 % | 15.8 to 24.7 min (16 epochs) | 6.5 MB |
| ResNet-18 trained end to end | 12.75 M | 0.72 G | 16.7 to 37.4 ms (whole model; the final candidate 37.4) | 1.4 to 3.1 % | 16.2 to 68.2 min (8 epochs; the long ones while other training ran; the mirror rerun, with no other training running, 19.2 to 21.4) | 51.1 MB |
| MobileNetV3 / EfficientNet-B0 backbone (frozen) | 4.2 / 4.0 M | not counted | 15.0 / 22.7 ms | 1.2 / 1.9 % | not run end to end | not saved |
| DINOv2-small / CLIP ViT-B/32 backbone at 224 x 224 | 21.6 / 87.5 M | not counted | 82.9 / 91.0 ms | 6.8 / 7.4 % | not run end to end | not saved |
| ACT, LeRobot (literature, M1 processor) | 52 M | not counted | 182 ms | 14.9 % | about 5 h per task on a GPU as published | not run |
| Diffusion policy (literature, other processors) | not counted | not counted | over 5,000 ms for the variants measured | over 400 % | hundreds of passes over the data | not run |

[measured; literature; shares calculated; "not counted": nobody counted it for this analysis]

- **Every model tested here fits the robot loop:** the time to finish a track is set by the arm's moves, not by the
  model, so the model is chosen on finishing (direction and stop), not on speed. [judgement]
- One ResNet-18 configuration on all six held-out sheets takes about 2 to 7 hours: about 2 to 2.5 hours when no other
  training runs (the mirror rerun took 19.2 to 21.4 minutes per fold), up to about 7 hours when other training shares
  the processor (folds of up to 68.2 minutes); the final candidate took 223 minutes while sharing. The exam model on all six
  sheets should take under about 1.5 hours per seed. The GPU pool (the university's graphics-card computers) is still
  unconfirmed, so the plan must not depend on it. [measured; calculated]
- Not yet timed: the whole decider on the robot (decoding the 640 x 480 JPEG, resizing it exactly as
  `training_dataset.py` does, the model and the joint solve). Only the fixed-wrist joint solve is measured: about 0.005 s
  per step. [measured by the audit; the rest not timed]
- Only one program may be connected to the robot, so the model must run inside the recording tool's decider, which is
  called once per fresh picture. [project rule; read in the code by the audit]

---

## 6. Training settings

### 6.1 How large a difference must be

- **Seeds.** On frozen ResNet-18 features the same setting trained with four seeds gave six-sheet means of 0.546 to
  0.577 (names head) and 0.591 to 0.616 (exact head); three more seeds of the names head gave 0.541 to 0.574 (3.3
  points). On single sheets the seed moved step within 45 by up to 9.1 points (names, sheet 01) and 9.8 (exact, sheet
  06), path within 45 by up to 16.0, the worst sheet by 12.4, and the share stopped right on one sheet by up to 32
  points. [measured]
- **Seed against seed on identical test pictures**, for the most lopsided seed pairs a sign test on the episode counts
  gave p of 0.05 to 0.07 although only the seed differed, so an episode-level test overstates certainty. [measured]
- **End-to-end runs have one seed each.** Their seed spread is unknown until the second seed of the final candidate is
  in (expected by about 05:30 on 10.11). [measured]
- **Consequence:** with one run each, a difference under about 3.5 points of six-sheet step within 45, or under about 7
  points of stopped right, is not evidence. [calculated; judgement]

### 6.2 Augmentation

Augmentation changes training pictures (light, blur, a small shift) so that the model copes with changes it will meet,
without changing what the right move is. The block, the gripper fingers, the puck and the line are all in the picture,
and moves are named in the picture's own frame, so a whole-picture shift keeps the label, a rotation turns every
direction, and a mirror swaps left and right. Settings of the final candidate at 160 x 120, measured ranges from the
audit:

| Change | Keeps the label? | Measured in our data | Now | Proposal |
|---|---|---|---|---|
| Whole-picture shift | yes | the camera view stepped once by about 15 px at 640 x 480 (about 3.75 px at 160 x 120) | up to 4 px | keep; test 8 px (about 5 %, the papers' typical size) after checking that a large shift does not cut off the fingers or the puck; for the exam model a wider shift (8 to 10 px) if the wrist is held at fixed values. The fold models of sheets 01 and 02 already kept their direction under 8-px shifts (within 2.0 points), with sheet 02's early stops at 4 instead of 2 of 45 (A.7) |
| Zoom (crop with a scale change) | no: shows step sizes that never occur | | off | off |
| Rotation in the picture plane | no: turns every direction; 18.8 % of moves lie within 5 degrees of a class border | roll under 0.5 degrees | off | off |
| Left-right mirror with swapped labels | the labels can be swapped, but the picture cannot occur: the bottom right is darker than its mirror in 90.7 % of the pictures | | off | off (see the mirror test below) |
| Brightness | yes | 0.74 to 1.12 of the median; natural light down to 0.674 | x0.65 to x1.35 | keep |
| Contrast | yes | 0.72 to 1.26 | x0.7 to x1.3 | keep |
| Gamma | yes | not measured | exponent 0.78 to 1.28 | keep, as a margin for an unknown exam room |
| Saturation | yes | not measured | up to 25 % either way | keep |
| Colour cast (gain per colour channel) | yes, while the puck stays redder than the paper | white-sheet blue/red ratio 0.91 to 1.28 of its median | up to 8 % | raise to 13 % (low priority) |
| Hue rotation | risky for the red puck (hue 137 to 163 in shade) | | off | off |
| Gaussian blur | yes | at most about 0.5 px at 640 x 480, invisible at 160 x 120 | 15 % of pictures, sigma 0.3 to 0.8 px | lower the top to 0.5 px (low priority) |
| Sensor noise | yes | about 0.11 grey levels at 160 x 120 | up to 0.03 of full scale | lower to 0.008 (harmless either way) |
| Cut-outs (erased rectangles) | no when they hide the puck or the line | | off | off; drop hand pictures instead |

[measured; calculated; proposals are judgements from the training-settings draft and its critics]

- **Literature.** Small shifts are the best-supported augmentation for imitation from pictures (robomimic real Can: 22 of
  30 with shifts against 8 of 30 without). A bundle of colour, blur and noise changes moved driving in an unseen town
  from 0 to 64 % (Codevilla et al. 2018), which guards against new light and tables, not new curves. Flips and rotations
  added to crops lowered scores in RAD. No paper measured mirrored, label-swapped pictures on robot success. [literature]
- **Mirror test so far (names head, old rules).** Step within 45 0.670 against 0.663 (sheet 03), 0.659 against 0.659
  (05) and 0.599 against 0.640 (06, -4.1 points); stopped right at 0.5 in 41 against 18 of 100 episodes. The comparison
  is confounded: the plain run kept its best-validation weights from epochs 7, 5 and 7, the mirrored run from epoch 8 on
  all three sheets; one seed; names head only. The rerun of 23:02 to 00:04 reproduced every mirrored score. Mirroring
  stays off. [measured; judgement]

### 6.3 Class weights and sampling

- **Names head (the backup's).** Square-root class weights (0.59 for forward up to 2.06 for back and 2.86 for stop).
  Without weights the frozen probe gained 1.6 points of step within 45 (inside the noise) but backed up far worse and
  stopped right 8.2 points less often; full inverse weights lost 6.8 points of step within 45, twice the largest seed
  difference. Keep square root. [measured]
- **Exact head (the first set-up's).** No direction weights: weighting by move name traded backward moves against all
  other moves at a net loss (section 4.1). [measured]
- **Stop output.** Positive weight 10 (68.6 non-stop rows per stop row after the cleaning); no other weight was tested.
  With a fixed threshold, this weight sets where 0.5 falls, so do not change it without re-checking the stop. [measured;
  judgement]
- **Sampling unit.** Uniform over pictures in every run, which gives sheet 04 up to 44.2 % of a fold's training rows.
  Sheet-balanced weights are untested; try them in the frozen probes first. [measured; judgement]
- **When correction data arrive:** retrain from scratch on all data rather than fine-tune on the corrections alone
  (fine-tuning on corrections lost 30 and 15.6 points in CR-DAgger), and compare a correction share of 10 to 25 % of each
  batch with plain mixing. [literature; judgement]

### 6.4 Score weighting

Keep scores 1 to 3 at equal weight (section 3.8). There is no measured basis for weighting score 1 down; if a score
weight is tried again, judge it on the stop, the only place where any difference appeared. [measured; judgement]

### 6.5 Picture size

160 x 120 colour pictures, the camera's 4:3 shape. At this size the 1-cm line is about 5.5 px wide and the puck about 15
px, and a sideways move shifts the picture content by about 5.2 px per cm. ResNet-18 needs 17 ms per picture at 160 x
120 against 26 ms at 224 x 224 (backbone only); 224 x 224 has 2.6 times the pixels. Papers used 84 x 84 to 320 x 240.
No size comparison exists on our data: keep 160 x 120, and try a larger size only on the GPU pool. [measured;
calculated; literature; judgement]

### 6.6 Optimiser and learning-rate schedule

AdamW, head learning rate 0.001, backbone 0.3 times that, weight decay 0.05, batch 64, a linear warm-up over the first
epoch, then a cosine decay to zero [measured]. Nothing on our data shows the rate is too high. Gentler fine-tuning is
suggested by the literature (an encoder at one tenth of the policy's rate gave Chi et al. their best results; training
the head first and then everything was on average 10 % better out of distribution in Kumar et al. 2022), but neither
result comes from a pushing task. Test "backbone factor 0.1 plus one head-only epoch" as one combined change only when
laptop time allows. [literature; judgement]

### 6.7 Epochs and which weights to keep

- **8 epochs, each on a random half of the rows (about 4 passes), no early stopping, last weights kept**, exactly as in
  the final candidate. [measured]
- Validation loss on held-out episodes of the training sheets is a poor guide, because those episodes are near-twins of
  training episodes (section 3.6), and in robomimic the last checkpoint beat the lowest-validation-loss one in all 8
  dataset and algorithm pairs. [measured; literature]
- The runs before the final candidate kept their best-validation weights: the names run epochs 7, 5 and 7, the small
  network epochs 15 to 16, the chunk run epoch 7 on sheet 05. So those numbers were chosen on the split the audit forbids
  for choosing, which is one more reason to compare them with care. [measured]
- The validation loss gives mixed signs about training longer. In the earlier exact run it was still creeping down at
  epoch 8 on sheets 03, 05 and 06 (0.487, 0.535 and 0.572), partly because the cosine schedule ends there; in the final
  candidate it rose slightly at epoch 8 on sheet 01 (0.5503 to 0.5507) and sheet 04 (0.6754 to 0.6774, lowest at epoch
  5, 0.6354) and fell on the other four. No held-out-sheet score per epoch was logged, so whether 16 epochs would beat 8
  is unknown; it is a low-priority test (9.1), with held-out-sheet scores logged every 2 epochs. [measured; judgement]

### 6.8 Regularisation

- Head: one hidden layer of 256 with dropout 0.2. A linear head lost 7.8 points of step within 45 (lower on all six
  sheets) and had 16.9 points more early stops; a 1,024-wide layer with dropout 0.3 lost 1.9 points. Dropout 0.2 in the
  head almost always helped on a real robot in Dasari et al. 2023. [measured; literature]
- No label smoothing: smoothing 0.1 lost 1.9 points of step within 45 on all six sheets (weak evidence, inside the
  noise). Weight decay 0.05 stays as it is; no comparison exists. [measured; judgement]

### 6.9 Seeds

Seeds needed per setting to detect a true difference (two-sided test at 5 %, power 80 %, from the measured frozen-probe
spreads) [calculated]:

| True difference | Step within 45 (spread 1.24 to 1.70 points) | Stopped right (spread 2.0 to 3.6 points) |
|---|---|---|
| 2 points | 8 to 13 seeds each | 17 to 53 |
| 3 points | 4 to 7 | 9 to 24 |
| 5 points | 3 to 4 | 4 to 10 |

Rules for Team M [judgement]:

1. Frozen probes (2 to 4 minutes each): at least 4 seeds per setting; act only on 3.5 points or more of six-sheet step
   within 45, or 7 points or more of stopped right.
2. End-to-end runs (about 2 to 7 hours for six sheets, section 5.5): 2 seeds per setting, and only for changes
   expected to move path within 45 by 5 points or more; smaller questions go to the robot or stay at the default.
3. Seed each fold from its sheet, not from its place in the list: today `train_policy.run_training` adds the fold's
   position to the seed and to the validation draw, so a run on sheets 03, 05, 06 and a run on all six sheets use
   different seeds and validation episodes for the same sheet. [measured; read in `train_policy.py`]
4. The exam model: one model with the recipe; keep a second seed as a spare. Do not spend robot time choosing between
   seeds: 24 paired blocks pick the better of a 90 % and an 80 % model only about 83 % of the time. [calculated;
   judgement]

---

## 7. Validation that predicts the exam

### 7.1 Held-out episodes against held-out sheets

Frozen ResNet-18 grid probes, one seed; share of moving pictures within 45 degrees of the demonstrated step [measured;
`splits.json`, re-scored per sheet]:

| Sheet | Always forward | Names: episodes / sheet held out (gap) | Exact: episodes / sheet held out (gap) | Pictures with a closer twin in another episode |
|---|---|---|---|---|
| 01 straight | 0.747 | 0.764 / 0.756 (+0.9) | 0.752 / 0.747 (+0.5) | 16.2 % |
| 02 arc | 0.565 | 0.540 / 0.530 (+1.1) | 0.629 / 0.583 (+4.6) | 17.7 % |
| 03 two curves | 0.522 | 0.654 / 0.565 (+8.9) | 0.692 / 0.624 (+6.9) | 51.4 % |
| 04 meander | 0.485 | 0.659 / 0.559 (+10.0) | 0.711 / 0.548 (+16.3) | 50.5 % |
| 05 loop | 0.509 | 0.705 / 0.583 (+12.1) | 0.754 / 0.563 (+19.0) | 41.7 % |
| 06 dashed loop | 0.402 | 0.690 / 0.467 (+22.3) | 0.749 / 0.483 (+26.6) | 46.3 % |
| Mean | 0.538 | 0.669 / 0.577 (+9.2) | 0.714 / 0.591 (+12.3) | |

- With four seeds the mean gaps are +10.9 (names) and +11.2 (exact) points. The stop suffers most: holding out the whole
  sheet halves the right stops (106 to 54 of 208 episodes for names, 119 to 58 for exact). [measured; calculated]
- **What it means** [judgement]: plan with held-out-sheet numbers for unseen tracks. The held-out-episode numbers are at
  most an upper bound for the known sheets laid out again at the exam. Never choose a model, a setting, an epoch or a
  stop threshold on held-out episodes.
- A leave-one-block-out test inside sheets 02 and 04 would measure "known sheet, new session", but there the blocks
  are tied to light and camera changes, so it cannot cleanly tell remembered views from track features. [measured;
  judgement]

### 7.2 Which offline numbers to trust, and what they cannot tell

Report every held-out sheet beside its own always-forward rule, and rank on these measures, in this order [judgement]:

1. path within 45, mean of six sheets and worst sheet;
2. step within 45, beside it;
3. the stop per episode under one fixed rule (0.5, two in a row) as counts right / early / 2 to 3 cm short / never, per
   sheet and per line end;
4. path and step within 45 on backward moves and on the pictures after a hand correction (the nearest thing to
   recovery pictures in our data);
5. the twist sign where the demonstration twisted.

Why path first: repeating the previous move, which a picture-only model cannot know, lifts step within 45 from 0.538 to
0.736 (+19.8 points) but path within 45 only from 0.604 to 0.648 (+4.4). Do not choose on validation loss or on 8-way
class agreement, which hardly separates anything (always forward 40.3 %, a leaky look-up 42.3 %). [measured]

From the literature [literature]: validation loss is the weakest guide to robot success (robomimic, above); thresholded
direction errors predicted closed-loop driving better than the mean squared error (correlations 0.65 and 0.64 against
0.39; Codevilla et al. 2018); offline numbers spot bad models but rank good ones poorly (correlations of 0.21 to 0.55
among the better half; within one data budget no significant link, Cowling 2026); inputs that reveal the previous move
flatter offline scores; in a 2026 preprint the validation action error tracked real success (Spearman -0.93 over 16
checkpoints) while the validation loss did not (Allshire et al. 2026); and recovery pictures in the validation set
raised the offline-to-driving correlation from 0.39 to 0.54 (Codevilla et al. 2018, in its main condition), which is why
the rows after a hand correction are measure 4.

**Offline gate before any robot time** (judge 2; the final candidate meets all three) [judgement]: path within 45 above
always forward on every held-out sheet (final candidate: +5.0 to +37.7 points), twist sign 0.6 or more on every sheet
(0.676 to 0.848), and early stops under about 15 % pooled under the fixed rule (25 of 209, 12 %). The backup passes the
same gate; the exam model, which has no held-out sheet, is checked on printed validation tracks instead (4.5, 7.4).

**Checking this order on our robot.** Write each candidate's per-sheet offline numbers (path within 45, the stop counts,
the rows after a correction) into the robot plan as a calibration table and set its phase 1a and phase 2 results beside
them (7.4). If the finishes follow the stop counts more closely than path within 45, rank on the stop first.
[judgement; validation draft]

**What offline numbers cannot tell** [judgement on the literature and our data]: how errors add up in a closed loop
(errors came in streaks: five or more wrong path pictures in a row in 25 of 45, 20 of 29 and 10 of 25 episodes on sheets
03, 05 and 06 for the final candidate); stalls and back-and-forth moves; recovery after a drift; the finishing time; an
executor fault; a new light, table or room; a moved camera; and the stop at a line end far from the paper edge. Only the
robot shows these.

### 7.3 How large an offline difference must be

| Measure | Seed spread seen (frozen probes, four seeds) | Count a single-run difference only above about |
|---|---|---|
| Step within 45, six-sheet mean | 3.1 (names), 2.4 (exact); 3.3 for three more names seeds | 3.5 points |
| Step within 45, one sheet | up to 9.1 (names), 9.8 (exact) | 10 points |
| Path within 45, six-sheet mean | 3.4 and 3.5 | 3.5 points |
| Path within 45, one sheet | up to 10.7 (names), 16.0 (exact) | 16 points |
| Path within 45, worst sheet | 12.4 (exact) | 12 points |
| Stopped right, six-sheet mean | 5.9 (names), 4.3 (exact) | 7 points |
| Stopped right, one sheet | 11 to 26 (names), 7 to 32 (exact) | 30 points |

[measured; thresholds are judgements] Also require the same sign on at least five of the six sheets, and compare designs
on three or more seeds. For one new track like these six the range is wide: for the final candidate's path within 45 a
prediction interval for one new track is about 0.66 to 0.91. [calculated by judge 2]

### 7.4 The robot test protocol

Nothing here has been run. It starts only after Emranul and Marshal agree the plan, and it doubles as the checklist the
project rules ask for before every robot session.

**A written plan, committed to git before the first trial under the students' own names.** It fixes the candidates by
code letter with their files and offline numbers, the locked executor settings, the tracks, the starts, the complete
trial order (from a fixed random seed), the success rule, the list of void faults, the analysis and the decision rules.
Any later change is listed as a deviation with its reason. [literature: pre-registration as recommended by the checked
evaluation sources; judgement]

**Tracks in four roles** [judgement]:

- *Known sheets* 01 to 06.
- *Practice tracks:* new printed tracks for settings pilots and correction rounds; once trained on, they are training
  tracks.
- *Validation tracks* (two or more): used only to choose between candidates and to check the exam model's stop scale,
  never trained on. Between them: a line end 10 cm or more from the paper edge, and parallel line parts about 2 cm
  apart (both beyond what the six sheets hold).
- *Sealed test tracks* (at least ten per set): A4, 1-cm line, round end caps, drawn by a written random rule (line length
  within the six sheets' 23 to 72 cm, every point inside the arm's demonstrated reach of 13.0 to 48.5 cm from the base,
  features at random: solid or dashed, a crossing or none, an end near or far from the paper edge). Two sets: a
  development set for checking a retrained model, and a final set used once. They stay out of sight until used.

**Starts.** The standard start of the recordings, with the puck placed using the recording tool's reference-picture
overlay (the "ghost"), so that both models start from the same picture; both ends of each known sheet, balanced. New
tracks are laid by the written exam lay-out rule; the exam-start note of 6.11.26 suggests one taped mark about 17 cm in
front of the robot's centre, an opinion and not yet a decision (an unseen line running more than about 24 cm from its
start may not fit it; Emranul expects 4 to 5 sheets in very different orientations, `requests_to_professor.md`
question 18, open). Offset and turned starts go in a separate block, reported apart, except in the comparison of the
first set-up with the backup: there starts with the puck laid beside the block are included, because backing up is
what the backup is for (judge 1; targets exam-risk critic). [measured set-up facts; judgement]

**Success rule ("finished"), all four must hold** [judgement]:

1. the policy stops by itself under the fixed stop rule, within the step limit (about 4 steps per cm of line);
2. at that moment the puck's centre lies within 2.0 cm of the centre of the line's end cap, measured on an overhead photo
   taken right after the stop, scaled by the sheet's 29.7-cm edge (2.0 cm is a judgement: the offline "right" measures
   the tip, so check on the first photos how many demonstrations would pass);
3. during the run the puck never left the line (a strip of bare paper visible between the puck and the line's course;
   on a dashed line, the course through the gaps; judged live, checked on the overhead video when in doubt), and it left
   every crossing on the line's own branch;
4. nobody touched the robot, the puck or the sheet, and no safety stop happened.

Secondary, fixed in advance: the same rule at 1.0 and 3.0 cm, and "reached the end without stopping by itself". When
the exam's rule is announced, re-score every trial under it, beside the pre-registered rule.

**Recorded in every trial:** time from the first model command to the stop; number of steps; the furthest progress
along the line; how far the puck's path strays from the line; the first failure type (left the line, wrong branch,
stopped early, stopped 2 to 3 cm short, never stopped, stalled or back-and-forth, meaning under 1 cm of progress over 10
steps, refused moves or joint limit, safety stop); the joint-4 range; the stop distance. A trial is *void* (re-run at the
end of the session) only for faults that are not the policy's doing: lost connection, no fresh picture, a set-up error,
a person in view. A policy that wanders or stops is never void. The operator ends a run early only when the puck has left
the paper, the arm has not moved for 5 steps, or for safety. One uncounted warm-up run on a practice track opens each
session. [judgement; validation draft]

**Light and drift.** The ceiling light of most recordings. At the start of every session, a camera check at one fixed
pose against a stored reference picture: the tool's own check takes the first picture at a new start position as its
reference, so on a new track it cannot see a camera move. Use the team's own tolerance, under 4 px (the forearm
camera-mount memory); if the check fails, stop the session and follow the camera-mount checklist, never touch the
camera. In the literature a new camera position was the hardest change for a trained policy, a new table texture the
next (Xie et al. 2023) [literature]. Log the brightness of each trial's first picture and flag trials outside 0.67 to
1.12 of the training median. Interleave the two candidates inside a session in the pre-generated random order, never one
per session. Run a separate light block under a changed light, reported apart. [measured; judgement]

**Roles.** One person keeps the code-letter key and runs the program; the other places the puck and scores. Scores are
written before the key is opened. With two people the blinding is imperfect, and the report says so.

**Session rules from the project's working rules.** A checklist before every session; motors off (FreeMotion) before
anyone touches the arm, never pushing on a held arm; only one program connected to the robot; the camera and its focus
ring not touched; only the automatic calibration after a restart; the robot away from the table edge; the step limit on.

**Phases.** Robot time is calculated from the measured 1.22 s per step, the median steps per sheet and the measured
set-up of 1.86 minutes per trial between demonstrations (0.75 from start to first row, 0.19 from last row to finish,
0.92 from finish to next start; `design/improvement/episode_timing.out.txt`). The validation draft used a guessed 1
minute instead; each phase below adds 0.86 minutes per trial to its figure. The slow case in brackets assumes 1.5 times
the steps and 2.5-minute resets, both guesses. Section 8.4 uses the same base. [calculated]

| Phase | What | Trials | Robot time | Reported as |
|---|---|---|---|---|
| 0a | Executor check without a model: replay recorded steps through the new model-step executor on sheets 01, 02 and 04 (both ends); force one refusal; drive joint 4 to its clamp; camera check | | not estimated | not a result |
| 0b | Pipeline check: the final candidate's fold model on sheet 01, 3 runs per end; measures the real reset time and steps per run | 6 | 15 min (21) | not a result |
| 1a | Unseen sheets without printing: each fold model on its own held-out sheet (`without_path_03` on sheet 03, and so on for 04, 05, 06), about 5 runs each, plus a few runs with one provided sheet laid on a larger white sheet (does the stop depend on the paper edge?); optionally `without_path_02` on sheet 02, about 5 runs, a far-from-edge stop test without printing (its line ends lie 6.2 and 6.4 cm from the paper edge, and its fold model never stopped in 24 of 45 episodes offline, two in a row) | about 20 plus (25 with sheet 02) | about 2 to 2.5 h, plus about 16 min for sheet 02 | exploratory |
| 1 | Model choice, only if two candidates survive and both finish most 1a runs: 24 paired blocks on the hard sheets and the validation tracks; in each block both candidates run from the same start in random order; an interim look after 12 blocks | 48 | 3.4 h (4.8); about 2.5 h (3.7) expected with the interim look | exploratory |
| 1b | Speed, if the time is to be improved: the chosen model at the 20 % cap against one faster setting, 12 paired blocks, decided by finish first, then time | 24 | about 1.6 h (2.4) | exploratory |
| 2 | Final test: the six known sheets x 5 trials and ten sealed tracks of the final set x 3 trials, in one random order over two sessions | 60 | 4.0 h (5.7) | the reported result |
| Light | The chosen model, 6 trials under a changed light | 6 | 0.4 h (0.5) | reported apart |

[calculated; validation draft and its critics] Phases 0b, 1, 2 and the light block take about 8.0 hours (11.4 in the
slow case). Changing sheets and ends between trials adds roughly 1.4 to 2.9 hours (from guessed change times of 1 to 2
minutes; validation evidence critic), so about 9.4 to 10.9 hours in all; phase 1a adds about 2 to 2.5 hours.

**If robot hours run short** (validation draft) [calculated; judgement]: phase 1 with 12 blocks only (it still picks the
better of a 90 % and a 70 % model 89 % of the time), and phase 2 with the six known sheets x 3 and the ten sealed
tracks x 2 (38 trials), reported as exploratory with its intervals. How many robot hours a week can be booked is an open
question (10.2).

**Calibration table.** The written plan lists each candidate's offline numbers per sheet (path within 45, the stop
counts, the rows after a hand correction); the phase 1a and phase 2 results are set beside them, to see which offline
measure follows finishing on this robot (7.2). [judgement; validation draft]

Robot time per trial by sheet (measured 1.22 s per step, median steps per demonstration, measured 1.86-minute set-up)
[calculated]:

| Sheet | Steps (median) | Driving time | With the 1.86-minute set-up |
|---|---|---|---|
| 01 straight | 31.5 | 38 s | about 2.5 min |
| 02 arc | 66 | 81 s | about 3.2 min |
| 03 two curves | 106 | 129 s | about 4.0 min |
| 04 meander | 168 | 205 s | about 5.3 min |
| 05 loop | 103.5 | 126 s | about 4.0 min |
| 06 dashed loop | 82 | 100 s | about 3.5 min |

**Decision rules, fixed in advance** [judgement]:

- After phase 1a: mostly stop failures, keep the set-up and make line-end data the next job; mostly lost pucks or no
  recovery, run the backup's fold models the same way; executor faults, fix them for every set-up.
- Phase 1: the candidate with more successes over the blocks run; a tie goes to the one that wins more blocks under
  "finish first, then time"; a remaining tie goes to the simpler, faster model. Report the counts and an interval for the
  difference, and claim no significance. The winner's phase-1 score is not its final score.
- Do not cancel phase 2 when fewer than half of the phase-1 trials finish: phase 1 is placed on hard tracks near 50 %
  success on purpose. Instead, write a bar on the validation tracks into the plan beforehand.
- Phase 2 analysis: finishes on the sealed tracks as k of 30 with a Wilson interval, plus k of 3 per track (trials on one
  track share its difficulty); the known sheets as k of 30 and k of 5 per sheet; a bar "above 70 % on new tracks" passes
  at 26 of 30 or more; finishing time per sheet among finished trials, and two models' times compared only inside paired
  blocks.

### 7.5 How many robot trials tell two models apart

95 % intervals for a success rate [calculated; Wilson intervals are the ones recommended for small samples, Brown et al.
2001]:

| Observed | Wilson interval |
|---|---|
| 5 of 10 | 0.24 to 0.76 |
| 7 of 10 | 0.40 to 0.89 |
| 9 of 10 | 0.60 to 0.98 |
| 10 of 10 | 0.72 to 1 |
| 14 of 20 | 0.48 to 0.85 |
| 18 of 20 | 0.70 to 0.97 |
| 24 of 30 | 0.63 to 0.90 |
| 27 of 30 | 0.74 to 0.97 |

| Question | Answer |
|---|---|
| Do 7 of 10 and 9 of 10 differ? | No: the interval for the difference is -0.16 to +0.51 [calculated] |
| Which of two models is better (90 % against 70 %), if we only want to choose? | right 87 % of the time with 10 trials each, 96 % with 24, 98 % with 30 [calculated] |
| The same for 90 % against 80 % | 81 % with 20 each, 83 % with 24, 92 % with 50 [calculated] |
| Prove that 90 % beats 70 % | 49 to 50 trials per model one-sided, 61 to 63 two-sided [literature, checked] |
| Show a gap of 8 to 9 points | about 500 trials per model (Snyder et al. 2025) [literature] |
| Interim look after 12 of 24 paired blocks (stop when one model leads by 3 successes) | keeps the chance of choosing the better of 90 % and 70 % at 0.961 (fixed 24 blocks: 0.962) and cuts the expected blocks to 18.2 [calculated] |

Trials on one track share its difficulty: with a guessed correlation of 0.15 between trials on one track, 30 trials on 3
tracks count like 12.8 independent trials, on 10 tracks like 23.1. So the final test uses many tracks with few trials
each. [calculated from a guess]

---

## 8. Improving after the first model

### 8.1 Where the first model goes wrong

The earlier end-to-end exact model (old rules) on its three held-out sheets (03, 05, 06; 7,738 moving pictures),
against always forward on the same pictures [measured; calculated for the margins; `situation_errors.csv`]:

| Situation | Step within 45 (moving pictures) | Margin over always forward (points) |
|---|---|---|
| Puck within 1 cm of the line | 0.744 (4,155) | +22.1 |
| Puck 1 to 2 cm off the line | 0.615 (1,360) | +19.1 |
| Puck more than 2 cm off the line | 0.571 (1,105) | +15.6 |
| Line radius 10 cm or more | 0.735 (2,937) | +13.7 |
| Line radius 1.5 to 10 cm | 0.633 to 0.656 | +24.1 to +25.1 |
| Line radius under 1.5 cm | 0.610 (390) | +8.4 |
| More than 8 cm of line to go | 0.683 (6,163) | +19.7 |
| Last 3 cm of the line | 0.607 (461) | +11.0 |
| Straight stretch | 0.798 (2,378) | +17.6 |
| Bend | 0.633 (5,360) | +20.2 |
| Puck within 2 cm of a crossing (sheets 05, 06) | 0.799 (432) | +18.3 |

- **Real weak spots, after allowing for how easy each situation is:** the puck off the line, the tightest bends (radius
  under 1.5 cm), the last 3 cm of the line, and the stop (section 4.5). Bends in general are not a weakness relative to
  always forward. The crossings look fine end to end, but only 2 crossings on 2 sheets are in the data, and the frozen
  probe scored below always forward there (0.590 against 0.616). [calculated; judgement]
- **The final candidate's weak spots, on its sheets 01 and 02:** pictures after a hand correction 0.493 within 45
  (always forward 0.389) against 0.762 elsewhere; within 3 cm of the goal 0.537 of 257 pictures against 0.502 for
  always forward; and, on all six sheets, backing up (section 2.1). [measured by critics]
- **The final candidate on all six held-out sheets, by situation** (path within 45, always forward on the same pictures
  in brackets; `training_error_analysis.py`, which counts slightly more pictures than the fold files) [measured; margins
  calculated]:

  | Situation | Pictures | Path within 45 | Margin (points) |
  |---|---:|---|---:|
  | Straight stretch | 5,857 | 0.860 (0.740) | +12.0 |
  | Bend | 12,107 | 0.735 (0.524) | +21.1 |
  | Start (first pictures) | 413 | 0.900 (0.780) | +12.0 |
  | Near the goal | 1,041 | 0.690 (0.566) | +12.4 |
  | Forearm straight / turned / far turned | 6,292 / 7,765 / 3,907 | 0.798 / 0.754 / 0.766 (0.716 / 0.518 / 0.518) | +8.2 / +23.6 / +24.8 |

  As for the earlier model, it is weakest near the goal, and bends are not a weakness relative to always forward. The
  puck-off-line, bend-radius and crossing splits were made only for the earlier model (`design/improvement/
  situation_errors.py` could redo them); light, camera view and score are in 3.7 and 3.8. [judgement]
- On the robot the model's own small errors will push the puck off the line more often than in the demonstrations, so
  the off-line weakness should grow in closed loop. [judgement; DAgger's argument, literature]

### 8.2 More different tracks or more episodes

Frozen ResNet-18 grid features with a names head, leave one sheet out, three seeds per setting [measured;
`data_amount.json`; training pictures calculated]:

| Training data per fold | Training pictures (mean) | Step within 45, mean of 6 sheets (seed range) | Stopped right, tuned, 2 in a row |
|---|---:|---|---|
| 1 sheet, all its episodes | 3,304 | 0.456 (0.431 to 0.471) | 0.097 |
| 2 sheets | 5,758 | 0.507 (0.491 to 0.523) | 0.182 |
| 3 sheets | 7,632 | 0.520 (0.514 to 0.533) | 0.255 |
| 4 sheets | 10,774 | 0.532 (0.531 to 0.533) | 0.323 |
| 5 sheets, 25 % of the episodes | 3,264 | 0.514 (0.512 to 0.518) | 0.207 |
| 5 sheets, 50 % | 6,527 | 0.546 (0.541 to 0.549) | 0.257 |
| 5 sheets, 100 % | 13,055 | 0.555 (0.541 to 0.574) | 0.320 |

- **At equal pictures, more sheets win:** five sheets at a quarter of their episodes beat one full sheet on all six
  folds, by 3.5 to 11.4 points; five sheets at half their episodes (0.546) beat four full sheets with 65 % more pictures
  (0.532). [measured]
- **A fit over all 126 fold results** gives +1.7 points of step within 45 per doubling of training pictures (bootstrap
  +0.8 to +2.7) and +2.5 per doubling of different sheets at equal pictures (+1.6 to +3.4). The sheet effect survives
  dropping the one-sheet runs (+2.13, +0.90 to +3.32). [calculated]
- **The stop grows with the amount of data** (+8.7 points per doubling of pictures), because each episode holds one stop
  picture; its sheet effect does not survive the same refit (+4.57, -0.85 to +9.63). For the stop, more stop examples
  helped; more sheets were not shown to help. [calculated]
- **Limits:** frozen features with a names head, not the end-to-end exact model; one to five training sheets, so
  anything beyond six is an extrapolation; the 3.3-point seed range at full data includes a different draw of validation
  episodes per seed, not only training noise. [judgement; measured]
- **Literature:** generalisation to unseen scenes and objects grew roughly with the number of training environments, not
  demonstrations (Lin et al. 2025; a simulation study, Xie et al. 2023, saw the gap fall from about 0.4 to under 0.1 from
  5 to 100 environments). Their "environments" are scenes and objects, not path shapes, and no study the checkers opened
  tested new path shapes of the same look, which is what the exam asks. [literature]

### 8.3 Correction rounds

**The idea** (DAgger, human-gated as in HG-DAgger): the trained model drives, a person watches, and when it goes wrong
the person takes over, so that the corrections come from the states the model really reaches. On real robots this
helped in HG-DAgger and BC-Z (ThriftyDAgger also reported a gain, but its behaviour-cloning baseline had less data and
the difference, 12 against 10 of 15, is within chance); about half of the opened studies used 1 to 3 rounds, with
rounds of 20 to 33 % of the initial data or 50 episodes; retraining from scratch on old plus new data was stable,
fine-tuning on the corrections alone was not. No study shows that corrections carry over to unseen layouts, and none
reports robot hours per round. [literature]

**Why the person takes over instead of labelling afterwards** (plain DAgger has a person label the model's own pictures
after the run) [judgement from the code audit; literature]: the exact step with its twist cannot be typed afterwards;
key-override corrections drop the twist and, with a turned forearm, re-solve for the least lean, which moves the picture
about 56 px from the training views; and labels given afterwards were unreliable in Laskey et al. 2017 (agreement with
what the people had done: correlations of 0.60 and 0.22; about 40 against 60 % success; mouse labels on half-speed
video).

**How with our tool** (from the code-reading audit; a proposal to be written as a robot checklist first):

1. Lay the puck as for a demonstration; the model drives one step per picture (about 1.2 s per step, so the watcher can
   react).
2. When it goes wrong (drifting off the line, stalling, turning the wrong way, stopping early, running past the end),
   pause and press h. Motors off (FreeMotion) before anyone touches the arm; guide it by its body, never by the camera or
   its tape.
3. Version 1 (the take-over as it exists in `hand_guided_exact`): guide from the failure to the end; the tool lifts,
   returns, the puck goes back with the ghost picture, and the guided path is replayed with exactly the demonstrations'
   labels (source replay_exact). Version 2 (a small change): guide only until the puck is back on the line, then let the
   model drive again; it finds more failures per hour.
4. In correction rounds the model's own stop must wait for Enter, as the replay's stop does now; otherwise early stops
   are never corrected.
5. Train only on the human's corrections, not on the model's own steps; keep the model steps, tagged, for analysis. The
   first replayed row (waypoint 0) is dropped by the cleaning, like the 239 correction starts of the demonstrations.
   New tags need new step columns (the writer raises on unknown keys).
6. If the puck has left the sheet or the view, end the episode instead of recording a long recovery.
7. After the episode, note the failure type in the review form. Keep the correction pictures also as an offline
   validation set with recoveries (Codevilla et al. 2018 advise validation data with recoveries; 7.2).

**Measured from the demonstrations:** a take-over cost a median 51.6 s (17.1 s guiding, 33.4 s lifting, returning and
putting the puck back), and a correction replays a median 22 rows. Offline, leaving the operators' own 239 corrections
out of the training rows cost 1.7 points of six-sheet step within 45 (a control removing as many first-pass rows cost
2.3) and 7.5 points of stopped right (control 2.3): no detectable recovery-specific effect on direction, but they matter
for the stop, because most stop pictures sit at their ends. These corrections came where the operators' own replays went
wrong, not where a model goes wrong, so this neither supports nor rules out on-policy corrections. [measured; judgement]

**Round size.** 50 correction episodes at a guessed 30 % first-model success give about 1,624 correction rows, 9 % of
the 18,053 rows, smaller than the 20 to 33 % rounds in the studies [calculated from a guess]. With version 1 most of
these rows are ordinary pushing from the failure point to the end, so expect a smaller effect than in the studies. The
improvement draft's choice for round 1 stands: 50 episodes, version 1 first, version 2 once it is built and tested,
since it collects more failures per hour; enlarge round 2 only if round 1 shows a gain [judgement]. Twenty
demonstrations per new track (step A of 8.4) is a judgement from the data-amount experiment: half the episodes of each
training sheet (about 8 to 19 per sheet) already gave within 0.9 points of the full result [measured; judgement].

### 8.4 What to record, how much, and the robot hours

**What to record more of**, in order of expected value [judgement on the measured weak spots]:

1. **Line ends and stops.** Printed tracks with line ends 10 cm or more from the paper edge, lines running along the edge
   mid-track, and dashed ends; near-end starts (2.4 minutes from start to next start, about 25 per hour) on every new
   track and about 10 more on each loop sheet. Before recording many near-end starts, check on the laptop whether they
   help the mid-track false stops at all: train once without the 42 existing near-end-start episodes and compare the
   early stops on held-out sheets.
2. **Off-line and recovery states.** Correction rounds (only they reach the model's own drifts), and deliberate recovery
   starts that need no model mode: the puck 1 to 2 cm beside the line mid-track, guided back (about 30 starts, 2.8 to 3.6
   robot hours). Recovery data has the strongest outside support: removing it cut driving success from 88 and 64 % to
   56 and 22 % (Codevilla et al. 2018; their training also used side cameras), and off-track data raised wrist-camera
   pushing with 1-cm moves from 20 to 80 % in 10 trials (DMD, with extra play data; the checker's corrected form)
   [literature].
3. **Tight bends and loops in new places**, both turning directions, including one long one-way turn that tests the
   joint-4 limit.
4. **Dashed lines of another rhythm** (sheet 06 is the only dashed sheet).
5. **Coverage:** a crossing near 90 degrees, close parallel stretches. Not needed: more of sheet 01; more daylight only if
   the exam runs in daylight.

**Robot hours per option** (calculated from measured cycle times; guesses: 1.0 to 1.5 times the demonstration's steps,
30 % first-model success, 0.5 h to print, tape and teach a new track; expected gains from the frozen-probe fit of 8.2,
extrapolated beyond five sheets) [calculated; guess]:

| Option | Robot hours | Data added | Expected offline gain (frozen fit; a rough guide) |
|---|---|---|---|
| 5 trials on each of the six sheets | 1.9 to 3.3 | none | finds the failures |
| 10 sealed tracks x 2 trials | 1.3 to 2.2 | none | unseen-track test |
| Pilot: 5 correction episodes | 0.4 to 0.6 | about 160 rows | measures a take-over in model mode |
| Correction round, 50 episodes | 3.7 to 6.3 | about 1,600 rows (9 %) | not estimable offline |
| Correction round, 100 episodes | 7.3 to 12.5 | about 3,250 rows (18 %) | not estimable offline |
| 100 more demonstrations on the six sheets | 9.4 to 13.6 | pictures x1.48 | +1.0 step within 45, +4.9 stopped right |
| 4 new tracks x 20 demonstrations | 9.5 to 12.8 | pictures x1.38, sheets x1.67 | +2.6 and +6.5 |
| 6 new tracks x 20 demonstrations | 14.3 to 19.3 | pictures x1.57, sheets x2.00 | +3.6 and +8.9 |
| About 30 recovery starts on the known sheets | 2.8 to 3.6 | recovery rows | not estimated |

Both projected direction gains (+2.6 for four new tracks, +1.0 for 100 more episodes on the six sheets) are below the
3.3-point spread of single runs, and within the bootstrap ranges their ratio could be anywhere from about 1.6 to 6. So
new tracks are the better bet, not a sure one. [calculated; judgement]

**Recommended order** (the improvement draft's steps A to E, re-ordered as its exam-risk critic proposed; robot steps
only after the team decides, each with its own checklist) [judgement; hours calculated]:

| Step | What | Robot hours |
|---|---|---|
| 0 | First closed-loop check (phases 0 and 1a of section 7.4) | about 2 to 2.5 plus the executor check |
| A | If printing is allowed: 4 new training tracks x 20 demonstrations (5 near-end starts each), plus 10 near-end starts on each of sheets 05 and 06; plus about 30 recovery starts on the known sheets (2.8 to 3.6); varied light; the camera check every sitting | 10.3 to 13.9, plus 2.8 to 3.6 |
| B | Baseline test of the model trained on all sheets (5 trials per sheet, 2 on each of 10 development tracks) and a pilot of 5 correction episodes; freeze the executor here, because corrections describe the executor they were recorded with | 3.6 to 6.1 |
| C | Correction round 1, 50 episodes, only if most failures are drift or wrong turns | 3.7 to 6.3 |
| D | Retrain from scratch on everything (plain mixing fixed in advance), then a paired, interleaved robot test of the new model against the first on the development tracks, not the final sealed set | 6.4 to 11.0 |
| E | Correction round 2 only if D showed a clear gain before the freeze | 2.2 to 6.3 |

Totals: 24 to 37 robot hours for A to D (26 to 44 with E); with the closed-loop check, the recovery starts, the final
test and a rehearsal about 31 to 46 hours (33 to 53 with E), about 6 to 9 hours a week over the five weeks from 16.11
to 20.12.26 (section 9 plans no robot hours this week). The per-trial times use the measured 1.86-minute set-up, as in
7.4 (for example 2.5 to 2.8 minutes a trial on sheet 01; `design/improvement/robot_hours.out.txt`). For scale: the 210
demonstrations took 28.5 hours of sittings. [calculated; measured]

### 8.5 If something is not allowed

- **No printing:** spend the hours on correction rounds on the six sheets, near-end starts, recovery starts, and the
  extra track files promised on 26.10 when they arrive; test unseen tracks with the fold models on their own held-out
  sheets. Expect a smaller gain on unseen tracks. [judgement]
- **No correction rounds:** record deliberate recovery demonstrations instead (the data already hold 60 offset or turned
  starts). [measured; judgement]
- **The first model finishes almost nothing:** a take-over would turn into a full demonstration from an odd state; record
  targeted demonstrations of the failing situations first and fix the executor before any correction round. [judgement]

---

## 9. The plan to the deadline

Fixed dates: meetings 3 (23.11.26) and 4 (7.12.26); exam registration in PRIMUSS 9 to 17.12.26 (both of you, binding);
all robot work finished by 20.12.26; freeze 8.1.27; hand-in 10.1.27 (23:59); live demo 11.1.27. Every robot step below
waits for Emranul's and Marshal's decision and a checklist. Hours come from sections 5, 7 and 8; the order is a
judgement.

| Week | Laptop (hours) | Robot (hours) | What it decides |
|---|---|---|---|
| 9.11 to 15.11 | The queue runs until about 06:30 on 10.11 (already started). Then: run `compare_runs.py`, which scores the backup and the second seed under the fixed stop rule (minutes); add an "all sheets" option to `train_policy.py` and train the exam model (under about 1.5 h per seed); stress-test all six fold models under light and shift changes, scored under the fixed rule (about a minute per condition per fold); build the model mode (not estimated), with the replay and fresh-picture checks and a run in `--simulated` mode (4.6); write the session-1 checklist, the robot test plan, the track-drawing rule (a small script) and the exam lay-out rule; fix `record_episodes.py` line 1120 before more recording (low priority: the cleaning relabels those rows); send the professor the questions of section 10.1 | none | the first set-up and the backup; the end-to-end seed spread; an executor ready for the robot |
| 16.11 to 22.11 | If wanted: the chunk rerun with the cleaning and last weights on six sheets (one six-sheet run, about 2 to 7 h); the near-end-start check of 8.4 (minutes as a frozen probe, one six-sheet run end to end) | Session 1 (needs the model mode): phase 0a executor check (not estimated); phase 0b on sheet 01 (15 to 21 min); phase 1a, each fold model on its own held-out sheet, with the white-underlay test (about 2 to 2.5 h, about 16 min more with sheet 02); a pilot of 5 correction episodes (0.4 to 0.6 h) | does the executor work; the real time per trial; the main failure type; does the stop rely on the paper edge |
| 23.11 (meeting 3) to 29.11 | Retrain only if session 1 shows a data or setting problem | If printing is allowed: print the practice, validation and two sealed sets; start step A (10.3 to 13.9 h in all) and the recovery starts (2.8 to 3.6 h). If not: corrections, near-end and recovery starts on the six sheets | the professor's answers set the data plan |
| 30.11 to 6.12 | Retrain from scratch on all data (about 1 to 1.5 h per model; a six-sheet check about 5 to 6 h, a guess); a 16-epoch test if time allows (4 to 7 h per seed) | Finish step A; step B, the baseline test of the all-sheets model with the correction pilot (3.6 to 6.1 h); phase 1 only if two candidates survive (about 2.5 to 4.8 h) | which model goes on; freeze the executor |
| 7.12 (meeting 4) to 13.12 | Retrain after the corrections (about 1 to 1.5 h) | Step C, correction round 1 (3.7 to 6.3 h), if failures are drift or wrong turns; step D, paired test of new against old on the development tracks (part of 6.4 to 11.0 h). PRIMUSS registration opens 9.12 | does the correction round help; freeze the model and the executor by about 13.12 |
| 14.12 to 20.12 | none needed | Phase 2 final test on the six sheets and the final sealed set (4.0 h, 5.7 in the slow case); the light block (0.4 to 0.5 h); one timed rehearsal of the exam procedure (not estimated). PRIMUSS closes 17.12. Robot work ends 20.12 | the reported result |
| 21.12 to 8.1.27 | Figures, tables and the documentation (7 pages per the course catalogue, if question 7 on `requests_to_professor.md` confirms it); slides | none | freeze on 8.1.27 |
| 9.1 to 11.1.27 | | the demo runs the locked model and executor | hand-in 10.1.27; live demo 11.1.27 |

Laptop total for the prioritised items: about 13 hours of queue tonight plus roughly 10 to 25 hours of retraining and
checks over the six weeks, well within the laptop's time if runs for decisions do not share the processor (the
training-settings draft listed 31 to 53 hours for every possible experiment, which its evidence critic corrected to
about 31 to 73, so most must stay optional; 9.1). [calculated from the items above; judgement] Robot total: about 31 to
46 hours if printing is allowed (section 8.4); a reduced robot plan is in 7.4.

**Who does what.** No owners are set; that is the team's decision (1.1, row 7). The team memory's provisional split has
Emranul running the robot sessions and Marshal building the recording tool and the dataset pipeline and sorting out
the GPU pool. The model mode is recording-tool work, and the trained copies are on Emranul's laptop, out of git (2.1).

**If the model mode is late.** Session 1 needs it. If it is not ready by about 15.11, session 1 and every robot date
after it move by the same amount; step A of 8.4 (new demonstrations, which need no model) can come first if printing
is allowed. [judgement]

### 9.1 Laptop experiments, in order

A six-sheet run takes about 2 to 2.5 hours when no other training runs, up to about 7 hours when shared (5.5); a
frozen probe 2 to 4 minutes. Priorities and rules are judgements, fixed before the results.

| # | Experiment | Question it answers | Laptop time | Priority | Rule fixed in advance |
|---|---|---|---|---|---|
| 1 | Score the backup and the second seed with the judges' scripts | keep, promote or drop the backup; the end-to-end seed spread | minutes | do (10.11) | section 2.5 |
| 2 | "All sheets" option in `train_policy.py`; the exam model, one seed plus a spare | the model for the exam | 1 to 1.5 h per seed | do | none: the same recipe |
| 3 | Replay and fresh-picture checks of the model mode (4.6) | does the robot side prepare pictures as training did | not estimated | do, before session 1 | its angles and stop probabilities equal the saved ones |
| 4 | Stress test of all six fold models (brightness 0.65 and 1.35, 8-px shift, 3-degree roll), scored under the fixed rule | how fragile the stop is | about a minute per condition per fold | do | none; it informs the light plan |
| 5 | Time the whole decider on a quiet laptop | the real decision time | about 0.5 h | do | none |
| 6 | Near-end-start check: train once without the 42 near-end-start episodes | do near-end starts help the mid-track false stops | minutes as a frozen probe; one six-sheet run end to end | do, before recording many near-end starts | record many more only if leaving them out raises the early stops |
| 7 | Chunk rerun with the cleaning and last weights, six sheets | is the chunk the better backup | one six-sheet run | optional | judge 3: stays a backup if its right stops are within about 7 points of the first set-up's; first if its path lead is above about 3.5 points on most sheets |
| 8 | Near-goal-window stop on the final recipe, six sheets | a better stop target (4.5) | one six-sheet run | optional | keep only if more than about 7 points more right stops (about 117 of 209 under the fixed rule) without more early stops |
| 9 | Combined stop rule on the saved models (re-run to read the step length) | a better stop rule without retraining (4.5) | minutes | optional | keep only if it cuts early stops by more than about 7 points without losing right stops |
| 10 | 8-px shift against 4 px, 2 seeds | a wider shift for the exam model | two six-sheet runs | optional | adopt only if path within 45 gains 5 points or more (6.9, rule 2) |
| 11 | Gentler fine-tuning (backbone factor 0.1 plus one head-only epoch), 2 seeds | robustness to new tracks | two six-sheet runs | optional | the same 5-point rule |
| 12 | 16 epochs against 8, held-out scores every 2 epochs | training length | about twice a six-sheet run per seed | low | change the budget only if the held-out curve still rises at epoch 8 by more than the seed spread |
| 13 | Sheet-balanced sampling weights, frozen probe, 4 seeds | the sampling unit | about 0.3 h | low | end to end only if 3.5 points or more (6.9, rule 1) |
| 14 | Leave-one-block-out inside sheets 02 and 04; an extra fold holding out 05 and 06 together | a known sheet in a new session; the crossings | not estimated | low | read with care (7.1) |
| 15 | Direction weights in the exact loss; mirroring with the exact head | backing up; the stop | | drop | the probe showed a net loss (4.1); mirrored pictures cannot occur (6.2) |

---

## 10. Open questions

### 10.1 For the professor

None of these is on `05_meetings/requests_to_professor.md` in this form; numbers in brackets point to related questions
already on that list, all still open. Meeting 2 was set for 9.11 at 09:50 and no note of it exists yet; if it answered
any of these, record the answer there first. Claude cannot send messages for the team: one of you adds questions 1, 3
and 5 to the list and sends them before meeting 3 on 23.11.

| # | Question | Why it matters |
|---|---|---|
| 1 | **The robot's joints in the controller.** The model sees only the camera picture. To carry out its output (move the tip forward and left, twist the forearm), the controller must use the robot's own joint angles, as every move command does. May it? May it set the wrist and hand to fixed start values and hold them there for the run (which keeps the camera's view close to the training views)? Do a fixed 1-cm step, clamps and a step budget count as rules outside the model? Does the model's own previous output count as an input? (audit 11.3, question 17) | the first set-up needs a yes |
| 2 | **Correction rounds.** After the first model drives, we would stop it where it goes wrong, guide the arm by hand with the motors off, and let the tool replay the correction exactly like a demonstration. Does this count as our own demonstrations? (listed for meeting 2 in the 31.10 decisions note; no answer recorded) | decides steps C and E |
| 3 | **More tracks.** When do the extra training tracks promised on 26.10 arrive (1)? May we print our own tracks of the same size and line thickness, for training and for testing (15)? | the stop and unseen-track tests need varied line ends |
| 4 | **What unseen tracks may contain:** sharper corners, crossings near 90 degrees, line ends far from the paper's edge, close parallel lines, other dash rhythms? (audit 11.3, question 20) | every line end in our six sheets lies 2.35 to 8.03 cm from a paper edge |
| 5 | **The exam run.** How is "finished" judged: the puck centre within what distance of the line's end? Must the robot stop by itself, and does pushing past the end count? When does the clock start and stop? How many attempts per track, and is there a time limit? Which table and which light (19; daylight appears only on sheets 02 and 04)? How is the start set (18)? | decides the stop rule (early against never) and the test's success rule |
| 6 | **Where the model runs** (10): on this laptop, offline? Is the GPU pool usable for training (6; our VPN was not yet connected on 3.11)? | the plan assumes the laptop only |
| 7 | May the exam model be trained on all six provided sheets, with none held out? (audit 11.3, question 19) | the exam model uses all six |
| 8 | May a fixed mask over the robot's own parts in the picture (the same pixels every time) be used as preprocessing? (audit 11.3, question 17) | a guard against shortcut cues |
| 9 | Low priority: may the model see the previous camera picture? Do mirrored copies of our demonstrations count as our own? | only if history or mirroring returns |
| 10 | Low priority: may the controller know an unseen track's line length (to set the step budget at about 4 steps per cm), or must it use a fixed cap of about 250 steps? (targets exam-risk critic) | a budget that is too short ends a good run |
| 11 | Low priority: may a mask computed from the joint angles and the taught corners run at the exam, not only a fixed one (question 8)? (audit 11.3, question 17) | a stronger guard against shortcut cues |

Already answered, no need to ask again: pretrained vision models such as an ImageNet ResNet may be used (question 10,
26.10.26); finishing the track counts first and the finishing time second, on the provided and unseen tracks of the same
size and line thickness (question 8, 26.10.26); we may choose the start and orientation and tape the sheet (question 9a,
26.10.26).

### 10.2 Facts only Emranul and Marshal can confirm (audit 11.3, questions 1 to 7; question 8 added here)

Write the answers into `05_meetings` or tell the Claude session, so that they reach the shared memory.

1. Sheet 02: were 02-01, 02-07, 02-08, 02-11, 02-12, 02-13, 02-14, 02-15, 02-33, 02-34 and 02-35 pushed from end B, as
   their paths and pictures say?
2. Lay-outs: were 05-09 and 05-22 laid turned right by mistake? Was 06-10 unturned or turned left? Was 04-37 left
   unturned? Was 05-24 an offset or a turned start? Were 05-01 and 05-02 laid square on purpose? Was sheet 02 laid
   further out than the start mark on 6.11 and 7.11?
3. What happened in the break on 8.11 between 18:07 and 19:33 (camera re-taped or knocked, block or rod re-mounted, robot
   restarted)? Is the camera still in that later view? A camera check with the sheet left in place would tell a camera
   move from a block move.
4. Was 02-11 try 3 (7.11, 19:01) really recorded in natural light?
5. 06-20: at which picture did the puck tip over? 01-18: are its last rows usable? 01-09: keep its 12 pictures?
6. The 8.11 session notes list 04-01 try 1 at 03:41 with 102 steps, but no such folder exists; was it saved as 03-27?
7. What did meeting 2 (9.11) answer?
8. How many robot hours a week can you book until 20.12.26? The plan needs about 6 to 9 (8.4); a reduced robot plan
   is in 7.4.

### 10.3 Training decisions for the team (audit 11.3, questions 8 to 16)

"Done" and "in the final candidate" mean applied by Claude in the final candidate before you decided; a retrain can
reverse each of them.

| # | Decision | What this analysis proposes | State |
|---|---|---|---|
| 8 | Drop the 449 null rows, or give them the next move? | drop them | done in the cleaning (final candidate) |
| 9 | Relabel the 51 duplicated final points as stop, or drop them? | relabel | done (50; one was in the dropped 05-16) |
| 10 | Targets: exact move plus continuous twist, soft classes, or both? Which rule for the wrist? | exact plus twist first, names plus exact as the backup; wrist and hand held at fixed start values | in the final candidate; the wrist rule is for the executor |
| 11 | Correct the label frame at large forearm turns (2.5 % of moving labels)? | not yet; low priority against the stop | open |
| 12 | Keep the rows just before corrections? Score 1? | keep the 1,186 rows before corrections, except the 21 the recording tool marked as mistakes; keep score 1 | in the final candidate |
| 13 | Step length: predicted and clipped, or fixed? | fixed 1 cm; a rescaled length only as a robot backup | executor rule |
| 14 | Sampling unit: picture, episode or sheet? | picture for now; sheet-balanced weights in the probes first | open |
| 15 | An extra fold holding out sheets 05 and 06 together (the only crossings)? | useful as a check of the crossings | not run |
| 16 | Which session unit for "known sheet, new session" tests? | leave-one-block-out inside sheets 02 and 04, read with care (7.1) | not run |

---

## 11. What was measured, calculated or guessed, and what is still open

### 11.1 Kinds of evidence

- **Measured:** every score of a model on held-out sheets (from its fold files), every count in the data audit, the
  laptop and robot-step timings, the take-over times, the stress test on two fold models.
- **Calculated:** robot hours per trial and per plan step, Wilson intervals and trial numbers, the seeds-needed table,
  the always-forward margins, the data-amount fit and its projected gains, the one-track prediction interval.
- **Literature:** only claims that an independent checker confirmed against their sources (360 of 397 checked claims were
  supported as stated, 36 corrected, 1 source could not be opened; the corrected forms are used).
- **Judgement:** the ranking of the set-ups; the executor rules (fixed 1 cm, twist gain 1, held wrist and hand, the joint-4
  clamp behaviour, the step budget value); the switch thresholds of 2.5; the order of the measures; the success rule's
  2.0-cm tolerance; the plan's order and dates.
- **Guess:** 1 to 2 minutes to change a sheet or an end between trials (the 1 minute to lay the puck and start that the
  validation draft guessed is replaced by the measured 1.86-minute set-up); 1.5 times the steps and 2.5-minute resets
  in the slow case; a 0.15 correlation between trials on one track; 30 % first-model success in a correction round;
  0.5 hours to print, tape and teach a new track; about 2 hours for a first closed-loop check.

### 11.2 What this analysis did not do

- It ran nothing on the robot and changed nothing in `03_data/raw`. No model has driven the robot, so there is no
  closed-loop evidence at all.
- No end-to-end run has a second seed yet (pending), so every end-to-end gap of a few points may be seed noise.
- No exam model trained on all six sheets exists; `train_policy.py` has no "all sheets" option yet.
- The recording tool's model mode does not exist.
- The predicted step length is not saved in the run files; it was re-run only for some folds.
- The stress test (light, colour, shift, roll, blur) covered only the fold models of sheets 01 and 02, and no change
  was scored under the proposed stop rule.
- Not run: leave-one-block-out; the extra fold holding out 05 and 06 together; the near-end-start check; a stop trained
  on a near-goal window end to end; the combined stop rule on the final candidate; mined hard negatives for the stop
  (started in a frozen probe, not finished); an 8-px shift; gentler fine-tuning; sheet-balanced sampling; the chunk
  target with the cleaning; the copy test of a history input; timing of the whole decider on a quiet laptop; the
  puck-off-line, bend-radius and crossing splits for the final candidate.
- The label frame at large forearm turns was not corrected, and the look-by-eye list of the cleaning (about 50 to 70
  rows, plus the audit's lists D1, D2 and D5) was not viewed.
- The design sections, critic notes, judges' notes and scripts, this report's parts and critiques, the queue script and
  the training logs were written in the session's temporary scratchpad; they were copied into `evidence/` at about 01:10
  on 10.11 (`design_review/`, `report_parts/`, `report_check/`, `laptop_queue/`; appendix A.6). The judges' scripts are
  written for named runs (for example `judge2_checks.py` names the final candidate's folder) and still point at the
  scratchpad paths, so they must be pointed at `evidence/` and the new run names before reuse. The fixed two-in-a-row
  stop rule no longer needs them: `compare_runs.py` scores it for every run.

### 11.3 The laptop queue

State at about 01:10 on 10.11, from the queue logs copied to `evidence/laptop_queue/logs/` (`e2e_queue.log`,
`e2e_queue_3.log`, `e2e_queue_4.log`): lines starting "===" give each run's start and finish, the lines after them its
sheets and scores. No number of a pending run is used or guessed in this report.

| Run | Sheets | Rules | State | What it decides |
|---|---|---|---|---|
| `resnet18_finetune_exact` | 01, 02, 04 (its 03, 05, 06 ran 16:14 to 17:26) | old rules: no audit cleaning, best-validation weights | finished 23:01; used in 2.1, 2.4 and A.4 [measured] | completes the earlier exact run on six sheets |
| `resnet18_finetune_mirror` | 03, 05, 06 again | old rules | rerun 23:02 to 00:04: every score equal to the first run's; only the minutes changed | nothing new: it repeats folds already on disk |
| `resnet18_finetune_chunk` | 03, 05, 06 again | old rules | rerun 00:04 to 01:08: every score equal to the first run's | nothing new: it repeated folds already on disk |
| `resnet18_backup_names_plus_exact` | all six | audit cleaning, last weights | running since 01:09 on 10.11, expected by about 03:30 | the backup's offline test (section 2.5) |
| `resnet18_final_candidate_seed_2` | all six | as the first set-up, another seed | queued, expected by about 05:30 on 10.11 | the end-to-end seed spread |
| `resnet18_finetune_no_augmentation` | 03, 05, 06 | old rules, names head (compare with `resnet18_finetune_classes`) | queued, expected by about 06:30 on 10.11 | how much of the end-to-end gain comes from the augmentation |

Why the mirror and chunk runs ran twice: at 17:26 the main session stopped the first queue to put the six-sheet test
first, but only the queue's outer shell stopped; its inner loop went on and trained the mirror run (18:18 to 20:37)
and the chunk run (20:39 to 22:07) beside the final candidate, sharing the processor (the final candidate's epochs took
about 4 minutes, against about 3 for a run alone on the processor). The later queue then ran both again as planned. The same seed gives the same
numbers, so no result is affected; the cost was about 2 hours of processor time. The process could not be stopped from
the session (stopping a process by its number was refused by the safety check). [measured]

Defined in `config.py` but not run, with the reason:

| Run | Why not run |
|---|---|
| `mobilenetv3_finetune_classes`, `efficientnet_b0_finetune_classes` | frozen tests showed no backbone better than ResNet-18, and speed is not a constraint |
| `resnet18_finetune_mixed` | superseded by the backup run with the cleaning (`resnet18_backup_names_plus_exact`) |
| `resnet18_finetune_exact_distance_stop` | the distance stop lost in the frozen test (tuned, two in a row: stopped right 0.15 to 0.16, never stopped 0.75 to 0.79) |

---

## 12. How to use the comparison set-up, and the words used

### 12.1 Running a comparison yourself

Everything runs from `D:\AI_Project\02_code` with the project's Python, once this branch is merged or checked out
there. Only `D:\AI_Project\.venv` exists, so from the worktree folder
`D:\AI_Project\.claude\worktrees\training-analysis\02_code` the same Python is `..\..\..\..\.venv\Scripts\python.exe`
(as the queue script uses). All settings are in `02_code/config.py`, in the dictionaries whose names start with
`TRAINING_`. Nothing touches the robot, and nothing in `03_data/raw` is changed.

| Step | Command (run from `02_code`) | What it does | Time on the laptop |
|---|---|---|---|
| 1 | `..\.venv\Scripts\python.exe training_dataset.py` | Builds one table of all kept pictures with their labels, plus picture caches, in `03_data/processed/v01_training_index/` | about 1 minute |
| 2 | `..\.venv\Scripts\python.exe training_features.py` | Turns every picture into numbers with six pretrained image models (frozen features), and times each model | about 55 minutes |
| 3 | `..\.venv\Scripts\python.exe training_probes.py backbones` | Trains small heads on the frozen features and scores them on held-out sheets; any experiment name of `TRAINING_PROBE_EXPERIMENTS`, `baselines`, or `all` | 2 to 40 minutes per experiment |
| 4 | `..\.venv\Scripts\python.exe train_policy.py resnet18_final_candidate --folds without_path_03` | Trains a whole network on the pictures for one named run of `TRAINING_RUNS`; without `--folds` it does all six held-out sheets | about 20 to 70 minutes per held-out sheet (the long ones while other jobs shared the processor) |
| 5 | `..\.venv\Scripts\python.exe compare_runs.py` | Collects every result into `04_runs/comparison.md` | seconds |
| 6 | `..\.venv\Scripts\python.exe training_error_analysis.py resnet18_imagenet_grid` | Shows where a model goes wrong: per sheet, bends against straight stretches, near the goal, light, camera view | under a minute |
| Test | `..\.venv\Scripts\python.exe test_training.py` | Checks the training code on small made-up examples and against the real labels | under a minute |

To try a new idea, add one entry to `TRAINING_RUNS` (or a variant to an experiment in `TRAINING_PROBE_EXPERIMENTS`)
that names only what differs from the defaults, run it, and run `compare_runs.py`. Every model is scored on the same
pictures with the same measures, so the numbers can be compared directly.

Quick check before a long run (one sheet, one epoch, 512 pictures; results in a folder ending in `_check` that git
ignores):

    ..\.venv\Scripts\python.exe train_policy.py resnet18_finetune_classes --folds without_path_04 --epochs 1 --train-row-limit 512

Notes for this round:

- `04_runs/comparison.md` was built at 18:19 and does not yet include the final candidate, the mirror and chunk runs or
  the new exact folds: run `compare_runs.py` again when the queue is done.
- `comparison.md` sorts by step within 45; read path within 45 first (section 7.2), and use the "same held-out sheets
  only" table when runs cover different sheets. Ignore rows whose split is "episodes" (today the first and third rows,
  0.707 and 0.655): they are optimistic, because held-out episodes have near-twins in training (3.6).
- A run trained in parts keeps only its last part in `metrics.json` (the earlier exact run's file now holds sheets 01,
  02 and 04); the fold files `fold_without_path_XX.json` keep every sheet, and `compare_runs.py` rebuilds from them.
- The fold files store the stop under a fixed 0.5 and under thresholds tuned on validation episodes; the fixed rule with
  two pictures in a row, proposed here, was computed by the judges' read-only scripts from the saved predictions
  (appendix A.6). Adding it to `training_metrics.py` would make it a standard column (a proposal, not done).
- Only one long run at a time when its numbers will decide something: runs that share the processor slow down (one
  mirror fold took 68.2 minutes while sharing, 21.4 when its rerun ran with no other training).

### 12.2 The files

| File (in `02_code`) | What it does |
|---|---|
| `training_dataset.py` | Reads `03_data/raw/main_data` without changing it and builds `03_data/processed/v01_training_index/`: `frames.csv` (one row per picture with its labels, the episode's facts and measured picture statistics), `episodes.csv`, small grey copies of every picture, and two picture caches (224 x 224 and 160 x 120) |
| `training_targets.py` | The rows a model trains on (scores 1 to 3, mistake rows left out, the audit cleaning when switched on), the targets (move class, exact step, chunk of future tip places, forearm twist, stop and near-goal), and the validation splits (one sheet held out; whole episodes held out) |
| `training_metrics.py` | The measures every model gets: push direction against the demonstrated step and against the path direction, class accuracy, the stop per episode (right, early, never), twist, lean and hand signs, chunk errors |
| `training_features.py` | Frozen features of every picture from six backbones (ResNet-18 pretrained and untrained, EfficientNet-B0, MobileNetV3, DINOv2-small, CLIP ViT-B/32), and the time per picture of each |
| `training_probes.py` | Quick comparisons: small heads on frozen features, one experiment per question, all folds, results in `04_runs/2026-11-09_frozen_feature_probes/`; also the rules without a model (`baselines`) |
| `train_policy.py` | Trains a whole network on the pictures (a small network from scratch, or a pretrained one fine-tuned), with label-aware augmentation, fold by fold; results in `04_runs/<date>_<run>/` |
| `training_error_analysis.py` | Where a model goes wrong (sheet, bends, near the goal, forearm turn, light, camera view, score), and what averaging the direction over the last pictures would give |
| `compare_runs.py` | One table of everything run so far: `04_runs/comparison.md` |
| `test_training.py` | Tests on small made-up examples, and a check that the direction rule gives the recording tool's own labels on all real moving rows |

Settings live in `config.py` only, in `TRAINING_DATA_PARAMETERS`, `TRAINING_FEATURE_PARAMETERS`,
`TRAINING_TARGET_PARAMETERS`, `TRAINING_METRIC_PARAMETERS`, `TRAINING_PROBE_PARAMETERS`, `TRAINING_PROBE_EXPERIMENTS`,
`TRAINING_RUN_PARAMETERS`, `TRAINING_RUNS`, `TRAINING_COMPARISON_PARAMETERS` and `TRAINING_ERROR_ANALYSIS_PARAMETERS`.
`config.main_checkout_folder()` lets the code in a git worktree find the data in the main folder. Model weights and
per-picture prediction files stay out of git (`.gitignore` and `04_runs/.gitignore`).

Installed into `.venv` on 9.11.26 (from PyPI, after a dry run showed that nothing already installed would change):
torch 2.15.0 (processor build), torchvision 0.29.2, timm 1.0.31, pandas 3.1.1, scikit-learn 1.9.1, pinned in
`02_code/requirements_training.txt`. Marshal installs the same versions on his Mac with `python -m pip install -r
02_code/requirements_training.txt`; on a Mac, PyTorch can use the Apple GPU ("mps"), which `train_policy.py` does not
use yet.

### 12.3 Glossary

| Word | Meaning here |
|---|---|
| A/B test | Two versions of one setting (for example one picture against two in a row) tried in paired robot runs to see which finishes more often. |
| ACT, diffusion policy | Two published kinds of robot policy: ACT (action chunking with transformers) predicts a chunk of future moves with a transformer network; a diffusion policy builds its move by removing noise step by step. Both are large and slow on a processor; not run here (5.3). |
| AdamW, learning rate, weight decay, batch, warm-up, cosine decay | Training settings in `config.py` that need no change: AdamW is the rule that updates the parameters, the learning rate how big each update is, weight decay a small pull of every parameter towards zero, a batch the 64 pictures used per update, the warm-up a first epoch with a rising learning rate, the cosine decay its smooth fall to zero. |
| Always forward | A rule that always pushes straight up the picture. Any model must beat it clearly, sheet by sheet. |
| Augmentation | Changing training pictures (light, blur, a small shift) so the model copes with changes it will meet, without changing what the right move is. |
| Backbone | The pretrained part of a network that turns a picture into numbers (for example ResNet-18, trained on ImageNet photos; `timm resnet18.tv_in1k` is the exact pretrained file, from the timm library). |
| Backward moves | Moves labelled back, back-left or back-right in the picture's frame; about half are ordinary pushing seen through a sideways-turned camera, some are recovery moves. |
| Best-validation weights, old rules | The runs before the final candidate kept the network as it was at the epoch with the lowest validation loss (best-validation weights) and trained on the uncleaned rows; together these are the "old rules". |
| Chunk | A target of several future steps (here the tip's place after each of the next 5 steps); the robot would still execute one step per picture. |
| Class weights (square-root) | Rare classes count more in the loss, each by the square root of its rarity. |
| Closed loop, offline, open loop | Offline: scoring a model on recorded pictures. Closed loop: the model drives and sees the results of its own moves; only closed loop shows whether it finishes. Open loop: adding up predictions without the robot moving or new pictures arriving. |
| Correction round | The model drives, a person takes over by hand where it goes wrong, and the tool replays the correction as a demonstration. |
| Decider | The part of the recording tool that picks the next step from the latest picture. |
| Design panel, critics, judges | Claude agents, not people: five drafted the design sections, ten critics attacked them, three judges ranked the whole set-ups. |
| DINOv2, CLIP | Two large pretrained picture networks (vision transformers); tested here only as frozen features (5.1). |
| Dithering | Stepping back and forth without progress. |
| End to end (fine-tuned) | The whole network, backbone included, learns from our pictures. Slower on a processor, much better here. |
| Episode-bootstrap interval | A range from re-drawing the held-out episodes many times; it shows the chance of which episodes were tested, not of the training seed. |
| Epoch, pass | An epoch is one training round; here each epoch uses a random half of the training rows. A pass is one look at every training row, so 8 epochs are about 4 passes. |
| Exact step | The demonstrated move as numbers: how far the tip moves forward and left in the picture (cm), and how much the forearm turned (degrees). |
| Exam model | A copy of the first set-up trained on all six sheets with the same recipe; not trained yet. |
| Executor | The rules that turn the model's output into robot moves (step length, clamps, joint solve, guards). |
| Feature map, 3 x 4 grid | The network's last picture-like layer; the head cuts it into 3 x 4 cells so that it knows where things are in the picture. |
| Fixed-wrist solve | Joints 5 and 6 held at fixed start values, joints 1 to 3 computed so the tip moves as asked (`solve_with_wrist_fixed`). |
| Fold | One train-and-test round with one sheet held out; "worst sheet" or "worst fold" is the sheet the model did worst on. |
| Frozen features, probe | The backbone is used as it is and only a small head on top learns. Fast, but the backbone never learns about lines and pucks. (For tables, rules and settings that must not change, the report says fixed or locked.) |
| GPU pool | The university's graphics-card computers; access not yet confirmed. |
| Guards, fallback | Checks before each move: the tip stays inside the sheet in use and at its pushing height, and a counter catches a refused move that keeps coming back. The fallback, what the robot then does, is not decided yet. |
| Head | The last small layers that turn the backbone's numbers into a move, a twist and a stop decision. |
| Held-out sheet (leave one sheet out) | Train on five sheets, test on the sixth, six times. It stands in for an unseen exam track. |
| Held-out episodes | Whole episodes held back from sheets the model trains on; optimistic, because episodes of one sitting are near-copies. |
| Hidden layer, dropout, parameters | A hidden layer is a middle layer of the head (here 256 numbers); dropout switches off a random 20 % of them in training, against learning by heart; parameters are the numbers training adjusts (12.75 million here). |
| Huber loss | A squared-error loss that is gentler on outliers. |
| ImageNet, ImageNet normalisation | A large collection of everyday photos; a network pretrained on it already knows edges, corners and shapes. The normalisation scales a picture's colours with those photos' mean and spread, as the pretrained network expects. |
| Implicit BC, robomimic Square | Published work cited in 2.4: robomimic Square is a simulated pick-and-place benchmark; implicit behaviour cloning picks a move by scoring many candidate moves instead of outputting one. |
| Joints 4, 5 and 6 | The forearm twist (which turns the taped camera), the wrist (rod lean) and the hand. |
| Last weights | The network as it is after the final epoch, not the epoch with the lowest validation loss. |
| Mean direction, most likely class | Two ways to read a names model: the mean direction averages the 8 move directions, each weighted by how likely the model finds it; the most likely class executes the single most likely name. |
| Model mode | The recording-tool mode, still to be built, in which the trained model drives the robot. |
| Move names (classes) | One of 8 fixed directions in the picture (forward, forward-left, ...), or stop. |
| Near-twin | A picture from another episode that looks almost the same as this one (closer than two consecutive pictures usually are). |
| Paired, interleaved, blind trials | Both models run from the same starts, in random order inside one session, scored by someone who does not know which model ran. |
| Path within 45 | The share of pictures where the predicted push points within 45 degrees of where the demonstrator's tip is three steps later. Less noisy than one hand-guided step, and hardly inflated by copying. |
| Picture's frame | Directions measured in the camera picture: up is forward, left is left; every label and output uses it. |
| Point | One percentage point (0.01 of a share). |
| Positive weight 10 | In training, a stop picture counts ten times as much as a non-stop picture in the stop output's loss. |
| Recipe | The training settings of the training-settings draft, now the final candidate's; "the recipe's former rule" was its stop threshold tuned on validation episodes. |
| Recording block, sitting | Episodes recorded in one go; their pictures are near-copies of each other. |
| ResNet-18 | A common 18-layer picture network; here pretrained on ImageNet and then trained on our pictures. |
| Row | One picture with the move recorded after it. |
| Rules hedge | A fall-back in case the professor forbids something. |
| Sealed track | A printed test track kept out of sight and out of training until its one use. |
| Seed | The random start of a training run; two seeds of the same setting give somewhat different results. |
| Sign test, p, power, prediction interval, bootstrap | Statistics checks used in 6.1, 6.9, 7.3 and 8.2; the plain conclusion is always in the sentence beside them. |
| Simulated robot | The recording tool's `--simulated` practice mode: it draws a plain page, a line and a puck, and the real robot does not move. |
| Small motion | A recorded step where the tip barely moved; mostly a forearm twist in place. |
| Step budget | The largest number of steps the executor allows in one run; a safety stop. |
| Step within 45 | The share of moving pictures where the predicted push direction is less than 45 degrees off the demonstrated next step. |
| Stop probability, threshold, two in a row | The stop output's value per picture; the robot stops when it reaches the threshold (here 0.5), on one picture or on two pictures in a row. |
| Stopped right / early / 2 to 3 cm / never | Per held-out episode: the model's first stop is within 2 cm of where the demonstration stopped / more than 3 cm before it / 2 to 3 cm before it / never comes. |
| Targets | What the network is trained to predict. |
| Twist sign | The share of twisting rows where the predicted forearm twist turns the same way as the demonstrated one. |
| Validation episodes | Episodes held back from the training sheets to watch a run; not to be used for choosing anything. |
| Wilson interval | A range for a success rate from few trials; 7 of 10 gives 0.40 to 0.89. |

---

## Appendix

### A.1 Rules without a model

Leave one sheet out, the same pictures as the frozen probes (before the cleaning) [measured; `baselines.json`]:

| Rule | 01 | 02 | 03 | 04 | 05 | 06 | Mean |
|---|---|---|---|---|---|---|---|
| Always forward, step within 45 | 0.747 | 0.565 | 0.522 | 0.485 | 0.509 | 0.402 | 0.538 |
| Always forward, path within 45 | 0.806 | 0.660 | 0.603 | 0.580 | 0.544 | 0.433 | 0.604 |
| Repeat the previous move, step within 45 (reference only) | 0.796 | 0.648 | 0.711 | 0.692 | 0.785 | 0.783 | 0.736 |

Repeat the previous move, path within 45: mean 0.648. On the final candidate's cleaned pictures always forward scores
path within 45 0.801, 0.657, 0.609, 0.582, 0.542 and 0.430 on sheets 01 to 06 [measured; judge 1].

### A.2 Frozen probes

ResNet-18 ImageNet grid features, six held-out sheets, mean of the folds; stop at a fixed 0.5 with one picture
[measured; `04_runs/comparison.md`]:

| Experiment | Variant | Step within 45 | Path within 45 | Stopped right | Stopped early |
|---|---|---|---|---|---|
| Designs | names, small motion as the next move (default) | 0.577 | 0.633 | 0.258 | 0.112 |
| | names, small motion dropped | 0.560 | 0.617 | 0.182 | 0.054 |
| | names, small motion as its own class | 0.560 | 0.620 | 0.207 | 0.059 |
| | exact step | 0.591 | 0.673 | 0.290 | 0.243 |
| | names plus exact (most likely class) | 0.565 | 0.620 | 0.234 | 0.060 |
| | chunk of 5 steps | 0.607 | 0.697 | 0.300 | 0.264 |
| Stop | names, own stop on the stop picture | 0.561 | 0.614 | 0.321 | 0.209 |
| | names, own stop within 2 cm of the goal | 0.571 | 0.626 | 0.138 | 0.658 |
| | exact, own stop within 2 cm of the goal | 0.583 | 0.657 | 0.176 | 0.512 |
| | exact, predicted distance to the goal | 0.595 | 0.670 | 0.110 | 0.013 |
| | names, predicted distance to the goal | 0.562 | 0.619 | 0.092 | 0.015 |
| Splits | exact, held-out episodes (5 folds) | 0.707 | 0.795 | 0.571 | 0.165 |
| | names, held-out episodes (5 folds) | 0.655 | 0.698 | 0.509 | 0.116 |
| History | plus the previous picture | 0.576 | 0.623 | 0.254 | 0.108 |
| | plus the change since the previous picture | 0.603 | 0.640 | 0.278 | 0.159 |
| | plus pictures 2 and 4 back | 0.586 | 0.625 | 0.186 | 0.122 |
| Class balance | no class weights | 0.593 | 0.662 | 0.175 | 0.058 |
| | full inverse weights | 0.509 | 0.552 | 0.292 | 0.212 |
| | square-root weights with label smoothing 0.1 | 0.558 | 0.610 | 0.233 | 0.054 |
| Scores (fewer test pictures; see 3.8 for like-for-like) | scores 2 and 3 | 0.565 | 0.623 | 0.186 | 0.205 |
| | score 3 only | 0.539 | 0.587 | 0.338 | 0.138 |
| | weights 0.4 / 0.7 / 1.0 | 0.573 | 0.627 | 0.306 | 0.121 |
| Head size | linear | 0.498 | 0.540 | 0.171 | 0.280 |
| | 1,024 wide with dropout 0.3 | 0.557 | 0.608 | 0.271 | 0.155 |
| Cleaning (old rules) | mistakes and flagged pictures dropped | 0.560 | 0.619 | 0.196 | 0.096 |
| | nothing dropped | 0.556 | 0.614 | 0.271 | 0.161 |
| | mistakes and still rows dropped | 0.566 | 0.620 | 0.268 | 0.115 |
| Seeds | names, seeds 11 / 12 / 13 | 0.561 / 0.546 / 0.556 | 0.612 / 0.600 / 0.605 | 0.317 / 0.292 / 0.317 | |
| | exact, seeds 11 / 12 / 13 | 0.616 / 0.593 / 0.611 | 0.708 / 0.674 / 0.707 | 0.332 / 0.289 / 0.300 | |

Stop with thresholds tuned on validation episodes, two in a row (right / early / never): names, stop as a class 0.317 /
0.274 / 0.352; names, own stop on the stop picture 0.382 / 0.255 / 0.319; exact, stop on the stop picture 0.221 / 0.414
/ 0.300; exact, stop near the goal 0.318 / 0.276 / 0.310; chunk 0.298 / 0.341 / 0.276 [measured; main session notes].

### A.3 Data amount

Per held-out sheet, step within 45, mean of three seeds [measured; `data_amount.json` via the improvement draft]:

| Training data per fold | 01 | 02 | 03 | 04 | 05 | 06 |
|---|---|---|---|---|---|---|
| 1 sheet, all episodes | 0.588 | 0.404 | 0.486 | 0.468 | 0.442 | 0.350 |
| 2 sheets | 0.663 | 0.437 | 0.533 | 0.507 | 0.500 | 0.404 |
| 3 sheets | 0.724 | 0.452 | 0.523 | 0.498 | 0.483 | 0.443 |
| 4 sheets | 0.740 | 0.437 | 0.550 | 0.542 | 0.545 | 0.378 |
| 5 sheets, 25 % of the episodes | 0.702 | 0.439 | 0.522 | 0.516 | 0.500 | 0.407 |
| 5 sheets, 50 % | 0.739 | 0.493 | 0.538 | 0.520 | 0.547 | 0.437 |
| 5 sheets, 100 % | 0.744 | 0.487 | 0.557 | 0.538 | 0.559 | 0.446 |

Seeds 1, 2, 3 for each row (six-sheet means): 25 % 0.518 / 0.513 / 0.512; 50 % 0.541 / 0.547 / 0.549; 100 % 0.574 /
0.551 / 0.541; one sheet 0.471 / 0.467 / 0.431; two 0.523 / 0.509 / 0.491; three 0.533 / 0.514 / 0.514; four 0.532 /
0.533 / 0.531 [measured].

### A.4 End-to-end runs per held-out sheet

Stop at a fixed 0.5 with one picture, as counts of episodes [measured; `04_runs/2026-11-09_<run>/fold_without_path_XX.json`]:

| Run (rules) | Sheet | Step within 45 | Path within 45 | Twist sign | Right / early / never | Minutes |
|---|---|---|---|---|---|---|
| `small_cnn_scratch_classes` (old) | 03 | 0.635 | 0.669 | 0.718 | 17 / 6 / 20 of 45 | 21.3 |
| | 05 | 0.635 | 0.683 | 0.405 | 1 / 2 / 27 of 30 | 21.7 |
| | 06 | 0.623 | 0.698 | 0.371 | 1 / 0 / 24 of 25 | 24.7 |
| `small_cnn_line_and_puck_classes` (old) | 03 | 0.630 | 0.676 | 0.659 | 17 / 2 / 25 of 45 | 15.8 |
| | 05 | 0.625 | 0.677 | 0.399 | 1 / 5 / 24 of 30 | 17.4 |
| | 06 | 0.623 | 0.696 | 0.398 | 0 / 0 / 25 of 25 | 18.2 |
| `resnet18_finetune_classes` (old) | 03 | 0.663 | 0.704 | 0.703 | 13 / 5 / 26 of 45 | 20.4 |
| | 05 | 0.659 | 0.711 | 0.372 | 4 / 5 / 21 of 30 | 26.7 |
| | 06 | 0.640 | 0.707 | 0.422 | 1 / 1 / 23 of 25 | 28.9 |
| `resnet18_finetune_mirror` (old) | 03 | 0.670 | 0.726 | 0.709 | 26 / 3 / 14 of 45 | 28.4 (rerun 19.2) |
| | 05 | 0.659 | 0.739 | 0.370 | 9 / 8 / 13 of 30 | 42.2 (rerun 21.0) |
| | 06 | 0.599 | 0.669 | 0.470 | 6 / 1 / 17 of 25 | 68.2 (rerun 21.4) |
| `resnet18_finetune_exact` (old) | 01 | 0.759 | 0.858 | 0.675 | 12 / 3 / 4 of 19 | 21.9 |
| | 02 | 0.631 | 0.726 | 0.807 | 21 / 2 / 20 of 44 | 20.6 |
| | 03 | 0.700 | 0.802 | 0.845 | 31 / 7 / 5 of 45 | 24.4 |
| | 04 | 0.635 | 0.729 | 0.758 | 15 / 17 / 4 of 45 | 16.2 |
| | 05 | 0.667 | 0.777 | 0.669 | 6 / 17 / 7 of 30 | 23.5 |
| | 06 | 0.675 | 0.791 | 0.802 | 5 / 1 / 19 of 25 | 22.6 |
| `resnet18_finetune_chunk` (old) | 03 | 0.692 | 0.837 | 0.839 | 26 / 5 / 12 of 45 | 28.2 |
| | 05 | 0.681 | 0.795 | 0.662 | 5 / 18 / 7 of 30 | 28.9 |
| | 06 | 0.638 | 0.770 | 0.778 | 0 / 1 / 24 of 25 | 30.8 |
| `resnet18_final_candidate` (cleaned, last weights) | 01 | 0.779 | 0.856 | 0.677 | 12 / 4 / 1 of 20 | 34.6 |
| | 02 | 0.632 | 0.762 | 0.802 | 24 / 2 / 18 of 45 | 48.8 |
| | 03 | 0.680 | 0.801 | 0.848 | 33 / 9 / 1 of 45 | 55.0 |
| | 04 | 0.642 | 0.735 | 0.773 | 18 / 13 / 4 of 45 | 23.4 |
| | 05 | 0.649 | 0.754 | 0.677 | 12 / 11 / 5 of 29 | 29.9 |
| | 06 | 0.684 | 0.810 | 0.786 | 2 / 2 / 21 of 25 | 31.5 |

Moving pictures per sheet for the final candidate: 578, 2,425, 3,458, 5,264, 2,461 and 1,809; path pictures 529,
2,243, 3,116, 4,502, 2,324 and 1,691; twisting rows 238, 1,195, 2,630, 4,535, 1,713 and 1,430. The old-rules runs used
the uncleaned rows (for example 569 moving pictures and 19 episodes on sheet 01). The earlier exact run on six sheets:
step within 45 mean 0.678, path within 45 mean 0.781 (worst 0.726), twist sign mean 0.759, stop 90 / 47 / 59 of 208
[calculated]. The mirror run was repeated from 23:02 to 00:04 with every score equal; its fold files now hold the
rerun's minutes. Milliseconds per picture (busy laptop): small network 13.5 and 2.5, ResNet-18 names 26.9, exact 21.5
(first part, in `comparison.md`) and 16.7 (second part, in its `metrics.json` now), chunk 27.0, final candidate 37.4.
[measured]

### A.5 The first set-up under every stop rule

The final candidate's saved predictions on all 209 held-out episodes; right / early / 2 to 3 cm / never [measured;
judge 2's re-scoring, which reproduces every fold file's fixed-0.5 counts]:

| Rule | Right | Early | 2 to 3 cm | Never | Right, Wilson 95 % | Early, Wilson 95 % |
|---|---:|---:|---:|---:|---|---|
| fixed 0.5, one picture | 101 | 41 | 17 | 50 | 0.416 to 0.551 | 0.148 to 0.255 |
| **fixed 0.5, two in a row** | **102** | **25** | **13** | **69** | **0.421 to 0.555** | **0.082 to 0.171** |
| fixed 0.4, two in a row | 104 | 31 | 14 | 60 | 0.430 to 0.565 | 0.107 to 0.203 |
| fixed 0.6, one picture | 104 | 32 | 16 | 57 | 0.430 to 0.565 | 0.111 to 0.208 |
| fixed 0.7, one picture | 101 | 24 | 13 | 71 | 0.416 to 0.551 | 0.078 to 0.165 |
| tuned on validation episodes, one picture (0.3 to 0.8) | 105 | 33 | 16 | 55 | 0.435 to 0.570 | |
| tuned on validation episodes, two in a row (0.05 to 0.4) | 98 | 58 | 20 | 33 | 0.402 to 0.537 | |
| chosen on the other five sheets (right minus early) | 86 | 24 | 13 | 86 | 0.347 to 0.479 | |

Per sheet (right / early / 2 to 3 cm / never):

| Sheet | Fixed 0.5, two in a row | Fixed 0.5, one picture | Early stops at 0.5, one picture: median distance before the goal |
|---|---|---|---|
| 01 (20 episodes) | 16 / 3 / 0 / 1 | 12 / 4 / 3 / 1 | 4.4 cm |
| 02 (45) | 19 / 1 / 1 / 24 | 24 / 2 / 1 / 18 | 4.0 cm |
| 03 (45) | 29 / 5 / 3 / 8 | 33 / 9 / 2 / 1 | 4.7 cm |
| 04 (45) | 21 / 9 / 7 / 8 | 18 / 13 / 10 / 4 | 3.7 cm |
| 05 (29) | 14 / 6 / 2 / 7 | 12 / 11 / 1 / 5 | 14.6 cm |
| 06 (25) | 3 / 1 / 0 / 21 | 2 / 2 / 0 / 21 | 20.3 cm (2 stops) |

Backing up and twist per sheet (final candidate) [measured; judge 2]:

| Sheet | Backward moves | Within 45 on backward / other moves | Predicted backward / demonstrated share | Twist gain (least squares) | Twist sign: all twisting rows / where 3 degrees or more is predicted |
|---|---|---|---|---|---|
| 01 | 37 of 578 | 0.054 / 0.828 | 0.002 / 0.064 | 0.65 | 0.676 / 0.723 |
| 02 | 280 of 2,425 | 0.139 / 0.696 | 0.021 / 0.115 | 0.55 | 0.802 / 0.865 |
| 03 | 340 of 3,458 | 0.412 / 0.709 | 0.081 / 0.098 | 0.94 | 0.848 / 0.928 |
| 04 | 523 of 5,264 | 0.300 / 0.680 | 0.062 / 0.099 | 0.96 | 0.773 / 0.851 |
| 05 | 416 of 2,461 | 0.320 / 0.715 | 0.064 / 0.169 | 0.66 | 0.677 / 0.787 |
| 06 | 298 of 1,809 | 0.591 / 0.703 | 0.292 / 0.165 | 1.24 | 0.786 / 0.955 |
| Pooled | 647 of 1,894 (0.342) | others 9,902 of 14,101 (0.702) | | | |

### A.6 Where the evidence is

In the repository (this worktree, `D:/AI_Project/.claude/worktrees/training-analysis/`):

- **Data audit:** `06_deliverables/research_notes/Model and training/evidence/audit_synthesis.md` (finding ids, tables;
  section 11 for the cleaning, the decisions and the open questions) and `evidence/corrected_ends.csv`.
- **Literature:** `06_deliverables/research_notes/Model and training/evidence/literature/<question>.md` with
  `<question>_check.md` for nine questions: action_chunking, action_representation, augmentation,
  diffusion_and_generative, evaluation_statistics, history_and_causal_confusion, interactive_correction,
  offline_vs_closed_loop, pretrained_representations.
- **Run results:** `04_runs/comparison.md` (built 18:19); `04_runs/2026-11-09_frozen_feature_probes/<experiment>.json`
  (backbones, baselines, balance, cleaning, data_amount, designs, history, head_size, scores, seeds, splits, stop,
  stop_distance and the fixed-threshold copies); `04_runs/2026-11-09_<run>/fold_without_path_XX.json` for each sheet of
  each end-to-end run, and `metrics.json` (for a run trained in parts only its last part: the earlier exact run's file
  was rewritten at 23:01 and now holds sheets 01, 02 and 04); per-picture predictions in
  `predictions_without_path_XX.npz` and weights in `checkpoints/` (both kept out of git).
- **Code and settings:** `02_code` (files in 12.2); `02_code/config.py`, `TRAINING_*` dictionaries (the final candidate
  at lines 1797 to 1804, the backup and the second seed at lines 1806 to 1814).
- **Project files used for context:** `05_meetings/requests_to_professor.md` (questions and answers so far), `STATUS.md`
  (Emranul's plan of 6.11 to train on the move names first).

Copied from the session's temporary scratchpad into `06_deliverables/research_notes/Model and training/evidence/` (text
files only; per-picture prediction files left out), at about 01:10 on 10.11:

- **Report parts from the main session:** `report_parts/data_section.md`, `how_to_compare.md`, `code_set_up.md`,
  `robot_time.md`, `professor_questions.md`, `results_so_far.md`, `literature_takeaways.md`; and
  `design_review/main_session_facts.md`.
- **Design panel drafts and critics:** `design_review/<key>/section.md`, `design_review/<key>/critic_evidence/note.md`
  and `design_review/<key>/critic_exam_risk/note.md` for the keys targets, models, training_settings, validation and
  improvement; each folder holds the scripts and printouts its numbers come from (for example
  `design_review/targets/stop_analysis.out.txt`, `design_review/improvement/situation_errors.csv`,
  `design_review/improvement/robot_hours.out.txt`, `design_review/validation/validation_numbers_output.txt`,
  `design_review/training_settings/critic_exam_risk/stress_test_final_candidate_output.txt`).
- **Judges:** `design_review/judge_1/note.md` (with `judge_rescoring.out.txt`, `judge_extra_checks.out.txt`,
  `judge_recovery_checks.out.txt`), `design_review/judge_2/note.md` (with `judge2_checks.out.txt`),
  `design_review/judge_3/note.md` (with `fold_evidence.out.txt`, `extra_pairs.out.txt`, `early_stop_distances.out.txt`).
  The scripts sit beside their printouts (`judge_rescoring.py`, `judge_extra_checks.py`, `judge_recovery_checks.py`,
  `judge2_checks.py`, `fold_evidence.py`, `extra_pairs.py`); they still name scratchpad paths inside.
- **Laptop queue:** `laptop_queue/run_end_to_end_queue.sh` and `laptop_queue/logs/` (`e2e_queue.log`,
  `e2e_queue_3.log`, `e2e_queue_4.log`, one `e2e_<run>.log` per run, the probe logs); the mirror rerun of 23:02
  overwrote `e2e_resnet18_finetune_mirror.log`, and the chunk rerun `e2e_resnet18_finetune_chunk.log`, with identical
  scores.
- **Critiques of this report:** `report_check/evidence.md`, `report_check/beginner.md`, `report_check/completeness.md`.

### A.7 Stress test of two fold models

The training-settings exam-risk critic re-ran the saved final-candidate fold models of sheets 01 and 02 on changed
copies of their held-out pictures; stop at 0.5 with one picture, right / early / never [measured by that critic;
`design/training_settings/critic_exam_risk/stress_test_final_candidate_output.txt`]:

| Change | Sheet 01: step within 45 | Sheet 01: stop (of 20) | Sheet 02: step within 45 | Sheet 02: stop (of 45) |
|---|---|---|---|---|
| none | 0.779 | 12 / 4 / 1 | 0.632 | 24 / 2 / 18 |
| brightness x0.5 | 0.785 | 12 / 4 / 1 | 0.627 | 21 / 10 / 11 |
| brightness x1.5 | 0.772 | 12 / 2 / 4 | 0.630 | 24 / 0 / 19 |
| warm colour cast | 0.775 | 12 / 4 / 1 | 0.633 | 25 / 2 / 16 |
| cool colour cast | 0.777 | 11 / 4 / 1 | 0.640 | 22 / 2 / 20 |
| shift 4 px | 0.791 | 13 / 3 / 3 | 0.638 | 26 / 1 / 17 |
| shift 8 px, right and down | 0.785 | 11 / 4 / 2 | 0.624 | 25 / 4 / 14 |
| shift 8 px, left and up | 0.779 | 13 / 3 / 2 | 0.612 | 25 / 4 / 13 |
| roll 3 degrees | 0.787 | 12 / 4 / 1 | 0.637 | 21 / 2 / 19 |
| blur 1 px | 0.766 | 11 / 4 / 1 | 0.634 | 21 / 2 / 20 |

The re-run reproduced the saved predictions exactly on the clean pictures. Half brightness is darker than any training
picture (natural light reaches down to 0.674 of the median). The printout also scores (0.3, two in a row) and the tuned
rule, but not the proposed (0.5, two in a row).
