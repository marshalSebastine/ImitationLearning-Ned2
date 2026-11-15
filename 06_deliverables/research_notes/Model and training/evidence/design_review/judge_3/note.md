# Judge 3 of 3: training set-ups ranked by the strength of what was measured

Written 9.11.26, about 22:00, for Team M (Emranul Hoque, Marshal Sebastine). Offline only: nothing was run on the robot
and nothing under D:/AI_Project was changed. My angle: only what was measured on our own data counts. A literature claim
is used only where its checker marked it supported (or in its corrected form); everything else is marked [judgement] or
[guess]. "Points" are percentage points.

Words used below. *Step 45*: the share of moving pictures whose predicted push direction lies within 45 degrees of the
demonstrated next step. *Path 45*: the same against the direction to where the tip is three steps later. *Right / early
/ never / 2-3 cm*: episode counts where the first stop came within 2 cm of the demonstrated stop, more than 3 cm before
it, not at all, or 2 to 3 cm before it. *Held-out sheet*: the model was trained on the other five sheets, the stand-in
for an unseen exam track.

## In short

1. **First model: the final candidate network exactly as it is being trained, stopped by one threshold fixed in advance
   (0.5), not by a threshold tuned on validation episodes.** It is the only set-up measured in its final form on more
   than three held-out sheets (five of six are done), it beats always-forward on all five, and on the two sheets it
   shares with the other end-to-end runs it stopped right more often than any of them (45 of 74 episodes, against 37 of
   75 or fewer).
2. **The stop limits finishing, and the stop rule hardly changes that.** Over five held-out sheets (184 episodes) the
   final candidate stopped right in 93 to 103 episodes under all seven rules tried; the rules only move runs between
   "early" and "never". The training-settings draft's rule (threshold tuned on validation episodes, two pictures in a
   row) gave the most early stops, 46 of 184, and no more right stops than a fixed 0.5 (99, with 24 or 39 early).
3. **Backup: the same recipe with the chunk-of-5 target**, executing only the first predicted step, rerun with the
   audit cleaning and the last weights. On the rows both scored it beat the final candidate's path 45 by 4.0 and 3.9
   points on sheets 03 and 05 and backed up better on sheet 05, but stopped right less often (31 against 45 of about 75
   episodes). It has two sheets, no cleaning and one seed, so it backs the first model up; it does not replace it yet.
4. **Not supported by the measurements:** ResNet-18 over the small CNN as a settled choice (read as a mean direction
   the two split by sheet), the class head as the backup (path 45 6.7 to 9.8 points lower, twist sign below 0.5 on both
   loop sheets, the weakest stop), mirroring (sheet 06 lost 4.1 points of step 45) and any picture history (frozen
   probes only).
5. **No end-to-end run has a second seed.** Every gap of a few points between end-to-end runs may be seed noise; in the
   frozen probes the seed alone moved path 45 on one sheet by up to 16 points and the right-stop share by up to 32.

## 1. How I ranked

- Rank 1 is the first model and rank 2 the backup. Ranks 3 and below follow how strongly the measurements support each
  set-up as written [judgement].
- What counts, in this order, for the first grading criterion (finishing): the stop, because an early stop ends a run
  and a run that never stops may not count [judgement; how "finished" is judged is open, audit section 11.3 point 20];
  then path 45, which copying the previous move inflates least (+4.4 points against +19.8 for step 45) [measured:
  main_session_facts.md, baselines]; then the twist sign; then backing up; then how complete the evidence is (sheets,
  final form, cleaning, seeds).
- A gap counts only if it is larger than the seed noise. End-to-end seed noise is not measured. The only floor is the
  frozen exact probe's four seeds: path 45 on one held-out sheet moved by 3.1 to 16.0 points, the right-stop share by
  6.7 to 32.0 points, the six-sheet means by 3.5 and 4.3 points [measured: design/validation/validation_numbers_output.txt
  part 2]. The episode-bootstrap intervals below show only which episodes happened to be held out, not the seed
  [judgement].
- A set-up never run in its own form ranks by its measured parts, below measured set-ups of similar promise [judgement].
- Offline numbers shortlist models; they do not choose between good ones. The checked takeaway is "Offline metrics
  shortlist, not replace" robot tests (Abbas et al. 2026, Sec. 5 heading; offline_vs_closed_loop check claim 28,
  supported), and within one data budget hold-out error and real success were not significantly linked (Cowling 2026,
  Table 3; check claim 23, supported).

## 2. What I measured for this note

Folds 04 (written 21:00) and 05 (21:30) of the final candidate came after every draft and critique, and the chunk run's
folds 03 and 05 after most of them. I re-scored all saved held-out predictions from their npz files and frames.csv with a
read-only script, `judge_3/fold_evidence.py`. It reproduces every stored step 45, path 45, moving-row count and stop share
of the 20 fold files exactly (`fold_evidence.out.txt`, section "STORED CHECK"). On top of that it computes always-forward
on each fold's own rows, backing up, the twist gain, the stop under every rule, a threshold chosen on the other held-out
sheets, and paired differences on shared rows (`fold_evidence.out.txt`, `extra_pairs.out.txt`,
`early_stop_distances.out.txt`). Sheet 06 of the final candidate and of the chunk run was still training.

**Table A. The final candidate per held-out sheet** (audit-cleaned rows, one seed; always-forward on the same rows in
brackets) [measured: R/2026-11-09_resnet18_final_candidate/fold_without_path_01.json to _05.json, fields within_45_deg,
path_within_45_deg, direction_rows, path_direction_rows, twist_sign_accuracy, twist_rows, episodes_stopped_*;
always-forward, backward moves and the 2-3 cm counts from J3/fold_evidence.out.txt]

| Held-out sheet | Episodes | Step 45 (moving pictures) | Path 45 (path pictures) | Twist sign (twisting rows) | Backward moves within 45 | Stop at 0.5, one picture: right / early / never / 2-3 cm |
|---|---|---|---|---|---|---|
| 01 straight | 20 | 0.779 (0.753) of 578 | 0.856 (0.802) of 529 | 0.677 of 238 | 0.054 of 37 | 12 / 4 / 1 / 3 |
| 02 arc | 45 | 0.632 (0.563) of 2,425 | 0.762 (0.657) of 2,243 | 0.802 of 1,195 | 0.139 of 280 | 24 / 2 / 18 / 1 |
| 03 two curves | 45 | 0.680 (0.522) of 3,458 | 0.801 (0.609) of 3,116 | 0.848 of 2,630 | 0.412 of 340 | 33 / 9 / 1 / 2 |
| 04 meander | 45 | 0.642 (0.486) of 5,264 | 0.735 (0.582) of 4,502 | 0.773 of 4,535 | 0.300 of 523 | 18 / 13 / 4 / 10 |
| 05 loop | 29 | 0.649 (0.507) of 2,461 | 0.754 (0.542) of 2,324 | 0.677 of 1,713 | 0.320 of 416 | 12 / 11 / 5 / 1 |
| Five sheets | 184 | mean 0.676 (0.566); worst 0.632 | mean 0.782 (0.638); worst 0.735 | pooled 0.777 of 10,311 | pooled 0.295 of 1,596 | 99 / 39 / 29 / 17 |

- It beats always-forward on all five sheets, by 2.6 to 15.8 points of step 45 and 5.5 to 21.2 points of path 45; the
  smallest margins are on the straight line and the arc, where forward is usually right [calculated from Table A].
- It backs up poorly: 471 of 1,596 backward moves within 45 degrees, against 0.70 of the other moves [calculated].
- Its twist gain is below 1 on every sheet: the least-squares gain of the demonstrated twist on the predicted one is
  0.655, 0.552, 0.935, 0.962 and 0.664 (sheets 01 to 05); where it predicts 3 degrees or more the sign is right on 0.72
  to 0.93 [measured: fold_evidence.out.txt]. The gain of 1 that the targets draft takes from the earlier run (0.97
  pooled) is therefore the upper end for this model, not a measured best [judgement].
- Its 39 early stops at 0.5 lie a median 4.7 cm before the goal. Twelve lie more than 8 cm before it, 8 of them on the
  loop sheet 05 (median 14.6 cm there); all 13 on the meander, sheet 04, lie 3.0 to 6.5 cm before it [measured:
  early_stop_distances.out.txt].
- Training took 23.4 to 55.0 minutes per sheet on the shared laptop [measured: fold files, field minutes].

**Table B. One network, seven stop rules** (the final candidate's same predictions on the 184 episodes of five held-out
sheets) [measured: J3/fold_evidence.out.txt, "POOLED STOP OUTCOMES"; the tuned thresholds are the fold files'
tuned_stop_threshold and tuned_two_stop_threshold]

| Stop rule | Thresholds it used | Right | Early | Never | 2-3 cm |
|---|---|---|---|---|---|
| fixed 0.5, one picture | 0.5 | 99 | 39 | 29 | 17 |
| fixed 0.5, two pictures in a row | 0.5 | 99 | 24 | 48 | 13 |
| threshold chosen on the other held-out sheets, one picture | 0.5 to 0.7 | 93 | 32 | 43 | 16 |
| the same, two in a row | 0.3 to 0.5 | 93 | 33 | 46 | 12 |
| tuned on validation episodes, one picture | 0.3 to 0.8 | 103 | 31 | 34 | 16 |
| tuned on validation episodes, two in a row (training-settings draft) | 0.1 to 0.4 | 93 | 46 | 25 | 20 |
| best fixed rule picked with hindsight on the test sheets | 0.6, one picture | 101 | 31 | 36 | 16 |

- Right stops stay at 93 to 103 of 184 (0.51 to 0.56) under every rule; the 95 % Wilson interval of 99 of 184 alone is
  0.47 to 0.61 [calculated]. No rule can be chosen on right stops.
- The rules differ in early stops (24 to 46). Against the fixed 0.5 with two in a row, the draft's tuned two-in-a-row
  rule had more early stops on four sheets and as many on sheet 01, fewer right stops on two (04: 15 against 21; 05: 12
  against 14), more on one (03: 31 against 29) and as many on two [measured: fold_evidence.out.txt, per-fold counts].
- Choosing the threshold on the other held-out sheets, as the targets draft proposes, landed on 0.5 to 0.7 for one
  picture and did no better than the untuned 0.5 (93 against 99 right) [measured]. With five sheets this confirms the
  targets evidence critic: no measured advantage.
- On sheet 02, pictures at half brightness raised the early stops of the tuned 0.1 two-in-a-row rule from 8 to 21 of 45,
  and of the fixed 0.5 one-picture rule from 2 to 10 [measured by the training-settings exam-risk critic:
  design/training_settings/critic_exam_risk/stress_test_final_candidate_output.txt]. A low tuned threshold makes the
  stop fragile [judgement].

**Table C. Paired differences on the rows both runs scored** (points, first run minus second, 95 % episode-bootstrap
interval in brackets; one seed per run; the path target is the first run's) [measured: J3/fold_evidence.out.txt, "PAIRED
DIFFERENCES"; J3/extra_pairs.out.txt]

| Comparison | Sheet | Step 45 | Path 45 | Backward moves within 45 |
|---|---|---|---|---|
| final candidate minus chunk run | 03 | -1.2 [-2.9, 0.4] | -4.0 [-5.7, -2.4] | -2.1 [-8.6, 3.6] of 340 |
| | 05 | -3.1 [-4.5, -1.8] | -3.9 [-6.0, -1.5] | -9.9 [-14.6, -5.4] of 416 |
| chunk run minus earlier exact run (both on uncleaned rows) | 03 | -0.8 [-1.7, 0.2] | +3.5 [1.8, 5.1] | 0.432 against 0.412 |
| | 05 | +1.3 [-0.3, 2.9] | +1.8 [-0.4, 4.1] | 0.418 against 0.279 |
| final candidate minus earlier exact run (cleaning, last weights, other seed) | 03 | -1.9 [-3.8, -0.3] | -0.7 [-2.8, 1.2] | 0.412 against 0.412 |
| | 05 | -1.8 [-3.3, -0.3] | -2.3 [-4.3, -0.5] | 0.320 against 0.279 |
| exact minus classes, most likely class | 03 / 05 / 06 | +3.6 / +0.8 / +3.5 | +9.8 / +6.7 / +8.4 (all intervals above 0) | 0.412 / 0.279 / 0.527 against 0.579 / 0.659 / 0.762 |
| exact minus classes read as a mean direction | 03 / 05 / 06 | +1.3 / -1.1 / +3.8 | +5.1 [2.9, 7.4] / +2.4 [-0.5, 5.7] / +6.3 [2.8, 9.7] | against 0.556 / 0.678 / 0.765 |
| ResNet-18 classes minus small CNN classes, most likely class | 03 / 05 / 06 | +2.9 / +2.5 / +1.7 | +3.5 / +2.8 / +0.9 | |
| the same, both read as a mean direction | 03 / 05 / 06 | +1.6 / +1.1 / -2.3 [-4.5, -0.1] | +1.9 / +1.9 / -4.4 [-6.8, -2.1] | +5.6 / +15.6 / +1.0 |
| mirrored classes minus plain classes, most likely class | 03 / 05 / 06 | +0.7 / 0.0 / -4.1 [-6.8, -1.5] | +2.2 / +2.9 / -3.8 [-7.0, -0.6] | |

**The stop on the two sheets every end-to-end run shares** (03 and 05; fixed 0.5, one picture; right / early): final
candidate 45 / 20 of 74 episodes (Wilson 0.49 to 0.71); earlier exact run 37 / 24 of 75 (0.38 to 0.60); mirrored
classes 35 / 11; chunk run 31 / 23 (0.31 to 0.53); small CNN classes 18 / 8; ResNet-18 classes 17 / 10 [measured: fold
files and fold_evidence.out.txt; intervals calculated]. With each run's validation-tuned two-in-a-row rule the order
holds: final candidate 43 of 74, against the earlier exact run 39, small CNN classes 37, mirrored classes 36 and chunk
run 32 of 75 [measured].

## 3. The ranking

| Rank | Set-up | Strongest measured support | Main risk |
|---|---|---|---|
| 1 (first) | A: final candidate network, fixed stop at 0.5 | five held-out sheets in final form; beats always-forward on all five; best stop on the shared sheets | the stop (99 of 184 right); backing up (0.295); no second seed, no closed loop |
| 2 (backup) | B: chunk-of-5 target on A's recipe, first step executed | best path 45 measured on sheets 03 and 05 (4.0 and 3.9 points above A, intervals above 0) | its stop (31 against 45 of about 75 right); two sheets, uncleaned rows, one seed |
| 3 | A2: A's network with the validation-tuned, two-in-a-row stop (training-settings draft) | the network is rank 1's | 46 of 184 early stops; thresholds 0.1 to 0.4 tuned on near-twin episodes |
| 4 | C: small CNN from scratch with A's exact-step target, stop output and recipe (models-draft backup) | its class version ties the ResNet-18 class model when both are read as a mean direction | never run with the exact head and its own stop output |
| 5 | D: names plus exact (mixed) head on A's recipe (targets-draft backup) | frozen probe: direction as exact, backing up 9 points better | never run end to end; better backing up did not show on after-correction pictures |
| 6 | E: ResNet-18 class head, square-root class weights, with or without mirroring (training-settings backup) | backs up best (0.58 to 0.76) | path 45 6.7 to 9.8 points lower, twist sign 0.37 and 0.42 on the loops, weakest stop |
| 7 | F: frozen backbone plus head (any of five backbones, any picture history) | four seeds on six sheets, the best-replicated numbers we have | 7.6 to 19.2 points of step 45 below end-to-end training |

### Rank 1, set-up A (first model): the final candidate network, stopped by a fixed 0.5

- **What it is.** timm resnet18.tv_in1k fine-tuned end to end (AdamW, head learning rate 1e-3, backbone 0.3 times
  that, weight decay 0.05, batch 64, one warm-up epoch then cosine to zero), 160 x 120 colour pictures, a 3 x 4 grid head
  with one hidden layer of 256 and dropout 0.2. Target: the exact next step in the current picture's frame (tip forward
  and left in cm, forearm twist in degrees; lean and hand trained but not executed), unweighted Huber loss, and its own
  stop output trained on the 259 stop pictures (positive weight 10). Rows: the audit cleaning (18,032 rows, 209
  episodes), scores 1 to 3 at equal weight. Augmentation as now (shift plus or minus 4 px, brightness, contrast,
  saturation, gamma, channel gain, light blur and noise, no mirror). 8 epochs of a random half of the rows, last weights.
  One picture as input [measured: config.py lines 1797 to 1804; settings in R/2026-11-09_resnet18_finetune_exact/metrics.json,
  which the run inherits].
- **How it stops.** The first picture whose stop probability reaches 0.5, fixed before any robot trial and never tuned.
  One picture or two in a row stays open: offline they tie on right stops (99 and 99 of 184) and trade 15 early stops
  for 19 never-stops [measured: Table B]. On the robot the second look means one more step or a still picture, which the
  offline count does not reproduce, so decide it in paired robot runs [judgement].
- **How it is validated.** Leave one sheet out, six folds, each fold beside its own always-forward rule; the stop as
  counts of right, early, never and 2-3 cm; no threshold tuned on validation episodes; two more seeds before trusting
  any gap under about 4 points of path 45; the exam model trained on all six sheets with the same recipe and the same
  fixed threshold [judgement on Tables A and B].
- **Executor** (from the targets draft and its critics; the model sees only the picture): the direction of the
  predicted vector, a fixed 1.0-cm step, the predicted twist clamped to plus or minus 10 degrees with joint 4 kept within
  plus or minus 115 degrees, wrist and hand held at their start values, joints 1 to 3 solved with the fixed-wrist solver,
  the step budget switched on [judgement; the measured basis is in the targets draft].
- **Why first.** Table A. The design is also replicated by a second training run: the earlier exact run (the same design
  without the cleaning) scored step 45 0.667 to 0.700 and path 45 0.777 to 0.802 on sheets 03, 05 and 06 [measured:
  R/2026-11-09_resnet18_finetune_exact/metrics.json, folds]. On the same rows, end-to-end training beat the frozen exact
  probe by 7.6, 10.4 and 19.2 points of step 45 [measured: validation_numbers_output.txt part 3], and the exact step
  beat the class head on path 45 on all three sheets (Table C). On the shared sheets 03 and 05 it stopped right more
  often than every other run (45 of 74 against 37 of 75 or fewer).
- **Main risk.** The stop: even this best stop leaves 85 of 184 held-out episodes not stopped right (39 early, 29
  never, 17 short by 2 to 3 cm), and sheet 06, where the earlier exact run never stopped in 19 of 25 episodes at 0.5, is
  still training [measured]. A fixed 0.5 needs no tuning set, but the stop probability's scale moved between probe seeds
  (median on the stop pictures 0.30 to 0.65; targets exam-risk critic, from targets/stop_analysis.out.txt), so check it
  on the six-sheet exam model with pictures it never trained on [judgement]. Next, backing up (0.295 of 1,596). Not
  measured at all: closed loop and seeds; in theory a cloned policy's cost can grow with the square of the track length
  (Ross et al. 2011, Theorem 2.1; offline_vs_closed_loop check claim 41, supported).

### Rank 2, set-up B (backup): the chunk-of-5 target on A's recipe

- **What it is.** A's network, input, cleaning, augmentation, schedule and last weights, with the chunk target (where
  the tip is after each of the next 5 steps, in the current picture's frame, plus twist, lean and hand) and the same
  own stop output. The robot executes only the direction of the first predicted step, one step per picture, through A's
  executor, with A's fixed 0.5 stop and A's validation. The run measured so far, resnet18_finetune_chunk, had neither the
  cleaning nor the last weights: it kept the lowest-validation-loss epoch, which was epoch 7 on sheet 05 [measured:
  config.py lines 1699, 1783 and 1801; train_policy.py lines 370 to 381; fold histories].
- **Why backup.** On the rows both scored, its first step beat A's path 45 by 4.0 points on sheet 03 and 3.9 on sheet
  05 (intervals above zero) and backed up better on sheet 05 (0.418 against 0.320 of 416 backward moves); against its
  own uncleaned exact sibling it gained 3.5 and 1.8 points of path 45 at the same step 45 (-0.8 and +1.3) and twist sign
  (0.839 against 0.845; 0.662 against 0.669) [measured: Table C; fold files]. Its path 45, 0.837 and 0.795, is the best
  measured on these two sheets by any run [measured: R/2026-11-09_resnet18_finetune_chunk/fold_without_path_03.json and
  _05.json, path_within_45_deg; the other runs' fold files]. It is the only alternative target measured end to end that is not worse than A on
  direction, and it needs no new executor [judgement].
- **Why not first.** Its stop was worse on both sheets: at 0.5 with one picture 26 against A's 33 right of 45 on sheet
  03 (never 12 against 1) and 5 of 30 against 12 of 29 on sheet 05 [measured: fold files]. Part of A's lead may come with
  the cleaning or the seed, since the uncleaned exact sibling stopped right in 37 of 75 [measured]. Two sheets, one seed.
- **Main risk.** It shares A's backbone, stop output and executor, so it hedges drift and backing up a little but not a
  stop that fails or a ruling against ImageNet weights [judgement].
- **Gate.** Rerun it with the cleaning and last weights on all six sheets. Keep it as backup if its pooled right stops
  under the fixed 0.5 rule are within about 7 points of A's (the probes' six-fold stop range reached 7.1 points;
  training-settings draft section 1). If it also keeps a path 45 lead above about 3.5 points on most sheets, it becomes
  the first model and A the backup [judgement].

### Rank 3, set-up A2: A's network with the validation-tuned, two-in-a-row stop (the training-settings draft as written)

- Measured: 93 right, 46 early, 25 never and 20 short of 184, against 99 / 24 / 48 / 13 for a fixed 0.5 with two in a
  row (Table B). The thresholds ran from 0.1 to 0.4 across folds and are tuned on validation episodes that are near-twins
  of training episodes (audit duplicates_leakage-3); the audit asks for settings chosen on sheet or block folds instead
  (audit section 11.2 point 4) [measured: fold files; audit].
- Why third: the network is rank 1's and only its rule is measured worse, so it is neither the first model nor a backup.
  It stays above C to F because even with this rule it stopped right more often than any other network measured
  [judgement on the shared-sheet counts].
- Main risk: one early stop in four held-out episodes, and a stop that darkness breaks (Table B, last point).

### Rank 4, set-up C: the small CNN from scratch with A's exact-step target, stop output and recipe

- Measured only in its class version (sheets 03, 05, 06, one seed, uncleaned rows). Read as the most likely class it
  trailed the ResNet-18 class model by 1.7 to 2.9 points of step 45; read as a mean direction the two split by sheet:
  ResNet-18 ahead by 1.6 and 1.1 points of step 45 on sheets 03 and 05, the small CNN ahead by 2.3 points of step 45 and
  4.4 of path 45 on sheet 06 (Table C). With a threshold from the other sheets and two pictures in a row its stop class
  stopped right in 47 of 100 episodes, against 30 for the ResNet-18 class model; it backed up less well (0.494, 0.526,
  0.728) [measured: fold_evidence.out.txt]. The two runs also differed in schedule (16 against 8 epochs, learning rate,
  kept epochs 15 to 16 against 5 to 7) [measured by the models evidence critic: design/models/critic_evidence/note.md].
- Literature: on robomimic Square with 200 demonstrations a ResNet-18 trained from scratch scored 0.94 against 0.92
  fine-tuned and 0.58 frozen (Chi et al. 2023, Table 5; pretrained_representations check claim 37, supported).
- Role: if the professor rules out ImageNet weights, this becomes the first model, after it passes A's six-fold test.
  Otherwise its case rests on the backbone tie [judgement].
- Main risk: the exact head and the stop output were never trained on it, so its stop is unknown.

### Rank 5, set-up D: names plus exact (mixed) head on A's recipe

- What it is: A with a second head of 8 move names (square-root class weights); the robot takes the direction from the
  names head's probability-weighted mean direction and the twist from the exact head, with A's own stop output.
  Configured as resnet18_finetune_mixed (config.py line 1611), never run end to end [measured: config].
- Measured parts: in the frozen probe this design, read as a mean direction, scored step 45 0.602 and path 45 0.677
  against 0.591 and 0.673 for the exact probe (the exact probe's four seeds span 0.591 to 0.616 and 0.673 to 0.708),
  backward moves 0.363 against 0.270, twist sign 0.699 against 0.700 [measured: R/2026-11-09_frozen_feature_probes/designs.json;
  targets/per_class_direction.out.txt via the targets draft]. End to end, a separate class model read as a mean
  direction backed up far better than the exact model (0.556 / 0.678 / 0.765 against 0.412 / 0.279 / 0.527) but lost
  5.1, 2.4 and 6.3 points of path 45 (Table C).
- Against it: on the 2,921 moving pictures that follow a hand correction on sheets 03, 05 and 06, the nearest thing to
  recovery pictures in our data, the class model and the exact model scored the same step 45 (0.605 and 0.605), though
  the class model recalled 284 of 516 backward moves and the exact model 146 [measured:
  design/training_settings/critic_exam_risk/recovery_stratum_output.txt; validation_numbers_output.txt part 4]. Better
  backing up did not show as better recovery offline.
- Main risk: unmeasured end to end; the names head's direction cost path 45 in the separate class model.

### Rank 6, set-up E: ResNet-18 class head with square-root class weights, with or without mirroring

- Measured (sheets 03, 05, 06, one seed, uncleaned rows): path 45 6.7 to 9.8 points below the exact model on all three
  sheets, step 45 0.8 to 3.6 below; twist sign 0.703 / 0.372 / 0.422 against 0.845 / 0.669 / 0.802, below a coin toss
  on both loop sheets; the stop as a class stopped right in 18 of 100 at 0.5 and 31 with thresholds from the other
  sheets, against 42 and 43 for the exact model's own stop output; it backs up best (0.579 / 0.659 / 0.762) [measured:
  metrics.json of both runs; fold_evidence.out.txt]. Mirroring raised the stop at 0.5 to 41 of 100 (36 and 40 with
  thresholds from the other sheets, one and two pictures) but lost 4.1 points of step 45 and 3.8 of path 45 on sheet 06
  [measured: Table C; fold_evidence.out.txt].
- Main risk: a camera that turns the wrong way on loops, and the weakest stop of the end-to-end runs.

### Rank 7, set-up F: a frozen backbone with a trained head

- Measured with four seeds on six sheets: the exact probe reached step 45 0.591 to 0.616, path 45 0.673 to 0.708 and
  stopped right in 0.289 to 0.332 of 208 episodes at 0.5; the class probes of five pretrained backbones reached 0.545 to
  0.577, against 0.538 for always-forward [measured: comparison.md]. On the same rows the end-to-end exact model beat
  the frozen exact probe by 7.6, 10.4 and 19.2 points of step 45 (above). A picture history (the change since the
  previous picture) raised the six-sheet step 45 from 0.577 to 0.603 but lowered the worst sheet from 0.467 to 0.454 and
  the worst path sheet from 0.526 to 0.498, and was never trained end to end [measured: comparison.md, history].
- Main risk: hardly better than pushing straight ahead on the straight line and the arc.

## 4. Decisive evidence

1. The final candidate is the only set-up measured in its final form on five held-out sheets, and it beat always-forward
   on all five by 2.6 to 15.8 points of step 45 and 5.5 to 21.2 points of path 45 (Table A).
2. Over those 184 episodes, seven stop rules gave 93 to 103 right stops but 24 to 46 early ones; the validation-tuned
   two-in-a-row rule gave the most early stops (Table B). The stop, not the rule, limits finishing.
3. On the shared sheets 03 and 05 the final candidate stopped right in 45 of 74 episodes, against 37 of 75 or fewer for
   every other end-to-end run (Table C note).
4. On the same rows, end-to-end training beat frozen features by 7.6 to 19.2 points of step 45, and the exact step beat
   the most likely class by 6.7 to 9.8 points of path 45 on three of three sheets (Table C): the two firm design facts.
5. On the same rows, the chunk run's first step beat the final candidate's path 45 by 4.0 and 3.9 points, with a worse
   stop (Table C): the strongest alternative, not yet settled.
6. Read as a mean direction, the ResNet-18 and the small CNN split by sheet (+1.9, +1.9, -4.4 points of path 45): the
   backbone is not settled by our data (Table C).
7. The exact designs back up poorly (0.295 of 1,596 backward moves for the final candidate) while class readings reach
   0.556 to 0.765, yet the two tie on after-correction pictures (0.605 against 0.605 of 2,921).
8. No end-to-end run has a second seed; the probes' seed ranges per sheet (path 45 up to 16.0 points, right stops up to
   32.0) are wider than most end-to-end gaps.

## 5. What would change the ranking

- **Sheet 06 of the final candidate and of the chunk run** (both training). If the final candidate never stops there,
  as the earlier exact run did in 19 of 25 episodes, its pooled right share falls to about one half; the order changes
  only if the chunk run stops better on sheet 06 [judgement].
- **The chunk run rerun with the cleaning and last weights on six sheets.** If its right stops under the fixed 0.5 rule
  are within about 7 points of A's and its path 45 lead stays above about 3.5 points on most sheets, B becomes first and
  A the backup; if its stop stays 10 or more points below A's, B drops below C [judgement].
- **Two more seeds of A and B.** If the per-sheet spread of path 45 between seeds exceeds the A to B gap (about 4
  points on sheets 03 and 05), the two are tied and A, the simpler and complete one, stays first [judgement].
- **The professor's answer on ImageNet weights.** If they are ruled out, C becomes the first model (after its six-fold
  run), and B and D move onto the small CNN too.
- **The exam's rule for "finished".** If pushing past the line end still counts, never-stops cost less than early
  stops and the fixed 0.5 with two pictures in a row (24 early) becomes the preferred rule; if the robot must stop
  within about 2 cm, early and never both fail and the rules tie (93 to 103 right).
- **The first robot smoke tests (after the team decides).** If A fails mainly by not recovering a puck that slid off
  the block, D moves above B as the backup, because the names head read as a mean direction backed up 0.556 to 0.765
  against 0.279 to 0.527 on the same rows; if both A and B fail mainly at the stop, no set-up here fixes it and new stop
  data matter more than the order.
- **A stop variant run end to end** that beats A's pooled right stops by more than about 7 points, or cuts early stops
  without losing right ones (for example the near-goal window target, which beat the stop-picture target in the frozen
  probes with tuned thresholds: 0.318 against 0.221 right, 0.276 against 0.414 early; comparison.md), would re-rank
  within A's family.
- **The mixed design run end to end.** If its backward moves reach 0.5 or more pooled with path 45 within about 3
  points of A's and a stop no worse, D rises above C and competes with B for the backup.

## 6. Where I differ from the drafts and their critics

- **Training-settings draft:** drop the validation-tuned, two-in-a-row stop. With five sheets it gives the most early
  stops (46 of 184) and no more right stops (Table B); its critics' warning now rests on 184 episodes instead of 65.
- **Targets draft:** a threshold chosen on the other held-out sheets has no measured advantage over an untuned 0.5 (93
  against 99 right); and the twist gain is below 1 on all five sheets (0.55 to 0.96), so gain 1 is a judgement and a
  blanket upscaling of confident twists has no support.
- **Models draft:** the ResNet-18's lead over the small CNN depends on how the class head is read (Table C), so "best
  wherever it was tested" overstates it; I keep the small CNN as the fallback for an ImageNet ruling, not as the backup.
- **Validation draft:** its measure 3 (the stop under the tuned two-in-a-row rule) should be the stop under one fixed
  rule, reported beside the other rules of Table B.
- **The chunk run:** the targets draft's bar (beat the exact run by more than about 3 points of path 45) is met on sheet
  03 (+3.5) and not on sheet 05 (+1.8); against the final candidate itself the margin is 4.0 and 3.9. That earns it the
  backup place and a rerun in final form, not the first place.

## Sources

- **J3:** `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/judge_3/`:
  `fold_evidence.py` writes `fold_evidence.out.txt` and `fold_evidence.out.json`; `extra_pairs.py` writes
  `extra_pairs.out.txt`; `early_stop_distances.out.txt` summarises the early stops from the json. They read the npz and
  fold files under R and `D:/AI_Project/03_data/processed/v01_training_index/frames.csv`, import no project code and write
  only into J3.
- **R:** `D:/AI_Project/.claude/worktrees/training-analysis/04_runs/`: comparison.md;
  2026-11-09_resnet18_final_candidate/fold_without_path_01.json to _05.json; 2026-11-09_resnet18_finetune_chunk/fold_without_path_03.json
  and _05.json; metrics.json of 2026-11-09_resnet18_finetune_exact, _finetune_classes, _finetune_mirror and
  _small_cnn_scratch_classes; 2026-11-09_frozen_feature_probes/designs.json; all predictions_without_path_*.npz.
- **Code:** `D:/AI_Project/.claude/worktrees/training-analysis/02_code/config.py` lines 1611, 1699, 1783, 1797 to 1804;
  `train_policy.py` lines 370 to 381; `training_metrics.py` stop_metrics and tuned_stop_threshold (lines 121 to 240).
- **Drafts and critics:** `.../scratchpad/design/<key>/section.md` for targets, models, training_settings, validation
  and improvement; `design/validation/validation_numbers_output.txt` parts 2 to 4;
  `design/training_settings/critic_exam_risk/stress_test_final_candidate_output.txt` and `recovery_stratum_output.txt`;
  `design/models/critic_evidence/note.md`; `design/targets/critic_evidence/extra_checks.out.txt`.
- **Audit:** `.../scratchpad/audit/AUDIT_SYNTHESIS.md`, section 11 (lines 1129 to 1277).
- **Literature, used only where supported:** `.../scratchpad/literature/offline_vs_closed_loop/check/note.md` (claims
  23, 28, 41); `.../literature/pretrained_representations/check/note.md` (claim 37).
- **Session facts:** `.../scratchpad/design/main_session_facts.md`.
