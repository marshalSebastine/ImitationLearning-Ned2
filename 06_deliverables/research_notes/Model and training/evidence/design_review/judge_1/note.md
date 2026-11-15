# Judge 1: which training set-up first, and which as backup (angle: finishing unseen tracks above all)

Written 9.11.26 (22:15) for Team M (Emranul Hoque, Marshal Sebastine). Offline only: nothing ran on the robot and
nothing under D:/AI_Project was changed. Markers: [measured] (file and field, or my re-count of saved predictions),
[calculated], [literature] (a claim its checker marked supported or corrected; the corrected form is used),
[judgement], [guess]. "Points" are percentage points. Episode counts are right / early / never unless said otherwise
("right": first stop within 2 cm of where the demonstration stopped; "early": more than 3 cm before it; "near": 2 to 3
cm before it; "never": no picture passes).

## Short answer

- **First set-up (S1): the final candidate as it is trained now.** Exact step (tip forward and left in cm, forearm
  twist) plus its own stop output trained on the 259 cleaned stop pictures; ImageNet ResNet-18 fine-tuned end to end on
  one 160 x 120 picture; the audit cleaning; 8 epochs on half the rows, last weights; validated leave-one-sheet-out.
  Two changes that need no retraining: stop at a **fixed threshold 0.5 with two pictures in a row** (a step of about
  1 cm between the two looks), not the threshold tuned on inner validation episodes; and the drafted executor rules
  (direction of the predicted step, a fixed 1-cm step, twist at gain 1 clamped, joint 4 within plus or minus 115 deg,
  step budget on).
- **Backup (S3): names-plus-exact on the same recipe.** The 8-name head with square-root class weights, read as its
  probability-weighted mean direction, gives the push direction; the exact head gives the twist; the same own stop
  output. Train it now (about 3.7 laptop hours for six folds [calculated from S1's fold minutes]) and score it on the
  six held-out sheets before any robot time.
- **Why, in three numbers.** (1) S1 is the only end-to-end set-up scored on all six held-out sheets, and on the three sheets every end-to-end run shares
  (03, 05, 06) it has the most right stops: 47 of 99 at (0.5, one picture), against 42 of 100 for the uncleaned exact
  run, 41 for the mirrored class head and 31 for the chunk run [measured; judge_rescoring.out.txt]. (2) Its path
  direction on all six held-out sheets is 0.735 to 0.856, 5.5 to 38.0 points above always forward on the same rows
  [measured; fold_without_path_01 to _06.json, path_within_45_deg]. (3) Its clearest weakness after the stop is
  recovery: in the first five rows after a hand correction on sheets 03, 05, 06 it points within 45 deg of the
  demonstrated step in 151 of 500 moving pictures, barely above always forward (138), while the class head read as a
  mean direction gets 216 of 500 [measured; judge_recovery_checks.out.txt]. S3 is the cheapest set-up aimed at that gap.
- **Caveat that outweighs the ranking.** The stop caps finishing for every set-up: S1 stops right in 102 of 209
  held-out episodes (0.49; Wilson 0.42 to 0.56) at its best untuned rule, and its never-stops sit at line ends far from
  the paper edge and on the dashed sheet. No training set-up on these six sheets fixes this [judgement on the measured
  pattern; in the targets section's tests on the frozen probes and the earlier exact run, even the best threshold for
  each held-out sheet stopped right in at most 0.43 of episodes, and extra loss weight on look-alike pictures did not
  help, section 2.6]; new printed tracks with varied line ends may, if the professor allows them [judgement].

## What I did

- Read main_session_facts.md, AUDIT_SYNTHESIS.md section 11, comparison.md, all five drafted sections and their ten
  critic notes, the code-reading note (summary and section f), the literature checks I cite, config.py TRAINING_* settings, and the
  scoring code (training_metrics.stop_metrics, training_probes.fold_measures and path_directions,
  training_targets.chunk_targets and audit_cleaned_frames).
- Read the fold files that finished after the drafts were written: the final candidate's sheets 04 (21:00), 05 (21:30)
  and 06 (about 22:00), the mirror run's sheet 06 (20:37), and the chunk run's sheets 03 (21:07), 05 (21:36) and 06
  (22:07).
- Wrote three read-only scripts in this folder that join the saved test predictions to frames.csv by frame_index. They
  reproduce every fold file's within_45_deg and path_within_45_deg exactly, for every run and sheet:
  - judge_rescoring.py (judge_rescoring.out.txt and .out.json): direction and strata, stop outcomes under fixed,
    inner-validation and leave-one-sheet-out rules, stop by finishing end, where early stops fire;
  - judge_extra_checks.py (judge_extra_checks.out.txt): error streaks per sheet, twist calibration, stop by line end
    under fixed rules;
  - judge_recovery_checks.py (judge_recovery_checks.out.txt): backward moves with a straight forearm, and the first
    five rows after a hand-correction restart.

## The set-ups ranked

| Rank | Set-up | What it predicts and how it stops | Backbone, input | State |
|---|---|---|---|---|
| 1 | **S1 final candidate** (targets and models drafts' main set-up; training-settings recipe with the stop rule changed) | exact step plus own stop output on the 259 cleaned stop pictures; stop at fixed 0.5, two in a row | ResNet-18 ImageNet, fine-tuned; one picture | six folds done, one seed |
| 2 | **S3 names-plus-exact** (targets draft's backup, critics' candidate two) | 8 names with square-root weights read as mean direction, plus exact twist, plus the same own stop output | as S1 | probes only; needs one six-fold run |
| 3 | **S6 chunk of 5** (targets draft's conditional option) | the tip's place after each of the next 5 steps, one step executed per picture, plus exact twist and own stop | as S1 | 03, 05, 06 done, uncleaned rows, best-validation weights |
| 4 | **S5 small CNN from scratch, exact plus stop** (models draft's backup) | as S1 | 1.61 M-parameter CNN from scratch | only the 8-name version exists (03, 05, 06) |
| 5 | **S4 8-name head with square-root weights and mirroring** (training-settings draft's backup) | 8 names, twist as a 3-way class, stop as a class | as S1, half the pictures mirrored | 03, 05, 06 done, uncleaned rows |
| 6 | **S8 one picture plus the change since the previous picture** | 8 names | frozen ResNet-18 features | frozen probe only |
| 7 | **S7 frozen ResNet-18 features plus a head** | exact or 8 names | frozen | six folds, four seeds |

Validation for every set-up: leave one sheet out (six folds) on the direction, stop, twist and recovery measures below,
then (after the team decides, with a checklist) robot runs of fold checkpoints on their own held-out sheets.

## Evidence that decides

### 1. The final candidate on all six held-out sheets

| Held-out sheet | Episodes | Step 45 (moving pictures) | Path 45 (path pictures) | Always forward, path 45, same rows | Twist sign (twisting rows) |
|---|---|---|---|---|---|
| 01 straight | 20 | 0.779 (578) | 0.856 (529) | 0.801 | 0.676 (238) |
| 02 arc | 45 | 0.632 (2,425) | 0.762 (2,243) | 0.657 | 0.802 (1,195) |
| 03 two curves | 45 | 0.680 (3,458) | 0.801 (3,116) | 0.609 | 0.848 (2,630) |
| 04 meander | 45 | 0.642 (5,264) | 0.735 (4,502) | 0.582 | 0.773 (4,535) |
| 05 loop | 29 | 0.649 (2,461) | 0.754 (2,324) | 0.542 | 0.677 (1,713) |
| 06 dashed loop | 25 | 0.684 (1,809) | 0.810 (1,691) | 0.430 | 0.786 (1,430) |
| Mean of six | 209 | 0.678 | 0.786 (worst 0.735) | 0.604 | |

[measured; 04_runs/2026-11-09_resnet18_final_candidate/fold_without_path_0X.json, fields within_45_deg,
path_within_45_deg, twist_sign_accuracy, twist_rows; always forward from judge_rescoring.out.txt]. Sheet 04 is the first
end-to-end result on the meander (frozen exact probe there 0.548 step 45 [measured; validation section table 1]). One
seed; 23.4 to 55.0 minutes per fold, 223 minutes for six with other jobs sharing the processor [measured; fold files,
minutes].

### 2. The stop decides more than any set-up difference

**Stop rules on S1, all 209 held-out episodes** [measured; judge_rescoring.out.txt]:

| Rule | Right | Early | Near | Never | Right (Wilson 95 %) |
|---|---|---|---|---|---|
| fixed 0.5, one picture | 101 | 41 | 17 | 50 | 0.48 (0.42 to 0.55) |
| **fixed 0.5, two in a row** | **102** | **25** | 13 | 69 | 0.49 (0.42 to 0.56) |
| fixed 0.3, two in a row | 101 | 40 | 14 | 54 | 0.48 |
| tuned on inner validation, one picture | 105 | 33 | 16 | 55 | 0.50 (0.44 to 0.57) |
| tuned on inner validation, two in a row (training-settings recipe) | 98 | **58** | 20 | 33 | 0.47 (0.40 to 0.54) |
| chosen on the other five held-out sheets, one picture | 99 | 30 | 16 | 64 | 0.47 |
| chosen on the other five, two in a row | 97 | 37 | 13 | 62 | 0.46 |
| one rule chosen with hindsight on all six (0.4, two in a row) | 104 | 31 | | 60 | optimistic ceiling |

- No rule separates the right counts (97 to 105 of 209, inside each other's intervals). The rules differ in early
  against never stops. An early stop ends a run unfinished; a never-stop may or may not count, depending on the exam
  rule nobody has asked about yet (audit 11.3 question 20) [judgement].
- The inner-validation two-in-a-row rule gives the most early stops (58 of 209), because its thresholds fall to 0.05
  to 0.1 on four of six folds (0.4, 0.1, 0.1, 0.1, 0.4, 0.05) [measured; fold files, tuned_two_stop_threshold]. Reject
  it. The untuned (0.5, two) is within 3 right stops of every rule and has the fewest early stops of all rules with
  95 or more right stops (a rule chosen on the other sheets to maximise right minus early had 24 early but only 87
  right and 85 never) [measured].
- Leave-one-sheet-out choice did not beat the untuned 0.5 (99 against 101 and 102 right) [measured]. Its value is as a
  safeguard against tuning on near-twin episodes (audit 11.2 point 4), not a measured gain.

**Where S1's stop fails** [measured; judge_rescoring.out.txt, judge_extra_checks.out.txt part 3]:

- Early stops moved to the end. Under the leave-one-sheet-out one-picture rule S1 stops early in 30 of 209 episodes, a
  median 4.5 cm before the goal; 24 of the 30 fire 3.0 to 7.1 cm before it and 6 fire more than 8 cm before it (four of
  them on the loop, sheet 05, 14.6 to 16.7 cm). The uncleaned exact run's 26 early stops on 03, 05, 06 fired a median
  14.3 cm before the goal, 17 of them more than 8 cm before.
- Never-stops pile up where the line end is far from the paper edge or dashed. At (0.5, one picture), 44 of S1's 50
  never-stops are on sheet 02 (18 of 45; ends 6.2 and 6.4 cm from the edge), sheet 05 end B (5 of 8; 6.0 cm) and sheet 06
  (21 of 25; the dashed loop, ends 4.4 and 8.0 cm). Line ends 5 cm or more from the edge: never 34 of 89; nearer ends:
  16 of 120 (edge distances from audit table 10.1, as used in improvement/stop_by_end.py).
- My reading [judgement]: the stop fires when the paper edge comes to a learned distance ahead, early on the meander
  whose ends lie 2.4 to 2.8 cm from the edge and not at all where the end lies 6 cm or more from it. This matches the
  audit's cue (every training line end 2.35 to 8.03 cm from a paper edge, stop pictures showing no line ahead; audit
  11.2 point 13). Sheet and edge distance are confounded in these data.

**Stop on the sheets every end-to-end run shares (03, 05, 06)** [measured; judge_rescoring.out.txt]:

| Set-up (run) | (0.5, one picture) | (0.5, two in a row) |
|---|---|---|
| S1 final candidate | 47 / 22 / 27 of 99 | 46 / 12 / 36 of 99 |
| exact, uncleaned rows (finetune_exact) | 42 / 25 / 31 of 100 | 36 / 23 / 40 |
| S6 chunk of 5, uncleaned | 31 / 24 / 43 | 23 / 15 / 61 |
| S4 class head plus mirroring (stop as a class) | 41 / 12 / 44 | 35 / 7 / 57 |
| class head without mirroring (stop as a class) | 18 / 11 / 70 | 10 / 8 / 81 |
| small CNN, class head (stop as a class) | 19 / 8 / 71 | 19 / 4 / 76 |

S1 has the most right stops under both rules. With two in a row it has 12 early stops, against 23 and 15 for the
other exact-type runs; the class-head runs have 4 to 8 early stops but stop right in only 10 to 35 of 100, because
their stop rarely fires at all (57 to 81 never). The cleaned rows are the likeliest reason S1 beats the earlier exact run
(the 50 duplicated final points no longer teach "not a stop" on the stop picture) [judgement]; the two runs also differ
in seed, so the size of the gain is uncertain.

### 3. Direction on normal pictures and on recovery pictures

On sheets 03, 05, 06 [measured; judge_rescoring.out.txt, judge_recovery_checks.out.txt, judge_extra_checks.out.txt
part 2; class heads read as their mean direction]:

| Set-up (run) | Path 45, all rows, mean of 3 sheets | First 5 rows after a hand correction: step 45 / path 45 | Backward moves with a straight forearm (under 30 deg) | Twist sign 03 / 05 / 06 |
|---|---|---|---|---|
| S1 final candidate | 0.788 | 151 of 500 / 234 of 477 | 72 of 218 (0.33) | 0.848 / 0.677 / 0.786 |
| exact, uncleaned | 0.790 | 169 / 260 of 556 | 66 (0.30) | 0.845 / 0.669 / 0.802 |
| S6 chunk of 5 | 0.801 | 164 / 300 of 556 | 92 (0.42) | 0.839 / 0.662 / 0.778 |
| class head (square-root weights), mean direction | 0.744 | 216 / 318 of 556 | 155 (0.71) | 0.703 / 0.372 / 0.422 |
| S4 class head plus mirroring, mean direction | 0.754 | 221 / 335 of 556 | 148 (0.68) | 0.709 / 0.370 / 0.470 |
| small CNN class head, mean direction | 0.746 | not computed | not computed | 0.718 / 0.405 / 0.371 |
| always forward | 0.527 | 138 / not computed | 0 | |

- The exact-type outputs lead over all path pictures (0.788 to 0.801 against 0.744 to 0.754) and on the twist, which
  keeps the forearm camera on the line. Class twist heads get the sign right on only 0.37 to 0.47 of twisting rows on
  the loops [measured].
- Right after a correction the picture is reversed: the class head read as a mean direction leads S1 by 13 points of
  step 45 (216 against 151 of 500) and by about 8 points of path 45 (0.57 against 0.49; S1's cleaned rows give 477 path
  pictures, the others 556); with the forearm straight it backs up within 45 deg on 0.71 of moves against 0.33. Over all six sheets S1 gets 415 of 1,124 in these rows (always forward 294) and 102
  of 458 straight back-ups (0.22) [measured; judge_recovery_checks.out.txt].
- Errors come in streaks for every set-up: episodes with 5 or more path pictures in a row more than 45 deg off are
  25 of 45 / 20 of 29 / 10 of 25 for S1 on 03 / 05 / 06, and 18 to 30 of 45, 21 to 24 of 30 and 15 to 19 of 25 for the
  other runs (always forward 36, 24, 25) [measured; judge_extra_checks.out.txt part 1]. Streaks do not separate the
  set-ups much, except S1's 10 of 25 on sheet 06.
- Why recovery matters for finishing: errors compound over a run, up to the square of the horizon times the per-step
  error in theory (Ross et al. 2011, Theorem 2.1) [literature: offline_vs_closed_loop check claim 41, supported]; in
  one-camera driving, removing the recovery data cut success from 88 and 64 % to 56 and 22 % (Codevilla et al. 2018,
  Table 1) [literature: augmentation check claim 22, supported with a caveat]; in wrist-camera pushing with 1-cm moves,
  off-track data raised success from 20 to 80 % in 10 trials, with extra play data (DMD) [literature:
  interactive_correction check claim 36, corrected]. And 130 of 210 demonstrations needed a hand correction (audit
  class_balance-10), so drift happens even to people [measured].

### 4. The chunk target after sheet 06

Chunk against the uncleaned exact run, which scored the identical test rows (4,059, 2,669 and 1,982) and, run on the
same three folds in the same order, used the same fold seeds and validation draws (the seed follows the fold's
position; training-settings section 10, rule 3) [measured; fold files, test_rows, and judge_rescoring.out.txt]: path 45 +3.5, +1.8 and -2.1 points on 03, 05, 06 (mean +1.1); step 45 -0.8, +1.3, -3.7. Its
stop never fires on sheet 06 (0 / 1 / 24 of 25 at 0.5; tuned one picture 0 / 0 / 25), and over the three sheets it
stops right in 31 of 100 at (0.5, one picture) against 42. Its gains are on recovery-like pictures (path in the first
five rows after a correction 300 against 260 of 556; straight back-ups 92 against 66 of 218). The checker found
support for training with a short chunk target and executing one step per picture, and noted that a longer prediction
horizon alone had only a transient effect (Zhang et al., Sec. 5) [literature: action_chunking check, section 3 and
claim 43, supported]. So the chunk is a near-tie with a worse stop, not a reason to change the first set-up.

### 5. The backbone

Read as a mean direction, the small CNN from scratch and the fine-tuned ResNet-18 class heads tie on path 45 (0.746
against 0.744 on 03, 05, 06) [measured; fold files, path_expected_within_45_deg]; the ResNet-18 leads only when the
most likely class is executed (0.707 against 0.683). In the literature a ResNet-18 trained from scratch, fine-tuned and
frozen scored 0.94, 0.92 and 0.58 on Square with 200 demonstrations (Chi et al. 2023, Table 5) [literature:
pretrained_representations check claim 37, supported], and an ImageNet start reduced the spread of closed-loop success
between seeds (Codevilla et al. 2019, Table 3) [literature: offline_vs_closed_loop check claim 44, supported]. So
keeping the ResNet-18 is a judgement for steadiness and readiness, and the small CNN is a rules hedge, not a finishing
gain.

### 6. Literature that limits what offline numbers can rank

- Within one model family, hold-out error and robot success were not significantly linked (+0.28 to -0.22; Cowling
  2026, Table 3), and offline metrics should shortlist, not replace robot tests (Abbas et al. 2026) [literature:
  offline_vs_closed_loop check claims 23 and 28, supported].
- In the closest real task, planar pushing from pictures, plain regression scored 35.0, 55.0, 6.7 and 19.6 % against
  85.0, 88.3, 83.3 and 48.3 % for an implicit model (Implicit BC, Table 6) [literature: action_representation check
  claim 27, supported], while from pictures with a ResNet-18, regression beat bins (17.8 against 14.7; Pearce et al.,
  Table 3) [supported]. The head choice is not settled by the literature; it is one more reason to have a class-head
  read-out (S3) ready.
- 24 paired robot trials pick the better of a 70 % and a 90 % model 96 % of the time, but of 80 % and 90 % only 83 %
  [literature: evaluation_statistics check, section 3 point 2]. A robot comparison can only settle large gaps, so the
  backup must differ from S1 where S1 is weak, not be a near copy of it [judgement].

## Ranking with reasons and main risk

1. **S1, final candidate (first set-up).** Reasons: complete six-fold validation; best right stops on shared sheets;
   path 0.735 to 0.856 and twist sign 0.68 to 0.85 on all six held-out sheets; early stops now near the end; ready now.
   Main risk: the stop at unseen line ends (never-stops at ends far from the paper edge and on dashed ends) and weak
   recovery right after a drift (151 of 500 against 138 for always forward on 03, 05, 06).
2. **S3, names-plus-exact (backup).** Reasons: aims at S1's measured recovery gap while keeping S1's twist and stop
   heads; the class head read as a mean direction gets 216 of 500 in post-correction rows and 0.71 of straight
   back-ups; in the frozen probes the combined design matched the exact step (step 0.602, path 0.677, backward 0.363,
   twist sign 0.699 against 0.591, 0.673, 0.270, 0.700) [measured; targets section table 2.2 from designs.json and
   per_class_direction.out.txt]. Main risk: never trained end to end; if it behaves like the separate class model it
   gives up about 4.6 points of path 45 over all path pictures (0.744 against 0.790 on 03, 05, 06) [calculated from
   the two runs], and two heads sharing one network may not keep both strengths [judgement].
3. **S6, chunk of 5.** Reasons: about level with the exact step on direction (+1.1 path), better on recovery-like
   pictures, same executor and time. Main risk: the stop (31 of 100 right at 0.5, never on sheet 06) and an uncleaned,
   best-validation run; the gain is inside any plausible seed spread [judgement].
4. **S5, small CNN from scratch with exact step and stop.** Reasons: ties the ResNet-18 on mean-direction path in the
   class form; needs no ImageNet weights; 2.5 to 13.5 ms per picture. Main risk: the exact-step version has never run;
   its class version stops right in 19 of 100 at 0.5.
5. **S4, class head with square-root weights and mirroring.** Reasons: the best measured post-correction numbers
   (step 45 221 of 500, path 45 335 of 556), 0.68 of straight back-ups, and few early stops (12 of 100 at 0.5) with more
   right stops than the other class heads (41 against 18 and 19). Main risk: twist sign 0.37 to 0.47 on the
   loops, so the forearm camera falls behind; mirroring cost 4.1 points of step 45 and 3.8 of path 45 on sheet 06
   (0.599 against 0.640, 0.669 against 0.707) [measured; mirror and classes fold_without_path_06.json]; mirrored pictures
   show a robot corner that never occurs (audit 11.2 point 26).
6. **S8, change since the previous picture.** Reasons: +2.6 points of step 45 on the mean in the frozen probe. Main
   risk: the worst fold fell (0.467 to 0.454; worst path fold 0.526 to 0.498) and the gain may come from copying the
   last move (repeat-previous scores 0.736) [measured; comparison.md history rows; literature: Wen et al. 2020,
   history check claims 6 and 7, supported].
7. **S7, frozen features.** Reasons: cheap. Main risk: six-fold path 0.673, worst fold 0.534, 7.6 to 19.2 points of
   step 45 below the fine-tuned exact model on 03, 05, 06 [measured; comparison.md; validation section V3].

## The first set-up in full (S1)

- **Targets:** tip forward and left (cm) and forearm twist (deg) of the next step in the current picture's frame,
  Huber loss, unweighted; lean and hand changes in the loss only, not executed; own stop output trained on the 259
  cleaned stop pictures, positive weight 10 [measured; config.py lines 1797 to 1804 and the training-settings
  section].
- **Backbone and input:** timm resnet18.tv_in1k, fine-tuned end to end with the 3 x 4 grid head; one 160 x 120 colour
  picture.
- **Training settings:** audit cleaning (18,032 rows, 209 episodes); scores 1 to 3 at equal weight; shift 4 px plus the
  photometric set; AdamW 1e-3 with backbone x0.3; 8 epochs of half the rows; last weights. Add a second seed (about
  3.7 h) to measure the end-to-end seed spread, which every later choice needs.
- **Validation:** six folds held out by sheet (done), scored on path 45, stop counts under fixed rules, the twist sign
  and the post-correction rows. Then, after the team decides and with a checklist, robot runs of the fold checkpoints
  on their own held-out sheets (unseen-track tests without printing). The exam model is the same recipe on all six
  sheets, run with the same untuned 0.5 rule, checked on sealed printed tracks if allowed.
- **Stop:** fixed 0.5, two pictures in a row, the second look after a step of about 1 cm (offline the second picture
  is the next demonstrated one, a median 1.05 cm on; targets critic). Robot A/B against one picture at 0.5. Step budget
  on as a safety stop: about 4 steps per cm of line (no demonstration needed more than 3.32; validation section V5), or
  a cap near 250 steps where the line length is not known (targets exam-risk critic).
- **Executor (all set-ups share it):** direction of the predicted step, fixed 1.0 cm (the predicted length shrinks to
  0.60 to 0.68 cm on curved sheets; targets section 2.3); twist at gain 1, clamped to plus or minus 10 deg per step,
  joint 4 within plus or minus 115 deg (per-sheet least-squares slopes of the demonstrated on the predicted twist 0.55
  to 1.24, so gain 1 is a judgement to A/B on the robot) [measured; judge_extra_checks.out.txt part 2]; wrist and hand
  held; one move per picture at the 20 % cap. Finishing time follows from this executor, not from the set-up.
- **Precondition for every set-up:** the recording tool has no model mode yet. The code note lists nine missing pieces
  (a model mode and decider, a fixed-wrist "model_step" action, clamps, guards for the table in use, a refusal counter
  with a fallback, the step budget switched on, a separate folder, output columns, Esc polling) [read; audit
  code_actions note, section (f)]. None of the ranking matters until these exist and are tested on the simulated
  robot.

## The backup in full (S3)

Same backbone, input, cleaning, augmentation, optimiser, 8 epochs and last weights as S1. Outputs: the 8-name head
with square-root class weights (its mean direction is executed), the exact head (its twist is executed), the own stop
output on the 259 stop pictures. Same executor and stop rule. Validate exactly as S1 on six folds; keep it as the
backup only if, against S1 (and against a second S1 seed's spread): path 45 on all rows no more than about 3 points
lower, the post-correction rows clearly higher (S1: 415 of 1,124 step 45, 573 of 1,062 path 45 over six sheets), and
the stop at (0.5, two) not worse than about 95 right and 30 early of 209 [judgement]. On the robot, compare S1 and S3
from the same starts, including starts with the puck laid beside the block, scored finish first, then time.

## What would change the ranking

1. **S3's six-fold run.** If it loses more than about 5 points of path 45 or its stop is clearly worse, S6 (cleaned,
   last weights, six folds) becomes the backup. If it stays within about 2 points of S1 on path with the recovery gain
   and an equal stop, S3 becomes the first set-up.
2. **A second seed of S1.** If S1's own seed spread exceeds the S1 to S3 or S1 to S6 gaps, the order of 2 and 3 is a
   coin flip and the robot decides.
3. **First robot runs of fold checkpoints** (after the team decides). Mostly lost pucks needing a back-up move: S3
   confirmed as backup. Mostly drift on bends with recovery working: S6 moves to second. Mostly stops: the ranking
   stays and line-end data becomes the only priority.
4. **The professor.** No ImageNet weights: S5 becomes first (with S3's heads if they win). Printed practice tracks
   allowed: the chosen set-up is retrained with line ends at varied distances from the paper edge, along-edge
   stretches and another dash rhythm, which the data-amount fit and the literature favour over more episodes of the
   six sheets (Lin et al. 2025, diffusion_and_generative check 4.2, checked).
5. **How "finished" is judged.** If pushing past the end counts, choose the stop rule with the fewest early stops ((0.5,
   two) or stricter); if the robot must stop within about 2 cm, the right count decides and no rule beats another.
6. **A line end far from the paper edge** (a printed track, or a provided sheet on a larger white sheet). If S1 stops
   right there, the paper-edge reading above is wrong and the stop risk is smaller than stated.
7. **Light and camera stress tests.** The training-settings exam-risk critic found S1's direction within 2.0 points
   under light and shift changes, but early stops on sheet 02 rising from 2 to up to 10 of 45 at half brightness
   [measured by that critic; training_settings/critic_exam_risk/stress_test_final_candidate_output.txt]. A set-up whose
   stop holds under these changes would move up.

## Where I differ from the drafts and critics

- **Stop rule:** the training-settings draft keeps the inner-validation two-in-a-row threshold; on six folds it has the
  most early stops (58 of 209). The targets draft and several critics want a threshold chosen across held-out sheets;
  on six folds that did no better than the untuned 0.5. The targets exam-risk critic prefers one picture; on S1's six
  folds two in a row at 0.5 cut early stops from 41 to 25 with 102 against 101 right, so I prefer two, with the robot
  A/B.
- **Backup:** the models draft's small CNN and the training-settings draft's mirrored class head are ranked 4 and 5;
  the targets draft keeps names-plus-exact for "if the robot fails to recover". The post-correction rows already
  measure that weakness offline, so I make it the backup now.
- **Chunk:** the targets draft would adopt it if it beat the exact run by more than about 3 points of path 45. With
  sheet 06 the lead is +1.1 on the mean, and its stop never fires on 06.

## Files

- Scripts and printouts (this folder):
  C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/judge_1/judge_rescoring.py,
  judge_rescoring.out.txt, judge_rescoring.out.json, judge_extra_checks.py, judge_extra_checks.out.txt,
  judge_recovery_checks.py, judge_recovery_checks.out.txt (all read only towards D:/AI_Project).
- Runs: D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_<run>/fold_without_path_XX.json and
  predictions_without_path_XX.npz for resnet18_final_candidate (01 to 06), resnet18_finetune_exact, _chunk, _classes,
  _mirror, small_cnn_scratch_classes, small_cnn_line_and_puck_classes (03, 05, 06); comparison.md.
- Drafts and critics: C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/<key>/section.md
  and critic_evidence/note.md, critic_exam_risk/note.md for targets, models, training_settings, validation, improvement.
- Audit: C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/audit/AUDIT_SYNTHESIS.md
  (section 11). Literature checks:
  C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/literature/<key>/check/note.md.
