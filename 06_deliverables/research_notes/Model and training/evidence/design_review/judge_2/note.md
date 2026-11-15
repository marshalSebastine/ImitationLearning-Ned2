# Judge 2: which whole training set-up first, and which as backup (robustness and simplicity, laptop CPU)

Written 9.11.26 between 21:55 and 22:40 for Team M (Emranul Hoque, Marshal Sebastine). Read only: nothing under
D:/AI_Project was changed, nothing was trained and nothing ran on the robot. I re-scored saved predictions with one
script of my own, `judge2_checks.py`, whose printout is `judge2_checks.out.txt` (both in
C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/judge_2/).
It reproduces the fold files' stop counts at threshold 0.5 exactly on all six final-candidate folds.

Markers: **[measured]** a number from a run file or my re-count of saved predictions (file and field named);
**[calculated]** arithmetic on measured numbers; **[literature]** a claim its checker marked supported (or corrected,
then in the corrected form); **[judgement]** or **[guess]** my own reasoning. Every end-to-end number below comes from
one training seed; the end-to-end seed spread is not measured yet.

State of the runs [measured: file times in 04_runs]: the final candidate finished its sixth held-out fold (sheet 06)
at 22:02, so all six folds are in. Since the drafts were written, these fold files also appeared: final candidate
folds 04, 05 and 06, the end-to-end chunk run's folds 03 and 05, and the mirror run's fold 06. I use them all.

---

## In short

Use the model that is already trained and measured, the final candidate, for the first robot work, and run it with
the simplest rules: step 1 cm in the direction the model predicts, twist the forearm by the amount it predicts (with
limits), keep the wrist and hand still, and stop when the stop output reaches 0.5 on two pictures in a row. Do not
tune the stop threshold for each model. On the 209 held-out episodes of the six folds, this fixed rule stopped in the
right place about as often as the best tuned rule (102 against 105 of 209) and early less often (25 against 33),
while the rule the recipe uses now (threshold tuned on validation episodes, two in a row) stopped right in 98 and
early in 58.

As the backup, train the same network once more with two outputs, the 8 move names and the exact step, and steer by
the names. On the same three held-out sheets (03, 05, 06; 1,054 backward moves), the final candidate gets 0.426 of the
demonstrated backward moves within 45 degrees, an earlier names model 0.663. Backing up is what a recovery after a
slipping puck needs, and offline numbers cannot rank recovery, so the robot decides between the two. The small
network trained from scratch is the fallback if the professor rules out ImageNet weights.

The stop is the weakest part of every set-up: about half of the held-out runs stop in the right place, and on the
dashed loop 21 of 25 never stop. No choice of network or target fixes it; new demonstrations at line ends do.

---

## 1. The set-ups compared

**Common parts** (unless a row says otherwise): ImageNet ResNet-18 (timm resnet18.tv_in1k) trained end to end on
160 x 120 colour pictures with the 3 x 4 grid head (hidden layer 256, dropout 0.2); one picture as input; the audit
cleaning as implemented (18,032 training rows in 209 episodes, 259 stop rows); scores 1 to 3 at equal weight; shift
plus or minus 4 px with brightness, contrast, gamma, saturation, channel-gain, light blur and noise; AdamW (head 1e-3,
backbone x0.3, weight decay 0.05, batch 64, one warm-up epoch, cosine); 8 epochs of a random half of the rows; last
weights kept [measured: config.py lines 1568 to 1602, 1713, 1802 to 1804; training_settings section].

**Common executor** (needed by every set-up; the model still sees only the picture): direction from the model, fixed
1.0 cm step, predicted twist at gain 1 clamped to plus or minus 10 deg per step, joint 4 kept within plus or minus 115
deg (at the clamp: keep stepping, hold the twist, log it), joints 5 and 6 held at their start values, joints 1 to 3
from `solve_with_wrist_fixed`, one move per picture at the recorded 20 % speed, workspace and height guards, a refusal
counter with a fallback, and the step budget switched on (4 steps per cm of line where the length is known, since no
demonstration needed more than 3.32 [measured: validation V5]; otherwise a cap of about 250 steps, the longest
demonstration having 226 [measured: targets critic exam_risk_checks.out.txt, part I]) [A-code (f) items 1 to 9;
targets section 3 points 5 to 7].

| Set-up | What is predicted, and how it stops | Network | Status |
|---|---|---|---|
| **S1** final candidate, simple run rules | exact step (tip forward and left in cm, forearm twist in deg; lean and hand trained but not executed), Huber loss unweighted; own stop output on the 259 stop pictures (positive weight 10); stop at a **fixed 0.5 on two pictures in a row**, no tuning | ResNet-18 fine-tuned | trained, all 6 folds measured |
| **S2** names plus exact | 8 move names (square-root class weights) plus the exact step; direction executed as the probability-weighted mean of the names; twist from the exact part; own stop output, same fixed rule | ResNet-18 fine-tuned | configuration only (design "mixed", is_stop_separate on); probe only so far |
| **S3** small CNN, exact plus stop | as S1 | 5 convolution blocks from scratch, 1.61 M parameters, learning rate 2e-3, 16 epochs of half the rows | configuration only; only its names version was run |
| **S4** chunk of 5 | tip places after each of the next 5 steps plus the exact step and stop; only the first step's direction executed | ResNet-18 fine-tuned | run on sheets 03 and 05 with the earlier uncleaned rows |
| **S5** final candidate as drafted | as S1, but stop threshold tuned per model on its inner validation episodes, two in a row | ResNet-18 fine-tuned | trained (same model as S1) |
| **S6** names only | 8 move names (square-root weights) plus a 3-way twist class executed as plus or minus 5 deg; stop as a class | ResNet-18 fine-tuned | run on 03, 05, 06, earlier rows |
| **S7** names with mirrored pictures | as S6, half the pictures mirrored with left and right swapped in the labels | ResNet-18 fine-tuned | run on 03, 05, 06, earlier rows |
| **S8** frozen features | names or exact head on fixed ResNet-18 grid features at 224 x 224 | ResNet-18 frozen | six folds, four seeds |

Not ranked, because the drafts and their checked literature already rule them out for now and I agree: a picture
history as input (offline gain 2.6 points while the worst fold fell, a copycat risk [measured: comparison.md, history;
literature: Wen et al. 2020, history check claims 6 and 7, supported]); ACT and diffusion policies (CPU cost, over
5,000 ms per decision on processors [literature: diffusion check claims 29 and 30, supported]); fine-tuned vision
transformers (cost, and collapse on small data [literature: pretrained check claims 12 and 17, supported]).

---

## 2. Ranking

### Rank 1: S1, the final candidate with simple run rules

Reasons:

- **It exists and is measured on all six unseen-track stand-ins** [measured: FC fold files, within_45_deg,
  path_within_45_deg, twist_sign_accuracy; always forward from baselines.json on the uncleaned rows, via
  main_session_facts.md]:

  | Held-out sheet | step 45 (moving pictures) | always forward | path 45 (path pictures) | always forward | twist sign (twisting rows) |
  |---|---|---|---|---|---|
  | 01 straight | 0.779 (578) | 0.747 | 0.856 (529) | 0.806 | 0.676 (238) |
  | 02 arc | 0.632 (2,425) | 0.565 | 0.762 (2,243) | 0.660 | 0.802 (1,195) |
  | 03 two curves | 0.680 (3,458) | 0.522 | 0.801 (3,116) | 0.603 | 0.848 (2,630) |
  | 04 meander | 0.642 (5,264) | 0.485 | 0.735 (4,502) | 0.580 | 0.773 (4,535) |
  | 05 loop | 0.649 (2,461) | 0.509 | 0.754 (2,324) | 0.544 | 0.677 (1,713) |
  | 06 dashed loop | 0.684 (1,809) | 0.402 | 0.810 (1,691) | 0.433 | 0.786 (1,430) |
  | mean (range) | 0.678 (0.632 to 0.779) | 0.538 | 0.786 (0.735 to 0.856) | 0.604 | 0.676 to 0.848 |

  It beats always forward on every fold, by 3.2 to 28.2 points of step 45 and 5.0 to 37.7 points of path 45
  [calculated]. For one new track like these six, a t prediction interval for path 45 is about 0.66 to 0.91
  [calculated: mean 0.786, standard deviation 0.045 over the six folds, t with 5 degrees of freedom, times the square
  root of 7/6].
- **Its two design choices are the best measured, like for like.** Fine-tuning end to end beat the frozen exact probe
  by 7.6, 10.4 and 19.2 points of step 45 on sheets 03, 05 and 06, with episode-bootstrap intervals of 6.0 to 9.1, 7.7
  to 13.1 and 15.2 to 22.6 [measured: validation section V3]. With the same network and sheets, the exact step beat the
  names on path 45 by 9.8, 6.7 and 8.4 points and on twist sign (0.845, 0.669, 0.802 against 0.703, 0.372, 0.422)
  [measured: resnet18_finetune_exact and _classes metrics.json]. A camera that twists the right way on curves is the
  main robustness gain of the exact target [judgement].
- **The cleaned recipe kept the direction of the earlier exact run and stopped right a little more often.** On the
  shared held-out sheets 03, 05 and 06 the final candidate averaged 0.671 step 45 and 0.788 path 45 against 0.681 and
  0.790, inside the probe seed ranges (up to 9.8 points of step 45 per sheet), and stopped right at 0.5 in 47 of 99
  episodes against 42 of 100, with 22 early against 25. The gain came from sheet 05 (12 right and 11 early of 29
  against 6 and 17 of 30); sheet 06 stopped right less often (2 of 25 against 5) [measured: fold files of both runs].
  Sheet 05 lost 05-16, whose stop label sat mid-track, from test and training [A-syn 11.1 B4], which may explain part
  of the sheet-05 gain [judgement].
- **It is robust to light, colour and small camera changes in direction.** On the saved fold models for sheets 01 and
  02, half and one-and-a-half brightness, warm and cool casts, 4 and 8 px shifts, a 3-degree roll and 1 px blur kept
  step 45 within 0.766 to 0.791 (clean 0.779) and 0.612 to 0.640 (clean 0.632) [measured: training_settings
  critic_exam_risk/stress_test_final_candidate_output.txt]. The stop is the fragile part there (section 3.1).
- **It is the simplest run-time recipe that works.** A fixed 1-cm step cannot stall (the predicted length shrinks to a
  median 0.68 cm on sheet 02 and to 0.22 to 0.36 cm within 2 cm of the goal [measured: same stress-test printout]);
  gain 1 needs no calibration and lies inside the per-sheet least-squares gains of 0.55 to 1.24 (section 3.2); the
  fixed-wrist solve reproduces the demonstrated next view within 0.5 px at the 90th percentile [calculated: A-code
  (f)]; the fixed 0.5 stop rule needs no tuning step and is the same for the fold models and for the exam model, which
  has no held-out sheet to tune on.
- **It fits the CPU.** 23.4 to 55.0 minutes per held-out fold, 223 minutes for all six with other jobs sharing the
  processor; 21.5 ms per picture against a 1.22 s robot step [measured: FC fold files, minutes; finetune_exact
  metrics.json, milliseconds_per_picture; A-code (e)].
- **Literature.** An end-to-end ResNet-18 with shift augmentation at about 200 demonstrations: 0.94 from scratch, 0.92
  fine-tuned, 0.58 frozen on robomimic Square (Chi et al. 2023, Table 5) [literature: pretrained check claim 37,
  supported]; shifts were the best-supported augmentation (robomimic real Can 22 of 30 against 8 of 30)
  [literature: augmentation claims 10 and 11, supported]; keeping the last weights of a fixed schedule beat choosing by
  validation loss in 8 of 8 rows (robomimic Table 29) [literature: offline_vs_closed_loop check claim 12, supported].

Main risk: **the stop, then recovery.** Under the fixed rule it stops right in 102 of 209 held-out runs (Wilson 0.421
to 0.555), early in 25 and never in 69 [measured: J2 section A]. On the dashed loop it hardly fires: 3 right, 1 early
and 21 never of 25 (its stop output caught 0.129 of that sheet's stop pictures at 0.5 [measured: fold 06,
stop_recall]). On the loop sheet the early stops fire mid-track (sheet 05 at 0.5, one picture: median 14.6 cm before
the goal, range 4.9 to 19.2 cm, against medians of 3.7 to 4.7 cm on sheets 01 to 04) [measured: J2 section D], where
the line runs along the paper edge [measured: critics of the models and training_settings sections, pictures viewed].
It backs up badly on most sheets: 647 of 1,894 backward moves within 45 deg (0.342) against 0.702 on the other moves,
with a backward direction predicted for 0.2 to 8.1 % of the pictures on sheets 01 to 05 against 6.4 to 16.9 %
demonstrated [measured: J2 section B].

### Rank 2 (backup): S2, names plus exact on the same pipeline

Reasons:

- **It targets S1's measured weakness and keeps S1's strengths.** The names model read as a probability-weighted mean
  direction got 0.663 of 1,054 backward moves within 45 deg against 0.392 for the exact model (sheets 03, 05, 06,
  earlier rows), at 0.674 against 0.730 on the other moves [measured: targets per_class_direction.out.txt]; the final
  candidate got 0.426 of the same 1,054 backward moves and 0.709 of the other 6,674 [calculated from J2 section B,
  per-sheet shares times counts]. In the frozen probes the combined design read this way tied the exact step overall
  (step 45 0.602 against 0.591, path 45 0.677 against 0.673) and backed up better (0.363 against 0.270), with the same
  twist sign (0.699 against 0.700)
  [measured: designs.json; per_class_direction.out.txt]. On the pictures after a hand correction, the closest thing to
  recovery pictures, the exact step's end-to-end lead over the names vanished (0.605 against 0.605) [measured:
  validation section V4].
- **Recovery matters on the robot, and offline numbers cannot rank it.** 130 of 210 demonstrations needed a hand
  correction [A-syn class_balance-10], and after one the first real move was backward in 148 of 245 cases [A-syn
  label_consistency-13]. Errors compound over a run, up to T squared times the per-step error in theory (Ross et al.
  2011, Theorem 2.1) [literature: offline_vs_closed_loop check claim 41, supported], and offline metrics shortlist but
  do not replace robot tests (Abbas et al. 2026) [literature: same check, claim 28, supported].
- **It is the cheapest backup to own.** Same data, augmentation, schedule, executor and stop rule; one configuration
  entry (design "mixed", is_stop_separate True, stop_target "stop_row", audit cleaning, 8 epochs, last weights) and,
  as far as I can read the code, no new training code [read in training_probes.design_outputs and training_arrays,
  train_policy.py line 326 and config.py; not run]. At run time it adds one weighted
  average of 8 angles [judgement]. Six folds take about 2.5 to 5.5 hours on the laptop [calculated from the final
  candidate's 23.4 to 55.0 minutes per fold].

Main risk: **untested end to end.** The names model read as a mean direction lost 4.6 points of path 45 against the
exact model (0.744 against 0.790, sheets 03, 05, 06, one seed) [measured: targets section 2.2], so S2 may follow the
line less well; its own stop output is untested in this design.

### Rank 3: S3, small CNN from scratch with the exact step and stop

Reasons: it is the fallback if the professor rules out ImageNet weights (models section, question 1). With the names
output it trailed the fine-tuned ResNet-18 by 2.9, 2.5 and 1.7 points of step 45 on sheets 03, 05 and 06 (bootstrap
intervals 1.1 to 4.6, -0.2 to 4.8, -0.7 to 4.0) and by 3.5, 2.8 and 0.9 points of path 45 [measured: validation V3;
small_cnn_scratch_classes and resnet18_finetune_classes metrics.json], and it beat every frozen network. It is small
(6.5 MB, 2.5 to 13.5 ms per picture) and trains in 15.8 to 24.7 minutes per fold [measured: models section, metrics
files]. From scratch tied fine-tuned at about 200 demonstrations in the literature (Chi et al. 2023, Table 5)
[literature: pretrained check claim 37, supported].

Main risk: it has never run with the exact output, and it shares every failure of S1 (stop, backing up) without
fixing any. ImageNet initialisation also narrowed the seed-to-seed spread of closed-loop success in driving (table
"variance" 4, 12 and 38 % against 23, 26 and 42 %, Codevilla et al. 2019, Table 3) [literature: offline_vs_closed_loop
check claim 44, supported], which favours S1 for robustness [judgement].

### Rank 4: S4, chunk of 5 steps, first step executed

Reasons: on the earlier uncleaned rows it raised path 45 over the exact model by 3.5 points on sheet 03 (0.837
against 0.802) and 1.8 on sheet 05 (0.795 against 0.777), with step 45 -0.8 and +1.4 and the same twist sign (0.839
and 0.662 against 0.845 and 0.669) [measured: resnet18_finetune_chunk fold_without_path_03.json and _05.json against
resnet18_finetune_exact]. Executing only the first step keeps the executor of S1, and executing chunks blind lost
under noise (Liu et al. 2025) [literature: action_chunking check, supported], so nothing more is needed at run time.

Main risk: the gain is below the drafts' own bar (about 3 points of path 45 over six folds; mean +2.65 on two folds
[calculated]), and its stop was worse: at 0.5, one picture, 26 right of 45 against 31 on sheet 03 and 5 of 30 against
6 on sheet 05, with 18 early stops on sheet 05 [measured: same files]. It has not been run on the cleaned rows.

### Rank 5: S5, the final candidate with the drafted stop rule

Reasons: same direction and twist as S1. Main risk: the threshold tuned on inner validation episodes (0.05 to 0.4,
and 0.1 or lower on four of six folds) gave 98 right, 58 early and 33 never of 209, against 102, 25 and 69 for the
fixed 0.5 rule on the same predictions; on sheet 06 it tuned to 0.05 and stopped early in 12 of 25 [measured: J2
section A; FC fold files, tuned_two_stop_threshold]. The validation episodes are near-twins of training episodes [A-syn
duplicates_leakage-3, -15], and the exam model trained on all six sheets has no held-out sheet to tune on [judgement].
It is dominated by S1 at no cost, because the rule acts only at run time.

### Rank 6: S6, names only (the training_settings draft's backup)

Reasons: it predicts back moves at about their real rate (4.7 % of pictures against 3.9 %, recall 26.6 %) [measured:
training_settings S5]. Main risk: its 3-way twist got the sign right on only 0.372 and 0.422 of the twisting rows of
the loop sheets (0.703 on sheet 03), and its path 45 was 8.3 points below the exact model (3 of 3 sheets lower)
[measured: resnet18_finetune_classes and _exact metrics.json]. S2 keeps its backing up and replaces its twist, so S6 is
dominated by S2 [judgement].

### Rank 7: S7, names with mirrored pictures

Reasons: the largest stop gain measured. At 0.5 it stopped right in 26, 9 and 6 episodes on sheets 03, 05 and 06,
against 13, 4 and 1 without mirroring (41 against 18 of 100) [measured: resnet18_finetune_mirror fold files;
training_settings section]. Main risk: sheet 06 lost 4.1 points of step 45 (0.599 against 0.640) and 3.8 of path 45
(0.669 against 0.707) [measured: mirror fold_without_path_06.json against finetune_classes]; a mirrored picture never
occurs with this camera (the bottom-right patch is darker than its mirror in 16,793 of 18,522 pictures) [A-syn 11.2
point 26]; it inherits S6's twist weakness (sign 0.470 on sheet 06); whether mirrored copies count as our own
demonstrations is a question for the professor (training_settings, question 3).

### Rank 8: S8, frozen features

Reasons: fastest to train (head 20 to 257 s for six folds) and a useful bench for stop experiments. Main risk: 7.6 to
19.2 points of step 45 below fine-tuning on the matched sheets (V3), and the pretrained backbones beat always forward
by only 0.7 to 3.9 points on the six-fold mean [measured: models section 3.1]. Not a candidate for the robot.

---

## 3. New numbers for this note (final candidate, all six held-out folds, 209 episodes)

### 3.1 Stop rules on the same predictions

Right: within 2 cm of where the demonstration stopped; early: more than 3 cm before; between: 2 to 3 cm; never: no
picture passes [measured: J2 section A; the fixed-0.5 one-picture counts match every fold file].

| Rule | right | early | between | never | right share (95 % Wilson) | early share (95 % Wilson) |
|---|---|---|---|---|---|---|
| fixed 0.5, one picture | 101 | 41 | 17 | 50 | 0.483 (0.416 to 0.551) | 0.196 (0.148 to 0.255) |
| **fixed 0.5, two in a row (S1)** | **102** | **25** | 13 | 69 | 0.488 (0.421 to 0.555) | 0.120 (0.082 to 0.171) |
| fixed 0.4, two in a row | 104 | 31 | 14 | 60 | 0.498 (0.430 to 0.565) | 0.148 (0.107 to 0.203) |
| fixed 0.6, one picture | 104 | 32 | 16 | 57 | 0.498 (0.430 to 0.565) | 0.153 (0.111 to 0.208) |
| fixed 0.7, one picture | 101 | 24 | 13 | 71 | 0.483 (0.416 to 0.551) | 0.115 (0.078 to 0.165) |
| tuned on inner validation episodes, one picture (0.3 to 0.8) | 105 | 33 | 16 | 55 | 0.502 (0.435 to 0.570) | 0.158 (0.115 to 0.213) |
| tuned on inner validation episodes, two in a row (0.05 to 0.4; S5) | 98 | 58 | 20 | 33 | 0.469 (0.402 to 0.537) | 0.278 (0.221 to 0.342) |
| chosen on the other five sheets, right minus early (the targets draft's rule) | 86 | 24 | 13 | 86 | 0.411 (0.347 to 0.479) | 0.115 (0.078 to 0.165) |

Per sheet, fixed 0.5 two in a row (right / early / between / never): 01 16 / 3 / 0 / 1 of 20; 02 19 / 1 / 1 / 24 of
45; 03 29 / 5 / 3 / 8 of 45; 04 21 / 9 / 7 / 8 of 45; 05 14 / 6 / 2 / 7 of 29; 06 3 / 1 / 0 / 21 of 25. With one
picture: 01 12 / 4 / 3 / 1; 02 24 / 2 / 1 / 18; 03 33 / 9 / 2 / 1; 04 18 / 13 / 10 / 4; 05 12 / 11 / 1 / 5; 06 2 / 2 /
0 / 21 [measured: J2 section A]. (The Wilson intervals of the last three early shares are my calculation.)

What I read from it:

- **The right-stop count is flat** from 0.4 to 0.7 with one picture (100 to 104) and from 0.25 to 0.5 with two (100 to
  104) [measured], so no tuning is needed and a shift in the exam model's probability scale costs little [judgement].
- **Two in a row at 0.5 keeps the right stops and removes 16 early stops**, turning them and others into never-stops
  (50 to 69). An early stop always fails a run; a never-stop is ended by the step budget and may still count if the
  puck reaches the end, which the exam rule does not say yet [judgement; A-syn 11.3 question 20]. Per sheet the change
  is mixed (02 lost 5 right stops, 01 gained 4, 03 lost 4, 04 gained 3) [measured], so one picture against two stays a
  robot A/B item. On the in-sample criterion right minus early, fixed 0.5 with two in a row (77) ties the best single
  rule over the six sheets (0.7 with one picture, also 77) [calculated from the table].
- **Choosing the threshold on the other sheets did worse than fixing it** (86 right against 102), because the chosen
  rule jumped between 0.4 and 0.7 and between one and two pictures from fold to fold [measured: J2 section A].
- **The dashed loop is a stop failure, not a threshold problem:** at every fixed threshold from 0.4 up, 0 to 6 of its
  25 runs stop right and 18 to 25 never stop; lower thresholds make it fire more often but also early (0.2, one
  picture: 8 right, 7 early, 10 never) [measured: J2 section A]. Its two early stops at 0.5 fired 20.2 and 20.4 cm
  before the goal [measured: J2 section D].
- **The stop is the fragile part under changed pictures.** At half brightness, sheet 02's early stops rose from 2 to
  10 of 45 (0.5, one picture) and from 8 to 21 (tuned, two in a row); an 8 px shift raised the tuned rule's early stops
  from 8 to 11 and 14 [measured: stress_test_final_candidate_output.txt]. Half brightness is beyond the training range
  (natural light reaches down to 0.674 of the median) [A-syn 11.2 point 22], so keep the recordings' ceiling light and
  check the camera and brightness at every session [judgement].

### 3.2 Backing up and twist, all six folds

| Held-out sheet | backward moves | within 45 on backward | on the other moves | predicted backward / demonstrated | twist gain (least squares through zero) | twist sign, all twisting rows / where 3 deg or more is predicted |
|---|---|---|---|---|---|---|
| 01 | 37 of 578 | 0.054 | 0.828 | 0.002 / 0.064 | 0.65 | 0.676 (238) / 0.723 (47) |
| 02 | 280 of 2,425 | 0.139 | 0.696 | 0.021 / 0.115 | 0.55 | 0.802 (1,195) / 0.865 (704) |
| 03 | 340 of 3,458 | 0.412 | 0.709 | 0.081 / 0.098 | 0.94 | 0.848 (2,630) / 0.928 (1,670) |
| 04 | 523 of 5,264 | 0.300 | 0.680 | 0.062 / 0.099 | 0.96 | 0.773 (4,535) / 0.851 (2,382) |
| 05 | 416 of 2,461 | 0.320 | 0.715 | 0.064 / 0.169 | 0.66 | 0.677 (1,713) / 0.787 (367) |
| 06 | 298 of 1,809 | 0.591 | 0.703 | 0.292 / 0.165 | 1.24 | 0.786 (1,430) / 0.955 (404) |
| pooled | 647 of 1,894 (0.342) | | 9,902 of 14,101 (0.702) | | | |

[measured: J2 sections B and C]. Backing up is weak on five of six sheets; on the dashed loop the model predicts a
backward direction more often than demonstrated. The twist gain is 0.55 to 1.24 per sheet, so gain 1 is a judgement,
not a fit, and no single other gain fits all sheets; I keep gain 1 for simplicity, clamped, and leave 0.7 against 1.0
to the robot only if the camera over-twists [judgement].

### 3.3 Runs that finished after the drafts

- Final candidate folds 04, 05 and 06: in the tables above.
- Chunk run (earlier rows): see rank 4.
- Mirror run, sheet 06: step 45 0.599 against 0.640 without mirroring; stop at 0.5 right 6, early 1, never 17 of 25
  against 1, 1 and 23; twist sign 0.470 [measured: resnet18_finetune_mirror/fold_without_path_06.json].

---

## 4. How to validate S1, simply and in stages

Offline (laptop, no robot):

1. Report every fold beside its own always-forward rule, with path 45 (mean and worst fold), step 45, stop counts
   under the fixed rule (right, early, between, never), the backward-move and after-correction strata and the twist
   sign. Gate for the robot: path 45 above always forward on every fold (met: +5.0 to +37.7), twist sign 0.6 or more on
   every fold (met: 0.676 to 0.848), and early stops under about 15 % pooled (met: 25 of 209, Wilson 8.2 to 17.1 %)
   [judgement on the measured numbers above].
2. Stress-test every fold checkpoint (brightness 0.65 and 1.35, shift 8 px, roll 3 deg); about a minute per condition
   per fold [measured: stress-test printout, 11 to 55 s per condition].
3. Train S2 and S3 on six folds, and two more seeds of S1, before any claim that they differ; count a difference only
   if it exceeds the seed spread (about 3.5 points on a six-fold mean, up to 16 points of path 45 on one sheet in the
   probes) [measured: training_settings section 1; validation critic].
4. Train the exam model on all six sheets with the identical recipe (needs an all-sheets option in train_policy.py
   [measured: training_settings section 8]). Keep one model; a second seed only as a spare. Choosing between seeds on
   the robot is poor value: 24 paired blocks pick the better of a 90 and an 80 % model only about 83 % of the time
   [calculated: validation V7, matching the evaluation_statistics checker's section 3, point 2].

Robot (only after the team decides, each session with its checklist, one program connected at a time):

5. Stage 0, executor shake-down without a model: replay recorded steps through the new model-step executor on sheets
   01, 02 and 04 (both ends), force one refusal and drive joint 4 to its clamp (27 of 45 sheet-04 demonstrations passed
   110 deg) [validation critic; A-syn 11.2 point 29].
6. Stage 1, unseen tracks without printing: the fold checkpoints on their own held-out sheets (without_path_03 on
   sheet 03, _04 on 04, _05 on 05, _06 on 06), about 5 runs each, about 2 to 2.5 robot hours [calculated by the models
   critic from 10.6 to 12.9 trials per hour], plus one provided sheet laid on a larger white sheet to see whether the
   stop depends on the paper edge [improvement critic]. Record finish, failure type, steps, time and joint 4.
7. Stage 2: if most failures are drift or failure to recover, run S2's fold checkpoints the same way, then a paired
   comparison only if both finish most runs (24 paired blocks choose the better of a 90 and a 70 % model 96 % of the
   time [literature: evaluation_statistics check claim 37, supported]). If most failures are stops, change the data,
   not the set-up (section 6).
8. Stage 3: the exam model on the known sheets and on sealed printed tracks, if printing is allowed, by a pre-registered
   rule (validation section 2.3).

---

## 5. Decisive evidence (what the ranking rests on)

1. Fine-tuning end to end beat frozen features by 7.6, 10.4 and 19.2 points of step 45 on sheets 03, 05, 06 (intervals
   exclude zero) [validation V3].
2. Exact step against names, same network and sheets: path 45 +9.8, +6.7, +8.4; twist sign 0.845 / 0.669 / 0.802
   against 0.703 / 0.372 / 0.422 [finetune_exact and finetune_classes metrics.json].
3. The final candidate on all six held-out sheets: step 45 0.632 to 0.779 (mean 0.678), path 45 0.735 to 0.856 (mean
   0.786), above always forward on every fold [FC fold files].
4. Fixed 0.5, two in a row: 102 right and 25 early of 209; the drafted tuned rule: 98 and 58 [J2 section A].
5. Backing up: final candidate 0.342 of 1,894 backward moves within 45 against 0.702 on others (six sheets); on the
   1,054 backward moves of sheets 03, 05, 06: final candidate 0.426, earlier exact 0.392, earlier names read as a mean
   direction 0.663 [J2 section B; per_class_direction.out.txt].
6. Direction robust to light, colour, 8 px shift, 3-degree roll and blur within 2 points; stop early stops up to 5
   times higher at half brightness [stress_test_final_candidate_output.txt].
7. Small CNN against ResNet-18, names output: 1.7 to 2.9 points of step 45, two of three intervals touching zero
   [validation V3].
8. Chunk against exact: path 45 +3.5 and +1.8, fewer right stops on sheet 03 (26 against 31 of 45) [chunk and exact
   fold files].
9. Mirror: stop right 41 against 18 of 100, but step 45 -4.1 on sheet 06 [mirror fold files].
10. CPU: 23.4 to 55.0 minutes per final-candidate fold; 21.5 ms per picture against a 1.22 s step [fold files;
    metrics.json; A-code (e)].
11. Offline numbers shortlist but rarely rank close models (Cowling 2026, Table 3, within-budget correlations +0.28 to
    -0.22, none significant) [literature: offline_vs_closed_loop check claim 23, supported], so S1 against S2 is a robot
    decision.

## 6. What would change the ranking

- **The professor rules out ImageNet weights:** S3 becomes first, with S2's head on the small CNN as its backup.
- **Stage 1 on the robot:** failures mostly in recovering a drifted or sliding puck move S2 to first; failures mostly
  in stopping change no rank but make line-end data the first job; a stall, livelock or joint-4 problem is an executor
  fix for every set-up.
- **S2's six-fold run:** path 45 within about 3 points of S1 and backward moves within 45 at about 0.45 or more would
  make S2 first; a loss of more than about 5 points of path 45 or a worse stop would drop it below S3.
- **S4 on the cleaned rows over six folds:** path 45 above S1 by more than the seed spread on at least 5 of 6 sheets,
  without more early stops, would make S4 first (the run-time rule stays the same).
- **Two more seeds of S1:** if the end-to-end seed spread exceeds the S1 to S3 gap, S1 and S3 tie offline; I would keep
  S1 for its narrower closed-loop seed spread with ImageNet initialisation unless the rules question goes the other way.
- **The exam rule on finishing:** if the robot must stop by itself, one and two pictures in a row tie (101 and 102 of
  209 right); if pushing past the end counts, two in a row is clearly better (25 against 41 early). It changes the stop
  rule inside S1, not the ranking of networks.
- **The white-underlay test or a printed line end away from the paper edge:** if the stop stops working without the
  paper edge, every set-up needs new end-of-line demonstrations (dashed ends included, given sheet 06) before the exam;
  the ranking stays, the data plan changes.
- **The GPU pool:** more seeds and a fair chunk comparison sooner; it does not change the first model.

## 7. Where I part from the drafts and critics

- **Stop threshold.** The training_settings draft keeps the inner-validation tuned two-in-a-row rule; it gave the most
  early stops (58 of 209). The targets draft chooses the threshold leave-one-sheet-out; over six sheets that gave 86
  right against 102 for a fixed 0.5, because the chosen rule jumps. I fix 0.5 (the code's default stop_threshold) for
  every model, including the exam model, and change it only by a pre-registered rule after robot practice runs.
- **One picture against two.** The targets exam-risk critic proposes one picture by default; on the pooled numbers two
  in a row has about the same right count (102 against 101) and 16 fewer early stops, so I make it the default and
  keep one picture as the robot A/B alternative.
- **Backup.** The training_settings draft's backup (names only) loses the twist on loops; S2 keeps its backing up with
  the exact twist. The models draft's backup (small CNN) guards against a rules decision, not against a measured
  failure, so I rank it third.
- **Mirroring.** Not the strongest candidate for the next retraining: sheet 06 lost 4.1 points of step 45, the picture
  is not left-right symmetric, and only the names head was tested.
- **Twist gain.** The quoted slope of 0.97 does not hold per sheet on the final candidate (0.55 to 1.24); gain 1 stays
  as the simplest choice, marked as a judgement.
- **Agree with the critics** on: early closed-loop tests with the existing fold checkpoints; the executor shake-down
  before any model run; line ends away from the paper edge, dashed ends and near-end demonstrations as the top data
  item, and asking the professor now about printing and how finishing is judged.

---

## Sources

Runs (D:/AI_Project/.claude/worktrees/training-analysis/04_runs/):

- [FC fold files] 2026-11-09_resnet18_final_candidate/fold_without_path_01.json to _06.json (measures: within_45_deg,
  path_within_45_deg, twist_sign_accuracy, stop_recall, episodes_stopped_*, tuned_*, minutes) and
  predictions_without_path_01.npz to _06.npz.
- 2026-11-09_resnet18_finetune_exact, _finetune_classes, _finetune_mirror, _finetune_chunk, small_cnn_scratch_classes:
  metrics.json and fold_without_path_*.json.
- comparison.md; 2026-11-09_frozen_feature_probes/designs.json, baselines.json (via main_session_facts.md).

This note's check: [J2] C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/judge_2/judge2_checks.out.txt
(sections A stop rules, B backing up, C twist, D early-stop distances), written by judge2_checks.py in the same folder.

Drafts and critics (C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/):

- targets/section.md and targets/per_class_direction.out.txt; targets/critic_evidence/extra_checks.out.txt;
  targets/critic_exam_risk/exam_risk_checks.out.txt.
- models/section.md; models/critic_exam_risk/note.md.
- training_settings/section.md; training_settings/critic_exam_risk/stress_test_final_candidate_output.txt.
- validation/section.md (V3, V4, V5, V7 from validation_numbers_output.txt); validation critic notes.
- improvement/section.md and its critic notes.
- main_session_facts.md.

Audit: [A-syn] C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/audit/AUDIT_SYNTHESIS.md
section 11 and the findings named; [A-code] audit/code_actions/note.md sections (e) and (f).

Literature (scratchpad/literature/<key>/check/note.md): pretrained_representations claims 12, 17, 37;
offline_vs_closed_loop claims 12, 23, 28, 41, 44; augmentation claims 10, 11; action_chunking (Liu et al. 2025);
history_and_causal_confusion claims 6, 7; diffusion_and_generative claims 29, 30; evaluation_statistics claim 37. All
marked supported by their checkers.
