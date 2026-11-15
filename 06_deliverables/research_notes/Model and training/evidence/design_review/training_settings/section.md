# Training settings

Written 9.11.26 (19:35) for Team M (Emranul Hoque, Marshal Sebastine). Nothing here was run on the robot. Every claim
carries a tag: **[measured]** (a number from a result file or from a re-count of saved predictions, file and field
named), **[literature]** (a source claim that its checker marked supported or corrected; the corrected form is used),
**[calculated]** (arithmetic on measured numbers), **[judgement]** or **[guess]**. Source codes are listed at the end.

## In short

Train the model the way the final candidate is already set up: the audit-cleaned rows, all three scores at equal
weight, small picture shifts plus light and colour changes, about four passes over the data and the last weights kept,
because nothing measured so far beats it by more than the noise between random seeds. That noise is large: training the
same setting again with another seed moves the held-out-sheet score by up to 3.4 points averaged over the six sheets
and by up to 10 points on a single sheet, so two settings can only be told apart with several seeds, or on the robot. One problem stands out:
the exact-step model almost never predicts a straight backward move (0.1 % of its predictions against 3.9 % in the
demonstrations), so the first change to test is giving rare directions more weight, as the class model already does.
Mirrored pictures with left and right swapped in the labels doubled the episodes that stopped in the right place on
the two held-out sheets finished so far (35 of 75 against 17 of 75) without losing direction accuracy, which makes
them the strongest candidate for the next retraining, while longer training, larger shifts and gentler fine-tuning are
untested. Settings that hurt by more than the seed noise: full inverse class weights, a linear head and training on
score-3 episodes only; label smoothing also scored lower, but by less.

*Sections 1 to 10 give the numbers and sources behind each point.*

---

## 1. Read this first: how large a difference must be

**Seeds.** On frozen ResNet-18 features (six held-out sheets), the same setting trained with four different seeds
gave a six-fold mean "within 45 degrees" (share of moving pictures whose predicted push direction is within 45 degrees
of the demonstrated one) of 0.546 to 0.577 for the class head (standard deviation 1.29 points) and 0.591 to 0.616 for
the exact-step head (1.24 points); three more seeds of the class head gave 0.540 to 0.574 (1.70 points). "Stopped
right" (share of episodes whose first stop falls within 2 cm of the demonstrated stop) varied by 2.0 to 3.6 points
(standard deviation) [measured: P seeds.json, designs.json, balance.json, data_amount.json, fields within_45_deg and
episodes_stopped_right; S1]. In these probes the seed also re-draws which 15 % of each training sheet's episodes are
held back for validation [measured: read in training_probes.run_variant and training_targets.inner_validation_mask], so
the spread includes that change of training data, as any rerun would.

**Seed against seed on identical test rows.** For pairs of seeds of one setting, the per-sheet difference in within 45
reached 9.1 points (class head) and 9.8 points (exact head), and the stop outcome changed in 30 to 55 of the 208 test
episodes; for the most lopsided seed pairs a sign test on those episode counts gave p = 0.05 to 0.07, although only the
seed differed [measured: S3]. So an episode-level test overstates certainty: a model change moves many episodes
together. In this section a paired count is called meaningful only if it is clearly beyond that floor [judgement].

**Determinism.** Three copies of the default probe gave identical predictions on every fold [measured: S2], so the
probes are deterministic. Any change to the training rows, even 21 rows, reshuffles the batches and acts like a new
seed [judgement]; for example, keeping the 21 mistake rows moved within 45 by -2.1 points, inside the seed floor
[measured: S2].

**End-to-end runs have one seed each.** Their seed spread is unknown. Seed spread in closed loop can be large: twelve
seeds of one driving model varied widely in success, and ImageNet initialisation reduced the spread (Codevilla et al.
2019, Sec. 5.3, Table 3) [literature: offline_vs_closed_loop-44, supported]. Offline numbers did not see seed
differences that showed on a real arm (Cowling 2026, preprint, Sec. 5.7) [literature: offline_vs_closed_loop-23,
supported].

**Sheets differ more than settings.** The class probe ranges from 0.467 (sheet 06 held out) to 0.756 (sheet 01) in
within 45; always-forward ranges from 0.402 to 0.747 [measured: P balance.json square_root_weights; F baselines]. All
comparisons below are paired on the same folds.

**Some probe variants were not like for like.** Filtering by score or by picture flag also filtered the held-out sheet:
"score 3 only" was scored on 487 to 3,456 test rows per fold instead of 658 to 6,348, and on 134 of the 208 episodes
[measured: P scores.json, cleaning.json, field test_rows]. I re-scored those variants from their saved predictions on
the rows both they and the default predicted [measured: S2]; those re-scored numbers are used below.

**The same sign on all six sheets is not proof either.** Two seeds of the exact head differed by 2.3 points, lower on
all six sheets (-0.7 to -4.0 points per sheet) [measured: S3, exact seed 11 against seed 12], because the six folds share
most of their training data [judgement]. The "folds higher" counts below are shown, but the size of a difference
against the seed range decides.

**Consequence for settings with one run each:** a difference under about 3.5 points of six-fold within 45, or under
about 7 points of stopped right, is not evidence of anything; the seed ranges above reach 3.4 and 7.1 points
[calculated from the seed ranges above].

---

## 2. Cleaning

What the audit found and what the final candidate now does (`training_targets.audit_cleaned_frames`, then the usual
mistake-row rule):

| Step | Rows | Evidence |
|---|---:|---|
| Drop row 0 of every episode and the first row after every hand correction (waypoint 0): null moves, often with hands | 449 | median tip move 0.11 and 0.14 mm; catches 53 of the 55 pictures in the hand census [measured: A label_consistency-7, 11.1 B1, B2] |
| Drop sudden-change pictures not already dropped | 1 | 6 of 6 viewed are real (hands or a person) [measured: A picture_quality-3] |
| Drop the half run 05-16 (12 rows beyond its row 0) and 04-01 try 6 rows 169 to 175 (puck pulled back, 7) | 19 | [measured: A 11.1 B4, B5] |
| Relabel 50 duplicated final points as stop | 50 | largest joint change to the stop row under 0.2 degrees [measured: A 11.1 C1] |
| Clip steps at 2 cm and twists at 15 degrees, keeping the rows | 25 steps, 13 twists | the largest are recording stalls [measured: A 11.1 E; my count in frames.csv] |
| Scores of the two unscored folders from review.json (both score 2) | 71 pictures kept | [measured: A score_differences-1] |
| Corrected start ends from the frozen table | 30 episodes changed | [measured: A metadata_starts-1 to -3] |
| Drop the 21 mistake rows (worktree default) | 21 | ordinary moves; costs little either way [measured: A label_consistency-13] |

Result: 18,032 training rows in 209 episodes (18,053 after the audit rules, 21 fewer after the mistake rule), 259 stop
rows (1.44 %), 1,765 small_motion rows that are mostly twists in place, longest step 2.00 cm [measured: S4].

**Like-for-like probe results** (class head, frozen features, six folds, re-scored on shared rows; S2):

| Variant against the default | Within 45, mean change | Folds higher | Stopped right | Episodes right only with variant / only with default |
|---|---:|---:|---|---|
| also drop the 504 static rows (tip under 1 mm, forearm under 1 degree) | -1.1 points | 2 of 6 | 0.268 against 0.258 | 23 / 20 (p 0.76) |
| also drop all 1,018 flagged pictures | -1.5 points | 2 of 6 | 0.196 against 0.218 | 14 / 23 (p 0.19) |
| keep the 21 mistake rows | -2.1 points | 1 of 6 | 0.271 against 0.258 | 21 / 17 (p 0.63) |

None of these differences is beyond the seed floor [measured: S2, S3]. Offline numbers therefore cannot show whether
cleaning helps; the case for it is that the dropped rows carry wrong labels for their pictures (a no-op move on the
same picture as the next real move; hands in view) [measured: A label_consistency-7, picture_quality-4], and that
single-step policies trained on idle actions got stuck on a real robot: LSTM-GMM stuck in 8 of 20 real Push-T runs,
with the authors calling removal of idle actions common practice (Chi et al. 2023, Sec. 6.1) [literature:
action_representation-31 and checker's evidence 8, supported].

**Flags.** Do not filter by picture flag: the 'no puck seen' flags are mostly from an older detector (today's finds the
puck in 377 of 524), dark-object flags are mostly shadows and the printed line, and none of the 17 blur flags is blurred
[measured: A picture_quality-3, -6, -7].

**Rows the implemented cleaning still keeps** (audit list D): named hand pictures 02-45 frame 043, 03-18 frames 051 and
053, 06-14 frame 071, 04-22 frame 134, 06-20 frame 065 and the last pictures of 02-02; 06-20 from the tip-over to the end
(22 to 37 rows; puck beside the block, outcome says puck_in_goal); 01-18 frames 039 to 043 (5 rows); 01-09 (12 pictures,
block hid the puck, 4 flagged); the 7 relabelled stops whose picture differs from the stop picture by more than 3 grey
levels [measured: A 11.1 C1, D3, D4, metadata_starts-8]. About 50 to 70 rows, 0.3 to 0.4 % [calculated]. No offline
effect is expected at that size [judgement]; they should be looked at and dropped for label correctness.

**The stale start-end text.** No training or scoring code reads start_end; only the loader corrects it [measured: read
in training_targets.py lines 95 to 96 and 143; grep of the training code]. It matters for per-end reports, per-end
sampling and any end-based baseline. Use the frozen corrected-ends table (18 of its rows were decided by hand) and
never regenerate it [measured: A metadata_starts-3]. Two code comments are stale and should be corrected when the code
is next edited: config.py lines 1566 to 1567 and train_policy.py line 12 say the camera creeps a few pixels, while it
moved once, by about 15 px at 640 x 480 [measured: A picture_quality-10; read in both files on 9.11.26].

---

## 3. Score weighting or filtering

Like for like (class head, frozen features, six folds, re-scored on the episodes both variants predict; S2):

| Variant | Test episodes | Within 45, mean change | Folds higher | Stopped right, variant against default | Discordant episodes |
|---|---:|---:|---:|---|---|
| score 3 only | 134 | -4.3 points | 0 of 6 | 0.338 against 0.264 | 24 / 16 (p 0.27) |
| scores 2 and 3 | 184 | -1.0 | 1 of 6 | 0.186 against 0.263 | 14 / 24 (p 0.14) |
| weights 0.4 / 0.7 / 1.0 | 208 | -0.3 | 1 of 6 | 0.306 against 0.258 | 17 / 8 (p 0.11) |

Score 3 only costs direction accuracy on every sheet, by 0.5 to 9.9 points per fold [measured: S2]. The stop
differences stay inside the seed floor (seed pairs reached 29 / 16 and 30 / 16) [measured: S3]. Score 1 cannot be told
apart from score 2 in motion, and it holds the hardest starts: dropping it thins the TR15 starts from 436 to 203 pictures
and the mid-track starts from 1,024 to 461 [measured: A score_differences-4, -11]. There is no measured basis for a
lower weight on score 1 [measured: A score_differences-13].

**Keep scores 1 to 3 at equal weight.** If a score weight is tried again, test it on the stop only, since that is the
only place where a gain appeared [judgement].

---

## 4. Class weights and balanced sampling

**Class head (the backup design).** Square-root weights (each class weighted by the square root of its rarity; after
cleaning 0.59 for forward up to 2.06 for back and 2.86 for stop) [measured: S4] against the alternatives, same rows
[measured: P balance.json; S2]:

| Weights | Within 45 | Path within 45 | Balanced class accuracy | Stopped right | Never stopped |
|---|---|---|---|---|---|
| none | +1.6 points (4 of 6 folds higher) | +2.9 (6 of 6) | -2.2 (0 of 6) | -8.2 (1 of 6); right only with none 9, only with square root 24 (p 0.014) | +14.9 (6 of 6) |
| square root (default) | 0.577 | 0.633 | 0.254 | 0.258 | 0.599 |
| full inverse | -6.8 (0 of 6) | -8.2 (0 of 6) | +1.6 (4 of 6) | +3.4 (3 of 6) | -16.6 (0 of 6) |

The direction cost of square-root weights (1.6 points) is inside the seed floor; their stop gain (24 against 9 episodes,
p 0.014, against 0.05 to 0.07 for the most lopsided seed pairs) is just beyond it [measured: S3]. Full inverse weights
cost 6.8 points of direction accuracy, twice the largest seed difference.

**Exact head (the final candidate) has no direction weights, and it shows.** Pooled over held-out sheets 03, 05 and 06
(7,738 moving pictures), the end-to-end exact model recalls 0.3 % of straight-back moves and predicts "back" for 0.1 % of
pictures against 3.9 % in the demonstrations; it over-predicts forward (44.8 % against 38.4 %). The class model with
square-root weights recalls 26.6 % of back moves (predicts 4.7 %) [measured: S5, from the predictions_*.npz files of
both runs]. The frozen probes show the same pattern: back recall 4.2 % for the exact head, 8.8 % for the class head
without weights, 16.0 % with square-root weights [measured: S5]. Back moves are 11.8 % of moving rows after cleaning
[measured: S4] and 60 % of the first moves after a hand correction [measured: A label_consistency-13]. Whether the
robot needs straight-back moves to recover is unknown [judgement]. Weighting the exact loss per row by the
square-root weight of the row's move name is untested.

**Stop output.** 68.6 non-stop rows per stop row after cleaning [measured: S4]; the stop output uses a positive weight
of 10 and a threshold tuned on the inner validation episodes (end-to-end exact: 0.15, 0.30, 0.15 for two pictures in a
row) [measured: E resnet18_finetune_exact metrics.json, settings.stop_positive_weight, tuned_two_stop_threshold]. With
a tuned threshold the positive weight mainly moves the threshold rather than the order of the stop scores
[judgement]; no other weight was tested.

**Sampling unit.** Picture-uniform sampling, used in every run, gives sheet 04 (the meander) 34.5 % of all rows from
21.5 % of the episodes, and 35.8 % to 44.2 % of the training rows of the folds where it trains; sheet 01 gets 4.0 % to
5.5 % [measured: S4]. Track variety matters more than episode count: at about the same number of pictures, five sheets
at half their episodes scored 0.546 against 0.508 to 0.532 for two to four sheets with all episodes, and doubling the
episodes of five sheets added 0.9 points while a fifth sheet added 2.3 (three seeds each) [measured: F data amount;
P data_amount.json]. Variety across training environments also drove generalisation in the literature (Xie et al.
2023, Factor World simulation: gap from about 0.4 to under 0.1 going from 5 to 100 environments) [literature:
pretrained_representations-44, corrected to simulation]. Whether sheet-balanced weights help is untested
[judgement].

**When correction data arrive later:** retrain from scratch on all data rather than fine-tune on the corrections only
(fine-tuning on corrections was worst, -30 % and -15.6 %; Xu et al. 2025, Findings 4 and 5) [literature:
interactive_correction-29, supported], and compare a correction share of 10 to 25 % of each batch with 50 % before
choosing [literature: interactive_correction checker's suggestion, judgement].

---

## 5. Augmentation

**Which changes keep the label right for this camera.** The block, the gripper fingers, the puck and the line are all
in the picture, and move names are given in the picture's own frame. A whole-picture shift moves all of them together,
so the move stays right; a rotation turns every direction by the rotation angle; a mirror swaps left and right; an
upside-down flip or a 90-degree turn puts the tool where it never is [literature: augmentation note section 4 and table
6.2, the researcher's reasoning, which the checker accepted as the best-supported part]. The measured spread of our
pictures sets the strengths [measured: A picture_quality-7, -10, -11, notes_context-10].

| Change | Label | Measured in our data | Current setting (all end-to-end runs, 160 x 120) | Recommendation |
|---|---|---|---|---|
| Whole-picture shift (pad by repeating edge pixels, then crop) | kept | the camera view stepped once by about 15 px at 640 x 480 (14 right, 5 down), about 3.75 px at 160 x 120 [calculated]; roll under 0.5 degrees | plus or minus 4 px (2.5 % of the width, 3.3 % of the height) [measured: E metrics.json settings.augmentation.shift_px] | keep 4 px; test 8 px (5 %), the typical size in the papers |
| Crop with a scale change (zoom) | broken for the step length: the camera's distance to the block is fixed, so a zoom shows a step size that never occurs [judgement] | none | not used | do not use |
| Rotation in the picture plane | changes by the rotation angle; 18.8 % of moves lie within 5 degrees of a class border | roll under 0.5 degrees; the forearm twist is not an in-plane rotation of the picture | not used | do not use |
| Left-right mirror with labels swapped | correctable (swap names; negate left, twist and hand) but the picture is impossible | the bottom-right patch is at least 30 grey levels darker than its mirror in 16,793 of 18,522 pictures (90.7 %); left-to-right move ratio by sheet 0.76 to 1.85 | off; one end-to-end test with the class head (sheets 03 and 05 done) | off in the first robot model; first candidate for the next retraining if sheet 06 agrees and a mirror-consistency check passes (see below) |
| Brightness (gain) | kept | 0.74 to 1.12 of the median (1st to 99th percentile); natural-light 1st percentile 0.674 | x0.65 to x1.35 | keep |
| Contrast | kept | 0.72 to 1.26 | x0.7 to x1.3 | keep |
| Gamma | kept | not measured | exponent 0.78 to 1.28; with the brightness gain, a picture of median brightness ends between about 0.49 and 1.43 of its brightness [calculated] | keep; the combined range is a margin for an unknown exam room [guess] |
| Saturation | kept | not measured | plus or minus 25 % | keep |
| Colour cast (gain per colour channel) | kept while the puck stays redder than the paper | white-sheet blue/red ratio 1.117 to 1.572 (median 1.228; 0.91 to 1.28 of it); ceiling 1.219, natural 1.354 | plus or minus 8 %, so the blue/red ratio moves by 0.85 to 1.17 [calculated] | raise to plus or minus 13 % (ratio 0.77 to 1.30, covering the measured 0.91 to 1.28) [calculated; low priority] |
| Hue rotation | risky for the red puck | puck hue 137 to 163 (OpenCV scale) in the robot's shadow against 159 to 161 in the open | none | none; the channel gain already makes violet pucks [judgement] |
| Gaussian blur | kept | at most about sigma 0.5 px at 640 x 480; the hazy lens of 6.11 lay between 0.5 and 0.75 px; that is 0.13 to 0.19 px at 160 x 120 [calculated] | 15 % of pictures, sigma 0.3 to 0.8 px at 160 x 120 (1.2 to 3.2 px at 640 x 480) [calculated] | lower the top to 0.5 px; the measured spread is invisible at 160 x 120, so any blur is only a margin for a smeared lens [judgement; low priority] |
| Sensor noise | kept | sigma 0.43 to 0.46 grey levels at 640 x 480, about 0.11 after downsizing to 160 x 120 [calculated, assuming independent pixel noise] | up to 0.03 of full scale (7.7 grey levels) | lower to 0.008 (2 grey levels), the audit's "light noise" [judgement; harmless either way] |
| Cut-outs (erase rectangles, for hands) | broken when the puck or line is hidden | hands were found in 15 of 210 first pictures and 36 of 266 first pictures after a pause; the cleaning removes all but about 6 named ones | none | none; remove hand pictures instead; nobody's hand should be in view during an exam run [guess] |
| Shadow patches, edge erasing | kept if away from puck and line | the robot's own shadow in daylight; dark objects beyond the table corner | none | optional, only if robot tests show sensitivity [guess] |

**What the literature measured.** Shifts are the best-supported augmentation for behaviour cloning: removing them cost
47 % and 35 % of success (relative) in robomimic's image tasks, and on the real robot 22 of 30 against 8 of 30 trials
(Mandlekar et al. 2021, Sec. 4.3, 4.7) [literature: augmentation-10, -11, supported; the checker computes p about
0.0007]. Shift sizes in the papers run from 3.3 % to 6.3 % of the picture [literature: augmentation checker's
calculation from claims 1, 13, 14, 33]; RAD found the gain comes from translation and in some cases two extra pixels
were enough [literature: augmentation-8, supported]. Random shift beat cutout, flips, rotation and intensity changes
(DrQ, curves only) [literature: augmentation-4, supported]; flips and rotations added to crops lowered RAD's Walker
score from 920 to 650 and 635 [literature: augmentation-7, corrected]. A bundle of colour, blur, noise and small
dropout changes moved driving in an unseen town from 0 % to 64 % (Codevilla et al. 2018, Table 1) [literature:
augmentation-19, -20, supported], but the checker notes this guards against new light and tables, not new curves.
Coloured cutout on its own did worst offline, worse than no augmentation (Young et al. 2020) [literature:
augmentation-35, corrected]. No paper measured mirrored, label-swapped pictures on robot success; a symmetry the labels
do not follow can hurt [literature: augmentation checker, points 3 and 42 at abstract level]. Mirrors would double the track shapes from 6 to 12, the one augmentation aimed at
unseen tracks [literature: augmentation checker's judgement].

**Mirror test so far.** ResNet-18 class head with half the pictures mirrored (labels swapped) against the same run
without mirroring, same seed and validation episodes, held-out sheets 03 and 05 (75 episodes; 3,458 and 2,471 moving
pictures) [measured: E resnet18_finetune_mirror fold_without_path_03.json, fold_without_path_05.json and
resnet18_finetune_classes metrics.json; S3]:

| Held-out sheet | Within 45, mirrored / plain | Path within 45 | Stopped right | Stopped early | Never stopped |
|---|---|---|---|---|---|
| 03 (45 episodes) | 0.670 / 0.663 | 0.726 / 0.704 | 26 / 13 | 3 / 5 | 14 / 26 |
| 05 (30 episodes) | 0.659 / 0.659 | 0.739 / 0.711 | 9 / 4 | 8 / 5 | 13 / 21 |

21 episodes stopped right only with mirroring and 3 only without (sign test p below 0.001, against 0.05 to 0.07 for the
most lopsided probe seed pairs) [measured: S3]. Direction accuracy did not change; path within 45 rose by 2.2 and 2.8
points, inside the seed floor; early stops rose on sheet 05 (8 against 5 of 30). Why the stop gains is not known; one
reading is that mirrored stop pictures give the rarest class (299 stop rows in that run, which predates the audit
cleaning) new views of the paper's end [guess; the count from A class_balance-1]. One seed,
two sheets, and only the class head: the exact head's mirrored labels (left part, twist and hand negated) are in the
code but untested [measured: read in train_policy.batch_targets]. Sheet 06 was still training at 19:30.

**No end-to-end run without augmentation exists** (resnet18_finetune_no_augmentation is configured but was never run)
[measured: 04_runs listing], so its value on our sheets rests on the literature [judgement]. A hand-made light-proof
input (darkness and redness maps instead of colour) did not help the small network offline: within 45 0.626 against
0.631 and path within 45 0.683 for both, sheets 03, 05, 06 [measured: E small_cnn_line_and_puck_classes and
small_cnn_scratch_classes metrics.json].

---

## 6. Input size

The end-to-end runs use 160 x 120 colour pictures (the camera's 4:3 shape) [measured: E metrics.json settings]. At that
size a sideways move shifts the picture content by about 5.2 px per cm, about 6.3 px for a typical left step [measured:
A label_consistency-14]; the 1-cm line is about 5.5 px wide and the puck about 15 px [calculated from about 22 px per cm
and a 60-px puck at 640 x 480, measured in one picture by the augmentation checker, claim 43 corrected]. ResNet-18
needs 17 ms per picture at 160 x 120 against 26 ms at 224 x 224 (4 threads, laptop busy) [measured: F speed]. Papers
used 84 x 84 and 120 x 120 with ResNet-18 (robomimic), found 128 x 128 enough (Schafer et al.), and 320 x 240 with
288 x 216 crops on real robots (Chi et al.) [literature: pretrained_representations checker's evidence 3, claim 50
supported, claim 39 supported]. 224 x 224 would have 2.6 times the pixels and roughly that much more training time
[calculated]. No size comparison exists on our data. **Keep
160 x 120**; try a larger size only on the GPU pool [judgement].

---

## 7. Optimiser, learning rate and schedule

Current [measured: E metrics.json settings]: AdamW, head learning rate 1e-3, backbone 0.3 times that (3e-4), weight
decay 0.05, batch 64, a linear warm-up over the first epoch, then a cosine decay to zero at the end of training.

- Learning rates matter: in robomimic a larger learning rate cut success by 35 % to 63 % (relative) [literature:
  Mandlekar et al. 2021, Sec. 4.4, as read by the action_representation checker while checking claim 24; the rates are
  not given in our notes].
- Fine-tuning a pretrained encoder at one tenth of the policy's learning rate gave Chi et al. their best results
  [literature: pretrained_representations-38, supported]; training the head first on frozen features and then
  fine-tuning everything (LP-FT) was on average 10 % better out of distribution than plain fine-tuning across 10 image
  datasets (Kumar et al. 2022) [literature: pretrained_representations-51, supported]. Unseen tracks are an
  out-of-distribution test [judgement]. Neither result comes from a ResNet-18 pushing task.
- On our data, the end-to-end exact runs' validation loss was still falling at the last epoch on all three sheets
  (0.487, 0.535, 0.572 at epoch 8) [measured: E fold_*.json history]; nothing shows the learning rate is too high.

**Keep the current optimiser and schedule** for the next model; test "backbone factor 0.1 plus one head-only epoch"
as one combined change when compute allows [judgement].

---

## 8. Epochs, early stopping and which weights to keep

- **Validation loss on held-out episodes of the training sheets is a poor guide.** Episodes of one recording block are
  near-twins (twin share 0.26 to 0.55 within a block, 0.000 to 0.003 across blocks), so early stopping on them rewards
  memorising the block [measured: A duplicates_leakage-3, -15]. In robomimic the last checkpoint beat the
  lowest-validation-loss checkpoint in all 8 dataset and algorithm pairs; against the best checkpoint the
  lowest-loss one lost 57 % to 100 % and the last one 12 % to 35 % [literature: offline_vs_closed_loop-12, supported;
  the percentages are the checker's calculation]. Real-robot studies kept the last checkpoint of a fixed budget (Chi et
  al., Lin et al.) [literature: offline_vs_closed_loop-29, -20, supported]. Cross-entropy can rise while accuracy still
  improves (Guo et al. 2017) [literature: offline_vs_closed_loop checker, checked].
- **What our runs show.** Exact head: the lowest validation loss came at the last epoch (8 of 8) on all three sheets,
  train and validation loss close (0.47 to 0.51 against 0.49 to 0.57). Class head: lowest at epochs 7, 5 and 7, last
  epoch 0.011 to 0.026 higher, train loss 1.35 to 1.39 against validation 1.64 to 1.94 [measured: E fold_*.json
  history; S1]. So for the exact design "best weights" and "last weights" were the same model; for the class design the
  reported numbers are from epochs 5 to 7. The final candidate (exact, audit cleaning, last weights): sheet 01 lowest at
  epoch 7 (0.5503) against 0.5507 at epoch 8 [measured: E resnet18_final_candidate fold_without_path_01.json].
- **Budget.** 8 epochs on a random half of the training rows each is about 4 passes [measured: E settings
  train_share_per_epoch 0.5, epochs 8]. Held-out-sheet scores per epoch were not logged, so whether 8 passes would beat 4
  is unknown. The checker's advice is to set the budget once from held-out action measures, then keep the last
  checkpoint [literature: offline_vs_closed_loop checker, judgement]; success has been seen to keep improving after the
  loss levels off (ACT tuning notes, not peer reviewed) [literature: offline_vs_closed_loop checker, checked].
- **Time.** 20.4 to 34.6 minutes per held-out sheet for the ResNet-18 runs (34.6 with a second run in parallel), so
  2.0 to 3.5 hours for six sheets [measured: E fold_*.json minutes; calculated].

**Fixed budget of 8 epochs at half the rows each, no early stopping, last weights** (exactly the final candidate). Keep
the 15 % inner validation episodes for the stop threshold and for watching the run [judgement]; holding them back costs
little, since doubling the episodes per sheet added only 0.9 points [measured: F data amount].

---

## 9. Regularisation

- **Head:** one hidden layer of 256 with dropout 0.2. A linear head lost 7.8 points of within 45 (0 of 6 sheets
  higher) and had 16.9 points more early stops (6 of 6); a 1,024-wide layer with dropout 0.3 lost 1.9 points (0 of 6
  higher) [measured: P head_size.json; S1]. Dropout 0.2 in the policy head almost always helped on a real robot,
  while it often hurt in simulation (Dasari et al. 2023, App. B, Sec. 5, Fig. 4) [literature:
  pretrained_representations-27, supported].
- **No label smoothing.** Uniform smoothing 0.1 lost 1.9 points of within 45 on all six sheets and raised "never
  stopped" by 7.8 points on all six [measured: P balance.json; S1]. That is inside the seed floor, so the evidence is
  weak, but nothing speaks for it [judgement]. Angular soft targets about 12 degrees wide, which the audit suggested for
  a class head, are untested [measured: A 11.2 point 9].
- **Weight decay 0.05 (AdamW)** and the augmentation of section 5 as they are; no comparison exists [judgement].

---

## 10. Seeds: how many to tell two settings apart

From the measured seed spread of the six-fold mean (section 1), the seeds needed per setting for a two-sided test at
5 % with 80 % power [calculated: S1, noncentral t, assuming the spread above]:

| True difference | Within 45 (spread 1.24 to 1.70 points) | Stopped right (spread 2.0 to 3.6 points) |
|---|---|---|
| 2 points | 8 to 13 seeds each | 17 to 53 |
| 3 points | 4 to 7 | 9 to 24 |
| 5 points | 3 to 4 | 4 to 10 |

With one seed each, the truly better of two settings shows the higher within 45 with probability 0.80 to 0.87 at a
2-point gap and 0.89 to 0.96 at a 3-point gap [calculated: S1]. The literature advises fixing the number of seeds in
advance from a pilot (Colas et al. 2018: 5 seeds gave about 49 % power and 10 about 81 % in their example)
[literature: evaluation_statistics-26, supported], reporting interval estimates over runs (Agarwal et al. 2021)
[literature: evaluation_statistics-25, supported], and remembering that a robot test compares trained checkpoints,
not training recipes (TRI LBM 2025) [literature: evaluation_statistics-15, supported].

**Rules for Team M** [judgement, built on the numbers above]:

1. Frozen probes (2 to 4 minutes each [measured: P *.json seconds]): at least 4 seeds per setting; act only on
   differences of 3.5 points or more of within 45, or 7 points or more of stopped right, unless more seeds confirm a
   smaller one.
2. End-to-end runs (2 to 3.5 hours each): 2 seeds per setting, and only for changes expected to move within 45 or
   path within 45 by 5 points or more; smaller questions go to the robot or are left at the default.
3. Seed each fold from its sheet, not from its place in the list: at present `train_policy.run_training` adds the
   fold's position to the seed and to the inner-validation draw, so a run on sheets 03, 05 and 06 and a run on all six
   sheets use different seeds and different validation episodes for the same sheet [measured: read in train_policy.py
   lines 446 to 464].
4. For the exam model trained on all six sheets, train 2 seeds and let paired robot trials choose (offline numbers
   could not rank seeds in Cowling 2026) [literature: offline_vs_closed_loop-23, supported].

---

## Recommended settings

| Setting | Value | Status |
|---|---|---|
| Rows | audit cleaning as implemented (18,032 rows, 209 episodes, 259 stops) | in the final candidate |
| Extra cleaning | look at and drop the about 50 to 70 rows of audit list D (named hands, 06-20 after the tip-over, 01-18 frames 039 to 043, 01-09, 7 relabelled stops) | next retraining |
| Flags | no filtering by flag | in the final candidate |
| Mistake rows | dropped (no measurable effect either way) | in the final candidate |
| Scores | 1, 2 and 3 at equal weight | in the final candidate |
| Start ends | the frozen corrected-ends table, for reports and any per-end sampling | in the final candidate |
| Direction weights | none in the exact loss for now; test square-root weights per row first (section 4) | test first |
| Stop weight | positive weight 10, threshold tuned on inner validation episodes, two pictures in a row | in the final candidate |
| Sampling | uniform over pictures; test square-root sheet weights in the probes | test |
| Shift | plus or minus 4 px at 160 x 120 | in the final candidate |
| Photometric | brightness x0.65 to 1.35, contrast x0.7 to 1.3, saturation plus or minus 25 %, gamma exponent 0.78 to 1.28 | in the final candidate |
| Colour cast | per-channel gain plus or minus 13 % (now 8 %) | next retraining, low priority |
| Blur | 15 % of pictures, sigma 0.3 to 0.5 px (now up to 0.8) | next retraining, low priority |
| Noise | up to 0.008 of full scale (now 0.03) | next retraining, low priority |
| Mirror with labels swapped | off now; half the pictures mirrored at the next retraining if sheet 06 agrees and the consistency check passes | test first (strongest stop gain measured) |
| Rotation, zoom crops, cut-outs | off | final |
| Input | 160 x 120 colour, ImageNet normalisation | in the final candidate |
| Optimiser | AdamW, head 1e-3, backbone x0.3, weight decay 0.05, batch 64, 1 warm-up epoch, cosine to zero | in the final candidate |
| Budget | 8 epochs of a random half of the rows (about 4 passes), no early stopping, last weights | in the final candidate |
| Head | one hidden layer of 256, dropout 0.2, no label smoothing | in the final candidate |
| Seeds | sheet-based seed per fold; 2 seeds for end-to-end comparisons; 2 seeds of the exam model, chosen on the robot | change the seeding code |

For the first robot tests, use the final candidate's recipe exactly as it was measured, so that its offline numbers
describe the model on the robot: a fold model on its own held-out sheet is a test of an unseen track, and the same
recipe trained on all six sheets is the exam candidate. That all-sheet model does not exist yet; `train_policy.py` has
no "all sheets" option [measured: read in train_policy.py lines 444 to 445]. Fold the low-priority augmentation changes
and the extra cleaning into the next retraining [judgement].

**Backup.** If the exact model fails to back up or recover on the robot (the risk of section 4), switch to the class
head with square-root class weights, with the same rows, augmentation, budget and last weights, and with mirroring if
sheet 06 confirms it. On sheets 03, 05 and 06 the class head predicts back moves at about their real rate (4.7 %
against 3.9 %) and recalls 26.6 % of them [measured: S5], at a cost of 2.6 points of within 45 and 8.3 points of path
within 45 against the exact model (3 of 3 sheets lower) [measured: E metrics.json of both runs; S1]; with mirroring it
stopped right in 35 of 75 episodes on sheets 03 and 05, against 37 of 75 for the exact model without mirroring
[measured: E fold files of both runs]. If compute runs short, keep the final candidate as is and spend the remaining
laptop time on seeds rather than on new settings [judgement].

---

## What is still unknown

- **How noisy end-to-end runs are.** Every end-to-end comparison so far rests on one seed (section 1).
- **Whether the cleaning, the score rule or the class weights matter on the robot.** Offline they sit inside the noise,
  except the stop gain of square-root class weights.
- **Whether the exact model's missing back moves matter** for recovery, and whether direction weights fix them without
  costing direction accuracy.
- **Whether the mirroring gain holds up**: on sheet 06, with other seeds, with the exact head, and on the robot; and
  whether the network learns to recognise mirrored pictures by the robot's dark corner.
- **The right training length.** No held-out-sheet curve per epoch exists.
- **Whether 8 px shifts, gentler fine-tuning, sheet-balanced sampling or a larger input** help; untested here.
- **The exam's light, table and track features** (daylight appears only on sheets 02 and 04) [measured: A
  picture_quality-9], which set how wide the photometric margins should be.
- **The final candidate's six-fold result.** At 19:35 only sheet 01 had finished: within 45 0.779 (578 moving
  pictures), path within 45 0.856 (529), stopped right 0.60 (12 of 20 episodes), 0.80 (16 of 20) with the tuned
  threshold and two pictures in a row, twist sign 0.68 (238 rows) [measured: E resnet18_final_candidate
  fold_without_path_01.json]; always-forward on that sheet scores 0.747 and 0.806 [measured: F baselines]. Its test
  rows are the cleaned ones (644 rather than 658 on sheet 01), so it is not directly comparable with the earlier runs.

**Experiments still needed**, in order (laptop processor; one end-to-end run on six sheets takes 2.0 to 3.5 hours
[calculated from E minutes]; a frozen probe 2 to 4 minutes [measured: P seconds]):

| What | Why | Hours |
|---|---|---|
| Let the queued runs finish: final candidate (5 sheets left), mirror (sheet 06), exact on 01, 02, 04, chunk on 03, 05, 06 | the main measurement and the only end-to-end augmentation test | 6 to 8 (queued) |
| Two more seeds of the final candidate on six sheets, seeded by sheet | the seed spread every end-to-end comparison needs | 4 to 7 |
| Train the exam candidate on all six sheets, 2 seeds (needs an "all sheets" option in train_policy.py) | no such model exists yet | 1 to 1.5 |
| Square-root direction weights in the exact loss: probe with 4 seeds, then end-to-end with 2 seeds | back moves recalled 0.3 % | 0.3 + 4 to 7 |
| 16 epochs (8 passes) against 8, last weights, held-out measures logged every 2 epochs | training length is unknown; validation loss still falling at epoch 8 | 4 to 7 per seed |
| Mirror consistency check (mirror held-out pictures, check the predictions mirror), before adopting mirroring | the picture is not symmetric | 1 |
| Exact head with half the pictures mirrored, 6 sheets, 2 seeds | the stop gain was measured with the class head only | 4 to 7 |
| Eye check of audit list D and the 7 relabelled stops | label correctness | 0.5 to 1 (a person) |
| Shift plus or minus 8 px against 4 px, 2 seeds | the papers' typical size is about 5 % | 4 to 7 |
| Backbone factor 0.1 plus one head-only epoch, 2 seeds | literature points to gentler fine-tuning out of distribution | 4 to 7 |
| Square-root sheet weights: probe with 4 seeds; end-to-end only if 3.5 points or more | sheet 04 is up to 44 % of training rows | 0.3 |
| Re-run score and flag filters in the probe code with filtering on training rows only | the original probes changed the test rows (partly repaired here by re-scoring) | 0.3 |
| Puck and tool positions near the picture border, before any larger shift | a large shift could cut the fingers or the puck off | 0.5 |
| No augmentation, and shift only, end to end | the value of the photometric part on our sheets | 2 to 3.5 each, low priority |

About 31 to 53 hours of processor time beyond the queued runs [calculated], so the order matters; the GPU pool would
change this.

**Questions for the professor**

1. May we print and demonstrate on extra practice tracks of the exam's size and line width and train on them? More
   different tracks helped more than more episodes.
2. May the exam model be trained on all six provided sheets, with no sheet held out?
3. Do mirrored copies of our own demonstrations, with left and right swapped in the labels, still count as training on
   our own demonstrations?
4. Is a fixed mask over the robot's own parts in the picture (the same pixels every time) allowed as preprocessing?
5. Under which light and on which table will the exam run?
6. Can we use the university GPU pool to train the final model?

---

## Sources

- **A**: data audit synthesis, `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/audit/AUDIT_SYNTHESIS.md` (finding ids, section 11).
- **P**: frozen-feature probes, `D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes/<experiment>.json` (per-fold fields).
- **E**: end-to-end runs, `D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_<run>/metrics.json` and `fold_<sheet>.json`.
- **F**: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/main_session_facts.md`.
- **L**: literature notes `.../scratchpad/literature/<key>/note.md` and their checks `.../check/note.md`.
- **S1**: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/training_settings/settings_evidence_output.txt` (per-fold differences, seed spread, seeds needed, loss histories).
- **S2**: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/training_settings/like_for_like_rescoring_output.txt` (probe variants re-scored on shared rows).
- **S3**: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/training_settings/noise_floor_rescoring_output.txt` (seed against seed, sign tests, mirror fold 03).
- **S4**: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/training_settings/cleaned_data_counts_output.txt` (rows before and after cleaning, class weights, sheet shares).
- **S5**: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/training_settings/per_class_recall_output.txt` (recall and predicted share per direction).

The scripts that wrote S1 to S5 are in the same folder (`settings_evidence.py`, `like_for_like_rescoring.py`,
`noise_floor_rescoring.py`, `cleaned_data_counts.py`, `per_class_recall.py`); they only read the project's files.
