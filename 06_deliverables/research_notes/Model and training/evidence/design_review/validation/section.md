# Validation that predicts the exam, and the robot test

Section of the Team M report (Emranul Hoque, Marshal Sebastine), draft of 9.11.26. Nothing in it was run on the robot;
the robot phases below start only after Emranul and Marshal have agreed the plan. "Points" means percentage points.
Every number carries its source in brackets (the short names are listed at the end) and says whether it is measured,
calculated, from the literature (with the independent checker's verdict), a judgement or a guess.

## Summary in plain words

To know how the robot will do on the exam's new tracks, a model has to be tested on a track it never saw in training.
Testing on held-out episodes of the six sheets is too kind: those episodes look almost like training episodes recorded
in the same sitting, and on our data they score 9 to 12 points higher on average than a whole held-out sheet, and up to
27 points higher on the dashed loop. So we compare models offline only by leaving one whole sheet out, by how often the
predicted push direction and the stop agree with the demonstrations, and we treat differences smaller than about 3
points (10 points on a single sheet) as noise. Offline numbers can rule out bad models but cannot rank good ones, so
the robot decides, under a plan written before the first trial: both candidates run from the same starts in random
order, and the final test uses at least ten new printed tracks. Ten trials cannot tell a 70 % model from a 90 % one;
about 24 trials per model are enough to choose between two models, and 50 to 63 per model to prove the difference.

## 1. Evidence

### 1.1 Why a test must hold out whole episodes, and why that is still not enough

- **Random pictures leak.** With pictures split at random, 80.6 to 81.2 % of the validation pictures have a near-twin
  in training, and a one-nearest-neighbour look-up (which cannot generalise at all) reaches a median angle error of
  23.6 degrees, against 52.8 degrees when whole sheets are held out (measured; A, table 5.2).
- **Episodes must stay whole.** 395 of 18,312 consecutive picture pairs differ by under 2 grey levels, and 314
  pictures have a near-copy two or more steps away in the same episode (measured; A, finding duplicates_leakage-2 and
  the further facts of section 5). The 18,522 pictures hold only about 1,175 to 1,736 independent views and about
  3,100 independent labels (calculated; A, duplicates_leakage-7), so every statistic must count episodes or larger
  units, not pictures.
- **Whole episodes still leak, through the recording block.** With episodes held out, 56.3 to 58.1 % of validation
  pictures still have a twin in training, against 2.8 % with sheets held out. The twin share between two episodes of a
  sheet is 0.316 when they were recorded within 15 minutes of each other and 0.000 beyond 4 hours, and 0.000 to 0.003
  across recording blocks of the same sheet (measured; A, duplicates_leakage-3, table 5.2).

### 1.2 Leaving one sheet out as the stand-in for an unseen track

- **It is honest.** It removes the twins (2.8 %), and the look-up fails as it should: 26.4 % agreement with the
  demonstrated move against 40.3 % for "always forward" (measured; A, table 5.2).
- **It is harsh.** Each fold also holds out that sheet's day, light, operator mix, camera period and some start kinds
  (measured; A, table 5.3), and it trains on five sheets while the exam model trains on six. More sheets helped more
  than more episodes: going from four to five training sheets added 2.3 points of "step within 45 degrees", doubling
  the episodes of the same five sheets added 0.9 points (frozen probes, three seeds each; measured; F, data amount).
  This matches studies in which generalisation to new environments grew with the number of training environments
  rather than demonstrations (Lin et al. 2025, ICLR, Sec. 4.1 to 4.3; checker: supported. Xie et al. 2023: the gap
  fell from about 0.4 to under 0.1 as training environments went from 5 to 100; checked by the augmentation checker).
- **Six tracks are a small sample of tracks.** The held-out-sheet share within 45 degrees has a between-sheet
  standard deviation of 9.7 points (classes) and 8.9 points (exact step), so the range for "a new track like these
  six" is wide: 0.475 to 0.678 and 0.498 to 0.685 (t interval over the six sheets; calculated; V2).
- **Each fold needs its own baseline.** "Always forward" gets 0.747 of the moving pictures within 45 degrees on sheet
  01 but 0.402 on sheet 06 (measured; V1). On sheet 02 the frozen class probe (0.530) is below always forward (0.565),
  which the six-fold mean (0.577 against 0.538) hides (measured; V1).

### 1.3 What the gap between held-out episodes and held-out sheets means

The saved predictions of the split experiment (frozen ResNet-18 grid features, 9.11.26) were re-scored sheet by sheet.
The re-scoring reproduces every stored fold number exactly: within 45 degrees, the number of moving pictures, and the
stop outcomes, for both splits and both designs (measured; V1 against R, splits.json).

**Table 1. Share of moving pictures whose predicted push direction is within 45 degrees of the demonstrated one**
(measured; V1; last column from A, table 5.1).

| Sheet | Moving pictures (episodes) | Classes, held-out episodes | Classes, held-out sheet | Gap | Exact step, held-out episodes | Exact step, held-out sheet | Gap | Always forward | Pictures with a twin in another episode closer than their own neighbour |
|---|---|---|---|---|---|---|---|---|---|
| 01 straight | 569 (19) | 0.764 | 0.756 | +0.9 | 0.752 | 0.747 | +0.5 | 0.747 | 16.2 % |
| 02 arc | 2,372 (44) | 0.540 | 0.530 | +1.1 | 0.629 | 0.583 | +4.6 | 0.565 | 17.7 % |
| 03 two curves | 3,458 (45) | 0.654 | 0.565 | +8.9 | 0.692 | 0.624 | +6.9 | 0.522 | 51.4 % |
| 04 meander | 5,271 (45) | 0.659 | 0.559 | +10.0 | 0.711 | 0.548 | +16.3 | 0.485 | 50.5 % |
| 05 loop | 2,471 (30) | 0.705 | 0.583 | +12.1 | 0.754 | 0.563 | +19.0 | 0.509 | 41.7 % |
| 06 dashed loop | 1,809 (25) | 0.690 | 0.467 | +22.3 | 0.749 | 0.483 | +26.6 | 0.402 | 46.3 % |
| Mean of the six sheets | | 0.669 | 0.577 | +9.2 | 0.714 | 0.591 | +12.3 | 0.538 | |

Fold spreads as stored: classes 0.647 to 0.664 over the five episode folds against 0.467 to 0.756 over the six sheet
folds; exact step 0.698 to 0.720 against 0.483 to 0.747 (measured; R, splits.json). The episode folds look steady only
because each one mixes all six sheets. Pooled over all 15,950 moving pictures the two splits give 0.655 against 0.556
(classes) and 0.707 against 0.572 (exact step) (measured; V1).

**The stop suffers most.** Stopped right (within 2 cm of where the demonstration stopped, first picture past 0.5):
classes 106 of 208 episodes with episodes held out, 54 of 208 with sheets held out; exact step 119 against 58. Never
stopped: 71 against 121 (classes), 34 against 85 (exact step). On the dashed loop the class probe stopped right in 12
of 25 episodes with episodes held out and in 1 of 25 with the sheet held out (measured; V1).

What this means (judgement, unless marked):

1. The gap is what a model gains from having seen the same track, recorded in the same sitting, during training. It
   is near zero on the straight line and the arc, and 7 to 27 points on the four hard sheets. For the stop, holding
   out the whole sheet halves the right stops (106 to 54 of 208 episodes for classes, 119 to 58 for the exact step).
2. Two measured facts line up with it. On sheets 03 to 06, 42 to 51 % of the pictures have a twin in another episode
   that is closer than their own neighbour in time, against 16 to 18 % on sheets 01 and 02 (measured; A, table 5.1).
   And only sheets 03 to 06 hold features no other sheet shows: the two hairpins of 04, the solid crossing of 05, the
   dashes and L-shaped crossing of 06, and bends under 2.5 cm radius only on 03 and 04 (measured; A, view_and_tracks-5
   to -7). These runs cannot tell which of the two causes the gap (remembered views, or track features that only that
   track teaches). A leave-one-block-out test inside sheets 02 and 04 can, because twins do not survive a change of
   recording block (0.000 to 0.003; measured; A, duplicates_leakage-3).
3. For the exam: an unseen track made only of familiar features should score like sheets 01 and 02, where both splits
   agree; an unseen track with a new feature should score like a held-out hard sheet; and the provided sheets, run in
   a new session at the exam, should land between the two splits. So the held-out-sheet number is the one to plan
   with for unseen tracks, and the held-out-episode number is at most an upper bound for the known sheets.
4. The gap depends on the design (mean +12.3 points for the exact step against +9.2 for classes; measured; V1), so a
   comparison of designs on held-out episodes partly compares how well they remember. Held-out episodes must never be
   used to choose a model, a setting or an epoch, which the audit already found for early stopping (A, section 11.2,
   item 4).

### 1.4 How much an offline number moves by chance

**Table 2. Noise in held-out-sheet numbers** (frozen ResNet-18 grid probes; measured; V2 and V3, from R, splits.json and
seeds.json).

| Source of noise | Size |
|---|---|
| Which episodes are on the held-out sheet (episode bootstrap, 4,000 draws, 95 % interval) | about plus or minus 2.0 to 3.1 points on sheets 02 to 06 and 5.7 to 6.0 on sheet 01 (19 episodes); six-sheet mean plus or minus 1.4 |
| Training seed, within 45 degrees (four seeds) | one sheet: up to 9.1 points (classes, sheet 01) and 9.8 points (exact step, sheet 06); six-sheet mean: 3.1 and 2.4 points; path within 45 degrees, six-sheet mean: 3.4 and 3.5 points |
| Training seed, stopped right | one sheet: ranges of 11 to 26 points (classes) and 7 to 32 points (exact step); six-sheet mean 5.9 and 4.3 points |
| Which tracks (the six sheets as a sample of tracks) | between-sheet standard deviation 9.7 and 8.9 points |

Two lessons:

- **The seed matters more than the episodes, and a bootstrap of one run hides it.** Seed 12 against seed 11 of the same
  class probe differs by -9.1 points on sheet 01, with an episode-bootstrap interval of -13.2 to -5.2: it looks
  "significant", yet only the seed changed (measured; V3).
- **One run per design is not enough to choose.** With one seed, exact step minus classes on held-out sheets was +5.4,
  +5.9 and +1.6 points on sheets 02, 03 and 06, but -0.9, -1.1 and -2.0 on sheets 01, 04 and 05 (measured; V3). With
  four seeds per design, the exact step led on all six sheets, by +0.5 to +8.7 points (mean +4.3) (calculated from the
  per-sheet values in V2).

Paired differences of single end-to-end runs on the same pictures (second minus first, 95 % episode-bootstrap
interval; one seed each; measured; V3):

| Comparison | Sheet 03 | Sheet 05 | Sheet 06 |
|---|---|---|---|
| ResNet-18 fine-tuned, exact step minus classes | +3.6 (+2.0 to +5.3) | +0.8 (-1.9 to +4.0) | +3.5 (+0.2 to +6.9) |
| ResNet-18 fine-tuned classes minus small CNN from scratch | +2.9 (+1.1 to +4.6) | +2.5 (-0.2 to +4.8) | +1.7 (-0.7 to +4.0) |
| ResNet-18 fine-tuned exact step minus frozen exact-step probe | +7.6 (+6.0 to +9.1) | +10.4 (+7.7 to +13.1) | +19.2 (+15.2 to +22.6) |

Only the last row is clearly larger than the seed noise of the probes. Offline evidence therefore says "fine-tune end
to end" firmly, and says little yet about exact step against classes, or ResNet-18 against the small CNN (judgement;
the seed noise of end-to-end runs is not measured yet).

### 1.5 Which offline numbers predict closed-loop success

From the literature (each claim checked against its source by an independent checker; verdict in brackets):

- **Validation loss is the weakest guide.** In robomimic the checkpoint with the lowest validation loss was 57 to 100 %
  (relative) below the best checkpoint, the last checkpoint 12 to 35 % below, and the last beat the lowest-loss
  checkpoint in all 8 rows of Table 29 (Mandlekar et al. 2021, App. G; percentages calculated by the checker)
  [supported]. In a 2026 preprint, validation action error correlated -0.93 (Spearman, 16 checkpoints) with real
  success, the validation loss 0.00 (Allshire et al. 2026, Sec. 3.4) [supported; the checker notes that their loss is a
  flow-matching loss, unlike ours, and that the 16 points mix model families].
- **Direction and thresholded measures beat mean squared error.** Unseen town, the better half of the checkpoints of
  45 driving models: correlation with success 0.65 for a quantised direction error and 0.64 for a thresholded error,
  against 0.39 for the mean squared error (Codevilla et al. 2018, Sec. 4.3, Fig. 3) [supported].
- **Recovery pictures in the validation set help.** The same paper's MSE correlation rose from 0.39 to 0.54 with
  recovery data and to 0.77 with side cameras (Fig. 2) [corrected: holds in the main-paper condition; in the training
  town other metrics did as well].
- **Offline numbers spot bad models but rank good ones poorly.** Correlations of 0.56 to 0.83 over all models fell to
  0.21 to 0.55 among the better half in the training town (Codevilla 2018, supplement) [supported]. Within one data
  budget the link between hold-out error and success was not significant, and seeds with almost the same hold-out
  error started the task in 2, 3, 6 and 10 of 10 runs (Cowling 2026, preprint, Table 3, Sec. 5.7) [supported]. Twelve
  seeds of one driving model varied by up to 42 % in success (Codevilla et al. 2019, Table 3) [supported].
- **Some failures have no offline counterpart.** A policy that never starts (96 % of one family's runs never left the
  rest pose; Cowling 2026, Sec. 5.4) [supported]; of the 53 to 75 driving scenarios with small offline errors, 75 to
  80 % still failed online, against 80 to 84 % of all scenarios (Haq et al. 2021, Table 2) [corrected numbers]; and
  compounding errors, up to T squared times the per-step error in theory (Ross et al. 2011, Theorem 2.1) [supported; a
  proof, not a measurement].
- **Inputs that reveal the previous move flatter offline scores.** Given the previous action, policies reached similar
  (in Pong lower) validation loss but consistently lower reward (de Haan et al. 2019, Sec. 3.2) [corrected]; a policy
  with history had a lower test loss than the single-picture policy in all six tasks, while a method with higher test
  loss earned more reward (Wen et al. 2020, App. 10.6) [checked by the history checker].
- **A closed-loop measure without reward ranks better than held-out action error.** Over 10,000 imitation agents, a
  distance between the states reached in rollouts and the demonstrations' states beat held-out action MSE for choosing
  settings (Hussenot et al. 2021, Sec. 2.1 and 3.1) [checked by the checker]. For us: how far the puck's path strays
  from the line.

Our own data show the same traps (measured):

- **The single-step measure rewards copying.** Repeating the previous move (information a picture-only model does not
  have) lifts step within 45 degrees on held-out sheets from 0.538 to 0.736 (+19.8 points), but path within 45 degrees
  (the direction to where the tip is three steps later) only from 0.604 to 0.648 (+4.4) (R, baselines.json).
- **A history input looked better on the mean and worse where it matters.** Adding the change since the previous
  picture raised step within 45 degrees from 0.577 to 0.603, while the worst fold fell from 0.467 to 0.454 and the worst
  path fold from 0.526 to 0.498 (R, comparison.md, history experiment).
- **Recovery pictures rank models differently.** The pictures after a hand correction (5,996 of the 15,950 moving
  pictures; the closest thing to recovery pictures in our data) are harder for every model on held-out sheets: 0.494
  against 0.594 (frozen classes), 0.605 against 0.731 (end-to-end exact step, sheets 03, 05, 06); always forward 0.431
  against 0.555. On them the end-to-end exact step's lead over classes vanishes (0.605 against 0.605; +1.5, -2.5 and
  -0.8 points on sheets 03, 05, 06), while on the other pictures it leads by 4.3 points (0.731 against 0.688) (V4).
- **Validation loss has not misled us yet, but only because training is short.** In the end-to-end runs the validation
  loss (on held-out episodes of the training sheets) was lowest at the last epoch in all three exact-step folds (8 of 8
  epochs) and at epochs 7, 5 and 7 of 8 in the class folds (V6).

### 1.6 What robot trials can show

All numbers in this subsection are calculated (V7 and I), assuming independent trials with one fixed success rate per
model, as every cited source does. V7 reproduces the numbers of the statistics literature note, which its checker
recomputed independently.

**Table 3. 95 % intervals for a success rate**

| Observed | Wilson | Exact (Clopper-Pearson) |
|---|---|---|
| 5 of 10 | 0.24 to 0.76 | 0.19 to 0.81 |
| 7 of 10 | 0.40 to 0.89 | 0.35 to 0.93 |
| 9 of 10 | 0.60 to 0.98 | 0.56 to 1.00 |
| 10 of 10 | 0.72 to 1 | 0.69 to 1 |
| 14 of 20 | 0.48 to 0.85 | 0.46 to 0.88 |
| 18 of 20 | 0.70 to 0.97 | 0.68 to 0.99 |
| 20 of 20 | 0.84 to 1 | 0.83 to 1 |
| 21 of 30 | 0.52 to 0.83 | 0.51 to 0.85 |
| 24 of 30 | 0.63 to 0.90 | 0.61 to 0.92 |
| 27 of 30 | 0.74 to 0.97 | 0.74 to 0.98 |
| 30 of 30 | 0.89 to 1 | 0.88 to 1 |

At an observed 80 % the Wilson interval is plus or minus 23 points with 10 trials, 17 with 20 and 14 with 30. The
Wilson interval is the one recommended for small samples (Brown et al. 2001, abstract) [supported].

**Table 4. How many trials tell 70 % from 90 %**

| Question | Answer | Source |
|---|---|---|
| Do 7 of 10 and 9 of 10 differ? | No: the difference interval is -0.16 to +0.51, and the chance that the second is really better is 0.845 | V7 |
| Is one model above a 70 % bar, if it is really at 90 % (exact one-sided test, 5 %, power 0.8)? | 28 trials (power 0.86), safely from 33; with 30 trials it passes at 26 of 30 or more (power 0.825) | V7 |
| Which of two models is better, if we only want to choose (ties split)? | right 87 % of the time with 10 trials each, 95 % with 20, 96 % with 24, 98 % with 30 | V7 |
| The same for 90 % against 80 % | 81 % with 20 each, 83 % with 24, 92 % with 50 | V7 |
| Prove that 90 % beats 70 % (exact unconditional test, power 0.8) | 49 to 50 trials per model one-sided, 61 to 63 two-sided | statistics note, Sec. 2.4 [supported] |
| The same with paired starts (mid-p McNemar, two-sided) | 33 to 63 pairs, fewer when a start that is hard for one model is hard for both | statistics note, Sec. 2.5 [supported] |
| Show a gap of 8 to 9 points | about 500 trials per model | Snyder et al. 2025, Table II [supported] |

**Trials on one track share its difficulty.** With a guessed intra-track correlation of 0.15, 30 trials on 3 tracks
count like 12.8 independent trials, on 10 tracks like 23.1 and on 15 tracks like 26.1 (calculated; V7, design effect
1 + (m - 1) x correlation; the correlation is a guess). The statistics checker's simulation gave 83 % coverage of a
nominal 95 % interval for 3 tracks x 10 trials at that correlation, against 93 % for 10 tracks x 3 trials (statistics
check, claim 41) [supported]. Many tracks with few trials each say more about "new tracks" than few tracks with many
trials.

**An interim look saves robot time without hurting the choice.** Running 12 of 24 paired blocks and stopping when one
model leads by 3 or more successes keeps the chance of choosing the better of a 90 % and a 70 % model at 0.961 (fixed
24 blocks: 0.962), stops early in 48 % of such comparisons and cuts the expected blocks to 18.2. When the two models
are equal it stops early in 26 % of comparisons, when the choice does not matter (calculated; I). It must be fixed in
the plan beforehand, and it supports a choice, not a claim of significance (Snyder et al. 2025, Sec. III and V-B1:
fix the level and the largest number of trials in advance; re-testing after every trial breaks the error rate)
[corrected wording, supported].

**Robot time.** A step takes 1.22 s (measured in the replays; F). With the median demonstration steps and a guessed
1 minute to lay the puck and start, a trial takes 1.6 minutes on sheet 01 to 4.4 minutes on sheet 04; if a model needs
1.5 times the steps and a reset takes 2.5 minutes (both guesses), 3.5 to 7.6 minutes (calculated; F and V7).

**A step limit must be set.** Standard-start demonstrations needed a median 2.14 steps per cm of line (90th percentile
2.75, at most 3.07; 71 episodes); over all 208 episodes at most 3.32, and none needed 4 (measured; V5). The tool's own
budget of 2 steps per cm (switched off by default; A, code_actions note on config.py) would cut off more than half of
the standard-start demonstrations.

## 2. Recommendation

### 2.1 Offline: what to compute and how to choose

1. **Split.** Choose only on leave-one-sheet-out (six folds). Report every fold with its always-forward rule and the
   worst fold. Report held-out episodes, if at all, only as an upper bound for the known sheets. Add leave-one-block-out
   inside sheets 02 (two blocks) and 04 (three blocks) as the "known sheet, new session" measure. Never split pictures at
   random; never choose epochs or settings on held-out episodes of the training sheets (A, section 11.2, items 1 to 4).
2. **Measures, in this order,** fixed in a list before the next runs (as Abbas et al. 2026, App. F, advise; checked by
   the offline-evaluation checker):
   1. path within 45 degrees, mean over the six folds and the worst fold: the main direction measure, because copying
      inflates it least (+4.4 against +19.8 points);
   2. step within 45 degrees beside it;
   3. the stop per episode under the run-time rule (threshold tuned on validation episodes, two pictures in a row):
      stopped right, early and never, as counts of the 208 episodes; early stops weigh most, because an early stop ends
      a run as a failure;
   4. path and step within 45 degrees on the pictures after a hand correction (the recovery measure);
   5. the twist sign where the demonstration twisted.

   Do not choose on validation loss or on 8-way class agreement, which hardly separates anything (always forward
   40.3 %, picture-free rules 39.9 to 40.4 %, a leaky look-up 42.3 %; measured; A, section 5).
3. **Noise rule.** Count an offline difference only if it is larger than the seed noise (about 3 points on a six-fold
   mean of within 45 degrees, about 10 points on one sheet, about 30 points on one sheet's stop share) and has the same
   sign on at least five of six sheets; compare designs on at least three seeds each. Otherwise call it a tie and let
   the robot, or simplicity and speed, decide (judgement, thresholds from Table 2).
4. **Checkpoints.** Fix the training length once from the held-out-sheet curve, then always keep the last checkpoint.
   For the ResNet-18 trained end to end at today's settings that is 8 epochs, which the final candidate already uses.
   Never pick an epoch per fold or by validation loss (robomimic, above; V6).
5. **Shortlist.** Send at most two candidates to the robot: the final candidate (ResNet-18 fine-tuned, exact step, own
   stop output; running since 18:18 on 9.11.26) and one alternative that differs in one thing the offline numbers could
   not settle (classes, the small CNN, or one extra picture after the copying check of Chuang 2022, which the history
   checker confirmed as an offline test). Write each candidate's offline numbers into the plan as a calibration table,
   to be set beside its robot results.

### 2.2 Decisions only the robot can make

| Decision | Why offline numbers cannot make it | Where it is tested |
|---|---|---|
| The final choice between the two shortlisted models | Their offline gaps (0.8 to 3.6 points between end-to-end designs) lie inside the probes' seed noise (Table 2) | Phase 1 |
| The stop rule (threshold, pictures in a row), and stopping at a line end far from the paper edge | Tuned thresholds varied from 0.05 to 0.30 between end-to-end folds and runs (V6); the stop share moves up to 32 points with the seed (Table 2); every training line end lies 2.35 to 8.03 cm from a paper edge and the stop pictures show no line ahead (measured; A, view_and_tracks-2) | Phase 0, validation tracks |
| How many steps of a predicted chunk to execute per picture (1, 2 or 3) | Under action noise, executing the whole chunk lost to re-planning (simulated Push-T with a VQ-BeT policy: success 26.9 against 38.3 % and 13.0 against 29.5 %), without noise it won (64.0 against 48.9 %) (Liu et al. 2025, Table 1) [supported] | Phase 1b, if chunks are used |
| Any picture history | Offline gains are the kind that hide closed-loop losses (1.5) | Only after the copying check, as a Phase-1 candidate |
| Recovery from drift, stalls, dithering, never starting | No offline counterpart (Cowling 2026; Haq et al. 2021; Ross et al. 2011) | Failure types in every trial |
| Finishing time, the exam's second criterion | No study measured it offline (offline-evaluation checker); it depends on the speed cap (20 % now), the steps taken and hesitation | Phase 1b |
| Executor settings: wrist rule, step length, clamps, refused moves, the workspace guard for the table in use | Only the robot can refuse or collide (A, code_actions note) | Phase 0 |
| Light, table and room | No split separates a new room (A, duplicates_leakage-15 and view_and_tracks-15) | Light block; ask the professor |
| Keeping or dropping score-1 episodes | The audit left it to a closed-loop test (A, section 11.2, item 19) | Only if robot hours remain |

### 2.3 The robot test protocol

**A written plan, committed to git before the first trial, under the students' own names.** It fixes the candidates by
code letter with their files and offline numbers, the frozen executor settings, the tracks, the starts, the complete
trial order (generated from a fixed random seed), the success rule, the list of void faults, the analysis and the
decision rules. Any later change is listed as a deviation with its reason (Vincent et al. 2024, Sec. III-B, and
Kress-Gazit et al. 2024, Sec. 2.3 [supported]; Snyder et al. 2025, Sec. III, in the checker's corrected form: the
significance level and the largest number of trials are fixed before the first trial). The plan doubles as the
checklist the project rules ask for before every robot session.

**Tracks, in four roles** (judgement, following the statistics and corrections checkers):

- *Known sheets* 01 to 06.
- *Practice tracks:* new printed tracks for settings pilots and correction rounds; once corrections are recorded on
  one, it is a training track.
- *Two validation tracks:* used only to choose between candidates, never trained on. Between them they carry the two
  cheapest risky features the audit names: a line end at least 8 cm from the paper edge, and parallel line parts closer
  than 3 cm (A, view_and_tracks-16).
- *At least ten sealed test tracks:* A4, 1-cm line, round end caps, drawn by a written random rule (a small script is
  easiest): line length within the six sheets' 23 to 72 cm, every point inside the arm's demonstrated reach (tip 13.0 to
  48.5 cm from the base; A, view_and_tracks-16), a start that fits the team's start mark, and features drawn at random
  per track (solid or dashed, a crossing or none, an end near or far from the paper edge). The script's seed is in the
  plan; the printed tracks stay out of sight until Phase 2 and are used once. If the team retrains after Phase 2, a
  second sealed set is printed for any new final number.

**Starts.**

- Main start, the standard start of the recordings: the sheet laid and taped as recorded, the puck centred on the
  round end cap of the chosen end, the robot at its taught start pose with the block just behind the puck (measured
  set-up facts; recording-tool memory). The puck is placed with the recording tool's reference-picture overlay (the
  "ghost", commit 12c40c5), so both models start from the same picture.
- Both ends of each known sheet where the recorded lay-out allows, balanced over the trials (the data hold both ends of
  every sheet: corrected A/B 16/4, 12/33, 25/20, 33/12, 9/21, 15/10; A, section 11.1).
- New tracks are laid on the start mark the team chooses for the exam (the exam-start note of 6.11.26 recommends one
  taped mark about 17 cm in front of the robot's centre; an opinion, not yet a decision).
- Offset and turned starts, as in the data, only in a separate robustness block if hours allow, reported apart.

**Success rule (primary, "finished"), all four must hold:**

1. the policy stops by itself (its stop output under the frozen stop rule) within the step limit of 4 steps per cm of
   line (no demonstration needed more than 3.32; V5);
2. at that moment the puck's centre lies within 2.0 cm of the centre of the line's end cap, measured on an overhead
   photo taken right after the stop (the sheet's 29.7-cm edge gives the scale; 2.0 cm is the tolerance of the offline
   "stopped right" measure);
3. during the run the puck never left the line (a strip of bare paper visible between the puck and the line's course;
   for a dashed line, the course through the gaps) and it left every crossing on the line's own branch, judged live and
   checked on the overhead video when in doubt;
4. nobody touched the robot, the puck or the sheet, and no safety stop happened.

Secondary, also fixed in advance: the same rule with 1.0 cm and with 3.0 cm in point 2, and "reached the end without
stopping by itself". When the exam's rule is announced, all trials are re-scored under it, beside the pre-registered
rule, which stays the primary one.

**Recorded in every trial:** time from the first model command to the stop and the number of steps (from the tool's
log); the furthest progress along the line as a share of its length, and how far the puck's path strayed from the line
(computed afterwards from the logged tip path and the track's geometry, checked on the video); the first failure type:
left the line, wrong branch at a crossing, stopped early (more than 3 cm before the end), stopped 2 to 3 cm short,
never stopped (step limit reached, or the puck pushed past the end or off the paper), stalled or dithered (under 1 cm
of progress over 10 steps), refused moves or a joint limit, safety stop. *Void* (re-run at the end of the session and
reported) only for faults that are not the policy's doing: lost robot connection, no fresh camera picture, an operator
set-up error, a person in view. A policy that wanders or stops is never void. The operator ends a run early only when
the puck has left the paper or the arm has not moved for 5 steps, or for safety.

**Light and drift.**

- Main condition: the ceiling light of most recordings (natural light only in 24 to 25 of the 210 episodes, on sheets
  02 and 04; measured; A, sections 4 and 9).
- At the start of every session: the camera check against the stored reference picture. The view jumped once by about
  15 px (8.11.26, 19:33; measured; A, duplicates_leakage-9); if the reading differs from the usual one, stop and follow
  the camera-mount checklist, never touch the camera.
- Log date, time, light and the first picture's brightness of every trial; flag trials outside the training range
  (the training pictures reach from about 0.67 of the median brightness, the 1st percentile of the natural-light
  pictures, up to 1.12; measured; A, section 11.2, item 22).
- One uncounted warm-up run on a practice track per session.
- Never run model A in one session and model B in another: interleave them inside a session in the pre-generated random
  order (Bose and Kuang 2021: weekly drift over five months on unchanged robots, removed by random assignment)
  [supported].
- A separate light block of 6 trials (sheets 03, 05 and 06, once from each end, with the chosen model) under a changed
  light, reported apart and never pooled with the main result.

**Roles and blinding.** One person keeps the code-letter key and runs the program, the other places the puck and
scores; scores are written before the key is opened; a borderline case is decided from the photo and video by the
written rule. With two people the separation is imperfect, and the report says so (Kress-Gazit et al. 2024, Sec. 2.3;
TRI LBM Team 2025, Sec. 4.1.1) [supported].

**Phases** (robot time calculated from V7 and I; the slow case uses the guessed 1.5 times the steps and 2.5-minute
resets; breaks, warm-ups and void re-runs not included):

| Phase | What | Trials | Robot time (slow case) | Reported as |
|---|---|---|---|---|
| 0, pipeline check | the final candidate on sheet 01, 3 runs per end: signs, the stop, the step limit, refused moves; measures the real reset time and steps per run, to re-plan the hours | 6 | 10 min (21) | not a result |
| 1, model choice | 24 paired blocks: sheets 04, 05 and 06 x 2 ends x 2 (12 blocks) and the two validation tracks x 2 ends x 3 (12 blocks), mixed in one random order; in each block both candidates run from the same start in random order; interim look after 12 blocks (lead of 3 or more stops) | 48 (about 36 expected at a 20-point gap) | 2.7 h (4.8); about 2.0 h (3.7) expected with the interim look | exploratory |
| 1b, speed (if time is to be improved) | the chosen model at the 20 % speed cap against one faster setting (a higher cap, or 2 steps per picture), 12 paired blocks on sheets 03 to 06 and the validation tracks, decided by finish first, then time | 24 | about 1.3 h (2.4) | exploratory |
| 2, final test | the chosen model and settings: the six known sheets x 5 trials (ends balanced) and the ten sealed tracks x 3 trials, in one random order over two sessions | 60 | 3.1 h (5.7) | the reported result |
| Light block | the chosen model, 6 trials under a changed light | 6 | 0.3 h (0.5) | reported apart |

All phases without 1b: 6.3 robot hours (11.4 in the slow case); with 1b about 7.6 (13.8) (calculated; V7, I).
Spending the comparison trials on the harder sheets and the validation tracks follows the TRI team, who made their tasks
hard enough to sit near 50 % success so that differences can show (TRI LBM Team 2025, Sec. 3) [supported]; on sheets 01
and 02 both candidates are likely to finish (guess).

**Decision rule of Phase 1** (fixed in advance): the candidate with more successes over the blocks run; a tie goes to
the candidate that wins more blocks under "finish first, then time"; a remaining tie goes to the simpler and faster
model. Report the counts, Newcombe's interval for the difference and the chance that the winner is really better (as
Kress-Gazit et al. 2024, Sec. 4.2, and the TRI team do) [supported], and claim no significance. The winner's Phase-1
score is not its final score, because choosing on a score makes that score optimistic (Cawley and Talbot 2010,
abstract, cited by the statistics checker).

**Analysis of Phase 2** (fixed in advance):

- Primary result: finishes on the sealed tracks, k of 30, with its Wilson interval, and per track k of 3; the report
  names the number of tracks, because trials on one track are not independent.
- Secondary: the known sheets, k of 30 and per sheet k of 5.
- A bar, if wanted: "above 70 % on new tracks" passes at 26 of 30 or more (exact one-sided test; power 0.825 if the
  true rate is 90 %; V7).
- Time: median and range of finishing time and steps per sheet among finished trials. Never compare two models' times
  from finished trials only, which is biased when their finish rates differ (statistics checker); use the paired
  blocks.
- Failure types with counts and one picture each; every void trial with its reason; the calibration table of offline
  numbers against robot results.

**Session rules from the project's working rules:** a checklist before every session; motors off (FreeMotion) before
anyone touches the arm, never push on a held arm; only one program connected to the robot at a time; the camera and its
focus ring are not touched; only the automatic calibration after a restart; the robot stays away from the table edge;
the step limit is switched on.

**Proposed dates** (judgement; the team decides): Phase 0 as soon as the final candidate is chosen (week of 16.11.26),
Phase 1 by about 27.11.26, correction rounds on practice tracks until about 10.12.26 if Phase 1 shows failures, Phase 2
by 18.12.26, two days before the robot cut-off of 20.12.26.

### 2.4 Backup

- **Robot hours short:** Phase 1 with 12 blocks only (it still chooses the better of a 90 % and a 70 % model 89 % of the
  time; calculated; I), and Phase 2 with the six known sheets x 3 and the ten sealed tracks x 2 (38 trials), reported as
  exploratory with its intervals.
- **Only one candidate survives offline:** skip Phase 1, run Phase 0 and Phase 2 with the final candidate, and spend the
  saved hours on a correction round on practice tracks.
- **Phase 0 shows an executor fault:** fix it and repeat Phase 0; no comparison runs before six runs on sheet 01 without
  a void fault.
- **Both candidates finish fewer than half of the Phase-1 trials:** do not run Phase 2; record corrections on the failing
  situations (practice tracks and known sheets), retrain, and repeat Phase 1; the sealed tracks stay sealed.

## 3. What is still unknown

- The exam's rule: how "finished" is judged, the time limit, the number of attempts, who lays out the sheets, when the
  clock starts and stops, the table and the light. The course page says the Project 6 evaluation format is not
  announced yet (01_course/moodle_snapshot.md).
- Whether the large gap on the hard sheets comes from remembered views or from features only that track teaches
  (leave-one-block-out answers it).
- The seed noise of end-to-end runs: every end-to-end run so far has one seed.
- The final candidate's six-fold numbers. Its run (audit-cleaned rows, 8 epochs, last weights) started at 18:18 on
  9.11.26; at 19:13 only the sheet-01 fold was done: within 45 degrees 0.779 of 578 moving pictures, path within 45
  degrees 0.856 of 529, stopped right 16 of 20 episodes under the tuned two-in-a-row rule (early 3, never 1), 34.6
  minutes for the fold (measured; R, 2026-11-09_resnet18_final_candidate/fold_without_path_01.json). Sheet 01 is the
  easiest fold, so this says little yet; the five harder folds decide.
- How much better the six-sheet exam model is than the five-sheet fold models (the four-to-five-sheet step of +2.3
  points suggests a little; guess).
- The real reset time, steps per robot run and void rate (Phase 0 measures them); the robot hours above rest on guesses.
- The intra-track correlation of robot outcomes (0.05 to 0.30 are guesses).
- How well any offline number tracks this robot: known only from the calibration table after Phase 1.
- Whether the university GPU pool becomes usable (it would allow more seeds and candidates, not change the protocol).

## Sources (short names used above)

- **V1 to V7:** `validation_numbers_output.txt`, parts 1 to 7, printed by `validation_numbers.py` (both in
  C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/validation/;
  reads frames.csv of the v01 index and the saved predictions, writes nothing elsewhere).
- **I:** `interim_rule_numbers_output.txt`, printed by `interim_rule_numbers.py` (same folder).
- **R:** D:/AI_Project/.claude/worktrees/training-analysis/04_runs/ (comparison.md;
  2026-11-09_frozen_feature_probes/splits.json, seeds.json, baselines.json and predictions/; the end-to-end folders
  2026-11-09_(run name)/fold_without_path_XX.json and predictions_without_path_XX.npz).
- **F:** C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/main_session_facts.md.
- **A:** C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/audit/AUDIT_SYNTHESIS.md
  (section, table or finding named), and audit/code_actions/note.md.
- **Literature notes and their checks:** scratchpad/literature/offline_vs_closed_loop, evaluation_statistics ("the
  statistics note"), history_and_causal_confusion, action_chunking, interactive_correction, diffusion_and_generative and
  augmentation, each with check/note.md.
- **Team files:** claude_memory/project-recording-tool.md and ai-project-workspace-setup.md;
  06_deliverables/research_notes/Recording tool/2026-11-06_exam_start_opinion.md; 01_course/moodle_snapshot.md.
