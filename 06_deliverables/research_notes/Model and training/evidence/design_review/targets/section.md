# What the model should predict, and how it stops

*Section of the training design for Team M (Emranul Hoque, Marshal Sebastine), written 9.11.26. Nothing in this section was
run on the robot. Markers: [measured] computed from the data or read from a run's output file; [calculated] derived from
measured numbers with the assumption stated; [literature] a claim its checker marked supported or corrected (the
corrected form is used); [judgement] or [guess] our own reasoning. Short source names are explained at the end.*

## In short

The model should predict the next move as numbers rather than as one of the eight move names: how far the magnet tip
moves forward and sideways in the picture, and how much the forearm, and with it the camera, twists. On the held-out
sheets these numbers pointed the push along the line better than the names did (by 5 to 8 points out of 100 when the
whole network was trained), and only they learn the twist well; their weak spot is that they rarely back up, and
weighting their training loss to fix that cost more than it gained (section 2.2). The robot should take only the
direction from the predicted move and always step 1 cm, because the model shortens its steps to about 0.6 cm on curved
sheets it has not seen, which would slow the runs and can leave the robot standing still; the wrist and the hand stay
where they start. Stopping is the weakest part: on unseen sheets the stop output stopped in the right place in only 3
to 4 runs out of 10, and most early stops came where the paper edge, not the end of the line, filled the view; a loss
weighting tried here did not help, so the stop needs more varied line ends to learn from and robot tests on a printed
track whose line ends away from the paper edge.

## 1. The questions

| Question | Options weighed |
|---|---|
| What does the model predict? | (a) one of 8 picture-frame move names; (b) the exact step: tip forward and left (cm), forearm twist, rod lean and hand turn (deg); (c) both; (d) a chunk: the tip's place after each of the next 5 steps |
| Small motions (twists with the tip still) and the 13 forearm turns | the next real move, dropped, a class of their own, or ordinary exact steps |
| Twist, rod lean (wrist, joint 5), hand (joint 6) | predicted, or held by a fixed rule in the executor |
| Step length at test time | as predicted, fixed, or the prediction rescaled |
| Stop | a ninth class; an output of its own trained on the stop picture, or on every picture within 2 cm; a predicted distance to the goal; one picture or two in a row; how the threshold is set |

Words used below. A *frozen probe* is a small head trained on fixed ResNet-18 picture features (seconds per run); an
*end-to-end* run trains the whole network (about 25 to 35 minutes per held-out sheet on the laptop). *Held-out sheet*:
the model is trained on five sheets and scored on the sixth, the stand-in for an unseen track. *step 45*: the share of
moving pictures whose predicted push direction is within 45 deg of the demonstrated step; *path 45*: the same against
the direction to where the tip is 3 steps later, a less noisy target.

## 2. Evidence

### 2.1 What the demonstrations contain (audit-cleaned rows)

- **Rows.** 18,053 rows after the audit's cleaning, 18,032 in 209 episodes after the score and mistake rules; 259 stop
  rows (209 stop rows plus 50 duplicated final points relabelled as stop) [measured; targets/data_facts.out.txt].
- **The names drop the twist.** The move name explains 84.7 % and 81.4 % of the variance of tip forward and tip left, but
  only 16.3 % of the forearm twist; a name executed at its nominal angle is 11.95 deg off the demonstrated move on
  average; 18.8 % of the moves lie within 5 deg of a bin edge [measured; audit motion_statistics-13, class_balance-5,
  label_consistency-3].
- **The forearm twists on most steps.** 11,741 of 17,773 non-stop rows (66.1 %) twist by 2 deg or more (38.2 % on the
  straight sheet 01, 66.9 to 75.1 % on sheets 03 to 06) [measured; data_facts]; median 4.04 deg per step, 99th
  percentile 11.6 deg; consecutive twists share their sign 94.2 % of the time [measured; audit motion_statistics-4].
  The camera is within 5 deg of straight in only 8.2 % of the pictures [measured; data_facts].
- **Small motions are twists in place.** 1,765 small_motion rows remain: 1,687 twist by 3 deg or more (median 5.98 deg),
  only 8 are still rows [measured; data_facts]. They take 10.5 % of the step time (4.6 % on sheet 05, 15.0 % on sheet
  04) [measured; data_facts] and cluster at tight bends (23.8 % of the pictures at a line radius under 2.5 cm against
  about 10 % elsewhere) [measured; audit view_and_tracks-7].
- **The 13 turn labels** are large twists; 4 of them are recording stalls; after the audit's clip all 13 are 15-deg
  twists [measured; audit motion_statistics-4; data_facts].
- **Backing up is common enough to matter.** Backward moves (back, back_left, back_right) are 1,894 of the 15,995 moving
  rows (11.8 %) [calculated from the label counts in data_facts]; about half of them are ordinary pushing seen through a
  sideways-turned camera, and 55.5 % lie in replayed hand corrections, though only 15.6 % in a correction's first six
  rows [measured; audit class_balance-10].
- **The hand hardly moves.** It changes by 3 deg or more on 26 of 17,773 non-stop rows [measured; data_facts], but its
  angle differs visibly between start kinds, a shortcut risk [measured; audit motion_statistics-8].
- **The wrist drifts slowly.** Joint 5 changes by a median 0.0 deg per step (95th percentile 0.44), yet it is more than
  5 deg from its episode-start value in 28.0 % of the rows; 52 of 210 episodes held it within 1 deg throughout
  [measured; audit code_actions-3 and the code_actions note].
- **Step length is a recording rule.** A waypoint was cut after 1 cm of tip travel or 5 deg on any joint, so short steps
  are mostly steps where the forearm reached 5 deg first; the moving-row median is 1.09 cm [measured; audit
  motion_statistics-3, notes_context-3].
- **The stop is rare and its cue is the paper edge.** Stop rows are 1.4 % of the rows. 90.0 % of the stop pictures show
  no line in their top half, but so do 27.0 % of the pictures more than 3 cm from the goal (sheets 01 to 06: 41.4, 20.1,
  22.4, 20.9, 37.9 and 46.4 %) [measured; targets/stop_analysis.out.txt, using the audit's line_share_top_half of at
  most 0.002]. Every line end lies 2.35 to 8.03 cm from a paper edge [measured; audit view_and_tracks-2]. The row before
  the first stop row lies a median 0.65 cm from the goal (90th percentile 1.13 cm, never over 2 cm); 523 non-stop rows
  lie within 2 cm of the goal, and the demonstrators kept moving inside that circle [measured; data_facts; audit
  notes_context-4].

### 2.2 Names, exact step, both or a chunk (held-out sheets)

| Model | Held-out sheets | step 45 | path 45 | twist sign | Source |
|---|---|---|---|---|---|
| always forward (no picture) | 6 | 0.538 | 0.604 | | baselines.json |
| probe, names, most likely class | 6 | 0.577 (folds 0.467 to 0.756); 4 seeds 0.546 to 0.577 | 0.633; seeds 0.600 to 0.633 | 0.408 | designs.json, seeds.json |
| probe, names, mean direction of the class probabilities | 6 | 0.605; 4 seeds 0.578 to 0.605 | 0.682; seeds 0.654 to 0.682 | 0.408 | same files ("expected") |
| probe, exact step | 6 | 0.591 (folds 0.483 to 0.747); 4 seeds 0.591 to 0.616 | 0.673; seeds 0.673 to 0.708 | 0.700 | designs.json, seeds.json |
| probe, names plus exact, mean direction | 6 | 0.602 | 0.677 | 0.699 | designs.json |
| probe, chunk of 5 | 6 | 0.607 (0.536 to 0.777) | 0.697 (place 3 steps ahead: 0.722) | 0.699 | designs.json |
| end-to-end ResNet-18, names, most likely class | 03, 05, 06 | 0.654 (0.640 to 0.663) | 0.707 (0.704 to 0.711) | 0.499 (0.372 to 0.703) | ..._finetune_classes/metrics.json |
| end-to-end ResNet-18, names, mean direction | 03, 05, 06 | 0.667 (0.637 to 0.687) | 0.744 (0.728 to 0.754) | 0.499 | same file ("expected") |
| end-to-end ResNet-18, exact step | 03, 05, 06 | 0.681 (0.667 to 0.700) | 0.790 (0.777 to 0.802) | 0.772 (0.669 to 0.845) | ..._finetune_exact/metrics.json |
| final candidate (exact, cleaned rows) | 01 and 02 so far | 0.779 and 0.632 | 0.856 and 0.762 | 0.677 and 0.802 | ..._final_candidate/fold_without_path_01.json, _02.json |

- **Against the names as they would run (most likely class),** the exact step leads by 4.3 points on step 45 and 7.8 on
  path 45 over four probe seeds (means 0.603 against 0.560, 0.691 against 0.613), and end to end by 2.6 and 8.3 points
  (path: 6.7 to 9.8 per sheet); the seed range alone is about 3 points [calculated from comparison.md and the metrics
  files]. On known sheets (held-out episodes) it also leads, 0.707 against 0.655 [measured; splits.json].
- **Against the names read as a mean direction** (the probability-weighted average of the 8 nominal angles, which the
  names model also offers), the lead shrinks: in the probes 0.9 points on step 45 and 2.2 on path 45, inside the seed
  range; end to end 1.3 and 4.6 points (path: 2.4 to 6.3 per sheet, one seed) [calculated from the same files]. The
  clear gains of the exact step are the path direction end to end and the twist (sign right on 0.77 against 0.50 of the
  twisting rows) [measured].
- **Where the exact step loses: backing up.** On the 1,054 backward moves of the three end-to-end held-out sheets, the
  exact model's direction is within 45 deg for 0.392 of them, the names model's mean direction for 0.663; on the other
  6,684 moves the exact step is better, 0.730 against 0.674. The exact model predicts a straight-back move on 0.1 % of
  the pictures against 3.9 % demonstrated. Over the six probe sheets: 0.270 against 0.400 on backward moves, 0.613
  against 0.616 on the others. Without class weights the names model is as poor at backing up (0.294 most likely class,
  0.273 mean direction), so its edge comes from the class weights in its loss, which the exact loss lacks [measured;
  targets/per_class_direction.out.txt]. The final candidate is no better: within 45 deg on 0.054 of the 37 backward
  moves of sheet 01 and 0.139 of the 280 of sheet 02 (0.828 and 0.696 on the other moves), and never a straight-back
  move [measured; targets/final_backing_up.out.txt]. A regression output is the average of what was demonstrated after
  similar pictures [literature: Bishop 1994, supported], and rare backward moves are averaged into forward ones
  [judgement].
- **Weighting the exact loss is not a free fix (tested for this section).** With the names' square-root class weights
  on the exact-step loss (forward 0.58 up to back 1.99), a frozen exact probe on the cleaned rows backed up better but
  lost on all other moves, and so overall:

  | Frozen exact probe, cleaned rows, mean of six held-out sheets | seeds | backward moves within 45 | other moves within 45 | step 45 | path 45 | twist sign |
  |---|---|---|---|---|---|---|
  | unweighted (as the final candidate) | 2 | 0.266, 0.277 | 0.639, 0.638 | 0.599, 0.602 | 0.691, 0.692 | 0.677, 0.669 |
  | weighted by move name | 2 | 0.383, 0.419 | 0.580, 0.580 | 0.563, 0.566 | 0.642, 0.652 | 0.682, 0.685 |

  [measured; targets/exact_probe.out.txt]. The weights gained 11 to 15 points on backward moves but lost 6 on all other
  moves, 3.6 on step 45 and 4.5 on path 45, far beyond the spread between the two seeds (1 point or less) [calculated
  from the table]. Backward moves are 12 % of the moving pictures and the path direction is the better guide to following the
  line, so the trade is a bad one [judgement]. The names model read as a mean direction
  keeps both (backward 0.400, others 0.616 in the probes), which is why the backup below combines it with the exact
  twist.
- **Predicting both** adds nothing when the names part is read as the most likely class (0.565), but read as a mean
  direction it matches the better of the two in the probes (step 0.602, path 0.677, backward moves 0.363, twist sign
  0.699) [measured; designs.json, per_class_direction]; it was never trained end to end.
- **The chunk's** first step (0.607) lies inside the exact step's seed range (0.591 to 0.616); only its place 3 steps
  ahead (path 0.722) is above all four exact seeds (best 0.708), by 1.4 points [measured; designs.json, seeds.json]. It
  backs up no better (0.290) [measured; per_class_direction]. The end-to-end chunk run is still queued.
- **Small motions in a name model.** Next move 0.577, dropped 0.560, own class 0.560 on step 45, all inside the seed
  range [measured; designs.json]. The next-move rule copies a move measured in a later picture's frame (another class in
  its own frame for 16.0 % of the relabelled rows) and would push 1 cm where the demonstrator twisted in place
  [measured; audit class_balance further facts]. In the exact target a small motion is simply a step with a short tip
  move and a twist, so no rule is needed [judgement].

### 2.3 What the exact model really outputs (checked for this section)

The saved end-to-end exact models (sheets 03, 05 and 06 held out) were re-run on their own test pictures to read the
predicted step length and twist, which the saved predictions leave out; the re-run reproduces the saved stop
probabilities, angles and twists exactly (largest difference 0) [measured; targets/exact_inference.log].

| | moving rows | twists in place | stop rows |
|---|---|---|---|
| predicted step, median | 0.60 cm (sheets 03, 05, 06: 0.57, 0.61, 0.65) | 0.48 cm | 0.30 cm |
| demonstrated step, median | 1.09 cm | 0.20 cm | 0 |

[measured; targets/exact_analysis.out.txt and length_by_error.out.txt; 7,738 moving, 598 twist-in-place and 100 stop
rows]

- **The length shrinks most where the direction is unsure:** median predicted step 0.70 cm when the direction is within
  15 deg, 0.58 cm at 30 to 45 deg, 0.48 cm beyond 90 deg [measured; length_by_error]. The average of moves in scattered
  directions is a short vector [judgement]. The final candidate, trained on the audit-cleaned rows, shrinks far less on
  the straight sheet 01 (1.00 against 1.12 cm, median direction error 13 deg) but about as much on the arc, sheet 02
  (0.68 against 1.11 cm) [measured; targets/final_fold_check.out.txt]; a frozen exact probe on the cleaned rows predicts
  a median 0.60 to 0.62 cm (two seeds; 0.45 to 0.95 by sheet) [measured; targets/exact_probe.out.txt]. So the cleaning
  does not remove the shrinkage.
- **The twist looks shrunk, but mostly where its sign is a guess.** On the 5,781 rows that twisted 2 deg or more: median
  predicted 2.48 deg against 5.62 demonstrated; sign right 0.782; under 1 deg for 21.9 % of them; correlation 0.63
  [measured; exact_analysis]. Split by the predicted size: below 1 deg the sign is right on 0.565 of the twisting rows
  (demonstrated median 2.37 deg), at 3 to 5 deg on 0.907 (demonstrated 5.27 deg against 3.85 predicted), at 5 to 8 deg
  on 0.943 (6.06 against 5.63). A least-squares fit of the demonstrated twist on the predicted one has a slope of 0.97,
  so on average the predicted twist is already the right size; small predictions are small because the model does not
  know which way to turn [measured; targets/twist_calibration.out.txt; 8,610 non-stop pictures]. Final candidate: 1.28
  against 4.92 deg on sheet 01 (sign right 0.676), 3.43 against 5.27 deg on sheet 02 (0.802) [measured;
  final_fold_check].
- **Standing still.** On 204 of 8,133 pictures more than 3 cm from the goal (2.5 %), in 48 of 100 episodes, the model
  predicts a step under 3 mm with a twist under 1 deg; 79 % of them show no line ahead [measured; exact_analysis]. In a
  closed loop such a step brings back almost the same picture (still rows change the next picture by 0.25 grey levels
  against 8.9 for moving steps), so a deterministic model can repeat it for ever [measured; audit code_actions-11 and
  further facts]. Final candidate: 6 of 560 pictures in 2 of 20 episodes (sheet 01), 18 of 2,398 in 8 of 45 episodes
  (sheet 02) [measured; final_fold_check].

### 2.4 What the robot can execute, and what the camera then sees

- **Names** run today on the keyboard path, but as motions unlike the data: no twist, a vertical rod with a straight
  forearm, turns as lifted 30-deg moves, and small_motion ends the session with a KeyError [measured; audit
  code_actions-6]. A camera that never twists shows the views of the 8 to 9 % of training pictures taken with a
  straight forearm [measured; data_facts, code_actions-6].
- **The exact step** has no executor yet. The gamepad path clips the tip move to 3 mm to 1 cm and re-solves towards the
  least rod lean, which within 1 to 6 steps moves the camera a median 16.3 to 16.9 mm and the demonstrated picture
  centre 48 to 56 px [calculated; audit code_actions-4, -5]. Keeping joints 5 and 6 and solving joints 1 to 3 with the
  existing `solve_with_wrist_fixed` reproduces the next demonstrated view per step (picture shift 90th percentile 0.5 to
  0.7 px) [calculated; code_actions-4]. Holding joint 5 at its start value for a whole run differs from that run's
  demonstration by more than 20 px in 23.3 % of the rows [calculated; code_actions-4], but such fixed-wrist views exist
  in the 52 episodes that held the wrist [judgement]. The solver returns joint 6 = 0, so the executor must put the hand
  back itself [measured; code_actions-4].
- **Twist by a fixed rule instead of the model.** Holding the forearm still shows the rare straight-forearm views on
  curved tracks (the forearm ranged a median 155 deg within an episode) [measured; audit motion_statistics-5]; a rule
  that twists towards the side of the predicted move would copy a correlation (forward_left rows twist left in 65.3 %
  of cases, right in 11.6 %) [measured; audit class_balance-9] that the model's own twist output already gets right 0.77
  of the time [measured]. So the twist is predicted [judgement].
- **The lean output** has no executor anywhere; executing it would need a joint-5 search, and joint-level outputs risk
  the height (1 deg on joint 2 moves the tip 5.4 mm against a 5-mm clearance) [calculated; code_actions-14]. Its sign is
  right on 0.81 to 0.89 of the rows that leaned 1 deg or more [measured; metrics.json], but per step the lean follows the
  reach (correlation 0.713 with tip forward), which a fixed-wrist solve produces by itself [measured; audit
  motion_statistics-7].
- **A chunk** can reuse the replay's waypoint path, but every waypoint is still a separate blocking move with a fresh
  picture [measured; audit code_actions-15].

### 2.5 Step length and finishing time

| Step (plain steps: the next row is the next waypoint) | steps | median time per step | per cm |
|---|---|---|---|
| tip 9 to 11 mm, twist under 1 deg | 1,280 | 0.95 s | 0.90 s |
| tip 13 to 16 mm, twist under 1 deg | 790 | 1.01 s | 0.73 s |
| tip 16 to 20 mm, twist under 1 deg | 100 | 1.15 s | 0.67 s |
| tip 9 to 12 mm, twist 3 to 5 deg | 1,123 | 1.19 s | |
| tip 9 to 12 mm, twist 5 to 8 deg | 1,146 | 1.35 s | |
| twist in place, 5 to 7 deg | 1,084 | 1.12 s | |

[measured; data_facts; over all 17,653 plain steps the median is 1.226 s, of which 0.175 s is the fresh picture and the
decision]

- **The raw predicted length (about 0.6 cm on curved sheets):** a 6-mm move turns the leading joint by about 1.1 to 1.2
  deg, and moves of 1 to 2 deg take 0.70 to 0.73 s, so about 0.9 s per 0.6 cm, roughly 1.5 s per cm against 0.9 s:
  about 1.6 to 1.7 times the driving time [calculated from audit code_actions-9 and -14; the data hold few pure 6-mm
  moves to check it].
- **A fixed 1.0 cm** is the demonstrated median, so about the demonstrated time per cm [calculated]. The demonstrated
  runs took a median 31.5 to 168 steps per sheet, 38 s (sheet 01) to 205 s (sheet 04) of driving at 1.22 s per step
  [calculated; main_session_facts.md].
- **1.5 cm** would save about 20 % per cm on straight stretches (0.73 against 0.90 s) [calculated from the table], but
  it jumps further between pictures than most training steps, at bends with a 0.43 to 0.6 cm radius [measured; audit
  view_and_tracks-5]; executing policies faster than demonstrated cost success in simulation (Square 0.83 to 0.55)
  [literature: SAIL, supported].
- **Twist size:** a 1-cm step with a 5 to 8 deg twist takes 0.40 s longer than without [measured; table]; a model that
  under-twists saves time per step but may need more steps on curves [judgement].
- **Chunks executed 2 or 3 steps per picture** save only the 0.175 s of picture and decision on 1 of 2 or 2 of 3
  steps, 7 to 10 % [calculated]; blending moves with pyniryo's `execute_trajectory` is untested [measured: audit
  code_actions-15]. Executing chunks without looking lost under noise (Push-T, strongest noise: 13.0 against 29.5 %
  success) and on an imprecise robot (0 of 30 against 47 of 60), while a precise arm ran 6 of 16 predicted steps open
  loop at 95 % [literature: Liu et al. 2025, Lee et al. 2024, Chi et al. 2023; supported]. The chunking theory needs a
  system that stays on course without feedback [literature: Zhang et al. 2025, supported]; a puck that can slide off the
  block is not such a system [judgement].

### 2.6 How it stops (held-out sheets)

Per episode the robot stops at the first picture whose stop probability passes the threshold; "right" means within
2 cm of where the demonstrator stopped, "early" more than 3 cm before it, "never" no picture passes. "Other sheets"
chooses the threshold on the other held-out sheets of the same run, so a sheet never sees its own pictures; "best" picks
the best threshold on the held-out sheet itself, an optimistic ceiling.

| Stop design | fixed 0.5 (right / early / never) | tuned on validation episodes, 2 in a row | other sheets, 1 picture | other sheets, 2 in a row | best, 1 picture |
|---|---|---|---|---|---|
| probe, names, stop as a class | 0.258 / 0.112 / 0.599 | 0.317 / 0.274 | 0.284 / 0.251 | 0.370 / 0.228 | 0.389 / 0.215 |
| probe, names, own stop on stop picture | 0.321 / 0.209 / 0.436 | 0.382 / 0.255 | 0.326 / 0.356 | 0.341 / 0.368 | 0.405 / 0.233 |
| probe, names, own stop within 2 cm | 0.138 / 0.658 / 0.090 | 0.329 / 0.215 | 0.297 / 0.240 | 0.298 / 0.287 | 0.340 / 0.276 |
| probe, exact, stop on stop picture (4 seeds) | 0.289 to 0.332 / 0.243 to 0.390 | 0.221 to 0.323 / 0.341 to 0.443 | 0.228 to 0.294 / 0.221 to 0.332 | 0.251 to 0.274 / 0.245 to 0.352 | 0.354 to 0.401 / 0.210 to 0.306 |
| probe, exact, stop within 2 cm | 0.176 / 0.512 / 0.174 | 0.318 / 0.276 | 0.226 / 0.270 | 0.325 / 0.244 | 0.349 / 0.286 |
| probe, exact, predicted distance to goal | 0.110 / 0.013 / 0.853 | 0.148 / 0.035 | 0.097 / 0.210 | 0.290 / 0.153 | 0.340 / 0.192 |
| probe, chunk, stop on stop picture | 0.300 / 0.264 / 0.367 | 0.298 / 0.341 | 0.293 / 0.286 | 0.290 / 0.310 | 0.356 / 0.304 |
| end-to-end exact, stop on stop picture (03, 05, 06) | 0.363 / 0.254 / 0.368 | 0.414 / 0.350 | 0.364 / 0.265 | 0.324 / 0.422 | 0.429 / 0.225 |
| end-to-end names, stop as a class (03, 05, 06) | 0.154 / 0.106 / 0.733 | | 0.278 / 0.407 | 0.216 / 0.432 | 0.371 / 0.292 |
| final candidate, sheet 01 (20 episodes) | 0.60 / 0.20 / 0.05 | 0.80 / 0.15 (one picture: 0.65 / 0.10) | | | |
| final candidate, sheet 02 (45 episodes) | 0.533 / 0.044 / 0.400 | 0.422 / 0.178 (one picture: 0.556 / 0.089) | | | |

[measured; comparison.md and the fold files for the first two columns, targets/stop_analysis.out.txt for the others;
six folds for the probes]

- **No threshold rescues the stop on unseen sheets.** Even the best threshold per held-out sheet stops right in at
  most 0.43 of the episodes with one picture (0.44 with two) and early in 0.19 to 0.31 [measured; stop_analysis]. On
  known sheets the same exact probe stops right in 0.571 and early in 0.165 [measured; splits.json]: the stop transfers
  poorly to new tracks.
- **Early stops happen mid-track, where no line is ahead.** With the threshold from the other sheets and two pictures in
  a row, the end-to-end exact model stopped early in 38 of 100 episodes, a median 15.7 cm before the goal, all 38 at
  pictures with no line in the top half (against 33 % of all pictures more than 3 cm from the goal on these sheets);
  22 of the 38 were on sheet 05, whose loop runs along the paper edge [measured; stop_analysis]. With one picture it
  stopped early in 26 episodes, 25 of them at no-line pictures, a median 14.3 cm before the goal [measured;
  targets/early_stops_one_picture.out.txt]. More than 5 cm from the goal its stop probability reached 0.282 at the 90th
  percentile with no line ahead, against 0.002 with a line in view. For the probes the share of early stops at no-line
  pictures was 56 to 97 % against 27 % [measured; stop_analysis]. The final candidate does better on its first two
  sheets: with one picture over its own tuned threshold, 2 early stops in 20 episodes on sheet 01 and 4 in 45 on sheet 02,
  close to the goal (medians 3.2 and 4.1 cm); its stop probability on far no-line pictures is 0.002 and 0.026 at the
  90th percentile against medians of 0.918 and 0.724 on the stop pictures [measured; targets/final_stop_check.out.txt].
  That sheet 01, with no line ahead on 41.4 % of its far pictures, hardly raises the stop there is encouraging; sheets 05
  and 06 (37.9 and 46.4 %), where the earlier models failed most, are still to come.
- **Stop as a class is badly calibrated, not useless.** At 0.5 it never fires in 0.60 (probe) and 0.73 (end to end) of
  the episodes, but with a threshold from the other sheets the probe version is among the best (0.370 / 0.228)
  [measured]. Once the move is predicted as numbers there are no classes, so the stop needs an output of its own
  [judgement].
- **Training on every picture within 2 cm** fires far too early at 0.5 (early 0.51 to 0.66) and needs thresholds of 0.7
  to 0.9; it is then about as good as the stop picture [measured]. The predicted distance to the goal fails at the
  validation-tuned threshold (right 0.148), but with a threshold from the other sheets it has the fewest early stops
  (0.153), and those lie close to the goal (median 4.3 cm, against 10.6 cm for the stop-picture probe) [measured;
  stop_analysis].
- **Two pictures in a row** help some designs and hurt others (end-to-end exact: 0.364 / 0.265 with one picture,
  0.324 / 0.422 with two; final candidate with validation-tuned thresholds: sheet 01 0.65 / 0.10 with one, 0.80 / 0.15
  with two; sheet 02 0.556 / 0.089 with one, 0.422 / 0.178 with two) [measured]. Even with two in a row the early stops
  sat at no-line pictures, which come several in a row [measured; stop_analysis]. On the robot a second look costs
  either a step past the end or, standing still, an almost identical picture [judgement].
- **Thresholds tuned on validation episodes of the training sheets** varied from 0.05 to 0.6 between folds [measured;
  fold files], so they do not carry over to an unseen sheet; the audit already asks for settings chosen on sheet folds
  [audit 11.2 point 4].
- **The predicted step length as a second stop cue** (stop only when the stop probability passes and the predicted step
  is under 0.5 to 0.8 cm) raised the ceiling of the end-to-end exact model from 0.439 / 0.288 to 0.494 / 0.177 (two in a
  row) and, with thresholds from the other sheets, from 0.324 / 0.422 to 0.350 / 0.329 [measured; exact_analysis]; three
  sheets only, so a hint, not a result.
- **Earlier pictures did not help the stop.** Adding the previous picture, the change since it, or the pictures 2 and 4
  steps back to the names probe gave 0.254, 0.278 and 0.186 stopped right at 0.5 against 0.258 for one picture, and no
  gain with tuned thresholds (0.305, 0.299, 0.226 against 0.317, two in a row) [measured; history.json via
  comparison.md].
- **What "right" means here.** The offline measure uses the tip's distance to where the demonstrator stopped, not the
  puck's distance to the line end that the exam judges [audit notes_context-23]; a robot test has to score the puck.

**Extra weight on look-alike pictures does not fix it (tested for this section).** A stop-only frozen probe on the
audit-cleaned rows (leave one sheet out, threshold from the other sheets) was trained without and with weight 5 on the
training pictures more than 3 cm from the goal that show no line ahead (4,523 of the 16,776 such far pictures over all
six sheets):

| Stop probe (cleaned rows) | seeds | right / early, 1 picture | right / early, 2 in a row | far no-line pictures, 90th percentile | stop pictures, median |
|---|---|---|---|---|---|
| stop pictures only | 3 | 0.284 to 0.346 / 0.242 to 0.258 | 0.269 to 0.313 / 0.279 to 0.319 | 0.115 to 0.224 | 0.410 to 0.487 |
| plus weight 5 on far no-line pictures | 3 | 0.231 to 0.271 / 0.268 to 0.304 | 0.254 to 0.319 / 0.252 to 0.371 | 0.086 to 0.107 | 0.250 to 0.324 |
| plus weight 10 on far no-line pictures | 1 | 0.248 / 0.235 | 0.301 / 0.294 | 0.054 | 0.178 |

[measured; targets/stop_probe.out.txt]

The weight lowered the stop probability on the look-alikes, but about as much on the real stop pictures; with one
picture the stop got worse, with two in a row it stayed within the seed range. With these picture features the model
cannot tell an end from a stretch of line along the paper edge, so a loss weight does not create the missing
information [judgement]. A second remedy, extra weight on the far pictures where a first model wrongly fired (mined
hard negatives, variant E of the script), did not finish in time.

### 2.7 What the literature adds (claims marked supported or corrected by the checkers)

- From pictures, plain regression was not worse than bins in the one controlled comparison (ResNet-18: 17.8 against
  14.7 for per-number bins and 16.8 for bins plus offsets), and a mixture head hardly helped multi-operator data with
  pictures (Transport 42.0 against 41.0) [literature: Pearce et al. 2023 Table 3; robomimic Table 26; supported].
- Single-step policies trained with idle actions got stuck (real Push-T: 8 of 20 runs), idle actions being one
  contributing cause [literature: Chi et al. 2023, Section 6.1; supported].
- Chunking helped most at 50 decisions per second in simulation (1 to 44 % success) [literature: ACT, supported]; our
  1-cm steps already give about 89 decisions per track [calculated; main_session_facts.md].
- Averaging the outputs of several pictures (temporal ensembling) did poorly and oscillated on a real robot, because the
  average of two valid actions need not be valid [literature: Black et al. 2025, supported].
- Thresholded direction errors predicted closed-loop driving better than the mean squared error (correlation -0.65
  against -0.39) [literature: Codevilla et al. 2018, supported], which is why step 45 and path 45 are used above.
- Ten robot runs cannot tell 70 from 90 % (7 of 10 gives a 95 % interval of 0.40 to 0.89) [literature: Wilson interval,
  evaluation_statistics, supported].

## 3. Recommendation

**What the model predicts**

1. One network with two outputs, as in the running final candidate (`resnet18_final_candidate`): the exact step of the
   next move in the current picture's frame (tip forward and tip left in cm, forearm twist in deg; Huber loss, a squared
   error that is gentler on outliers), and a stop probability from an output of its own trained on the 259 stop
   pictures. The lean and hand numbers may stay in the loss as extra targets but are not executed; the hand could also
   be dropped (26 changes of 3 deg or more in 17,773 rows).
2. Keep the exact-step loss unweighted: weighting it by move name brought back some backing up but cost more on all
   other moves, on step 45 and on path 45 (section 2.2). The weak backing up is a known risk to watch on the robot
   (backup, third point).
3. Small motions stay ordinary rows of the exact target (a short tip move with a twist); the 13 turns are 15-deg twists
   after the clip; the null rows stay dropped (already in the audit cleaning). No relabelling with the next move.
4. A chunk target only if the queued end-to-end chunk run beats the exact run on the same held-out sheets by more than
   the seed range (about 3 points of path 45); even then the robot executes one step per picture.

**How the robot executes it (executor rules; the model still sees only the picture)**

5. Direction: the angle of the predicted (forward, left) vector. Length: a fixed 1.0 cm, not the predicted length, which
   is shrunk to a median 0.60 cm on the earlier run's curved held-out sheets and 0.68 cm on the final candidate's sheet
   02, and would leave the robot standing still on 0.75 to 2.5 % of the pictures away from the goal.
6. Twist: the predicted twist as it comes (gain 1, which the least-squares slope of 0.97 supports; when the model is
   unsure of the turn it then hardly twists), clamped to plus or minus 10 deg per step, with joint 4 kept within plus
   or minus 115 deg (the arm-model limit is 119.75 deg; 27 of 45 sheet-04 demonstrations passed 110 deg, audit
   code_actions-8).
7. Wrist (joint 5) and hand (joint 6) held at their start values; joints 1 to 3 solved with `solve_with_wrist_fixed`;
   one `move_joints` per picture at the 20 % speed cap the data were recorded with; the workspace, height and refusal
   guards of audit code_actions-11.

**How it stops**

8. Stop at the first picture whose stop probability reaches the threshold. Choose between one picture and two in a row
   (the second look after a short 0.5-cm step) by the leave-one-sheet-out result of the final candidate: the earlier
   exact run's three sheets and the final candidate's sheet 02 favour one picture, its sheet 01 favours two.
9. Choose the threshold the same way, pooling the other five held-out sheets' predictions of the final candidate,
   never on validation episodes of the training sheets. Prefer the later stop when two thresholds stop right equally
   often: an early stop always fails the run, a stop 1 to 2 cm late may still count [judgement; the exam rule is open].
10. Do not add loss weight on pictures without a line ahead (it did not help, and with one picture it made the stop
    worse, section 2.6). The stop needs more and more varied ends to learn from: two or three printed test tracks whose
    line ends 10 cm or more from the paper edge or runs along the edge mid-track, with near-end demonstrations on them,
    because more different tracks helped held-out sheets more than more episodes of the same tracks [measured:
    main_session_facts.md, data amount]. This is robot time, so it is the team's decision.
11. Switch the step budget on (`is_step_budget_enforced` is False in config.py today) as a safety stop only.

Why: the exact step is the best direction and twist output we have on held-out sheets; it needs no rule for small
motions; with the fixed-wrist solve the camera sees per step what it saw in training; a fixed 1-cm step keeps the
demonstrated pace and cannot stall; and a stop chosen on held-out sheets is the only fair guess for an unseen track
[judgement on the measured numbers above].

## 4. Backup

- **If fixed 1-cm steps lose the puck at sharp bends on the robot:** use the predicted length rescaled by 1.81 (so that
  its median on moving rows equals the demonstrated 1.09 cm) and clamped to 0.5 to 1.5 cm. On the three held-out sheets
  this gives a median step of 1.07 cm, the lower clamp on 7.8 % and the upper on 17.4 % of the pictures [measured;
  targets/rescaled_length.out.txt]; steps become about 1.27 cm where the direction is usually right (error under 15 deg)
  and 0.87 cm where it is usually wrong (over 90 deg) [calculated from length_by_error], so a wrong step does less harm.
  It does not bring back the twist in place (0.88 cm executed where the demonstrators moved 0.19 cm) and cannot stall
  [measured; rescaled_length].
- **If the camera falls behind on curves** (the forearm lagging the path on sheet 03), scale up only the confident
  twists: about 1.2 times on predicted twists of 3 deg or more, where the sign is right 0.91 to 0.94 of the time and the
  demonstrated twist was 1.1 to 1.4 times the predicted one [calculated from twist_calibration]. A blanket gain of 2
  would mostly enlarge twists whose sign is a guess [judgement on the same numbers].
- **If the robot fails to back up or recover the puck:** the names-plus-exact design, its direction read as the mean
  direction of the names part and its twist taken from the exact part. In the probes it matched the exact step overall
  (step 0.602, path 0.677) and backed up better (0.363 against 0.270); end to end, the names model read this way backed
  up far better than the exact step (0.663 against 0.392) at a cost of 4.6 points of path 45 [measured; section 2.2]. It
  needs one end-to-end run (about 3.5 hours on the laptop for six sheets).
- **If the stop still fires early on the robot:** the stop output trained on every picture within 2 cm with a high
  threshold (0.7 to 0.9), or the combined rule (stop probability passes and the predicted step is under about 0.6 cm),
  both about as good offline [measured]; two pictures in a row as a robot A/B test.

## 5. What is still unknown

- **The final candidate on sheets 03 to 06.** Four of its six folds were still training at the time of writing (about
  35 to 50 minutes each): its direction, twist, backing up and stop there, above all on the loops 05 and 06, and the
  leave-one-sheet-out threshold over all six sheets. The checks of sections 2.3 and 2.6 take minutes once the folds
  exist (`exact_inference.py --final <fold>`, `final_fold_check.py`, `final_stop_check.py`).
- **Closed loop.** Every number above scores the model on the demonstrators' pictures; driving with its own moves, the
  robot sees other pictures [literature: offline_vs_closed_loop, supported]. Fixed against rescaled length, the twist
  as predicted against a larger gain on confident twists, and the stop threshold can only be ranked on the robot (finish
  first, time second), with paired runs from matched starts.
- **The stop cue at the exam.** On the six sheets every line ends 2.35 to 8.03 cm from a paper edge; whether unseen
  tracks do the same is not known, and a line ending in the middle of the paper may never trigger the stop.
- **The rules of the exam and the course.** How "finished" is judged (puck within what distance of the line end; does
  overshoot count); whether the executor may read the joints to solve joints 1 to 3 and hold the wrist while the model
  sees only the picture; whether a fixed step length, clamps and a step budget count as rules outside the model (audit
  11.3 items 17 and 20).
- **Twist in place.** A fixed 1-cm step moves the tip where the demonstrators twisted in place (10.5 % of the step time);
  whether that matters at sharp bends needs the robot.
- **Backing up.** The exact models got 0.27 of the backward moves within 45 deg in the probes (six sheets pooled), 0.39
  end to end (sheets 03, 05, 06) and 0.05 and 0.14 as the final candidate (sheets 01 and 02); whether the robot then
  loses pucks that slide off the block, or simply pushes on from the side, only robot runs can show. Watch for it from
  the first run on sheet 02 or 03.
- **Speed.** The 20 % speed cap was never raised for pushing steps; a higher cap would save time without changing the
  pictures, but how the puck behaves is untested (audit 11.3 item 24).

## Sources

- **targets/** is `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/targets/`:
  the scripts and printouts written for this section (data_facts, stop_analysis, early_stops_one_picture,
  exact_inference, exact_analysis, length_by_error, twist_calibration, rescaled_length, per_class_direction,
  final_fold_check, final_stop_check, final_backing_up, stop_probe, exact_probe; each `.py` writes the `.out.txt` or
  `.log` of the same name; `common_frames.py` re-implements the audit cleaning without importing project code; all
  read the project's files and write only into this folder).
- **comparison.md, designs.json, seeds.json, splits.json, stop.json, stop_distance.json, baselines.json, metrics.json,
  fold files**: `D:/AI_Project/.claude/worktrees/training-analysis/04_runs/` (probes in
  `2026-11-09_frozen_feature_probes/`, end-to-end runs in `2026-11-09_<run>/`).
- **audit <dimension>-<n>**: findings in `.../scratchpad/audit/AUDIT_SYNTHESIS.md`; code_actions also in
  `.../scratchpad/audit/code_actions/note.md`.
- **main_session_facts.md**: `.../scratchpad/design/main_session_facts.md`.
- **literature**: `.../scratchpad/literature/<key>/note.md` with its `check/note.md` (keys action_representation,
  action_chunking, offline_vs_closed_loop, evaluation_statistics, diffusion_and_generative).
