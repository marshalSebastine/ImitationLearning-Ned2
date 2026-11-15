# Facts measured by the main session on 9.11.26

All measured on Emranul's laptop unless marked otherwise.

## Laptop and software

- Lenovo ThinkPad 21QKS0BK00, AMD Ryzen AI 5 PRO 340 (6 cores, 12 threads), 27.6 GB RAM, AMD Radeon 840M integrated
  graphics (no CUDA). Free disk: C 427 GB, D 1,232 GB.
- Python 3.11.9 in D:/AI_Project/.venv. Installed on 9.11.26 from PyPI (dry run first: nothing existing was upgraded or
  downgraded): torch 2.15.0+cpu (MKL, oneDNN, AVX-512 in use), torchvision 0.29.2, timm 1.0.31, pandas 3.1.1,
  scikit-learn 1.9.1 and their dependencies. Already there: numpy 2.4.6, opencv 5.0, onnxruntime 1.31.1, pyniryo 1.2.5.
- Pretrained weights downloaded from the Hugging Face hub by timm: resnet18.tv_in1k, vit_small_patch14_dinov2.lvd142m,
  efficientnet_b0.ra_in1k, mobilenetv3_large_100.ra_in1k, vit_base_patch32_clip_224.openai.

## The index (02_code/training_dataset.py, 71 s to build)

- 211 episodes indexed (208 scored 1 to 3, 3 unscored), 18,522 pictures (18,451 in scored episodes); score-0 episodes
  (32) left out. Every picture readable; every file's MD5 equals the MD5 logged at capture (no picture changed since).
- The recording tool's own nearest-direction rule, re-implemented in training_metrics.nearest_direction, reproduces the
  logged move name on more than 99.9 percent of moving rows (test_training.py).
- tip_forward_m and tip_left_m recomputed from consecutive arm-model tip positions and picture_heading_deg agree with the
  logged values to 0.02 mm median and 0.25 mm at the 95th percentile, so a row's label is the move from its picture to
  the next one, and chunk targets (where the tip is k steps later, in the current picture's frame) can be computed from
  the logged poses.
- small_motion rows: 2,265; median forearm change 5.5 degrees, median tip move 2 mm, 9.3 percent of them on row 0.
- Median time from one row to the next 1.23 s, 90th percentile 1.53 s (seconds_to_next_row).
- Start ends of the 17 re-planned episodes of sheets 05 (plans 14 to 19, 28 to 30) and 06 (plans 14 to 21, 24 to 25):
  the tip's path agrees with the plan in config, not with the text in episode.json. Sheet 05 plans 14 to 19 end near
  y = +0.09 m with positive total turning, plans 1 to 13 near y = -0.08 m; sheet 06 plans 14 to 21 end near y = -0.12 m,
  plans 1 to 13 near y = +0.08 m. training_targets.load_training_frames uses the plan's end.

## Compute measured here (with up to 20 agents sharing the processor, so these are upper bounds)

- Frozen feature extraction of all 18,522 pictures at 224 x 224, 4 threads: ResNet-18 285 s, DINOv2-small 1,404 s,
  EfficientNet-B0 about 4 minutes (others: see 03_data/processed/v01_training_index/features/extraction_summary.json and
  speed.json for the per-picture milliseconds).

## How the comparison is set up (02_code)

- training_targets.py: rows used are scores 1 to 3 without the 21 mistake rows; small motions become the next real move
  by default ("next_move"), or are dropped, or form their own class; targets: move class (8 directions plus stop),
  exact step (tip forward and left in cm, forearm, lean and hand changes in degrees), chunk (tip place after each of the
  next 5 steps), twist class (2 degree threshold), stop (stop picture, or within 2 cm of the goal).
- Splits: leave one sheet out (6 folds) and whole episodes held out (5 folds over all sheets). Early stopping on whole
  episodes held back from the training sheets (15 percent per sheet).
- training_probes.py: frozen features (global summary, or a 4 x 4 grid of the last feature map) with a head of one hidden
  layer of 256 (or linear), AdamW, early stopping. train_policy.py: end-to-end fine-tuning on 160 x 120 pictures with
  colour, gamma, blur, noise and a 3-pixel shift (mirroring with label swap as an option), same heads and measures.

## Audit cleaning, now in training_targets.py (is_audit_cleaning)

Checked on 9.11.26 18:11: 18,522 rows become 18,053 in 209 episodes; 259 stop rows (50 relabelled duplicated final points); 1,765 small_motion rows left (twists in place); steps clipped at 2 cm, twists at 15 degrees; corrected ends per sheet A/B: 01 16/4, 02 12/33, 03 25/20, 04 33/12, 05 8/21 (05-16 dropped), 06 15/10. The end-to-end runs and probes before 18:15 used the earlier rules (score from the folder name, mistakes dropped, plan end, no row drops).

## Pending at 18:20

Running: the line-and-puck small CNN (folds 03, 05 done, 06 running). Queued in this order: resnet18_final_candidate (exact step, stop output on the stop pictures, audit cleaning, 8 epochs, last weights, all six held-out sheets), resnet18_finetune_exact on sheets 01, 02, 04, resnet18_finetune_mirror and resnet18_finetune_chunk on 03, 05, 06. The goal-distance stop was tested with frozen probes and lost (tuned, two in a row: stopped right 0.15 to 0.16, never stopped 0.75 to 0.79), so it gets no end-to-end run.

# Results collected so far (main session notes for the report, 9.11.26)

## Baselines (leave one sheet out, 6 folds, mean of folds)

| Rule | step within 45 | path within 45 | path median error |
|---|---|---|---|
| always forward | 0.538 | 0.604 | 33.7 |
| repeat previous move (uses the previous move: reference only) | 0.736 | 0.648 | 28.6 |

Per sheet, always forward, path within 45: 01 0.806, 02 0.660, 03 0.603, 04 0.580, 05 0.544, 06 0.433.
Per sheet, always forward, step within 45: 01 0.747, 02 0.565, 03 0.522, 04 0.485, 05 0.509, 06 0.402.

## Frozen backbones, classes head (leave one sheet out, mean of 6 folds)

| Backbone, features | step 45 | path 45 | path 45 worst | expected path 45 | stopped right | early |
|---|---|---|---|---|---|---|
| ResNet-18 ImageNet, grid | 0.577 | 0.633 | 0.526 | 0.682 | 0.258 | 0.112 |
| MobileNetV3, grid | 0.576 | 0.640 | 0.556 | 0.691 | 0.215 | 0.070 |
| EfficientNet-B0, grid | 0.547 | 0.606 | 0.478 | 0.667 | 0.253 | 0.081 |
| DINOv2-small, grid | 0.545 | 0.598 | 0.459 | 0.654 | 0.287 | 0.095 |
| CLIP ViT-B/32, grid | 0.547 | 0.606 | 0.480 | 0.673 | 0.209 | 0.217 |
| ResNet-18 untrained, grid | 0.490 | 0.522 | 0.409 | 0.568 | 0.117 | 0.172 |
| (global summaries: 0.522 to 0.548 step 45) | | | | | | |

ResNet-18 grid per sheet (step / path 45): 01 0.756/0.801, 02 0.530/0.629, 03 0.565/0.570, 04 0.559/0.652,
05 0.583/0.621, 06 0.467/0.526.

Error analysis (ResNet-18 grid, path 45, model vs always forward): bend 0.580 vs 0.515; straight 0.701 vs 0.739;
start 0.650 vs 0.766; near goal 0.477 vs 0.540; forearm turned 0.600 vs 0.511; far turned 0.617 vs 0.514; camera view
B 0.581 vs 0.505.

Averaging the direction over the last pictures (path 45, pooled): ResNet-18 grid 1: 0.617, 2: 0.631, 3: 0.635,
5: 0.620; MobileNetV3 grid 1: 0.607, 2: 0.629, 3: 0.638, 5: 0.632.

Speed (ms per picture, 4 threads, laptop busy): ResNet-18 26 (224) / 17 (160x120); MobileNetV3 25 / 15; EfficientNet-B0
20 / 23; DINOv2-small 83; CLIP B/32 91. Parameters: 11.2, 4.2, 4.0, 21.6, 87.5 million.

## Frozen designs (ResNet-18 grid, leave one sheet out)

| Design | step 45 | stopped right | early |
|---|---|---|---|
| classes, small motion as next move | 0.577 | 0.258 | 0.112 |
| classes, small motion dropped | 0.560 | 0.182 | 0.054 |
| classes, small motion own class | 0.560 | 0.207 | 0.059 |
| exact step | 0.591 | 0.291 | 0.243 |
| mixed classes and exact | 0.565 | 0.234 | 0.060 |
| chunk of 5 steps | 0.607 | 0.300 | 0.264 |

## Splits (frozen ResNet-18 grid)

| Split | classes step 45 | exact step 45 | classes stopped right | exact stopped right |
|---|---|---|---|---|
| held-out episodes (5 folds) | 0.655 | 0.707 | 0.509 | 0.571 |
| held-out sheets (6 folds) | 0.577 | 0.591 | 0.258 | 0.291 |

## History (frozen ResNet-18 grid, classes)

one picture 0.577; plus previous picture 0.576; plus change since previous 0.603; plus pictures 2 and 4 back 0.586
(stopped right 0.186).

## Class balance (frozen ResNet-18 grid, classes)

no class weights 0.593 (stopped right 0.176, early 0.058); square-root weights 0.577 (0.258, 0.112); full inverse
weights 0.509 (0.292, 0.212); square-root weights with label smoothing 0.1: 0.558 (0.233, 0.054).

## Scores (frozen ResNet-18 grid, classes, 6 folds)

scores 1, 2, 3: 0.577 (stopped right 0.258); scores 2, 3: 0.565 (0.186); score 3 only: 0.539 (0.338); all scores
weighted 0.4 / 0.7 / 1.0: 0.573 (0.306).

## Head size (frozen ResNet-18 grid, classes, 6 folds)

linear 0.499 (stopped right 0.171, early 0.280); one hidden layer of 256: 0.577 (0.258, 0.112); 1024 with dropout 0.3:
0.558 (0.271, 0.156).

## Cleaning (frozen ResNet-18 grid, classes, 6 folds)

mistake rows dropped, flags kept (default) 0.577 (0.258, 0.112); mistakes and flagged pictures dropped 0.560 (0.196,
0.096); nothing dropped 0.556 (0.271, 0.161); mistakes and still rows dropped 0.566 (0.268, 0.115).

## Data amount (frozen ResNet-18 grid, classes, 6 folds, step 45; three seeds each: 1, 2, 3)

| Training data per fold | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| 25 % of the episodes of all 5 training sheets | 0.518 | 0.513 | 0.512 | 0.514 |
| 50 % of the episodes of all 5 | 0.541 | 0.547 | 0.549 | 0.546 |
| 100 % of all 5 | 0.574 | 0.551 | 0.541 | 0.555 |
| 1 training sheet (all its episodes) | 0.471 | 0.467 | 0.431 | 0.456 |
| 2 sheets | 0.523 | 0.509 | 0.491 | 0.508 |
| 3 sheets | 0.533 | 0.514 | 0.514 | 0.520 |
| 4 sheets | 0.532 | 0.533 | 0.531 | 0.532 |

Read: at about the same number of pictures, five sheets at half their episodes (0.546) beat two to four sheets with all
their episodes (0.508 to 0.532). Doubling the episodes of the same five sheets added 0.9 points (0.546 to 0.555);
the step from four to five sheets added 2.3. Seed spread at full data: 0.541 to 0.574, so about 3 points between
single runs are noise. (Frozen features; the sheet draws are random per fold.)

## End-to-end, ResNet-18 fine-tuned, exact step (stop head on the stop picture; tuned stop thresholds in fold files)

| Held-out sheet | step 45 | stopped right (0.5) | early (0.5) | twist sign |
|---|---|---|---|---|
| 03 | 0.700 | 0.689 | 0.156 | 0.845 |
| 05 | 0.667 | 0.200 | 0.567 | 0.669 |
| 06 | 0.675 | 0.200 | 0.040 | 0.802 |

Path within 45: 0.802 / 0.777 / 0.791 (folds 03 / 05 / 06). Tuned stop, two pictures in a row (right / early /
never, threshold): 03 0.756 / 0.178 / 0.044 (0.15); 05 0.167 / 0.633 / 0.200 (0.30); 06 0.320 / 0.240 / 0.440 (0.15).
Twist mean error 2.9 / 3.6 / 3.4 degrees; lean sign 0.89 / 0.81 / 0.81.

Run 16:14 to 17:26. Against classes on the same folds: +3.6, +0.8, +3.5 points step 45; twist sign 0.84/0.67/0.80
against 0.70/0.37/0.42. Folds 01, 02, 04 follow after the line-and-puck run (queue 2). The line-and-puck run started
17:26 by the first queue could not be stopped (process stop refused by the safety check); it continues as planned.

## Seeds (frozen ResNet-18 grid, 6 folds, step 45)

classes: seeds 11, 12, 13 give 0.561, 0.546, 0.556 (default seed 0.577): range 0.546 to 0.577, mean of four 0.560.
exact: 0.616, 0.593, 0.611 (default seed 0.591): range 0.591 to 0.616, mean of four 0.603. Exact leads by about 4.3
points; the seed range is about 3 points. Stopped right (fixed 0.5) classes 0.29 to 0.32, exact 0.29 to 0.33.

## Matched folds 03, 05, 06 (compare_runs.py, step 45 / path 45)

end-to-end ResNet-18 fine-tuned classes 0.654 / 0.707; end-to-end small CNN from scratch classes 0.631 / 0.683;
probe chunk of 5 steps 0.578 / 0.661; probe MobileNetV3 grid 0.563 / 0.624; probe classes without class weights
0.563 / 0.618; probe exact 0.557 / 0.628; probe ResNet-18 grid classes 0.538 / 0.573.

## Stop, rerun with thresholds tuned on validation episodes (frozen ResNet-18 grid, 6 folds)

Columns: stopped right / early / never with two pictures in a row past the tuned threshold (mean threshold).
classes, stop as a class 0.317 / 0.274 / 0.352 (0.13); classes, own stop output on the stop picture 0.382 / 0.255 /
0.319 (0.25); classes, own stop output near the goal 0.329 / 0.215 / 0.370 (0.78); exact, stop on the stop picture
0.221 / 0.414 / 0.300 (0.14); exact, stop near the goal 0.318 / 0.276 / 0.310 (0.67); chunk 0.298 / 0.341 / 0.276
(0.23). One picture past the threshold (right / early): 0.299 / 0.234, 0.356 / 0.228, 0.273 / 0.301, 0.266 / 0.335,
0.241 / 0.266, 0.282 / 0.344. Path within 45 in the same runs: classes 0.633, exact 0.673, chunk 0.697 (chunk's place
3 steps ahead: 0.722), mixed 0.620 (its vector 0.675).

## Stop (fixed threshold 0.5; first run)

classes, stop as a class: right 0.258, early 0.112; classes, own stop output on the stop picture: right 0.321, early
0.209; classes, own stop output on every picture near the goal: right 0.138, early 0.659; exact, stop on the stop
picture: right 0.291, early 0.243; exact, stop near the goal: right 0.176, early 0.512.

## End-to-end, ResNet-18 fine-tuned, classes (folds 03, 05, 06; 8 epochs on half the pictures each)

| Held-out sheet | step 45 | path 45 | expected path 45 | stopped right | early | never | twist sign |
|---|---|---|---|---|---|---|---|
| 03 | 0.663 | 0.704 | 0.751 | 0.289 | 0.111 | 0.578 | 0.703 |
| 05 | 0.659 | 0.711 | 0.754 | 0.133 | 0.167 | 0.700 | 0.372 |
| 06 | 0.640 | 0.707 | 0.728 | 0.040 | 0.040 | 0.920 | 0.422 |

ms per picture 26.9 (160 x 120, laptop busy). 20 to 29 minutes per fold.

## End-to-end, small CNN from scratch, classes

| Held-out sheet | step 45 | stopped right | early | twist sign |
|---|---|---|---|---|
| 03 | 0.635 | 0.378 | 0.133 | 0.718 |
| 05 | 0.635 | 0.033 | 0.067 | 0.405 |
| 06 | 0.623 | 0.040 | 0.000 | 0.371 |

16 epochs on half the pictures each, about 22 minutes per fold (13:47 to 16:14 for both runs' three folds).

# Literature takeaways (training-literature-check workflow, 9.11.26; each claim checked against its source)

Claim checks: offline_vs_closed_loop 48 (45 supported, 3 corrected); action_representation 40 (34, 6); action_chunking
49 (44, 5); diffusion_and_generative 44 (42, 2); pretrained_representations 57 (51, 6); augmentation 43 (40, 3);
history_and_causal_confusion 38 (33, 4, 1 could not open); interactive_correction 36 (34, 2); evaluation_statistics 42
(37, 5). Notes: scratchpad/literature/<key>/note.md and check/note.md.

1. Offline numbers against robot success. Validation loss is the weakest predictor (robomimic, Mandlekar et al. 2021,
   Table 29: the lowest-validation-loss checkpoint 57 to 100 percent below the best, the last checkpoint 12 to 35 percent
   below, per the checker). Direction-type measures correlate better (Codevilla et al. 2018: quantised classification
   error about 0.65 against 0.39 for MSE, unseen town). Validation pictures with recoveries raised the MSE correlation
   from 0.39 to 0.77 in one condition. Among good models and between seeds the offline numbers say little. So: shortlist
   offline on held-out sheets with direction measures, never pick on validation loss, fix the training length once and
   take the last checkpoint, then decide on the robot. Score robot runs also by how far the path strays from the line
   (Hussenot et al. 2021) and count stalls and dithering.
2. Classes against continuous outputs. From pictures, MSE was not worse than bins in the one controlled comparison
   (Pearce et al., ResNet-18: MSE 17.8, per-number bins 14.7, k-means plus residual 16.8). Big multimodality gains came
   with memory, states or many operators (robomimic with pictures, multi-operator Transport: 42.0 against 41.0 without
   the GMM). Two operators are closer to one operator. Pauses and idle actions made single-step policies stall (Diffusion
   Policy real Push-T: LSTM-GMM stuck in 8 of 20 runs; the authors call removing idle actions common practice).
3. Action chunks. ACT's gain from chunking (1 to 44 percent) was at 50 Hz in simulation with joint state; our 1 cm steps
   already give about 89 decisions per track. Real Push-T ran 6 of 16 predicted steps open loop at 95 percent. Under
   noise, executing long chunks blind lost (Liu et al. 2025). Supported plan: train with a 5 to 10 step target, execute
   one step per picture by default, try 2 or 3 per picture only for speed, on held-out tracks.
4. Diffusion and generative policies. Strong on Push-T but trained for hours on GPUs; with the same network and chunks,
   regression matched flow policies on most benchmarks (Pan et al. 2026). Not realistic on the laptop's processor now;
   a later GPU comparison only with the same encoder and a held-out sheet. Lin et al. 2025 (ICLR): generalisation to
   unseen environments grew roughly as a power law in the number of training environments, not demonstrations.
5. Pretrained backbones. With 100 to 200 demonstrations a ResNet-18 trained end to end with shift augmentation matched
   or beat frozen features (Chi et al. 2023, Square: 0.94 from scratch, 0.92 fine-tuned, 0.58 frozen; real Push-T 95
   percent with a ResNet-18 trained from scratch). Fully fine-tuning large ViTs on small data hurt. No study tested
   unseen track shapes. Candidates worth adding: a small CNN from scratch and a ResNet-18 cut after layer 3.
6. Augmentation. Small shifts or crops are the best supported (robomimic real Can 22 of 30 against 8 of 30 without);
   photometric changes guard against light and table changes. Mirrors with swapped labels: no measured robot benefit;
   safe only if the camera view is left-right symmetric and every label is mirrored; test first. Recorded recovery
   demonstrations are better supported than synthetic corrective views. Track variety matters (Xie et al.: the
   generalisation gap fell from about 0.4 to under 0.1 as training environments went from 5 to 100).
7. Picture history. History helped only where the current picture lacked needed state; otherwise it hurt closed-loop
   results while offline scores improved (Wen et al. 2020: lower test loss, less reward). Default: current picture only;
   test one extra frame only in closed loop, with a frozen-history copy test first (Chuang 2022). The crossing and the
   dashed gaps might need memory: check on the robot.
8. Corrections after the first model. Human-gated corrections helped on real robots (HG-DAgger, ThriftyDAgger, BC-Z),
   mostly 1 to 3 rounds; corrections recorded with the same tool keep the same labels; keep the correction share near 10
   to 25 percent of a batch, not 50; pilot 5 episodes to measure the time per correction (no source reports hours per
   round); keep at least one practice track for evaluation only.
9. Robot test statistics. Report counts with Wilson intervals; 10 trials cannot tell 70 from 90 percent (7/10 is 0.40 to
   0.89, 9/10 is 0.60 to 0.98); a formal test of 70 against 90 percent needs about 50 trials per model one-sided; choose
   between two candidates with about 24 paired, blind, interleaved trials from matched starts; run one pre-registered
   final test; on unseen tracks prefer many distinct tracks with 2 to 3 trials each (at least 10 tracks) over few tracks
   with many trials, because trials on one track share its difficulty; time among successful runs only is biased when
   finish rates differ, so use finish first, then time per paired block.

## Robot time per trial (calculated from measured numbers)

Measured (code-reading audit, 17,863 replayed steps): a step takes 1.22 s median (the move 1.03 s, waiting for a fresh
picture 0.11 s, deciding 0.06 s). Steps per demonstrated episode, median by sheet: 01 31.5, 02 66, 03 106, 04 168,
05 103.5, 06 82 (all start kinds mixed). A model adds its own time per picture: 15 to 26 ms for ResNet-18 or
MobileNetV3 on 160 x 120 pictures (measured, 4 threads, laptop busy), under 3 percent of a step.

| Sheet | Steps (median) | Driving time at 1.22 s per step | With about 1 minute to lay the puck and start |
|---|---|---|---|
| 01 straight | 31.5 | 38 s | about 1.6 min |
| 02 arc | 66 | 81 s | about 2.4 min |
| 03 two curves | 106 | 129 s | about 3.2 min |
| 04 meander | 168 | 205 s | about 4.4 min |
| 05 loop | 103.5 | 126 s | about 3.1 min |
| 06 dashed loop | 82 | 100 s | about 2.7 min |

So 10 trials on each of the six sheets take about 3 hours of driving and set-up (calculated: 10 x the last column
summed, 174 minutes), plus resets after failures and breaks: plan 4 hours. 20 trials per sheet take about 6 to 8
hours. The guess of 1 minute per set-up is not measured.
