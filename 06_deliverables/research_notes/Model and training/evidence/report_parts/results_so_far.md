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

## End-to-end, ResNet-18 exact step on the other three sheets (queue 3, 22:02 to 23:02; old rules)

| Held-out sheet | step 45 | stopped right (0.5) | early (0.5) | twist sign |
|---|---|---|---|---|
| 01 | 0.759 | 0.632 | 0.158 | 0.675 |
| 02 | 0.631 | 0.477 | 0.045 | 0.807 |
| 04 | 0.635 | 0.333 | 0.378 | 0.758 |

Six-fold mean of this plain exact model 0.678, the same as the final candidate's 0.678 (per sheet -2.0 to +2.0
points: within noise). Fixed-0.5 stopped right, mean of six: 0.422 against the final candidate's 0.460.

## End-to-end mirror and chunk runs (3 folds, old rules; trained 18:18 to 22:07 by the first queue, whose inner loop
## kept running after its outer shell was stopped at 17:26; the mirror run was repeated 23:02 to 00:04 with identical
## numbers, the chunk run is being repeated from 00:05)

| Held-out sheet | mirror (names head, mirrored pictures) step 45 | names head, no mirror | chunk of 5, step 45 | exact, step 45 |
|---|---|---|---|---|
| 03 | 0.670 | 0.663 | 0.692 | 0.700 |
| 05 | 0.659 | 0.659 | 0.681 | 0.667 |
| 06 | 0.599 | 0.640 | 0.638 | 0.675 |

Mirror: stopped right (0.5) 0.578 / 0.300 / 0.240, early 0.067 / 0.267 / 0.040; twist sign 0.709 / 0.370 / 0.470.
Chunk: stopped right 0.578 / 0.167 / 0.000, early 0.111 / 0.600 / 0.040; twist sign 0.839 / 0.662 / 0.778.
Mirroring costs 4.1 points on the dashed loop and is neutral on 03 and 05 (the audit: the view is not left-right
symmetric). The chunk target is about even with the exact step (mean 0.670 against 0.681 step 45; on path within 45
the chunk leads on 03 and 05 and trails on 06, about +1.2 points over the three sheets per the report's evidence
critic).

## End-to-end, small CNN from scratch on line-and-puck maps instead of colour, classes

| Held-out sheet | step 45 | stopped right | early | twist sign |
|---|---|---|---|---|
| 03 | 0.630 | 0.378 | 0.044 | 0.659 |
| 05 | 0.625 | 0.033 | 0.167 | 0.399 |
| 06 | 0.623 | 0.000 | 0.000 | 0.398 |

Mean 0.626 against 0.631 for the same network on colour pictures: no gain (within seed noise). Run 17:26 to 18:18.

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

## Final candidate, error analysis by situation (training_error_analysis.py, pooled over the six held-out sheets, path within 45; always forward beside it)


resnet18_final_candidate
  situation      value  pictures  step_within_45  path_within_45  always_forward_path_within_45
      sheet    path_01       633           0.778           0.864                          0.808
      sheet    path_02      2613           0.632           0.757                          0.673
      sheet    path_03      3955           0.680           0.804                          0.612
      sheet    path_04      6227           0.642           0.738                          0.587
      sheet    path_05      2600           0.649           0.758                          0.545
      sheet    path_06      1936           0.684           0.814                          0.429
      place     middle     16303           0.659           0.772                          0.583
      place  near_goal      1041           0.619           0.690                          0.566
      place      start       413           0.763           0.900                          0.780
      place       stop       207             NaN             NaN                            NaN
      shape       bend     12107           0.619           0.735                          0.524
      shape   straight      5857           0.749           0.860                          0.740
    forearm far_turned      3907           0.650           0.766                          0.518
    forearm   straight      6292           0.686           0.798                          0.716
    forearm     turned      7765           0.643           0.754                          0.518
camera_view          A     11693           0.660           0.769                          0.629
camera_view          B      6271           0.659           0.777                          0.518
   lighting    ceiling     15026           0.671           0.785                          0.585
   lighting    natural      2938           0.600           0.702                          0.605
      score        1.0      2269           0.635           0.735                          0.582
      score        2.0      5060           0.618           0.723                          0.551
      score        3.0     10635           0.685           0.804                          0.608
 window  pictures  path_within_45
      1     14294          0.7720
      2     14294          0.7753
      3     14294          0.7668
      5     14294          0.7308

[exited with code 0]
