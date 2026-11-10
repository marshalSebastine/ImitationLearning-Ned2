# Which model, and does it fit the laptop and the robot loop

Written 9.11.26 for Team M (Emranul Hoque, Marshal Sebastine). Offline analysis only: nothing here was run on the robot.

## In short

We compared several picture-to-move models on our own 208 demonstrations, always scoring them on a sheet the model
had never seen, which is our stand-in for the unseen exam tracks. A ready-made picture network that is not retrained
(frozen) hardly beat the simple rule "always push forward", while retraining the whole network on our pictures beat
that rule on every held-out sheet tested so far, by 3 to 27 points (least on the straight line and the arc, where
pushing forward is often right). The best model so far is the ImageNet ResNet-18 retrained on our pictures with the
exact-step output; a much smaller network trained from scratch came within 2 to 3 points of the ResNet-18 when both
used the same output, and is our backup. Both are fast enough for the robot: they need 2 to 27 milliseconds per
picture, while one robot step takes 1.22 seconds, and they train in under an hour per held-out sheet on the laptop
(16 to 49 minutes measured), so no GPU is needed. ACT and diffusion policies were not tested here; the checked
literature and our own timings say they would cost days of laptop training, with no evidence that they handle unseen
tracks better, so we leave them out for now.

How to read the marks: every number is **measured** on our data or laptop unless it says **calculated** (arithmetic on
measured numbers), **literature** (a published result whose checker marked it supported or corrected), **judgement**
or **guess**. Codes in square brackets point to the files in "Sources" at the end. A "point" is one percentage point.

## 1. How the comparison was made

- **Task of the model.** Look at one camera picture and give the next push: a direction (as one of 8 move classes, or
  as an exact step in centimetres), the forearm twist, and when to stop.
- **Held-out sheet test.** Train on five sheets, test on the sixth, six times (one per sheet). This is the stand-in for
  an unseen track. The end-to-end runs so far covered sheets 03 (two curves), 05 (loop with a crossing) and 06 (dashed
  loop), with 3,458, 2,471 and 1,809 moving test pictures from 45, 30 and 25 episodes [R-fte: direction_rows,
  test_episodes]. The six-sheet frozen tests had 569, 2,372, 3,458, 5,271, 2,471 and 1,809 moving test pictures on
  sheets 01 to 06 [P-bb: direction_rows].
- **Main measure, "step within 45".** The share of moving test pictures whose predicted push direction lies within 45
  degrees of the demonstrated step. **"Path within 45"** compares with the direction to where the tip was 3 steps
  later, which is less noisy than one hand-guided step [CFG lines 1684 to 1688].
- **The bar to clear.** The picture-free rule "always forward" scores 0.538 on average over the six sheets, from
  0.402 (sheet 06) to 0.747 (sheet 01) [P-base: within_45_deg].
- **Noise.** The same frozen ResNet-18 probe trained with four different seeds gave means of 0.546 to 0.577 (3.1
  points apart), and on single sheets ranges of 1.5 points (sheet 03) up to 9.1 points (sheet 01) [P-seed, P-bb]. So
  a gap of about 3 points between two single runs is not evidence. No end-to-end run has been repeated with other
  seeds yet.
- **Offline is a screen, not the verdict.** Within one model family, offline scores rarely pick the winner: in Cowling
  (2026, Table 3) hold-out error and robot success were not significantly linked within one data budget (+0.28 to
  -0.22), and Abbas et al. (2026) conclude that offline metrics can shortlist but not replace robot tests
  [literature, L-off check claims 23 and 28: supported].
- **Timing conditions.** All laptop timings were taken while up to 20 agents shared the processor, so they are upper
  bounds [M]. The same small network was timed at 13.5 ms in one run and 2.5 ms in another [R-cnn, R-lp:
  milliseconds_per_picture]; its architecture is identical in both runs [R-cnn, R-lp: settings].

## 2. What was tested here, and what is judged from the literature only

| Option | Tested on our data? | How | Result in one line |
|---|---|---|---|
| Small CNN from scratch (5 convolution blocks, 1.61 M parameters) | yes, end to end | sheets 03, 05, 06; move classes; colour picture, and a hand-made "line and puck" input | step within 45: 0.623 to 0.635 per sheet |
| ResNet-18 ImageNet, frozen plus a head | yes | all six sheets; 4 x 4 grid and summary features; four seeds | 0.577 mean, close to "always forward" (0.538) |
| ResNet-18 ImageNet, fine-tuned | yes, end to end | sheets 03, 05, 06 with move classes and with the exact step; the final candidate is running on all six (sheets 01 and 02 done) | best so far: 0.667 to 0.700 on sheets 03, 05, 06 (exact step) |
| MobileNetV3-Large and EfficientNet-B0, frozen plus a head | yes | all six sheets | 0.576 and 0.546 mean |
| MobileNetV3-Large and EfficientNet-B0, fine-tuned | **no**: configured in config.py, not run | | |
| DINOv2-small and CLIP ViT-B/32, frozen plus a head | yes | all six sheets | 0.545 and 0.547 mean |
| DINOv2 and CLIP, fine-tuned | **no**: literature only | | |
| ACT | **no**: literature only. Its main idea, predicting several future steps, was tested with frozen features; an end-to-end run is queued | chunk-of-5 target, all six sheets | 0.607, inside the seed range of the exact-step probe (0.591 to 0.616) |
| Diffusion policy | **no**: literature only | | |
| Short picture history as input | frozen ResNet-18 only | previous picture; change since the previous picture; pictures 2 and 4 back | at best +2.6 points offline, with a copying risk (section 3.3) |
| Any model on the robot | **no** | | |

Sources for the results column: [R-cnn, R-lp, R-ftc, R-fte, R-fin, P-bb, P-des, P-seed, P-his]; configured but unrun
runs: [CFG lines 1613 to 1615].

## 3. Evidence

### 3.1 Frozen networks barely beat "always forward"

Step within 45, frozen backbone with a head of one hidden layer of 256 on a 4 x 4 grid of its last feature map, 224 x
224 pictures, move-class output [P-bb: within_45_deg per fold; P-base]:

| Frozen backbone | 01 | 02 | 03 | 04 | 05 | 06 | mean | worst sheet | beats "always forward" on | path within 45, mean (worst) |
|---|---|---|---|---|---|---|---|---|---|---|
| "Always forward" (no picture) | 0.747 | 0.565 | 0.522 | 0.485 | 0.509 | 0.402 | 0.538 | 0.402 | | 0.604 (0.433) |
| ResNet-18, ImageNet | 0.756 | 0.530 | 0.565 | 0.560 | 0.583 | 0.467 | 0.577 | 0.467 | 5 of 6 | 0.633 (0.526) |
| MobileNetV3-Large, ImageNet | 0.708 | 0.535 | 0.587 | 0.526 | 0.567 | 0.535 | 0.576 | 0.526 | 4 of 6 | 0.640 (0.556) |
| EfficientNet-B0, ImageNet | 0.685 | 0.513 | 0.514 | 0.532 | 0.588 | 0.447 | 0.546 | 0.447 | 3 of 6 | 0.606 (0.478) |
| DINOv2-small | 0.700 | 0.493 | 0.530 | 0.557 | 0.574 | 0.417 | 0.545 | 0.417 | 4 of 6 | 0.598 (0.459) |
| CLIP ViT-B/32 | 0.714 | 0.477 | 0.587 | 0.552 | 0.513 | 0.438 | 0.547 | 0.438 | 4 of 6 | 0.606 (0.480) |
| ResNet-18, random weights (control) | 0.647 | 0.439 | 0.521 | 0.495 | 0.469 | 0.372 | 0.490 | 0.372 | 1 of 6 | 0.522 (0.409) |

- **Frozen features are not enough.** The pretrained backbones average 0.545 to 0.577, only 0.7 to 3.9 points above
  the picture-free rule, and each loses to it on one to three sheets, mostly the straight line (01) and the arc (02)
  [P-bb, P-base].
- **Pretraining does help a frozen network.** ImageNet ResNet-18 beat the same network with random weights on all six
  sheets, by 4.4 to 11.4 points [P-bb].
- **No clear winner among the pretrained ones.** ResNet-18 and MobileNetV3 lead the other three by about 3 points on
  the mean, which is the size of the seed range (3.1 points). ResNet-18 was ahead of DINOv2-small on all six sheets
  (by 0.3 to 5.6 points) and ahead of CLIP and EfficientNet-B0 on five of six; MobileNetV3 had the best worst sheet
  (0.526 against 0.467) [P-bb, P-seed].
- **The transformers cost more.** DINOv2-small takes 82.9 ms and CLIP ViT-B/32 91.0 ms per 224 x 224 picture,
  against 26.1 ms for ResNet-18; extracting features for all 18,522 pictures took 1,404 s and 935 s against 285 s
  [F-speed: milliseconds_square_224_median; F-ext: seconds].
- **Literature agrees.** On robomimic Square with 200 demonstrations, a ResNet-18 scored 0.58 frozen, 0.92 fine-tuned
  and 0.94 trained from scratch (Chi et al. 2023, Table 5) [literature, L-pre check claim 37: supported].

### 3.2 Training the whole network on our pictures

Trained end to end on 160 x 120 pictures with colour, gamma, blur, noise and 4-pixel shift augmentation and a 3 x 4
grid head; scored on held-out sheets 03 / 05 / 06, mean in brackets [R-cnn, R-lp, R-ftc, R-fte; frozen reference from
P-bb; rule from P-base]:

| Model | Output | Step within 45 | Path within 45 | Balanced class accuracy | Twist sign right | Stopped right (threshold 0.5) |
|---|---|---|---|---|---|---|
| "Always forward" (no picture) | | 0.522 / 0.509 / 0.402 (0.478) | 0.603 / 0.544 / 0.433 (0.527) | 0.125 | | |
| Frozen ResNet-18 plus head (reference) | classes | 0.565 / 0.583 / 0.467 (0.538) | 0.570 / 0.621 / 0.526 (0.573) | 0.292 / 0.296 / 0.272 (0.287) | 0.672 / 0.442 / 0.379 | 0.378 / 0.067 / 0.040 |
| Small CNN from scratch, colour | classes | 0.635 / 0.635 / 0.623 (0.631) | 0.669 / 0.683 / 0.698 (0.683) | 0.317 / 0.368 / 0.369 (0.351) | 0.718 / 0.405 / 0.371 | 0.378 / 0.033 / 0.040 |
| Small CNN from scratch, line-and-puck maps | classes | 0.630 / 0.625 / 0.623 (0.626) | 0.676 / 0.677 / 0.696 (0.683) | 0.336 / 0.350 / 0.369 (0.352) | 0.659 / 0.399 / 0.398 | 0.378 / 0.033 / 0.000 |
| ResNet-18 ImageNet, fine-tuned | classes | 0.663 / 0.659 / 0.640 (0.654) | 0.704 / 0.711 / 0.707 (0.707) | 0.347 / 0.385 / 0.412 (0.381) | 0.703 / 0.372 / 0.422 | 0.289 / 0.133 / 0.040 |
| ResNet-18 ImageNet, fine-tuned | exact step plus stop | 0.700 / 0.667 / 0.675 (0.681) | 0.802 / 0.777 / 0.791 (0.790) | not comparable | 0.845 / 0.669 / 0.802 | 0.689 / 0.200 / 0.200 |

Denominators: moving pictures 3,458 / 2,471 / 1,809; path pictures 3,186 / 2,381 / 1,708; twisting rows 2,630 /
1,721 / 1,430; episodes 45 / 30 / 25 [R-fte: direction_rows, path_direction_rows, twist_rows, test_episodes].

- **Training on our pictures is the largest gain measured.** Against the frozen ResNet-18, the fine-tuned ResNet-18
  (classes) gained 9.8, 7.6 and 17.3 points on sheets 03, 05 and 06, and the small CNN from scratch 6.9, 5.1 and 15.6
  points. Five of these six gaps exceed the frozen probe's seed range on the same sheet (1.5, 6.1 and 2.9 points)
  [R-ftc, R-cnn, P-bb, P-seed]. Caveat: the end-to-end runs also changed the augmentation, the picture size (160 x 120
  against 224 x 224) and the grid (3 x 4 against 4 x 4) [R-ftc and P-bb: settings], so the gain is not the fine-tuning
  alone. The configured run without augmentation would separate the two [CFG line 1613].
- **Against the picture-free rule,** the end-to-end models gain 10.8 to 27.3 points of step within 45 on each sheet;
  the fine-tuned exact-step model gains 17.7, 15.9 and 27.3 points, and 19.9, 23.4 and 35.8 points of path within 45
  [R-fte, R-cnn, R-lp, R-ftc, P-base].
- **ImageNet start against no pretraining.** With the same head, augmentation and pictures (but a different learning
  rate and 8 against 16 epochs), the fine-tuned ResNet-18 beat the small CNN on all three sheets: by 2.9, 2.5 and 1.7
  points of step within 45, 3.5, 2.8 and 0.9 points of path within 45, and 3.0, 1.7 and 4.3 points of balanced class
  accuracy; the twist and the stop were mixed [R-ftc, R-cnn]. The lead is consistent but about one seed range wide,
  so it is suggestive, not settled (judgement). The literature found a tie at this data size (Chi et al. 2023, Table
  5: 0.94 from scratch against 0.92 fine-tuned), and the checker found no opened study with a clear winner between
  the two at about 200 demonstrations [literature, L-pre check claim 37 and its conclusion].
- **The hand-made line-and-puck maps gave no offline gain:** -0.4, -1.0 and 0.0 points of step within 45 against the
  colour picture [R-lp, R-cnn]. Their aim, robustness to new light and tables, is hardly tested by these sheets
  (judgement).
- **The output matters as much as the network.** The same fine-tuned ResNet-18 with the exact-step output beat its
  move-class version by 3.6, 0.8 and 3.5 points of step within 45 and 9.8, 6.7 and 8.4 points of path within 45, and
  got the twist sign right on 0.845, 0.669 and 0.802 of twisting rows against 0.703, 0.372 and 0.422 [R-fte, R-ftc].
  The choice of output has its own section.
- **First results of the final candidate** (exact step plus its own stop output, the audit's cleaning, last weights
  after a fixed 8 epochs) [R-fin: within_45_deg, path_within_45_deg, twist_sign_accuracy,
  tuned_two_episodes_stopped_right; baselines and frozen probes from P-base, P-des, P-seed]:
  - Sheet 01 (straight line): step within 45 0.779 of 578 moving pictures, path within 45 0.856 of 529, and 16 of 20
    episodes stopped right with the tuned two-in-a-row rule. "Always forward" scores 0.747 there (569 pictures before
    the cleaning), so the gain is only about 3 points; forward is nearly always right on a straight line, so this
    sheet says little about bends (judgement).
  - Sheet 02 (arc): step within 45 0.632 of 2,425 moving pictures, path within 45 0.762 of 2,243, twist sign right on
    0.802 of 1,195 twisting rows, and 19 of 45 episodes stopped right with the tuned two-in-a-row rule. That is 6.7
    and 10.1 points above "always forward" (0.565 and 0.660), and 4.8 to 6.0 points of step within 45 above the frozen
    exact-step probe over four seeds (0.572 to 0.584). The arc came out harder than expected: here the reference rule
    "repeat the previous move" (0.648) scores higher than the model.
  - The other four sheets are still running.

### 3.3 One picture or a short history

Frozen ResNet-18 grid with the move-class head, step within 45 per held-out sheet [P-his; P-base]:

| Input | 01 | 02 | 03 | 04 | 05 | 06 | mean | worst sheet | path within 45, mean | stopped right, mean |
|---|---|---|---|---|---|---|---|---|---|---|
| One picture | 0.756 | 0.530 | 0.565 | 0.560 | 0.583 | 0.467 | 0.577 | 0.467 | 0.633 | 0.258 |
| Plus the previous picture | 0.758 | 0.532 | 0.548 | 0.583 | 0.594 | 0.444 | 0.576 | 0.444 | 0.623 | 0.254 |
| Plus the change since the previous picture | 0.770 | 0.558 | 0.597 | 0.604 | 0.636 | 0.454 | 0.603 | 0.454 | 0.640 | 0.278 |
| Plus pictures 2 and 4 back | 0.772 | 0.517 | 0.612 | 0.573 | 0.554 | 0.488 | 0.586 | 0.488 | 0.625 | 0.186 |
| Rule "repeat the previous move" (needs the last move; reference only) | 0.796 | 0.648 | 0.711 | 0.692 | 0.785 | 0.783 | 0.736 | 0.648 | 0.648 | |

- **Only "change since the previous picture" gained:** +2.6 points on the mean, up on five sheets (+1.4 to +5.3)
  but down on sheet 06 (-1.3), and its worst sheet fell from 0.467 to 0.454. The previous picture itself added
  nothing (0.576); pictures 2 and 4 back added 0.9 points but stopped right less often (0.186 against 0.258) [P-his].
- **Why we distrust that gain.** In our stop-and-go loop both pictures are taken with the arm at rest, so the change
  between them mostly shows the last 1 cm step (judgement). The last move alone predicts the next one well: "repeat
  the previous move" scores 0.736, more than any one-picture model offline [P-base]. A history input can therefore
  earn offline points by copying the last move and then fail where the line bends, the copycat problem: history
  policies copy the previous action when consecutive actions are alike and easy to read from the frames (Wen et al.
  2020, Sec. 1 and 4, Table 1) [literature, L-his check claims 6 and 7: supported].
- **Offline gains from history have hidden closed-loop losses before.** A driving policy with history had the better
  validation score (perplexity 0.834 against 0.989) but drove 144.92 against 268.95 units of distance (de Haan et al.
  2019, Table 1, from Wang et al. 2019) [literature, L-his check claim 4: supported]; in Wen et al. 2020 the plain
  history policy had a lower test loss than the authors' fixed version, yet earned less reward (App. 10.6), so the
  offline loss ranked them the wrong way round [literature, L-his check, missed evidence: checked].
- **But the literature is mixed.** With images and 200 demonstrations, robomimic's recurrent policy beat single-step
  cloning (Mandlekar et al. 2021, Table 3), and the image Diffusion Policy worked best with a short history of more
  than one frame, two being a good compromise (Chi et al. 2024, App. B.1) [literature, L-his check claim 29:
  supported; robomimic Table 3: checked]. Those ran at 10 to 20 Hz, where two frames show motion; ours are taken at
  rest (judgement).
- **The previous action as an input is already ruled out** until the professor allows it and it is tested, because it
  would inflate offline scores [A-syn section 11.2, point 16].
- **Cost is no argument either way:** the history heads took 207 s (one picture) to 530 s (pictures 2 and 4 back) to
  train over all six sheets [P-his: seconds], and at run time the previous picture's features can be kept (judgement).

### 3.4 Options judged from the literature only

**ACT (action chunking with transformers).**

- As published: about 80 M parameters, four cameras plus the joint positions, 50 decisions per second, about 5 hours
  of training per task on one RTX 2080 Ti, about 0.01 s per decision on that GPU (Zhao et al. 2023, Sec. IV-C); the
  official code shares one ResNet-18 across all cameras [literature, L-chk check claims 15 (corrected) and 16].
- What its gains rest on: predicting chunks raised success from 1 % (chunk of 1) to 44 % (chunk of 100) in simulation
  at 50 Hz (Fig. 8a), and the CVAE mattered for human data (35.3 % against 2 % without it) [literature, L-chk check
  claims 4 and 12: supported]. A track of ours takes about 89 decisions (calculated, 18,500 pictures / 208
  demonstrations) against 400 to 700 steps per ACT episode [L-chk note 4.1].
- Inputs: the original code always feeds joint positions, which our rule forbids; LeRobot's version can run on
  pictures only [literature, L-chk check claims 21 and 25: supported].
- Cost on a processor: LeRobot's ACT (52 M parameters) took 182 ms per decision on a 2021 MacBook M1 processor
  (Cadene et al. 2026, Table 3) [literature, L-dif check claim 30: supported]; LeRobot's own guide advises against
  training on a processor only [literature, L-chk check, missed evidence 7: checked].
- In our loop every step is its own blocking move with about 0.61 s of fixed cost [A-code (e)], so executing several
  predicted steps per picture saves at most the picture wait and the decision: 0.106 + 0.058 s plus the model's 0.02
  s, about 0.19 s of a 1.22 s step (calculated from A-code (e)).
- Judgement: ACT's useful idea for us, predicting several steps, can be tested inside the ResNet-18 model. Its frozen
  probe scored 0.607 against 0.591 to 0.616 for the exact step over four seeds [P-des, P-seed], and the end-to-end
  chunk run is queued [M]. A full ACT is not justified now.

**Diffusion policy.**

- Strong on the closest published task: real Push-T reached 95 % success with 136 demonstrations, an end-to-end
  ResNet-18, two fixed cameras and 10 Hz (Chi et al. 2023, Table 6) [literature, L-dif check claim 5: supported]. But
  its policies also got the end-effector position, used absolute position targets, and the released simulated Push-T
  configs train on at most 90 demonstrations [L-dif check sections 1 and 3.1: checked].
- With the same network and chunks, plain regression matched flow policies on most of 28 benchmarks, Push-T included
  (Pan et al. 2026, Sec. 3.1) [literature, L-dif check claim 35: supported]. A diffusion policy trained on 30
  demonstrations replayed memorised training trajectories even for unrelated pictures (He et al. 2026, Sec. 3.3 to
  3.4), and generalisation to new scenes grew with the number of training environments, not demonstrations (Lin et
  al. 2025, Sec. 4) [literature, L-dif check sections 4.2 and 4.3: checked].
- Cost: measured processor times elsewhere are 5,244 ms per decision (Lee et al. 2024, Table 7, a 4-core Intel
  processor) and over 5,000 ms (Cadene et al. 2026, Table 3, M1 processor) [literature, L-dif check claims 29 and 30:
  supported]; a right-sized 17 M-parameter head might take tens of ms (the checker's own count, unchecked arithmetic)
  [L-dif check section 6]. Training recipes use 75 to 3,000 passes over the data, about 500 in LeRobot's Push-T recipe
  and 3,000 in Chi et al. [L-dif check section 3.4 and claim 7]. One full pass of our ResNet-18 encoder takes about 4
  to 8 minutes (calculated: twice the measured epoch of half the pictures [R-fold]; config.py notes about 7 minutes
  [CFG line 1675]), so 500 passes would take about 33 to 67 hours and 3,000 passes 8 to 17 days, per model and per
  held-out sheet, before the denoising network (calculated).
- Judgement: it does not fit the laptop, and nothing shows it would handle unseen tracks better; a fair comparison,
  with the same encoder and a held-out sheet, belongs on the GPU pool if that ever becomes available.

**DINOv2 or CLIP, fine-tuned.**

- Fully fine-tuning a large vision transformer on few demonstrations collapsed in VC-1 (Adroit 59.3 to 15.9,
  Majumdar et al. 2023, Table 5), and fine-tuning the MVP transformer hurt (44.0 frozen against 31.1 fine-tuned with
  augmentation) while ResNets gained when augmentation was used (Hansen et al. 2023, Table 5) [literature, L-pre check
  claims 12 and 17: supported]. On the other side, CLIP ViT-B/16 fine-tuned with a 10 times lower learning rate was
  the best encoder on Square (0.98) [literature, L-pre check claims 37 and 38: supported]. DINOv2 ViT-S/14 did worse than expected in a robustness study
  (Burns et al. 2024, App. A.1) [literature, L-pre check claim 34: supported].
- Cost: a ViT-B/16 needs 17.56 G multiply-adds per 224 x 224 picture against 1.81 G for ResNet-18 [literature, L-pre
  check claim 54: supported]. Our count gives 0.72 G for ResNet-18 and 0.19 G for the small CNN at 160 x 120 [K].
- Measured here: frozen, both were below ResNet-18 (section 3.1) and 3 to 5 times slower per picture [F-speed].
- Judgement: not worth days of processor time.

## 4. Does it fit the laptop?

Measured on the laptop (AMD Ryzen AI 5 PRO 340, 6 cores, no NVIDIA GPU, PyTorch 2.15 on the processor) [M]. An
"epoch" here is a random half of the fold's training pictures (train_share_per_epoch 0.5) plus a pass over the
validation episodes [R-fte: settings; TP].

| Model | Parameters (backbone + head) | Multiply-adds per 160 x 120 picture | Seconds per epoch | Minutes per held-out sheet | Weights file |
|---|---|---|---|---|---|
| Small CNN from scratch | 1.02 M + 0.59 M = 1.61 M [K] | 0.19 G [K] | 52.7 to 115.5 (16 epochs) [R-fold] | 15.8 to 24.7 [R-cnn, R-lp: minutes] | 6.5 MB |
| ResNet-18 fine-tuned | 11.18 M + 1.57 M = 12.75 M [K] | 0.72 G [K] | 131.5 to 229.3 (8 epochs); 240.8 to 290.0 with two runs at once; 215.2 to 572.8 as more jobs started [R-fold, R-fin] | 20.4 to 28.9; 34.6 with two runs at once; 48.8 with more jobs [R-ftc, R-fte, R-fin: minutes] | 51.1 MB |
| Frozen backbone plus head | head only | | head training 20 to 257 s for all six sheets [P-bb: seconds] | features once: 116 s (MobileNetV3) to 1,404 s (DINOv2) [F-ext] | |

(Weights file sizes are the saved checkpoints in the run folders, 6,489,127 and 51,137,519 bytes.)

- **One configuration on all six held-out sheets** takes about 1.6 to 2.5 hours (small CNN) or 2.0 to 2.9 hours
  (ResNet-18), up to about 5 hours when other jobs share the processor (calculated: six times the measured minutes per
  sheet). Three seeds take three times that. The load matters: on sheet 02 the final candidate's epochs grew from 215
  to 573 s while other jobs started [R-fin: history[].seconds], so runs for decisions should not share the processor
  (judgement).
- **The exam model** (trained on all six sheets once the settings are fixed, as the audit asks [A-syn section 11.2,
  point 8]) sees about 1.0 to 1.8 times the training pictures of one fold, depending on the fold it is compared with
  and on whether 15 % is still held back for validation (calculated from the 18,053 cleaned rows [M] and the test rows
  per sheet [P-bb, R-fin]), so it should take about 20 to 60 minutes (calculated).
- **An independent check** in the audit measured 37.6 ms per training picture for ResNet-18 at 160 x 120 (batch 32,
  2 threads), which it turned into about 11.6 minutes per pass over 18,451 pictures [A-syn line 943].
- **What does not fit:** diffusion recipes (days to weeks per model, section 3.4, calculated) and fine-tuning large
  vision transformers (about 10 times the multiply-adds of ResNet-18 for ViT-B/16; literature and calculated). The
  GPU pool is still unconfirmed (VPN not connected on 3.11) [M], so the plan must not depend on it.
- Judgement: the laptop is enough for the recommended model, its backup, seeds and the remaining planned runs
  (about 20 to 32 hours of processor time in all, section 8), well before 20.12.26.

## 5. Does it fit the robot loop?

One step of the recording tool takes 1.222 s (median of 17,863 replayed steps; 5th to 95th percentile 0.826 to
1.586 s): the move itself 1.031 s (84 %), waiting for a fresh picture 0.106 s, deciding and drawing 0.058 s, logging
0.003 s [A-code (e)]. A model adds its own time to the deciding part:

| Model | Time per picture (measured, busy laptop) | Share of a 1.222 s step (calculated) | Added to a sheet 04 run of 168 steps, about 205 s of driving (calculated) |
|---|---|---|---|
| Small CNN | 2.5 to 13.5 ms [R-lp, R-cnn] | 0.2 to 1.1 % | 0.4 to 2.3 s |
| ResNet-18 at 160 x 120 | 16.7 ms backbone, 90th percentile 19.1 [F-speed]; 21.5 to 26.9 ms whole model [R-fte, R-ftc] | 1.4 to 2.2 % | 2.8 to 4.5 s |
| MobileNetV3, EfficientNet-B0 at 160 x 120 (backbone) | 15.0 and 22.7 ms [F-speed] | 1.2 and 1.9 % | 2.5 and 3.8 s |
| DINOv2-small, CLIP ViT-B/32 at 224 x 224 (backbone) | 82.9 and 91.0 ms [F-speed] | 6.8 and 7.4 % | 13.9 and 15.3 s |
| ACT, 52 M (literature, M1 processor) | 182 ms [L-dif check claim 30] | 14.9 % | 30.6 s |
| Diffusion policy (literature, other processors) | over 5,000 ms [L-dif check claims 29 and 30] | over 400 % | about 14 to 15 minutes |

Sheet 04's 168 steps and 205 s of driving are the median of its demonstrations at 1.22 s per step [A-code (e); M].

- **Every model tested here fits easily:** the time to finish a track is set by the arm's moves, not by the model.
  So the model should be chosen on finishing (direction and stop), not on speed (judgement).
- **How the times were taken:** one random picture of the network's input size through the model, median of 20 runs
  after 3 warm-up runs, without decoding and resizing the JPEG [TP lines 386 to 402]. The run-time decider must decode
  the robot's 640 x 480 JPEG and resize it exactly as training_dataset.py does (BGR to RGB, INTER_AREA to 160 x 120)
  [A-code (c)]; that cost is not measured (guess: a few ms). The fixed-wrist joint solve adds 0.005 s per step
  [A-code (e)].
- **One program on the robot:** the model has to run inside the recording tool's decider, which is called once per
  fresh picture, because only one program may be connected to the robot [A-code (c); project rules].

## 6. Recommendation

**Main model.** The ImageNet ResNet-18 (timm resnet18.tv_in1k), trained end to end on our 160 x 120 pictures with the
3 x 4 grid head, the exact-step output (tip forward and left, forearm twist, lean and hand changes) plus its own stop
output, one picture as input, the audit's cleaning, a fixed 8 epochs on half the pictures each, and the last weights.
This is the configured run resnet18_final_candidate [CFG lines 1797 to 1804]. Why:

1. Best offline numbers wherever it was tested: step within 45 0.700 / 0.667 / 0.675 and path within 45 0.802 /
   0.777 / 0.791 on sheets 03 / 05 / 06 [R-fte]. With the final settings it gave 0.779 and 0.856 on the straight sheet
   01, level with the best frozen probes there (up to 0.782 and 0.851), and 0.632 and 0.762 on the arc, sheet 02,
   above every frozen probe there (up to 0.584 and 0.713) [R-fin; all probe files in P-bb's folder, fields
   within_45_deg and path_within_45_deg].
2. With the same move-class output, a large lead over frozen features (7.6 to 17.3 points on each sheet) and a small
   but consistent lead over the small CNN (1.7 to 2.9 points) [R-ftc, R-cnn, P-bb].
3. It fits: 20 to 49 minutes per held-out sheet to train, 16.7 to 26.9 ms per picture, about 2 % of a robot step
   [R-fold, R-fin, F-speed, R-fte, A-code].
4. The literature supports an end-to-end ResNet-18 with shift augmentation at 100 to 200 demonstrations (Chi et al.
   2023, Tables 5 and 6; Hansen et al. 2023, Tables 2 and 5) [literature, L-pre check claims 15 to 17, 37, 39].
5. Keeping the last weights of a fixed schedule follows the literature: in robomimic the checkpoint with the lowest
   validation loss was 57 to 100 % worse than the best one, the last checkpoint 12 to 35 % worse (Mandlekar et al.
   2021, Table 29) [literature, L-off check claim 12 and the checker's calculation].

Before any robot run: finish the six sheets, add two more seeds, then train the exam model on all six sheets.

**Backup.** The small CNN from scratch (five convolution blocks, 1.61 M parameters, 0.19 G multiply-adds, 6.5 MB of
weights) with the same head, augmentation, cleaning and the exact-step plus stop output. That exact-step version
still has to be run. Why:

1. With the same move-class output it came within 1.7 to 2.9 points of the fine-tuned ResNet-18 on each held-out
   sheet and beat every frozen network [R-cnn, R-ftc, P-bb].
2. It needs no outside weights, so it is safe if the professor objects to a network that started from ImageNet
   (judgement; see the questions).
3. It trains in 16 to 25 minutes per sheet and runs in 2.5 to 13.5 ms per picture [R-cnn, R-lp].
4. Small networks trained from scratch with shift augmentation matched pretrained ones at about 100 to 200
   demonstrations (Hansen et al. 2023, Table 2 and Fig. 8; Chi et al. 2023, Table 5) [literature, L-pre check claims
   15, 16 and 37: supported].

Switch to the backup if the professor rules out ImageNet weights, if seeds show the ResNet-18 lead is noise, or if
it wins the robot comparison.

**How to choose between the two:** on the robot, after the team decides and with a checklist, with about 24 paired
blocks of two trials from matched starts on the harder sheets and new tracks. With a real gap of 20 points (70
against 90 %) this picks the better model about 96 % of the time, with 10 points (80 against 90 %) only about 83 %
[literature, L-eva check section 3, point 2]. The note estimates 2.2 to 3.0 robot hours for these 48 trials [L-eva
note section 3.4]; with slower resets 3.7 to 4.5 hours (calculated from 10.6 to 12.9 trials per hour, L-eva check
claim 40).

**Not now:** DINOv2-small, CLIP ViT-B/32 and EfficientNet-B0 (lower frozen results, slower, and the literature warns
against fine-tuning transformers on small data); ACT and diffusion policies (literature only, days of processor
training, no evidence for unseen tracks); a picture history as input (an offline gain that copying could explain;
test it only after the copy test in section 8, then on the robot).

## 7. What is still unknown

1. **Closed-loop behaviour of any model.** Nothing has driven the robot. Offline scores shortlist; they rarely pick
   the winner among good models of one family [L-off check claims 23 and 28].
2. **Most of the six-sheet picture for the final settings.** The final candidate has finished sheets 01 and 02 only
   [R-fin]; sheet 04 (the meander, the longest sheet) has no end-to-end result at all yet. On the arc the model scored
   below the reference rule "repeat the previous move" (0.632 against 0.648) [R-fin, P-base], which is worth a look
   in the error analysis before the robot.
3. **Seed noise of the end-to-end runs.** None was repeated. The 1.7 to 2.9 point lead of the ResNet-18 over the small
   CNN is about the 3.1-point seed range of the frozen probe [P-seed].
4. **What part of the end-to-end gain comes from fine-tuning and what from augmentation and picture size:** not
   separated (the no-augmentation run is configured, not run) [CFG line 1613].
5. **Fine-tuned MobileNetV3 and EfficientNet-B0:** configured, not run. Frozen, MobileNetV3 matched ResNet-18 and had
   the best worst sheet (0.526 against 0.467) [P-bb].
6. **True run-time speed:** all timings come from a shared processor (the same network timed 2.5 and 13.5 ms), and the
   whole decider (decoding, resizing, model, solver) has not been timed.
7. **Memory at the crossing and the dashed gaps.** Offline, the one-picture fine-tuned model did no worse on the loop
   sheets 05 and 06 than on sheet 03 (0.667 and 0.675 against 0.700) [R-fte], but only robot runs tell whether one
   picture is enough there.
8. **Training length.** At epoch 8 the validation loss had nearly levelled off (exact-step run: 0.4885 to 0.4867,
   0.5405 to 0.5349, 0.5723 to 0.5721 from epoch 7 to 8) [R-fold], so a longer schedule might help a little; this
   must be judged on held-out sheets with the direction measures, not on validation loss.
9. **Exam tracks may hold features none of the six sheets has** (sharper corners, line ends away from the paper edge,
   other dash rhythms) [A-syn section 11.3, question 20]; no model study in the literature tested unseen track shapes
   [L-pre note 4.6 and its check].

## 8. Experiments still needed

| Experiment | Why | Laptop or robot hours |
|---|---|---|
| Finish the queued runs: final candidate on sheets 03 to 06, exact step on 01, 02 and 04, mirror on 06, chunk on 03, 05 and 06 | complete the six-sheet picture before choosing | about 4 to 9 laptop hours (calculated from 20 to 49 minutes per sheet, the upper end with other jobs sharing the processor) |
| Two more seeds of the final candidate on all six sheets | the main model's lead is about one seed range | about 5 to 7 laptop hours |
| Small CNN with the exact-step plus stop output, audit cleaning, all six sheets, three seeds | the backup in its final form | about 5 to 7.5 laptop hours |
| ResNet-18 without augmentation on sheets 03, 05, 06 (configured) | separate fine-tuning from augmentation | about 1.2 to 1.5 laptop hours |
| Fine-tuned MobileNetV3 on sheets 03, 05, 06 (configured) | the only frozen network level with ResNet-18; untested end to end | about 1 to 1.5 laptop hours (guess: its epoch time is not measured) |
| Training length: final candidate with 16 epochs on sheets 03, 05, 06 | validation loss still creeping down at epoch 8 | about 2.5 to 3 laptop hours |
| Time the whole decider on a quiet laptop (decode, resize, model, solve; 1, 4 and 6 threads) | replace the busy-laptop upper bounds | about 0.5 laptop hours |
| Copy test of the "change since previous" probe: give it the current picture twice and count how often it repeats the last move or stops (Chuang et al. 2022, Table 5) [L-his check claim 18, corrected: an offline test] | check whether its +2.6 points come from copying | about 0.5 laptop hours |
| Paired robot comparison of the main model and the backup, 24 blocks of two trials | offline numbers cannot pick between close models | about 2.2 to 4.5 robot hours (after the team decides) |
| Train the exam model on all six sheets | the model for the demo | about 0.5 to 1 laptop hour per seed (calculated) |

## 9. Questions for the professor

1. May the network start from ImageNet weights (pictures from elsewhere) if all its training for the task uses only
   our demonstrations? If not, the small CNN from scratch becomes the main model.
2. Must the final policy run on the laptop next to the robot, and may the university GPU pool be used for training if
   it becomes available?
3. May the model see the previous camera picture as a second input, or average its own output over the last few
   pictures? Both still use only the camera.

## Sources

Run results (folder D:/AI_Project/.claude/worktrees/training-analysis/04_runs/):

- [R-cmp] comparison.md (all runs, means and worst folds; matched sheets 03, 05, 06).
- [R-cnn] 2026-11-09_small_cnn_scratch_classes/metrics.json (fields within_45_deg, path_within_45_deg,
  class_balanced_accuracy, twist_sign_accuracy, episodes_stopped_right, minutes, milliseconds_per_picture, settings).
- [R-lp] 2026-11-09_small_cnn_line_and_puck_classes/metrics.json (same fields).
- [R-ftc] 2026-11-09_resnet18_finetune_classes/metrics.json (same fields).
- [R-fte] 2026-11-09_resnet18_finetune_exact/metrics.json (same fields, plus direction_rows, path_direction_rows,
  twist_rows, test_episodes, tuned_two_episodes_stopped_right).
- [R-fin] 2026-11-09_resnet18_final_candidate/fold_without_path_01.json and fold_without_path_02.json (the first two
  finished sheets of the final candidate, written 18:53 and 19:42; fields as above, history[].seconds).
- [R-fold] the fold_without_path_*.json files of the four end-to-end runs above (history[].seconds and
  history[].validation_loss per epoch); checkpoint sizes from checkpoints/*.pt.
- [P-bb] 2026-11-09_frozen_feature_probes/backbones.json (per-fold within_45_deg, path_within_45_deg,
  class_balanced_accuracy, twist_sign_accuracy, episodes_stopped_right, seconds, settings).
- [P-his] 2026-11-09_frozen_feature_probes/history.json (same fields).
- [P-des] 2026-11-09_frozen_feature_probes/designs.json (chunk_of_5_steps, exact_step).
- [P-seed] 2026-11-09_frozen_feature_probes/seeds.json (classes and exact, seeds 11 to 13).
- [P-base] 2026-11-09_frozen_feature_probes/baselines.json (sheets: always_forward, repeat_previous_move).

Laptop speed and features (folder D:/AI_Project/03_data/processed/v01_training_index/features/):

- [F-speed] speed.json (parameters_million, milliseconds_square_224_median, milliseconds_wide_160_median and 90th).
- [F-ext] extraction_summary.json (seconds per backbone for all 18,522 pictures).

Code and settings (folder D:/AI_Project/.claude/worktrees/training-analysis/02_code/):

- [CFG] config.py (TRAINING_RUN_PARAMETERS lines 1568 to 1602, TRAINING_RUNS lines 1607 to 1616, epoch note line
  1675, path measure lines 1684 to 1688, final candidate lines 1797 to 1804).
- [TP] train_policy.py (models lines 45 to 121, training loop lines 306 to 382, timing lines 386 to 402).
- [K] my count of parameters and multiply-adds, rebuilt from those definitions without training or timing:
  C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/models/count_model_size.py
  and its output count_model_size.out.json in the same folder (ResNet-18 at 224 x 224 gives 1.81 G, matching
  torchvision's card).

Audit and session facts (folder C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/):

- [A-code] audit/code_actions/note.md, sections (c) decider interface, (e) timings, (f) outputs.
- [A-syn] audit/AUDIT_SYNTHESIS.md, section 11 (lines 1129 to 1277) and line 943 (compute check).
- [M] design/main_session_facts.md (laptop, timing conditions, queue, robot time per trial).

Literature notes, each claim used only where its checker marked it supported or corrected (folder
.../scratchpad/literature/, each with note.md and check/note.md):

- [L-pre] pretrained_representations; [L-chk] action_chunking; [L-dif] diffusion_and_generative;
  [L-his] history_and_causal_confusion; [L-off] offline_vs_closed_loop; [L-eva] evaluation_statistics.
