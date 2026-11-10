# Check: Diffusion policy and other generative policies

Key `diffusion_and_generative`. A sceptical check of the 44 claims and the conclusion in the researcher's note
(`../note.md`), written 9 November 2026 by a checking sub-agent. Nothing was run on the robot or trained on the laptop.

How I checked: every web page was re-opened by me through an automatic page reader (WebFetch), and every number in a claim
was compared digit by digit with what the page reader returned. Where a number mattered I asked a second, differently worded
question. Configuration files from the Diffusion Policy code were read as raw text. My own cost count is the small script
`unet_cost.py` in this folder (arithmetic only, it downloads nothing).

Labels used below: **checked** = I read it in the source myself; **unchecked** = my own calculation, inference or judgement.
**measured** = a number from an experiment; **speculation** = an explanation, an estimate or an opinion.

---

## 1. The result in plain words

1. **42 of the 44 claims are supported.** The papers say what the claims say, and the numbers match. Two claims need a
   correction: claim 13 changes one word inside a quotation, and claim 40 compares its own estimate with CPU measurements of
   different networks. Several supported claims need a caveat (section 3).
2. **The most important thing the researcher missed (checked).** The authors' own released Push-T settings train on at most
   **90** of the 206 demonstrations (`max_train_episodes: 90` in `task/pusht_image.yaml`, `task/pusht_lowdim.yaml` and the
   published experiment config for image Push-T). The README says the released experiment configs reproduce the paper's
   tables. So the well-known 0.91 on image Push-T most likely came from **90 demonstrations, not 200** (this last step is my
   inference; the paper itself says 200).
3. **The verdict "no diffusion policy now, keep the classifier" survives, but the strongest reasons are different from the
   ones given.** (a) Behaviour Transformers' own ablation shows that classifying into bins does most of the work, and uniform
   bins (like our move names) did about as well as learned ones. (b) A study presented at ICLR 2026 found that a diffusion
   policy trained on few demonstrations mostly replays memorised training trajectories, which is the opposite of what unseen
   tracks need. (c) Diffusion recipes train for hundreds to thousands of passes over the data, which is days on our CPU even
   with a small network.
4. **"Seconds added to each step" is not established.** The 256 million-parameter network is a choice, not a requirement:
   Diffusion Policy's own transformer variant has 9 million parameters, a 2026 comparison used about 20 million, and a
   follow-up study found that a bigger U-Net did not help. By my count a 17 million-parameter U-Net with 10 steps costs about
   2.5 GFLOP per decision (1.5 for the ten denoising steps, 1.0 for encoding two 84x84 pictures): tens of milliseconds at the
   assumed CPU speed (unchecked; time it).
5. **Diffusion Policy's published advantage came with things we do not have.** Its Push-T policies also get the end-effector
   position as input (our model may use only the camera), and it did best with absolute position targets; our steps are
   relative moves, closer to the velocity control under which its lead over the baselines shrank.

---

## 2. Verdict for every claim

| Claim | Verdict | What I found (location in the source) |
|---|---|---|
| 1 | supported | Sec. 2 to 3 and Table 7: 2 observations in, 16 predicted, 8 executed, 6 for real Push-T. The CNN U-Net beats the transformer on Push-T (Tables 1 and 2). Code: FiLM gives a learned scale and bias when `cond_predict_scale` is true (the configs set it true; the code default is false). Block Push uses 3 observations. |
| 2 | supported | Table 2, Push-T column: CNN 0.91/0.84, transformer 0.78/0.66, LSTM-GMM 0.69/0.54, IBC 0.75/0.64. |
| 3 | supported | Table 1: CNN 0.95/0.91, transformer 0.95/0.79, IBC 0.90/0.84, BET 0.79/0.70, LSTM-GMM 0.67/0.61; 3 seeds, 50 initial conditions, checkpoints every 50 epochs (Sec. 5.2). The robomimic tasks used only 22 initial conditions because of an evaluation bug (footnote); Push-T is not affected. |
| 4 | supported, caveat | Sec. 5.1, Table 3, Table 7 as claimed. But the released configs cap training at 90 episodes (see 3.1). |
| 5 | supported | Table 6 and App. C.1 digit by digit. Success means an end-state IoU above the lowest IoU among the demonstrations (Table 6 caption). Two of five recorded cameras were given to the policy. |
| 6 | supported | App. C.1.2: fixed 12 h per method. The only GPU named in the paper is the RTX 3080, for inference. README: Linux machine with an Nvidia GPU, no training times. |
| 7 | supported | Sec. 5.2 (3000 / 4500 epochs) and the generic config (3050 epochs, batch 64, lr 1e-4, weight decay 1e-6, 500 warm-up steps, EMA, `cuda:0`). The generic config's default task is robomimic Lift, not Push-T; the published image Push-T config has the same values. |
| 8 | supported | Sec. 4.1, Fig. 3, 40 steps of the best checkpoint. |
| 9 | supported | Sec. 4.1 and 4.3 (authors' explanation). |
| 10 | supported | Sec. 3.2, App. A.3. |
| 11 | supported, caveat | Table 7: 256, 264 (Transport), 67 M; encoder 22 M (45 M for Transport). LeRobot Table 3: 263 M. Caveat: 22 M for one-camera Push-T equals two ResNet-18s; one is about 11 M (torchvision lists 11.7 M including its classifier layer), so that entry looks like a copy of the two-camera rows (my inference). The transformer variant has only 9 M (Table 8). |
| 12 | supported | Table 5 (robomimic Square, 500 epochs) and Table 6. Fine-tuning used a 10 times lower learning rate (Sec. 5.4). |
| 13 | **corrected** | Numbers right, quotation wrong by one word: Sec. 3.4 says the latency is 0.1 s "on a Nvidia 3080 GPU" (not "an"). App. A.4 and Table 7 say 16 DDIM steps on the real robot, Sec. 3.4 says 10: the inconsistency is real. |
| 14 | supported | Sec. 9. Missed from the same section: it also lists poor results with too few demonstrations as an inherited behaviour-cloning limitation. |
| 15 | supported | Sec. 4.2, Sec. 5.3, Fig. 4. |
| 16 | supported | Sec. 4.2 (authors' explanation). |
| 17 | supported | Sec. 4.3, 5.3, Fig. 5. The latency test is simulated. |
| 18 | supported | Table 3 and Sec. 7.3 to 7.5. App. A.5 is one sentence; the Fig. 15 caption names neither the task nor the sizes. I could not read the figure either. |
| 19 | supported | Sec. 4.4, Fig. 6. The minimum-training-error checkpoint rule is stated for **real** Push-T (App. C.1.2). |
| 20 | supported | Sec. 2, App. D, App. E.2. The 4-stage conv and max-pool encoder (32, 64, 128, 256 channels) is the pushing-task encoder; the sweeping tasks used 26-layer ResNets. |
| 21 | supported | Table 3 and App. C.1 (Table 5) digit by digit. |
| 22 | supported | Table 4, App. C.1 and C.3 digit by digit. Insert-Blue and the sorting task give partial credit. |
| 23 | supported | App. C.2, Tables 6 and 7. The D4RL timing ran on TPUv3. The paper's figure of 5.6 times slower probably refers to training speed (17.9 against 101.3 steps per second, my reading); total train-plus-evaluation time differs by 5.2 times. |
| 24 | supported | Sec. 3, Fig. 4 (low-data regime). |
| 25 | supported | Table 6: IBC 0.00 success with position and with velocity control. |
| 26 | supported | Sec. 2.1 to 2.4, Fig. 2, App. B.2. |
| 27 | supported | Tables 1 and 3. Missed from Table 3: uniform quantisation instead of k-means kept 0.90 / 0.96 / 0.90 of full BeT. |
| 28 | supported | VQ-BeT Tables 2 and 9. VQ-BeT's "max" is the highest coverage within a rollout, not Diffusion Policy's "best checkpoint". Missed from Table 2: MLP behaviour cloning scored 0.13 on image Push-T. |
| 29 | supported | VQ-BeT Table 7 digit by digit (573.49 / 5243.82, 18.06 / 207.25, 4.73 / 83.28 ms). Table 3's caption says the simulated timing used 10 diffusion iterations, following the Diffusion Policy real-world settings. Neither the real-robot step count nor the network size is given. |
| 30 | supported | LeRobot Sec. 3.3, Table 3 digit by digit. Column order inferred from the platform list in Sec. 3.3; latency is averaged over the calls that did not time out. |
| 31 | supported | Sec. IV-C, V-B, Tables I and II. Real-task final success 88, 96, 84, 20, 64 and 92 %: "80 to 96 % on most" means 4 of 6 tasks. |
| 32 | supported | Sec. VI-A and VI-B, Fig. 7a and 7c. The chunk-size result averages two simulated tasks with scripted and human data, with temporal ensembling off. |
| 33 | supported | Tables I, III, IV and Sec. IV-C. The P5000 timings were measured on robomimic Square; the DDIM Push-T score (0.78) was taken from another paper (ParaDiGMS); Push-T is averaged over 200 rollouts. |
| 34 | supported | Tables 1 and 5; abstract. Missed: in the same paper's Table 2, Consistency Policy reached only 0.595 on Push-T. |
| 35 | supported | Sec. 3.1 and 3.2, 4.3, Table 1, Table 4. Table 1 names neither the metric nor which Push-T version. All models were capacity-matched to about 20 M parameters (App. B.2). |
| 36 | supported | Sec. 4, Sec. 5.1, Table 6. |
| 37 | supported, caveat | Dataset card: 206 episodes, 25,650 frames, 96x96x3, 10 fps, credits Diffusion Policy. But Diffusion Policy's own configs train on at most 90 of them, so "comparable to our data" overstates what Diffusion Policy was trained on. |
| 38 | supported | Model card: step 175k of 200k, batch 64, 500 episodes, 65.4 % vs 64.2 %. Success here means at least 95 % overlap with the target; the average best overlap is 0.955 vs 0.957. No hardware or time. |
| 39 | supported | 11,689,512 parameters, 1.81 GFLOPS; the page gives no input size (its inference transform crops to 224). |
| 40 | **corrected** | Arithmetic confirmed by my own count (251.5 M parameters, 1.17 G multiply-adds per step, 24.4 / 235 GFLOP per decision). But the "10 to 40 times slower" comparison is not like for like (see 3.3). |
| 41 | supported (arithmetic) | My count gives 32 to 129 days, 7.5 to 30 days, 1.1 to 4.3 days, and 1.2 to 4.7 h. It holds only for the full 256 M network, 84x84 crops and the assumed 50 to 200 GFLOP/s (see 3.4). |
| 42 | supported (as reasoning) | Consistent with BeT Sec. 2 and Table 3. Caveat: a per-step classifier can still switch between options from one step to the next (see 5.3). |
| 43 | supported | Diffusion Policy Sec. 5: LSTM-GMM, IBC, BET; no plain MSE baseline. VQ-BeT does include one (see 4.4). |
| 44 | supported | True for the opened papers (VQ-BeT's real tests also only randomise positions within a fixed scene, Sec. 4.7, App. A.2). Diffusion Policy also ran qualitative perturbation tests (Sec. 6.1). Two papers that do bear on generalisation were missed (4.2, 4.3). |

---

## 3. Corrections and caveats in detail

### 3.1 Diffusion Policy's Push-T was trained on 90 demonstrations, not 200 (checked; one inference)

- Source: Chi et al., code repository real-stanford/diffusion_policy, files `diffusion_policy/config/task/pusht_image.yaml` and
  `task/pusht_lowdim.yaml` (both set `max_train_episodes: 90`, `val_ratio: 0.02`, `seed: 42`), the published experiment
  config https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml (run dated
  16 January 2023: 3050 epochs, batch 64, 96x96 images cropped to 84x84, 100 DDPM steps, `max_train_episodes: 90`), and
  `diffusion_policy/dataset/pusht_image_dataset.py` (the training mask is cut down to at most `max_train_episodes` episodes by
  a seeded random choice; the episodes left out go to the validation set).
- The README (https://github.com/real-stanford/diffusion_policy) says of the experiment folder: "For each experiment used to
  generate Table I,II and IV in the paper", it provides the config, logs and checkpoints.
- So the published simulated Push-T results most likely used 90 training demonstrations out of 206 (my inference from the
  configs; the paper's Table 3 says 200). For Team M this cuts both ways: Diffusion Policy did well with even fewer
  demonstrations than we have, but the "same size as our data" comparison in claim 37 is not right for what it was trained on.

### 3.2 The quotation in claim 13 (checked)

The paper writes "a Nvidia 3080 GPU" in Sec. 3.4 (arXiv v5). The rest of claim 13 is right.

### 3.3 Claim 40: the comparison with measured CPU times (checked and unchecked parts)

- My independent count from `conditional_unet1d.py` and `conv1d_components.py` (`unet_cost.py`, unchecked arithmetic)
  reproduces the researcher's figures: 251.5 M parameters for the Push-T network (255.6 M with robomimic's 7-D actions, which
  matches the published 256 M), 1.17 G multiply-adds per denoising step, 24.4 GFLOP per decision with 10 steps and 235 GFLOP
  with 100 steps.
- Claim 29's 5,244 ms is for the **transformer** variant, a different and much smaller network: Diffusion Policy's Table 8
  lists 9 M parameters in simulation and 80 M for real Push-T (checked). VQ-BeT states neither its size nor its number of
  steps. Comparing it with a U-Net estimate gives a meaningless ratio.
- Claim 30's LeRobot network is a U-Net-type model of 263 M with 10 steps, so it is the comparable one. LeRobot's current
  default configuration (`configuration_diffusion.py`, checked) predicts 64 actions and executes 32, without random crops and
  with BatchNorm, unlike the 16 / 8 of the original Push-T setting. If the benchmark used these defaults, each denoising step
  would be about 4 times the work I counted (my inference; the paper does not say which settings it timed).
- The 50 to 200 GFLOP/s speed of the laptop CPU remains an assumption. The researcher's own suggestion stands: time one
  forward pass with random weights on the laptop before deciding anything.

### 3.4 Claim 41: what the estimate does and does not show (unchecked arithmetic)

- The arithmetic is right for the full 256 M network. The diffusion head does not need to be that size (section 4.5). With a
  smaller U-Net the image encoder dominates and the cost per training picture falls from about 10 GFLOP to about 3.2 to
  4.8 GFLOP (two frames at 84x84). The real obstacle is then the number of passes over the data that diffusion recipes use:
  3000 epochs (Chi), about 500 (LeRobot: 200k steps x 64 / 25,650 frames) and 75 to 800 (Lin, Hu et al.). At 500 epochs my
  count gives about 2 to 8 days for a 17 M U-Net at 50 to 200 GFLOP/s, against 1.2 to 4.7 hours for 30 epochs of the
  classifier.
- Using full 640x480 pictures instead of 84x84 crops would multiply the encoder cost by about 44 (22 GFLOP per picture for a
  ResNet-18 forward pass), for the classifier too.

---

## 4. Evidence the researcher missed

### 4.1 Diffusion Policy's own configs: 90 training demonstrations (checked)

See 3.1.

### 4.2 Lin, Hu, Sheng, Wen, You, Gao (2025), Data Scaling Laws in Imitation Learning for Robotic Manipulation, ICLR 2025 (oral)

URL opened: https://arxiv.org/html/2410.18647v3 (v3 lists Hu first) and https://www.iclr.cc/virtual/2025/oral/31769.
All checked:

- **Measured:** a Diffusion Policy (CNN U-Net, DDIM, fine-tuned DINOv2 ViT-L/14 encoder, data from a hand-held UMI gripper)
  generalised to unseen environments and objects roughly as a power law of the **number of training environments and
  objects**, while the number of demonstrations alone showed no clear power law (Sec. 4.1 to 4.3, Fig. 5 and 7).
  Performance stopped improving at about 50 demonstrations per environment-object pair (Sec. 4.3).
- **Measured:** 32 pairs with 50 demonstrations each gave 85 to 92.5 % success in 8 unseen environments on four tasks
  (Sec. 5, Table 1).
- **Measured:** a larger visual encoder helped; a larger diffusion U-Net did not (Sec. 6, Table 2).
- **Cost:** the largest model (over 396 M parameters) trained 75 h on 8 A800 GPUs (App. C).
- **Why it matters for us (judgement):** for unseen tracks, the variety of track shapes we train on is probably the main
  lever, whatever the action head. Six sheets is a small number of "environments".

### 4.3 He, Liu, Sznaier Camps, Bruno, Sartoretti, Schwager (2026), Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative, ICLR 2026

URLs opened: https://arxiv.org/html/2505.05787v1 (arXiv title "Demystifying Diffusion Policies: Action Memorization and
Simple Lookup Table Alternatives", five authors) and https://iclr.cc/virtual/2026/poster/10009689 (ICLR title, adds Joseph
Bruno). All checked:

- **Measured:** a Diffusion Policy trained with the standard code on 30 cup-grasping demonstrations reproduced one of the
  training trajectories almost exactly, even between training positions and even when shown pictures of a cat or a dog
  (Sec. 3.3 and 3.4).
- **Measured:** a lookup table that retrieves the nearest training picture (pretrained ResNet-18 with a contrastive loss)
  matched its 100 % rollout success in this one set-up, at 0.009 s instead of 2.65 s per decision (hardware not stated)
  (Sec. 4, Table 1).
- **Authors' limitation:** all their datasets are small; behaviour at larger scale is unknown (Sec. 6).
- **Why it matters for us (judgement):** with about 200 demonstrations, a diffusion head may learn to replay stretches of the
  six training tracks rather than local "follow the line" rules. Long action chunks (Chi's 16 predicted, 8 executed, which
  is 8 cm blind in our step size) make this worse on unseen tracks.

### 4.4 A plain-regression baseline that did badly on image Push-T (checked, weak)

Lee et al. (2024), VQ-BeT, ICML 2024, https://arxiv.org/html/2403.03181v2, Table 2: an MLP-based behaviour-cloning baseline reached
0.13 final coverage on image Push-T and 0.65 on state Push-T, against 0.66 and 0.73 for the Diffusion Policy CNN. The paper
does not describe this baseline (loss, chunking, encoder), so it is weak evidence, but it points the other way from Much Ado
and should be mentioned for balance.

### 4.5 Small diffusion networks exist and work (checked)

- Chi et al., Diffusion Policy arXiv v5, Table 8: the transformer variant has 9 M parameters for simulated tasks (including
  Push-T) and 80 M for real Push-T.
- Pan et al. (2026), Much Ado About Noising, ICLR 2026, https://arxiv.org/html/2512.01809v3, App. B.2: all compared models were
  capacity-matched to about 20 M parameters, and the flow policy still scored 0.97 on Push-T (Table 1).
- Wu, Wang, Chen, Pang, Xu (2025), On-Device Diffusion Transformer Policy for Efficient Robot Manipulation (LightDP),
  https://arxiv.org/html/2508.00697v1 (ICCV 2025 according to a CVF search listing I did not open): the transformer Diffusion
  Policy has 8.97 M parameters; pruned to 4.76 M with 4 steps it ran in 2.72 ms on an iPhone 13 (Core ML; the page does not
  say which compute unit), against 90.6 ms for the original with 100 steps (Table 1), with Push-T success 0.747 against 0.754
  for the re-evaluated original (Table 2). On a Jetson Orin NX the original took 244.68 ms (App. I). No CPU measurements.

### 4.6 Uniform bins work as well as learned ones (checked)

Shafiullah et al. (2022), BeT, NeurIPS 2022, https://arxiv.org/html/2206.11251v2, Table 3: replacing k-means bins by uniform
quantisation kept 0.90 / 0.96 / 0.90 of full BeT (CARLA / block push / Kitchen); dropping the offsets kept 0.94 / 0.95 /
0.78. Our hand-made move names are closer to uniform bins, so this is the most direct support for the classifier.

### 4.7 Other details that change how the numbers read (checked)

- Diffusion Policy Sec. 9 also lists poor results with too few demonstrations as a limitation inherited from behaviour
  cloning.
- Push-T scores are not comparable across papers: Diffusion Policy reports coverage (best checkpoint / last 10), VQ-BeT a
  within-rollout maximum, LeRobot a success rate at 95 % overlap (about 65 %) next to an average best overlap of about 0.96.
- OneDP's Table 2 re-run gave Consistency Policy 0.595 on Push-T, against 0.82 in the Consistency Policy paper.
- IBC's real-robot implicit model trained in 5.0 h against 5.8 h for the explicit one (App. C.2, Table 7): the energy-based
  method was not slower to train there.

### 4.8 Outside generative policies, but on our exact question (abstract only, checked)

Giusti et al. (2016), A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots, IEEE Robotics
and Automation Letters 1(2):661-667, https://infoscience.epfl.ch/record/218621 (abstract page only): a deep network used as
an image **classifier** of the trail direction from one camera reached about human accuracy and gave preliminary quadrotor
control on unseen trails. Only the abstract was read; classes and numbers not checked.

---

## 5. Challenge to the conclusion

### 5.1 The evidence paragraph

- "0.91 coverage on simulated image Push-T (200 demonstrations)": change to "the paper says 200, the released configs train on
  at most 90". These policies also received the end-effector position as input. In VQ-BeT's re-run, the same CNN policy
  scored 0.66 final coverage, so the number depends on who runs it and how it is scored.
- "95 % success on real Push-T (136 demonstrations, two cameras; LSTM-GMM 20 %, IBC 0 %)": correct (Table 6). It also used
  two fixed cameras plus absolute position actions; we have one moving camera and relative steps.
- "67M to 256M parameter denoising network": correct for the CNN variant, but the transformer variant is 9 M (80 M for real
  Push-T), so "heavy" describes the published U-Net, not diffusion as such.
- "Measured CPU inference takes seconds per call (5.2 s, 4-core Intel; over 5 s, M1)": correct as measured, but neither is our
  x86 laptop, the 5.2 s is for a different network with unstated steps, and LeRobot's settings for its timing are not stated.
- "Regression matched generative (flow) policies on most benchmarks, including Push-T": correct (Much Ado Sec. 3), with the
  balancing VQ-BeT result in 4.4.
- "Discretising actions into classes already captures several modes (BeT)": correct, and BeT's ablation (4.6) makes it
  stronger than stated.

### 5.2 The judgement

- **Training on the laptop CPU: the verdict holds.** Even a small diffusion head needs many more passes over the data than
  a classifier, and that multiplies the image-encoder cost (3.4). Days, not weeks, for a right-sized model, but still days.
- **Running it: overstated.** "Seconds added to each 1.2 s step" applies to the 256 M network with many steps, or to timings
  of other code on other CPUs. A right-sized head (5 to 20 M) with 10 steps should cost tens of milliseconds by my count
  (unchecked). Only a timing test on the laptop settles this.
- **Why it would or would not beat the classifier.** The conclusion bets on multimodality. Much Ado's authors argue that
  multimodality is not where generative policies gain; their hypothesis is that iterative computation with noise keeps
  actions sensible on unfamiliar states. That would matter for unseen tracks. But He et al. (4.3) found the opposite kind of
  behaviour with small data: memorised trajectories replayed even for unrelated pictures. Neither tested line following. So
  there is no evidence that a diffusion head generalises better to unseen tracks, and some evidence that with about 200
  demonstrations it may generalise worse. The conclusion is right not to expect gains, partly for the wrong reason.
- **Diffusion Policy's lead depended on absolute position actions** (Sec. 4.2, Fig. 4: with velocity control the baselines
  did better and Diffusion Policy worse). Our relative 1 cm or 5 degree steps are closer to velocity control, and our robot
  stands still while the model decides, so the latency advantage (Sec. 4.3, Fig. 5) does not apply either.

### 5.3 The "borrow the cheap parts" list

- **Random crops:** in Push-T the camera is fixed and the crop moves the picture by at most 6 of 96 pixels. In our task the
  position of the line in the picture **is** the information that decides the move, so a large crop shift can make the label
  wrong. Keep shifts small compared with the image distance that changes the correct move, or compare with and without
  (my reasoning, untested).
- **GroupNorm:** Diffusion Policy changed BatchNorm to GroupNorm for stable training together with weight averaging (EMA). No
  paper here shows a gain for a classifier trained without EMA. Optional, not a known improvement.
- **Spatial softmax:** reasonable for a task where "where is the line" matters; not tested for classifiers in these papers.
- **Short action chunks:** in a stop-and-go loop, executing k predicted steps per picture divides the number of pictures (and
  part of the time) by k but moves k cm without looking. Chi's best of 8 steps is 0.8 s of motion at 10 Hz; for us it would
  be 8 cm blind. Memorisation (4.3) argues for short chunks on unseen tracks. Predicting several steps but executing one
  still gives the training benefit.
- **Per-step mode switching:** a classifier cannot average two opposite moves, but it can still switch between them from one
  picture to the next. In Diffusion Policy's Fig. 3, BET, which is itself a bin classifier with offsets, switched between
  pushing from the left and from the right within one rollout. At the loop crossing and the gaps of the dashed loop,
  commitment needs history (two frames, the previous move) or chunking, not a diffusion head.

### 5.4 The plan "try diffusion later on a GPU pool, as a comparison"

Good, if the comparison is fair: the same encoder, the same pictures and crops, the same number of predicted and executed
steps, a plain-regression or MIP baseline as in Much Ado, and evaluation on a held-out sheet (train on five, test on the
sixth). Lin, Hu et al. (4.2) suggest that adding more sheet shapes would probably help unseen tracks more than any action
head.

---

## 6. My recalculation (unchecked arithmetic; `unet_cost.py`)

Counting the layers of `ConditionalUnet1D` (kernel 5, two residual blocks per level, two in the middle, FiLM with scale and
bias, horizon 16, Push-T conditioning 2 x (64 + 2) plus a 128-wide step embedding):

| Network | Parameters | Multiply-adds per denoising step | GFLOP per decision, 10 steps (with two 84x84 ResNet-18 frames) |
|---|---|---|---|
| down_dims 512 / 1024 / 2048 (published) | 251.5 M | 1,170 M | 24.4 (235 at 100 steps) |
| 256 / 512 / 1024 (my guess at the 67 M real-world size) | 64.9 M | 294 M | 6.9 |
| 128 / 256 / 512 | 17.3 M | 75 M | 2.5 |
| 64 / 128 / 256 | 4.9 M | 19 M | 1.4 |

At the assumed 50 to 200 GFLOP/s (not measured), 2.5 GFLOP is 12 to 50 ms. Training per picture (forward plus backward, about
3 times forward, two frames): 10.1, 4.8, 3.5 and 3.2 GFLOP for the four rows; a ResNet-18 classifier on one 84x84 frame
1.5 GFLOP. These counts ignore Python and memory overheads, which the measured GPU timings (370 ms on an RTX 4090 in LeRobot,
against 100 ms on an RTX 3080 in Chi et al.) show can dominate.

---

## 7. What I could not check

- The RSS 2023 PDF (I used arXiv v5 and the RSS abstract page, which lists seven authors without Tedrake, in a different
  order: Chi, Feng, Du, Xu, Cousineau, Burchfiel, Song).
- Values that exist only in figures (Diffusion Policy Fig. 15; Much Ado Fig. 2 and 8).
- Which settings LeRobot used for its timing benchmark, and how many steps VQ-BeT used on the robot.
- Venue pages I saw only in search listings: BeT (NeurIPS 2022), VQ-BeT (PMLR 235:26991-27008), ACT (RSS XIX paper 016),
  Consistency Policy (RSS XX paper 071), OneDP (PMLR 267:63399-63416), Much Ado (ICLR 2026 proceedings), LightDP (ICCV 2025).
  The arXiv papers themselves were opened.
- Any measurement of diffusion-policy training on a CPU: I found none either.
