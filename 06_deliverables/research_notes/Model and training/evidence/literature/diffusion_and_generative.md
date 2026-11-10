# Diffusion policy and other generative policies

Key: `diffusion_and_generative`. Literature notes for Team M (Emranul Hoque, Marshal Sebastine), written 9 November 2026 by a
research sub-agent. Nothing here was run on the robot or on the laptop. Claim numbers in square brackets, for example [DG-5],
match the ids `diffusion_and_generative-5` in the structured result.

How to read the labels:

- **measured**: a number the paper's authors measured in an experiment.
- **speculation**: the authors' explanation or opinion, or my own estimate or judgement (always marked "my" or "judgement").
- **definition**: what a method is or how it was set up.
- **checked**: I read it in the source myself. **unchecked**: my own calculation or judgement, or something I could not open.

Numbers were read from the HTML text of each paper with an automatic page reader. The numbers that matter most here (Diffusion
Policy Tables 1, 2, 6 and 7 and Section 3.4; IBC Tables 3, 4 and 7; LeRobot Table 3; VQ-BeT Table 7; Consistency Policy
Table III; ACT Section IV-C) were read twice with different questions and agreed both times. Values that exist only inside
figures could not be read.

---

## The short answer, in plain words

1. **What a diffusion policy is.** A normal policy network looks at the picture and outputs one action straight away. A
   diffusion policy starts from random noise and "cleans it up" in 10 to 100 small steps, each one a pass through a large
   network, until the noise has become a short sequence of future actions. Because it starts from random noise, it can
   produce "go round the left" on one run and "go round the right" on another, instead of averaging them into "go straight
   into the block", which is what a plain mean-squared-error (MSE) regressor does.
2. **It works very well on the published task closest to ours.** Push-T (pushing a T-shaped block to a target from camera
   pictures) with 136 to 200 demonstrations: 95 % success on a real robot, where the older methods reached 0 to 20 % [DG-5].
3. **It is expensive.** The recipes use a denoising network of 67 to 256 million parameters, thousands of passes over the
   data or 12 hours of training per model, and a GPU [DG-6, DG-7, DG-11]. On a CPU, two groups measured about 5 s or more per
   decision [DG-29, DG-30]. Our loop is about 1.2 s per step now.
4. **Our classifier already avoids the main problem diffusion solves.** A softmax over our 10 move names can give
   probability to two directions at once, and taking the most likely one commits to a single choice. Turning actions into
   classes is exactly how Behaviour Transformers handle several modes [DG-26, DG-27]. A controlled study from 2026 found
   that, with the same network and the same action chunks, plain regression and diffusion-type (flow) policies did about
   equally well on most benchmarks, including Push-T [DG-35].
5. **Judgement.** A diffusion policy is not realistic on the laptop CPU, and I do not expect it to beat a well-built
   classifier on our task. The cheap parts of the Diffusion Policy recipe (its image encoder, random crops, predicting a few
   steps at once) are worth copying into the classifier. A real diffusion policy is worth a try only later, on a GPU pool,
   as a comparison.

---

## Sources I opened

| # | Authors, year | Title | Venue | URL(s) I opened |
|---|---|---|---|---|
| S1 | Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song (2023; extended 2024) | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | RSS 2023 (RSS XIX, paper 026, DOI 10.15607/RSS.2023.XIX.026; the RSS author list has seven names, without Tedrake). Extended journal version in IJRR | https://arxiv.org/html/2303.04137v5 (full text with appendix); https://www.roboticsproceedings.org/rss19/p026.html (abstract page); https://diffusion-policy.cs.columbia.edu/ (project page) |
| S2 | Same group | Code and README | GitHub | https://github.com/real-stanford/diffusion_policy ; raw files `conditional_unet1d.py`, `train_diffusion_unet_hybrid_workspace.yaml`, `task/pusht_image.yaml` |
| S3 | Florence, Lynch, Zeng, Ramirez, Wahid, Downs, Wong, Lee, Mordatch, Tompson (2021) | Implicit Behavioral Cloning | Proceedings of the 5th Conference on Robot Learning, PMLR 164:158-168 (2022) | https://ar5iv.labs.arxiv.org/html/2109.00137 ; https://proceedings.mlr.press/v164/florence22a.html |
| S4 | Shafiullah, Cui, Altanzaya, Pinto (2022) | Behavior Transformers: Cloning k modes with one stone (BeT) | NeurIPS 2022 | https://arxiv.org/html/2206.11251v2 ; https://mlanthology.org/neurips/2022/shafiullah2022neurips-behavior/ |
| S5 | Lee, Wang, Etukuru, Kim, Shafiullah, Pinto (2024) | Behavior Generation with Latent Actions (VQ-BeT) | ICML 2024, PMLR 235:26991-27008 | https://arxiv.org/html/2403.03181v2 ; https://arxiv.org/abs/2403.03181 |
| S6 | Zhao, Kumar, Levine, Finn (2023) | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT) | RSS 2023 (paper 016) | https://arxiv.org/html/2304.13705v1 ; https://www.roboticsproceedings.org/rss19/p016.html |
| S7 | Prasad, Lin, Wu, Zhou, Bohg (2024) | Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation | RSS 2024 (paper 071) | https://arxiv.org/html/2405.07503v1 ; https://roboticsproceedings.org/rss20/p071.html |
| S8 | Wang, Li, Mandlekar, Xu, Fan, Narang, Fan, Zhu, Balaji, Zhou, Liu, Zeng (2025) | One-Step Diffusion Policy: Fast Visuomotor Policies via Diffusion Distillation (OneDP) | ICML 2025, PMLR 267:63399-63416 | https://arxiv.org/html/2410.21257v1 ; https://proceedings.mlr.press/v267/wang25ba.html |
| S9 | Pan, Anantharaman, Huang, Jin, Pfrommer, Yuan, Permenter, Qu, Boffi, Shi, Simchowitz (2026) | Much Ado About Noising: Dispelling the Myths of Generative Robotic Control | ICLR 2026 | https://arxiv.org/html/2512.01809v3 ; https://mlanthology.org/iclr/2026/pan2026iclr-much/ |
| S10 | Cadene, Alibert, Capuano and others, Wolf (2026) | LeRobot: An Open-Source Library for End-to-End Robot Learning | arXiv 2602.22818 (no venue stated) | https://arxiv.org/html/2602.22818 |
| S11 | Hugging Face (LeRobot team) | Model card `lerobot/diffusion_pusht`; dataset card `lerobot/pusht` | Web pages | https://huggingface.co/lerobot/diffusion_pusht ; https://huggingface.co/datasets/lerobot/pusht |
| S12 | PyTorch team | torchvision ResNet-18 model page | Documentation | https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html |

About S1: the project page says the RSS 2023 paper is arXiv version 4 and the IJRR paper is arXiv version 5. I could not read
the RSS PDF or the arXiv v4 PDF as text with my tools, so **every section, table and figure number for S1 below refers to
arXiv v5** (the extended version). The experiments I cite (Push-T, real Push-T, encoder, DDIM) are the RSS ones, but the
numbering in the RSS PDF may differ.

**Not opened** (seen only as citations or search results): Ho et al. 2020 (DDPM); Song et al. (DDIM); Nichol and Dhariwal 2021
(iDDPM); Mandlekar et al. 2021 (robomimic, the LSTM-GMM baseline); He et al. 2016 (ResNet); the IJRR article page on Sage
(it refused access, HTTP 403).

---

## 1. Diffusion Policy (Chi et al., RSS 2023; extended IJRR version)

### 1.1 What it is [DG-1]

- The policy is a conditional denoising diffusion model over a short **sequence** of future actions. At test time it starts
  from Gaussian noise and removes noise in K steps, conditioned on features of the last two camera pictures (S1, Sec. 2 and 3).
- Two denoising networks: a 1-D temporal convolution U-Net ("CNN"; it did best on Push-T) and a transformer. The image
  features enter through FiLM layers (a learned scale and shift) inside every residual block (S2, `conditional_unet1d.py`).
- Typical horizons: look at the last 2 observations, predict 16 actions, execute 8 of them, then look again (S1, Table 7).
  For real Push-T: execute 6.
- The CNN network for simulation has 256 M parameters; the real-world CNN network has 67 M (S1, Table 7) [DG-11]. LeRobot's
  re-implementation lists 263 M (S10, Table 3).

### 1.2 Push-T in simulation [DG-2, DG-3, DG-4]

- Task (S1, Sec. 5.1): a circular end-effector pushes a grey T-shaped block onto a fixed red target; start positions are
  random. The paper says Push-T was adapted from IBC. Two versions: RGB picture input, or 9 keypoints of the block; **both
  also get the end-effector position** (proprioception). Our model may only use the camera, so this is a difference.
- Data: 200 proficient human demonstrations (S1, Table 3). Pictures 96x96, randomly cropped to 84x84 in training (Table 7).
  LeRobot's copy of the data has 206 episodes and 25,650 frames at 10 frames per second (S11, dataset card) [DG-37]. So in
  number of demonstrations and pictures, **Push-T is about the size of our data** (208 demonstrations, about 18,500
  pictures), but its pictures are tiny.
- Metric: how much of the target the block covers, not a success rate (S1, Sec. 5.2). Results are "best checkpoint /
  average of the last 10 checkpoints", 3 training seeds, 50 start positions, checkpoints every 50 epochs (Sec. 5.2).

| Method (S1) | Image Push-T (Table 2) | State Push-T (Table 1) |
|---|---|---|
| LSTM-GMM (robomimic BC-RNN with a mixture head) | 0.69 / 0.54 | 0.67 / 0.61 |
| IBC (implicit, energy-based) | 0.75 / 0.64 | 0.90 / 0.84 |
| BET (Behaviour Transformer) | not run | 0.79 / 0.70 |
| Diffusion Policy, CNN | **0.91 / 0.84** | **0.95 / 0.91** |
| Diffusion Policy, transformer | 0.78 / 0.66 | 0.95 / 0.79 |

Important for us: **none of Diffusion Policy's baselines is a plain classifier or a plain MSE regressor.** LSTM-GMM, IBC and BET
are all themselves "multimodal" methods. So these tables do not tell us whether a diffusion policy beats a simple CNN
classifier (judgement) [DG-43].

LeRobot's re-implementation reached 65.4 % success (a yes/no score per episode) over 500 episodes, against 64.2 % for the
original checkpoint, after 175 k of a planned 200 k training steps at batch size 64 (S11, model card) [DG-38].

### 1.3 Real Push-T [DG-5]

Set-up (S1, App. C.1 and D): UR5 arm; 5 RealSense cameras recorded, of which **2 views** were given to the policy, at
320x240 and 10 Hz (cropped to 288x216 in training); demonstrations by SpaceMouse teleoperation at 10 Hz; **136
demonstrations**; 20 test episodes per method from the same start positions; an episode ends when the end-effector stays in
the end zone for more than 0.5 s, or after 60 s.

Results (S1, Sec. 6, Table 6):

| Method | IoU with target | Success | Time (s) |
|---|---|---|---|
| Human demonstrations | 0.84 | 1.00 | 20.3 |
| IBC, position / velocity control | 0.14 / 0.19 | 0.00 / 0.00 | 56.3 / 41.6 |
| LSTM-GMM, position / velocity | 0.24 / 0.25 | 0.20 / 0.10 | 47.3 / 51.7 |
| Diffusion Policy, transformer, trained end to end | 0.53 | 0.65 | 57.5 |
| Diffusion Policy, ImageNet-pretrained encoder | 0.24 | 0.15 | 55.8 |
| Diffusion Policy, R3M-pretrained encoder | 0.66 | 0.80 | 31.7 |
| **Diffusion Policy, CNN, encoder trained end to end** | **0.80** | **0.95** | **22.9** |

(The page reader could not see which of the last three columns are CNN; the text says the best model was the CNN with an
end-to-end encoder.)

The robustness tests (a hand waved in front of the camera, the block moved during the push) are single, qualitative episodes
with no numbers (Sec. 6.1, Fig. 8).

### 1.4 How it handles several possible actions (multimodality) [DG-8, DG-9]

- **Measured (qualitative):** Fig. 3 (Sec. 4.1) shows one Push-T state where the block can be pushed from the left or the
  right, with 40 rollout steps from the best checkpoint. Diffusion Policy uses both sides across runs but commits to one side
  within a run. LSTM-GMM and IBC lean towards one side; BET switches back and forth because it has no temporal consistency.
- **Authors' explanation (speculation):** the random starting noise decides which "basin" (mode) the action falls into, and
  the noise added during denoising lets samples move between modes (Sec. 4.1). Predicting a whole action sequence keeps the
  robot on one mode for several steps (Sec. 4.3).
- A 2026 study tested this explanation and disagreed for Push-T, see Section 3.5 [DG-35].

### 1.5 The image encoder [DG-10, DG-12]

- **Definition (S1, Sec. 3.2; App. A.3):** a standard ResNet-18, **not pretrained**, trained end to end with the policy; one
  encoder per camera view. Two changes: the global average pooling is replaced by a **spatial softmax** (keeps where things
  are in the picture), and **BatchNorm is replaced by GroupNorm** (for stable training together with an exponential moving
  average of the weights). Random crops in training, a fixed centre crop at test time.
- **Measured (S1, Sec. 5.4, Table 5; robomimic Square task, 500 epochs):**

| Encoder | Trained from scratch | Pretrained, frozen | Pretrained, fine-tuned |
|---|---|---|---|
| ResNet-18 (ImageNet-21k) | 0.94 | 0.58 | 0.92 |
| ResNet-34 (ImageNet-21k) | 0.92 | 0.40 | 0.94 |
| ViT-B/16 (CLIP) | 0.22 | 0.70 | 0.98 |

  So frozen pretrained features did worse, and the small ResNet-18 trained from scratch was as good as anything except a
  fine-tuned CLIP ViT. On real Push-T, the end-to-end encoder (95 %) beat R3M (80 %) and ImageNet (15 %) (Table 6).

### 1.6 Number of demonstrations [DG-4, DG-5, DG-18]

From S1 Table 3 and Sec. 7: sim Push-T 200; real Push-T 136; real mug flip 250; sauce pouring 90; sauce spreading 90;
bimanual egg beater 210, mat unrolling 162, shirt folding 284. Appendix A.5 says Diffusion Policy beat LSTM-GMM at every
training-set size tried (Fig. 15), but the text does not say which task or which sizes, and I could not read the figure.

### 1.7 Training length, time and hardware [DG-6, DG-7]

- **Simulation:** image tasks trained for 3000 epochs, state tasks for 4500 (S1, Sec. 5.2). The released generic config for
  image tasks uses 3050 epochs, batch size 64, learning rate 1e-4, weight decay 1e-6, 500 warm-up steps, weight averaging on,
  device `cuda:0` (S2, config file).
- **Real Push-T:** a fixed **12 hours of training per method** (S1, App. C.1.2). Sauce tasks: 1000 epochs (App. C.2.2).
- **Hardware:** the paper text I read does not name the training GPU. The README says to install on a Linux machine with an
  NVIDIA GPU and gives no training times (S2). The only GPU named in the paper is the RTX 3080 used for the latency figure.
- I found **no measurement of Diffusion Policy training on a CPU** anywhere.

### 1.8 Denoising steps and inference time [DG-13, DG-14]

- Training always uses 100 noise levels. In simulation, inference uses 100 steps of iDDPM; on the real robot, DDIM with
  **16** steps, to cut the latency (S1, App. A.4 and Table 7).
- Section 3.4 states that with 100 training and **10** DDIM inference steps, DDIM "enables 0.1s inference latency on an
  Nvidia 3080 GPU". Note the paper says 10 in one place and 16 in another for the real robot.
- **Limitations (authors' own assessment, Sec. 9):** more computation and latency than simpler methods such as LSTM-GMM;
  predicting action sequences only partly hides this, and may not be enough where high-rate control is needed. They point to
  faster solvers and consistency models as future work.

### 1.9 Other findings that touch our design [DG-15, DG-16, DG-17, DG-19]

- **Position versus velocity control (Sec. 4.2, Fig. 4).** Measured: Diffusion Policy did better when predicting absolute
  positions than velocities, while BC-RNN and BET did worse with positions. The authors' reasons are explicitly speculation:
  multimodality is stronger with position actions, and position actions suffer less from compounding error. Our relative
  "1 cm or 5 degree" steps are closer to velocity actions, but our robot stops while the model decides, so the latency part of
  the argument does not apply to us (judgement).
- **Action horizon (Sec. 4.3, 5.3, Fig. 5).** Measured: executing more than one predicted action helps temporal consistency
  and makes the policy less likely to copy pauses in the demonstrations; too long a horizon slows reactions; 8 steps was best
  for most tasks. With position control, performance held with up to 4 steps of simulated latency.
- **Training stability (Sec. 4.4, Fig. 6).** Measured: IBC's training loss went down smoothly while its test success
  oscillated, which made choosing a checkpoint hard; for Push-T, IBC checkpoints were chosen by the lowest training-set action
  error for this reason (App. C). Diffusion avoids estimating the normalising constant that energy-based models need.

---

## 2. Implicit Behavioral Cloning (Florence et al., CoRL 2021)

### 2.1 What it is [DG-20]

- The policy is the action with the **lowest energy** under a learned energy function E(picture, action). Training uses an
  InfoNCE loss with sampled "counter-example" actions. Choosing an action at test time is an optimisation: sample many
  candidate actions and keep refining towards low energy (derivative-free optimisation), or use an autoregressive version or
  Langevin sampling (S3, Sec. 1 and 2, App. B). For the 2-D pushing tasks they used derivative-free optimisation (App. D).
- Image encoder: a small 4-layer convolution and max-pooling network followed by an MLP, not a ResNet (App. E.2).

### 2.2 Pushing results [DG-21, DG-22]

**Simulated pushing (S3, Sec. 4, Table 3; App. C.1):** xArm6 in PyBullet, **2,000 scripted demonstrations** (the script
re-aims when the block slips), 2-D actions, 3 seeds of 100 test episodes.

| Method | Single target, from states | Two targets, from states | Single target, from pixels (180x240) |
|---|---|---|---|
| Implicit (EBM) | 100 | 99.0 | **100** |
| Mixture density network | 100 | 99.7 | 10.0 |
| Explicit MSE regression | 98.3 | 89.7 | 87.0 |
| Nearest neighbour | 4.0 | 0.0 | 4.3 |

**Real robot (S3, Sec. 4, Table 4; App. C.3):** xArm6 moving in a plane, **RGB pictures only** (recorded 640x360, fed at
90x120), 2-D change-of-position actions, policy at 5 Hz, demonstrations given with a mouse; 20 runs x 3 seeds = 60 trials
per method and task.

| Task (demonstrations) | Implicit (EBM) success % | Explicit MSE success % |
|---|---|---|
| Push red then green block (95) | 85.0 +/- 5.0 | 35.0 +/- 18.0 |
| Push red/green, two valid orders (410) | 88.3 +/- 7.6 | 55.0 +/- 18.0 |
| Insert blue block, about 1 mm gap (223) | 83.3 +/- 3.8 | 6.7 +/- 9.4 |
| Sort blue from yellow (502) | 48.3 +/- 4.6 | 19.6 +/- 1.5 |

This is the **closest published analogue to our task** (a planar arm pushing objects from RGB alone, 95 to 502
demonstrations). The implicit model beat MSE regression by a wide margin. But again the comparison is with MSE regression,
not with a classifier over discrete moves (judgement).

### 2.3 Cost [DG-23]

S3, App. C.2, Table 7 (real-robot models): 100 k training steps, batch 128, on **8 V100 GPUs**: implicit 5.0 h, explicit 5.8 h
(the authors attribute the similar times to image processing dominating). Inference on one RTX 2080 Ti: **7.22 ms** for the
implicit model (1024 candidate actions, 3 refinement rounds) against 3.49 ms explicit; 9.25 ms with 2048 candidates. On the
D4RL tasks (Table 6, TPU) the implicit model took 3.4 h against 0.66 h for train plus evaluation, about 5 times more.
Limitations (Sec. 7, App. B.4): more compute, plain derivative-free sampling fails above about 5 action dimensions, Langevin
training needs extra stabilisation.

### 2.4 Generalisation and later evidence [DG-24, DG-25]

- **Measured, toy task (S3, Sec. 3, Fig. 4):** regressing the pixel position of a dot from an image with only 10 or 30
  training images, the implicit model had 1 to 2 orders of magnitude lower test error than MSE, mainly outside the region
  covered by training data. This is the only evidence I found that a generative-style policy extrapolates better, and it is a
  toy problem.
- **Later evidence (S1, Table 6):** in Diffusion Policy's real Push-T, IBC reached 0 % success.

---

## 3. Other generative policies and close relatives

### 3.1 Behaviour Transformers, BeT (NeurIPS 2022) [DG-26, DG-27]

- **Definition (S4, Sec. 2):** cluster all demonstration actions with k-means into k "bins"; a transformer predicts which bin
  (classification with a focal loss) **plus a small correction (offset) for each bin**; only the offset of the true bin is
  trained. The motivation (Sec. 2, Fig. 2): MSE regression assumes one Gaussian, so with two valid actions it outputs their
  average, which fits neither.
- **Measured (Table 1):** on the multimodal block-pushing task, plain MLP regression with MSE reached one block 67 % of the time
  but never pushed both blocks into targets (0 %); BeT pushed both in 71 %. In Kitchen, MSE regression completed no tasks.
- **Ablation (Table 3, scores relative to full BeT, CARLA / block push / Kitchen):** no offsets 0.94 / 0.95 / 0.78; no
  binning 0.94 / 0.25 / 0.68. So **the classification into bins did most of the work; the offsets mattered most in the
  highest-dimensional task**. k used: 24 for block push, 32 for CARLA, 64 for Kitchen (Table 4).
- Relevance (judgement): our "move name plus exact step" is the same idea as BeT, with hand-made bins instead of k-means.

### 3.2 VQ-BeT (ICML 2024) [DG-28, DG-29]

- **Definition (S5, Sec. 2.3, 3.2):** learned residual vector-quantised codes for action chunks, instead of k-means bins.
- **Measured, image Push-T (Table 2, final coverage):** VQ-BeT 0.68, Diffusion Policy CNN 0.66, Diffusion Policy transformer
  0.45, BeT 0.01. Best coverage (Table 9): Diffusion Policy CNN 0.82 ahead of VQ-BeT 0.73.
- **Measured, real-robot inference time (Table 7):** on an RTX A4000 GPU, Diffusion Policy (transformer) 573 ms, VQ-BeT 18 ms,
  plain BC 4.7 ms. **On the robot's onboard 4-core Intel CPU: Diffusion Policy 5,244 ms, VQ-BeT 207 ms, plain BC 83 ms.** The
  number of denoising steps behind Table 7 is not stated; in simulation (Table 3) they used 10.

### 3.3 ACT, Action Chunking with Transformers (RSS 2023) [DG-30, DG-31, DG-32]

- **Definition (S6, Sec. III to IV):** a conditional variational autoencoder (CVAE) with transformers; ResNet-18 image
  backbones; 4 cameras; predicts 100 future actions at once; about 80 M parameters.
- **Measured (Sec. IV-C):** training takes about 5 hours on one 11 GB RTX 2080 Ti; inference about 0.01 s on the same machine.
  50 demonstrations per real task (100 for one task); 80 to 96 % success on most real tasks (Tables I and II).
- **Measured ablation (Sec. VI-B, Fig. 7c):** without the CVAE, success with **human** demonstrations fell from 35.3 % to 2 %;
  with scripted (consistent) demonstrations it made almost no difference. Chunk size (Fig. 7a): about 1 % with 1 step, about
  44 % with 100 steps.
- **Measured on CPU (S10, Table 3):** ACT (52 M) took 182 ms per call on a 2021 MacBook Pro M1 CPU [DG-30].
- Relevance (judgement): hand-guided human demonstrations are noisy, as in ACT's human data, but ACT controls two 7-joint arms
  from 4 cameras, far from our 10 move names.

### 3.4 Faster diffusion: Consistency Policy (RSS 2024) and One-Step Diffusion Policy (ICML 2025) [DG-33, DG-34]

- Both first train a normal diffusion policy, then **distil** it into a model that needs 1 to 3 steps. So they **add**
  training cost: OneDP says 2 to 10 % extra (S8, abstract).
- **Consistency Policy (S7):** simulation on an NVIDIA P5000: DDPM 100 steps 110 ms, DDIM 15 steps 11 ms, Consistency Policy
  1 step 1 ms (Table III). Real robot, on a laptop RTX 3070-Ti GPU: DDPM 100 steps about 1.5 s per forward pass (Sec. IV-C),
  DDIM 192 ms, Consistency Policy 21 ms (Table IV). State Push-T coverage: DDPM 0.87, DDIM 0.78, Consistency Policy 0.82
  (1 step) and 0.84 (3 steps) (Table I). The introduction also states (not from their own table) that DDPM Diffusion Policy can
  take about one second per action on an NVIDIA T4. No CPU numbers.
- **OneDP (S8):** on a V100: picture encoding 9 ms; DDPM 100 steps 660 ms; DDIM 10 steps 66 ms; OneDP 7 ms (Table 5). Push-T
  coverage: DDPM 0.863, DDIM 10 steps 0.823, OneDP 0.816, and a diffusion policy run with only 1 step and no distillation
  0.000 (Table 1). No CPU numbers.

### 3.5 Is the "generative" part needed at all? Much Ado About Noising (ICLR 2026) [DG-35, DG-36]

- **Measured (S9, Sec. 3.1, Fig. 2):** on 28 behaviour-cloning benchmarks, regression policies given the **same networks and
  action chunks** as flow-based generative policies performed about equally on most tasks. Gaps appeared mainly on
  high-precision tasks such as Tool-Hang and Transport. Push-T, often called multimodal, showed no substantial gap. The
  authors' practical message: the network and action-chunking choices mattered more than generative versus regression.
- **Measured (Sec. 3.2, Table 1):** for a trained flow policy on Push-T, starting from zero noise, from random noise, or
  averaging 64 samples gave 0.97, 0.97 and 0.95. If Push-T needed distinct modes, averaging samples should have hurt more.
- **Measured (Sec. 4, Fig. 8, Table 4):** a two-step regression policy (their Minimal Iterative Policy) matched the flow
  policies with 2 network passes instead of 9.
- **Authors' hypothesis (speculation, Sec. 4 and 5.1):** the gains come from supervised iterative computation combined with
  some injected noise, which keeps actions plausible on unfamiliar states (manifold adherence, their term); they say the
  mechanism is still unknown.

---

## 4. What a diffusion policy costs on a CPU

### 4.1 Measured by others [DG-13, DG-29, DG-30, DG-33, DG-34]

| Source | Policy | Hardware | Time per call |
|---|---|---|---|
| VQ-BeT (S5), Table 7 | Diffusion Policy, transformer (steps not stated) | robot's onboard 4-core Intel CPU | **5,244 ms** |
| VQ-BeT (S5), Table 7 | VQ-BeT / plain BC | same CPU | 207 ms / 83 ms |
| LeRobot (S10), Table 3 | Diffusion Policy, 263 M, 10 denoising steps, fp32 | 2021 MacBook Pro M1, CPU only | **over 5,000 ms** (all 100 calls hit the 5 s time-out) |
| LeRobot (S10), Table 3 | ACT, 52 M | same CPU | 182 ms |
| LeRobot (S10), Table 3 | Diffusion Policy | M1 GPU (MPS) / RTX 4090 / A100 | 3,454 / 370 / 614 ms |
| Diffusion Policy (S1), Sec. 3.4 | DDIM 10 steps | RTX 3080 | about 100 ms |
| Consistency Policy (S7) | DDPM 100 / DDIM 15 steps | laptop RTX 3070-Ti | about 1,500 / 192 ms |
| OneDP (S8), Table 5 | DDPM 100 / DDIM 10 steps | V100 | 660 / 66 ms |

Two independent groups measured **about 5 s or more per decision on a CPU**. On GPUs the same kind of model ranges from about
10 ms (the DDIM baseline in Consistency Policy, on a P5000) to 1.5 s depending on steps and code, so the implementation
matters a great deal (judgement; for example LeRobot's 370 ms on an RTX 4090 with 10 steps is almost 4 times Chi's 100 ms on
an older RTX 3080).

### 4.2 My calculation (not measured; treat as a rough order of magnitude) [DG-39, DG-40, DG-41]

Inputs I used:

- From the released code (S2): U-Net channels 512, 1024, 2048, kernel size 5, two residual blocks per level, two in the
  middle, prediction horizon 16, 2 observation frames. Counting the convolutions gives about **1.2 billion multiply-adds per
  denoising step** for the 256 M network (my count of parameters from the same layer list comes to about 251 M, close to the
  256 M in Table 7, which suggests the layer list is right).
- torchvision lists ResNet-18 at 1.81 GFLOPS (S12; the page does not say the image size; I assume the usual 224x224). Scaled
  by pixel count: about 0.26 billion multiply-adds for one 84x84 crop, about 0.7 for 160x120.
- **Assumption, to be measured:** the 6-core laptop CPU sustains 50 to 200 GFLOP/s in PyTorch on these layers.

| Case | Work | Time at 50 to 200 GFLOP/s |
|---|---|---|
| Diffusion Policy, one decision, 10 DDIM steps | about 24 GFLOP | 0.12 to 0.5 s |
| Diffusion Policy, one decision, 100 DDPM steps | about 233 GFLOP | 1.2 to 4.7 s |
| ResNet-18 classifier, one 84x84 picture | about 0.5 GFLOP | 3 to 10 ms |
| Diffusion training, per picture (forward and backward, 2 frames) | about 10 GFLOP | |
| Classifier training, per picture (1 frame, 84x84) | about 1.6 GFLOP | |
| Diffusion, published recipe: 3000 epochs x 18,500 pictures | about 5.7 x 10^17 FLOP | 33 to 131 days |
| Diffusion, LeRobot recipe: 200 k steps x batch 64 | about 1.3 x 10^17 FLOP | 8 to 30 days |
| Diffusion, short run: 100 epochs | about 1.9 x 10^16 FLOP | 1 to 4.4 days |
| Classifier: 30 epochs | about 9 x 10^14 FLOP | 1.2 to 5 hours |

Loading and resizing the 640x480 JPEG pictures adds CPU time on top of this. The measured CPU inference times in 4.1 are 10 to
40 times slower than my estimate for 10 steps; I do not know why (possible reasons: more steps than stated, larger images,
code overheads, thread settings). **The honest summary: expect seconds per decision unless our own timing test shows
otherwise.**

A cheap test the team could run before deciding (judgement): build LeRobot's diffusion policy with random weights on the
laptop and time 50 calls at 10 steps. Timing does not need a trained model and takes minutes.

---

## 5. What this means for Team M (judgement, based on the evidence above)

### 5.1 Does our data have the kind of multimodality diffusion is good at? [DG-42]

- Diffusion (and IBC, BeT, ACT's CVAE) earn their keep when, for the same observation, demonstrations do clearly **different**
  things and the alternative is an **MSE regressor**, which averages them (S3 Table 4, S4 Table 1, S6 Sec. VI-B).
- Our move names are classes. A classifier trained with cross-entropy already represents "left or right" as two separate
  probabilities and picks one; it cannot average them into a third, wrong move. That is BeT's whole argument for binning.
- Where averaging could still hurt us: if the **exact step** (tip forward and left, twist, lean, turn) is regressed with MSE
  on its own. If two demonstrations at a similar picture used clearly different steps, the regressor would output their
  middle. A BeT-style head (predict the move name, then a separate exact-step correction **per move name**, training only the
  correction of the true class) avoids this at almost no cost.
- The places on our sheets where several actions are genuinely valid are likely few: the crossing of the loop (where one
  picture can show four line branches) and the gaps of the dashed loop. At the crossing the problem is more likely that
  one picture does not show where the puck came from (aliasing), which diffusion does not fix by itself; a short history of
  pictures, the previous move, or predicting a few moves at once would address it more directly.
- A quick check in our own data: at those places, look at the classifier's top two move names. If the top two are opposite
  directions and both likely, that is the multimodality diffusion handles; if they are neighbours (forward and forward-left),
  any averaging is harmless.

### 5.2 Is a diffusion policy realistic on the laptop CPU?

- **Training: no**, not with the published recipes (my estimate, days to weeks or more; Section 4.2). A cut-down version (the
  67 M real-world network size, one frame, small pictures, about 100 epochs) might fit in one or two days, but then it is no
  longer the recipe that produced the published results, and there is no evidence how well it works.
- **Running it: possible but slow.** Measured CPU numbers from others are about 5 s per decision; that would make each of our
  steps about 6 s instead of 1.2 s, roughly 5 times slower, and finishing time is the second exam criterion. Executing several
  predicted steps per picture would cut this, but at the price of moving 6 to 8 cm with no new picture.
- **With a GPU pool: realistic.** The published real Push-T models trained in 12 hours each, and with DDIM at 10 steps the
  decision time on a GPU is around 0.1 s.

### 5.3 When would a diffusion policy beat our classifier or a regressor? [DG-44]

From the evidence, mainly when all of these hold: (a) actions are continuous and the demonstrations disagree for the same
picture; (b) the alternative is an MSE regressor rather than a classifier; (c) the task needs high precision (Much Ado's gaps
were on high-precision tasks); (d) long action sequences need to stay consistent. Our task has coarse, low-dimensional steps,
already-discrete move names, and a stop-and-go loop; so I expect little or no gain. Nothing in these papers tests unseen
layouts of the kind our exam uses; Diffusion Policy's tests vary only start positions on the same task. Generalising to unseen
tracks will depend mostly on the image encoder, augmentation and the variety of sheets, not on the action head.

### 5.4 Cheap things worth borrowing from these papers

1. Diffusion Policy's encoder recipe for our classifier: ResNet-18 with **GroupNorm instead of BatchNorm**, **spatial
   softmax** instead of average pooling, **random crops** in training and a centre crop at test, an exponential moving average
   of the weights (S1, Sec. 3.2, App. A.3). Frozen pretrained encoders did worst in their tests.
2. **Predict a few steps at once** (action chunking) and either execute one and re-plan, or average overlapping predictions;
   Diffusion Policy and ACT both found chunking helps (S1 Sec. 4.3; S6 Fig. 7a, 7b).
3. **Two observation frames** instead of one (Diffusion Policy's default), if the crossing turns out to be ambiguous.
4. A **BeT-style per-class correction** for the exact step.
5. If a generative head is still wanted later: ACT's CVAE ran at 182 ms on a laptop CPU (S10) and trained in 5 hours on one
   old GPU (S6); this is far cheaper than diffusion.

### 5.5 If we get the GPU pool: a fair comparison

Train our best classifier and a diffusion policy with the **same** ResNet-18 encoder, the same pictures and crops, and the
same number of predicted steps; compare on held-out sheets (for example train on five sheets, test on the sixth) rather than
only on the training sheets. Much Ado's main lesson is that unfair comparisons (different networks or chunking) explain many
published gaps.

---

## 6. What I could not check

- The RSS 2023 PDF itself (my PDF reader failed); I used arXiv v5 and the RSS abstract page. Table and section numbers may
  differ in the RSS PDF.
- The IJRR article page (access refused), so volume, issue and pages of the journal version are not given here.
- The GPU used to train Diffusion Policy (not named in the text I read).
- Figure-only values: Diffusion Policy's data-efficiency figure (Fig. 15), BeT's sweep over k (Fig. 6), Much Ado's per-task
  bars (Fig. 2).
- The number of denoising steps behind VQ-BeT's CPU measurement (Table 7), and why LeRobot's measured times are so much
  slower than Chi's.
- Any CPU training measurement for a diffusion policy: I found none.
