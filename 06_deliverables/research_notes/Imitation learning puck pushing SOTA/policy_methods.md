# State-of-the-art imitation-learning policy methods for a small-data, single-camera, low-rate robot pushing task (as of October 2026)

Context carried through every judgement below: two beginner master's students, deadline 10 January 2027; Niryo Ned2 with blocking move commands (about one 1 cm tool step per second, no streaming control); wrist camera only (640x480 JPEG, about 9 fps, puck always at the bottom of the picture, about 9 cm wide window, about 7 cm of line visible ahead); task = push a 2.4 cm red puck along a 1 cm black line on an A4 sheet to the line's end; evaluation on an unseen sheet (six training sheets: straight, arc, S-curve with hairpin, meander, two self-crossing loops); planned action = relative tool step of about 1 cm (forward / sideways) plus a "finished" flag, one decision per picture; feasible data = 100 to 300 episodes of 25 to 40 steps (about 2,500 to 12,000 image-action pairs); compute = laptop CPU, possibly a university GPU; PyTorch, simple code. Professor's hard constraint: imitation learning only.

Wherever a statement is my own inference rather than a cited finding, it is placed under "Inferences". Older results (2016 to 2022) are marked as such.

---

## Key question 1: Plain behaviour cloning with a small CNN or a pretrained encoder (ResNet-18, R3M, VC-1, DINOv2): data needs, success rates, failure modes

### Takeaway
On the benchmarks that matter here, plain BC with a ResNet-18-style encoder, random-crop augmentation and about 200 demonstrations reaches 97 to 100 percent on easy pick-and-place tasks and 55 to 62 percent on harder ones; with only 20 percent of the data (about 40 demos) easy tasks still reach 83 to 100 percent while harder tasks collapse to about 30 percent. A well-augmented from-scratch encoder is competitive with frozen pretrained encoders (R3M, MVP, PVR); pretrained encoders help most when demos are very few (about 10) and when they are fine-tuned with augmentation, and no pretrained encoder is universally best.

### Cited Findings
- robomimic (2021, older) used 200 proficient-human (PH) demos per task (300 for multi-human, MH) and 200 real-robot trajectories per task. Image-based BC success (PH): Lift 100.0, Can 97.3, Square 62.0, Transport 55.3, Tool Hang 20.0; BC-RNN: 100.0, 98.0, 82.0, 72.0, 67.3 — [robomimic, arXiv 2108.03298](https://arxiv.org/abs/2108.03298)
- robomimic dataset-size ablation (image, PH, success percent at 20 / 50 / 100 percent of the 200 demos): Lift 100 / 100 / 100; Can 83.3 / 97.3 / 98.0; Square 29.3 / 64.7 / 82.0; Transport 30.7 / 60.1 / 72.0. MH image: Can 77.3 / 87.3 / 96.0; Square 27.3 / 50.7 / 76.7 — [robomimic appendix](https://arxiv.org/abs/2108.03298)
- robomimic image pipeline: 84x84 inputs in simulation (120x120 real robot), random crop ("pixel shift") to 76x76 (108x108 real), ResNet-18 encoder with SpatialSoftmax (64 keypoints), learning rate 1e-4, GMM head with 5 modes; real raw 640x480 frames centre-cropped to 480x480 then resized — [robomimic appendix](https://arxiv.org/abs/2108.03298)
- robomimic hyperparameter sensitivity: learning rate 1e-3 instead of 1e-4 dropped image agents by 35 to 63 percent; a shallow conv encoder instead of ResNet dropped performance 25 to 62 percent; larger MLPs reduced performance across the board (overfitting); replacing the GMM head with a deterministic head caused large drops on MH data (low-dim Transport minus 58 percent). Recommendation: "a GMM policy and a ResNet encoder" — [robomimic](https://arxiv.org/abs/2108.03298)
- robomimic policy selection: choosing the checkpoint by lowest validation loss or by the final checkpoint gave policies 10 to 100 percent worse than the best checkpoint, so offline validation loss is an unreliable selector — [robomimic](https://arxiv.org/abs/2108.03298)
- robomimic lesson L6: simulation findings transferred to the real robot without real-world tuning — [robomimic](https://arxiv.org/abs/2108.03298)
- Hansen et al. (ICML 2023): a shallow 5-layer stride-2 ConvNet trained from scratch with random-shift augmentation ("LfS +aug") is competitive with frozen R3M, MVP and PVR on BC. BC with 100 sim demos: Adroit Pen 22.0 (no aug) to 74.8 (+aug) vs 81.3 best frozen pretrained; Relocate 16.9 to 51.4 vs 47.5. Real robot with 10 (reach) or 20 (pick) human demos: Reach 80 to 85 vs 90; Pick 25 to 55 vs 35 — [Hansen et al., arXiv 2212.05749](https://arxiv.org/abs/2212.05749)
- Hansen et al. data-efficiency sweep (10, 25, 100 demos, DMControl BC): 100 demos generally favour from-scratch; frozen pretrained representations do "marginally better" at 10 demos; performance drops quickly as demos decrease, read by the authors as a policy-learning bottleneck rather than a representation bottleneck — [Hansen et al.](https://arxiv.org/abs/2212.05749)
- Hansen et al. fine-tuning: fine-tuning R3M with augmentation reached 80.5 vs 64.0 frozen vs 63.1 LfS +aug on Adroit; fine-tuning without augmentation did not help; MVP (ViT) benefited least — [Hansen et al.](https://arxiv.org/abs/2212.05749)
- R3M (2022, older): frozen R3M plus 2-layer MLP BC with 5, 10 or 25 demos per task reaches about 62 percent average over 12 sim tasks, more than 20 points above from-scratch and more than 10 above prior representations; real Franka with 20 demos per task: R3M 56 percent average vs CLIP 24 percent (e.g., "pushing mug to goal" 70 vs 40); images resized/cropped to 224x224 — [R3M, arXiv 2203.12601](https://arxiv.org/abs/2203.12601)
- VC-1 (2023): across 17 CortexBench tasks "none [of the pretrained visual representations] are universally dominant"; VC-1 "outperforms all prior PVRs on average but does not universally dominate either"; "task- or domain-specific adaptation of VC-1 leads to substantial gains" — [VC-1, arXiv 2303.18240](https://arxiv.org/abs/2303.18240)
- Burns et al. (2023): among 15 pretrained vision models, "visual representations designed for manipulation and control tasks do not necessarily generalize" under lighting, texture and distractor shifts; emergent segmentation ability of ViTs predicts out-of-distribution generalisation; real ALOHA check used 50 demos — [Burns et al., arXiv 2312.12444](https://arxiv.org/abs/2312.12444)
- DINOv3-Diffusion Policy (2025): on Push-T / Lift / Can / Square with 84x84 inputs cropped to 76x76, frozen DINOv3 ViT-S/16 scored 0.39 / 1.00 / 0.50 / 0.60 vs ResNet-18 from scratch 0.78 / 1.00 / 0.90 / 1.00; fine-tuned DINOv3 0.84 / 1.00 / 0.90 / 0.90 vs fine-tuned ResNet-18 0.89 / 1.00 / 0.80 / 1.00; the paper's sample-efficiency claim is not backed by a low-data experiment — [arXiv 2509.17684](https://arxiv.org/abs/2509.17684)
- VINN (2021, older) on a real cabinet door with 71 demos: end-to-end BC 0 percent, BC on frozen BYOL representations 53.3 percent, VINN nearest-neighbour 80 percent — [VINN, arXiv 2112.01511](https://arxiv.org/abs/2112.01511)
- LeRobot practical guidance: "We suggest recording at least 50 episodes, with 10 episodes per location. Keep the cameras fixed and maintain consistent grasping behavior... A good rule of thumb is you should be able to do the task yourself by only looking at the camera images." and "Avoid adding too much variation too quickly" — [LeRobot il_robots tutorial](https://huggingface.co/docs/lerobot/en/il_robots)
- Classic single-camera, discrete-direction CNN (2016, older): Giusti et al. trained a deep network as a supervised classifier on one monocular image to estimate the trail direction relative to the heading, reaching accuracy comparable to humans on the same task, and steered a quadrotor on trails not seen in training — [Giusti et al., IEEE RA-L 2016 (EPFL record)](https://infoscience.epfl.ch/record/218621); [project page](https://people.idsia.ch/~giusti/forest/web/)
- End-to-end CNN imitation for robot soccer (2018, older) notes that a plain CNN "can only map the current image to an action" and that performance drops when demonstrations cover a limited part of the state space — [arXiv 1807.09205](https://arxiv.org/abs/1807.09205)

### Inferences
- The team's dataset (about 2,500 to 12,000 labelled images) is in the same order as robomimic's 20 to 100 percent subsets (40 to 200 episodes of about 50 to 200 steps), where easy, visually unambiguous tasks already saturate. Following a clearly visible high-contrast line with the puck fixed at the bottom of the frame is closer to "Lift/Can" difficulty than "Square/Transport", so plain BC with a small CNN plus random-crop augmentation is a realistic baseline, not a fallback.
- The robomimic hyperparameter lessons translate directly: learning rate about 1e-4, a ResNet-18-sized (or smaller, see Hansen's 5-layer ConvNet) encoder, no oversized MLP head, and random crop always on. The warning about checkpoint selection means the team must evaluate checkpoints on the robot (or on held-out episodes with a task-level metric), not just pick the lowest validation loss.
- A pretrained ResNet-18 (ImageNet weights, as LeRobot's ACT uses) fine-tuned with augmentation is the lowest-risk encoder choice on a CPU laptop; frozen DINOv2/DINOv3 features gave the worst results in the only controlled comparison found and need a ViT forward pass per image. R3M/VC-1 are worth trying only if demos stay near 10 to 20 per path type, which is below the planned budget.
- Failure modes to expect from the literature: (a) compounding drift once the puck leaves the demonstrated distribution (no recovery data), (b) brittleness to lighting/background changes if augmentation is weak, (c) overfitting with a large head on a few thousand images, (d) copycat/inertia if past actions are fed in (see Key question 5).

### Gaps
- No paper was found that trains BC on exactly a wrist-only camera with a quasi-static 1-step-per-second robot; all numbers above are from 10 to 50 Hz teleoperation data.
- I could not verify the exact figures of Giusti et al. (number of classes, number of frames, human accuracy) from the fetched pages; only the qualitative claims above are sourced.
- No controlled DINOv2 (as opposed to DINOv3) vs ResNet-18 BC comparison at 50 to 300 demos was found.

---

## Key question 2: ACT and Diffusion Policy — minimum demonstrations, compute, inference, chunking and temporal ensembling, and sense at one action per second

### Takeaway
ACT is reported to work from 50 demonstrations (about 5 hours of training on an 11 GB RTX 2080 Ti, about 10 ms inference), and Diffusion Policy from 136 to 200 demos (12 hours of training in its real Push-T experiment, about 0.1 s inference on an RTX 3080 with DDIM); both were designed for 10 to 50 Hz control and their chunking/ensembling benefits are tied to high-rate, temporally correlated actions. A 2026 analysis finds the "temporal consistency" benefit of chunking matters only at high frequencies, so at one 1 cm step per second chunking is optional and the heavy machinery of ACT and DP is not what makes the difference.

### Cited Findings
- ACT: 50 demos per real task (100 for Thread Velcro); 50 Hz; chunk size k = 100; four 480x640 cameras through ResNet-18; about 80 M parameters trained from scratch per task; about 5 hours on one 11 GB RTX 2080 Ti; inference about 0.01 s — [ACT, arXiv 2304.13705](https://arxiv.org/abs/2304.13705)
- ACT ablations: without temporal ensembling, success is about 1 percent at k = 1 and about 44 percent at k = 100 (averaged over 4 settings); temporal ensembling adds 3.3 points for ACT and 4 for BC-ConvMLP and hurts VINN; removing the CVAE made almost no difference on scripted data but dropped human-data success from 35.3 to 2 percent — [ACT](https://arxiv.org/abs/2304.13705)
- ACT vs baselines (sim cube transfer, scripted | human demos): BC-ConvMLP 1 | 0, BeT 27 | 1, RT-1 2 | 0, VINN 3 | 0, ACT 86 | 50 — [ACT](https://arxiv.org/abs/2304.13705)
- LeRobot on ACT: "first model we recommend when you're starting out", "Trains in a few hours on a single GPU", "Only ~80M parameters", "Often achieves high success rates with just 50 demonstrations"; "Expect a few hours for 100k training steps on a single GPU"; start with batch size 8 — [LeRobot ACT docs](https://huggingface.co/docs/lerobot/act)
- LeRobot Colab notebook (seen via a mirror of docs/source/notebooks.mdx): about 1.5 hours for ACT training on an A100 at batch size 64 — [notebooks.mdx mirror](https://gitea.d-robotics.cc/chao.wu/Lerobot/raw/branch/GH/dev/docs/source/notebooks.mdx); a mirrored FAQ gives 2 to 4 hours for 100k ACT steps on Push-T on one unnamed GPU — [FAQ mirror (secondary)](https://www.mintlify.com/huggingface/lerobot/resources/faq)
- Diffusion Policy demo counts: robomimic tasks 200 PH / 300 MH; sim Push-T 200; real Push-T 136; real sauce pouring and spreading 50 each (90 percent used for training) — [Diffusion Policy, arXiv 2303.04137](https://arxiv.org/abs/2303.04137)
- Diffusion Policy horizons and encoder: observation horizon To = 2, action execution horizon Ta = 8, prediction horizon Tp = 16; Ta = 8 "found optimal for most tested tasks"; vision-based DP "performs best with a low [observation] horizon above 1, and 2 is a good compromise"; ResNet-18 from scratch with GroupNorm and spatial softmax; 84x84 inputs cropped to 76x76 in sim, 320x240 real; learning rate 1e-4 — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- Diffusion Policy compute: real Push-T models "trained for a fixed 12 hours"; 100 DDPM steps in training; DDIM with 16 (appendix) or 10 (Sec. 3.4, about 0.1 s latency on an RTX 3080) inference steps; policy at 10 Hz interpolated to 125 Hz; keeps peak performance with up to 4 steps of latency — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- Diffusion Policy results: real Push-T 95 percent vs 0 percent best IBC and 20 percent best LSTM-GMM; sim visual Push-T DP-C 0.91 vs IBC 0.75 vs LSTM-GMM 0.69; average 46.9 percent improvement in simulation; no dedicated low-data experiment — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- Diffusion Policy on position vs velocity: DP is consistently better with position control, baselines (BC-RNN, BeT) generally better with velocity control; velocity control is more affected by latency — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- VQ-BeT measured inference cost: real robot, VQ-BeT 18 ms GPU / 207 ms CPU vs DP-T 573 ms GPU / 5,244 ms CPU — [VQ-BeT, arXiv 2403.03181](https://arxiv.org/abs/2403.03181)
- Why chunking helps (2026): with Libero-90 (50 demos per task, 20 Hz) a Markovian diffusion policy got 68.9 percent vs 89.2 with chunking (execute 10 of 20) vs 94.0 with a "delayed" (history-conditioned, single-action) policy; robomimic Can 83.7 / 97.2 / 93.5, Square 69.0 / 85.4 / 80.8, Tool Hang 28.0 / 75.2 / 51.6. "the temporal consistency of action chunking does matter, but only at high frequencies" (50 to 60 Hz, where delayed policies failed); the gain is attributed to non-Markovian expressivity, reduced compounding error from conditioning on past observations, and implicit ensembling; the authors conjecture a link to human operating frequency of about 2 to 10 Hz — [arXiv 2608.02547](https://arxiv.org/abs/2608.02547)
- Bidirectional Decoding (2024): "there is no universally optimal choice of action horizon across all conditions"; in a 1-D diagnostic the best chunk length was 10 with no noise, 5 with moderate noise and 1 with high noise; temporal-ensembling (EMA) "best decay rate varies by task" and at high noise EMA hurt VQ-BeT (18.4 vs 29.5) — [arXiv 2408.17355](https://arxiv.org/abs/2408.17355)
- Data scaling laws (2024): the largest UMI Diffusion-Policy models (DINOv2 ViT-L encoder, over 396 M parameters) trained 75 hours on 8 A800 GPUs — [arXiv 2410.18647](https://arxiv.org/abs/2410.18647)
- Frequency sub-sampling evidence (tactile cable tracing, 2026): a policy trained at 60 Hz reached 93 percent SR@10 offline; re-trained at 30 Hz it fell to 23 percent and at 15 Hz to 0 percent (12 demos, offline metric) — [Touch2Trace, arXiv 2609.15921](https://arxiv.org/abs/2609.15921)

### Inferences
- At one decision per second with 1 cm steps, an ACT chunk of k = 100 would mean 100 s of open-loop motion; the equivalent wall-clock horizon of DP's Ta = 8 at 10 Hz is 0.8 s, i.e. one step here. The published chunking benefits (temporal smoothness, jitter removal, shortened effective horizon of 50 Hz sequences) are therefore mostly irrelevant for this robot; a chunk of 2 to 3 steps could at most cut camera/inference latency per cm but would push the puck 2 to 3 cm blind while only about 7 cm of line is visible, which is risky at the hairpin.
- The 2026 finding that chunking's remaining benefit comes from conditioning on past observations and implicit ensembling suggests the cheap substitute for this team: a single-step policy that optionally sees the previous frame(s), or an ensemble of a few small CNNs, rather than ACT.
- ACT is still feasible if the university GPU is available (one 640x480 camera, so training time would be well below the paper's 5 hours; LeRobot handles the code), but its CVAE, 80 M parameters and chunking are complexity the team does not need to explain to the professor. Diffusion Policy on a laptop CPU is slow to train (12 hours on a GPU in the real Push-T experiment) and slow at inference (seconds on CPU per VQ-BeT's measurement), though at one action per second a few seconds of inference could still be tolerated.
- The Touch2Trace frequency drop is for a dynamic, contact-rich task where demos were recorded at 60 Hz and sub-sampled; it does not transfer to a quasi-static push where the demonstrations themselves are collected one step at a time at the deployment rate. The relevant risk at 1 Hz is not dynamics but out-of-distribution puck drift between decisions.

### Gaps
- No published result trains ACT or Diffusion Policy at about 1 Hz with step-wise (blocking) demonstrations; all claims about their behaviour at that rate are inference.
- Minimum-demonstration curves for ACT below 50 demos were not found in the primary paper; the LeRobot "50 demonstrations" claim is a documentation statement, not a measured curve.
- Diffusion Policy training time per sim task on a given GPU is not stated in the paper text fetched (only the 12-hour real Push-T budget).

---

## Key question 3: Implicit BC, VINN, BeT / VQ-BeT and simpler alternatives — when they beat plain BC on small data

### Takeaway
These methods beat explicit MSE-BC mainly when the demonstration data is multimodal or discontinuous (several valid actions for one image) or when demos are very few (VINN under about 20 demos); on unimodal, visually unambiguous tasks explicit BC with a mixture or classification head matches them, and each adds cost (IBC: about 5x slower training and unstable; BeT/VQ-BeT: discretisation machinery; VINN: nearest-neighbour search over all training frames).

### Cited Findings
- Implicit BC (2021, older): energy-based implicit models "generally [perform] better, on average, than commonly used explicit models" (MSE or MDN); real robot (3 seeds x 20 rollouts): Push-Red-then-Green (95 demos) EBM 85.0 vs MSE 35.0; Push-Red/Green-Multimodal (410 demos) 88.3 vs 55.0; Insert-Blue (223) 83.3 vs 6.7; Sort (502) 48.3 vs 19.6; planar sweeping with 50 human demos 82.6 vs 75.6; the 10-example coordinate-regression toy shows 1 to 2 orders of magnitude lower test MSE for the EBM — [IBC, arXiv 2109.00137](https://arxiv.org/abs/2109.00137)
- IBC costs and weaknesses: D4RL training 17.9 steps/s vs 101.3 for MSE (3.4 h vs 0.66 h); real-robot inference 7.22 ms vs 3.49 ms; Langevin training needs gradient penalties and spectral normalisation; derivative-free inference fails beyond about 5 action dimensions; not uniformly better (hammer-human worse) — [IBC](https://arxiv.org/abs/2109.00137)
- Later benchmarks place IBC below newer methods: Diffusion Policy real Push-T 95 vs IBC 0 percent; VQ-BeT reports BeT block-push training under an hour vs about 14 hours for IBC — [Diffusion Policy](https://arxiv.org/abs/2303.04137); [BeT, arXiv 2206.11251](https://arxiv.org/abs/2206.11251)
- VINN (2021, older): ImageNet ResNet-50 fine-tuned with BYOL on demo frames (3.5 h on an RTX 8000), then k-NN (k = 10 best) over demonstration embeddings; real door opening with 71 demos: VINN 80 percent vs end-to-end BC 0 vs BC-on-representations 53.3; "significantly better on stacking and door opening with fewer than 20 demonstrations", while end-to-end BC can match or beat VINN with large datasets; pushing "was hard for all methods with few demos"; inference about 0.074 s encode + 0.038 s k-NN — [VINN](https://arxiv.org/abs/2112.01511)
- VINN generalisation: with signs and handle covered VINN kept 70 percent vs BC-rep 10; with all visual markers covered both 0 — [VINN](https://arxiv.org/abs/2112.01511)
- BeT (2022, older): k-means action bins plus residual offset on a minGPT trunk (10^4 to 10^6 parameters); datasets 100 (CARLA) to 1,000 (block push) demos; block push "push 2" BeT 0.71 vs IBC 0.00; Kitchen 4-of-4 0.44 vs IBC 0.24; removing binning drops block push to 0.25; replacing the transformer with an MLP drops block push to 0.00; no dedicated small-data experiment; context lengths 2 to 10, k = 24 to 64 clusters — [BeT](https://arxiv.org/abs/2206.11251)
- VQ-BeT (2024): residual VQ with 2 codebook layers, GPT backbone (6 layers, 120 dims), focal loss plus L1 offset; PushT 206 human demos; image PushT 0.68 vs DP-C 0.66 vs BeT 0.01; real robot with 45 demos per task: single-phase 47/50 vs DP-T 45/50 vs BC 29/50; two-phase 19/30 vs 11/30 vs 7/30; no reduced-dataset experiment; advantage smallest on the largest dataset (block push, 1,000 trajectories) — [VQ-BeT](https://arxiv.org/abs/2403.03181)
- IMLE Policy (2025): claims it "requires 38% less data on average to match the performance of baseline methods" and single-step inference "97.3% [faster] compared to Diffusion Policy"; abstract gives no benchmark names or robot numbers — [arXiv 2502.12371](https://arxiv.org/abs/2502.12371)
- robomimic: BC-RNN (an LSTM over 10 frames with a GMM head) was the strongest simple method and is recommended with "a GMM policy and a ResNet encoder" — [robomimic](https://arxiv.org/abs/2108.03298)
- ACT paper's view of the older baselines on its tasks: BeT, VINN and BC-ConvMLP all at 0 to 3 percent on human-demo sim tasks vs ACT 20 to 50 — [ACT](https://arxiv.org/abs/2304.13705)

### Inferences
- A line-following push has little multimodality: given the picture, there is normally one correct 1 cm step. The exceptions are the self-crossing loops (two lines leave the crossing) and the moment the hairpin enters the frame. This is the setting where IBC/BeT/VQ-BeT give the least over a classification head, which already represents multiple modes as a softmax over "forward / left / right / stop".
- VINN is the one "beats BC" candidate the team could actually run: it is a few dozen lines on top of a pretrained encoder, needs no policy training, and shines below about 20 demos. Its weakness here is that nearest neighbours in embedding space are defined by appearance (which sheet, lighting), so on an unseen sheet it may retrieve the wrong path type. It is a good diagnostic baseline, not the final method.
- IBC should be ruled out for this team: unstable training, several-times slower, and later papers show it losing to simpler methods.

### Gaps
- No small-data (under 100 demos) controlled comparison of BC vs BeT vs VQ-BeT vs IBC on a single-camera task was found; the papers' own low-data claims are qualitative (VINN) or absent (BeT, VQ-BeT).
- IMLE Policy's real-robot numbers were not retrieved (abstract only).

---

## Key question 4: Action representation — discrete classes vs continuous regression; relative vs absolute; chunking at low rate; multimodality

### Takeaway
Theory published in 2025 to 2026 supports the team's plan: smooth deterministic continuous-action imitators provably suffer exponentially compounding error, whereas BC over quantised (discrete) actions with log-loss attains optimal sample complexity with only polynomial horizon dependence; classic single-camera direction classifiers (trail following) worked with a handful of classes. Relative (delta) actions are the natural choice when the only sensor is a wrist camera; continuous regression, if used at all, should have a mixture head.

### Cited Findings
- Simchowitz et al. (COLT 2025): "any smooth, deterministic imitator policy necessarily suffers error on execution that is exponentially larger" than its training error, even with stable dynamics and a smooth expert; escaping this needs "improper" policies, "those which are non-smooth, non-Markovian, or which exhibit highly state-dependent stochasticity"; the authors say this helps explain why action chunking and diffusion policies work — [arXiv 2503.09722](https://arxiv.org/abs/2503.09722)
- Cao and Xie (ICML 2026): BC with quantised actions and log-loss "achieves optimal sample complexity, matching existing lower bounds", with polynomial horizon dependence on quantisation error "provided the dynamics are stable and the policy satisfies a probabilistic smoothness condition"; they characterise which quantisation schemes satisfy this — [arXiv 2603.20538](https://arxiv.org/abs/2603.20538); [ICML 2026 proceedings](https://proceedings.mlr.press/v306/cao26x.html)
- Discretisation in practice: BeT's k-means bins plus offset (removing bins dropped block push from 1.0-scale to 0.25); VQ-BeT's hierarchical VQ tokens with a single forward pass, 18 ms GPU / 207 ms CPU — [BeT](https://arxiv.org/abs/2206.11251); [VQ-BeT](https://arxiv.org/abs/2403.03181)
- robomimic: a deterministic (MSE) head versus a 5-mode GMM head lost heavily on multi-human data (low-dim Transport minus 58 percent) — [robomimic](https://arxiv.org/abs/2108.03298)
- Diffusion Policy: position (absolute target) control was consistently better for DP, velocity control better for BC-RNN and BeT; velocity control is more latency-sensitive; sequence prediction avoids "jitter from per-step mode switching"; baselines "often get stuck" when idle actions remain in training data — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- Chunking at different rates: temporal consistency matters "only at high frequencies"; at 50 to 60 Hz sub-chunks of 5 (effective 10 to 20 Hz) restored performance; real-world setup used delta joint position control at 15 Hz with 50 demos per task — [arXiv 2608.02547](https://arxiv.org/abs/2608.02547)
- Chunk length vs noise: best horizon 10 (no noise), 5 (moderate), 1 (high noise) in the diagnostic; EMA temporal ensembling can hurt under noise — [arXiv 2408.17355](https://arxiv.org/abs/2408.17355)
- Multimodality handling by method: DP samples both approach sides on Push-T "with similar likelihood"; LSTM-GMM and IBC biased to one side; BeT "fails to commit to one mode" — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- ACT's CVAE exists to absorb the stochasticity of human demos (35.3 to 2 percent without it on human data, no effect on scripted data) — [ACT](https://arxiv.org/abs/2304.13705)
- Single-camera direction classification (2016, older): Giusti et al. estimate trail direction relative to heading with an image classifier at human-level accuracy and flew on unseen trails — [Giusti et al.](https://infoscience.epfl.ch/record/218621)
- PilotNet (2016, older): a CNN mapping "raw pixels from a single front-facing camera directly to steering commands" learned to drive "on local roads with or without lane markings" and "in areas with unclear visual guidance such as in parking lots and on unpaved roads" — [arXiv 1604.07316](https://arxiv.org/abs/1604.07316)

### Inferences
- The planned "forward / left / right / finished" classification is exactly the quantised-action BC that Cao and Xie analyse, and the demonstrations are produced one discrete step at a time, so there is no quantisation error at all; a cross-entropy head also represents the two-way ambiguity at a crossing as two probable classes instead of averaging them into a diagonal step, which is the failure mode of an MSE head. Keep a diagonal class (forward-left, forward-right) if the demonstrators actually use such steps, because the puck will otherwise zig-zag on curves.
- Relative steps are forced by the sensor: the absolute sheet position is not observable from a wrist camera that always shows the puck at the bottom, so an absolute-target head would have to memorise sheet layouts. DP's "position control is better" finding is about target-pose vs velocity on a fast arm with global cameras and does not transfer.
- Multimodality in the training set will come mainly from inconsistent demonstrators (LeRobot's advice "maintain consistent ... behavior") and from the crossings. A discrete head plus consistent demonstration rules (e.g., "always keep the line centred under the puck; at a crossing continue straight") removes most of it without any generative model.
- A "finished" flag is an extra binary output; the DP idle-action warning suggests making sure the stop label is tied to a visually distinct cue (line end plus a printed end marker) rather than to the demonstrator pausing.

### Gaps
- No empirical paper was found that compares a 3- to 5-class discrete head against continuous regression on the same small visuomotor dataset; the support is theoretical (2025 to 2026) plus the old trail-following and driving results.
- Nothing was found on action chunking at about 1 Hz specifically.

---

## Key question 5: Image input — resolution, cropping, augmentation, frame stacking or history

### Takeaway
The strongest and most repeatedly measured single design choice is random-crop ("pixel shift") augmentation: removing it cost 35 to 47 percent relative success in robomimic and it more than tripled from-scratch BC success in Hansen et al.; resolutions of 84x84 to 120x120 (random crop to about 90 percent) are standard for BC and Diffusion Policy. History helps BC-RNN on long-horizon tasks but a long observation history can hurt vision-based policies (DP's best To is 2) and invites the copycat shortcut.

### Cited Findings
- robomimic: removing pixel-shift randomisation caused relative drops of 47 percent (Square) and 35 percent (Transport); removing the wrist camera 9 and 43 percent; real Can: 26.7 percent without randomisation and 43.3 without wrist camera vs a proficient policy with both; resolutions 84x84 sim (crop 76x76), 120x120 real (crop 108x108); ResNet-18 plus SpatialSoftmax; adding end-effector velocities hurt image agents 2 to 29 percent — [robomimic](https://arxiv.org/abs/2108.03298)
- Hansen et al.: random shift (pad 12 on 224x224 centre crops of 256x256) raised from-scratch BC from 22.0 to 74.8 (Adroit Pen), 16.9 to 51.4 (Relocate), 261 to 657 (Reacher Easy), real pick 25 to 55 percent; fine-tuning pretrained encoders only helped with augmentation; the paper also compares random shift, colour jitter and random overlay and notes a domain gap remains with augmentation on the DMControl generalisation benchmark — [Hansen et al.](https://arxiv.org/abs/2212.05749)
- Diffusion Policy: 84x84 cropped to 76x76 in sim, 320x240 real at 10 fps; To = 2 is the best observation horizon for vision ("best with a low horizon above 1"); CNN-variant performance decreases as the observation horizon increases — [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- ACT uses full 480x640 frames with ResNet-18 to 15x20x512 feature maps and no observation history — [ACT](https://arxiv.org/abs/2304.13705)
- RoboSaGA (2026): random crop and colour jitter (0.2 on brightness, contrast, saturation, hue) are the usual BC baselines, "random crop is a strong default", but they "fall short under large visual shifts such as shadows, distractors, and backgrounds"; proposes saliency-guided superimposition — [arXiv 2608.11870](https://arxiv.org/abs/2608.11870)
- Wang (2025): "applying any form of randomization improved policy generalization", and diverse trajectories were especially useful for closing the visual gap — [arXiv 2511.09932](https://arxiv.org/abs/2511.09932)
- ReShoot (2026): generative re-rendering of recorded demos under varied appearance restored 42.9 and 47.5 percent success under colour shifts on two platforms without new demos — [arXiv 2609.19661](https://arxiv.org/abs/2609.19661)
- Burns et al.: robustness to lighting, texture and distractors is not guaranteed by manipulation-oriented pretrained encoders — [arXiv 2312.12444](https://arxiv.org/abs/2312.12444)
- robomimic history: BC-RNN (10-frame LSTM) beat BC everywhere, e.g. image Square 82.0 vs 62.0, Tool Hang 67.3 vs 20.0; "particularly effective in learning from human datasets" — [robomimic](https://arxiv.org/abs/2108.03298)
- Copycat problem (NeurIPS 2020, older): in partially observed settings with autocorrelated expert actions, BC from observation histories can do worse than BC from the latest observation because the imitator "learns to cheat by predicting the expert's previous action, rather than the next action"; the 2022 follow-up calls it the inertia problem; gains from fixes were smaller on images than on states — [Wen et al., arXiv 2010.14876](https://arxiv.org/abs/2010.14876); [arXiv 2207.09705](https://arxiv.org/abs/2207.09705)
- Why chunking helps (2026): history-conditioned ("delayed") single-action policies matched or beat chunking at 20 Hz (Libero-90 94.0 vs 89.2) but "may suffer from higher sample complexity in low-data regimes" — [arXiv 2608.02547](https://arxiv.org/abs/2608.02547)
- Bidirectional Decoding: extending context length "helps at first, then hurts" — [arXiv 2408.17355](https://arxiv.org/abs/2408.17355)

### Inferences
- A 640x480 wrist frame downscaled to about 160x120 (then random-cropped to about 144x108) keeps the 1 cm line at roughly 2 to 3 pixels wide at the top of the frame given the 9 cm field of view, which is enough for a CNN while keeping CPU training fast; going below about 96x72 risks losing the line in the far field. This is my estimate from the stated field of view, not a measured result.
- Cropping is doubly useful here: a fixed crop can remove the pusher rod and anything outside the sheet, and random crops of about 90 percent provide the shift augmentation that every study above found essential. Colour jitter at about 0.2 is the standard second augmentation and matters because the evaluation sheet may be printed or lit differently; horizontal flips are usable only if the left/right labels are swapped accordingly.
- Start with a single frame. The next step on a line is almost fully determined by the current picture, the DP evidence favours To of 1 to 2 for vision, and the copycat/inertia shortcut is a real risk when a human pushes forward 20 times in a row. If history is added later (for the crossings or to detect the end), use 2 frames or a BC-RNN-style GRU over a short window, never the previous action as an input.

### Gaps
- No study was found that measures resolution (e.g., 160x120 vs 224x224) vs success for a wrist-only policy; the resolutions above are what papers used, not ablations.
- No measured comparison of frame stacking vs single frame for a line-following or pushing task was found.

---

## Key question 6: Generalisation to unseen layouts (new paths) from the same method families

### Takeaway
The best evidence says generalisation scales with the diversity of training situations, not with more demos of the same situation: Diffusion Policy with UMI data plateaued at about 50 demos per environment-object pair and reached about 90 percent in unseen environments once about 32 diverse pairs were covered. Visual-imitation methods fail when the visual cue they rely on is removed, and single-camera direction classifiers (trails, roads) have generalised to unseen routes when training covered varied scenes.

### Cited Findings
- Data scaling laws (2024): generalisation follows "roughly a power law" in the number of environments, objects or environment-object pairs; demonstrations per pair plateau (plateaus at about 400, 800, 1,600 demos for 8, 16, 32 pairs); recommendation of about 50 demos per environment-object pair, 32 pairs (about 1,600 demos) gave 85 to 92.5 percent success in 8 unseen environments with unseen objects for four tasks; "more demonstrations per environment added little once diversity was high"; environment generalisation is harder than object generalisation — [arXiv 2410.18647](https://arxiv.org/abs/2410.18647)
- VINN on modified cabinets: covering signs and handle dropped BC-on-representations from 90 to 10 percent while VINN kept 70; covering all visual markers dropped both to 0 — [VINN](https://arxiv.org/abs/2112.01511)
- Burns et al.: out-of-distribution robustness under lighting/texture/distractor shifts is predicted by emergent segmentation of ViT features, not by ImageNet accuracy — [arXiv 2312.12444](https://arxiv.org/abs/2312.12444)
- Trail following (2016, older): a monocular direction classifier steered a quadrotor on trails not seen in training — [Giusti et al.](https://infoscience.epfl.ch/record/218621)
- PilotNet (2016, older): generalised to roads "with or without lane markings" and unpaved roads from a single front camera — [arXiv 1604.07316](https://arxiv.org/abs/1604.07316)
- A 2023 UT Austin course presentation on an imitation-learning line-following project lists "How is the generalization ability to more unseen tracks?" as open future work — [CS391R slides](https://cs.utexas.edu/~yukez/cs391r_fall2023/slides/pre_10-03_Shaoheng.pdf)
- Robot soccer end-to-end IL (2018, older): "if the teacher's demonstrations cover very limited part of the state space or the learning model is not able to generalize well, the performance of the imitation learning decreases" — [arXiv 1807.09205](https://arxiv.org/abs/1807.09205)
- robomimic: "Larger human datasets show promise for complex tasks" (L5), and simulation findings transferred to real tasks (L6) — [robomimic](https://arxiv.org/abs/2108.03298)
- LeRobot supports a `dagger` rollout strategy for "Human-in-the-loop data collection" — [LeRobot il_robots tutorial](https://huggingface.co/docs/lerobot/en/il_robots)
- Wang (2025): diverse trajectories were especially useful for closing the visual gap — [arXiv 2511.09932](https://arxiv.org/abs/2511.09932)

### Inferences
- The unseen evaluation sheet is an "unseen environment" in the data-scaling-laws sense. Translating their rule of thumb: the "environment" is not the whole sheet but the local line geometry in the 9x7 cm window (straight, gentle curve, tight curve, hairpin, crossing, end). Six sheets already contain all of these; the team should record episodes starting at many points along each sheet, in both directions, with the puck deliberately placed off-centre, so that each local geometry appears in dozens of frames from varied positions. Spreading 100 to 300 episodes across six sheets like this is a better use of the budget than 50 identical runs per sheet.
- Because the wrist camera sees only the local window, a new sheet is "unseen" only in the sequence of local geometries, which is favourable: a per-frame classifier that handles each local geometry should chain them. The risk is the far-field cue at crossings and hairpins, where a new sheet may combine curvatures not seen before.
- Recovery data is the standard remedy for compounding drift in BC (the ACT paper's motivation). With a blocking step interface the team can implement DAgger-style corrections cheaply: run the policy, let the puck drift, let the human label the corrective step, append to the dataset, retrain. This stays within "imitation learning" because the labels are human.

### Gaps
- No paper was found that measures generalisation of a visual imitation policy to new line or track shapes with a held-out track; the closest evidence is trail/road following (2016) and the UMI scaling study on scenes and objects.
- No quantitative study of how many distinct path shapes are needed for layout generalisation.

---

## Key question 7: 2024 to 2026 surveys and practical guides on small-data visuomotor imitation learning

### Takeaway
No 2024 to 2026 survey specifically about small-data visuomotor imitation learning was found; the closest are a 2025 systematic review of imitation learning for manipulation, a 2025 survey of diffusion models for manipulation, a 2024 general survey of IL methods, and LeRobot's documentation, which is the most concrete practical guide (50 episodes, ACT first, a few GPU-hours).

### Cited Findings
- "Robotic Manipulation via Imitation Learning: Taxonomy, Evolution, Benchmark, and Challenges" (Aug 2025, v2 Sept 2025): covers each paper's "hierarchical classification, input formats, key priors, strengths and limitations, and citation metrics" and "where available, we report benchmark results"; the abstract states no data-efficiency conclusions — [arXiv 2508.17449](https://arxiv.org/abs/2508.17449)
- "Diffusion Models for Robotic Manipulation: A Survey" (Apr 2025): reviews grasp learning, trajectory planning and data augmentation, stresses diffusion models' "ability to model multi-modal distributions" and diffusion-based augmentation "to enhance generalizability and data scarcity"; no quantitative data-efficiency guidance in the abstract — [arXiv 2504.08438](https://arxiv.org/abs/2504.08438)
- "A Survey of Imitation Learning Methods, Environments and Metrics" (2024): notes the proliferation of metrics and environments creates a lack of standardisation in evaluation — [arXiv 2404.19456](https://arxiv.org/abs/2404.19456)
- "Interactive Imitation Learning for Dexterous Robotic Manipulation: Challenges and Perspectives" (2025) exists as a survey of human-in-the-loop IL — [arXiv 2506.00098](https://arxiv.org/abs/2506.00098)
- LeRobot documentation as a practical guide: ACT is "the first model we recommend when you're starting out"; "Often achieves high success rates with just 50 demonstrations"; record "at least 50 episodes, with 10 episodes per location"; "Avoid adding too much variation too quickly"; "Expect a few hours for 100k training steps on a single GPU"; Colab and HF Jobs paths for people without a GPU — [LeRobot ACT docs](https://huggingface.co/docs/lerobot/act); [LeRobot il_robots tutorial](https://huggingface.co/docs/lerobot/en/il_robots)
- Trend statements from the 2026 chunking paper for practitioners: chunking "is not strictly required in every setting"; history-conditioned single-action policies or small ensembles capture much of its benefit; minimise inference pauses on real robots — [arXiv 2608.02547](https://arxiv.org/abs/2608.02547)
- The older "What Matters" study remains the most cited practical guide for small-data BC design (encoder, augmentation, GMM head, learning rate, checkpoint selection) — [robomimic](https://arxiv.org/abs/2108.03298)

### Inferences
- For the written report, the team can cite robomimic (design rules), LeRobot docs (demo counts and compute), the data-scaling-laws paper (diversity over repetition), and the 2025 to 2026 theory papers (why discrete actions and non-smooth policies limit compounding error) as the "state of the art" framing, and state explicitly that no survey addresses their exact regime.

### Gaps
- Full texts of the two 2025 surveys were not read; whether they tabulate ACT/DP demo counts is unverified.

---

## Synthesis: which methods are realistic for this team (comparison across methods, propagated to the use case)

### Takeaway
Realistic ranking for this team: (1) single-frame small CNN or ImageNet-ResNet-18 classifier with random crop plus colour jitter, 4 to 6 discrete step classes plus a stop flag, trained on a CPU in minutes to an hour, with DAgger-style recovery rounds; (2) the same with a 2-frame or GRU history, or a 3-model ensemble, if crossings/hairpins fail; (3) VINN as a no-training diagnostic baseline; (4) ACT via LeRobot only if a GPU is available and the professor values a "modern" architecture; Diffusion Policy, IBC, BeT/VQ-BeT are not justified by the evidence for this unimodal, low-rate, small-data task.

### Cited Findings
- Demo counts and compute at a glance (all cited above): BC 40 to 200 demos, learning rate 1e-4, ResNet-18 encoder ([robomimic](https://arxiv.org/abs/2108.03298)); ACT 50 demos, 80 M params, about 5 h on an RTX 2080 Ti, 10 ms inference ([ACT](https://arxiv.org/abs/2304.13705)); Diffusion Policy 136 to 200 demos, 12 h training budget in the real Push-T experiment, about 0.1 s GPU inference with DDIM ([Diffusion Policy](https://arxiv.org/abs/2303.04137)); VQ-BeT 45 to 206 demos, 18 ms GPU / 207 ms CPU vs DP-T 573 ms / 5.2 s ([VQ-BeT](https://arxiv.org/abs/2403.03181)); VINN 71 demos, 3.5 h BYOL fine-tune on an RTX 8000 (or none if the encoder is frozen), about 0.11 s inference ([VINN](https://arxiv.org/abs/2112.01511)); IBC about 5x slower training than MSE BC, unstable ([IBC](https://arxiv.org/abs/2109.00137)); Hansen's 5-layer ConvNet from scratch with random shift is competitive with frozen pretrained encoders on 10 to 100 demos ([Hansen et al.](https://arxiv.org/abs/2212.05749))
- Chunking/ensembling benefits are tied to high control rates; at low rates the gain is from history conditioning and ensembling ([arXiv 2608.02547](https://arxiv.org/abs/2608.02547)); chunk length should shrink as environment noise rises ([arXiv 2408.17355](https://arxiv.org/abs/2408.17355))
- Discrete-action BC with log-loss has optimal sample complexity ([arXiv 2603.20538](https://arxiv.org/abs/2603.20538)); smooth deterministic continuous imitators compound error exponentially ([arXiv 2503.09722](https://arxiv.org/abs/2503.09722))
- Diversity of training situations, about 50 demos per situation, drives generalisation to unseen environments ([arXiv 2410.18647](https://arxiv.org/abs/2410.18647))

### Inferences
- Compute for option (1): a 5-layer ConvNet or ResNet-18 on about 10,000 images of 144x108 is a few minutes per epoch on a modern laptop CPU, so the full loop (record, train, test on the robot, add corrections) fits a weekly iteration cycle before the January deadline. Inference per frame is tens of milliseconds on CPU, negligible against the 1 s move.
- Expected behaviour: straight lines and gentle arcs should work from a few dozen episodes (robomimic's 20 percent-data results on easy tasks); the hairpin and the crossings are the "Square/Transport" of this task and will need targeted demonstrations and recovery data. Report success per path type, not one number.
- What would make ACT or DP worthwhile: if the team moves to continuous sub-centimetre steps, multiple cameras, or a non-blocking controller, the chunking and generative machinery starts to pay; none of that is planned.

### Gaps
- All compute estimates for the team's laptop are extrapolations; no source benchmarks CPU training of a small BC policy on about 10,000 wrist-camera frames.
- Nothing in the literature covers a blocking, one-step-per-second robot interface; the mapping from the cited findings to this regime is inference throughout.
