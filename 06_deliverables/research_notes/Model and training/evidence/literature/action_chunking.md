# Action chunking and ACT: literature notes

Key: `action_chunking`. Written 9 November 2026 by a Claude research sub-agent for Team M (Emranul Hoque and Marshal Sebastine), Project 6, Niryo Ned2 puck pusher.

Question: what action chunking and ACT fix, what their own numbers show (chunk sizes, temporal ensembling, CVAE, data, inputs, network size, compute), what later papers found about chunk length and about reacting to new pictures versus executing chunks, whether chunking would help our stop-and-go one-camera pusher whose finishing time is graded, and whether ACT can run without joint positions.

## How to read these notes

- **[checked]**: I read the statement in the source myself (PDF text, HTML page, figure image or code). **[unchecked]**: I did not verify it in the source.
- **M** = the paper measured it. **A** = the authors' explanation, argument or hypothesis (not directly measured). **T** = a theory result (proved under stated assumptions, not an experiment). **D** = a description of the method, the setting or the code. **J** = my own judgement for our project (no paper says it). **C** = my own arithmetic on stated numbers (assumptions given).
- "points" means percentage points.
- Figure numbers for the ACT paper follow the arXiv PDF (v1). The arXiv HTML version numbers its figures one lower (the ablation figure is Fig. 8 in the PDF and Fig. 7 in the HTML).
- PDF texts I extracted are in `sources/` next to this note (ACT and Diffusion Policy). Other papers were read through their arXiv HTML pages.

## 0. Short answer in plain words

1. ACT (Zhao et al., 2023) makes the network output the next k actions at once instead of one. In their simulation tests, at 50 decisions per second, this raised success from 1% (k = 1) to 44% (k = 100). [checked, M]
2. Why it helps, according to the authors: fewer decisions per episode means fewer chances for small errors to pile up (compounding error), and a chunk can carry a pause or other behaviour that depends on time rather than on what the camera sees. [checked, A]
3. Later papers agree that chunks help when the world behaves predictably, but executing a chunk without looking again (open loop) hurts when there is noise or when the robot's motion is imprecise (Liu et al., 2025; Lee et al., 2024; Chi et al., 2023). [checked, M]
4. ACT can run without joint positions: the LeRobot version of ACT accepts images only, and a 2025 study ran ACT without the robot state and it generalised better to moved objects. [checked, D and M]
5. For us: chunking is worth testing mainly as a way to save time (fewer pictures per track), but as "predict several steps, execute only 1, 2 or 3 per picture", because the puck can slip and finishing counts most. [J]

## 1. The ACT paper

**Source.** Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. Robotics: Science and Systems XIX (RSS 2023, Daegu), paper 016, DOI 10.15607/RSS.2023.XIX.016; arXiv 2304.13705v1. Opened: https://arxiv.org/pdf/2304.13705 (text extracted with pdftotext), https://arxiv.org/html/2304.13705v1, https://www.roboticsproceedings.org/rss19/p016.html, and the ablation figure image https://arxiv.org/html/2304.13705v1/fm_data.png (the published image still shows the small data tables the plots were drawn from).

### 1.1 What chunking is meant to fix

- **Compounding error.** The policy predicts the target joint positions for the next k steps rather than one step, modelling π(a_t:t+k | s_t) instead of π(a_t | s_t). The authors say this cuts the effective horizon of the task k-fold and so mitigates compounding errors (Sec. I; Sec. IV-A). [checked, D + A]
- **Pauses and other time-dependent behaviour.** Predicting sequences also helps with "pauses in demonstrations that are hard to model with Markovian single-step policies" (Sec. I). Sec. IV-A explains: during a pause the state looks the same, but the right action depends on how long the pause has lasted; a chunk handles this when the pause lies inside one chunk, without the causal-confusion risk of feeding past observations. [checked, A]
- **What they saw in the baselines.** Single-step baselines got worse towards the end of an episode, and the robot could stop and stay still forever in some states. The authors attribute this to compounding errors and non-Markovian data (Sec. V-C). The attribution is not tested separately. [checked, A]
- **Naive chunking is jerky.** Executing a whole chunk and only then looking again brings in new information abruptly every k steps, which they say gives jerky motion. This motivated temporal ensembling (Sec. IV-A). [checked, A]

### 1.2 Chunk sizes tested and their effect (Sec. VI-A, Fig. 8a)

Setting: two simulated bimanual tasks (Cube Transfer, Bimanual Insertion), each trained on scripted or on human demonstrations; success averaged over these four settings; temporal ensembling switched off; a separate policy trained for each k. k = 1 means no chunking; the figure's axis marks k = 400 as fully open loop (the whole episode planned from the first picture). Values from the data tables inside the published figure image. [checked, M]

| Chunk size k | 1 | 10 | 100 | 200 | 400 (open loop) |
|---|---|---|---|---|---|
| ACT (no temporal ensembling), success % | 1 | 21 | 44 | 42 | 41 |
| BC-ConvMLP with chunking added | 1 | 1 | 24 | 25 | 20 |
| VINN with chunking added | 0 | 6 | 23 | 27.25 | 37 |

- The authors call the fall after k = 100 a slight dip and blame less reactive behaviour and the difficulty of modelling long action sequences (Sec. VI-A). [checked, A]
- Only these five values of k appear on the axis, so the best k could lie anywhere between 10 and 200. [checked, D]
- There is no chunk-size test on the real robot; all real tasks used k = 100 (Table III). [checked, D]
- (J) These are simulated tasks with fixed physics; the only randomness is where the objects start, which the first picture already shows. That is why even fully open-loop control reaches 41%. The test therefore says little about reacting to disturbances such as a puck slipping.
- (C) k = 100 at 50 Hz is 2 seconds of motion, about 15 to 25% of a 400 to 700 step real episode.

### 1.3 Temporal ensembling (Sec. IV-A, Algorithm 2, Fig. 5, Fig. 8b)

- **How it works.** The policy is asked for a new chunk at every time step, so chunks overlap and each time step receives several predicted actions (one from each recent chunk). The executed action is their weighted average with weights w_i = exp(-m * i), where w_0 belongs to the oldest prediction; a smaller m lets new observations in faster. It costs nothing in training, only extra computation when running. [checked, D]
- **Value of m.** The paper gives no number. The official code (`imitate_episodes.py`) sets `k = 0.01` for this weight and queries the policy every step when ensembling is on. [checked, D]
- (C) With m = 0.01 and k = 100, the newest prediction gets about exp(-0.99) ≈ 0.37 times the weight of the oldest one. Each executed action is therefore an almost even average of up to 100 predictions made over the last 2 seconds: strong smoothing, slow reaction.
- **Ablation** (best success over k, the same four simulated settings, each variant tuned separately): ACT 44 to 47.3 (+3.3 points); BC-ConvMLP 25 to 29 (+4); VINN 37 to 17 (-20). [checked, M]
- The authors' explanation: ensembling smooths the modelling errors of a parametric network; VINN replays recorded actions, so it has no such errors to smooth (Sec. VI-A). [checked, A]
- Real-robot use: the paper presents ACT as chunking plus temporal ensembling and gives the ensembled inference as Algorithm 2, but it does not say task by task whether ensembling was on for the real robot. [checked, D]
- (J) Important for us: with ensembling, ACT still takes a new observation at every step (every 20 ms). Its real-robot success is the success of a policy that looks very often, not of one that runs blind for 2 seconds.

### 1.4 The CVAE and its role (Sec. IV-B, IV-C, App. C, Fig. 4, Fig. 11, Fig. 8c)

- Human demonstrations vary: the same situation can be followed by different motions, and people are less precise where precision matters less (Sec. IV-B). [checked, A]
- ACT is trained as a conditional variational autoencoder (CVAE). A small transformer encoder reads a learned [CLS] token, the current joint positions and the demonstrated k-step action sequence (images left out to train faster) and outputs the mean and variance of a style variable z with 32 numbers (Fig. 11; code `self.latent_dim = 32`). The decoder, which is the actual policy, reads the images, the joint positions and z and outputs the k actions. Loss: L1 reconstruction plus beta times the KL term, beta = 10 (Algorithm 1; Table III). [checked, D]
- At test time the encoder is thrown away and z is set to 0 (the mean of the prior), so the policy gives the same output for the same input (Sec. IV-B; App. C). [checked, D]
- **Ablation** (Fig. 8c, two simulated tasks): scripted data 59% with the CVAE and 58% without; human data 35.3% with and 2% without. "Without" means the same network trained with plain L1 on the action sequence. [checked, M]
- (J) The CVAE matters when demonstrations disagree with each other. For our move names, a classifier with a softmax already copes with several possible answers for one picture. For the continuous "exact step" output trained with L1, disagreement could average out into a wrong step; a CVAE or a classification head would help there. Not tested.

### 1.5 Inputs and outputs (Sec. III, IV, IV-C, App. B)

- Observation: 4 RGB webcams (two on the wrists, one at the front, one on top), 480 x 640 each, plus the 14 joint positions of the two follower arms (7 + 7). [checked, D]
- Action: the 14 absolute target joint positions read from the leader arms (what the human commanded), tracked by the motors' own PID controllers. Teleoperation and recording at 50 Hz; cameras stream at 30 frames per second. [checked, D]
- The authors noted worse results with delta (relative) joint positions as actions, and more precise modelling with L1 than with L2 loss, but give no numbers (Sec. IV-C). [checked, A]
- The checkpoint with the lowest validation loss is used for testing (Sec. IV). [checked, D]

### 1.6 Network size, compute and time

- A ResNet18 per camera turns each 480 x 640 image into a 15 x 20 x 512 feature map (300 tokens); four cameras give 1200 tokens, plus one token for the joints and one for z. Transformer encoder 4 layers, decoder 7 layers, hidden size 512, feed-forward size 3200, 8 heads; around 80M parameters; trained from scratch for each task (Sec. IV-C; Table III). [checked, D]
- Other settings (Table III): learning rate 1e-5, batch size 8, chunk size 100, beta 10, dropout 0.1. [checked, D]
- Training takes around 5 hours on one 11 GB RTX 2080 Ti; inference around 0.01 s on the same machine (Sec. IV-C). [checked, M]
- Code details [checked, D]: the ResNet18 is built with torchvision's pretrained weights (ImageNet by torchvision's convention) and frozen batch norm (`detr/models/backbone.py`), so training from scratch applies to the rest of the network. Each training epoch draws one random time step from each episode (`utils.py`). The README trains the simulated tasks for 2000 epochs and advises at least 5000 epochs for real data, or 3 to 4 times as long as it takes the loss to level off, saying success and smoothness can keep improving after that (README advice, not a measured result).
- A known quirk [checked, D]: the original code computes 7 decoder layers but uses only the first layer's output (`detr/models/transformer.py` returns all layers stacked, `detr/models/detr_vae.py` takes element 0). LeRobot's ACT configuration says so in a comment and sets `n_decoder_layers = 1` to match.
- (C) Parameter count by hand: ResNet18 about 11.7M, encoder 4 x 4.3M ≈ 17M, decoder 7 x 5.4M ≈ 38M, CVAE encoder ≈ 17M, total ≈ 84M, in line with the paper. At test time only about 34M take part (CVAE encoder dropped, 6 decoder layers unused).

### 1.7 Data sizes (Sec. V-B)

- 50 demonstrations per real task (100 for Thread Velcro). Each episode takes 8 to 14 s, that is 400 to 700 steps at 50 Hz; about 10 to 20 minutes of data per task, 30 to 60 minutes of wall-clock time with resets and mistakes. Simulated tasks: 50 successful demonstrations each, scripted or human. [checked, D]
- (C) 50 x (400 to 700) = 20,000 to 35,000 frames per task, one model per task. Our 18,500 pictures are about the same number of frames, but spread over six sheets, meant to generalise to unseen tracks, and taken one per 1 cm step.

### 1.8 Results (Tables I and II)

- Real robot, one seed, 25 trials per task: final success Slide Ziploc 88%, Slot Battery 96%, Open Cup 84%, Thread Velcro 20%, Prep Tape 64%, Put On Shoe 92%. Four baselines (BC-ConvMLP, BeT, RT-1, VINN) reached 0% final success on Ziploc and Battery; BeT, the only baseline run on the other four, reached 0% final success on them. [checked, M]
- Simulation, 3 seeds x 50 trials: ACT final success Cube Transfer 86% (scripted data) and 50% (human data); Bimanual Insertion 32% and 20%. The best baseline (BeT) reached 27% and 1%, and 3% and 0%. [checked, M]
- Test conditions: objects placed at random along a 15 cm line (real) or in 2D regions (simulation), always on the task the model was trained on. There was no test on new tasks or new layouts. [checked, D]

### 1.9 Control frequency user study (Sec. VI-C, Fig. 8d)

Six people teleoperated two fine tasks at 50 Hz and at 5 Hz: zip tie 33 s at 5 Hz against 20 s at 50 Hz; unstacking cups 16 s against 10 s; overall 62% slower at 5 Hz (p < 0.001, repeated-measures test). This is about how fast humans teleoperate, not about the learned policy. [checked, M]

## 2. Can ACT run without joint positions?

- **Original ACT code: no.** `detr/models/detr_vae.py` always projects `qpos` (the joint positions) into the policy; there is no image-only path. https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py [checked, D]
- **LeRobot ACT: yes.** The docstring in `configuration_act.py` says the policy may optionally work without an `observation.state` key (the robot's joint state); the feature check only requires at least one image or an environment state. In `modeling_act.py` the state tokens exist only if the robot state is given; without it the CVAE encoder reads just [CLS] and the action sequence. Defaults: chunk_size 100, n_action_steps 100 (the whole chunk is executed before the next picture), temporal ensembling off (temporal_ensemble_coeff None; a comment gives 0.01 as the original value), latent_dim 32, kl_weight 10, resnet18 backbone. https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py and .../modeling_act.py (main branch, opened 9.11.26). [checked, D]
- **Evidence that it works.** Juntu Zhao, Wenbo Lu, Di Zhang, Yufeng Liu, Yushen Liang, Tianluo Zhang, Yifeng Cao, Junyuan Xie, Yingdong Hu, Shengjie Wang, Junliang Guo, Dequan Wang, Yang Gao (2025). Do You Need Proprioceptive States in Visuomotor Policies? arXiv 2509.18644v2 (preprint, no venue stated). Opened: https://arxiv.org/html/2509.18644v2.
  - Pick Pen task, relative end-effector actions, two wide-angle wrist cameras. Demonstrations collected at a table height of 80 cm; "height" score averages tests at 72 and 90 cm; "horizontal" score averages tests with the pen holder moved 5 and 10 cm. The pick-and-place tasks used about 300 episodes each (stated for the task family). Table V [checked, M]:

| Policy | With state (height / horizontal) | Without state (height / horizontal) |
|---|---|---|
| ACT | 0 / 0.084 | 0.933 / 0.517 |
| Diffusion Policy | 0 / 0 | 0.867 / 0.533 |
| π0 | 0 / 0 | 0.984 / 0.584 |

  - Same paper, π0 without state (Table III): absolute end-effector, relative joint-angle and absolute joint-angle actions all reached 1.0 in the training layout but 0 after the layout changes; only relative end-effector actions generalised (0.984 / 0.584). [checked, M]
  - Same paper, cameras (Table IV, π0 without state): overhead camera only 0.217 / 0.133; one normal wrist camera plus the overhead camera 0.867 / 0.267; two wide-angle wrist cameras without the overhead camera 1.0 / 1.0. No test used exactly one camera and nothing else. [checked, M]
  - The authors argue that the state input gives the policy a shortcut to memorise training trajectories; they name sensitivity to background changes as a weakness of vision-only policies (Sec. I; Sec. VIII). [checked, A]
- (J) For us: the course rules already allow only the camera as model input, so image-only ACT is the only allowed form anyway. Our step labels are relative to the current pose (tip forward and left, forearm twist, rod lean, hand turn), which is the kind of action that generalised in this study. The camera turns with the forearm roll, so the picture itself carries part of the arm's pose. Dropping joints from the CVAE encoder should cost little because that encoder only exists during training. (Feeding joint positions to the CVAE encoder during training only would still leave the camera as the only input at test time; check with the course rules before trying it.)

## 3. Later evidence on chunk length and reactivity

### 3.1 Diffusion Policy (Chi et al.)

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023; arXiv 2303.04137v5 (2024) is the extended journal version (the journal is not named on the arXiv page). Opened: https://arxiv.org/html/2303.04137v5 and https://arxiv.org/pdf/2303.04137v5 (text extracted).

- The policy predicts Tp steps, executes Ta of them, then looks again (receding horizon; Sec. 2.3). Usual settings To = 2 observations, Ta = 8, Tp = 16 (Table 7). [checked, D]
- Action horizon trade-off (Sec. 5.3, Fig. 5 left; the legend lists Push-T and Square): an action horizon above 1 helps consistent actions and covers idle parts of demonstrations, but "too long a horizon reduces performance due to slow reaction time"; 8 steps was best for most tasks they tested. The tick values are not in the extracted text, so I cannot say which horizons were tried. [checked, M]
- Robust to a delay of up to 4 steps in simulation with position control; velocity control suffers more (Sec. 5.3, Fig. 5 right). [checked, M]
- Idle actions (pauses), Sec. 4.3: they argue single-step policies overfit to pauses and report that BC-RNN and IBC often got stuck on the real robot when pauses were not removed from the training data. An argument plus an observation, not a controlled test. [checked, A]
- Real-world Push-T, a pushing task like ours (Sec. 6.1, Table 6, Table 7, App. C.1, App. D.0.1): UR5 robot, 10 Hz commands, two cameras at 320 x 240, 136 demonstrations, To = 2, Tp = 16, Ta = 6 (0.6 s of motion per look). Diffusion Policy (CNN, vision trained end to end) succeeded in 95% of trials, IoU 0.80 against 0.84 for the human demonstrations; average episode 22.9 s against 20.3 s for the humans (my reading of the table layout in the extracted PDF text). IBC and LSTM-GMM reached 0 to 20% success. Fixed training time of 12 hours per method. [checked, M]

### 3.2 VQ-BeT (Lee et al., 2024)

Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, Lerrel Pinto. Behavior Generation with Latent Actions. ICML 2024 (PMLR 235). Opened: https://arxiv.org/html/2403.03181v2.

- On a low-cost mobile manipulator (Hello Robot Stretch), Diffusion Policy run closed loop (one action per prediction) succeeded in 47 of 60 trials over six tasks; executing three actions per prediction succeeded in 0 of 30 (Sec. 4.7, Table 11). [checked, M] The authors blame the robot's imprecise motion control, which pushed the policy out of its training distribution within three open-loop steps. [checked, A]
- In their own ablation, chunking did not help VQ-BeT (Sec. 4.6, Fig. 5; I did not read the plotted numbers), and they judge chunking unnecessary for their small, fast model. [checked, M]

### 3.3 BAKU (Haldar et al., 2024)

Siddhant Haldar, Zhuoran Peng, Lerrel Pinto. BAKU: An Efficient Transformer for Multi-Task Policy Learning. NeurIPS 2024 (poster). Opened: https://arxiv.org/html/2406.07539v2 and https://neurips.cc/virtual/2024/poster/93282.

- Ablation (Sec. 4.4, Table 3): a chunk of 10 actions combined with temporal ensembling (App. A.3), against predicting a single action. LIBERO-90 0.90 against 0.76; Meta-World 0.79 against 0.78; DeepMind Control locomotion 0.70 against 0.74. Chunking helped manipulation and slightly hurt locomotion. [checked, M]

### 3.4 Bidirectional Decoding (Liu et al., 2025)

Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, Chelsea Finn. Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling (earlier versions: "via Closed-Loop Resampling"). ICLR 2025 (project page). arXiv 2408.17355 (v1 August 2024, v4 April 2025). Opened: https://arxiv.org/html/2408.17355v4 and https://bid-robot.github.io/.

- Theory (Sec. 3, Proposition 1, Corollaries 2 and 3): a longer chunk keeps the policy on one strategy and lets it reproduce time-dependent behaviour (consistency); a shorter chunk lets it use fresher observations (reactivity). Longer wins in near-deterministic settings, shorter wins in stochastic ones, and no single chunk length is best everywhere. [checked, T]
- One-dimensional test with a pausing expert (Sec. 5.1, Table 3): the expert steps forward from s0 to s10 but stays at s5 until it has spent five steps there; each forward move fails with probability δ (noise). Distance to the expert's behaviour (total variation, lower is better) for action horizons h [checked, M]:

| Noise δ | h = 1 | h = 3 | h = 5 | h = 7 | h = 10 |
|---|---|---|---|---|---|
| 0.0 | 4.03 | 2.07 | 1.54 | 1.06 | 0.00 |
| 0.2 | 1.43 | 0.94 | 0.39 | 0.71 | 1.32 |
| 0.4 | 0.36 | 0.42 | 0.59 | 0.835 | 1.11 |

- Push-T with a VQ-BeT policy and temporally correlated noise added to the executed actions (Table 1), success % at noise 0 / 1.0 / 1.5 [checked, M]:
  - vanilla open loop (execute the chunk): 64.0 / 26.9 / 13.0
  - vanilla closed loop (predict a chunk every step, execute only its first action): 48.9 / 38.3 / 29.5
  - BID closed loop: 54.4 / 45.3 / 31.7
- How BID works: it samples many candidate chunks at each step and picks the one that agrees with the previous plan (backward coherence) and is likely under the trained model but unlikely under a weak early checkpoint (forward contrast). Default 16 samples from each model. On an A5000 GPU with VQ-BeT: 13.2 ms for one sample against 25.2 to 26.8 ms for 8 to 32 samples (Table 2). [checked, D + M]
- Temporal ensembling (called EMA here, λ = 0.5) averaged 34.9 (open loop) and 35.6 (closed loop) against BID's 37.8 and 43.8 on Push-T with VQ-BeT (App. A.5, Table 4). [checked, M]
- On two real tasks with a moving target cup, BID reached at least twice the success of the other methods; with a static cup it matched them (Sec. 5.3, Fig. 9; exact numbers only in the figure). [checked, M]
- Stated limitations: the extra computation may be too much for high-frequency control on cheap robots; the analysis covers short-context policies only (Sec. 6). [checked, A]
- (J) BID needs a policy that can draw several different samples. ACT with z = 0 gives one answer per picture, so BID applies only if we sample (random z, or sampling from the move-name probabilities).

### 3.5 Theory: when chunking stops errors piling up (Zhang et al., 2025)

Thomas T. Zhang, Daniel Pfrommer, Chaoyi Pan, Nikolai Matni, Max Simchowitz. Action Chunking and Exploratory Data Collection Yield Exponential Improvements in Behavior Cloning for Continuous Control. arXiv 2507.09061v5 (preprint, November 2025; I could not confirm a venue). Opened: https://arxiv.org/html/2507.09061v5.

- Theory (Sec. 3, Assumption 3.1, Proposition 3.1, Theorem 1): if the system stays stable when driven without feedback (open-loop stable, as with position control and a good low-level tracking controller) and the policy is smooth, executing long enough chunks keeps the error bound independent of the task length. The chunk length needed grows only logarithmically with the stability constants, and longer chunks beyond that add little. If the system is not open-loop stable, chunking alone is not enough (Sec. 3, Sec. 4). [checked, T]
- Their remedy for unstable systems: record some demonstrations with small noise added to the executed actions, while storing the clean intended actions as labels (Sec. 4, their Practice 2). Using the noisy actions as labels performed badly (Fig. 10 left). [checked, T + M]
- Experiments: on robomimic tool_hang (state input), executing slightly longer chunks at test time improved success drastically (Fig. 5); on MuJoCo HalfCheetah without end-effector control, naive chunking hurt (Fig. 10 right). I did not read the plotted values. [checked, M]
- (J) Pushing a round puck with a flat block is not open-loop stable: the puck can slide sideways off the block. By this theory, chunks help the arm's own motion but cannot replace looking at the puck. Their noise-injection idea fits our replay-based recording tool (replay with small deliberate offsets, keep the clean step as the label); that belongs to the recovery-data question.

### 3.6 Real-time chunking and pauses between chunks (Black et al., 2025)

Kevin Black, Manuel Y. Galliker, Sergey Levine. Real-Time Execution of Action Chunking Flow Policies. NeurIPS 2025 (arXiv comments). arXiv 2506.07339v2. Opened: https://arxiv.org/html/2506.07339v2.

- Problem: computing the next chunk only after the current one ends (synchronous inference) makes the robot stop and wait, while switching chunks carelessly makes it jump. The abstract warns of "pauses or out-of-distribution jerky movements at chunk boundaries". [checked, A]
- They argue the pauses both slow the task and change the robot's dynamics, so testing no longer looks like training (Sec. 2). [checked, A]
- Real robot (π0.5, 50-step chunks at 50 Hz, the synchronous baseline executes 25 actions per chunk; six bimanual tasks, 10 trials per task and method): synchronous throughput fell roughly in proportion to the added delay; real-time chunking had the best throughput at every delay and finished tasks faster than synchronous inference even when the pauses were removed (Sec. 4.2, Fig. 6). [checked, M]
- Temporal ensembling did poorly across the board in simulation, even with no delay, and on the real robot oscillated enough to trigger the protective stop at +100 and +200 ms of delay (Sec. 4.1, Sec. 4.2). Their explanation: the average of two valid actions need not be a valid action. [checked, M + A]

### 3.7 Speed: chunking and faster-than-demonstration execution

- **OpenVLA-OFT.** Moo Jin Kim, Chelsea Finn, Percy Liang (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. RSS 2025. Opened: https://arxiv.org/html/2502.19645v2. On LIBERO, parallel decoding plus chunking (K = 8; the full chunk is executed before the next look) raised throughput from 4.2 to 108.8 actions per second on an A100 and average success from 76.5% to 90.2% (Table I, Table II). Adding chunking raised the time per model call from 0.0629 to 0.0735 s. [checked, M]
- **SAIL.** Nadun Ranawaka Arachchige, Zhenyang Chen, Wonsuhk Jung, Woo Chul Shin, Rohan Bansal, Pierre Barroso, Yu Hang He, Yingyang Celine Lin, Benjamin Joffe, Shreyas Kousik, Danfei Xu (2025). SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies. CoRL 2025. Opened: https://arxiv.org/html/2506.11948v2.
  - Simply playing Diffusion Policy faster lowered success in simulation: Can 0.97 to 0.87, Square 0.83 to 0.55, Mug 0.68 to 0.56 (Table 1). [checked, M]
  - Their full system applied to ACT reached 1.7 to 3.7 times the demonstration speed, while success fell: Lift 1.00 to 0.94, Can 0.77 to 0.56, Square 0.51 to 0.43, Stack 0.73 to 0.68 (App. K, Table K.7). [checked, M]
  - Their metric "throughput with regret" (App. B) scores 1/duration for each success and subtracts 1/t_max for each failure, so it rewards finishing first and speed second. [checked, D]
- (J) For us the model call is probably a small part of the 1.2 s step, so OpenVLA-OFT's speed-up does not carry over directly; SAIL shows that running faster than the demonstrations costs success, which matters because finishing counts most.

## 4. What this means for our pusher (judgement and calculation)

### 4.1 ACT on ALOHA compared with our set-up

| | ACT on ALOHA | Our pusher |
|---|---|---|
| Decision rate | 50 per second | about 1 per 1.2 s, stop and go |
| Size of one action | target joints 20 ms ahead (millimetres) | about 1 cm or 5 degrees |
| Decisions per episode | 400 to 700 | about 89 (C: 18,500 / 208) |
| Cameras | 4 (two on the wrists) | 1, on the forearm, turning with the forearm roll |
| Joint positions as input | yes | not allowed by the course |
| Action type | absolute joint targets | relative steps (move names and exact steps) |
| Data | 50 demonstrations, one model per task | 208 demonstrations over six sheets, one model |
| Test | same task, object moved along 15 cm | same sheets plus unseen tracks |

### 4.2 Would chunking help us?

- (J) Horizon: our 1 cm steps already make each decision a coarse, larger action, and a track takes only about 89 decisions. ACT's jump from 1% to 44% came at 50 decisions per second; the gain from shortening our horizon further is likely much smaller.
- (J) Pauses: our demonstrations are replayed step by step with a picture before each step, so hesitations during hand guiding probably do not appear as repeated pictures with no move. Count how often "stop" and "small motion" labels occur in the middle of tracks; if often, a one-step policy may get stuck there (as Chi et al. saw for BC-RNN and IBC), and chunks would help.
- (J) Consistency: at the crossing of the loop sheet and over the gaps of the dashed loop, one picture may fit more than one move. A chunk commits to a direction for several steps, which may help (the consistency side in Liu et al.).
- (J) Reactivity: the puck can slip along the block, especially on curves and on unseen tracks. The evidence that open-loop steps hurt under noise or imprecise control (Liu et al. Table 1; Lee et al. Table 11) argues for executing short chunks only.
- (C) Time: if a fraction f of the 1.2 s per step goes to taking the picture and deciding, executing n steps per picture saves about (1 - 1/n) x f x 1.2 s per step. Example: f = 0.5 and n = 3 give 0.4 s per step, about 36 s on an 89-step track that now takes about 107 s. Measure f first. If the arm can run several steps as one smooth move (check the Niryo API), the saving is larger, but the puck may then move differently from the stop-and-go demonstrations.
- (J) Suggested test: train the model to output the next K steps (for example K = 5 to 10), then run it with n = 1, 2 and 3 steps per picture. Compare on a held-out sheet (leave one sheet out), ranking by finishing rate first and time second (SAIL's throughput with regret is one way to combine both). Leave temporal ensembling out at first: it needs a picture every step, so it saves no time, it gave only +3.3 points in ACT, and it did badly in Black et al.

### 4.3 ACT on our laptop CPU (C, rough)

- Assumptions: one camera at 480 x 640, k = 100, training costs about 3 times a forward pass, a 6-core laptop CPU delivering 100 to 200 GFLOP/s in PyTorch.
- ResNet18 at 480 x 640 is about 11 billion multiply-adds per picture; the transformer about 10 billion (encoder about 5.6, the 7 computed decoder layers about 4.5); so training costs about 60 billion multiply-adds (about 125 GFLOP) per sample, roughly 0.6 to 1.3 s per sample.
- ACT's real-robot budget (5000 epochs x 50 samples = 250,000 samples, about 13 passes over our 18,500 pictures) would take roughly 40 to 90 hours.
- Halving the picture's width and height (240 x 320), one decoder layer and a short chunk (k about 10) cut the cost about 5 times (about 26 GFLOP per sample), to roughly 10 to 20 hours.
- One forward pass at full size (about 42 GFLOP) would take roughly 0.2 to 0.4 s, a noticeable part of our 1.2 s step; at the smaller size well under 0.1 s.
- These are guesses: time one training epoch on the laptop before deciding.

### 4.4 Conclusion (evidence first, then judgement)

Evidence: In ACT's simulated ablation (50 Hz), chunking raised success from 1% (k = 1) to 44% (k = 100), with 41% fully open loop; temporal ensembling added 3.3 points; the CVAE mattered only for human data (35.3% vs 2%). Later work shows the price of not looking: open-loop chunks win when things are predictable and lose under noise or imprecise control (Liu et al. Push-T: 64.0 vs 48.9 without noise, 13.0 vs 29.5 with strong noise; three open-loop steps on a Stretch: 0/30 vs 47/60), and Diffusion Policy found about 8 steps at 10 Hz best. LeRobot's ACT accepts images only, and a 2025 study ran ACT without state, generalising better with relative actions.

Judgement: our 1 cm steps already shorten the horizon (about 89 decisions per track) and our replayed demonstrations probably hold few pauses, so ACT's gains will likely be smaller for us. Chunking's main value here is time (fewer pictures), but the puck can slip. Train a model predicting 5 to 10 steps, compare executing 1, 2 or 3 per picture on a held-out sheet, ranking by finishing first and time second; use image-only input and relative steps.

## 5. Sources opened

| Source | URL opened |
|---|---|
| Zhao, Kumar, Levine, Finn (2023), ACT / ALOHA, RSS 2023 | https://arxiv.org/pdf/2304.13705 ; https://arxiv.org/html/2304.13705v1 ; https://arxiv.org/html/2304.13705v1/fm_data.png ; https://www.roboticsproceedings.org/rss19/p016.html |
| ACT official code (tonyzhaozh/act) | https://github.com/tonyzhaozh/act ; .../blob/main/imitate_episodes.py ; .../blob/main/utils.py ; .../blob/main/detr/models/detr_vae.py ; .../blob/main/detr/models/transformer.py ; .../blob/main/detr/models/backbone.py |
| LeRobot ACT code (Hugging Face) | https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py ; https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py |
| Chi et al., Diffusion Policy (RSS 2023; extended arXiv v5) | https://arxiv.org/html/2303.04137v5 ; https://arxiv.org/pdf/2303.04137v5 |
| Lee et al. (2024), VQ-BeT, ICML 2024 | https://arxiv.org/abs/2403.03181 ; https://arxiv.org/html/2403.03181v2 |
| Haldar, Peng, Pinto (2024), BAKU, NeurIPS 2024 | https://arxiv.org/html/2406.07539v2 ; https://neurips.cc/virtual/2024/poster/93282 |
| Liu et al. (2025), Bidirectional Decoding, ICLR 2025 | https://arxiv.org/html/2408.17355v4 ; https://bid-robot.github.io/ |
| Zhang et al. (2025), chunking and exploratory data, preprint | https://arxiv.org/html/2507.09061v5 |
| Black, Galliker, Levine (2025), real-time chunking, NeurIPS 2025 | https://arxiv.org/html/2506.07339v2 |
| Kim, Finn, Liang (2025), OpenVLA-OFT, RSS 2025 | https://arxiv.org/html/2502.19645v2 |
| Arachchige et al. (2025), SAIL, CoRL 2025 | https://arxiv.org/html/2506.11948v2 |
| Zhao, J. et al. (2025), Do You Need Proprioceptive States in Visuomotor Policies?, preprint | https://arxiv.org/html/2509.18644v2 |

## 6. Seen cited but not opened

- Swamy, Choudhury, Bagnell, Wu (2022). Causal imitation learning under temporally correlated noise. ICML 2022. Cited by ACT as reference [61] for the idea of temporally correlated confounders. Not opened.
- Lai, Huang, Gershman (2022). Action chunking as policy compression. PsyArXiv. Cited by ACT as [35] for the psychology origin of the term. Not opened.
- Chi et al. (2024). Universal Manipulation Interface (UMI). Its public checkpoint was used in the BID real-robot test. Not opened.
- A third-party listing places Zhang et al. (2025) at ICLR 2026 under a slightly different title; not opened, venue unconfirmed.
