# Action representation for picture-based behaviour cloning: classes, continuous actions, bins and mixtures

Key: `action_representation`. Written 9.11.26 by a Claude sub-agent for Team M (Emranul Hoque and Marshal Sebastine).
Literature only: nothing in this note was tested on our robot or on our data.

How to read this note

- **Checked** means I opened the source and read the statement there. **Unchecked** means I did not (or, for my own
  calculations, that nobody has tested it).
- **Measured** means the paper ran an experiment and reports the number. **Speculation** means the authors (or I) explain
  or guess, without an experiment that isolates the point. **Definition** means a method description or a mathematical fact.
- Numbers are copied from the papers' tables. Success is given as the paper gives it (a fraction from 0 to 1, or a percentage).
- My own calculations and judgements are only in section 6 and in the conclusion, and are labelled there.
- Most papers were read through their arXiv HTML render (ar5iv). Tables in such renders can be garbled; where I cross-checked
  a table, I say how.

---

## 1. The question in plain words

Our model sees one picture and must say how the arm moves next. There are three ways to write that output.

1. **Continuous regression.** The network outputs the exact step (tip forward, tip left, forearm twist, rod lean, hand turn)
   and is trained with mean squared error (MSE: the average of the squared differences to the demonstrated numbers).
   NVIDIA's PilotNet steers a car this way.
2. **Classes (bins).** The network outputs one probability per move name (8 picture-frame directions, stop, small motion)
   and is trained with cross-entropy (it is rewarded for giving a high probability to the demonstrated class). At run time it
   usually takes the most probable class (the "argmax"). RT-1 does this with 256 bins for each action number.
3. **Outputs with several peaks.** A mixture of Gaussians (GMM, or "mixture density network", MDN), classes plus a fine
   correction inside each class (Behavior Transformers, "bins plus offsets"), or generative models (diffusion, CVAE).

The worry with option 1 is **mode averaging**. A "mode" is one distinct way of acting. If the same picture was followed by
forward-left in some demonstrations and by forward-right in others, MSE learns the average, forward, which nobody
demonstrated and which may push the puck off the line.

---

## 2. Short answer

**What the evidence shows**

- MSE training gives the average of the demonstrated actions for each input (Bishop 1994). When two different actions
  are both valid, that average can be wrong: in Bishop's two-link arm, positioning error fell from 0.0578 (MSE) to 0.0053
  (two-component mixture); in Rahmatizadeh et al. (2018), a push task went from 0% (LSTM with MSE) to 95% (LSTM with MDN).
- Discrete or mixture outputs helped most where the data were multimodal or came from several people. RT-1 lost 29 points
  on seen tasks (97 to 68) and 33 on unseen tasks (76 to 43) when its 256-bin outputs were replaced by a Gaussian with MSE.
  robomimic's recurrent BC lost up to 58% of its success on multi-operator data when the GMM head was replaced by a plain
  (deterministic) output, but almost nothing on single-operator data. ACT fell from 35.3% to 2% on human data without its
  CVAE, and barely changed on scripted data.
- MSE is not always worse. In Pearce et al. (2023) MSE beat per-dimension bins and BeT-style bins plus offsets on the
  kitchen task (0.69 against 0.34 and 0.34). In Implicit BC, an MDN trained from pixels reached 10% where MSE reached 87%.
  Offline steering error was lower for regression than for plain classification in Eraqi et al. (2017).
- Coarse, fixed classes have driven real closed loops: three trail classes (Giusti et al. 2016) and six soft classes
  (TrailNet, 100% autonomy on a 250 m forest trail).
- Behavior Transformers' bins plus offsets beat an MSE network on multimodal tasks. Removing the offsets cost 5 to 22% of
  its success, and fixed uniform bins scored 0.90 to 0.96 of its k-means bins.

**My judgement for our project** (section 6): two people guiding by hand with free twist, lean and hand turn is closer to
robomimic's multi-operator data and ACT's human data than to a single expert. The move-name classifier is already
mode-safe if decoded by argmax. The risk sits in regressing the exact step with MSE, especially the joint numbers. A
BeT-style design fits our labels directly: the move name as the bin, the exact step as the offset. It should be compared
against a plain MSE head on the same network, on the robot.

---

## 3. Steering: regression versus classification

### 3.1 PilotNet: steering as regression

Bojarski et al. (2016), *End to End Learning for Self-Driving Cars*, arXiv:1604.07316 (NVIDIA; arXiv preprint, no
peer-reviewed venue listed). Opened: https://ar5iv.labs.arxiv.org/html/1604.07316

- The network outputs one number, the inverse turning radius 1/r, chosen because it does not depend on the car's geometry,
  stays finite when driving straight and passes smoothly through zero between left and right turns (Sections 2 and 4).
  Checked, definition.
- Training minimises the MSE between that output and the human's command, or an adjusted command for shifted and rotated
  images that steers the car back within two seconds (Sections 2 and 4). Checked, definition.
- About 72 hours of driving had been collected by 28 March 2016 (Section 3). On-road, the car was autonomous about 98% of
  the time on a typical drive (lane changes and turns excluded), and drove 10 miles on the Garden State Parkway without an
  intervention (Section 7.2). Autonomy counts 6 seconds per intervention (Section 7.1, Equation 1). Checked, measured.
- Relevance: lane keeping has one sensible answer per picture most of the time, so averaging does little harm there. The
  paper does not compare against classification.

### 3.2 Steering as classes or bins

**Giusti et al. (2016)**, *A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots*, IEEE
Robotics and Automation Letters (DOI 10.1109/LRA.2015.2509024). Opened: https://rpg.ifi.uzh.ch/docs/RAL16_Giusti.pdf
(text extracted from the PDF).

- Trail following is posed as three classes, turn left, go straight, turn right, split at plus or minus 15 degrees between
  the trail direction and the viewing direction (Section II). Labels come from a hiker wearing three head-mounted cameras
  pointing 30 degrees left, straight ahead and 30 degrees right, so each picture's class is fixed by which camera took it
  (Section III-A). This is the closest published analogue to our fixed "nearest of 8 directions" label. Checked, definition.
- The network reached 85.2% accuracy on the three classes; two human observers reached 86.5% and 82.0% (Table I). A drone
  was flown with yaw proportional to P(turn right) minus P(turn left) and speed proportional to P(go straight)
  (section "Control of a Robot for Autonomous Trail Following"). Checked, measured.

**Smolyanskiy et al. (2017)**, *Toward Low-Flying Autonomous MAV Trail Navigation using Deep Neural Networks for
Environmental Awareness*, IROS 2017 (arXiv comment says IROS 2017; proceedings page not opened). Opened:
https://ar5iv.labs.arxiv.org/html/1705.02550

- TrailNet outputs six classes: three for the view direction (left, centre, right) and three for the sideways offset on the
  trail (left, centre, right), trained with cross-entropy on soft labels plus an entropy reward and a penalty for swapping
  sides (Section IV-A, Equations 1 and 2). The steering angle is a weighted sum of right-minus-left probabilities with
  weights of 10 degrees (Equation 3). Checked, definition.
- They chose classification with soft labels over regression because labels are easier to collect and less sensitive to
  noise; no experiment compares the two (Section IV-A). Checked, speculation.
- On a 250 m trail (Table I): Giusti et al.'s three-class network, 79% accuracy and 80% autonomy; TrailNet, 84% and 100%;
  SqueezeNet, 86% and 98%; ResNet-18, 92% and 88%. Note that higher offline accuracy did not mean higher autonomy (ResNet-18).
  Checked, measured.

**Xu, Gao, Yu and Darrell (2017)**, *End-to-end Learning of Driving Models from Large-scale Video Datasets*, CVPR 2017
(arXiv comment "camera ready for CVPR2017"). Opened: https://ar5iv.labs.arxiv.org/html/1612.01079

- Continuous driving actions (future angular speed) are cut into many small bins and predicted as a multinomial
  distribution, with each label smoothed by a Gaussian (sigma 0.5) over nearby bins. The authors prefer this because a
  single parametric family may not fit all situations (Sections 3.4 and 5.2). Checked, definition.
- Data-driven bins beat log-spaced and linear bins (Table 3: -4.83, -3.66 and -2.82, against 1.86 for a random guess; the
  column is labelled perplexity but is probably a log value, lower is better). They did not compare against regression.
  Figure 7 shows two peaks at an intersection (turn or go straight) that collapse to one after the turn. Checked, measured.

**Eraqi, Moustafa and Honer (2017)**, *End-to-End Deep Learning for Steering Autonomous Vehicles Considering Temporal
Dependencies*, NIPS 2017 Workshop on Machine Learning for Intelligent Transportation Systems. Opened:
https://ar5iv.labs.arxiv.org/html/1710.03804

- Offline steering error on the Comma.ai dataset, degrees RMSE (Table 2): regression 17.77 (CNN) and 16.01 (C-LSTM);
  plain classification 18.70 and 17.84; their sine-wave encoding over 95 output units 17.44 and 14.93. So plain
  classification was slightly worse than regression here, offline. Checked, measured.
- They argue that plain classification treats neighbouring classes as unrelated, so a near miss is penalised like a far miss
  (Figure 3), and that data needs grow with the number of classes. Checked, speculation.

**Michelmore, Kwiatkowska and Gal (2018)**, *Evaluating Uncertainty Quantification in End-to-End Autonomous Driving
Control*, arXiv:1811.06817 (no venue listed). Opened: https://ar5iv.labs.arxiv.org/html/1811.06817

- Two PilotNet-like networks in the Udacity simulator: regression with MSE, and classification over 200 bins of 0.25
  degrees from -25 to +25 degrees with softmax cross-entropy (Section IV-A). The regression car crashed about twice per lap
  and moved jerkily; the classification car crashed 0 to 1 times per lap and drove more smoothly. Classification accuracy
  was only 67%, which they blame on the bins losing detail (Section V-A). Checked, measured. Caution: the two networks
  differ in more than the output (dropout placement, uncertainty method), so this is not a controlled comparison.

**What section 3 shows**: offline error and closed-loop driving can disagree (Eraqi: regression slightly better offline;
Michelmore: classification crashed less in closed loop; TrailNet: best offline accuracy was not best autonomy). Coarse fixed
classes with feedback at every frame have flown and driven real vehicles.

---

## 4. Why MSE averages modes, and what it costs

**Bishop (1994)**, *Mixture Density Networks*, Neural Computing Research Group Report NCRG/94/004, Aston University.
Opened: https://research.aston.ac.uk/en/publications/mixture-density-networks (abstract) and
https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf (full text extracted from the PDF).

- Minimising a sum-of-squares error makes the network output the conditional average of the targets for each input
  (Abstract and Section 2). For problems with several correct outputs, "the average of several correct target values is not
  necessarily itself a correct value" (Abstract). A mixture density network instead predicts a whole probability
  distribution with several peaks. Checked, definition.
- Robot example (Section 6, Table 1): a two-link arm's inverse kinematics has two answers (elbow up, elbow down) for part of
  the workspace. An MSE network's root-mean-square positioning error was 0.0578; an MDN with two components reached
  0.0053, an order of magnitude lower. The MSE errors were largest where two answers exist, because averaging the two joint
  solutions puts the arm somewhere else. Checked, measured.
- Relevance: our exact step contains joint numbers (forearm twist, rod lean, hand turn). If two operators reach the same
  tip move with different joint mixes, an MSE head averages the joint mixes, just like elbow up and elbow down (my
  judgement, not tested).

**Rahmatizadeh, Abolghasemi, Behal and Bölöni (2018)**, *From Virtual Demonstration to Real-World Manipulation Using LSTM
and MDN*, AAAI 2018 (venue from the AAAI listing found by search; the AAAI page itself blocked me). Opened:
https://ar5iv.labs.arxiv.org/html/1603.03833

- Simulated tasks with object poses as input (not pictures), 20 trials each, one demonstrator (section "The benefits of LSTM
  and MDN", unnumbered table): feedforward with MSE 0% (pick and place) and 0% (push to pose); LSTM with MSE 85% and 0%;
  feedforward with MDN 95% and 15%; LSTM with MDN 100% and 95%. On the real Baxter robot, LSTM with MDN reached 80% and 60%.
  Checked, measured.
- Their explanation: both tasks can be done in more than one way, and MSE averages them, so the gripper can end up pushing
  the box's corner. This is their reasoning, not a separate test. Checked, speculation.

Behavior Transformers (Section 2, Figure 2) and Pearce et al. (Section 2) make the same point: MSE equals fitting a single
Gaussian, whose best guess is the average (both checked, definition; details in section 5).

---

## 5. Bins, bins plus offsets, and mixtures in robot behaviour cloning

### 5.1 RT-1: 256 bins for each action number

Brohan et al. (2022/2023), *RT-1: Robotics Transformer for Real-World Control at Scale*, Robotics: Science and Systems XIX
(2023), DOI 10.15607/RSS.2023.XIX.025. Opened: https://ar5iv.labs.arxiv.org/html/2212.06817 and
https://roboticsproceedings.org/rss19/p025.html

- Eleven action numbers (seven for the arm, three for the base, one mode switch between arm, base and terminate), each cut
  into 256 bins spread evenly between that number's limits, trained with cross-entropy (Section 5.1). Data: about 130,000
  demonstrations, over 700 instructions, 13 robots, 17 months; the policy runs at 3 Hz (Sections 1 and 5.2). Checked,
  definition.
- Ablation (Appendix D.4, Table 13). Full RT-1: seen 97, unseen 76, distractors 83, backgrounds 59 (percent). With
  continuous actions (MSE loss with a normal-distribution output): 68, 43, 37 and 35. With autoregressive actions (each
  number conditioned on the previous ones): 85, 71, 67 and 65, and inference took 36 ms instead of 15 ms. Checked, measured.
- The authors explain the continuous result by multimodality: bins can represent several peaks, a Gaussian only one
  (Appendix D.5). This explanation is not isolated by a separate test. Checked, speculation.
- Note the scale: RT-1 has about 600 times more demonstrations than we do. Note also the mode switch with a terminate value,
  which is like our stop class.

**FAST** (Pertsch et al. 2025, *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, arXiv:2501.09747,
no venue on the page). Opened: https://arxiv.org/html/2501.09747

- In a synthetic curve-prediction test (Section IV, Figure 3), simple 256-bin tokens per time step had low error at low
  sampling rates but failed as the rate rose (25 to 800 time steps per sequence), the model ending up copying the first
  action. Checked, measured (synthetic task only).
- The authors blame strongly correlated consecutive tokens, which carry almost no new information each. Checked,
  speculation.
- Relevance: we take one decision per picture and per 1 cm step, a very low rate, so this failure mode should not apply to
  per-step classes (my judgement).

### 5.2 Behavior Transformers: k-means bins plus offsets

Shafiullah, Cui, Altanzaya and Pinto (2022), *Behavior Transformers: Cloning k modes with one stone*, NeurIPS 2022.
Opened: https://ar5iv.labs.arxiv.org/html/2206.11251 and https://mlanthology.org/neurips/2022/shafiullah2022neurips-behavior

- Method (Sections 2.1 to 2.3, Appendix B.2, Table 4): all demonstrated actions are clustered with k-means into k bins.
  Each action becomes a bin number plus an offset from that bin's centre. A transformer over recent observations predicts
  the bin with focal loss (cross-entropy that down-weights easy examples; the authors say it helps with unevenly filled
  bins) and an offset for every bin, but only the demonstrated bin's offset is trained (masked MSE). k was 32 (CARLA), 24
  (block push) and 64 (kitchen); history 10, 5 and 10 steps; datasets 100, 1,000 and 566 demonstrations. CARLA used
  224 x 224 pictures through a frozen ResNet-18; block push and kitchen used states. The authors note that discretising
  "invariably causes loss of fidelity", which the offsets repair. Checked, definition.
- Main results (Table 1), MLP with MSE (called RBC in the paper) against BeT: CARLA 0.98 against 0.98; block push, two blocks pushed, 0
  against 0.71; kitchen, four tasks done, 0 against 0.44. Checked, measured.
- Ablations (Section 3.5, Table 3), as a fraction of full BeT's success for CARLA, block push and kitchen: no offsets 0.94,
  0.95, 0.78; no binning 0.94, 0.25, 0.68; no history 0.65, 0.95, 0.88; uniform quantisation instead of k-means 0.90, 0.96,
  0.90; a transformer with an MDN head 0.30, 0.83, 0.86. The paper does not say how the uniform-bin variant was built.
  Checked, measured.
- Relevance: bins plus offsets is exactly "move name plus exact step". Fixed uniform bins scored close to k-means bins.

### 5.3 Mixture heads

**robomimic**: Mandlekar et al. (2021), *What Matters in Learning from Offline Human Demonstrations for Robot
Manipulation*, Conference on Robot Learning 2021, PMLR 164, pages 1678 to 1690. Opened:
https://ar5iv.labs.arxiv.org/html/2108.03298 and https://proceedings.mlr.press/v164/mandlekar22a.html

- Data (Section 3.2, Appendix B): proficient-human (PH) sets have 200 demonstrations by one experienced operator;
  multi-human (MH) sets have 300, from six operators (two better, two okay, two worse) giving 50 each. The paper says data
  from several people can differ in skill and in solution strategy (Section 2). Checked, definition.
- MH results were lower than PH despite 100 more demonstrations, which the authors think is probably due to suboptimal and
  multimodal data (Section 4.1). Checked, speculation.
- GMM against a plain output for the recurrent policy BC-RNN (Section 4.4 and Appendix Table 26; default GMM with 5
  modes). Success in percent, default then no GMM:
  - MH, states: Square 78.0 then 58.0; Transport 65.3 then 27.3 (the 58% relative drop quoted in Section 4.4).
  - MH, pictures: Square 76.7 then 61.3; Transport 42.0 then 41.0.
  - PH, states: Square 84.0 then 82.0; Transport 71.3 then 69.3.
  - PH, pictures: Square 82.0 then 84.0; Transport 72.0 then 74.0.
  Checked, measured. Cross-check: the HTML render left the picture rows' column alignment unclear, so I checked the other
  columns of the same rows against the percentage ranges in Section 4.4 (learning rate 35 to 63%, shallow encoder 25 to 62%,
  smaller RNN 3 to 58%); they match, so the no-GMM values above are the remaining column.
- 5, 10 and 100 GMM modes made no significant difference (Appendix D.2). At test time they set every mode's spread to 1e-4,
  that is, near-deterministic sampling (Appendix J); without this trick results were mostly a little lower (Table 21, for
  example Transport PH states 71.3 with it, 64.7 without; Square MH pictures 76.7 with, 68.0 without). Checked, measured.
- Relevance: robomimic's own summary is that a GMM policy appears uniformly better (Section 4.4), but the numbers show the
  gain is on multi-operator data; on single-operator data the plain output did as well.

**Implicit BC**: Florence et al. (2021), *Implicit Behavioral Cloning*, Conference on Robot Learning 2021, PMLR 164, pages
158 to 168. Opened: https://ar5iv.labs.arxiv.org/html/2109.00137 and https://proceedings.mlr.press/v164/florence22a.html

- Simulated block pushing, 2,000 scripted demonstrations (Table 3): single target from pixels, energy-based model 100%,
  MSE 87.0%, MDN 10.0%; multiple targets from states, 99.0%, 89.7% and 99.7%. Checked, measured.
- Real robot with human teleoperation, 95 to 502 demonstrations (Table 4), energy-based against MSE: push red then green
  85.0 against 35.0; red or green, multimodal, 88.3 against 55.0; insert blue 83.3 against 6.7. Checked, measured.
- Relevance: an MDN head trained from pixels can fail badly (10%) where MSE works; mixture heads need care with images.

**Rahmatizadeh, Abolghasemi, Bölöni and Levine (2018)**, *Vision-Based Multi-Task Manipulation for Inexpensive Robots Using
End-To-End Learning from Demonstration*, ICRA 2018, pages 3758 to 3765 (venue from the authors' listing found by search).
Opened: https://ar5iv.labs.arxiv.org/html/1707.02920

- A cheap six-axis Lynxmotion arm, one camera, 128 x 128 pictures, 398 to 909 demonstrations per task. The output is an
  MDN with 50 Gaussian components over joint angles, each joint conditioned on the ones before it. Removing that
  conditioning cut success from 76, 80, 88, 76 and 88% to 16, 20, 52, 64 and 20% on the five tasks (Table I). Checked,
  measured.
- Relevance: when several joint numbers are predicted together, how they depend on each other matters here. RT-1 found the
  opposite for its bins (autoregression did not help), so the evidence is mixed.

**Diffusion Policy**: Chi et al. (2023), *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, Robotics:
Science and Systems XIX, DOI 10.15607/RSS.2023.XIX.026. Opened: https://ar5iv.labs.arxiv.org/html/2303.04137 and
https://roboticsproceedings.org/rss19/p026.html

- Push-T (pushing a T-shaped block onto a target; 200 human demonstrations in simulation) is the closest benchmark to our
  task. From pictures (Table 2, best and average of the last 10 checkpoints): LSTM-GMM 0.69 and 0.54; IBC 0.75 and 0.64;
  Diffusion Policy (CNN) 0.91 and 0.84. LSTM-GMM fell from single-operator to multi-operator data: Square 0.82 to 0.64,
  Transport 0.88 to 0.44 (best checkpoint). Checked, measured.
- From states (Table 1), BeT on Push-T 0.79 and 0.70; on Transport MH BeT 0.21 and 0.06, LSTM-GMM 0.62 and 0.20,
  Diffusion Policy (CNN) 0.68 and 0.46. Checked, measured.
- Real Push-T (Table 6): LSTM-GMM 20% (position control) and 10% (velocity), IBC 0%, Diffusion Policy 95%. Checked,
  measured.
- Push-T can be solved by going round the T on either side. LSTM-GMM and IBC kept to one side; BeT did not commit to one
  side; Diffusion Policy used both and committed within each run (Figure 3, qualitative). Checked, measured.
- Single-step policies can overfit to pauses in the demonstrations (idle actions); BC-RNN and IBC often got stuck in the
  real experiments when pauses were not removed (Section 4.3). Checked, measured (qualitative report).
- Picking each step independently from a multimodal output can alternate between modes and jitter (Section 4.3). This is
  an argument, not a separate measurement. Checked, speculation.
- Cost: 10 denoising steps gave 0.1 s per decision on an NVIDIA RTX 3080 GPU (Section 3.4; Appendix A.4 says 16 steps for
  the real robot, the paper does not reconcile the two). No CPU timing is given. Checked, measured.

### 5.4 Human data against scripted data

**ACT**: Zhao, Kumar, Levine and Finn (2023), *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*,
Robotics: Science and Systems XIX, DOI 10.15607/RSS.2023.XIX.016. Opened: https://ar5iv.labs.arxiv.org/html/2304.13705 and
https://www.roboticsproceedings.org/rss19/p016.html

- Removing the CVAE (the part that lets the policy model several ways of acting) made almost no difference with scripted
  demonstrations but dropped success on human demonstrations from 35.3% to 2%, averaged over two simulated tasks with 50
  demonstrations each (Section VI-B, Figure 6c). Checked, measured. Caution: Table I gives full ACT 82% and 60% on the same
  human-data tasks, which does not match 35.3%; the paper does not explain the difference.
- The authors explain it by human demonstrations being noisy and multimodal, more random where precision matters less
  (Section IV-B). Checked, speculation.

### 5.5 Where MSE held up

**Pearce et al. (2023)**, *Imitating Human Behaviour with Diffusion Models*, ICLR 2023. Opened:
https://ar5iv.labs.arxiv.org/html/2301.10677 and https://mlanthology.org/iclr/2023/pearce2023iclr-imitating

- Same networks for all output types (Section 4). *Discretised* means 20 uniform bins per action number, each number
  independent; *K-Means* uses 64 clusters; *K-Means+Residual* is BeT-style (Section 4, Appendix B.2). Checked, definition.
- Kitchen, 566 demonstrations, states, share of runs finishing four or more tasks (Table 1). Transformer: MSE 0.69,
  Discretised 0.34, K-Means 0.00, K-Means+Residual 0.34, Diffusion-X 0.88. MLP: MSE 0.50, Discretised 0.18, K-Means+Residual
  0.23, Diffusion-X 0.77. Checked, measured.
- Counter-Strike from 280 x 150 pictures, one skilled player, 45,000 samples, game score (Table 3). ResNet-18: MSE 17.8,
  Discrete 14.7, K-Means+Residual 16.8, Diffusion-X 24.0 (human 36.5). Small CNN: MSE 6.9, Discrete 7.8, Diffusion-X 17.3.
  Checked, measured.
- MSE learns the average and can output an action between two modes; per-dimension bins learn each number separately and
  can give uncoordinated actions, while bins over the joint space need B to the power of the number of dimensions
  (Section 2). Checked, definition.
- Relevance: the ranking of MSE, bins and bins plus offsets changed between papers and tasks. Results depend on the
  implementation, so our own comparison is needed.

### 5.6 Soft labels for neighbouring bins

- Xu et al. smoothed each label over nearby bins with a Gaussian of sigma 0.5 (Section 5.2); TrailNet trained on soft labels
  (Section IV-A). Both checked, definition.
- **Farebrother et al. (2024)**, *Stop Regressing: Training Value Functions via Classification for Scalable Deep RL*, ICML
  2024, PMLR 235, pages 13049 to 13071. Opened: https://ar5iv.labs.arxiv.org/html/2403.03950 and
  https://mlanthology.org/icml/2024/farebrother2024icml-stop. For value functions in reinforcement learning (not behaviour
  cloning), cross-entropy over bins with Gaussian-smoothed targets (called HL-Gauss) beat MSE in most settings and degraded
  more gracefully when targets were noisy (Section 5.2.1). The paper does not test behaviour cloning. Checked, measured.

---

## 6. What applies to our set-up (my judgement unless a source is named)

Our case: two operators guiding by hand with motors off, replay of every joint, one picture before each step of about 1 cm
or 5 degrees, a move name (8 picture-frame directions, stop, small motion) and the exact step (tip forward and left,
forearm twist, rod lean, hand turn) for every picture, 208 scored demonstrations, about 18,500 pictures, a laptop CPU.

### 6.1 Where our data can have more than one mode

1. **Two operators' styles**: how far behind the puck the block is kept, how early a drift is corrected, whether a
   correction is a tip step or a forearm twist. Closest evidence: robomimic multi-operator sets (GMM mattered) and ACT's
   human data (CVAE mattered). Ours has two operators, robomimic six, so the effect may be smaller (judgement).
2. **Joint redundancy**: the same tip move can come from different mixes of twist, lean and hand turn. Closest evidence:
   Bishop's elbow-up and elbow-down arm. The team's memory notes that in the exact replay the rod leaned 16 to 22 degrees
   (median) and the forearm twisted a little at almost every step without a turn label (6.11.26 analysis in
   `claude_memory/project-recording-tool.md`; numbers from the team, not checked by me).
3. **Real decision points**: the crossing on the loop sheet, the gaps on the dashed loop, winning back a puck that has
   drifted, when to stop. Here two different answers can both be right, or the picture alone may be ambiguous (a history
   question, not only an output question; BeT's ablation without history lost 35% on CARLA).
4. **Boundary noise**: a line heading about 22 degrees from forward is labelled forward on one step and forward-left on the
   next. This is label noise rather than a real second mode. Soft labels address it (Xu, TrailNet, Farebrother).

### 6.2 What the fixed 8-direction label fixes, and what it does not

Fixes (supported by RT-1, BeT, Giusti, TrailNet): decoded by argmax, the classifier commits to one direction, so
forward-left and forward-right cannot be averaged into forward. Because each class is a direction in the picture plane,
forward and left are binned together, which avoids the per-number independence problem Pearce et al. describe.

Does not fix:

- **Detail**: the classes are 45 degrees apart. Calculated, not measured: the largest heading error from snapping to the
  nearest direction is 22.5 degrees; on a 1 cm step that is up to 1 cm x sin 22.5 degrees, about 0.38 cm sideways, against
  a 1 cm line and a 2.4 cm puck. Feedback at every picture can correct it, but probably as a zig-zag; this is what the exact
  step (BeT's offset) is for.
- **The joint numbers**: an MSE head on twist, lean and hand turn averages the two operators' joint habits (point 2 above).
- **Uneven classes**: stop occurs about once per episode. The team's 6.11 check of the exact replay found 13 to 41% of rows
  moving under 3 mm (team memory, not checked by me), so "small motion" may be common. BeT used focal loss for uneven bins.
  Diffusion Policy reports single-step policies getting stuck on pauses; a model that keeps choosing "small motion" could
  stall, which costs finishing and time.
- **Decoding by average**: Giusti and TrailNet turned class probabilities into an average steering value. If two distant
  classes are both likely, that average re-creates mode averaging. Argmax, or averaging only the winning class and its two
  neighbours, avoids it.
- **Switching between steps**: picking a different mode at consecutive pictures could zig-zag (Diffusion Policy's argument,
  not measured for a stop-and-go loop like ours).

### 6.3 A design that fits our labels, and a cheap comparison (judgement)

1. **Head A, move name**: one class per move name (10 as described in the task; the team memory lists 11 move names, so use
   whatever the final data contain), cross-entropy with class weights or focal loss; soft labels shared between the
   two nearest directions by the exact angle, wrapping round (direction 8 neighbours direction 1); stop and small motion as
   ordinary classes. Decode by argmax.
2. **Head B, exact step**: for each class, a correction from the class centre for tip forward and left, plus twist, lean and
   hand turn; trained with MSE only for the demonstrated class (BeT's masked loss). At run time use the offset of the argmax
   class.
3. **Baseline C**: plain MSE on the exact step with the same network.
4. Optional: a 5-component GMM on the exact step with near-deterministic decoding (robomimic). Diffusion did best in Pearce
   and Chi but needs many network passes per decision; I found no CPU timings, so its cost on our laptop is unknown.
5. Judge by closed-loop runs on held-out tracks, not offline accuracy: Eraqi, Michelmore and TrailNet all show offline and
   closed-loop rankings can differ. All three heads share one backbone, so training cost on the CPU is nearly the same.

---

## 7. Claim list

| id | claim (short) | source | where | status | kind |
|---|---|---|---|---|---|
| action_representation-1 | PilotNet outputs 1/r, trained with MSE | Bojarski et al. 2016 | Sec. 2, 4 | checked | definition |
| action_representation-2 | PilotNet: 72 h data, about 98% autonomy on road | Bojarski et al. 2016 | Sec. 3, 7.1, 7.2 | checked | measured |
| action_representation-3 | Three fixed trail classes, 85.2% vs humans 86.5/82.0%, yaw from P(TR) minus P(TL) | Giusti et al. 2016 | Sec. II, III-A, Table I | checked | measured |
| action_representation-4 | TrailNet 6 soft classes: 84% accuracy, 100% autonomy; Giusti net 79%, 80% | Smolyanskiy et al. 2017 | Table I | checked | measured |
| action_representation-5 | TrailNet chose classification for labelling ease and noise, untested | Smolyanskiy et al. 2017 | Sec. IV-A | checked | speculation |
| action_representation-6 | Binned driving actions with Gaussian label smoothing; data-driven bins best | Xu et al. 2017 | Sec. 3.4, 5.2, Table 3, Fig. 7 | checked | measured |
| action_representation-7 | Offline RMSE: regression 17.77/16.01, plain classes 18.70/17.84, sine 17.44/14.93 | Eraqi et al. 2017 | Table 2 | checked | measured |
| action_representation-8 | Classification (200 bins) crashed 0 to 1 per lap, regression about 2; not controlled | Michelmore et al. 2018 | Sec. IV-A, V-A | checked | measured |
| action_representation-9 | MSE gives the conditional average; average of correct targets may be wrong | Bishop 1994 | Abstract, Sec. 2 | checked | definition |
| action_representation-10 | Two-link arm: MSE error 0.0578, MDN 0.0053 | Bishop 1994 | Sec. 6, Table 1 | checked | measured |
| action_representation-11 | MSE vs MDN: 0%/0%, 85%/0% vs 95%/15%, 100%/95% | Rahmatizadeh et al. 2018 (AAAI) | unnumbered table | checked | measured |
| action_representation-12 | MSE averages two valid push orders, hits the corner | Rahmatizadeh et al. 2018 (AAAI) | same section | checked | speculation |
| action_representation-13 | RT-1: 11 numbers x 256 uniform bins, cross-entropy | Brohan et al. 2023 | Sec. 5.1 | checked | definition |
| action_representation-14 | RT-1 continuous actions: 97/76/83/59 to 68/43/37/35 | Brohan et al. 2023 | App. D.4, Table 13 | checked | measured |
| action_representation-15 | RT-1 authors: bins capture multimodality, Gaussian one mode | Brohan et al. 2023 | App. D.5 | checked | speculation |
| action_representation-16 | RT-1 autoregressive actions: 85/71/67/65, 36 ms vs 15 ms | Brohan et al. 2023 | Table 13 | checked | measured |
| action_representation-17 | BeT: k-means bins, focal loss, masked offset MSE; k 32/24/64 | Shafiullah et al. 2022 | Sec. 2, App. B.2, Table 4 | checked | definition |
| action_representation-18 | BeT vs MLP-MSE: block push 0.71 vs 0, kitchen 0.44 vs 0, CARLA 0.98 vs 0.98 | Shafiullah et al. 2022 | Table 1 | checked | measured |
| action_representation-19 | BeT ablations: no offsets 0.78 to 0.95; no binning 0.25 to 0.94; uniform 0.90 to 0.96; MDN 0.30 to 0.86 | Shafiullah et al. 2022 | Sec. 3.5, Table 3 | checked | measured |
| action_representation-20 | Kitchen: MSE 0.69, discretised 0.34, k-means+residual 0.34, diffusion 0.88 | Pearce et al. 2023 | Table 1 | checked | measured |
| action_representation-21 | CSGO pixels: MSE 17.8, discrete 14.7, k-means+residual 16.8, diffusion 24.0 | Pearce et al. 2023 | Table 3 | checked | measured |
| action_representation-22 | MSE averages modes; per-number bins uncoordinated; joint bins grow as B^d | Pearce et al. 2023 | Sec. 2 | checked | definition |
| action_representation-23 | robomimic PH: 200 demos, one operator; MH: 300 demos, six operators | Mandlekar et al. 2021 | Sec. 3.2, App. B | checked | definition |
| action_representation-24 | No GMM: MH Transport 65.3 to 27.3, Square 78.0 to 58.0; PH unchanged | Mandlekar et al. 2021 | Sec. 4.4, Table 26 | checked | measured |
| action_representation-25 | MH lower, probably from suboptimal, multimodal data (authors' view) | Mandlekar et al. 2021 | Sec. 4.1 | checked | speculation |
| action_representation-26 | 5/10/100 modes same; near-deterministic GMM decoding mostly better | Mandlekar et al. 2021 | App. D.2, App. J, Table 21 | checked | measured |
| action_representation-27 | Pixel pushing: EBM 100%, MSE 87%, MDN 10%; real pushing EBM 85 to 88% vs MSE 35 to 55% | Florence et al. 2021 | Tables 3, 4 | checked | measured |
| action_representation-28 | Push-T pictures: LSTM-GMM 0.69, IBC 0.75, Diffusion 0.91; LSTM-GMM PH to MH drops | Chi et al. 2023 | Table 2 | checked | measured |
| action_representation-29 | States: BeT Push-T 0.79; Transport MH BeT 0.21, LSTM-GMM 0.62 | Chi et al. 2023 | Table 1 | checked | measured |
| action_representation-30 | Push-T modes: GMM and IBC one side, BeT no commitment, diffusion commits | Chi et al. 2023 | Fig. 3 | checked | measured |
| action_representation-31 | Single-step policies get stuck on pauses (idle actions) | Chi et al. 2023 | Sec. 4.3 | checked | measured |
| action_representation-32 | Independent per-step sampling can alternate modes and jitter | Chi et al. 2023 | Sec. 4.3 | checked | speculation |
| action_representation-33 | ACT without CVAE: human data 35.3% to 2%, scripted unchanged | Zhao et al. 2023 | Sec. VI-B, Fig. 6c | checked | measured |
| action_representation-34 | ACT authors: human demos noisy and multimodal | Zhao et al. 2023 | Sec. IV-B | checked | speculation |
| action_representation-35 | Cheap arm, one camera, 50-component MDN; no joint conditioning cuts 76 to 88% to 16 to 64% | Rahmatizadeh et al. 2018 (ICRA) | Table I | checked | measured |
| action_representation-36 | Per-step 256-bin tokens fine at low rates, fail at high rates | Pertsch et al. 2025 | Sec. IV, Fig. 3 | checked | measured |
| action_representation-37 | Value functions: Gaussian-smoothed bins beat MSE, robust to noisy targets | Farebrother et al. 2024 | Sec. 5.2.1 | checked | measured |
| action_representation-38 | Snapping to 8 directions: up to 22.5 degrees, about 0.38 cm per 1 cm step | own calculation | this note, 6.2 | unchecked | speculation |
| action_representation-39 | Exact replay: lean 16 to 22 degrees median, twists at most steps, 13 to 41% rows under 3 mm | team memory, 6.11.26 | project-recording-tool.md | unchecked | measured |
| action_representation-40 | Diffusion Policy: 10 denoising steps, 0.1 s per decision on an RTX 3080 GPU; no CPU timing | Chi et al. 2023 | Sec. 3.4, App. A.4 | checked | measured |

## Conclusion for the project (at most 200 words)

Evidence: MSE learns the average action per picture (Bishop 1994); where two actions are valid, the average can fail
(Bishop's arm error 0.0578 against 0.0053 with a mixture; a push task 0% against 95%, Rahmatizadeh 2018). Multimodal
outputs mattered most with several human operators: robomimic without its GMM fell from 65.3 to 27.3% on six-operator
Transport, not on one-operator data; ACT without its CVAE fell from 35.3 to 2% on human data only; RT-1 with a Gaussian
instead of 256 bins fell from 97 to 68% (seen) and 76 to 43% (unseen). MSE is not always worse: on Pearce's kitchen task
it scored 0.69 against 0.34 for per-number bins and for BeT-style bins plus offsets, which beat MSE in BeT's own tests.
Coarse fixed classes have steered real robots (Giusti, TrailNet).

Judgement: two hand-guiding operators with free twist, lean and hand turn resemble robomimic's multi-operator data. Train
the move name as classes (argmax, class weights, soft neighbouring labels), add a per-class offset head for the exact step,
and compare with plain MSE on the same network, in closed loop on held-out tracks. Watch whether 'small motion' makes the
robot stall.

---

## 8. Not opened, and caveats

- Not opened: ALVINN (Pomerleau 1989, NIPS), which is often described as using a row of steering output units with a
  Gaussian-shaped target, and Pomerleau's 1991 paper on that output coding; the NeurIPS server refused the connection and the
  other copy needed a login. Not opened: reference [20] that both Eraqi et al. and Michelmore et al. cite for "classification
  often beats regression". Nothing above relies on these.
- Venues taken from listings rather than opened proceedings pages: Xu et al. (arXiv comment), TrailNet (arXiv comment
  says IROS 2017 submission), both Rahmatizadeh papers (search listings of AAAI and the authors' page).
- Table caveats: ACT's 35.3% does not match its Table I; Xu et al.'s metric sign is ambiguous; BeT does not say how its
  uniform-bin variant was built; Michelmore et al.'s comparison changes more than the output; Diffusion Policy gives two
  different numbers of denoising steps.
- Scale caveat: RT-1 (130,000 demonstrations) and Block push (1,000 scripted) are far from our 208 demonstrations; robomimic
  (200 to 300), Push-T (200), BeT CARLA (100) and ACT (50) are closer.
