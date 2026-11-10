# Interactive correction after a first model: DAgger and human-gated corrections

Key: interactive_correction. Written 9 November 2026 by a literature sub-agent for Team M (Emranul Hoque, Marshal Sebastine).

Question: after a first behaviour-cloning model, how should the team collect corrections (DAgger, HG-DAgger, EnsembleDAgger, ThriftyDAgger, Mandlekar et al. 2020, later work)? How many rounds and episodes are typical, which states should be collected, how should corrections be labelled when the original data came from hand guiding with exact replay, and what does it cost in robot time?

## 0. How to read this note

- Each claim has an id (interactive_correction-N) that matches the claims list in section 6.
- Checked: I read the statement in the source myself. Unchecked: I did not.
- Measured: the paper reports it from an experiment. Speculation: the authors argue or guess it. Definition: how a method works, or a theoretical result.
- "Calculated" lines use the team's own figures (208 demonstrations, about 18,500 pictures, about 1.2 s per step). "Judgement" lines are my opinion, not evidence.
- Rendering warning for anyone reading these papers in HTML: ar5iv and arxiv.org/html print numbers written in LaTeX maths twice. HG-DAgger's Table II reads "11" and "66" in ar5iv, while the PDF says 1 and 6. Sirius's HTML says a reaction time of "22 seconds", the PDF says 2 seconds. I therefore re-read every number that matters in the PDF (text extracted locally; the extracted text and the small extraction script are in `working/` next to this note). Where only the HTML could be read, the claim says so.

## 1. Short answer in plain language

- DAgger's idea: let the robot drive with its own model, collect the states it actually reaches (including its mistakes), get the correct action for those states from the expert, add them to the data and retrain. Repeat.
- With a human expert, labelling pictures afterwards without feeling the robot is awkward and was inconsistent in two studies (Laskey et al. 2017; Ross et al. 2013). Human-gated variants (HG-DAgger) let the human take over when the robot goes wrong and record only the human's recovery. This is the usual real-robot recipe.
- Typical real-robot effort: one to three rounds of corrections (five in HG-DAgger), each round about 20 to 33 percent of the original data, or 50 to 100 correction episodes. The first round often gives most of the gain.
- For this project, hand guiding plus exact replay is already the demonstration method, so a correction can be a short demonstration that starts from the state where the model went wrong, with the same two labels (move name and exact step).
- Almost no paper reports robot hours per round. Calculated for this project: a round of about 50 episodes costs roughly 3 to 5 robot hours (with guessed handling times, section 4.4).

## 2. The methods side by side

| Method | Who decides when the human helps | Which states get labels | What the policy is trained on | Real robot in the paper |
|---|---|---|---|---|
| DAgger (Ross et al. 2011) | nobody; the robot (or a random mix with the expert) drives | every visited state, labelled by the expert | all data aggregated, retrained each iteration | no (games; a human expert in Super Tux Kart) |
| HG-DAgger (Kelly et al. 2019) | the human: takes over when unsafe, hands back when safe | only the human's recovery stretches | initial data plus recovery stretches | yes, a test car |
| EnsembleDAgger (Menda et al. 2019) | a rule: robot acts only if its ensemble is confident and close to the expert | states where the expert acts | aggregated data | no (simulation) |
| ThriftyDAgger (Hoque et al. 2021) | the robot: asks when a state is novel or risky, within a budget | human-controlled stretches | policy on human stretches; a success predictor on all data | yes, da Vinci research kit, camera images |
| IWR (Mandlekar et al. 2020) | the human (remote, by smartphone) | intervention stretches, robot stretches kept separately | 50/50 batches of intervention and robot samples | no (simulation) |
| Sirius (Liu et al. 2023) | the human | demonstration, intervention, robot, pre-intervention classes | weighted behaviour cloning; pre-intervention weight zero | yes, Franka arm |
| CR-DAgger (Xu et al. 2025) | the human pushes a handle while the policy keeps running | correction deltas, plus zero-correction data | a residual policy on top of the base policy | yes, robot arm |

## 3. Paper by paper

### 3.1 DAgger: Ross, Gordon and Bagnell (2011)
A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011. Opened: https://arxiv.org/abs/1011.0686, https://arxiv.org/html/1011.0686v3, PDF https://arxiv.org/pdf/1011.0686 (text extracted).

- interactive_correction-1 (definition, checked). Section 3, Algorithm 3.1: at iteration i the robot runs a mix of expert and current policy (expert with probability beta_i); every visited state is labelled with the expert's action; the new data join all earlier data; the next policy is trained on everything. A parameter-free choice is beta = 1 in the first iteration and 0 afterwards. The best policy on validation is returned.
- interactive_correction-2 (definition, checked). Section 1 and Theorems 3.1 to 3.4: the number of iterations needed grows roughly linearly with the task horizon (more with finite samples). This is a worst-case bound, not a recipe.
- interactive_correction-3 (measured, checked). Section 5.1, Figure 2: in Super Tux Kart a human expert gave joystick steering labels for the game images; a linear controller on a 25 x 19 colour image at 5 Hz; one lap (about 1,000 data points) per iteration; 20 iterations. Behaviour cloning on expert laps did not improve with more data; DAgger almost never fell off after 5 iterations and never after 15; SMILe still fell about twice per lap after 20.
- interactive_correction-4 (measured, checked). Section 5.2: Super Mario Bros. with a planner as expert, 5,000 data points per iteration, 20 iterations; behaviour cloning stagnated (the authors attribute this to Mario getting stuck next to obstacles, a state the expert never shows); DAgger beat SMILe and SEARN.

### 3.2 HG-DAgger: Kelly, Sidrane, Driggs-Campbell and Kochenderfer (2019)
HG-DAgger: Interactive Imitation Learning with Human Experts. ICRA 2019, pp. 8077 to 8083 (venue as cited in Mandlekar et al. 2020, reference 18). Opened: PDF https://arxiv.org/pdf/1810.02890 (text extracted), https://ar5iv.labs.arxiv.org/html/1810.02890.

- interactive_correction-5 (definition, checked). Section II, Algorithm 1, Eq. 1 and 2: the novice drives until the human judges it unsafe; the human takes over, steers back to a safe region and hands back. The paper puts it this way: "Expert action labels are only collected and added to D during these recovery trajectories". On the test car the expert took over by turning an auxiliary steering wheel and handed back with a button (Section III-A2).
- interactive_correction-6 (speculation, checked). Sections I and IV-A: the authors argue that DAgger's random switching gives the human poor feedback and a perceived actuator lag (with beta = 0 no feedback at all), which can degrade labels; they hypothesise this caused DAgger's instability in later epochs.
- interactive_correction-7 (measured, checked). Section III-A1, footnotes 2 and 3: all methods started from behaviour cloning on 10,000 expert labels, then 5 epochs adding 2,000 labels each (the last HG-DAgger epoch needed fewer because the novice rarely needed help). Rollout counts are not reported. DAgger used beta = 0.85, decayed by 0.85 per epoch.
- interactive_correction-8 (measured, checked). Section IV-C, Table II (PDF): real car, first 5,000 test samples on five fixed obstacle layouts: behaviour cloning 1 collision and 6 road departures, DAgger 1 and 1, HG-DAgger 0 and 0. Steering closest to the human for HG-DAgger (Bhattacharyya 0.0834, against 0.1057 DAgger and 0.1173 behaviour cloning). The authors say the small test set limits statistical significance. ar5iv shows the counts as 11 and 66.
- interactive_correction-9 (measured, checked). Section IV-A, Figures 3 and 4: in simulation HG-DAgger learned faster and more stably per expert label than DAgger and behaviour cloning; curves only.
- Also: HG-DAgger learns a doubt threshold from an ensemble (mean of the last 25 percent of doubt values at intervention moments, Eq. 4), used only for evaluation.

### 3.3 EnsembleDAgger: Menda, Driggs-Campbell and Kochenderfer (2019)
EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning. IROS 2019 (arXiv comments). Opened: https://arxiv.org/abs/1807.08364, https://ar5iv.labs.arxiv.org/html/1807.08364 (HTML only).

- interactive_correction-10 (definition, checked). Section III: the novice is an ensemble; it may act only when the ensemble's spread (doubt) is below one threshold and its action is close enough to the expert's; otherwise the expert acts.
- interactive_correction-11 (measured, checked, HTML only). Section IV, Appendix A: simulation only (inverted pendulum, HalfCheetah); the doubt rule gave fewer failures than a discrepancy-only rule; numbers only in figures.
- interactive_correction-12 (speculation, checked). Section V: the method assumes unfamiliar states are risky, which can fail at bottleneck states; no method is given for choosing the thresholds.

### 3.4 ThriftyDAgger: Hoque, Balakrishna, Novoseller, Wilcox, Brown and Goldberg (2021)
ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning. CoRL 2021, PMLR 164, pp. 598 to 608. Opened: https://proceedings.mlr.press/v164/hoque22a.html, PDF https://proceedings.mlr.press/v164/hoque22a/hoque22a.pdf (text extracted), https://ar5iv.labs.arxiv.org/html/2109.08273 (appendix).

- interactive_correction-13 (definition, checked). Sections 4.1 to 4.5, Algorithm 1: robot-gated. The robot asks for help when a state is novel (spread of a 5-member ensemble) or risky (a learned estimate of the chance of finishing is low); thresholds follow from one user setting, the target share of steps with a request (0.01 in the experiments). Human-controlled steps train the policy; all steps train the success predictor; the policy is updated after each episode.
- interactive_correction-14 (measured, checked). Section 5.5, Table 3: physical cable routing, da Vinci Research Kit, 64 x 64 x 3 images, (x, y) moves. 25 offline demonstrations (1,381 state-action pairs) plus 1,500 interactive steps (at most 100 per episode). Autonomous success: behaviour cloning 0/15, HG-DAgger 10/15, ThriftyDAgger 12/15; 15/15 for both when a human may help. Per successful training episode: 1.4 to 1.6 interventions, 14 to 15 human actions, 46 to 56 robot actions (per-episode counts are over successful episodes only, Section 5.1).
- interactive_correction-15 (measured, checked). Section 5.3, Table 1: simulated peg insertion, 30 demonstrations (2,687 pairs) plus 10,000 interactive steps. Autonomous success: behaviour cloning 24/100, HG-DAgger 57/100, ThriftyDAgger 73/100; ThriftyDAgger used fewer interventions (0.88 against 1.49 per episode) but more human actions (43.6 against 20.3).
- interactive_correction-16 (measured, checked). Section 5.4, Table 2: user study, 10 people, 3 simulated robots: ThriftyDAgger 7.9 interventions against 10.6 for HG-DAgger, throughput 9.2 against 5.1. The authors guess that HG-DAgger was harder because people had to watch three robots (speculation). No robot hours are reported for the physical experiment.

### 3.5 IWR: Mandlekar, Xu, Martín-Martín, Zhu, Fei-Fei and Savarese (2020)
Human-in-the-Loop Imitation Learning using Remote Teleoperation. arXiv 2012.06733 (no venue on the arXiv page). Opened: https://arxiv.org/abs/2012.06733, PDF https://arxiv.org/pdf/2012.06733 (text extracted), https://ar5iv.arxiv.org/html/2012.06733.

- interactive_correction-17 (definition, checked). Section IV-C, Figure 3: Intervention Weighted Regression keeps the human's intervention samples and the robot's own samples in two datasets and draws equal-size batches from each, so interventions are up-weighted while robot samples act as a regulariser. Their HG-DAgger baseline keeps only intervention samples (Section V-B).
- interactive_correction-18 (measured, checked). Section V-C: 30 initial demonstrations per operator; 3 rounds; each round until intervention samples reached about 33 percent of the initial dataset; 50 evaluation rollouts per checkpoint; simulation only (Sawyer arm, robosuite).
- interactive_correction-19 (measured, checked). Tables I and II, success in percent. Threading (one operator): base 58.0; Full Demos 76.7; HG-DAgger 57.3, 62.7, 75.3 (round 1, round 2, final); IWR 84.0, 90.7, 87.3. Coffee Machine (three operators): base 52.0; Full Demos 64.9; HG-DAgger 70.2, 71.1, 69.6; IWR 79.6, 79.5, 87.5. Full Demos is matched on number of samples, not on human or robot time.
- interactive_correction-20 (measured, checked). Section V-D: one large round with the same number of intervention samples was no different on Threading and about 7 points worse on Coffee Machine than three rounds. The authors suggest tasks with more bottlenecks gain more from iterating (speculation).
- interactive_correction-21 (speculation, checked). Section I: they argue offline DAgger relabelling is often infeasible on real robots (a 30 s task at 20 Hz needs 600 labels per trajectory) and prone to wrong labels.

### 3.6 Sirius: Liu, Nasiriany, Zhang, Bao and Zhu (2023)
Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment. RSS XIX, 2023. Opened: https://roboticsproceedings.org/rss19/p005.html, PDF https://arxiv.org/pdf/2211.08416 (text extracted), https://arxiv.org/html/2211.08416.

- interactive_correction-22 (definition, checked). Sections IV-B and IV-D: every sample is tagged demonstration, intervention, robot or pre-intervention; weights re-balance the classes (interventions to half of the data, pre-intervention to zero). The pre-intervention window is 15 robot actions, from an observed human reaction time of about 2 seconds (PDF; the HTML prints "22").
- interactive_correction-23 (measured, checked). Sections V-A and V-B, Table V, abstract: 3 rounds, each until intervention samples reach one third of the initial demonstration samples; real tasks on a Franka arm with a SpaceMouse, 30 initial demonstrations each; real-robot evaluation with one seed and 32 trials on the best checkpoint; abstract reports 27 percent better than prior methods on real hardware; per-round real numbers only in Figure 5.

### 3.7 Small robot, ten students: Laskey, Chuck, Lee, Mahler, Krishnan, Jamieson, Dragan and Goldberg (2017)
Comparing Human-Centric and Robot-Centric Sampling for Robot Deep Learning from Demonstrations. ICRA 2017 (camera-ready file on the Goldberg lab site; arXiv v2 says submitted to ICRA 2017). Opened: https://goldberg.berkeley.edu/pubs/2017-laskey-RC_HC_ICRA_CAMERA_READY.pdf (text extracted), https://arxiv.org/pdf/1610.00850v2, https://arxiv.org/abs/1610.00850v2.

- interactive_correction-24 (measured, checked). Section IV-C, Figure 6: 10 students taught a Zymark robot to singulate objects from a webcam image (controls per step: base rotation within 1.5 degrees, arm extension within 1 cm, close to this project's steps). Human-centric: 60 teleoperated demonstrations. Robot-centric (DAgger-like): 20 demonstrations plus 2 iterations of 20 robot rollouts labelled afterwards on video at half speed with a mouse. Success on 30 held-out layouts: about 40 percent robot-centric against 60 percent human-centric. About 2 hours per person.
- interactive_correction-25 (measured, checked). Section IV-D, Table I, Figure 7: when 5 participants relabelled their own teleoperated runs afterwards, the labels correlated with what they had done only 0.60 (rotation) and 0.22 (translation); robot-centric data also gave higher held-out error.
- interactive_correction-26 (speculation, checked). Sections IV-D and conclusion: the authors conclude robot-centric labelling can lose the supervisor's intent and that human-centric collection may be preferable for expressive models with human supervisors.

### 3.8 Small robot, human labels from video: Ross, Melik-Barkhudarov, Shankar, Wendel, Dey, Bagnell and Hebert
Learning Monocular Reactive UAV Control in Cluttered Natural Environments. arXiv 1211.1690 (2012); usually cited as ICRA 2013, venue not shown on the pages I opened. Opened: https://arxiv.org/abs/1211.1690, https://ar5iv.labs.arxiv.org/html/1211.1690 (HTML only; the PDF was too large to fetch).

- interactive_correction-27 (measured, checked, HTML only). Sections III-C, IV-A, IV-C: DAgger on a small quadrotor with one cheap camera; after the first iteration the pilot labelled the drone's own flights with a joystick, partly offline on video replayed 3 times slower; 3 iterations indoors and 3 outdoors; without feedback the pilot often turned too much; indoors, after iteration 3 the pilot no longer needed to intervene.

### 3.9 Hand-guided corrections: CR-DAgger, Xu, Hou, Xin, Liu and Song (2025)
Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections. arXiv 2506.16685 v5, an extended version of the NeurIPS 2025 paper (the NeurIPS version lists four authors, per mlanthology). Opened: https://arxiv.org/abs/2506.16685, PDF https://arxiv.org/pdf/2506.16685v5 (text extracted), https://mlanthology.org/neurips/2025/xu2025neurips-compliant.

- interactive_correction-28 (measured, checked). Sections 3.1, 4.3, Finding 3, Figures 10 and 11: corrections given kinesthetically (a person pushes a handle on the arm) while the policy kept running gave 45 percent higher success on book flipping than take-over corrections (HG-DAgger style, on the same interface); take-over data showed larger, more variable speeds just after control switched and drifted further from the base policy's data. Base policies used 150 to 405 demonstrations; 50 to 100 correction episodes per method.
- interactive_correction-29 (measured, checked). Section 4.4, Findings 4 and 5: retraining from scratch on old plus correction data was stable but barely used the small correction set (book flipping minus 1.67 points averaged over collection methods; belt assembly plus 18.8); fine-tuning on corrections alone was unstable (minus 30 and minus 15.6 points).
- interactive_correction-30 (measured, checked). Section 4.5, Figures 12 and 13: one batch of 50 correction episodes was more stable than updating every 10 episodes (book flipping reached 100 percent with one batch, from a 40 percent base); two batches beat three on belt assembly; sampling more densely just after a correction starts helped, while also sampling the moments before it (mostly bad behaviour) lowered success.
- interactive_correction-31 (speculation, checked). Section 5: the authors recommend starting correction collection once the base policy has at least 10 to 20 percent success (experience, not a tested threshold).

### 3.10 Scale and robot time: BC-Z, Jang et al. (2021)
Jang, E., Irpan, A., Khansari, M., Kappler, D., Ebert, F., Lynch, C., Levine, S. and Finn, C. BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning. CoRL 2021, PMLR 164, pp. 991 to 1002. Opened: https://arxiv.org/abs/2202.02005, https://ar5iv.labs.arxiv.org/html/2202.02005 (HTML only; the PDF was too large).

- interactive_correction-32 (measured, checked, HTML only). Section 4, Section 6.3 Table 4, introduction, Appendix B.2: 11,108 expert demonstrations, then 14,769 HG-DAgger episodes over 16 iterations (VR teleoperation with an override switch); 125 hours of robot time, 12 robots, 7 operators, 5 months. At a fixed total amount of data, 50 percent manual plus 50 percent HG-DAgger gave 53 percent success against 27 percent manual only (one task), and 47 against 23 percent (8 tasks).

### 3.11 Small car, learning from the timing of interventions: EIL, Spencer et al. (2020)
Spencer, J., Choudhury, S., Barnes, M., Schmittle, M., Chiang, M., Ramadge, P. and Srinivasa, S. Learning from Interventions: Human-robot interaction as both explicit and implicit feedback. RSS XVI, 2020. Opened: https://roboticsproceedings.org/rss16/p055.html, PDF https://roboticsproceedings.org/rss16/p055.pdf (text extracted).

- interactive_correction-33 (definition, checked). Sections III to V: robot steps the human did not interrupt count as good enough, the last H robot steps before an intervention as bad, and the human's intervention actions as demonstrations; tested on a 1/10-scale MuSHR car (Jetson Nano) whose state was a localised pose with hand-made features, not camera images.
- interactive_correction-34 (measured, checked). Abstract (proceedings page): the car learned collision avoidance from scratch with a few hundred samples (about one minute) of expert control.

### 3.12 Alternatives that need little correction time
- interactive_correction-35 (measured, checked, abstract only). DART: Laskey, M., Lee, J., Fox, R., Dragan, A. and Goldberg, K. (2017). DART: Noise Injection for Robust Imitation Learning. CoRL 2017. https://arxiv.org/abs/1703.09327. Noise is injected while the human demonstrates, so demonstrations contain recoveries; on a Toyota HSR grasping in clutter, a 62 percent average improvement over behaviour cloning.
- interactive_correction-36 (measured, checked). Diffusion Meets DAgger: Zhang, X., Chang, M., Kumar, P. and Gupta, S. (2024). RSS 2024. https://arxiv.org/abs/2402.17768, https://arxiv.org/html/2402.17768v2. Wrist camera; the policy predicts a direction and the robot moves 1 cm (Section IV-A1), close to this project. Off-track views are synthesised with a diffusion model and labelled from a later frame (Section III-C). Pushing with 8 demonstrations: behaviour cloning 20 percent, DMD 80 percent (10 trials, Table II and abstract). No human-correction DAgger was run, and the diffusion model needs a GPU (compute not reported).

## 4. Synthesis

### 4.1 How many rounds and episodes are typical?

| Study | Initial data | Rounds | Size of each round | Platform |
|---|---|---|---|---|
| DAgger, Super Tux Kart | expert laps | 20 (good after 5) | 1 lap, about 1,000 labels | game |
| Ross et al., UAV | pilot demonstrations | 3 | not stated per round | small quadrotor |
| HG-DAgger | 10,000 labels | 5 | 2,000 labels (20 percent of initial) | car |
| Laskey et al. 2017 | 20 demonstrations | 2 | 20 rollouts | Zymark arm |
| Mandlekar et al. 2020 | 30 demonstrations | 3 | intervention samples about 33 percent of initial | simulation |
| Sirius | 30 demonstrations (real) | 3 | intervention samples one third of initial | Franka arm |
| ThriftyDAgger, cable routing | 25 demonstrations | update after every episode | 1,500 steps in total | da Vinci kit |
| CR-DAgger | 150 to 405 demonstrations | 1 to 3 batches | 50 episodes per batch | arm |
| BC-Z | 11,108 demonstrations | 16 | about 920 episodes per iteration (calculated: 14,769 / 16) | 12 robots |

- Evidence: real-robot studies mostly use 1 to 3 rounds (claims 18, 20, 23, 24, 27, 30); HG-DAgger used 5; larger counts come from games or a large industrial team.
- Evidence: the first round often brings most of the gain (IWR Threading 58.0 to 84.0 after round 1, claim 19); more rounds did not always help (claims 20, 30); small frequent updates were unstable (claim 30).

### 4.2 Which states to collect?
- Evidence: DAgger labels every visited state (claim 1); HG-DAgger and IWR take the states where the human sees trouble coming plus the path back to safety (claims 5, 17); ThriftyDAgger and EnsembleDAgger take states the model has not seen or from which it is unlikely to finish (claims 10, 13). Interventions cluster at bottlenecks, the few moments where precision matters (Mandlekar et al. Figure 3; ThriftyDAgger Figure 2).
- Evidence against wandering far: robot-centric data put the Zymark robot in odd states needing complex recoveries, which were harder to learn (claims 24 to 26).
- Judgement for this project: run the first model on the six sheets and on new practice tracks drawn to the exam rules (same size and line width), because the exam includes unseen tracks and DAgger's point is to collect where the model will be tested. Expect trouble at the loop's crossing, the dashed loop's gaps, tight curves, the start where the ink begins, the stop at the end, and "stall" states where the model keeps choosing stop or small motion. If the puck is far off the line, end the episode instead of recording a long recovery.

### 4.3 How to label corrections when demonstrations come from hand guiding with exact replay
Evidence:
- Taking over and driving the recovery oneself (claims 5, 8) is the standard real-robot recipe; labels given afterwards on recorded pictures were inconsistent (claims 25, 27), and Mandlekar et al. argue offline relabelling is error-prone (claim 21).
- Do not train on the robot's last steps before a takeover: Sirius gives them zero weight (claim 22), EIL treats them as bad (claim 33), CR-DAgger found sampling them lowered success (claim 30).
- Up-weight the corrections: plain retraining swamps a small correction set (claim 29); IWR's 50/50 batches beat plain HG-DAgger aggregation (claim 19).
- Keeping the robot's accepted steps is optional: IWR keeps them as a regulariser, HG-DAgger drops them (claim 17).
- Smooth kinesthetic corrections beat abrupt take-overs in CR-DAgger (claim 28), but there the policy ran continuously, unlike this project's stop-and-go loop.

Judgement (a proposal, untested):
1. The loop already pauses between steps (about 1.2 s), so the person watching can stop the model at any step. At a bad state: stop, motors off (FreeMotion), hand-guide a short recovery until the puck is back on the line and moving well, put the puck back where it was at the takeover (the tool's ghost overlay), then replay the guided path with the camera exactly as for a demonstration. The correction then carries the same move-name and exact-step labels as the demonstrations. A puck put back off its place makes the replayed step slightly wrong for the new picture (label noise); before the overlay the tool measured 8 to 21 mm errors on 9 of 52 pucks.
2. Tag every picture with its class (demonstration, correction, robot accepted, pre-takeover), round, model version and the model's own predicted move at the takeover picture.
3. Exclude the 1 to 3 model steps before each takeover (calculated: a 2 s reaction time is under 2 steps at 1.2 s per step).
4. Sample corrections at about half of each training batch (IWR style) and retrain on all data.
5. Offline relabelling of move names (classic DAgger) costs less robot time, and 10 move names may be easier than continuous mouse labels, but test it first: each student relabels about 200 of their own demonstration pictures blind; if agreement with the recorded move names is low, do not rely on it (this repeats Laskey et al.'s consistency check).
6. Human corrections are the team's own demonstrations, so they should fit the course rule. An automatic labeller (for example from puck tracking) would not be; ask the course staff before using one.

### 4.4 What does it cost in robot time?
Evidence:
- Robot hours per round are rarely reported. Exceptions: about 2 hours per participant for the whole Laskey et al. study (claim 24); 125 hours for 25,877 BC-Z episodes on 12 robots (claim 32); ThriftyDAgger gives wall-clock times only for the simulated user study.
- Comparisons are per sample, not per hour (Mandlekar et al. Full Demos, BC-Z Table 4), so "corrections beat extra demonstrations" is shown per picture, not per robot hour.
- Calculated from ThriftyDAgger Table 3: human actions were about 20 percent (HG-DAgger, 13.9 of 69.4) and 25 percent (ThriftyDAgger, 15.2 of 60.7) of training steps, and about 9 to 11 human actions per intervention.

Calculated for this project (guesses marked):
- One track is about 89 pictures (18,500 / 208), about 107 s of stepping at 1.2 s per step.
- The "a third of the initial data per round" convention would mean about 6,200 correction pictures per round: replay alone about 2.1 hours, and if corrections were 20 to 25 percent of steps the model would also take about 18,500 to 24,700 steps itself (6 to 8 hours). With guiding and puck resets, 8 to 10 hours or more per round: too much for this course.
- A round like ThriftyDAgger's 1,500 steps or CR-DAgger's 50 episodes: 50 model runs are about 1.5 hours; with 1.5 takeovers per run (as in ThriftyDAgger; assumed) and about 10 steps per correction (assumed), that is 75 corrections and about 750 correction pictures (about 4 percent of the current data); at a guessed 2 minutes per correction (motors off, guide, puck back, replay) about 2.5 hours more. Total about 4 hours per round (3 to 5 with other guesses), plus sheet changes and retraining on the laptop.
- Offline relabelling instead: the same 50 runs (about 1.5 robot hours) give about 4,450 pictures to label at a desk; at a guessed 2 s per picture, about 2.5 hours of human time and no extra robot time.

## 5. Conclusion for the project

Evidence: after a first model, human-gated corrections raised success on real robots: HG-DAgger beat behaviour cloning on a car (0 against 6 road departures, small test); with 1,500 extra steps ThriftyDAgger reached 12/15 where behaviour cloning got 0/15; at equal data, corrections beat extra demonstrations (BC-Z; up-weighted in Mandlekar et al., simulation). Real-robot studies mostly used 1 to 3 rounds (HG-DAgger 5), each 20 to 33 percent of the initial data or 50 episodes. Steps just before a takeover are best excluded; corrections need up-weighting or they are swamped. Labels typed afterwards were inconsistent (correlations 0.22 and 0.60) and lost to plain demonstrations in a ten-student study. Robot hours per round are rarely reported, and comparisons are per sample, not per hour.

Judgement: once the first model finishes some tracks, run one or two rounds of about 50 episodes on the six sheets and new practice tracks. At a failure, stop, hand-guide a short recovery and replay it with the existing tool, so corrections carry the demonstrations' labels; tag them, drop the 1 to 3 steps before takeover, and sample corrections at half of each batch. Calculated with guessed handling times: about 3 to 5 robot hours per round.

## 6. Claims list

| id | kind | status | short claim |
|---|---|---|---|
| interactive_correction-1 | definition | checked | DAgger: run policy, expert labels visited states, aggregate, retrain |
| interactive_correction-2 | definition | checked | DAgger theory: iterations grow about linearly with horizon |
| interactive_correction-3 | measured | checked | Super Tux Kart: human labels, 1 lap per iteration, good after 5 of 20 |
| interactive_correction-4 | measured | checked | Super Mario: 5,000 points per iteration, BC stagnates |
| interactive_correction-5 | definition | checked | HG-DAgger: human takes over, only recovery labels kept |
| interactive_correction-6 | speculation | checked | DAgger switching degrades human labels (perceived lag) |
| interactive_correction-7 | measured | checked | HG-DAgger: 10,000 BC labels, then 5 x 2,000 |
| interactive_correction-8 | measured | checked | HG-DAgger car: BC 1 and 6, DAgger 1 and 1, HG-DAgger 0 and 0 |
| interactive_correction-9 | measured | checked | HG-DAgger simulation: faster, more stable learning (figures) |
| interactive_correction-10 | definition | checked | EnsembleDAgger gating by doubt and discrepancy |
| interactive_correction-11 | measured | checked | EnsembleDAgger: simulation only, fewer failures with doubt rule |
| interactive_correction-12 | speculation | checked | EnsembleDAgger may fail at bottlenecks; no threshold method |
| interactive_correction-13 | definition | checked | ThriftyDAgger: novelty and risk gating within a budget |
| interactive_correction-14 | measured | checked | Cable routing: BC 0/15, HG-DAgger 10/15, Thrifty 12/15 |
| interactive_correction-15 | measured | checked | Peg insertion: BC 24, HG-DAgger 57, Thrifty 73 of 100 |
| interactive_correction-16 | measured | checked | User study: fewer interventions, higher throughput |
| interactive_correction-17 | definition | checked | IWR: 50/50 batches of intervention and robot samples |
| interactive_correction-18 | measured | checked | Mandlekar: 30 demos, 3 rounds of about 33 percent |
| interactive_correction-19 | measured | checked | Mandlekar Tables I and II results |
| interactive_correction-20 | measured | checked | One big round vs three: same or about 7 points worse |
| interactive_correction-21 | speculation | checked | Offline relabelling often infeasible (600 labels per 30 s) |
| interactive_correction-22 | definition | checked | Sirius: four classes; pre-intervention weight zero; 2 s window |
| interactive_correction-23 | measured | checked | Sirius: 3 rounds of one third; 30 real demos; plus 27 percent |
| interactive_correction-24 | measured | checked | Laskey: RC 40 vs HC 60 percent, 2 hours per person |
| interactive_correction-25 | measured | checked | Retroactive labels correlate 0.60 and 0.22 with own teleoperation |
| interactive_correction-26 | speculation | checked | RC loses intent; HC preferable with human supervisors |
| interactive_correction-27 | measured | checked | UAV: 3 iterations, video 3 times slower, over-turning |
| interactive_correction-28 | measured | checked | Kinesthetic on-policy corrections beat take-over by 45 percent |
| interactive_correction-29 | measured | checked | Retrain dilutes corrections; finetune unstable |
| interactive_correction-30 | measured | checked | Batch of 50 stable; 2 batches beat 3; sample after start |
| interactive_correction-31 | speculation | checked | Start corrections at 10 to 20 percent base success |
| interactive_correction-32 | measured | checked | BC-Z: 16 iterations, 125 robot hours, 53 vs 27 percent |
| interactive_correction-33 | definition | checked | EIL: non-intervened good, pre-intervention steps bad |
| interactive_correction-34 | measured | checked | EIL: learned from about one minute of expert control |
| interactive_correction-35 | measured | checked | DART: noise injection, 62 percent over BC (abstract) |
| interactive_correction-36 | measured | checked | DMD: synthetic views, 80 vs 20 percent with 8 demos |

## 7. Sources

Opened (all URLs above). Seen only as citations, not opened: SafeDAgger (Zhang and Cho 2017), LazyDAgger (Hoque et al. 2021), SMILe (Ross and Bagnell 2010), RoboCopilot (2025), HACTS (2025), the Duckietown imitation-learning paper by Lőrincz et al. (2022, search result only), the interactive imitation learning survey by Celemin et al. (2022). One search found no paper using DAgger or corrections on a Niryo Ned2 or a similar educational arm.
