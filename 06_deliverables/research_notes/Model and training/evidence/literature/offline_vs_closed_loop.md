# Which offline numbers predict closed-loop success? (key: offline_vs_closed_loop)

Literature note for Team M (Niryo Ned2, puck pushed along a black line, one forearm camera, behaviour cloning from
208 hand-guided demonstrations). Written on 9 November 2026 by a research sub-agent. Every paper below was opened
(URL given); the two 2018 and 2024 PDFs were also converted to text and read line by line
(`sources/codevilla2018.txt`, `sources/foster2024.txt` in this folder). "Checked" means I read the claim in the source
itself. "Measured" means the paper measured it; "speculation" means the authors' (or my) interpretation, advice or a
theoretical bound that was not measured on a real system; "definition" means a metric or protocol definition.
Sections marked **Judgement** are mine, not the papers'.

---

## 1. The short answer in plain words

* **Validation loss is the weakest predictor.** In robomimic the checkpoint with the lowest validation loss was 10 % to
  100 % worse than the best checkpoint found by running the robot (in simulation), and even worse than simply taking the
  last checkpoint. Two 2026 real-robot studies found validation loss uncorrelated with success.
* **Measures built on the actions themselves do better, especially "is the direction right?" measures.** In driving,
  a left/straight/right classification error and a "fraction of clearly wrong predictions" error correlated about
  0.65 with success, against 0.39 for mean squared error. In a 2026 robot study, validation action error correlated
  -0.93 (Spearman) with real success over 16 checkpoints.
* **Which pictures you validate on matters more than which formula you use.** Adding pictures of the expert
  recovering from disturbances lifted the correlation of plain MSE from 0.39 to 0.54, and adding side views
  (off-path pictures) to 0.77.
* **All offline numbers are good at spotting bad models and poor at ranking good ones.** Over all driving models the
  correlations were 0.56 to 0.83; among the better half, on the basic validation data, they fell to 0.21 to 0.65.
  Between training seeds on the same data, offline numbers said nothing about the real difference.
* **Some failures cannot be seen offline at all**: a policy that never starts moving, drifts slowly off the line, or
  loops. Offline tests are optimistic: low offline error did not prevent many closed-loop failures.
* So offline numbers can **shortlist**; only the robot can **decide**.

---

## 2. Codevilla, Lopez, Koltun and Dosovitskiy (2018), the core study

*On Offline Evaluation of Vision-based Driving Models*, ECCV 2018. Opened: https://arxiv.org/abs/1809.04843 (venue in
the arXiv comments), https://ar5iv.labs.arxiv.org/html/1809.04843 and the arXiv PDF (main paper plus supplement).

**Set-up.** 45 driving models trained in the CARLA simulator, varying the amount of data (0.2 to 80 hours), data
distribution (one or three cameras, with or without injected steering noise), depth, data balancing, loss and
regularisation. Each model was evaluated offline and online after 2K, 4K, ... 500K mini-batches, so each scatter point
is a checkpoint (Sec. 4.1, 4.2). Online: success rate of goal-directed navigation, average completion, km per
infraction (Sec. 3.3). Pearson correlation over all points of a plot; the main figures show the unseen town (Town 2,
unseen weather) and keep only the best 50 % of points by the offline metric (Sec. 4.3).

**Offline metrics compared (Sec. 3.3, Table 1).** MSE of steering; absolute error; speed-weighted absolute error;
cumulative speed-weighted error over T steps; quantised classification error (steering binned into left, straight,
right with a threshold); thresholded relative error (TRE: fraction of samples whose error is at least a fraction
alpha of the true value, alpha = 0.1).

**Results.**
* Same validation data (one forward camera, no noise), unseen town: |r| = 0.39 for MSE, 0.61 absolute error,
  0.57 speed-weighted, 0.61 cumulative over 64 steps, 0.65 quantised classification, 0.64 TRE (Sec. 4.3, Fig. 3).
  Speed weighting and accumulating over time did not help.
* Same metric (MSE), different validation data: 0.39 (central camera, no noise), 0.54 (with recovery-from-noise data),
  0.77 (three cameras, no noise) (Sec. 4.3, Fig. 2). The authors' reading: a good policy must also act correctly away
  from the expert's path, and "Proper validation data should therefore include examples of recovery from
  perturbations."
* Supplement (Sec. 3, Figs. 1 to 7): in the training town among the best half, correlations were weaker still (MSE
  0.21; other metrics 0.53 to 0.55; validation data variants 0.21, 0.45, 0.50). With all models included, the
  correlations rose to 0.75 to 0.83 (training town) and 0.56 to 0.75 (unseen town). Offline numbers separate bad from
  reasonable models much better than they rank reasonable ones.
* Case study (Sec. 4.3, Fig. 5): two models with similar prediction accuracy; one crashed in every trial, typically
  from wrong timing or direction in a turn. A short large error in a turn is hidden by an average.
* Model selection (Sec. 4.5, Table 3): TRE picked the best-driving model in 10 of 12 comparisons, MSE in 6 of 12.
* Online metrics among themselves (Fig. 4): success rate correlated 0.80 with average completion and 0.77 with km per
  infraction.
* Real-world BDDV data, offline only (Sec. 4.4, Table 2): overall accuracy of three models 78.0 %, 81.8 %, 83.3 %; on
  turns 32.4 %, 49.3 %, 50.7 %. The turn subset separated the models most.
* Authors' limits (Sec. 5): mostly simulated data; correlation coefficient as the only measure of relation; even the
  best offline metric is far from perfectly correlated with driving quality.

---

## 3. Mandlekar et al. (2021), robomimic

Mandlekar, Xu, Wong, Nasiriany, Wang, Kulkarni, Fei-Fei, Savarese, Zhu, Martin-Martin, *What Matters in Learning from
Offline Human Demonstrations for Robot Manipulation*, CoRL 2021 (PMLR 164:1678-1690). Opened:
https://proceedings.mlr.press/v164/mandlekar22a.html and https://ar5iv.labs.arxiv.org/html/2108.03298.

* Protocol (Sec. 3.3, App. B.2): every checkpoint run for 50 rollouts in simulation (image agents every 20 epochs);
  reported number is the maximum success over training, 3 seeds.
* Sec. 4.5, Fig. 4(a): choosing the checkpoint by lowest validation loss (supervised-learning habit) or the last
  checkpoint (offline-RL habit) gave 10 % to 100 % lower success than the best checkpoint. Authors: this motivates
  better offline evaluation metrics. Lesson L3 (Sec. 5): better offline policy selection matters for real robots.
* App. G, Table 29 (success %, lowest-validation-loss / last / best checkpoint): BC-RNN Square image 35.3 / 64.7 / 82.0;
  BC-RNN Transport image 0.0 / 58.7 / 72.0; BC Square low-dim 20.0 / 65.3 / 78.7. The last checkpoint beat the
  lowest-validation-loss checkpoint in all eight dataset and algorithm pairs of Table 29.
* App. G: the lowest validation loss came early (epochs 100 to 300), the best success much later; validation loss kept
  rising while success kept rising. Using 30 % of the data for validation did not fix it (Square: 2.7 % for the
  lowest-loss checkpoint against 80.7 % for the best).
* Real robot (Sec. 4.7): they reported the final checkpoint over 30 rollouts because real evaluation takes too long to
  test every checkpoint.

---

## 4. Later robot-learning work

**Li et al. (2024), SIMPLER.** Li, Hsu, Gu, Mees, Pertsch, Walke, Fu, Lunawat, Sieh, Kirmani, Levine, Wu, Finn, Su,
Vuong, Xiao, *Evaluating Real-World Robot Manipulation Policies in Simulation*, CoRL 2024 (PMLR 270:3705-3728).
Opened: https://arxiv.org/html/2405.05941 and https://proceedings.mlr.press/v270/li25c.html. Six policies (three RT-1
checkpoints, RT-1-X, RT-2-X, Octo-Base), three Google Robot tasks. Ranking by validation action MSE against real
success: mean Pearson r 0.31, MMRV 0.375; their simulator with visual matching: 0.92 and 0.056 (Sec. VI-B, Table I).
MMRV (Sec. III-B) is the mean over policies of the worst rank swap, weighted by the real success gap (0 is perfect).
Authors' conclusion: validation MSE is not a good proxy for real performance in imitation learning.

**Lin et al. (2025), data scaling laws.** Lin, Hu, Sheng, Wen, You, Gao, *Data Scaling Laws in Imitation Learning for
Robotic Manipulation*, ICLR 2025. Opened: https://arxiv.org/html/2410.18647 (v4) and
https://www.iclr.cc/virtual/2025/poster/28305. App. E.1, Fig. 17: validation action MSE on 30 held-out human
demonstrations per environment or object. As the number of training environment-object pairs grew, MSE tracked the
real score almost perfectly (Pearson -0.98, Spearman -1.00); for object generalisation only -0.73, with MSE rising at
16 objects. LoRA had lower MSE than full fine-tuning (0.0049 against 0.006) but a lower real score (0.72 against 0.90).
Authors: MSE is a quick debugging tool for obviously broken policies, not a substitute for real scoring; they evaluated
final checkpoints (App. C).

**Allshire et al. (2026), preprint.** Allshire, Singh, Singh, Rashid, Choi, McAllister, Yu, Chen, Huang, Abbeel, Chen,
Duan, Isola, Malik, Shentu, Shi, Wu, Kanazawa, *Scalable Behavior Cloning with Open Data, Training, and Evaluation*,
arXiv 2606.27375 (not peer reviewed). Opened: https://arxiv.org/html/2606.27375v1. Sec. 3.4, Fig. 8, 16 checkpoints of
diffusion-transformer and VLA policies, three real tasks: validation action error against strict success Spearman
-0.93 (Pearson -0.89); training loss -0.83; validation loss 0.00 (not significant). Fig. 7: validation loss fell, then
rose while real performance kept improving. Advice: use action error and training loss for small design decisions,
the robot for major comparisons.

**Cowling (2026), preprint, low-cost arm.** Cowling, *Does Validation Loss Predict Real-Robot Success? Evidence from Four
Imitation-Learning Policies on a Low-Cost Arm*, arXiv 2611.00283 (30 October 2026, single author, not peer reviewed).
Opened: https://arxiv.org/abs/2611.00283 and https://arxiv.org/html/2611.00283. SO-101 arm, pick-and-place, ACT,
Diffusion Policy, SmolVLA, pi0.5; 82 checkpoints scored on 25 held-out demonstrations against 930 bench rollouts.
* Within one policy across data budgets, hold-out action error correlated with success (Spearman -0.23 to -0.88), but
  no more than the number of training episodes alone explains; within one budget the link largely vanished; between
  SmolVLA seeds on identical data hold-out error differed about 1 % while bench success differed, mainly in how often
  the arm started (Abstract, Table 3, Sec. 5.7).
* Across policy families the ranking inverted (Spearman +0.90 at 200 episodes); the lowest-hold-out-error policy
  succeeded 3 of 110 times, mostly because the arm never left its rest pose (96 % of its rollouts), a failure that a
  held-out set of demonstrations cannot contain (Sec. 5.3, 5.4).
* Hold-out loss of diffusion and flow policies rose two- to five-fold during training while action error fell by up
  to 29 % (Abstract, Sec. 6).
* Ten rollouts per checkpoint carry about plus or minus 15 percentage points; report "never started" separately from
  other failures; use action error, not loss, for stopping training; use the robot to choose seeds, converged
  checkpoints and model families (Sec. 8.5, 9).

**Abbas et al. (2026), preprint, dexterous hand.** Abbas, Jain, Viswanath et al., *From Teleoperation to Long-Horizon
Dexterity: Evaluating Visuomotor Skills and Offline Proxies for Dexterous Manipulation*, arXiv 2610.40828 (not peer
reviewed). Opened: https://arxiv.org/html/2610.40828. 281 policies, about 1,700 real trials, 67 offline metrics (Sec. 4,
App. F). Best validation loss uncorrelated with insertion success over 21 checkpoints (r = +0.08, p = 0.72). Only five
metrics passed their reliability checks, all about smoothness and directness of the predicted path (r -0.70 to -0.80),
and only within one policy family; nothing tracked the grasp rate. Authors: offline metrics can shortlist but not
replace real-robot evaluation.

**Chi et al. (2023), Diffusion Policy.** Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song, *Diffusion Policy:
Visuomotor Policy Learning via Action Diffusion*, RSS 2023 (extended journal version on arXiv). Opened:
https://arxiv.org/abs/2303.04137 and https://ar5iv.labs.arxiv.org/html/2303.04137. Simulation results reported as the
best checkpoint and as the average of the last 10 checkpoints (Sec. 5.2); real-robot experiments used the last
checkpoint after a fixed training budget (App. C.1.2, C.2.2). For implicit BC, training loss fell smoothly while
success oscillated, which made checkpoint choice hard (Sec. 4.4, Fig. 6).

**Kim et al. (2024), OpenVLA.** Kim, Pertsch, Karamcheti, Xiao, Balakrishna, Nair et al., *OpenVLA: An Open-Source
Vision-Language-Action Model*, CoRL 2024 (PMLR 270:2679-2713). Opened: https://arxiv.org/html/2406.09246 and
https://proceedings.mlr.press/v270/kim25c.html. Real-robot performance kept improving with more epochs, so they trained
until training action-token accuracy passed 95 % (Sec. 3.4). An 8-bit model had about the same offline token accuracy
(Sec. 5.4 footnote) but lower real success (58.1 % against 71.3 %, App. B.3 Table 5) because slower inference changed
the timing under non-blocking control; with blocking (stop-and-go) control all precisions performed alike (App. D.4).

**Foster, Block and Misra (2024).** *Is Behavior Cloning All You Need? Understanding Horizon in Imitation Learning*,
NeurIPS 2024. Opened: https://arxiv.org/abs/2407.15007 and the arXiv PDF (read as text). Theory: behaviour cloning
with log loss can reach horizon-independent sample complexity under stated conditions (Abstract, Sec. 2). Experiment
(Sec. 5.2, Fig. 6, one Walker2d training run, H = n = 500): with log-loss training, validation loss and Hellinger
distance tracked rollout reward closely, "acting as excellent proxies for rollout performance"; with MSE training,
validation error was less well correlated (policies with similar validation loss but different reward).

**Irpan et al. (2019), off-policy classification (reinforcement learning, not BC).** Irpan, Rao, Bousmalis, Harris,
Ibarz, Levine, *Off-Policy Evaluation via Off-Policy Classification*, NeurIPS 2019. Opened:
https://arxiv.org/abs/1906.01624 and https://ar5iv.labs.arxiv.org/html/1906.01624. 15 vision-based grasping models on a
real robot (17 % to 91 % success): SoftOPC, a classification score that uses success and failure labels, correlated
Spearman 0.94 (R2 0.91) with real success; TD error 0.48 (R2 0.17) (Sec. 6.2, Table 2). Shows that an offline score
built on outcome labels from the deployment setting can rank policies well.

**Li et al. (2025), WorldEval.** Li, Zhu, Wen, Shen, Xu, *WorldEval: World Model as Real-World Robot Policies
Evaluator*, arXiv 2505.19017 (no venue stated). Opened: https://arxiv.org/html/2505.19017. A video world model scored
four policies on three tasks with Pearson 0.94 and MMRV 0.044 against real success (Table 1), but it is a 14-billion
parameter video model fine-tuned on 8 H800 GPUs. Out of reach for a laptop CPU.

---

## 5. More evidence from driving and testing

**Haq, Shin, Nejati, Briand (2021).** *Can Offline Testing of Deep Neural Networks Replace Their Online Testing? A Case
Study of Automated Driving Systems*, Empirical Software Engineering 26 (2021). Opened:
https://ar5iv.labs.arxiv.org/html/2101.11118. Three steering networks, 90 simulated scenarios each (Sec. 3.3, Fig. 9,
Table 2): offline and online verdicts disagreed in 45.5 % to 63.3 % of scenarios (Spearman below 0.5). Large offline
errors almost always meant online lane-keeping violations (36/37, 15/15, 23/24), but 40 to 57 scenarios with small
offline errors still failed online. The authors blame small errors that accumulate in closed loop (Sec. 3.3.2).

**Dauner, Hallgarten, Geiger, Chitta (2023).** *Parting with Misconceptions about Learning-based Vehicle Motion
Planning*, CoRL 2023. Opened: https://arxiv.org/abs/2306.07962 and https://ar5iv.labs.arxiv.org/html/2306.07962. In
nuPlan the open-loop score and the closed-loop score traded off: a planner with open-loop score 86 had closed-loop 54,
another with open-loop 42 had closed-loop 92 (Sec. 3.2, Tables 1 and 2). They discourage open-loop metrics as the main
indicator (Sec. 5).

**Dauner et al. (2024), NAVSIM.** *NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking*,
NeurIPS 2024 Datasets and Benchmarks. Opened: https://arxiv.org/abs/2406.15349 and https://arxiv.org/html/2406.15349.
Over 37 rule-based and 114 learned planners, a short non-reactive simulation score correlated better with closed-loop
score than the displacement-based open-loop score (Sec. 4.1, Figs. 3 and 4; the coefficients are in the figures and
were not read).

---

## 6. Why offline numbers miss closed-loop failures

* **Compounding errors.** Ross, Gordon, Bagnell (2011), *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning*, AISTATS 2011 (PMLR 15:627-635). Opened: https://proceedings.mlr.press/v15/ross11a.html and
  https://ar5iv.labs.arxiv.org/html/1011.0686. Theorem 2.1 (Sec. 2.1): a cloned policy with per-step error epsilon on
  the expert's states can cost up to T squared times epsilon more than the expert over T steps, because one mistake
  leads to states the demonstrations never showed. With about 90 steps per episode here, a small per-picture error rate
  can still end many episodes.
* **Shortcuts (causal confusion).** de Haan, Jayaraman, Levine (2019), *Causal Confusion in Imitation Learning*,
  NeurIPS 2019. Opened: https://arxiv.org/abs/1905.11979 and https://ar5iv.labs.arxiv.org/html/1905.11979. Giving the
  policy more information (the previous action) lowered validation loss but lowered reward (Sec. 3.2, Fig. 4; App. E,
  Fig. 11). Their Table 1, reproduced from Wang et al. (2019, not opened): a driving model with history had better
  validation perplexity (0.834 against 0.989) but more interventions (2.94 against 1.30) and collisions (6.49 against
  3.38).
* **Seed luck.** Codevilla, Santana, Lopez, Gaidon (2019), *Exploring the Limitations of Behavior Cloning for Autonomous
  Driving*, ICCV 2019. Opened: https://ar5iv.labs.arxiv.org/html/1904.08980 and
  https://mlanthology.org/iccv/2019/codevilla2019iccv-exploring/. The same model trained with 12 seeds varied widely in
  closed-loop success (table values up to 42 % in dense traffic, labelled "variance"); ImageNet initialisation reduced
  the spread (Sec. 5.3, Table 3). Cowling (2026) found hold-out metrics blind to such seed differences.
* **Failures with no offline counterpart.** Never starting (Cowling 2026), timing effects under non-blocking control
  (OpenVLA), and slow drift (Haq 2021).

---

## 7. How to run and read a robot test

**Kress-Gazit et al. (2024).** Kress-Gazit, Hashimoto, Kuppuswamy, Shah, Horgan, Richardson, Feng, Burchfiel, *Robot
Learning as an Empirical Science: Best Practices for Policy Evaluation*, arXiv 2409.09491 (no venue). Opened:
https://arxiv.org/html/2409.09491v1. Report trial counts, not only percentages (Sec. 4.1); use intervals or Bayesian
estimates (Sec. 4.2); match and document start conditions, interleave policies so the evaluator does not know which
runs (Sec. 2.2, 2.3); fix success criteria in advance; add progress rubrics, completion time and failure descriptions
(Sec. 3, 4.3). Worked examples: 15/18 against 11/17 gives only about an 11 % posterior chance that the second policy is
better (Sec. 4.2); 13/20 against 14/20 cannot name a winner (App. 6.4).

**Snyder et al. (2025), STEP.** Snyder, Hancock, Badithela, Dixon, Miller, Ambrus, Majumdar, Itkina, Nishimura, *Is Your
Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping*, RSS 2025. Opened:
https://arxiv.org/abs/2503.10966 and https://arxiv.org/html/2503.10966. A sequential test that may stop early with
controlled error: 19 trials sufficed for 0.56 against 0.92, 8 trials for 0.28 against 0.80 (Table I), but 59 % against
68 % would need about 500 (Sec. V-C2). Re-running an ordinary test after every new trial inflates false positives
(Sec. V-B1, Fig. 3). A non-significant result is not evidence that two policies are equal.

**Calculated (mine, normal approximation, not from a source).** With 10 runs at a true 50 % success the standard error
is about 16 points. To tell 60 % from 90 % with 80 % power at the 5 % level needs about 32 runs per model; 70 % from 90 %
about 62. At about 90 steps of 1.2 s, one run takes roughly 2 minutes plus the reset.

---

## 8. What this means for Team M (Judgement, built on the evidence above)

The project's code (read, not changed) already scores leave-one-sheet-out folds, angle errors, the share within
30/45/60 degrees, class and balanced class accuracy, stop precision and recall, episode-level stop outcomes, and two
reference rules ("always_forward", "repeat_previous_move") (`02_code/training_metrics.py`, `02_code/training_probes.py`).
That already follows the best evidence (thresholded and class-type measures over mean errors; held-out conditions).

**What to measure offline**
1. Keep the held-out-sheet score as the main number for unseen tracks (Codevilla's unseen-town analysis; Lin et al.'s
   held-out environments). Six folds of different shapes: treat differences of a few points as noise.
2. Rank by thresholded or class-type measures (within-45-degree share, balanced accuracy, stop outcomes per episode),
   not by validation loss or mean error. Look at the worst sheet and the worst segment, not only the mean.
3. Add per-segment scores: curves, the crossing, dashed gaps, the line end. Codevilla's turns were the most telling
   subset, and crashes came from short errors in turns.
4. Build a small **recovery validation set**: a few episodes where the puck starts off the line or the rod leans
   wrongly and the demonstrator brings it back, labelled the usual way and never trained on. This was the single
   biggest lever in Codevilla (0.39 to 0.54 or 0.77).
5. Report each model's gain over "always_forward" and "repeat_previous_move". Treat offline gains from picture history
   with suspicion until the robot confirms them (causal confusion).
6. Log validation loss but do not select on it. `train_head` early-stops on validation loss; for linear heads on
   frozen features this is probably harmless, but compare once on the robot against the last epoch.

**How to pick checkpoints**
* Fix a training budget and take the last checkpoint as the default (robomimic: last beat lowest-loss in every row;
  Diffusion Policy and Lin et al. used last checkpoints on real robots). If the held-out-sheet score clearly peaks
  earlier, keep that one too as a second candidate.
* Use offline numbers to throw out clearly worse models and to shortlist two or three candidates (for example the
  last checkpoint, the best held-out-sheet score, and a different design). Do not spend robot time on candidates that
  differ by a few offline points; seeds and near-equal checkpoints cannot be ranked offline.

**What the robot test must decide**
* Go or no-go per candidate: does it finish each of the six sheets and some new tracks of the exam's kind? Record
  finish (yes/no), progress along the line when it fails (lower noise than finish alone; Codevilla found completion and
  success strongly related), number of steps and time, and the failure type: left the line, stuck or oscillating
  between moves, stopped early, never stopped, puck lost from the magnet, never got going.
* Choose between candidates only when the gap is large; with 10 runs each, only gaps of roughly 30 points or more are
  trustworthy. Use the same start positions, interleave the candidates, fix the number of runs in advance (or use a
  sequential test), and do not stop as soon as one looks ahead.
* Calibrate the offline numbers: write each tested candidate's offline scores next to its robot results. After a few
  candidates the team learns which offline number tracks its own robot.
* Turn failures into data: the states where the robot failed can be demonstrated by hand and added (the DAgger idea of
  Ross et al. 2011; still imitation learning from the team's own demonstrations).

---

### Conclusion for the project (at most 200 words)

**Evidence:** no offline number reliably ranks good checkpoints. Validation loss is weakest: in robomimic the
lowest-loss checkpoint was 10 to 100 % worse than the best, and two 2026 robot studies found it uncorrelated.
Action-based measures do better (Codevilla: classification or thresholded error about 0.65 against 0.39 for MSE; a
2026 study: validation action error Spearman -0.93 over 16 checkpoints), but correlations shrink among good models and
vanish between seeds. Validation pictures off the demonstrated path raised correlation most (0.39 to 0.77). Offline
tests miss closed-loop failures (Haq: 40 to 57 of 90 low-error scenarios still failed; a policy that never starts
looks fine offline).

**Judgement:** keep the held-out-sheet split and the existing within-45-degree, balanced-accuracy and stop measures;
add per-segment scores and a small recovery validation set; never select on validation loss; distrust gains from
picture history until the robot confirms them. Default to the last checkpoint of a fixed budget, shortlist two or three
candidates offline, then let the robot decide: same starts, interleaved runs, counting finishes, progress, failure type
and time, on the six sheets and unseen tracks. Ten runs give about plus or minus 15 points, so only large differences
are decidable.

---

## 9. Claim register (same ids as the structured result)

Format: id (kind, status; source, location): claim. All "checked" claims were read in the source itself.

* -1 (measured, checked; Codevilla 2018, Sec. 4.3, Figs. 2 and 3): basic validation data (one forward camera, no
  noise), unseen town, best half of checkpoints: steering MSE correlates with success only |r| = 0.39.
* -2 (measured, checked; Codevilla 2018, Sec. 4.3, Fig. 3): on the same data, absolute error 0.61, speed-weighted 0.57,
  cumulative over 64 steps 0.61, quantised classification 0.65, thresholded relative error 0.64.
* -3 (measured, checked; Codevilla 2018, Sec. 4.3, Fig. 2): MSE on validation data with recovery-from-noise 0.54, with
  three cameras 0.77.
* -4 (speculation, checked; Codevilla 2018, Sec. 4.3): authors conclude a policy must act well away from the expert's
  path, so validation data should contain recoveries.
* -5 (measured, checked; Codevilla 2018, supplement Sec. 3, Figs. 1, 2, 4 to 7): training town best half 0.21 to 0.55;
  all models 0.75 to 0.83 (training town) and 0.56 to 0.75 (unseen town).
* -6 (measured, checked; Codevilla 2018, Sec. 4.3, Fig. 5): two models with similar accuracy; one crashed in every trial
  from wrong timing or direction in turns.
* -7 (measured, checked; Codevilla 2018, Sec. 4.5, Table 3): TRE picked the best-driving model in 10 of 12 cases, MSE in
  6 of 12.
* -8 (measured, checked; Codevilla 2018, Sec. 4.3, Fig. 4): success rate correlates 0.80 with average completion and 0.77
  with km per infraction.
* -9 (measured, checked; Codevilla 2018, Sec. 4.4, Table 2): on real BDDV data, turn accuracy (32.4 % to 50.7 %)
  separated models more than overall accuracy (78.0 % to 83.3 %); no online test.
* -10 (definition, checked; Codevilla 2018, Sec. 3.3, Table 1): definitions of the six offline metrics, including
  quantised classification error and thresholded relative error.
* -11 (measured, checked; Mandlekar 2021, Sec. 4.5, Fig. 4(a)): lowest-validation-loss or last checkpoint gave 10 % to
  100 % lower success than the best checkpoint.
* -12 (measured, checked; Mandlekar 2021, App. G, Table 29): for example BC-RNN Square image 35.3 / 64.7 / 82.0 % (lowest
  loss / last / best); last beat lowest loss in all eight pairs.
* -13 (measured, checked; Mandlekar 2021, App. G): lowest validation loss at epochs 100 to 300, best success later;
  validation loss rose while success rose; 30 % validation split did not help (2.7 % against 80.7 %).
* -14 (definition, checked; Mandlekar 2021, Sec. 3.3, App. B.2, Sec. 4.7): 50 rollouts per checkpoint, maximum
  reported; real robot: final checkpoint, 30 rollouts, because real evaluation is slow.
* -15 (speculation, checked; Mandlekar 2021, Sec. 4.5 and Sec. 5 L3): better offline policy selection is needed for real
  robots.
* -16 (measured, checked; Li 2024 SIMPLER, Sec. VI-B, Table I): validation MSE ranks six policies poorly (Pearson 0.31,
  MMRV 0.375) against simulation with visual matching (0.92, 0.056).
* -17 (definition, checked; Li 2024 SIMPLER, Sec. III-B): MMRV definition.
* -18 (measured, checked; Lin 2025, App. E.1, Fig. 17): held-out MSE tracked real score across environment-object
  diversity (Pearson -0.98, Spearman -1.00) but only -0.73 for object generalisation.
* -19 (measured, checked; Lin 2025, App. E.1): LoRA lower MSE (0.0049 against 0.006) but lower real score (0.72 against
  0.90).
* -20 (speculation, checked; Lin 2025, App. E.1 and App. C): MSE only a debugging tool; final checkpoints evaluated.
* -21 (measured, checked; Allshire 2026, Sec. 3.4, Fig. 8): over 16 checkpoints, validation action error Spearman -0.93
  with real success, training loss -0.83, validation loss 0.00.
* -22 (measured, checked; Allshire 2026, Sec. 3.4, Fig. 7): validation loss fell then rose while real performance
  improved.
* -23 (measured, checked; Cowling 2026, Abstract, Table 3, Sec. 5.7): within-policy correlation (-0.23 to -0.88) no
  stronger than episode count alone; vanishes within one budget; blind to seed differences.
* -24 (measured, checked; Cowling 2026, Sec. 5.3, 5.4): ranking inverted across families (+0.90); lowest-error policy 3
  of 110, mostly never leaving rest.
* -25 (measured, checked; Cowling 2026, Abstract, Sec. 6): hold-out loss of diffusion and flow policies rose two- to
  five-fold while action error fell up to 29 %.
* -26 (speculation, checked; Cowling 2026, Sec. 8.5, 9): about plus or minus 15 points per 10 rollouts; report
  initiation failures separately; robot decides seeds, checkpoints, families.
* -27 (measured, checked; Abbas 2026, Sec. 4, App. F): validation loss uncorrelated (r = +0.08, p = 0.72, 21
  checkpoints); only five smoothness metrics passed (r -0.70 to -0.80), within one family.
* -28 (speculation, checked; Abbas 2026, Sec. 4 and conclusion): offline metrics can shortlist but not replace robot
  evaluation.
* -29 (definition, checked; Chi 2023, Sec. 5.2, App. C.1.2 and C.2.2): best and last-10 average reported; real robot used
  last checkpoints.
* -30 (measured, checked; Chi 2023, Sec. 4.4, Fig. 6): implicit BC training loss fell smoothly while success oscillated.
* -31 (measured, checked; Kim 2024 OpenVLA, Sec. 3.4): real performance kept improving with epochs; trained until
  training token accuracy passed 95 %.
* -32 (measured, checked; Kim 2024 OpenVLA, Sec. 5.4 footnote, App. B.3 Table 5, App. D.4): int8 similar offline
  accuracy, lower real success (58.1 % against 71.3 %) under non-blocking control; equal under blocking control.
* -33 (measured, checked; Foster 2024, Sec. 5.2, Fig. 6): one Walker2d run: log-loss validation tracks reward; MSE
  validation less so.
* -34 (speculation, checked; Foster 2024, Abstract, Sec. 2): theory: log-loss BC can be horizon-independent under
  conditions (a proof, not a measurement).
* -35 (measured, checked; Irpan 2019, Sec. 6.2, Table 2): 15 real grasping models: SoftOPC Spearman 0.94, TD error 0.48.
* -36 (measured, checked; Li 2025 WorldEval, Table 1 and training details): Pearson 0.94, MMRV 0.044; 14B video model on
  8 H800 GPUs.
* -37 (measured, checked; Haq 2021, Sec. 3.3, Fig. 9, Table 2): offline and online disagreed in 45.5 % to 63.3 % of 90
  scenarios; large offline errors meant violations; 40 to 57 low-error scenarios still failed.
* -38 (speculation, checked; Haq 2021, Sec. 3.3.2): disagreement attributed to small errors accumulating in closed loop.
* -39 (measured, checked; Dauner 2023, Sec. 3.2, Tables 1 and 2, Sec. 5): open-loop and closed-loop scores traded off
  (86/54 against 42/92); open-loop metrics discouraged.
* -40 (measured, checked; Dauner 2024 NAVSIM, Sec. 4.1, Figs. 3 and 4): pseudo-simulation score correlated better with
  closed loop than displacement score over 151 planners (coefficients not read).
* -41 (speculation, checked; Ross 2011, Sec. 2.1, Theorem 2.1): cloned policy can cost up to T squared times epsilon (a
  proof, not a measurement).
* -42 (measured, checked; de Haan 2019, Sec. 3.2, Fig. 4, App. E Fig. 11): more input information lowered validation
  loss but lowered reward.
* -43 (measured, checked in de Haan; de Haan 2019, Sec. 3.2, Table 1, from Wang et al. 2019, not opened): history model
  better perplexity, more interventions and collisions.
* -44 (measured, checked; Codevilla 2019, Sec. 5.3, Table 3): 12 seeds of one model varied widely in success (up to 42 %
  "variance" in dense traffic).
* -45 (speculation, checked; Kress-Gazit 2024, Sec. 2 to 4): recommendations for robot tests.
* -46 (measured, checked; Kress-Gazit 2024, Sec. 4.2, App. 6.4): 15/18 against 11/17 gives P about 0.11; 13/20 against
  14/20 undecided.
* -47 (measured, checked; Snyder 2025, Table I, Sec. V-C2): 19 trials for 0.56 against 0.92, 8 for 0.28 against 0.80,
  about 500 for 59 % against 68 %.
* -48 (measured, checked; Snyder 2025, Sec. V-B1, Fig. 3, Sec. V-C2): re-testing after each trial inflates false
  positives; non-significance is not equality.

---

## 10. Not opened or not used

* Wang et al. (2019), the driving history study, seen only through de Haan et al. Table 1: not opened.
* Stocco, Pulfer, Tonella (2023), *Model vs system level testing of autonomous driving systems: a replication and
  extension study*, Empirical Software Engineering: not opened (Springer page needed a login). The arXiv links I tried
  led to two other papers by the same group (*Mind the Gap!*, arXiv 2112.11255, about simulator versus physical car;
  *Two is Better Than One*, arXiv 2305.08060); neither tests offline error against online failure, so I did not use them.
* *Much Ado About Noising* (arXiv 2512.01809) and WorldGym (arXiv 2506.00613): seen only in search results, not opened.
