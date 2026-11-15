# Check of the evaluation_statistics claims: how many robot trials, and which success rule

Key: `evaluation_statistics`. Checked on 9 November 2026 by a separate checking sub-agent (Claude) for Team M
(Emranul Hoque and Marshal Sebastine). I opened every cited source myself and recomputed every calculated
number with my own script. Nothing here has been tried on the robot.

Files in this folder:

- `note.md`: this note.
- `recompute_planning_numbers.py`: my independent recomputation, Python standard library only. Where possible
  it uses other methods than the researcher's script: Clopper-Pearson limits from Beta quantiles (continued
  fraction), the closed-form sum for comparing two Beta posteriors (checked by a midpoint integral), exact
  unconditional tests built by a binary search over the ordered tables and then checked on a fine nuisance grid
  (2001 points plus golden-section refinement). It also computes new numbers used in the challenge
  (precision at other rates, choice probability for other gaps, clustering of trials within tracks, robot time).
- `recompute_planning_numbers_output.txt`: its printed output.

Verdict words: **supported** (the source says this), **corrected** (the source says something different; the
correct statement is given), **not supported**, **could not open**.

---

## 1. Verdicts at a glance

| Id | Verdict | In one line |
|---|---|---|
| 1 | supported | Kress-Gazit 2.3, all four recommendations as stated |
| 2 | supported | Kress-Gazit 4.1, items 1 to 4, including in or out of distribution |
| 3 | supported | 0.11 in the paper (rendered ".11.11" in the HTML, a maths artefact); I get 0.1128 |
| 4 | supported | Figure 7 counts exactly as stated |
| 5 | supported | 13/20 before, 0/8 after (2 runs each on initial conditions 0, 1, 3, 4) |
| 6 | supported | sample size against diversity named as open; no "preregistration" anywhere |
| 7 | supported | abstract "10 or 50", introduction "10 to 60" per setting; a characterisation |
| 8 | corrected | STEP fixes the level and the maximum number of trials in advance for its sequential test too |
| 9 | supported | up to 32 % fewer trials; strongest in the hardest comparisons |
| 10 | supported | Table II 0.324 and 0.337 at 100, 0.823 and 0.804 at 500; CarrotOnPlate on a real WidowX |
| 11 | corrected | decisions after 17 to 23 trials (17 to 20 only with a 50-trial cap); "nearly equal loss" is not the paper's reading |
| 12 | corrected | the Bonferroni multitask comparison was in simulation, and the "not equal" remark is in V-C2, not VII |
| 13 | supported | 50 per task, policy and condition; 1,800 is in the Introduction |
| 14 | supported | Sections 4.1.4 and 4.1.5 as stated |
| 15 | supported | posteriors per checkpoint, no training randomness |
| 16 | supported | difficult tasks, about 50 % success targeted |
| 17 | supported | "statistical noise" warning, authors' judgement |
| 18 | supported | fewer than 50 rollouts, empirical success rate; arXiv comment "submitted to the IEEE" |
| 19 | supported | testing plan before rollouts; adaptive stopping needs confidence sequences |
| 20 | supported | 0.641 and 0.035; plain Clopper-Pearson gives 0.640 and 0.028, so theirs is tighter, as stated |
| 21 | supported | 20 trials per method, IoU above the demonstrations' minimum, 600 steps, no spread |
| 22 | supported | 10 % to 100 % relative drop; in simulation |
| 23 | supported | best checkpoint, 50 rollouts, 3 seeds in simulation; final checkpoint, 30 rollouts on the robot |
| 24 | supported | only one of twenty weekly intervals contained zero; random groups did |
| 25 | supported | 3 to 10 runs; stratified percentile bootstrap good from 10 runs |
| 26 | supported | power 49 % and 81 % follow from type II errors of 51 % and 19 % |
| 27 | supported | abstract only (full text would not load for me either) |
| 28 | supported | with a nuance: the small-sample warning is about the symmetric normal-approximation limits |
| 29 | supported | abstract only; it does not name Barnard or Boschloo |
| 30 | supported | 9595 scenarios; mid-p and exact unconditional "can be recommended" |
| 31 | supported | abstract only |
| 32 | supported | documented values right; "validates the power calculations" is too strong |
| 33 | corrected | all twelve limits right; "plus or minus 10 points needs about 60 trials" holds only near 80 % |
| 34 | supported | recomputed exactly |
| 35 | supported | recomputed exactly, with a finer nuisance grid |
| 36 | supported | recomputed exactly |
| 37 | supported | recomputed exactly |
| 38 | supported | recomputed exactly |
| 39 | supported | recomputed exactly |
| 40 | supported | arithmetic right; it is an estimate that can be 40 % too optimistic |
| 41 | supported | my clustering simulation supports it; but the conclusion then ignores it |
| 42 | corrected | the logic is sound, but robomimic does not show a winner's score is overstated; Cawley and Talbot 2010 does |

---

## 2. Details, claim by claim

### Kress-Gazit et al. 2024 (claims 1 to 6)

Opened https://arxiv.org/html/2409.09491v2 and, for Example 6, https://arxiv.org/html/2409.09491v1 (same
rendering). arXiv preprint, 20 September 2024, no venue named.

- **1, supported.** Section 2.3 has four bullets: define detailed success criteria ahead of time (the behaviour
  designer writes them, but is not the person evaluating); reduce unintended variability between rollouts,
  above all initial conditions, with overlays or scene markings, and run the compared policies in the same
  session; A/B testing, interleaved and blind to the evaluator; consistent evaluators and separate demonstrator
  and evaluator roles.
- **2, supported.** Section 4.1, items 1 to 4: success overall and in sub-goals; number of evaluations per
  condition, not only percentages; timing (A/B or not, one session or several); initial conditions shown or
  described, their relation to the training conditions, and whether they were chosen in or out of distribution.
- **3, supported.** Example 6 is in Section 4.2 and points to Figure 6 (appendix, Section 6.3). The probability
  appears as ".11.11" in both HTML versions, which is the MathML text plus its alternative text, so 0.11. My
  closed form gives 0.1128 (midpoint integral 0.1128). A text search finds no "hypothesis", "p-value",
  "Wilson", "Clopper" or statistical "power", and no minimum number of trials.
- **4, supported.** Figure 7 (Section 6.3): (a) A 5 successes and 1 failure, B 6 and 3; (b) A 150 and 30, B 110
  and 60; in (b) a difference of zero is no longer in the support. The panel (a) rates are only roughly equal
  to panel (b) (83 % and 67 % against 83 % and 65 %), which the paper explains by whole-number counts.
- **5, supported.** Example 2 (Section 2.2): after a lab reorganisation, 8 runs of policy A, 2 each on initial
  conditions 0, 1, 3 and 4, gave 0/8; before, the same initial conditions gave 13/20.
- **6, supported.** The closing discussion names balancing sample size with diversity of conditions under a
  fixed budget as a difficulty and does not solve it. No form of "preregistration" appears in the text.

### Snyder et al. 2025, STEP (claims 7 to 12)

Opened https://arxiv.org/html/2503.10966v4 (whole text, in three parts) and
https://roboticsproceedings.org/rss21/p077.html (RSS XXI, DOI 10.15607/RSS.2025.XXI.077).

- **7, supported.** Abstract: a small feasible sample size (for example 10 or 50) because of human effort and
  limited inference throughput. Introduction: a small number of trials (for example 10 to 60) per setting,
  citing robomimic, Implicit BC, Diffusion Policy, Octo, OpenVLA and pi-0. It is the authors' characterisation,
  not a survey.
- **8, corrected.** What is right: the captions of Figure 3 (400 simulated sequences per alternative) and
  Figure 6 (1000 per alternative) say that using a batch method in sequence violates type I error control;
  Section V-B1 says the same of Barnard's test; the Introduction calls topping up a batch test with more trials
  p-hacking; Supplement X-B shows that a Bonferroni-corrected sequential Barnard test restores error control but
  loses much power. What needs correcting: Section III has the evaluator fix the significance level and the
  maximum number of trials (N_max) in advance for STEP itself. So a sequential test is not an alternative to
  fixing them. Correct statement: fix the level and the maximum number of trials before the first trial in every
  case; a batch test is run once at its planned size; stopping early needs a valid sequential test (STEP, SAVI
  or Lai) that was also set up in advance.
- **9, supported.** RSS abstract: up to 32 % fewer evaluation trials than state-of-the-art baselines with
  probabilistic correctness and power kept; strongest in the most challenging comparisons. The Introduction
  says the baselines are state-of-the-art sequential methods.
- **10, supported.** Supplement X-D, Table II: oracle SPRT power 0.324 and 0.337 at 100 trials, 0.823 and 0.804
  at 500; the text says 500 trials are needed before the probability reaches about 80 %. Section V-C2:
  CarrotOnPlate ran on a real WidowX 250S with N_max = 100 (0.59 against 0.68 and 0.68 against 0.76) and no
  method gave a significant result at 0.05.
- **11, corrected.** Right: single-task diffusion policies on a bimanual Franka (Section V-C1, Table I);
  validation action MSE 1.61e-3 for the baseline and 1.35e-3 for the other checkpoint; success 0.56 against 0.92
  from 5 in-distribution initial conditions times 10. Stopping times (SAVI, Lai, STEP, oracle): 20, 17, 19, 17
  with N_max = 50; 20, 21, 21, 17 with N_max = 200; 20, 23, 23, 17 with N_max = 500; the text says 17 to 23
  trials apiece. So "about 17 to 20" holds only for the 50-trial cap. The paper does not call the two losses
  "nearly equal"; it reads the 36-point gap as a sign that the baseline was under-trained. The lower-loss
  checkpoint was the better one, so in this example the validation loss ranked the two correctly; it only did
  not show how large the gap was. This example should not be used as evidence that offline loss misleads.
- **12, corrected.** Right: Supplement X-A randomises the trial order so that drift (hardware wear is the example)
  hits all settings equally, and separates the evaluator from the demonstrator. Wrong location and setting: the
  Bonferroni-corrected multitask comparison (Section V-C3; three tasks at 0.01 each, 0.03 in total) was run in
  simulation (SimplerEnv, Octo-Small against Octo-Base), not on hardware; and the remark that a non-significant
  result does not mean the null hypothesis should be accepted is in Section V-C2, not Section VII (which is
  Limitations and does not discuss it).

### TRI LBM Team 2025 (claims 13 to 17)

Opened https://arxiv.org/html/2507.05331v1 (all three parts). arXiv preprint, 7 July 2025; author list in
Section 6 starts Barreiros, Beaulieu, Bhat, Cory, Cousineau, as claimed.

- **13, supported.** Section 3 (Results, opening text): 50 real-world rollouts per task, per policy, per
  condition; 200 per task, per policy, per condition in simulation (a few tasks fewer because of missing data).
  The 1,800 real-world trials figure is in Section 1, not Section 3. Section 4.1.1 and Figure 8: blind
  evaluators; bundles as large as the number of policies, one initial condition per bundle, random policy order;
  an overlay of the camera image and the desired initial scene.
- **14, supported.** Section 4.1.4: violins are uniform-Beta-prior posteriors; overlapping confidence intervals
  can lead to a wrong "not separated" conclusion. Section 4.1.5: a sequential test on paired outcomes with type I
  error at most 5 %; Bonferroni across the pairwise comparisons inside one plot (not across tasks); compact letter
  display, used in the captions of Figures 2, 3, 4, 6 and 7.
- **15, supported.** Section 3: the distributions are per checkpoint and leave out training randomness.
- **16, supported.** Section 3: tasks made difficult, targeting about 50 % success so that relative success rates
  are as informative as possible; actual rates ended higher or lower.
- **17, supported.** Section 5, before 5.1: many effects were measurable only with larger-than-standard samples
  and careful, non-standard testing, and many robotics papers risk measuring statistical noise.

### Vincent et al. 2024 (claims 18 to 20)

Opened https://arxiv.org/html/2405.05439v2 and https://arxiv.org/abs/2405.05439v2.

- **18, supported.** Section I: common to evaluate with fewer than 50 rollouts, recording the empirical success
  rate. The arXiv comment field says the work was submitted to the IEEE.
- **19, supported.** Section III-B: write a testing plan (sampling rule for initial conditions, number of
  rollouts) before any rollout; running until enough good outcomes, or lighting that changes (sunset), breaks the
  assumptions; confidence sequences suit adaptive stopping.
- **20, supported.** Section IV: a randomised, tighter version of the Clopper-Pearson bound. Section VI-B: a
  diffusion policy pouring ice from a red cup into a sink, 50 rollouts in each of two OOD settings: 4/50
  (hypothesised harmful, another red object in view) and 38/50 (benign); lower bounds 0.035 and 0.641. My plain
  one-sided Clopper-Pearson bounds are 0.028 and 0.640, so theirs are a little tighter, as expected.

### Other robot-learning sources (claims 21 to 24)

- **21, supported.** Chi et al., opened https://ar5iv.labs.arxiv.org/html/2303.04137 (the extended version of the
  RSS 2023 paper). Section 6.1: LSTM-GMM stuck in 8 of 20 evaluations, IBC left early in 6 of 20; Table 6
  caption: success means an end-state IoU above the minimum IoU in the demonstrations; Table 3: real-world
  Push-T allows at most 600 rollout steps; Table 6 gives single numbers with no spread.
- **22, supported.** Mandlekar et al., opened https://ar5iv.labs.arxiv.org/html/2108.03298. Section 4.5 (in
  simulation): the "relative performance drop" from choosing by validation loss or by the last checkpoint is
  10 % to 100 % against the best checkpoint, which motivates better offline metrics. Note: that "best" is itself
  the maximum over many noisy 50-rollout evaluations.
- **23, supported.** Section 3.3: maximum success over training, 50 rollouts, average of 3 seeds. Section 4.7: the
  final checkpoint, 30 rollouts, because real-world evaluation takes so long.
- **24, supported.** Bose and Kuang, opened https://research.google/blog/the-importance-of-ab-testing-in-robotics/
  (10 June 2021, not peer reviewed): about 33k trials on two robots over more than five months, same software and
  model, foam dice moved between bins. Most weekly intervals were expected to contain zero but only one did; ten
  randomly assigned groups gave intervals around zero. They recommend a concurrent baseline with random
  assignment, which also keeps operators from knowing the arm.

### Statistics sources (claims 25 to 32)

- **25, supported.** Agarwal et al., opened https://ar5iv.labs.arxiv.org/html/2108.13264: 3 to 10 runs common
  (Section 1; Figure 1 says 5 or fewer); point estimates misled the Atari 100k comparisons (Section 3, Figure 2);
  stratified percentile bootstrap intervals are good from N = 10 runs (Section 4.1).
- **26, supported.** Colas et al., opened https://ar5iv.labs.arxiv.org/html/1806.08295: pilot study and power
  analysis (Section 4); Welch's t-test over the bootstrap test (Sections 5.1 and 6; the paper's small-sample
  threshold is 10 in one place and 20 in another); Example 2: type II error 51 % with 5 seeds and 19 % with 10.
- **27, supported.** Brown, Cai and DasGupta: the ASAIP abstract (https://asaip.psu.edu/?p=677) says what the claim
  says. Project Euclid and the Wharton copy did not load for me either, so the cut-off at n = 40 is still
  unverified (one search snippet gave "n > 40" for larger n).
- **28, supported.** NIST 7.2.4.1 (opened): the Wilson interval (under a heading that names Agresti and Coull)
  was recommended by Agresti and Coull for virtually all combinations of n and p; the exact binomial method comes
  under the bold heading "Exact Intervals for Small Numbers of Failures and/or Small Sample Sizes". The warning
  sentence there is about symmetric normal-approximation limits, not about Wilson.
- **29, supported.** Lydersen et al. 2009, Europe PMC abstract (opened): as claimed; Barnard and Boschloo are the
  researcher's examples, not named in the abstract.
- **30, supported.** Fagerland et al. 2013, Europe PMC abstract (opened): mid-p never exceeded the nominal level in
  9595 scenarios, almost as powerful as the asymptotic test; mid-p and exact unconditional "can be recommended";
  the exact conditional test did not perform well.
- **31, supported.** Nosek et al. 2018, Europe PMC abstract (opened): as claimed.
- **32, supported, with one overstatement.** SciPy pages (opened): Barnard example `[[7, 12], [8, 3]]`,
  alternative "less", statistic -1.894, p 0.03407; Fisher on the same table 0.0640; Boschloo example
  `[[74, 31], [43, 32]]`, alternative "greater", statistic 0.0483, p 0.0355; Boschloo is described as uniformly
  more powerful than Fisher. My code gives 0.03411, 0.06407 and 0.03556 (SciPy's optimiser seems to stop a little
  below the true maximum), the researcher's 0.0341, 0.0641 and 0.0356: agreement to about three significant
  figures. Matching three p-values checks the p-value code, not the power enumeration; the power numbers are
  confirmed by my separate recomputation (claim 35).

### Calculated numbers (claims 33 to 40), recomputed independently

- **33, corrected.** All twelve limits agree to three decimals (for example 7/10: Wilson 0.397 to 0.892,
  Clopper-Pearson 0.348 to 0.933). But "plus or minus 10 points needs about 60 trials" holds only near a true
  rate of 80 %. A Wilson half-width of 0.10 needs 93 trials at 50 %, 78 at 70 %, 60 at 80 % and 37 at 90 %.
- **34, supported.** Power first reaches 0.8 at 28 trials (0.858), falls below it once at 32 (0.789; 31 gives
  0.807) and stays above it from 33 (0.894). With 30 trials, pass at 26 or more (size 0.030, power 0.825). With 10,
  only 10/10 passes (power 0.349).
- **35, supported.** Barnard 49 and Boschloo 50 (one-sided 0.05); 61 and 63 (one-sided 0.025, which for Barnard
  with equal groups is exactly the two-sided 0.05 test, by symmetry); Fisher 56 and 69; formula 62 (61.6), 72
  with continuity correction (71.25). Two-sided Barnard power 0.143, 0.354, 0.480 at 10, 20, 30. On the fine grid
  the largest size of the Barnard and Boschloo regions is 0.0500 (and 0.0249 to 0.0250 at 0.025), so the
  researcher's 201-point grid did not hide any excess.
- **36, supported.** Mid-p, two-sided: 33, 47, 63 pairs; exact conditional: 39, 54, 71.
- **37, supported.** 0.870, 0.947, 0.962, 0.977 (70 against 90 %); 0.811 and 0.861 (80 against 90 %).
- **38, supported.** Newcombe: -0.160 to 0.514, -0.054 to 0.431, -0.005 to 0.390, 0.057 to 0.336. Posterior:
  0.845, 0.935, 0.971.
- **39, supported.** 0.741, 0.861, 0.905; 1 - 0.95^9 = 0.370; 0.05/9 = 0.00556.
- **40, supported as arithmetic.** 18,500/208 = 88.9 steps; 106.7 s of motion; 2.78 and 3.78 minutes per trial;
  21.6 and 15.9 trials per hour. Sensitivity (my calculation): if a test run needs 1.5 times the demonstration
  steps (a stop-and-go model can hesitate) and a reset takes 2 to 3 minutes, it is 10.6 to 12.9 trials per hour,
  and the proposed 108 trials need 8.4 to 10.2 robot hours instead of 4.9 to 6.8.

### Judgements (claims 41 and 42)

- **41, supported, and quantified.** I simulated tracks whose own success rates vary around a mean of 0.80, with an
  intra-class correlation (how much trials on one track resemble each other) of 0.05, 0.15 or 0.30. These values
  are guesses; nobody has measured them. Coverage of the nominal 95 % Wilson interval for the mean success on new
  tracks, 30 trials in every design (20,000 simulated studies each):

  | Design (tracks x trials) | correlation 0.05 | 0.15 | 0.30 |
  |---|---|---|---|
  | 3 x 10 | 0.918 | 0.826 | 0.706 |
  | 6 x 5 | 0.944 | 0.901 | 0.837 |
  | 10 x 3 | 0.955 | 0.934 | 0.904 |
  | 30 x 1 | 0.964 | 0.964 | 0.965 |

  With correlation 0.15, 3 x 10 is worth only about 13 independent trials (design effect 2.35). This fits Vincent
  et al. 2024, whose bounds are "valid for the environment the samples were drawn from" (Section VII), with i.i.d.
  starts as the sufficient condition (Section VI). The conclusion's final run (3 tracks x 10) contradicts this
  judgement.
- **42, corrected.** The logic (choose on separate data, then report one untouched run) is sound, and the numbers
  follow from claims 35 to 37. But robomimic Section 4.5 measures how much worse offline-chosen checkpoints are
  than the online-best one; it does not measure how far a chosen winner's own score is overstated. Direct support
  is Cawley and Talbot 2010 (see Section 4). And by claim 41 the sealed run should use more tracks with fewer
  trials each.

---

## 3. Does the evidence support the conclusion for Team M?

Mostly yes. Fixing the rule and the plan in advance, blind and randomised paired trials from matched starts,
reporting counts with Wilson intervals, and not trying to prove small differences are all well supported and fit a
two-person project with one robot. But five points need changing.

1. **The final test design measures three tracks, not "unseen tracks".** The exam uses new tracks of the same size
   and line thickness. Ten trials on each of three tracks mostly repeat each track's difficulty, so a Wilson
   interval on k/30 is too narrow for new tracks (83 % instead of 95 % coverage at a guessed correlation of 0.15).
   Printing tracks is cheap, so use as many distinct sealed tracks as possible (for example 10 x 3 or 15 x 2),
   drawn by a written random rule from a defined family, and analyse per track. The six known sheets are a fixed
   set that the exam reuses, so 6 x 5 is fine there.
2. **The "96 %" depends on a 20-point gap.** With 24 trials each it is 0.96 for 70 against 90 %, 0.92 for 50
   against 70 % or 30 against 50 % (the harder sheets may sit there), 0.83 for 80 against 90 % and 0.70 for 85
   against 90 %. The plan is still sensible, because a small gap costs little at the exam, but the conclusion
   should say so.
3. **The time measure.** "Finishing matters most, time second" is a two-level rule. Comparing time only among
   successful runs is biased when the two models finish at different rates. Score each paired block by the rule
   (finish first, then time), or treat failures as censored times. The Project 6 evaluation format is "not
   announced yet" (`01_course/moodle_snapshot.md`, Dates table), so the number of attempts per track and any time
   limit are unknown; the plan should state its assumption and copy the real rule once it is published.
4. **Robot time is a guess.** At 11 to 13 trials per hour (longer runs, slower resets) the plan needs 8 to 10
   hours. Measure one reset and ten model runs first.
5. **Two smaller points.** "7/10 and 9/10 give overlapping intervals" is the very reasoning the TRI team warns
   against; quote the difference interval (-0.16 to 0.51) or the posterior (0.85) instead. And STEP still needs the
   level and the maximum number of trials fixed in advance.

How well does the literature transfer? The trial-count figures (10 to 50, or 10 to 60) are authors'
characterisations from labs with continuous-action, GPU-trained policies, not surveys. The statistics (binomial
trials, pairing, blinding, drift) transfer directly to a CPU-trained classifier run in a stop-and-go loop, and
fewer candidates (two or three checkpoints, since training is slow on a CPU) make the "choose, then one sealed run"
plan realistic. The evidence that offline loss is a poor guide is mixed: robomimic (simulation) and the
data-scaling study (Section 4) support it, but in the STEP towel example the lower-loss checkpoint was the better
one. Offline accuracy on held-out tracks remains a cheap screen before robot time, not a replacement for it.

---

## 4. Important evidence the researcher missed (all opened by me unless marked)

- **Clustering of trials within a track.** Design effect 1 + (m - 1) x rho for m trials per cluster and
  intra-class correlation rho; Wikipedia "Design effect", Section 4.3, credits Kish 1965 (book not opened); Killip,
  Mahfoud and Pearce 2004, *Annals of Family Medicine* 2(3), 204 to 208
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC1466680/; the formula is an image there, so I took it from Wikipedia):
  a smaller design effect means a larger effective sample. Together with Vincent et al. 2024 (Sections VI and VII)
  and my simulation, this is the argument for many tracks with few trials each.
- **Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, Yang Gao (2024).** *Data Scaling Laws in
  Imitation Learning for Robotic Manipulation.* arXiv 2410.18647 (opened https://arxiv.org/html/2410.18647v3; the
  ICLR 2025 listing was seen only in search results). A generalisation study of imitation learning on a real arm:
  8 unseen environments (or 8 unseen objects) with 5 trials each, 40 per policy (Section 4.1); a normalised score
  with partial credit, because success is "an overly sparse signal lacking the granularity to distinguish between
  policies" (Section 3); blind tests with policies shuffled at identical starts (Appendix E.2); validation MSE often
  did not track real-world performance (Appendix E.1). [checked]
- **Gavin C. Cawley and Nicola L. C. Talbot (2010).** *On Over-fitting in Model Selection and Subsequent Selection
  Bias in Performance Evaluation.* JMLR 11, 2079 to 2107 (abstract opened at
  https://jmlr.org/papers/v11/cawley10a.html). Over-fitting the selection criterion makes common evaluation
  practices optimistic, by amounts comparable to real differences between methods. This is the direct support for
  keeping the reported test separate from model choice. [abstract only]
- **Stephane Ross, Geoffrey Gordon, Drew Bagnell (2011).** *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning.* AISTATS, PMLR 15, 627 to 635 (abstract opened at
  https://proceedings.mlr.press/v15/ross11a.html). Future observations depend on the policy's own past actions,
  which breaks the i.i.d. assumption and hurts performance. With about 89 decisions per run, per-picture accuracy of
  the move-name model cannot stand in for closed-loop trials. [abstract only]
- **A combined success-and-time measure.** Anderson et al. 2018, *On Evaluation of Embodied Navigation Agents*
  (arXiv 1807.06757; only the abstract page opened, the SPL definition not checked) is the usual source for
  "success weighted by path length". Survival analysis with failures as censored times is standard; no source
  opened. Either fixes the bias of "time among successes only".
- **The course's own rule.** `01_course/moodle_snapshot.md`: evaluation on the provided tracks and new unseen ones of
  the same overall dimensions and line thickness; the Project 6 evaluation format is not announced yet. The test
  protocol should mirror it once known.
- **How the sealed tracks are made, and what happens if they are used up.** Vincent et al. 2024 (Section III-B) ask
  for a written sampling rule for initial conditions; the same applies to drawing unseen tracks. The plan also needs
  a rule for when the sealed run disappoints and the team retrains: a second sealed set, or reporting the first
  result as the honest one.
- **Partial progress as a more informative measure.** Lin et al. (above) and the TRI team's task-completion analysis
  (Section 11 of 2507.05331, not opened) use partial credit; a furthest-mark score on each line gives more
  information per trial than finish or fail and can guide model choice, while the exam rule stays the main result.

---

## 5. What I would change in the conclusion

1. Final run: 6 known sheets x 5 trials, plus as many distinct sealed unseen tracks as can be printed (at least 10,
   2 to 3 trials each), drawn by a written random rule; report per-track results and the pooled count.
2. State that the 96 % figure assumes a 20-point gap, and give the 80 against 90 % figure (83 %).
3. Replace "overlapping intervals" with the difference interval or posterior.
4. Time: a finish-then-time rule per paired block, or a censored-time analysis; not time among successes only.
5. Measure reset time and steps per test run before fixing the trial numbers.
6. Cite Cawley and Talbot 2010 for the sealed final run, and drop the STEP towel example as evidence against offline
   loss.
