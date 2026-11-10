# Evaluation statistics: how many robot trials, and which success rule

Key: `evaluation_statistics`. Written 9 November 2026 by a literature sub-agent (Claude) for Team M
(Emranul Hoque and Marshal Sebastine). Nothing here has been tried on the robot.

Files in this folder:

- `note.md`: this note.
- `planning_numbers.py`: every number in Section 2, computed with the Python standard library only
  (exact binomial sums, no approximations unless stated).
- `planning_numbers_output.txt`: the printed output of that script.
- `check_unconditional_tests.py` and `check_unconditional_tests_output.txt`: a check that the test code
  reproduces the worked examples in SciPy's documentation.

Labels used below:

- **[checked]**: I read it in the source itself.
- **[abstract only]**: I could open only the abstract; the full text would not load.
- **[not opened]**: only seen cited elsewhere; not used as evidence.
- **[calculated]**: computed by this note's script, not measured on any robot.
- **[judgement]**: my own reasoning, built on the evidence but not taken from a source.
- **measured** means a paper reports it from its own experiments; **speculation** means the authors
  (or I) say it without measuring it.

---

## 0. Plain-language summary

1. **Ten trials cannot tell 70 % from 90 %.** 7 out of 10 gives a 95 % Wilson interval of 0.40 to 0.89;
   9 out of 10 gives 0.60 to 0.98. The two ranges overlap a lot. [calculated]
2. **To prove with a proper test that one model (90 %) beats another (70 %)** you need about 50 trials per
   model (one-sided test) or about 62 per model (the usual two-sided test). That is 100 to 125 trials,
   roughly 5 to 8 robot hours at the current speed. [calculated]
3. **To simply choose the better of two models** (not prove it), about 20 to 24 trials each is usually
   enough: with a real 20-point gap the better model shows more successes about 95 % of the time.
   [calculated]
4. **To show one model is above 70 % when it is really at 90 %**, about 30 trials are needed. [calculated]
5. **Write the plan before the first trial** (success rule, number of trials, order, analysis) and commit
   it to git. That is a light form of pre-registration. (Vincent et al. 2024; Nosek et al. 2018)
6. **Randomise the order, run both models from the same start in the same session, and hide which model
   is running from the person who scores.** (Kress-Gazit et al. 2024; TRI LBM Team 2025; Bose and Kuang
   2021)
7. **Report counts ("27 out of 30") for every sheet and track**, with Wilson 95 % intervals, the finishing
   times of the successful runs, and the types of failure. (Kress-Gazit et al. 2024; Brown et al. 2001)
8. **Keep the unseen test tracks sealed** until one final run; choose the model on separate validation
   tracks. [judgement]

---

## 1. What the literature does and recommends

### 1.1 Kress-Gazit et al. 2024, the best-practice paper

Hadas Kress-Gazit, Kunimatsu Hashimoto, Naveen Kuppuswamy, Paarth Shah, Phoebe Horgan, Gordon Richardson,
Siyuan Feng, Benjamin Burchfiel (2024). *Robot Learning as an Empirical Science: Best Practices for Policy
Evaluation.* arXiv preprint 2409.09491, version 2, 20 September 2024 (Toyota Research Institute; Cornell
University). No peer-reviewed venue is shown on the arXiv page.
Opened: https://arxiv.org/html/2409.09491v2

- **Section 2.3, four recommendations** [checked]:
  1. Define detailed success criteria ahead of time; the person who writes the criteria should not be the
     person who scores.
  2. Reduce unintended variation in the environment, above all the initial conditions, for example with
     image overlays or markings, and evaluate the compared policies in the same session.
  3. A/B testing: interleave the rollouts of the policies so the evaluator does not know which policy is
     running.
  4. Keep the evaluators consistent, and separate the roles of demonstrator and evaluator.
- **Section 4.1, what to report** [checked]: the definition of overall success and of sub-goal success;
  the number of evaluations per condition, not only percentages; the timing (A/B or not, one session or
  several, days or weeks); the initial conditions, shown as pictures or described, and whether they are
  inside or outside the training distribution.
- **Section 4.2, statistical analysis** [checked]: a single success number can mislead. Frequentist
  options are interval estimates (they cite Agarwal et al. 2021) or bounds (they cite Vincent et al.
  2024); the Bayesian option follows Kruschke 2013 [not opened]. In Example 6 (pancake task) policy A
  succeeded 15 of 18 times and policy B 11 of 17; with a uniform prior on each success probability they
  report a 0.11 probability that B is really better. I recomputed 0.113 [calculated]. The paper names
  neither Wilson nor Clopper-Pearson intervals, and gives no hypothesis test, no power analysis and no
  minimum number of trials.
- **Section 6.3, Figure 7** [checked]: the same success percentages with different counts (5 successes
  and 1 failure against 6 and 3; then 150 and 30 against 110 and 60) lead to different conclusions; only
  the larger counts separate the policies.
- **Section 2.2, Example 2** [checked, measured]: after the lab was reorganised, policy A went from 13/20
  to 0/8 on four initial conditions. Learned policies can be very sensitive to unnoticed changes.
- **Sections 3.1.1 and 4.3** [checked]: rubrics for sub-goals (Table 1: a policy with 4/19 overall
  success was 23/23 on picking up the spatula and flipping), and a description of failure modes with
  categories, counts and examples.
- **Section 6.4, energy-bar example** [checked]: 10 initial conditions, each run twice per policy over two
  days, A/B with a blind evaluator: 13/20 against 14/20, not distinguishable.
- **Discussion** [checked]: the authors leave open the problem of "balancing sample-size with diversity of
  conditions and comparisons given a fixed resource budget".
- **Pre-registration** is not mentioned; the nearest advice is to define the success criteria in advance.

### 1.2 Snyder et al. 2025, comparing two policies with a sequential test (STEP)

David Snyder, Asher James Hancock, Apurva Badithela, Emma Dixon, Patrick Miller, Rares Andrei Ambrus,
Anirudha Majumdar, Masha Itkina, Haruki Nishimura (2025). *Is Your Imitation Learning Policy Better than
Mine? Policy Comparison with Near-Optimal Stopping.* Robotics: Science and Systems XXI (RSS 2025),
paper 077. Opened: https://arxiv.org/html/2503.10966v4 and https://roboticsproceedings.org/rss21/p077.html

- **Abstract and Section I** [checked; the authors' characterisation, not a survey]: robot policy
  comparisons are limited to small numbers of trials (they give 10 or 50 as examples, and cite papers with
  about 10 to 60 per setting) because of human effort and inference time.
- **Sections II and III** [checked]: each trial is a success or failure, trials are assumed independent
  and identically distributed; the null hypothesis is "the new policy is no better"; the significance
  level and the maximum number of trials are fixed by the evaluator before any data are collected.
- **Section VI-A, Figure 3, Figure 6 and Section X-B** [checked, measured in simulation]: the batch
  baseline is Barnard's exact test (Fisher's and Boschloo's tests are mentioned). Running a batch test
  again after each new trial, and stopping when it becomes significant, breaks the promised false-positive
  rate.
- **RSS abstract** [checked, measured]: their sequential test needed up to 32 % fewer trials than the best
  existing sequential tests while keeping the error guarantees.
- **Section V-C1, Table I** [checked, measured]: on a bimanual Franka towel-folding task, two
  diffusion-policy checkpoints had success 0.56 and 0.92 (50 trials each: 5 initial conditions times 10).
  The sequential tests decided after about 17 to 20 trials. The two checkpoints had almost the same
  validation action error (1.61e-3 against 1.35e-3), so the offline loss hid a large difference in
  success.
- **Supplement X-D, Table II** [checked, measured in simulation]: for gaps of about 8 to 9 points
  (0.59 against 0.68, and 0.68 against 0.76), even the ideal sequential test (an oracle SPRT) has power of
  only about 0.32 to 0.34 at 100 trials per policy, and needs about 500 per policy for power near 0.8.
- **Supplement X-A and Section V** [checked]: the order of trials was randomised so that drift affects all
  settings equally; the evaluator was separate from the demonstrator; several tasks were combined with a
  Bonferroni correction. A non-significant result does not show that two policies are equal.

### 1.3 TRI LBM Team 2025, a large evaluation done carefully

TRI LBM Team (Jose Barreiros, Andrew Beaulieu, Aditya Bhat, Rick Cory, Eric Cousineau and others) (2025).
*A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation.* arXiv preprint
2507.05331, version 1, 7 July 2025. Opened: https://arxiv.org/html/2507.05331v1

- **Section 3** [checked]: 50 real-world rollouts per task, per policy, per condition (about 1,800 in
  total); 200 per task in simulation.
- **Section 4.1.1, Figure 8** [checked]: the evaluators did not know which policy was running. Trials ran
  in bundles; each bundle used one initial condition, matched to an overlay picture of the target scene,
  and the policies ran in random order within the bundle (a paired, interleaved design).
- **Sections 4.1.4 and 4.1.5** [checked]: success rates are shown as posteriors from a uniform Beta prior
  (violin plots) instead of confidence intervals, because overlapping intervals can wrongly suggest "no
  difference". Differences are decided by a sequential paired test at a 5 % false-positive rate, with a
  Bonferroni correction inside each set of comparisons, summarised by letters (compact letter display).
- **Section 3** [checked]: the posteriors are per trained checkpoint and do not include the randomness of
  training, so they compare trained models, not training recipes.
- **Section 3** [checked]: tasks were made hard, aiming at about 50 % success, so that the difference
  between policies is visible (a policy at 100 % cannot show that it is better).
- **Section 5, just before 5.1** [checked; the authors' judgement]: they warn of a "significant risk that
  many robotics papers are measuring statistical noise" because studies are too small.

### 1.4 Vincent et al. 2024, bounds on success with few rollouts

Joseph A. Vincent, Haruki Nishimura, Masha Itkina, Paarth Shah, Mac Schwager, Thomas Kollar (2024).
*How Generalizable Is My Behavior Cloning Policy? A Statistical Approach to Trustworthy Performance
Evaluation.* arXiv preprint 2405.05439, version 2, 18 July 2024 (submitted to the IEEE).
Opened: https://arxiv.org/abs/2405.05439 and https://arxiv.org/html/2405.05439v2

- **Section I** [checked; characterisation]: papers commonly evaluate with fewer than 50 rollouts and
  report the raw success rate.
- **Section III-B, Assumptions** [checked]: "a testing plan should be developed before collecting any
  policy rollouts", fixing how initial conditions are sampled and how many rollouts are run. Running
  rollouts until enough good outcomes appear breaks the assumptions (confidence sequences are needed for
  adaptive stopping), and so do conditions that change during testing, such as lighting.
- **Sections IV and VI-B** [checked, measured]: a lower confidence bound on the success probability (a
  tighter, randomised version of the Clopper-Pearson bound). For a pouring diffusion policy with 50
  rollouts per setting, 38/50 gave a 95 % lower bound of 0.641 and 4/50 gave 0.035.
- **Section VI-B** [checked]: test directly in the changed (out-of-distribution) settings that matter,
  because intuition about which changes hurt was unreliable.

### 1.5 What two well-known imitation-learning papers actually ran

- Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran
  Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023; arXiv 2303.04137
  version 5 (2024) is the extended journal version. Opened: https://arxiv.org/abs/2303.04137 and
  https://ar5iv.labs.arxiv.org/html/2303.04137 [checked]
  - Real-world Push-T (Section 6.1, Table 6 caption, Table 3): 20 trials per method (from "8 out of 20
    evaluations"), success means a final overlap (IoU) above the lowest overlap reached in the human
    demonstrations, at most 600 steps, average episode time reported. No error bars on the real-world
    numbers.
  - Simulation (Section 5.2): 3 training seeds times 50 initial conditions, reporting the best checkpoint
    and the mean of the last 10.
- Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio
  Savarese, Yuke Zhu, Roberto Martín-Martín (2021). *What Matters in Learning from Offline Human
  Demonstrations for Robot Manipulation.* CoRL 2021 (oral). Opened: https://arxiv.org/abs/2108.03298 and
  https://ar5iv.labs.arxiv.org/html/2108.03298 [checked]
  - Section 3.3: simulation results are the best checkpoint over training (50 rollouts per evaluation,
    3 seeds).
  - Section 4.5, Figure 4a [measured]: choosing the checkpoint by lowest validation loss gave policies
    10 % to 100 % (relative) worse than the best checkpoint; choosing the last checkpoint fell in the same
    range.
  - Section 4.7: on the real robot, the final checkpoint with 30 rollouts.

### 1.6 Drift and random assignment on real robots

Arnab Bose and Yuheng Kuang (10 June 2021). *The Importance of A/B Testing in Robotics.* Google Research
blog (not peer reviewed). Opened: https://research.google/blog/the-importance-of-ab-testing-in-robotics/
[checked, measured]

- About 33,000 trials of the same bin-to-bin task on two robots over more than five months, with the same
  software and model: weekly success rates drifted away from the baseline, and most weekly 95 % intervals
  excluded zero. When the same data were split into ten randomly assigned groups, the intervals sat at
  zero.
- They recommend random assignment with the baseline run at the same time (A/B). It also stops operators
  knowing which arm a trial belongs to. Lighting, surfaces, wear and battery charge then affect all arms
  alike.
- Kress-Gazit et al. cite a related article by V. Vanhoucke [not opened].

### 1.7 General statistics sources

- Lawrence D. Brown, T. Tony Cai, Anirban DasGupta (2001). *Interval Estimation for a Binomial
  Proportion.* Statistical Science 16(2), 101 to 133. [abstract only, via the ASAIP portal:
  https://asaip.psu.edu/?p=677; the Project Euclid full text did not load] The usual textbook (Wald)
  interval has erratic coverage; they recommend the Wilson interval or the equal-tailed Jeffreys interval
  for small n, and the Agresti-Coull interval for larger n. (A search snippet put the cut-off at n of 40
  or less; I could not check that in the paper.)
- NIST/SEMATECH *e-Handbook of Statistical Methods*, Section 7.2.4.1 "Confidence intervals".
  Opened: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm [checked] Gives the Wilson
  formula, notes that Agresti and Coull recommend it for virtually all sample sizes and rates, and advises
  the exact binomial method when the sample or the number of failures is very small.
- Stian Lydersen, Morten W. Fagerland, Petter Laake (2009). *Recommended tests for association in 2 x 2
  tables.* Statistics in Medicine 28(7), 1159 to 1175. [abstract only, via Europe PMC:
  https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1002/sim.3531&format=json&resultType=core]
  Unconditional tests (the Barnard and Boschloo family) keep the false-positive rate and are more powerful
  than Fisher's exact test for small and moderate samples; Fisher's test with the mid-p adjustment gives
  about the same results; the "traditional Fisher's exact test should practically never be used".
- Morten W. Fagerland, Stian Lydersen, Petter Laake (2013). *The McNemar test for binary matched-pairs
  data: mid-p and asymptotic are better than exact conditional.* BMC Medical Research Methodology 13, 91.
  [abstract only, via Europe PMC:
  https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1186/1471-2288-13-91&format=json&resultType=core]
  For paired success/failure data the mid-p McNemar test never exceeded the nominal level in 9,595
  scenarios and is recommended; the exact conditional McNemar test performed poorly (too cautious).
- Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare (2021). *Deep
  Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 2021 (Outstanding Paper).
  Opened: https://arxiv.org/abs/2108.13264 and https://ar5iv.labs.arxiv.org/html/2108.13264 [checked]
  3 to 10 runs are common (Section 1); comparing point estimates misled earlier comparisons (Section 3,
  Figure 2); report interval estimates, for example stratified percentile bootstrap intervals, which work
  with as few as 10 runs (Section 4.1).
- Cédric Colas, Olivier Sigaud, Pierre-Yves Oudeyer (2018). *How Many Random Seeds? Statistical Power
  Analysis in Deep Reinforcement Learning Experiments.* arXiv 1806.08295. Opened:
  https://arxiv.org/abs/1806.08295 and https://ar5iv.labs.arxiv.org/html/1806.08295 [checked] Choose the
  number of training seeds in advance from a pilot study and a power analysis (Section 4); prefer Welch's
  t-test to the bootstrap test for small samples (Sections 5 and 6); in their example 5 seeds gave about
  49 % power and 10 seeds about 81 % (Section 4).
- Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, David T. Mellor (2018). *The preregistration
  revolution.* PNAS 115(11), 2600 to 2606. [abstract only, via Europe PMC:
  https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1073/pnas.1708274114&format=json&resultType=core]
  Pre-registration means fixing the research questions and the analysis plan before seeing the outcomes;
  it separates tested predictions from explanations made after the fact.
- SciPy documentation, `scipy.stats.barnard_exact` and `scipy.stats.boschloo_exact`. Opened:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.barnard_exact.html and
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boschloo_exact.html [checked]
  Worked examples: 7/15 against 12/15 gives a one-sided Barnard p-value of 0.03407 (Fisher 0.0640);
  74/117 against 31/63 gives a Boschloo p-value of 0.0355. Boschloo's test is uniformly more powerful than
  Fisher's. This note's code gives 0.0341, 0.0641 and 0.0356 (`check_unconditional_tests_output.txt`), so
  the power numbers in Section 2.4 rest on checked code.

**Not opened** (only seen cited; not used as evidence): Wilson 1927 (Journal of the American Statistical
Association); Clopper and Pearson 1934 (Biometrika); Agresti and Coull 1998 (The American Statistician;
the copy I reached was a scanned image I could not read); Newcombe 1998 (the difference interval used in
Section 2.6); Kruschke 2013; Barnard 1945; Boschloo 1970; Hanley and Lippman-Hand 1983 (the "rule of
three"); Vanhoucke's blog article.

---

## 2. Worked numbers [calculated]

All numbers come from `planning_numbers.py` (output in `planning_numbers_output.txt`). They assume what
every paper above assumes: trials are independent and each model has one fixed success probability. Real
robot trials break this when conditions drift or when some starts are harder than others; Section 3 is
designed to limit that.

### 2.1 95 % intervals for 10, 20 and 30 trials

| Observed | Wilson score | Clopper-Pearson (exact) | Uniform-prior Bayesian (credible) |
|---|---|---|---|
| 5/10 | 0.237 to 0.763 | 0.187 to 0.813 | 0.234 to 0.766 |
| 7/10 | 0.397 to 0.892 | 0.348 to 0.933 | 0.390 to 0.891 |
| 8/10 | 0.490 to 0.943 | 0.444 to 0.975 | 0.482 to 0.940 |
| 9/10 | 0.596 to 0.982 | 0.555 to 0.997 | 0.587 to 0.977 |
| 10/10 | 0.722 to 1 | 0.692 to 1 | 0.715 to 0.998 |
| 10/20 | 0.299 to 0.701 | 0.272 to 0.728 | 0.298 to 0.702 |
| 14/20 | 0.481 to 0.855 | 0.457 to 0.881 | 0.478 to 0.854 |
| 16/20 | 0.584 to 0.919 | 0.563 to 0.943 | 0.581 to 0.918 |
| 18/20 | 0.699 to 0.972 | 0.683 to 0.988 | 0.696 to 0.970 |
| 20/20 | 0.839 to 1 | 0.832 to 1 | 0.839 to 0.999 |
| 15/30 | 0.332 to 0.668 | 0.313 to 0.687 | 0.331 to 0.669 |
| 21/30 | 0.521 to 0.833 | 0.506 to 0.853 | 0.520 to 0.833 |
| 24/30 | 0.627 to 0.905 | 0.614 to 0.923 | 0.625 to 0.904 |
| 27/30 | 0.744 to 0.965 | 0.735 to 0.979 | 0.742 to 0.964 |
| 30/30 | 0.886 to 1 | 0.884 to 1 | 0.888 to 0.999 |

How to read it: Clopper-Pearson is always the widest because it guarantees at least 95 % coverage; Wilson
aims at 95 % on average and is the one Brown et al. and the NIST handbook point to for small samples. The
uniform-prior Bayesian interval (the analysis used by Kress-Gazit et al. and the TRI team) is almost the
same as Wilson here. Precision grows slowly: at an observed 80 % the Wilson half-width is about 0.23 with
10 trials, 0.17 with 20, 0.14 with 30, 0.10 with 60, 0.08 with 100 and 0.06 with 150. Knowing one success
rate to within plus or minus 10 points needs about 60 trials.

### 2.2 One model against a fixed bar (exact binomial test, one-sided, 5 %)

| Trials | Question: "is it above 70 %?" (truth 90 %) | Question: "is it below 90 %?" (truth 70 %) |
|---|---|---|
| 10 | pass only with 10/10; power 0.35 | fail it at 6/10 or fewer; power 0.35 |
| 20 | pass with 18/20 or more; power 0.68 | at 15/20 or fewer; power 0.76 |
| 25 | pass with 22/25 or more; power 0.76 | at 19/25 or fewer; power 0.81 |
| 30 | pass with 26/30 or more; power 0.83 | at 23/30 or fewer; power 0.84 |

Power 0.8 is first reached at 28 trials (and stays from 33) for the first question, and at 25 (stays from
29) for the second. Exact tests have a saw-tooth power curve, so one trial more can occasionally lower the
power slightly.

### 2.3 After a clean sweep

All successes only proves so much (one-sided 95 % lower bound): 5/5 shows at least 0.55, 10/10 at least
0.74, 20/20 at least 0.86, 30/30 at least 0.905, 60/60 at least 0.95. This matches the "rule of three"
(failure rate below about 3/n).

### 2.4 Two models, independent trials: how many to tell 70 % from 90 %

Exact power computed by enumerating every possible result, for true rates 0.70 and 0.90, power 0.8:

| Test | Trials per model, one-sided 5 % | Trials per model, two-sided 5 % | Largest false-positive rate found (nominal 0.05 / 0.025) |
|---|---|---|---|
| Barnard exact unconditional (pooled z ordering) | 49 | 61 | 0.050 / 0.025 |
| Boschloo exact unconditional | 50 | 63 | 0.050 / 0.025 |
| Fisher exact with mid-p | 49 | 61 | 0.057 / 0.029 (slightly above nominal) |
| Fisher exact (traditional) | 56 | 69 | 0.043 / 0.021 (too cautious) |
| Pooled z-test (normal approximation) | 46 | 59 | 0.073 / 0.038 (too many false positives) |
| Textbook formula, without / with continuity correction | 49 / 58 | 62 / 72 | not applicable |

Power with small samples (two-sided 5 %, Barnard): 0.14 with 10 per model, 0.35 with 20, 0.48 with 30,
0.60 with 40, 0.79 with 60. With the traditional Fisher test: 0.06, 0.24, 0.37, 0.53, 0.73. In plain
words: with 20 trials per model, a real 20-point gap is missed about two times in three.

### 2.5 Two models from the same starts (paired design)

If both models run from the same start (same sheet, same end, same puck placement), the test only uses the
starts where exactly one model succeeded. Whether pairing saves trials depends on whether a start that is
hard for one model is also hard for the other. Pairs needed for power 0.8:

| How much difficulty is shared | Exact conditional McNemar, two-sided 5 % | Mid-p McNemar, two-sided 5 % | Mid-p McNemar, one-sided 5 % |
|---|---|---|---|
| Not at all (outcomes independent) | 71 | 63 | 48 |
| Half-way | 54 | 47 | 38 |
| Fully (the 90 % model never fails where the 70 % model succeeds) | 39 | 33 | 27 |

So pairing can halve the robot time (33 pairs, 66 trials, instead of about 125) when hard starts are hard
for both, and costs nothing otherwise if the mid-p test is used. The exact conditional version is worse
than unpaired testing when outcomes are independent, which fits Fagerland et al. 2013.

### 2.6 Choosing without a formal test

If the aim is only to pick the model for the exam, the useful number is the chance that the truly better
model shows more successes (ties split by a coin):

| Trials per model | 10 | 12 | 15 | 20 | 24 | 30 | 50 |
|---|---|---|---|---|---|---|---|
| 90 % against 70 % | 0.87 | 0.89 | 0.92 | 0.95 | 0.96 | 0.98 | 0.995 |
| 90 % against 80 % | 0.73 | 0.75 | 0.78 | 0.81 | 0.83 | 0.86 | 0.92 |

When the gap is small, picking wrongly also costs little, so 20 to 24 trials per candidate is a sensible
budget for a decision. These trials are not proof that one model is better.

What an observed 20-point gap means at different sizes (95 % Newcombe interval for the difference, and the
uniform-prior probability that the better-looking model really is better):

| Observed | Difference interval | Probability the second is better |
|---|---|---|
| 7/10 against 9/10 | -0.16 to 0.51 | 0.85 |
| 14/20 against 18/20 | -0.05 to 0.43 | 0.94 |
| 21/30 against 27/30 | -0.005 to 0.39 | 0.97 |
| 42/60 against 54/60 | 0.06 to 0.34 | not computed |

For comparison, Kress-Gazit et al.'s energy-bar result (13/20 against 14/20) gives only 0.63.

### 2.7 Many tracks and robot time

- Nine separate tests (six sheets and three unseen tracks) at 5 % each give a 37 % chance of at least one
  false "significant" result. A Bonferroni correction tests each at 0.0056, which needs many more trials,
  so declare one main (primary) result in advance instead.
- Robot time (assumed, must be measured): 18,500 pictures over 208 demonstrations is about 89 steps per
  run; at 1.2 s per step that is about 107 s of motion. With a reset (puck back, robot home, sheet check) of
  60 to 120 s, which is a guess, one trial takes 2.8 to 3.8 minutes: **about 16 to 22 trials per robot
  hour**.

---

## 3. Proposed protocol for Team M [judgement, built on Sections 1 and 2]

### 3.1 Before the first trial: write the plan (pre-registration)

One file, committed to git with its date before trial 1 (it also serves as the checklist the project rules
ask for before a robot session). Later changes are allowed but are listed as deviations with the reason.
It fixes:

1. The success rule (Section 3.2) and the list of "void" faults.
2. The tracks: which are training sheets, which are validation tracks, which are sealed test tracks
   (Section 3.3).
3. The number of trials per condition, and the start conditions (which end, puck placement).
4. The trial order: a full list generated in advance with a fixed random seed.
5. Who runs the program, who resets and scores, and how the model identity is hidden.
6. The analysis: main result, intervals, tests, and how time is summarised (Section 3.7).
7. The stopping rule: a fixed number of trials (no extra trials after peeking), or a sequential test chosen
   in advance.

### 3.2 The success rule

- **Success (main measure):** copy the exam's finishing rule exactly. If the exam does not define a
  tolerance, write one now (for example, the puck passes the end mark of the line within the step limit,
  without anyone touching the puck or the arm). Diffusion Policy tied its threshold to the human
  demonstrations; in the same spirit a step limit could be twice the step count of the longest
  demonstration of that sheet, and for unseen tracks a limit per centimetre of line.
- **Failure:** time or step limit reached; the policy stops before the end; the puck leaves the line beyond
  the tolerance; a safety stop or collision; any human touch.
- **Void (re-run at the end of the block, and reported):** only faults on a list written in advance that
  are not the policy's doing (robot connection lost, camera returned no picture, set-up error by the
  operator). A policy that stops or wanders is a failure, never a void.
- **Time (second measure):** seconds and steps from the first command to success.
- **Partial progress (third measure):** marks at 25, 50 and 75 % of each line; record the furthest mark the
  puck passed. This makes failures informative, like the TRI team's task-completion score and Kress-Gazit
  et al.'s rubrics.
- **Scoring:** by the person who does not know which model ran; borderline cases decided from video by the
  written rule before the model codes are revealed.

### 3.3 Three kinds of tracks

- **Training sheets:** the six known sheets (straight, arc, two curves, meander, loop with a crossing,
  dashed loop).
- **Validation tracks:** two new tracks, never used for demonstrations, used to choose between models and
  checkpoints.
- **Test tracks:** three or more new tracks of exam size and line thickness, drawn by one person, kept out
  of sight and out of every decision until the final run, used once. Trials on one track share its
  difficulty, so three tracks tell you about those three tracks. If printing more is cheap, six test
  tracks with five trials each say more about "new tracks in general" than three with ten each, for the
  same robot time.

### 3.4 Phase 1: choosing the model (exploratory, not the reported score)

- At most two candidates (or two checkpoints). Offline loss is a poor guide (robomimic Section 4.5; the
  STEP towel checkpoints), so the choice is made on the robot.
- 24 paired blocks, 2 trials each, 48 trials, about 2.2 to 3.0 robot hours: the three harder known sheets
  (meander, loop with a crossing, dashed loop), both ends, twice (12 blocks), plus the two validation
  tracks, both ends, three times (12 blocks). In each block both candidates run from the same start, in
  random order. The easy sheets are left out because both models are likely to score 100 % there (the TRI
  team's reason for aiming at about 50 %).
- Decision: the candidate with more successes; with a 20-point real gap this picks the better one about
  96 % of the time. A tie is broken by the finishing time.
- If robot time is short: 12 blocks first; stop if one candidate leads by 3 or more successes, otherwise
  run the other 12. This kind of looking is acceptable for a decision but not for a significance claim.

### 3.5 Phase 2: final evaluation of the chosen model (confirmatory, reported)

- 60 trials, about 2.7 to 3.8 robot hours, spread over two sessions: six known sheets times 5 trials
  (balanced over the two ends) and three test tracks times 10 trials, all mixed in the pre-generated random
  order, so drift during a day does not hit only the unseen tracks.
- Main result declared in advance: success on the test tracks pooled (k out of 30) with its Wilson 95 %
  interval; second: known sheets pooled (k out of 30). For example 27/30 gives 0.74 to 0.97 and 21/30 gives
  0.52 to 0.83.
- If a bar is wanted ("better than 70 % on unseen tracks"), the one-sided exact test passes at 26/30 or
  more (power 0.83 if the true rate is 90 %).

### 3.6 Optional Phase 3: a formal A/B claim for the report

Only if the report must claim that one design beats another with a test. For a 70 % against 90 % gap:
33 to 63 pairs with the mid-p McNemar test (two-sided), depending on shared difficulty, or 61 to 63 trials
per model unpaired with Barnard's or Boschloo's test; that is 3 to 8 robot hours. A sequential test
(STEP) can stop earlier when the gap is large (its authors report up to 32 % fewer trials). If this is too
expensive, report the comparison as exploratory: counts, the interval for the difference and the posterior
probability, and say plainly that the study was not large enough to detect a 20-point difference reliably.
A comparison of two trained models is a comparison of those two checkpoints; claims about a training
method need several training seeds (Colas et al. 2018; TRI LBM Team 2025, Section 3).

### 3.7 Randomisation, blinding and drift

- **Order:** generate the full list of trials in advance with a fixed random seed and store it in git;
  inside each paired block the model order is random (STEP Supplement X-A; TRI LBM Section 4.1.1; Bose and
  Kuang 2021).
- **Blinding:** the program loads models by code letter; one student keeps the code-to-model key and runs
  the program, the other resets the puck and scores, and the key is opened only after all trials are
  scored (Kress-Gazit et al. Section 2.3; TRI LBM Section 4.1.1). With two people the separation is not
  perfect; say so in the report.
- **Matched starts:** place the puck with the reference-picture overlay the recording tool already has,
  like the TRI team's overlay pictures.
- **Drift:** at the start of each session compare the camera view with the stored reference picture; if it
  differs, stop and follow the camera-mount checklist rather than adjusting anything. Log date, time and
  lighting. Never run model A on one day and model B on another.
- **Warm-up:** one uncounted run on a practice sheet per session to check the robot and the camera,
  declared in the plan.

### 3.8 Analysis and reporting

- Counts per sheet and per track (k/n), pooled counts, Wilson 95 % intervals (Brown et al. 2001; NIST
  7.2.4.1). Clopper-Pearson if a guaranteed bound is needed (Vincent et al. 2024).
- Comparisons: Barnard's or Boschloo's exact test, or Fisher's test with mid-p, for unpaired data
  (Lydersen et al. 2009); mid-p McNemar for paired blocks (Fagerland et al. 2013). Report the difference
  with its interval, or the posterior probability as Kress-Gazit et al. and the TRI team do, not only a
  p-value.
- Time: median and range of finishing time and steps among the successful runs, per sheet and track; to
  compare two models on time, use the blocks where both finished. Time comes second, as in the exam.
- Failure types with counts and one example picture each (Kress-Gazit et al. Section 4.3), and every void
  trial with its reason.
- The Kress-Gazit et al. Section 4.1 list: success definition, trials per condition, when the trials ran,
  start conditions with pictures, and whether the tracks were seen in training.

### 3.9 Robot-hour budget (at 16 to 22 trials per hour, an estimate)

| Phase | Trials | Robot time |
|---|---|---|
| Plan | 0 | 0 |
| Phase 1, model choice | 48 | 2.2 to 3.0 h |
| Phase 2, final evaluation | 60 | 2.7 to 3.8 h |
| Total | 108 | 4.9 to 6.8 h |
| Optional Phase 3, formal A/B claim | 66 to 126 | 3.0 to 7.9 h |

### 3.10 What this protocol cannot tell you

It cannot prove small differences (10 points or less needs hundreds of trials per model; STEP Table II),
it describes the test tracks used rather than every possible track, and it describes the trained
checkpoints rather than the training recipe.

---

## 4. Claims list (ids as in the structured result)

| Id | Short claim | Status |
|---|---|---|
| evaluation_statistics-1 | Kress-Gazit 2.3: success rule in advance, matched starts, blind interleaved A/B, separate roles | checked |
| evaluation_statistics-2 | Kress-Gazit 4.1: report counts per condition, timing, start conditions, success definition | checked |
| evaluation_statistics-3 | Kress-Gazit Example 6: 15/18 vs 11/17, P(B better) 0.11 (recomputed 0.113); no power analysis | checked |
| evaluation_statistics-4 | Kress-Gazit Fig 7: same percentages, different counts, different conclusions | checked |
| evaluation_statistics-5 | Kress-Gazit Example 2: 13/20 fell to 0/8 after a lab reorganisation | checked |
| evaluation_statistics-6 | Kress-Gazit: sample size against diversity left open; no pre-registration | checked |
| evaluation_statistics-7 | STEP: typical comparisons use about 10 to 60 trials | checked |
| evaluation_statistics-8 | STEP: re-testing after each trial breaks error control; fix level and maximum first | checked |
| evaluation_statistics-9 | STEP: up to 32 % fewer trials than sequential baselines | checked |
| evaluation_statistics-10 | STEP Table II: 8 to 9 point gaps need about 500 trials per policy | checked |
| evaluation_statistics-11 | STEP Table I: 0.56 vs 0.92 with near-equal validation loss; decided after 17 to 20 trials | checked |
| evaluation_statistics-12 | STEP X-A: randomised order, separate evaluator, Bonferroni | checked |
| evaluation_statistics-13 | TRI LBM: 50 real rollouts per task and policy, blind, randomised bundles, overlay starts | checked |
| evaluation_statistics-14 | TRI LBM: uniform-prior posteriors, sequential paired test, Bonferroni, letter display | checked |
| evaluation_statistics-15 | TRI LBM: posteriors per checkpoint, no training randomness | checked |
| evaluation_statistics-16 | TRI LBM: tasks made hard, aiming at about 50 % success | checked |
| evaluation_statistics-17 | TRI LBM: warning that many papers measure noise | checked |
| evaluation_statistics-18 | Vincent: papers commonly use fewer than 50 rollouts | checked |
| evaluation_statistics-19 | Vincent: testing plan before rollouts; no stopping on good results | checked |
| evaluation_statistics-20 | Vincent: 38/50 gives lower bound 0.641, 4/50 gives 0.035 | checked |
| evaluation_statistics-21 | Diffusion Policy: 20 real trials per method, threshold from human demonstrations, no error bars | checked |
| evaluation_statistics-22 | robomimic: validation-loss checkpoint choice 10 to 100 % worse than best | checked |
| evaluation_statistics-23 | robomimic: best checkpoint in simulation, final checkpoint and 30 rollouts on the robot | checked |
| evaluation_statistics-24 | Bose and Kuang: five months of drift; random assignment removes it | checked |
| evaluation_statistics-25 | Agarwal: 3 to 10 runs common; interval estimates, stratified bootstrap | checked |
| evaluation_statistics-26 | Colas: seeds chosen by power analysis; Welch's t-test | checked |
| evaluation_statistics-27 | Brown et al.: Wilson or Jeffreys for small n (abstract only) | checked |
| evaluation_statistics-28 | NIST 7.2.4.1: Wilson recommended; exact method for very small samples | checked |
| evaluation_statistics-29 | Lydersen et al.: unconditional or mid-p tests, not traditional Fisher (abstract only) | checked |
| evaluation_statistics-30 | Fagerland et al.: mid-p McNemar for paired data (abstract only) | checked |
| evaluation_statistics-31 | Nosek et al.: pre-registration fixes questions and analysis before outcomes (abstract only) | checked |
| evaluation_statistics-32 | SciPy examples reproduced by this note's code | checked |
| evaluation_statistics-33 | Intervals for 10, 20, 30 trials | calculated |
| evaluation_statistics-34 | One model against a 70 % bar: about 28 to 33 trials | calculated |
| evaluation_statistics-35 | Two models, 70 % vs 90 %: about 49 to 50 (one-sided) or 61 to 63 (two-sided) per model | calculated |
| evaluation_statistics-36 | Paired: 33 to 63 pairs (mid-p McNemar, two-sided) | calculated |
| evaluation_statistics-37 | Picking the better: 0.87 at 10, 0.95 at 20, 0.96 at 24 trials each | calculated |
| evaluation_statistics-38 | A 20-point observed gap: intervals and posterior probabilities | calculated |
| evaluation_statistics-39 | Clean sweeps; nine tests give 37 % false-positive risk | calculated |
| evaluation_statistics-40 | Robot time about 16 to 22 trials per hour (reset time guessed) | unchecked |
| evaluation_statistics-41 | More test tracks with fewer trials each | judgement |
| evaluation_statistics-42 | Separate validation tracks; sealed test tracks used once | judgement |

---

## 5. Conclusion for the project (at most 200 words)

Evidence: robot-learning papers usually run 10 to 50 trials per condition (Snyder 2025; Vincent 2024).
Best-practice papers ask for success rules fixed in advance, blind and interleaved A/B trials from matched
starts, randomised order, counts per condition, and interval or Bayesian analysis (Kress-Gazit 2024; TRI
LBM 2025; Bose and Kuang 2021). Statisticians favour Wilson intervals for small samples and unconditional
or mid-p tests over the traditional Fisher test (Brown 2001; Lydersen 2009). Calculated here: with 10
trials, 7/10 and 9/10 give overlapping intervals (0.40 to 0.89, 0.60 to 0.98); a formal test of 70 %
against 90 % needs about 50 trials per model one-sided or 62 two-sided, fewer only when paired starts share
difficulty; one model against a 70 % bar needs about 30. Judgement: Team M should not try to prove every
design choice. Choose between two candidates with about 24 paired, blind trials each on harder sheets and
separate validation tracks (96 % chance of picking the better if the gap is 20 points), then run one
pre-registered final test of 60 trials (6 sheets times 5, 3 sealed unseen tracks times 10) and report
counts, Wilson intervals, finishing times and failure types.
