# Check of "Which offline numbers predict closed-loop success" (key offline_vs_closed_loop)

Sceptical check of 48 claims and the conclusion, 9 November 2026. I opened every cited source myself (arXiv HTML,
ar5iv, arXiv abstract pages, PMLR pages). For Codevilla et al. (2018) the figure panels are missing from ar5iv, so I
pulled the printed correlation labels out of the vector figures in the arXiv PDF with a small script
(`tools/pdf_text.py`; output in `extracted/codevilla2018.txt`). Same for the Foster et al. PDF (`extracted/foster2024.txt`).
"Checked" means I read it in the source. Where I calculated something myself, it says "my calculation".

## Verdicts in one table

| Claim | Verdict | What the source says (location) |
|---|---|---|
| 1 | supported | 45 models (Sec. 4.2), 11 checkpoints 2K to 500K (Sec. 4.1); Fig. 2 and Fig. 3 panels read "Correlation -0.39" (Town 2, best 50 %). The "better half" is chosen by the offline metric itself, separately for each metric. |
| 2 | supported | Fig. 3 panel labels: absolute -0.61, speed-weighted -0.57, "Cumulative error, 64 steps" -0.61, classification @0.03 -0.65, TRE @0.1 -0.64. |
| 3 | **corrected** | Numbers right (Fig. 2: -0.39, -0.54, -0.77). But "data mattered more than the metric" holds only in this condition; see below. |
| 4 | supported | Sec. 4.3, paragraph "The effect of validation data". Authors' inference. |
| 5 | supported | Supp. Sec. 3 figure labels: Town 1 best half 0.21 (MSE) to 0.55; all models Town 1 0.75 to 0.83, Town 2 0.56 to 0.75. Source flaw: supp. captions of Figs. 1, 4, 6 say "absolute error" though the axes and file names show steering MSE. |
| 6 | supported | Sec. 4.3 case study, Fig. 5. No numbers given for the two models; trajectories are in Town 1. |
| 7 | supported | Sec. 4.5, Table 3. I recounted the table: TRE right 10 of 12, MSE 6 of 12. Caveat below. |
| 8 | supported | Fig. 4: 0.80 and 0.77 (and 0.44). Caption names no town; the figure files are "Town02". Town 1 (supp. Fig. 3): 0.98 and -0.87. |
| 9 | supported | Sec. 4.4, Table 2: 78.0/81.8/83.3 and 32.4/49.3/50.7; text says turns give the largest separation. Models are Xu et al.'s 4-way classifiers. |
| 10 | supported | Table 1, Sec. 3.3 (fraction with error at least alpha times the true value); alpha = 0.1 set in Sec. 4.5; sigma = 0.03 in the axis labels. |
| 11 | supported | Sec. 4.5, Fig. 4(a): "10% to 100% decrease". The range mixes two rules (my calculation below). |
| 12 | supported | App. G, Table 29: all three triples match; last beats lowest-validation-loss in 8 of 8 rows. |
| 13 | supported | App. G: "epoch 100-300"; loss rising with success "in many cases" (Fig. G.1). The 30 % split run is low-dim BC-RNN: Square 80.7 vs 2.7, Transport 64.0 vs 0.7. |
| 14 | supported | Sec. 3.3, App. B.2: 50 rollouts every 20 (image) or 50 (low-dim) epochs, maximum over training, mean of 3 seeds; Sec. 4.7: final checkpoint, 30 rollouts. |
| 15 | supported | Sec. 4.5 last sentence; Lesson L3 heading. PMLR 164:1678-1690 confirmed. |
| 16 | supported | Sec. VI-B, Table I, means over 3 tasks: MSE r 0.308, MMRV 0.375; visual matching r 0.924, MMRV 0.056. PMLR 270:3705-3728 confirmed. Only 6 policies per task. |
| 17 | supported | Sec. III-B, Eqs. 1 and 2; range 0 to 1, lower better. |
| 18 | supported | App. E.1, Fig. 17 (Pour Water): 30 validation demos per environment or object; pairs r -0.98, rho -1.00; objects r -0.73, rise at 16 objects. "Only weakly" is the researcher's word for r = -0.73. Points are data-scaling levels, not checkpoints. |
| 19 | supported | App. E.1: LoRA 0.0049 vs 0.006; scores 0.72 vs 0.9 (Sec. 6, Table 2a). |
| 20 | supported | App. E.1 (debugging tool for obvious problems); Sec. 3 tester scores 0 to 3, random interleaving; App. C final checkpoint. |
| 21 | supported | arXiv 2606.27375v1 (25 Jun 2026) exists. Fig. 8, n = 16: action error r -0.89, rho -0.93; training loss r -0.84, rho -0.83; validation loss r -0.04, rho 0.00 (p 1.000). The 16 points mix DiT and VLA at three batch sizes; validation loss is the flow-matching loss, action error is L2 of sampled chunks; 50 trials per task. |
| 22 | supported | Sec. 3.4 text (figure itself not seen). |
| 23 | supported | arXiv 2611.00283v1 (dated 30 Oct 2026, one author) exists. Table 3: ACT -0.88, pi0.5 -0.55, SmolVLA -0.68, DP -0.48 and -0.23; episode-count baseline for ACT -0.89; within budget +0.28 to -0.22, none significant; Sec. 5.7: SmolVLA seeds started 2, 3, 6, 10 of 10. Source flaw: Table 3 rows sum to 75 checkpoints, not 82. |
| 24 | supported | Sec. 5.3 (+0.90 at 200 episodes); Sec. 5.4: untrimmed Diffusion Policy, 3 of 110, 96 % never left rest (SmolVLA 47 %, trimmed DP 9 %). |
| 25 | supported | Sec. 6: loss up 2.0 to 5.3 times; action error down up to 29 % (untrimmed DP, 200 episodes). |
| 26 | supported | Location: plus or minus 15 points is in Sec. 8.4 and Sec. 9 and is never defined; advice is in Sec. 8.5. The stop-training advice is against hold-out loss for diffusion and flow policies specifically. |
| 27 | supported | arXiv 2610.40828v1 (29 Oct 2026) exists; page author list garbled. r +0.08, p 0.72 is for insert-given-pick rate over 21 iDP3 checkpoints; five motion-quality metrics r -0.70 to -0.80, iDP3 only; none tracked pick rate. |
| 28 | supported | Abstract; Sec. 5 takeaway heading "Offline metrics shortlist, not replace". |
| 29 | supported | Sec. 5.2 (last 10 checkpoints, every 50 epochs, plus best); App. C.1.2 (12-hour budget, last checkpoint; IBC by minimum training-set action MSE); App. C.2.2 (1000 epochs, last). |
| 30 | supported | Fig. 6 caption (smooth energy loss, no gain in action accuracy) and Sec. 4.4 (checkpoint selection difficult). |
| 31 | supported | Sec. 3.4 "Training Epochs": past 95 % training token accuracy, 27 epochs. |
| 32 | supported | Sec. 5.4, Table 2: bf16 71.3, int8 58.1, int4 71.9 (8 Bridge tasks, 80 rollouts each). Location fix: per-task Table 5 is App. B.1.3 (B.3 is data-efficient adaptation); blocking: App. D.4, Table 11 (70.0, 74.4, 68.8). |
| 33 | supported | Sec. 5.2, Fig. 6: one training run, Walker2d, H = n = 500. |
| 34 | supported | Abstract and Sec. 2; NeurIPS 2024 per arXiv comments. Theory. |
| 35 | supported | Sec. 6.2, Table 2: SoftOPC R2 0.91, rho 0.94; TD error 0.17, 0.48; App. E.4 Table 3: 16.67 to 90.71 %. Needs success/failure labels. |
| 36 | supported | arXiv 2505.19017v1, no venue; Table 1, 4 policies, 3 tasks: r 0.942, MMRV 0.044; WAN2.1 14B, 8 H800, about 11 h. |
| 37 | supported | Sec. 3.3.2, Table 2: 45.5, 63.3, 60.0 %; 36/37, 15/15, 23/24; 40, 57, 53; thresholds MAE 0.1 and MDCL 0.7. Per-network Spearman only in Fig. 9 (not read). |
| 38 | supported | Sec. 3.3.2 and 4.2. Authors' explanation. |
| 39 | supported | Table 2: PDM-Open OLS 86, CLS-R 54 (CLS-NR 50); PDM-Closed OLS 42, CLS-R 92 (CLS-NR 93), IDM-based; Sec. 5. |
| 40 | supported | Sec. 4.1: 37 and 114 planners; Fig. 4 caption: PDMS better than OLS for every planner type; no coefficients in the text. |
| 41 | supported | Sec. 2.1, Theorem 2.1 (credited to Ross and Bagnell 2010), tight; PMLR 15:627-635 confirmed. |
| 42 | **corrected** | Lower validation loss is stated only for Pong; elsewhere all policies reach near-zero validation error. See below. |
| 43 | supported | Sec. 3.2, Table 1 (from Wang et al. 2019, not opened): 0.834/0.989, 2.94/1.30, 6.49/3.38; also distance 144.92/268.95; no units. |
| 44 | supported | Sec. 5.3, Table 3: CILRS 23/26/42 %, with ImageNet 4/12/38 % (empty/regular/dense). |
| 45 | **corrected** | Completion time is not among the recommendations. See below. |
| 46 | supported | Sec. 4.2 Example 6 (posterior .11); App. 6.4.2 (13/20 vs 14/20, no winner). My normal approximation also gives 0.11. |
| 47 | supported | Sec. V-C1, Table I: 19 trials (FoldRedTowel 0.56 vs 0.92), 8 trials (CleanUpSpill 0.28 vs 0.80, one policy in vs out of distribution); Sec. V-C2: 59 vs 68 % needs N_max 500. |
| 48 | supported | Sec. V-B1, Fig. 3; Sec. V-C2. |

All 48: checked. No claim was unsupported and every source could be opened.

## The three corrections

**Claim 3.** The comparison "validation data mattered more than the metric" is the researcher's, not the authors'. It
holds in the main-paper condition (Town 2, better half: three cameras 0.77 against the best metric 0.65). In the
supplement it does not: Town 1, better half, MSE went from 0.21 to 0.45 (noise) and 0.50 (three cameras) while other
metrics reached 0.53 to 0.55; all models, Town 1: 0.75, 0.83, 0.77 against metrics 0.75 to 0.83; Town 2: 0.56, 0.60,
0.74 against 0.56 to 0.75. The authors say only that both avenues improve the correlation (Sec. 5). Their one-line lesson,
which stands: "Proper validation data should therefore include examples of recovery from perturbations." (Sec. 4.3)

**Claim 42.** de Haan et al. state that all policies reach near-zero validation error (Sec. 3.2) and that losses are
frequently near zero for both the original and the confounded policy (App. E). A lower validation loss for the policy
given the previous action is stated only for Pong. Fig. 4 shows MountainCar, Hopper and Pong; App. E adds Enduro and
UpNDown. Correct statement: given the previous action, policies reached similar (in Pong lower) validation loss but
consistently lower reward.

**Claim 45.** Kress-Gazit et al. recommend success criteria fixed in advance, matched and documented starts, blind
interleaved A/B runs, counts with intervals or Bayesian estimates, rubrics and failure analysis. Their performance metrics
(Sec. 3.2) are STL robustness and SPARC smoothness; time to finish appears only inside examples, never as advice.

## Small location and detail fixes (verdict still "supported")

- Claim 26: the plus or minus 15 points is in Sec. 8.4 and Sec. 9, not 8.5, and the paper does not say what it is.
- Claim 32: Table 5 is in App. B.1.3; the blocking-control result is App. D.4, Table 11.
- Claim 47: the 19 and 8 trials are Sec. V-C1; the CleanUpSpill pair is one policy in two conditions, not two policies.
- Claim 7: one CARLA evaluation is 25 trials (Sec. 4.1), so success rates move in steps of 0.04. Two of TRE's ten "hits"
  are ties (Town 1, data type 0.84/0.84; architecture 0.76/0.76) and several Town 2 winners differ by one or two trials.
- Claim 21: the 16 points are not one training run; they mix two model families and three batch sizes.
- Venues I confirmed on the publisher page: PMLR 164:1678-1690 (robomimic), 270:3705-3728 (SIMPLER), 270:2679-2713
  (OpenVLA), 15:627-635 (Ross). Seen only in search results, not opened: ICLR 2025 (Lin), RSS 2025 (Snyder), Empirical
  Software Engineering 26(5) article 90 (Haq), CoRL 2023 PMLR 229 (Dauner), NeurIPS 2024 Datasets and Benchmarks (NAVSIM).
  Not re-checked: ECCV 2018, ICCV 2019, NeurIPS 2019 (Irpan, de Haan), RSS 2023 (Chi). The three 2026 papers are arXiv
  preprints with no venue.

## My calculations (not in the sources)

- Robomimic Table 29, drop from the best checkpoint: lowest validation loss 57.0 to 100 % (relative), last checkpoint
  11.9 to 34.7 % (6.0 to 17.3 points). The "best" is the maximum of many noisy 50-rollout estimates, so these gaps are
  somewhat inflated (my inference).
- Haq Table 2: small-error scenarios were 53, 75 and 66 of 90; of these 40, 57 and 53 failed online (75 to 80 %). Online
  violations overall: 76, 72, 76 of 90 (80 to 84 %).
- Ten runs: at 50 % one standard error is about 16 points; 95 % Wilson intervals are 24 to 76 % for 5/10 and 49 to 94 %
  for 8/10.
- r = -0.89 from 16 points has a 95 % interval of about -0.71 to -0.96 (Fisher transform, treating points as independent).

## Does the conclusion hold for this project?

Partly. Three factual slips first:

1. "In robomimic the lowest-loss checkpoint was 10 to 100 % worse" mixes two rules: lowest validation loss was 57 to 100 %
   worse, the last checkpoint 12 to 35 % worse (my calculation).
2. "Haq: 40 to 57 of 90 low-error scenarios still failed" is wrong: 40 to 57 failures were among 53 to 75 low-error
   scenarios. And those networks failed online in 80 to 84 % of all scenarios, so a high base rate drives the misses.
3. "Ten runs give about plus or minus 15 points" understates it: that figure is undefined in Cowling and is about one
   standard error; a 95 % interval is about plus or minus 26 to 30 points. Ten runs per candidate decide only very large
   gaps.

Then the transfer problems:

- **Loss versus action error.** The "validation loss uncorrelated" results of 2026 are about diffusion and flow-matching
  losses, which are not errors on the action. The team's model is a move-name classifier plus a step regressor, where the
  loss nearly is the action error. Not choosing on validation cross-entropy is still wise, because cross-entropy can rise
  while accuracy improves (Guo et al. 2017, below), but that is a different reason.
- **Between runs versus within a run.** The -0.93 of Allshire et al. spans model families and batch sizes. The team's real
  choice is among checkpoints or seeds of one model on the same 208 demonstrations, which is Cowling's "within one budget"
  case, where the correlation was not significant. So offline numbers will mainly catch broken runs and set the training
  length; they will rarely pick the winner.
- **Off-path validation pictures.** 0.39 to 0.77 holds only in Codevilla's main condition (see claim 3). One forearm
  camera has no side cameras; recovery pictures must be recorded by hand (start beside the line, guide back) or made by
  warping recorded pictures of the flat sheet (Bojarski et al. 2016, below). Either set will be small and noisy.
- **Stop-and-go is an advantage.** With blocking control OpenVLA's latency effect disappeared (App. D.4), so per-step
  predictions are not confounded by timing here. Compounding errors over the tens of steps of a track still are (Ross).
- **Last checkpoint by default** rests on large models with heavy augmentation. For a CPU-trained model on six sheets, set
  the budget once from the held-out-sheet accuracy curve (as Cowling advises: action error to decide when to stop), then
  keep the last checkpoint of that budget.
- **Small samples behind the correlations:** SIMPLER 6 policies per task, Allshire 16 points, Abbas 21 checkpoints.
- **Time to finish** (the exam's second criterion) was measured by no study; only robot runs can rank it.

The practical judgement (held-out sheets; within-45-degree, balanced-accuracy, stop and per-segment scores; a small
recovery set; never choose on validation loss; distrust picture history until the robot agrees; two or three candidates
decided on the robot with the same starts, interleaved runs, finishes, progress, failure type and time) fits the evidence.
I would add: pool runs across tracks, use a sequential test or a Bayesian comparison rather than a fixed count, record a
path-to-line distance for every run (see Hussenot below), and count stalls and back-and-forth dithering separately, the
stop-and-go version of Cowling's "never started".

## Evidence the researcher missed

- **Hussenot et al. (2021). Hyperparameter Selection for Imitation Learning. ICML 2021.**
  https://ar5iv.labs.arxiv.org/html/2105.12034, Sec. 2.1 and 3.1, Figs. 2, 3 and 8 (checked). Over 10,000 agents in 9
  environments, held-out action MSE was the only fully offline proxy, and it was beaten by state-distribution divergence
  measured on rollouts without any reward; validation or training demos made little difference. Measured. They call action
  MSE "a fully offline metric unaware of the system dynamics".
- **Guo, Pleiss, Sun, Weinberger (2017). On Calibration of Modern Neural Networks. ICML 2017.**
  https://ar5iv.labs.arxiv.org/html/1706.04599, Sec. 3, Fig. 3 (checked). On CIFAR-100, test error fell from 29 to 27 %
  while "NLL overfits during the remainder of training". Measured; the overconfidence explanation is theirs.
- **Bojarski et al. (2016). End to End Learning for Self-Driving Cars. arXiv 1604.07316.**
  https://ar5iv.labs.arxiv.org/html/1604.07316, Sec. 2, 5.2, 6, 7.1 (checked). Closed-loop test built from recorded video:
  each picture is warped to the car's new pose (flat-ground assumption), with a virtual intervention beyond 1 m and an
  autonomy score. A method, not validated against road results in the paper. An A4 sheet is flat, so a warp-based replay
  is cheap on a CPU (puck and rod would distort); any rule-based labels must stay out of training (imitation only).
- **Mandlekar et al. (2021), robomimic, Sec. 3.2 and App. B.1 (checked):** the PH datasets have 200 demonstrations
  (Transport 300), a 10 % validation split by demonstration and GMM action heads. Closest data scale to the team's 208.
- **Kim et al. (2024), OpenVLA, App. D.4, Table 11 (checked):** blocking (stop-and-go) control removed the precision gap
  (70.0, 74.4, 68.8 %).
- **ACT repository README (Zhao), https://github.com/tonyzhaozh/act, "ACT tuning tips" (checked, not peer reviewed):**
  "Success rate and smoothness can improve way after loss plateaus." For real data it advises 3 to 4 times the plateau.
- **Abbas et al. (2026), App. F (checked):** a recipe for choosing offline metrics honestly: list fixed before robot
  results, Benjamini-Hochberg q 0.05, same sign when any one policy is left out, at least ten checkpoints.
- **Not opened, worth a look:** Kadian et al. (2020), Sim2Real Predictivity; Laskey et al. (2017), DART (noise while
  demonstrating, for recovery data); Paine et al. (2020), Hyperparameter Selection for Offline RL.

## Files

- This note: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/literature/offline_vs_closed_loop/check/note.md`
- Extracted PDF text: `.../check/extracted/codevilla2018.txt`, `.../check/extracted/foster2024.txt`
- Helper scripts: `.../check/tools/pdf_text.py`, `pdf_images.py`, `pdf_streams.py`
- The fetch tool saved the two PDFs itself in `C:\Users\e7dc\.claude\projects\D--AI-Project\997601d4-4a2e-4eb2-91d1-890ff53278fd\tool-results\`.
