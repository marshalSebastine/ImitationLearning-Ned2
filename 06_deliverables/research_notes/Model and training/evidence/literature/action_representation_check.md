# Check of the action-representation claims (classes, continuous actions, bins, mixtures)

Key: `action_representation`. Checked 9.11.26 by a Claude sub-agent acting as a sceptical checker, for Team M (Emranul
Hoque and Marshal Sebastine). Literature only: nothing here was tested on the robot or on the team's data.

The note under check is `../note.md` (same folder level as this `check/` folder). Its 40 claims and its conclusion are
judged below.

---

## 1. How I checked

- I downloaded every cited page myself and searched its text for each number, digit by digit. Pages and my plain-text
  versions are in `check/sources/<paper>/` (`page.html` or `page.pdf`, `text.txt`; for PDFs also `joined.txt`). The two
  small extraction scripts are in `check/scripts/`.
- The two PDFs (Giusti et al. 2016; Bishop 1994) were read through my own rough text extractor (no PDF library was
  installed). The numbers I report from them come from the extracted text layer, and they read cleanly.
- For Michelmore et al. (2018) I also looked at Figures 1 and 2 (the two network diagrams) as images.
- Versions matter in three places:
  - **Diffusion Policy**: ar5iv shows arXiv v5, which arXiv's own comment calls an extended journal version of the RSS
    2023 paper. I also opened the RSS XIX PDF (https://roboticsproceedings.org/rss19/p026.pdf). The numbers I checked are
    the same in both, but the labels differ: RSS has the multimodal figure as Fig. 4 (v5: Figure 3), the real Push-T table
    as Table V (v5: Table 6), the timing in Section III-D (v5: 3.4) and the denoising steps in Appendix D, Table VI (v5:
    A.4, Table 7). The researcher's locations follow v5.
  - **Implicit BC**: ar5iv labels the real-robot table "Table 4". In the published PMLR PDF
    (https://proceedings.mlr.press/v164/florence22a/florence22a.pdf, opened) it is **Table 6**.
  - **FAST**: the arXiv HTML page is v1 (16 Jan 2025); no venue on the page.
- Venues I confirmed on the venue's own page: RT-1, ACT and Diffusion Policy (roboticsproceedings.org, DOIs
  10.15607/RSS.2023.XIX.025, .016, .026); robomimic (PMLR 164, pages 1678 to 1690), Implicit BC (PMLR 164, 158 to 168),
  Stop Regressing (PMLR 235, 13049 to 13071); BeT and Pearce et al. (ML Anthology pages). From arXiv comments (opened):
  Xu et al. "camera ready for CVPR2017"; TrailNet "IROS2017 conference submission"; Eraqi et al. NIPS 2017 Machine
  Learning for Intelligent Transportation Systems workshop. **Not opened** (search listings only): AAAI-18 for
  Rahmatizadeh et al. (1603.03833) and ICRA 2018 pages 3758 to 3765 for Rahmatizadeh et al. (1707.02920).
- Every verdict below is **checked** (read in the source by me). Claim 38 is the researcher's own calculation: I redid the
  arithmetic. Claim 39 comes from the team's memory file, which I read.

---

## 2. Verdicts at a glance

| id | verdict | short reason |
|---|---|---|
| 1 | supported | Sections 2 and 4 say this |
| 2 | supported | numbers right; the 6 s rule belongs to the simulation metric (see note) |
| 3 | supported | beta 15 degrees, cameras 30 degrees, Table I 85.2 / 86.5 / 82.0, control law as stated |
| 4 | **corrected** | the accuracies are offline (IDSIA dataset), not on the 250 m trail; Giusti row is a re-implementation |
| 5 | supported | Section IV-A; no comparison experiment |
| 6 | **corrected** | the "no single parametric family" reason is for binning, not for the Gaussian smoothing |
| 7 | supported | Table 2 right; but "plain classification" there decodes by expected value (see note) |
| 8 | **corrected** | results right; the paper does not say dropout differs; layer widths and decision rule differ |
| 9 | supported | Abstract and Section 2 |
| 10 | supported | Section 6, Table 1, Figures 12 to 15 |
| 11 | supported | unnumbered tables, numbers right |
| 12 | **corrected** | the corner-push argument is a design conjecture in the method section, not in the results subsection |
| 13 | supported | Section 5.1, 5.2; 3 Hz in Fig. 1 caption and Section 5.1 |
| 14 | supported | Table 13 |
| 15 | supported | Appendix D.4 and D.5 |
| 16 | supported | numbers right; the paper does not define what "auto-regressive actions" conditions on |
| 17 | supported | Sections 2.1 to 2.4, 3.1, Appendix A, B.2, Table 4 |
| 18 | supported | Table 1 |
| 19 | supported | Table 3; uniform-bin variant indeed undescribed |
| 20 | supported | Table 1; the "MLP" rows are "MLP Sieve" |
| 21 | supported | Table 3 |
| 22 | supported | Section 2 |
| 23 | supported | Sections 2 (C2), 3.2 |
| 24 | supported | Table 26; I redid the column alignment independently |
| 25 | supported | Section 4.1 |
| 26 | **corrected** | the low-noise trick still samples which mode; not "near-deterministic" |
| 27 | supported | numbers right; real-robot table is Table 6 in the published PDF |
| 28 | supported | numbers right; Push-T is scored by target-area coverage, not success |
| 29 | supported | Tables 1 and 6 (v5) |
| 30 | supported | Figure 3 (v5) caption; qualitative |
| 31 | supported | Section 4.3; qualitative, no count there (a count is in Section 6.1) |
| 32 | supported | Section 4.3; argument only |
| 33 | **corrected** | the "mismatch" with Table I is a misreading; Table I gives 50 and 20, mean 35 |
| 34 | supported | Section IV-B (and VI-B) |
| 35 | supported | Table I, Sections III and IV |
| 36 | supported | Section IV, Figure 3 |
| 37 | supported | Sections 4, 5.1, 5.2.1 |
| 38 | supported | arithmetic correct (0.383 cm sideways) |
| 39 | supported | memory file, 6.11.26 night entry, finding (2) |
| 40 | supported | Section 3.4 (10 steps, 0.1 s, RTX 3080) against Appendix A.4 / Table 7 (16 steps); no CPU timing |

No claim was unsupported or unopenable. Six need correction (4, 6, 8, 12, 26, 33).

---

## 3. Details

### Corrections

**Claim 4 (TrailNet, Smolyanskiy et al. 2017, Section V-A, Table I).** Correct statement: TrailNet reached 84%
classification **accuracy on the IDSIA Swiss Alps dataset** (offline; the authors held out trails 003, 008 and 010) and
**100% autonomy on a 250 m Pacific Northwest trail** (share of time without a human override). Other rows: Giusti et al.'s
network 79% / 80%, ResNet-18 92% / 88%, SqueezeNet 86% / 98%, mini AlexNet 81% / 97%. The Giusti row is TrailNet's own
re-implementation (tanh replaced by ReLU, footnote 5) trained on TrailNet's own split, so its 79% is not comparable with
Giusti's own 85.2%. Loss and architecture are confounded in Table I: TrailNet and SqueezeNet used all three loss terms
(cross-entropy, entropy reward, side-swap penalty), ResNet-18 and mini AlexNet only cross-entropy. Steering (Eq. 3) adds
the right-minus-left differences of **both** heads (view orientation and lateral offset), each weighted by 10 degrees.

**Claim 6 (Xu et al. 2017).** The reason "it can be hard to find one parametric family that fits all scenarios" is given in
Section 3.4 for **discretising into bins instead of regressing the parameters of a distribution**, not for the label
smoothing. The smoothing is a separate sentence in Section 5.2: a Gaussian with standard deviation 0.5 over nearby bins,
with **no unit stated** ("0.5 bins" is an inference; for the linear scheme 1 bin = 1 degree, so it would be the same).
Each scheme has 180 bins. Table 3 values are as stated (Random Guess 1.86, Linear -2.82, Log -3.66, Data-Driven -4.83);
Table 2's caption calls the metric log perplexity, lower is better; no regression comparison; Figure 7(c) shows the two
peaks (turn left or go straight).

**Claim 8 (Michelmore et al. 2018, Sections IV-A, IV-C, V-A; Figures 1 and 2).** The results are as stated: regression
about two crashes per lap and jerky; classification 67% accuracy, zero to one crash per lap, smoother but **still jerky**.
But the paper does not say the networks differ in dropout: dropout (p = 0.05 after every layer but the first and last) is
described only for the regression network. What the paper does show differing: the fully connected layers are
1152-100-50-10-1 in the regression network (Fig. 1) and 1152-512-256-200 in the classification network (Fig. 2), and the
decision rule differs (regression: mean of 128 dropout passes; classification: the most frequent per-pass argmax). The
conclusion "not a controlled comparison" stands, for these reasons.

**Claim 12 (Rahmatizadeh et al., AAAI 2018).** The two-push-orders argument (averaging makes the gripper push the box's
corner) is in the section "The neural network based robot controller", offered as a **conjecture that motivated choosing
the MDN**, before any experiment. The subsection "The benefits of LSTM and MDN" reports and summarises the success rates
without this explanation. "No separate test" is correct. (A side note from the physical-robot section: the trained
LSTM-MDN itself pushes the box's corner on purpose to rotate it.)

**Claim 26 (robomimic, Appendix D.2, Appendix J, Table 21).** Modes 5, 10 and 100 and minimum spreads 1e-2, 1e-4, 1e-6
made no significant difference: correct. At rollout every mode's spread is set to 1e-4, which the authors say "amounts to
sampling one of the GMM modes" (Appendix J). Which mode is still drawn at random by the mixture weights, so the policy is
**not near-deterministic**; only the spread within a mode is removed. Table 21 numbers as stated (Transport PH states 71.3
against 64.7; Square MH pictures 76.7 against 68.0). Two cells are slightly higher without the trick (Square MH states
79.3 against 78.0; Transport MH pictures 43.3 against 42.0).

**Claim 33 (ACT, Zhao et al. 2023, Section VI-B, Fig. 6(c), Table I).** The first two sentences are correct (scripted:
almost no difference; human: 35.3% to 2%; two simulated tasks; 50 demonstrations each, Section V-B). The third sentence
is a misreading. Table I gives each simulated subtask as [scripted | human]. ACT's **final** human-data success is
**50%** (Cube Transfer, "Transfer") and **20%** (Bimanual Insertion, "Insert"): mean 35%, which matches 35.3% (three seeds).
82 and 60 are ACT's human-data rates for the earlier subtasks "Touched" and "Lifted" of Cube Transfer. The text's gains
over the best baseline (59, 49, 29 and 20 points, Section V-C) confirm this reading. Also: the variant without the CVAE is
trained with an L1 loss on action chunks, and the full ACT decodes deterministically at test time (style variable set to
zero, Section IV-B).

### Supported, with notes worth keeping

- **Claim 2 (PilotNet).** The 6-second penalty per intervention is the simulation metric (Section 7.1). The on-road 98% is
  "fraction of time the car steers itself", lane changes and turns excluded (Section 7.2). The 10 miles on the Garden
  State Parkway had zero interventions.
- **Claim 3 (Giusti et al.).** Also in the paper (control section): on wide trails the real quadrotor followed the trail
  for a few hundred metres; on narrow trails it crashed often, because the classifier only corrected sideways shifts of
  about a metre. This is a qualitative, preliminary field result.
- **Claim 7 (Eraqi et al., Table 2).** The "plain classification" baseline is classification followed by a softmax
  **expected-value** refinement (the DEX method of Rothe et al., their reference [20]), not argmax. The sine-wave method is
  trained by a least-squares (RMSE) fit of the predicted waveform. So this is not a test of argmax classes against
  regression. The regression row (17.77) is ResNet-152, the best of Table 1.
- **Claim 16 (RT-1).** Numbers right (85, 71, 67, 65; 36 ms against 15 ms). The paper only says "auto-regressively
  conditioning on actions, as used in" Gato, Decision Transformer and Multi-Game DT; it does not say whether this is within
  one step's 11 numbers or across past steps. The parenthesis in the claim is the researcher's reading.
- **Claim 20 (Pearce et al.).** The second group is the "MLP Sieve" architecture (Basic MLP rows exist only for the
  diffusion variants). K-Means Transformer had a single seed (Appendix B.2). The authors' own K-Means+Residual reached
  0.34 against BeT's reported 0.44, which they could not close (Appendix B.2).
- **Claim 24 (robomimic Table 26).** I aligned the columns from the cell order (low-dim rows have no shallow-conv value,
  picture rows no larger-MLP value) and checked them against Section 4.4's ranges (larger learning rate 35 to 63%; shallow
  conv 25 to 62%; smaller RNN 3 to 58%; no GMM low-dim Transport MH 58%). Same values as the researcher.
- **Claim 27 (Implicit BC).** In the published PMLR version the real-robot results are Table 6. The real task is the
  closest published analogue to ours: a cylindrical end-effector pushing blocks on a table, the robot held to a plane,
  RGB pictures only, 2D actions at 5 Hz, teleoperated demonstrations by mouse (Appendix C.3).
- **Claim 28 (Diffusion Policy).** Push-T numbers are **target-area coverage**, not success rates (Section 5.2). Square and
  Transport numbers are success rates.
- **Claim 31 (Diffusion Policy).** Section 4.3 gives no count. Section 6.1 does: on real Push-T, LSTM-GMM got stuck near
  the T block in 8 of 20 runs and IBC left it too early in 6 of 20; the authors say not removing idle actions contributed to
  the baselines overfitting small actions and getting stuck.
- **Claim 38 (own calculation).** 1 cm x sin 22.5 degrees = 0.383 cm sideways. The full end-point error is 0.390 cm
  (2 x sin 11.25 degrees) and the forward shortfall 0.076 cm. Assumes 8 evenly spaced directions and an exact 1 cm step.
- **Claim 39 (team memory).** The file says exactly this, in the 6.11.26 night entry, finding (2). It describes the exact
  replay before mass collection, not the final data. The same file (9.11.26 entry) gives the final data as 211 kept
  episodes and 18,522 pictures with 3 old episodes unscored, and lists 11 move classes; the task text says 208
  demonstrations and 10 move names. Worth reconciling before training.
- **Claim 40 (Diffusion Policy).** The RSS version also lists limitations: higher compute and latency than simpler methods
  such as LSTM-GMM. The real Push-T CNN model has about 67 million diffusion and 22 million vision parameters (Table 7).

---

## 4. Challenge to the conclusion

The quoted numbers are right, apart from the points in section 3. My concern is how far they carry to our set-up: one
forearm camera, one picture per decision and no memory, about 208 demonstrations from two people, CPU training, 1 cm
stop-and-go steps and unseen tracks. Below, "measured" means a paper's experiment; the rest is my judgement.

1. **Most of the big gains came with memory or history.** Rahmatizadeh's 0% to 95% on push to pose compares LSTM-MSE with
   LSTM-MDN. With a feedforward (memoryless) network, which is what a one-picture model is, the MDN reached only 15%
   (measured). robomimic's GMM results are for BC-RNN, a recurrent policy over 10 steps. BeT's bins plus offsets ran on a
   transformer over 5 to 10 past observations; with an MLP trunk instead, BeT fell to 0.90, 0 and 0.05 of its full score on
   CARLA, block push and kitchen (Table 3, measured). RT-1 sees six past pictures.
2. **States are not pictures.** robomimic's large drop without the GMM (Transport MH 65.3 to 27.3) is with ground-truth
   object states. With pictures, the multi-operator Transport hardly changed (42.0 to 41.0) and Square fell from 76.7 to
   61.3 (Table 26, measured). The only picture-based head comparison in the set, Pearce et al.'s Counter-Strike with a
   ResNet-18, had MSE at 17.8 against 14.7 for per-number bins and 16.8 for BeT-style bins plus offsets (measured). In
   Implicit BC, an MDN from pictures reached 10% where MSE reached 87% (measured). So, from pictures, the evidence is mixed.
3. **"Several operators" is not what ACT shows.** ACT's human data were recorded by one person ("even though a single
   person collects all of the demonstrations", Section V-B). Its result says human data are noisy, even from one operator.
   robomimic's single-operator results point the other way (no GMM needed). Two people are much closer to one than to
   robomimic's six operators of deliberately mixed skill.
4. **Whether our labels are multimodal at all is untested.** Following a line mostly has one right move per picture. Real
   forks are rare: the crossing on the loop sheet, the gaps on the dashed loop, and winning back a drifted puck. Much of the
   disagreement is probably boundary noise between neighbouring classes, and each person's habit in twist, lean and hand
   turn (my judgement). This can be measured cheaply on our own data before choosing a head (see section 5, last item).
5. **A per-class offset head does not remove the joint-mix averaging that the judgement worries about.** In Pearce et al.'s
   words, "The residual learns the 'average' of the actions that fall within each bin" (Section 2). If the two operators
   reach the same move name with different mixes of twist, lean and hand turn, the offset head averages them, exactly as
   plain MSE would. In Pearce's controlled re-implementation, bins plus offsets were below MSE (kitchen 0.34 against 0.69 and
   0.23 against 0.50, measured).
6. **Argmax and soft labels are not what the cited robot papers tested.** BeT, robomimic and RT-1 sample from their output
   distribution. Giusti and TrailNet steer with probability differences, which is a soft average. Eraqi's classifier
   decodes by expected value. Only Michelmore's classifier used an argmax-like rule, in an uncontrolled comparison. Soft
   labels have support only from authors' statements without ablations (Xu; TrailNet says label smoothing plus an entropy
   reward were critical to stable flight) and from value functions in reinforcement learning (Farebrother et al.). Argmax is
   still a sensible choice, but it is a choice, not a finding.
7. **Coarse classes on real robots: weaker than it sounds.** Giusti's real flights were qualitative and preliminary (a few
   hundred metres on wide trails, frequent crashes on narrow ones). TrailNet flew one 250 m trail and argued that three
   heading classes cannot correct a sideways offset; our 8 directions include sideways moves, so that particular worry is
   smaller for us.
8. **RT-1's bins-against-Gaussian result** comes from 130,000 demonstrations and over 700 instructions, with sampling; its
   multimodality explanation is untested by the authors.

What the evidence **does** support for our case:

- MSE on a variable with genuinely separate right answers gives a wrong average (Bishop, measured on a two-link arm).
- Offline scores do not predict closed-loop success (TrailNet Table I; Eraqi against Michelmore; robomimic Section 4.5,
  where picking a checkpoint by validation loss lost 10% to 100% of the best success). So the comparison must be in closed
  loop on held-out tracks, and the checkpoint should be picked on the robot too.
- Single-step policies trained on pauses get stuck (Diffusion Policy Section 6.1: LSTM-GMM stuck in 8 of 20 real Push-T
  runs; ACT Section V-C: prior single-step methods could pause indefinitely). So "small motion" should be handled **before**
  training (drop, merge or down-weight near-zero rows, or test the stall rate on the robot early), not only watched.
- In the most similar real task (Implicit BC: planar pushing from pictures, 95 to 502 teleoperated demonstrations), plain
  MSE did badly: 35.0, 55.0, 6.7 and 19.6% against 85.0, 88.3, 83.3 and 48.3% for the implicit model (measured). That argues
  against relying on MSE alone, though the winner there was an energy-based model, not classes.

**My suggested wording of the conclusion (judgement, under 200 words):** MSE gives the average action for each input
(Bishop 1994), which fails where two different answers are both right (two-link arm: 0.0578 against 0.0053). Whether that
matters for us is unmeasured: line following mostly has one right move per picture. The largest gains from mixtures or
bins came with memory or history (LSTM-MDN 95% against feedforward MDN 15% on pushing; BeT collapsed with an MLP trunk) or
with state inputs (robomimic six-operator Transport 65.3 to 27.3 without GMM; with pictures 42.0 to 41.0). From pictures,
MSE beat bins in one study (Counter-Strike 17.8 against 14.7) and a mixture failed in another (10% against 87%). In real
planar pushing from pictures, MSE did poorly (35, 55 and 6.7%). Offline scores did not predict closed-loop success. The
move-name classifier is a sensible first model because it matches our labels and is cheap on a CPU, not because the
literature shows it beats MSE here. First measure how often similar pictures carry different move names; deal with "small
motion" before training; then compare classes, classes plus offsets and MSE on one backbone, in closed loop on held-out
tracks, with enough runs per variant.

---

## 5. Evidence the researcher missed (all checked by me unless marked)

1. robomimic, pictures, multi-operator, no GMM: Transport 42.0 to 41.0, Square 76.7 to 61.3 (Table 26). The conclusion
   quotes only the state-based 65.3 to 27.3.
2. robomimic Appendix D.2: the GMM's advantage was smaller for BC-RNN than for plain BC (no numbers given). Plain BC is the
   memoryless case, like ours, so the GMM may matter more for us; the paper gives no figure to say how much.
3. robomimic Sections 4.3 and 4.5: removing pixel-shift image randomisation cost 47% and 35% (relative) on Square and
   Transport; a shallow encoder instead of ResNet-18 cost 25 to 62%; choosing the checkpoint by validation loss lost 10 to
   100%. For unseen tracks, encoder and augmentation may matter as much as the output head (my judgement).
4. BeT Table 3, trunk ablation: MLP trunk 0.90, 0, 0.05; temporal convolution 0.72, 0.01, 0.26; LSTM 0.03, 0.03, 0.04
   (fractions of full BeT on CARLA, block push, kitchen). BeT also samples its bin at test time (Section 2.4), and its CARLA
   and block-push demonstrations were scripted with designed 50/50 modes (Appendix A).
5. Pearce et al. Section 2: the residual of K-Means+Residual still averages within each bin.
6. Pearce et al. Table 3: from pictures with ResNet-18, MSE 17.8 beat per-number bins 14.7 and K-Means+Residual 16.8.
7. Implicit BC real-world pushing, PMLR Table 6: MSE 35.0, 55.0, 6.7, 19.6% against the implicit model 85.0, 88.3, 83.3,
   48.3%; and simulated pushing from pictures (Table 3): MDN 10.0% against MSE 87.0%. The researcher listed these numbers in
   the note, but the conclusion uses neither, and the optional GMM head in section 6.3 of the note ignores the MDN failure.
8. Diffusion Policy Section 6.1: LSTM-GMM stuck in 8 of 20 real Push-T runs, IBC left the block early in 6 of 20; idle
   actions not removed contributed. Removing idle actions is called common practice there.
9. ACT: human data from one person (Section V-B); no-CVAE variant uses L1 on action chunks; test-time decoding is
   deterministic (z = 0); single-step prediction (chunk size 1) scored 1% against 44% at chunk size 100 (Section VI-A,
   averaged over four simulated settings); continuous ACT beat the binned BeT and RT-1 on its tasks (Tables I and II),
   confounded by chunking.
10. Giusti et al.: the real-flight outcome (wide trails only, frequent crashes on narrow ones). Giusti and TrailNet both
    decode by probability differences, not argmax.
11. Eraqi et al.: the classification baseline decodes by expected value (DEX), so it does not compare argmax classes with
    regression.
12. Michelmore et al.: the classification network had much wider dense layers (512 and 256 against 100, 50 and 10).
13. Baselines are implementation-sensitive: BeT's MLP-with-MSE scored 0 at every kitchen level (BeT Table 1), while Pearce
    et al.'s MSE MLP Sieve scored 0.50 on four or more tasks (Pearce Table 1). One paper's MSE failure is weak evidence.
14. Data facts to reconcile: team memory gives 211 kept episodes, 18,522 pictures and 11 move classes; the task text says
    208, about 18,500 and 10 (8 directions, stop, small motion).
15. Gaps nobody filled: none of the papers tests generalisation to unseen track shapes, a 1 cm stop-and-go loop, or CPU
    training. A direct check on our own data would answer the multimodality question better than any paper: for pictures
    that look alike (for example nearest neighbours in a pretrained feature space, same sheet region), how often do the two
    operators' move names differ; what share of rows lie within a few degrees of a class boundary (from the exact angle);
    and what share is "small motion". Unchecked: this is my proposal, not a published method.

---

## 6. Files

- Researcher's note: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/literature/action_representation/note.md`
- This check: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/literature/action_representation/check/note.md`
- Downloaded sources and text: `.../action_representation/check/sources/` (one folder per paper; `arxiv_abs/` and
  `venues/` for the arXiv abstract pages and proceedings pages; `diffusion_rss/` and `ibc_pmlr_pdf/` for the published PDFs)
- Extraction scripts: `.../action_representation/check/scripts/` (`html_text.py`, `pdf_text.py`, `join_lines.py`)
- Team memory read for claim 39: `D:/AI_Project/claude_memory/project-recording-tool.md`
