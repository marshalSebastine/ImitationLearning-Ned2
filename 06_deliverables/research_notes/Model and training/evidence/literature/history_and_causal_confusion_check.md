# Check: picture histories, copycat problems and causal confusion

Key: `history_and_causal_confusion`. Sceptical check of the 38 claims and the conclusion written by the researcher
agent. Checked on 9 November 2026.

## How I checked

- I opened every cited page myself (the URL in the claim). The only exception is claim 38: the NeurIPS review page
  refused the connection.
- I read the pages with a web-fetch tool. It turns a page into text and answers questions about it. I asked for numbers
  digit by digit, and asked again for the key ones. Values shown only inside figures cannot be read this way, so they are
  marked "figure values not read".
- I did not check venues separately. Only the Wen 2021 arXiv page states its venue (ICML 2021). The pages for Wang 2019,
  Spencer 2021, Codevilla 2019 and Torne 2025 state no venue.
- "Checked" below means I read the statement in the source myself. Quotes are kept to three, each under 15 words, at most
  one per source.

## Verdicts in short

- Supported: 33 claims (1 to 15, 17, 19, 21 to 29, 31 to 37). Several carry a note.
- Corrected: 4 claims (16, 18, 20, 30).
- Could not open: 1 claim (38).
- Not supported: none.

## Verdict per claim (all checked except 38)

| id | verdict | what the source says, and where |
|---|---|---|
| 1 | supported | de Haan 2019, Sec. 1 and Sec. 3. Cloned policies fail because they mistake the causes of the expert's actions. The other factors are nuisance variables. This does no harm when training and test data come from the same distribution, but the imitator's own actions shift its inputs. |
| 2 | supported, location note | Sec. 3.2. In MountainCar and Hopper the previous action is added to the state at an index the learner does not know (in the original it is replaced by noise). In Atari a symbol for the previous action is drawn at an unknown place. Both Atari versions use two stacked frames. With the original input, reward tends towards the expert's as data grow; the confounded input needs many more demonstrations or fails. Fig. 4 shows only MountainCar, Hopper and Pong (5 runs; values only in the figure). The Enduro and UpNDown results are in Appendix E. |
| 3 | supported | Appendix E (Fig. 11). Training and validation losses are often near zero for both policies, yet the confounded one consistently gets significantly lower reward. Sec. 3.2 adds that in Pong the confounded policy even had the lower validation loss. No loss values are given in the text. |
| 4 | supported | Table 1 matches digit by digit. History: 0.834, 144.92, 2.94 ± 1.79, 6.49 ± 5.72. No history: 0.989, 268.95, 1.30 ± 0.78, 3.38 ± 2.55. Note from Wang 2019: "history" there means the car's own past positions drawn into the plan view, and perplexity is the negative log-likelihood of the expert's actions (lower is better). |
| 5 | supported | The general statement is in Sec. 1. In Sec. 3.2 the evidence about history comes from Wang 2019 and Bansal 2019. The authors' own confounded inputs are an explicit cue for the previous action, and Sec. 6 says the method was tested only in simpler synthetic settings. |
| 6 | supported | Wen 2020, Sec. 1. A copycat policy copies the expert's previous action. It needs two conditions: expert actions strongly correlated in time, and past actions easy to recover from the history. Error is low on training and held-out data, but the policy fails when it runs (driving example). Sec. 4: copying nearly maximises the training objective. |
| 7 | supported | Table 1 (Sec. 4). Humanoid 0.18 ± 0.01 vs 6.93 ± 0.32 (x10^-1). Walker2d 0.46 ± 0.02 vs 2.47 ± 0.07 (x10^-2). BC-OH is lower in all six tasks, by factors of about 3 to 38. H = 2 is stated in Sec. 4; joint positions only in Sec. 6.2. Detail: the probe is a two-layer MLP that predicts a_t from the previous 9 actions (k = 9). |
| 8 | supported, location note | Table 2 matches (BC-SO / BC-OH / authors' method). HalfCheetah -38 / 820 / 1250. Walker2d 363 / 592 / 1296. Humanoid 587 / 565 / 671. Hopper 275 / 293 / 1086. The table is in Sec. 6.3, not 6.2. Five seeds, mean ± std. The Hopper (293 ± 83 vs 275 ± 40) and Humanoid (565 ± 80 vs 587 ± 58) gaps are within one std, so "5 of 6" rests on means alone. |
| 9 | supported | Sec. 7: the gain in image-based experiments is not significant yet, and causal confusion with images is called an open question. No image results appear in the main text or the appendix. |
| 10 | supported | Codevilla 2019, Sec. 3.1: one image, the ego speed and a high-level command. Sec. 3.2: the chance that a stopped car stays stopped is "overwhelming" in the training data. This links low speed with no acceleration and causes too much stopping and hard restarts. The CILRS design is in Sec. 3.3. |
| 11 | supported | Sec. 5.3: an inertia failure means the car stays still, with no intent to accelerate, for at least 8 s before the timeout. Fig. 4 and the text: the share of such failures rises with the amount of training data. No values are given in the text. |
| 12 | supported | Sec. 3.3 gives the speed branch's purpose. Sec. 5.3 says it is not a final solution, because inertia still occurs. Sec. 6 lists inertia among the open causal-confusion problems. |
| 13 | supported | Sec. 5.3 and Table 3: 12 training runs. Dense traffic: 42% for CILRS, 38% for CILRS with ImageNet. Empty: 23% and 4%. Regular: 26% and 12%. The caption calls these estimated variances, in percent; the text says success can change by up to 42% on tasks with moving objects. |
| 14 | supported | Wen 2021, Table 1 (CARLA with speed). BC-SO 42.667 ± 8.668, BC-OH 33.000 ± 4.190, HistoryDropout 34.000 ± 2.625, DAGGER (120K) 35.222 ± 3.067, keyframe (step) 43.444 ± 0.786. H = 6 (Sec. 5); speed in the input (Sec. 6). The authors say their method does not beat BC-SO significantly here. Three policies, each evaluated three times. |
| 15 | supported | CARLA without speed: 9.222 ± 2.380, 25.667 ± 0.981, keyframe (step) 36.778 ± 5.808. MuJoCo with images (Table 2; H = 1, i.e. previous plus current 128x128 frame). HalfCheetah 615 ± 41 (BC-OH) vs 4 ± 5 (BC-SO). Hopper 740 vs 601. Walker2D 614 vs 481. |
| 16 | **corrected** | See below. |
| 17 | supported | Chuang 2022. Sec. 5.1 withholds ego speed from all methods, and Appendix B.1 sets the input speed to zero. Table 2 (3 seeds), regular / dense: BC-SO 37.6 ± 6.1 / 13.1 ± 1.8; BC-OH 67.1 ± 10.8 / 34.1 ± 7.5; residual 78.1 ± 0.9 / 52.0 ± 2.3. H = 6. |
| 18 | **corrected** | See below. |
| 19 | supported | Bansal 2019, Sec. 4.2. For 50% of the examples only the current position is kept in the past-pose input. The reason given: otherwise the network can cheat by extending past motion (stop-sign example). Sec. 6.2: model M0 is behaviour cloning with this dropout, and the later models build on it. No ablation removes it (Sec. 6.3.1). |
| 20 | **corrected** | See below. |
| 21 | supported | Spencer 2021, Sec. 1: causal-confusion errors are a more visible case of covariate shift driven by feedback. Fig. 1: a learner latches onto the previous action and keeps braking. Fig. 1 is a drawing, not an experiment. Sec. 3 also argues that de Haan's effect comes from too little data or the wrong model class, not from a true causal confound. |
| 22 | supported | Mandlekar 2021, Table 1 (low-dimensional inputs, 3 seeds, best success during training). Transport PH 71.3 ± 6.6 vs 17.3 ± 2.5. Square MH 78.0 ± 4.3 vs 52.7 ± 6.6. Lift MG 70.7 ± 3.4 vs 65.3 ± 2.5. Tool Hang PH 19.3 ± 5.0 vs 29.3 ± 0.9. Missed: with images (Table 3), Tool Hang PH flips to 67.3 ± 4.1 vs 20.0 ± 5.9. |
| 23 | supported, qualified | Sec. 2, C1: human demonstrations may be non-Markovian, because people may not act only on what they see now; earlier work found that models using history help. Sec. 4.1 credits history and its heading is tagged C1 and C2, but the paragraph itself never says non-Markovian. The link rests on that tag. |
| 24 | supported | Appendix D.2: sequence lengths 10, 30 and 50 were tried, and longer ones did not help significantly. Length 10 was used (Tables 8 and 9). |
| 25 | supported | Sec. 4.5 (C4), Fig. 4a: picking the checkpoint by lowest validation loss, or the last one, gave a policy 10% to 100% worse than the best. Figure values not read. |
| 26 | supported | Zhao 2023, Sec. III: current joint positions plus 4 camera images. Sec. IV-A: the policy predicts a_{t:t+k} from s_t, with no history. Recording at 50 Hz (Sec. III). 50 demonstrations per task, 100 for Thread Velcro (Sec. V-B). |
| 27 | supported | Sec. I: chunking handles pauses and other confounders that are correlated in time. Sec. IV-A: it does so when the pause falls within one chunk, without the causal confusion of history-conditioned policies (citing de Haan). This is argued, not tested. The only baselines with history are BeT (100 steps) and RT-1 (6), per Appendix D. |
| 28 | supported | Sec. VI-A: from 1% at k = 1 to 44% at k = 100, tapering a little at 200 and 400. Averaged over Transfer Cube and Bimanual Insertion, scripted and human data, temporal ensembling off. Sec. V-C blames compounding errors and non-Markovian data for earlier methods' poor results, with pausing for ever as one symptom. This is not quantified. |
| 29 | supported | Chi 2024, Appendix B.1 and the Fig. 14 caption. Policies on state inputs do not care about the observation horizon. Policies on images, the CNN version most of all, get worse as the horizon grows and do best with a low horizon above 1. A horizon of 2 is good for most tasks. Tables 7 and 8 use To = 2 for all image and real tasks (exceptions: Block Push 3; Kitchen with the transformer 4). |
| 30 | **corrected** | See below. |
| 31 | supported | Octo 2024, Sec. III-D: 2 frames of history; in early tests the gains beyond the first extra frame fell away (the paper's word is significantly, not sharply). Appendix E: one frame of history beat none in zero-shot tests, and longer history did not help on the few tasks tried. No numbers. |
| 32 | supported, hedged | Appendix E: policies with proprioceptive inputs "seemed generally worse". Causal confusion (citing de Haan) is offered only as a hypothesis, not tested. |
| 33 | supported | Brohan 2023, Appendix D.4, Table 13. Seen 97 to 82. Unseen 76 to 62. Distractors 83 to 50 (hard 64 to 14). Backgrounds 59 to 59. The authors say history mostly helps generalisation to distractors. Note that seen and unseen tasks also fell by 14 to 15 points. |
| 34 | supported | Torne 2025, Appendix C, Table 3 (3 seeds), no history / history without PTP / PTP. Square 0.79 ± 0.06 / 0.17 ± 0.01 / 0.89 ± 0.01. Tool-Hang 0.51 ± 0.14 / 0.00 / 0.75 ± 0.10. Transport 0.60 ± 0.08 / 0.00 / 0.67 ± 0.08. Push-T 0.67 ± 0.03 / 0.59 ± 0.01 / 0.62 ± 0.02. Sec. 5 describes the baseline as using the current frame and one past frame, so it very likely includes one past frame. 16 steps is the default history (Sec. 5). |
| 35 | supported | Sec. 3: a two-layer MLP predicts the current action from the previous one. The ratio of expert error to policy error is above 1 for regression policies (beyond the expert) and well below 1 for diffusion policies, which the authors read as under-use of the past. Three robomimic tasks, not named. The caption says Figure 2 but the text cites Fig. 3. |
| 36 | supported | Sec. 5.2: PTP averages 70%, no history 15%, and history without PTP is near zero on three of four tasks. Tape replacement: 80%, with both baselines failing. The tasks (block switch, two scoops, mug swap, tape swap) were built to need memory. Context 16 steps, chunks of 8. |
| 37 | supported | Wang 2019, Table II has the same values. The column headers are garbled, so which row has history rests on de Haan. Sec. V: including the agent's history scores much better on off-policy perplexity but fails when driving, which the authors blame on behaviour cloning overfitting to the expert's data. History is the car's own past positions drawn into the plan view. The page states no venue. |
| 38 | could not open | papers.nips.cc, proceedings.neurips.cc and papers.neurips.cc all refused the connection, and the web archive could not be fetched. A search-engine summary hinted that one reviewer offered mode collapse over noisy, conflicting labels as another explanation, and another doubted the clean TRPO expert data. This is unverified: do not cite. |

### The four corrections

- **16 (Wen 2021).** The mechanism is right. Sec. 4.3 defines an action-prediction error from a model that sees only
  past actions, and samples with a high error are likely changepoints ("keyframes"). Sec. 4.4 says the model is a
  two-layer MLP and names the threshold and weight of the step function. But the example values are not in Sec. 4.3.
  Appendix 9.3 upweights the top 10% by a factor of 5 for CARLA only, for the step version, after a grid search over 10% or 20% and
  weights 3, 5 or 10. The MuJoCo image runs used a softmax weighting instead, with temperature 0.2 for Hopper and
  Walker2D and 0.1 for HalfCheetah.
- **18 (Chuang 2022).** The numbers are right (Sec. 5.5, Table 5: 40.88, 22.96 and 16.18%), but this was an offline
  test on validation samples, not driving. The samples are those where the car was moving and its last action was not a
  stop. The history was replaced by copies of the current frame, and the authors counted how often the prediction
  switched to stop (lower is better). The MSE row in the same table measures how well the previous action can be read
  back from the memory feature (higher means less copying information). It is not the policy's own loss.
- **20 (Spencer 2021).** Appendix A.6, Table 3 (25 expert demonstrations). The means fell on Ant (3352.9 ± 1800.9 to
  2919.2 ± 1967.9) and on Walker2d (5349 ± 634 to 4451 ± 1491). HalfCheetah barely moved (4388.4 ± 494.8 to 4338.1 ±
  791.7, about 1%). MountainCar (-107.8 to -105.4) and Reacher (-4.9 to -4.5) rose a little, CartPole did not change,
  and Acrobot went from -78.4 to -80.0. The spreads all overlap heavily, and only Walker2d dropped by much. The
  appendix draws no conclusion about BC+H, and the caption says plain BC is mostly within the expert's error margin. A
  fair reading is no clear effect, except a drop on Walker2d.
- **30 (Chi 2024).** The first sentence is the authors' argument, not a measurement. Sec. 4.3 argues that single-step
  policies can easily overfit to pauses, and adds an informal remark that BC-RNN and IBC often get stuck in real tests
  when idle actions are kept. Sec. 6.1 says that keeping idle actions "contributed to" LSTM-GMM and IBC overfitting on
  small actions and getting stuck, so it is one factor among others. What was measured: LSTM-GMM got stuck near the T
  in 8 of 20 runs, and IBC left it early in 6 of 20. Sec. 5.3 measured that executing 8 steps was best for most tasks
  tested (Fig. 5). That this makes up for idle stretches is the authors' explanation, and they add that too long a
  horizon slows the reaction.

## Challenge to the conclusion

The conclusion is partly supported, but it is stronger than the evidence allows for this project.

1. **The settings are different.** Every study is continuous control at 3 to 50 Hz, in simulation or driving, and most
   use far more data (CARLA100 is 100 hours of driving; RT-1 about 130,000 episodes). None uses a stop-and-go loop, a
   single moving camera as the only input, or about 200 demonstrations tested on unseen layouts. What the conclusion
   says about the team's loop is reasoning, not measurement.
2. **Most "history hurts" results use an easy previous-action cue.** That cue was an explicit symbol or state entry (de
   Haan, Spencer), a drawn trail of past positions (Wang), or a speed input next to the image history (Wen 2021). In
   the team's loop the camera shift between two still pictures shows the last move, so Wen 2020's second condition is
   probably met. The first condition (long runs of identical moves) costs nothing to measure from the 208 labelled
   demonstrations, and it should be measured rather than assumed.
3. **The headline number is weak.** In 42.7% vs 33.0%, BC-SO has a spread of 8.7 points from three policies evaluated
   three times, and Codevilla saw success change by up to 42 points between seeds. Spencer's own table shows no clear
   effect of the previous action except on Walker2d (claim 20).
4. **It undersells the case for one extra frame.** Diffusion Policy's image models did best with a horizon above 1.
   Octo's one frame of history beat none. Robomimic with images and 200 skilled human demonstrations (Table 3) favoured
   BC-RNN on every task not already at the ceiling (Tool Hang 67.3 vs 20.0). These run at 10 to 20 Hz, where two
   frames show velocity, and the team's still pictures do not need that. So they do not carry over directly. Still,
   "current picture only" is a hypothesis to test, not a safe fact.
5. **The tracks may need memory.** At the crossing of the loop, and at the gaps of the dashed loop, one picture may not
   show which way to go. Torne 2025 shows that single-frame policies fail badly when memory is really needed (15% vs
   70%). Before ruling history out, look in the data for near-identical pictures that carry different move names.
6. **The chunking evidence is from fine-grained control.** ACT used k = 100 at 50 Hz (about 2 s), and Diffusion Policy
   8 steps at 10 Hz. One of the team's steps is already about 1 cm, so predicting 2 or 3 steps is untested. Also, those
   papers executed the chunk; using it only as extra training targets is a different idea. Codevilla's inertia came
   from the speed input, which a current-picture-only model lacks. But the "stop" and "small motion" classes can still
   cause stalls.
7. **The testing advice is the best-supported part, but it needs a budget.** Support comes from de Haan App. E, Wang
   2019, robomimic Sec. 4.5, Wen 2020 App. 10.6 and Codevilla Table 3. The cost is at least 3 seeds per variant on the
   laptop CPU and several closed-loop runs per sheet. A cheap pre-screen is Chuang's frozen-history test: give a
   two-frame model the current picture twice and count how often it switches to stop or small motion. A big offline
   gain from the extra frame should be read as a warning, not a success.

Net judgement: the default (current picture only, no extra inputs) is a reasonable, cautious start for 208
demonstrations. Robomimic Sec. 4.3 found that extra inputs the task does not need hurt. But the literature does not
settle the question for this setting. The decision must come from the team's own closed-loop comparison, and that
comparison should include the crossing and dashed sheets and at least one unseen track.

## Evidence the researcher missed

Checked (I read it in the source):

- **Robomimic with images** (Mandlekar 2021, Table 3). With images, BC-RNN (10 steps at 20 Hz) beat BC on every task
  not already at the ceiling. Square PH 82.0 vs 62.0, Transport PH 72.0 vs 55.3, Tool Hang PH 67.3 vs 20.0, Square MH
  76.7 vs 46.0, Transport MH 42.0 vs 18.7. The PH sets hold 200 demonstrations (Sec. 3.2), close to the team's 208. The
  inputs, though, are a front camera and a wrist camera plus the gripper's pose.
- **Robomimic, extra inputs** (Sec. 4.3). Adding end-effector velocity or joint data to BC-RNN cut success by 49 to 88%
  (relative) with low-dimensional inputs and by 2 to 29% with images. The authors suspect overfitting to information
  the task does not need. This supports keeping the input minimal.
- **Wen 2020, Appendix 10.6.** BC-OH had a lower test loss than BC-SO in all six tasks. The authors' method had a
  higher test loss than BC-OH but more reward. So offline loss ranks methods wrongly.
- **History dropout does not cure copying.** Wen 2021, Table 1: HistoryDropout 34.0 vs BC-OH 33.0 with speed, and 25.3
  vs 25.7 without. Wen 2020, Table 2: Dropout-BC is below BC-OH in 5 of 6 tasks, and Sec. 6.3 says it does poorly on
  all of them.
- **Chuang 2022, Sec. 5.5, Table 5.** The frozen-history probe is an offline copying test that needs no robot time.
- **Torne 2025, Sec. 5.2.** Single-frame policies fail on tasks that need memory (15% vs 70%). This raises the question
  of whether the crossing and the dashed gaps need memory.
- **Zhao 2023, Sec. VI-A.** Chunking also improved two other methods (BC-ConvMLP and VINN). Their numbers are only in
  Fig. 6a.
- **Diffusion Policy, real Push-T** (Chi 2024, Sec. 6.1, Table 6), the task closest to pushing a puck. 136
  demonstrations at 10 Hz, observation horizon 2, action horizon 6. Success: Diffusion Policy 95%, LSTM-GMM 20%, IBC
  0%.
- **Codevilla 2019, Table 2.** The single-frame CILRS fell from 83 to 56 (regular traffic) and from 42 to 24 (dense)
  in a new town with new weather. On unseen tracks, generalisation may matter more than the history choice.
- **Spencer 2021, Sec. 3.** The authors argue that de Haan's effect comes from too little data or the wrong model
  class. So more varied data could shrink it.
- **Wang 2019, Sec. IV-A and V.** The GTA "history" is a drawn trail of past positions, not stacked camera frames.

Leads, not opened (only seen cited or remembered; do not cite before opening):

- Ortega et al. (2021), "Shaking the foundations: delusions in sequence models for interaction and control", arXiv.
- Swamy et al. (2022), "Causal Imitation Learning under Temporally Correlated Noise", ICML 2022.
- Park et al. (2021), "Object-Aware Regularization for Addressing Causal Confusion in Imitation Learning", NeurIPS 2021.
- Wen et al. (2022), "Fighting Fire with Fire: Avoiding DNN Shortcuts through Priming", ICML 2022.
- Shi et al. (2023), "Waypoint-Based Imitation Learning for Robotic Manipulation", CoRL 2023 (fewer, coarser
  decisions).

Project data check (not literature): from the 208 labelled demonstrations, count per sheet how often the move name
equals the previous one. That is the accuracy a two-frame model could reach just by copying.

## Sources I opened (9 November 2026)

- de Haan, Jayaraman, Levine (2019). Causal Confusion in Imitation Learning. NeurIPS 2019. https://ar5iv.labs.arxiv.org/html/1905.11979
- Wen, Lin, Darrell, Jayaraman, Gao (2020). Fighting Copycat Agents in Behavioral Cloning from Observation Histories. NeurIPS 2020. https://ar5iv.labs.arxiv.org/html/2010.14876
- Codevilla, Santana, López, Gaidon (2019). Exploring the Limitations of Behavior Cloning for Autonomous Driving. ICCV 2019 (venue not on page). https://ar5iv.labs.arxiv.org/html/1904.08980
- Wen, Lin, Qian, Gao, Jayaraman (2021). Keyframe-Focused Visual Imitation Learning. ICML 2021 (stated on arXiv). https://ar5iv.labs.arxiv.org/html/2106.06452 and https://arxiv.org/abs/2106.06452
- Chuang, Yang, Wen, Gao (2022). Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction. ECCV 2022. https://ar5iv.labs.arxiv.org/html/2207.09705
- Bansal, Krizhevsky, Ogale (2019). ChauffeurNet. RSS 2019 (venue not checked). https://ar5iv.labs.arxiv.org/html/1812.03079
- Spencer, Choudhury, Venkatraman, Ziebart, Bagnell (2021). Feedback in Imitation Learning: The Three Regimes of Covariate Shift. arXiv preprint. https://ar5iv.labs.arxiv.org/html/2102.02872
- Mandlekar et al. (2021). What Matters in Learning from Offline Human Demonstrations for Robot Manipulation. CoRL 2021. https://ar5iv.labs.arxiv.org/html/2108.03298
- Zhao, Kumar, Levine, Finn (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. RSS 2023. https://ar5iv.labs.arxiv.org/html/2304.13705
- Chi et al. (2024). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. IJRR 2024. https://arxiv.org/html/2303.04137v5
- Octo Model Team et al. (2024). Octo: An Open-Source Generalist Robot Policy. RSS 2024. https://arxiv.org/html/2405.12213v2
- Brohan et al. (2023). RT-1: Robotics Transformer for Real-World Control at Scale. RSS 2023. https://ar5iv.labs.arxiv.org/html/2212.06817
- Torne, Tang, Liu, Finn (2025). Learning Long-Context Diffusion Policies via Past-Token Prediction. CoRL 2025 (not on arXiv page; arXiv lists the first author as Marcel Torne). https://arxiv.org/html/2505.09561v2 and https://arxiv.org/abs/2505.09561
- Wang, Devin, Cai, Krähenbühl, Darrell (2019). Monocular Plan View Networks for Autonomous Driving. Venue not on page. https://ar5iv.labs.arxiv.org/html/1905.06937
- Not opened (connection refused): https://papers.nips.cc/paper_files/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Review.html
