# Picture histories, copycat problems and causal confusion

Key: `history_and_causal_confusion`
Written 9.11.26 by a literature sub-agent for Team M (Emranul Hoque, Marshal Sebastine). Nothing in this note has been tried on the Ned2. Every number below comes from a paper; everything marked "judgement" is my own reasoning and has not been tested.

## The question

Should the model see earlier pictures as well as the current one (frame stacking, also called "history")? Does that help or hurt behaviour cloning? What is the safe default for our one-camera pusher, where small motions (forearm twists with the tip still) come before real moves?

## Short answer

What the papers measured:

- Extra history helped closed-loop behaviour cloning when the current input was missing something the decision needed, such as the car's speed or a robot's joint velocities (Wen et al. 2020, 2021; Chuang et al. 2022).
- When the current input already held what was needed, adding history or the previous action made closed-loop behaviour worse. In some cases offline scores looked better at the same time (de Haan et al. 2019; Wen et al. 2021; Spencer et al. 2021).
- Policies trained on histories tend to copy the previous action (the "copycat" problem). This happens when consecutive expert actions are alike and the previous action can be read off the earlier frames (Wen et al. 2020; Torne Villasevil et al. 2025 for regression-type policies).
- Demonstrations with stops, pauses or small actions made policies stall. Codevilla et al. 2019 called this the inertia problem; Diffusion Policy and ACT report robots getting stuck. Predicting a chunk of several future actions helped (ACT; Diffusion Policy).
- Recent manipulation work keeps history short. Diffusion Policy found two frames (the current one plus one earlier) a good compromise, and longer horizons hurt its image-based CNN variant. Octo found one extra frame helped and more did not. Long histories hurt diffusion policies unless training was changed (Torne Villasevil et al. 2025). In other settings history helped a lot: RT-1 with six frames, and robomimic's recurrent policy on human teleoperated data.

My judgement for our robot:

- In our stop-and-go loop the arm stands still when each picture is taken. So an earlier picture mostly tells the model what the last step was, which is exactly the copycat cue. It adds no speed information.
- Safe default: the current picture only, with no stacked frames and no previous move as input. Have the model also predict the next two or three steps, so that a forearm twist is learnt together with the move that follows it.
- Try one extra picture only as a separate test, judged by closed-loop runs on held-out sheets, over several training seeds.

## Words used in this note

- Behaviour cloning (BC): supervised learning from demonstrations. The model gets a picture and learns to output what the demonstrator did next.
- History, frame stacking: giving the model earlier pictures (or earlier actions) as well as the current one.
- Offline score versus closed loop: an offline score is accuracy or loss on stored demonstration pictures. Closed loop means letting the model drive the robot, so each new picture is a result of its own earlier choices.
- Causal confusion (de Haan et al. call it causal misidentification): the model relies on something that only goes together with the right action in the data, instead of what actually causes it.
- Copycat problem: a history model learns "do what was done last step", because that is usually right in the demonstrations.
- Inertia problem: a stopped car in the data usually stays stopped, so the model learns that being still means staying still, and then fails to start again.
- Action chunk: the model predicts several future steps at once from the current input.
- Keyframe or changepoint: a moment in a demonstration where the action changes, for example where the line bends.

## 1. The three classic papers

### 1.1 de Haan, Jayaraman and Levine 2019, "Causal Confusion in Imitation Learning", NeurIPS 2019

Opened: https://arxiv.org/abs/1905.11979 (comments: published at NeurIPS 2019) and the HTML version https://ar5iv.labs.arxiv.org/html/1905.11979

- Definition (Sec. 1, Sec. 3): a cloned policy can lean on nuisance correlates of the expert's actions (things that merely go together with them) instead of their causes. This still gives low training error, but it fails under the shift in what the policy sees once its own actions drive the scene.
- Set-up (Sec. 3.2, Fig. 3): each task had an "original" version and a "confounded" version, with the same experts and networks. The confounded version added the previous action to the input. In MountainCar and Hopper this was an extra entry in the state vector at an unknown index. In Atari Pong, Enduro and UpNDown it was a symbol drawn at an unknown place on the screen. The Atari states were two stacked frames in both versions.
- Measured (Sec. 3.2, Fig. 4): with the original input, reward tended towards expert level as the dataset grew. With the confounded input, the policy needed many more demonstrations to reach the same level, or never reached it.
- Measured (App. E, Fig. 11): training and validation losses were often near zero for both versions, yet the confounded policy consistently earned clearly lower reward when run. The offline loss did not show the problem.
- Measured, reported from Wang et al. 2019 (Sec. 3.2, Table 1, GTA-V driving). The policy with history had better validation perplexity (0.834 against 0.989) but drove worse:
  - distance 144.92 against 268.95
  - interventions 2.94 ± 1.79 against 1.30 ± 0.78
  - collisions 6.49 ± 5.72 against 3.38 ± 2.55

  The table caption sums this up: "Accessing history yields better validation performance, but worse actual driving performance."
- Illustration only (Sec. 1, Fig. 1): a car that can see its own dashboard brake light learns to brake when the light is on. This example was not an experiment.
- Speculation (Sec. 1, Sec. 3.2): the authors say causal misidentification occurs commonly, especially when the input includes history. Their own experiments only add the previous action, so the general claim about history rests on earlier work (Wang et al. 2019 and others).
- Their fix (Sec. 4) learns which input parts to trust, using expert queries or reward from running the policy. For us that would cost robot time and needs a reward signal (judgement).

### 1.2 Wen, Lin, Darrell, Jayaraman and Gao 2020, "Fighting Copycat Agents in Behavioral Cloning from Observation Histories", NeurIPS 2020

Opened: https://ar5iv.labs.arxiv.org/html/2010.14876 and the venue page https://neurips.cc/virtual/2020/poster/17210

- Definition (Sec. 1, Sec. 4): a policy trained on observation histories learns to predict the expert's "previous action, rather than the next action". This happens under two conditions, paraphrased:
  - (i) consecutive expert actions are strongly correlated;
  - (ii) the previous action is easy to recover from the history.

  Such a policy scores well on training and held-out demonstrations but fails when it runs.
- Set-up (Sec. 4, Sec. 6.2): six MuJoCo tasks (Ant, Hopper, Humanoid, Reacher, Walker2d, HalfCheetah) were made partially observed by keeping only joint positions, with no velocities. The history length was H = 2.
- Measured (Table 1): the history policy (BC-OH) produced next actions that were much easier to predict from its previous actions than the expert's were. This held in all six tasks; for example Humanoid had a prediction error of 0.18 against 6.93 (x10^-1), and Walker2d 0.46 against 2.47 (x10^-2).
- Measured (Table 2): in these partially observed tasks, plain BC-OH still beat single-observation BC (BC-SO) in five of six. Examples are HalfCheetah 820 against -38, Walker2d 592 against 363 and Ant 1750 against 1300; Humanoid was the exception at 565 against 587. Their fix beat both in all six, for example Hopper 1086 against 275 (BC-SO) and 293 (BC-OH).
- So the lesson is not "history always hurts". When one observation lacks needed state, history helps, but a plain history model under-uses it because copying is easier.
- Fix (Sec. 5): adversarial training plus an information bottleneck, which removes information about the previous action from the learnt features. It works offline, with no extra robot runs.
- The authors' own limit (Sec. 7): they say their gains in image-based experiments were not significant yet. No numbers are given.

### 1.3 Codevilla, Santana, López and Gaidon 2019, "Exploring the Limitations of Behavior Cloning for Autonomous Driving", ICCV 2019

Opened: https://ar5iv.labs.arxiv.org/html/1904.08980 and the venue record https://mlanthology.org/iccv/2019/codevilla2019iccv-exploring/ (DOI 10.1109/ICCV.2019.00942)

- Inputs (Sec. 3.1): one image, the car's measured speed and a route command. There are no stacked frames. The speed input plays the part that history plays elsewhere (my reading).
- Inertia problem (Sec. 3.2): in the training data a stopped car, for example at a red light, nearly always stays stopped. The model therefore ties low speed to "do not accelerate", stops too often and has trouble restarting.
- Measured (Sec. 5.3, Fig. 4): an episode counted as an inertia failure if the car stood still without throttle for at least 8 s until timeout. The share of such episodes grew as the amount of training data grew. The exact values are only in the figure, which I could not read.
- Speed-prediction branch (Sec. 3.3): the model also learns to predict speed from the image. This helped, but the authors call it "not a final solution to this problem", and the inertia problem stays open in their conclusion.
- Measured (Sec. 5.3, Table 3): training the same model 12 times with different random seeds changed the success rate by up to 42% in dense traffic. ImageNet initialisation reduced the spread, to 38% in dense traffic.

## 2. Follow-up work on copycat policies

### 2.1 Wen, Lin, Qian, Gao and Jayaraman 2021, "Keyframe-Focused Visual Imitation Learning", ICML 2021 (PMLR 139)

Opened: https://ar5iv.labs.arxiv.org/html/2106.06452 and https://proceedings.mlr.press/v139/wen21d.html

- Measured (Table 1, Sec. 5 and 6): CARLA driving, with the car's speed part of the observation and a history of H = 6 frames.

  | Method | Success |
  |---|---|
  | Single frame (BC-SO) | 42.7% ± 8.7 |
  | History (BC-OH) | 33.0% ± 4.2 |
  | History dropout | 34.0% |
  | DAgger, listed as "120K" in the table | 35.2% |
  | Their keyframe method | 43.4% ± 0.8 |

- Measured (Table 1): with speed removed from the input the order flipped. BC-SO reached 9.2%, BC-OH 25.7% and the keyframe method 36.8%.
- Measured (Table 2): in image-based MuJoCo, where the history was the previous plus the current frame, history also helped plain BC, for example HalfCheetah 615 against 4. The keyframe method was best (Hopper 951, HalfCheetah 819, Walker2D 769).
- Method (Sec. 4.3): first train a small model that predicts each expert action from past actions only. The demonstration moments where it fails most are changepoints. These are given more weight in training, for example the top 10% weighted by a factor of 5.
- What stands out: with speed available, the plain single-frame model did as well as the best history fix and better than DAgger or history dropout.

### 2.2 Chuang, Yang, Wen and Gao 2022, "Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction", ECCV 2022

Opened: https://ar5iv.labs.arxiv.org/html/2207.09705 and https://arxiv.org/abs/2207.09705 (comments: ECCV2022)

- Set-up (Sec. 5.1, App. B.1): the car's speed was withheld from all methods, so one frame could not show speed. CARLA used a history of H = 6.
- Measured (Table 2): history helped plain BC. In the training town with regular traffic, success went from 37.6 to 67.1; in dense traffic from 13.1 to 34.1. Their residual method did better again, with 78.1 and 52.0.
- Measured (Table 5): they replaced the history with copies of the current frame, so the scene looked still. Cars driven by the history policy then stopped in 40.88% of cases where they had been moving. The rate was 22.96% for the keyframe method and 16.18% for theirs. This is the inertia effect, now caused by a history that looks still.
- Method (Sec. 4): one branch looks at the history and is trained to predict only the change from the previous action to the next. A second branch looks at the current frame and outputs the action.

### 2.3 Bansal, Krizhevsky and Ogale 2019, "ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst", RSS 2019

Opened: https://ar5iv.labs.arxiv.org/html/1812.03079 and https://m.roboticsproceedings.org/rss15/p31.html

- Method and rationale (Sec. 4.2): "past motion dropout". In half of the training examples, the input showing past positions contained only the current position. Without this, the authors say, the network can cheat by extrapolating past motion. Their example is a car that stops at a stop sign only after it sees itself slowing down, and so never stops when driving on its own.
- No experiment isolates this trick: all their model variants used it (Sec. 6.2). In Wen et al. 2021 a similar history dropout gained only one point (34.0% against 33.0%).

### 2.4 Spencer, Choudhury, Venkatraman, Ziebart and Bagnell 2021, "Feedback in Imitation Learning: The Three Regimes of Covariate Shift", arXiv preprint

Opened: https://ar5iv.labs.arxiv.org/html/2102.02872 (no venue given on the page)

- Measured (App. A.6, Table 3): adding the previous action to the observation ("BC+H") made BC worse on Ant, Walker2d and HalfCheetah. It was about the same on others, such as MountainCar and Reacher.
- Argument (Sec. 1, Fig. 1, Sec. 3): causal confusion is "simply a more visible case of feedback-driven covariate shift". Their example is a learner that wrongly brakes once and then keeps braking (they call this latching), because the previous action predicts the current one so well.

### 2.5 Wang, Devin, Cai, Krähenbühl and Darrell 2019, "Monocular Plan View Networks for Autonomous Driving"

Opened: https://ar5iv.labs.arxiv.org/html/1905.06937. The venue is IROS 2019 according to a search result; I did not open the venue record.

- This is the primary source of de Haan's driving table (Table II). The same numbers appear there, but the HTML column layout was garbled, so I relied on de Haan's labels for which row is which.
- The authors say history improves the off-policy score but fails on-policy, and they blame behaviour cloning overfitting the expert's data distribution.

## 3. Newer robot-manipulation evidence

### 3.1 Mandlekar et al. 2021, "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation" (robomimic), CoRL 2021 (PMLR 164)

Opened: https://ar5iv.labs.arxiv.org/html/2108.03298 and https://proceedings.mlr.press/v164/mandlekar22a.html

- Measured (Sec. 4.1, Table 1, simulated tasks with low-dimensional observations, averaged over 3 seeds): the recurrent policy with history (BC-RNN) beat single-step BC on most human datasets. Examples:
  - Transport, proficient human: 71.3% against 17.3%
  - Square, multi-human: 78.0% against 52.7%

  It gained little on machine-generated data (Lift: 70.7% against 65.3%) and lost on Tool Hang, proficient human (19.3% against 29.3%).
- Speculation (Sec. 2, C1): the authors explain this by human demonstrations being non-Markovian, meaning people do not act only on what they currently see.
- Measured (App. D.2): longer BC-RNN sequence lengths (10, 30, 50) did not improve performance significantly.
- Measured (Sec. 4.5, Fig. 4a): picking the checkpoint with the lowest validation loss, or the last one, gave a policy 10% to 100% worse than the best checkpoint.

### 3.2 Zhao, Kumar, Levine and Finn 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), RSS 2023

Opened: https://ar5iv.labs.arxiv.org/html/2304.13705, https://arxiv.org/abs/2304.13705 and https://www.roboticsproceedings.org/rss19/p016.html

- Design (Sec. III, IV, V-B): ACT sees only the current observation, which is four camera images and the joint positions, with no frame history. It predicts a chunk of k future actions. Data were recorded at 50 Hz, with 50 demonstrations per task (100 for one task).
- Argument, not a head-to-head test (Sec. I, Sec. IV-A): chunking can model non-Markovian habits in human demonstrations, such as pauses inside a chunk, "without introducing the causal confusion issue for history-conditioned policies". The paper does not test an ACT that uses history.
- Measured (Sec. VI-A, Fig. 6a): success rose from 1% at chunk size k = 1 to 44% at k = 100, then tapered slightly. This was averaged over two simulated tasks with scripted and human data, without temporal ensembling. Calculated, not stated: at 50 Hz, k = 100 is 2 s of motion.
- Observation with the authors' attribution (Sec. V-C): with earlier methods, behaviour degraded towards the end of episodes and the robot could pause indefinitely in some states. The authors blame compounding errors and non-Markovian data.

### 3.3 Chi et al. 2023/2024, "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023, extended in IJRR 2024

Opened: https://arxiv.org/html/2303.04137v5 (journal version), https://ar5iv.labs.arxiv.org/html/2303.04137, https://arxiv.org/abs/2303.04137 and the project page https://diffusion-policy.cs.columbia.edu/ (which lists both venues)

- Set-up (Sec. 2.3; App. A, Tables 7 and 8): the observation horizon To is the number of latest observations given to the policy. It is 2 for most tasks. For Push-T it is 2, with 8 actions executed out of 16 predicted. Push-T means pushing a T-shaped block onto a target with a round end-effector. Its inputs are images plus the end-effector position, and the real robot is commanded at 10 Hz.
- Measured (App. B.1, Fig. 14): the state-based policy did not care about the observation horizon. The image-based policy, especially with a CNN backbone, got worse as the horizon grew; it did best with a low horizon above 1, with 2 a good compromise. I could not read the figure's values.
- Observation with the authors' attribution (Sec. 4.3, Sec. 6.1): "single-step policies can easily overfit to this pausing behavior". In the real Push-T task idle actions were kept in the data. The authors say this contributed to LSTM-GMM (a BC-RNN-style policy with history) and IBC overfitting on small actions and getting stuck.
- Measured (Sec. 5.3, Fig. 5): predicting action sequences helps with idle stretches, but too long a sequence slows reactions. Executing 8 steps was best for most tasks.

### 3.4 Octo Model Team (Ghosh, Walke, Pertsch, Black, Mees et al.) 2024, "Octo: An Open-Source Generalist Robot Policy", RSS 2024

Opened: https://arxiv.org/html/2405.12213v2, https://arxiv.org/abs/2405.12213 and https://octo-models.github.io (the BibTeX gives Proceedings of Robotics: Science and Systems, Delft, 2024)

- Measured, no numbers given (Sec. III, App. E): Octo trains with 2 frames, the current one plus one earlier. Models with one frame of history did better in zero-shot tests than models without. Preliminary tests showed "significantly diminishing gains beyond the first additional frame". Longer history gave no benefit on the few tasks they tried, though other tasks might benefit.
- Measured, then speculation (App. E): policies that also got proprioception (joint readings) were generally worse. The authors guess this is causal confusion between the proprioception and the target actions, citing de Haan et al.

### 3.5 Brohan et al. 2023, "RT-1: Robotics Transformer for Real-World Control at Scale", RSS 2023

Opened: https://ar5iv.labs.arxiv.org/html/2212.06817, https://arxiv.org/abs/2212.06817 and https://roboticsproceedings.org/rss19/p025.html

- Measured (App. D.4, Table 13): cutting the input from six images to one hurt success:

  | Test | Six images | One image |
  |---|---|---|
  | Seen tasks | 97 | 82 |
  | Unseen tasks | 76 | 62 |
  | With distractors | 83 | 50 |
  | Backgrounds | 59 | 59 (unchanged) |

  The authors say history mainly helps with distractors.
- Context (judgement): this is continuous control with a large dataset, where earlier frames carry motion. It is a different situation from our still-at-every-picture loop.

### 3.6 Torne Villasevil, Tang, Liu and Finn 2025, "Learning Long-Context Diffusion Policies via Past-Token Prediction", CoRL 2025 (PMLR 305)

Opened: https://arxiv.org/html/2505.09561v2 and https://proceedings.mlr.press/v305/villasevil25a.html (an earlier workshop title was "... Robot Policies via Past-Token Prediction")

- Measured (App. C, Table 3, Fig. 9): diffusion policies given a 16-step history, without the authors' fix, fell far below versions without history (success rate, 3 seeds):

  | Task | No history | 16-step history | With past-token prediction (their fix) |
  |---|---|---|---|
  | Square | 0.79 | 0.17 | not reported here |
  | Tool-Hang | 0.51 | 0.00 | not reported here |
  | Transport | 0.60 | 0.00 | not reported here |
  | Push-T (pushing task) | 0.67 | 0.59 | 0.62 |

  On Push-T, no history was best. The "no history" baseline is described as taking the current and past single frame, which is ambiguous and may mean two frames.
- Measured (Sec. 3, Fig. 2, which the text calls Fig. 3): they compared how well each policy's current action can be predicted from its previous one, in robot runs against demonstrations. Regression policies were above the expert's level, which means copycat behaviour. Diffusion policies were well below, a "surprising underuse of past action information".
- Measured (Sec. 5.2): on four real tasks built to need memory, the fix averaged 70% success against 15% without history. Plain history was near zero on three of the four.

## 4. The pattern across the evidence

| Setting | Does the current input already hold what is needed? | Effect of plain history | Source |
|---|---|---|---|
| Control tasks with the previous action added | yes | worse, even with near-zero offline loss | de Haan 2019 (Fig. 4, Fig. 11) |
| GTA-V driving, history of positions | mostly yes | better offline score, worse driving | de Haan 2019 Table 1 (from Wang 2019) |
| MuJoCo with the previous action added | yes | worse on 3 of the tasks, same elsewhere | Spencer 2021 (Table 3) |
| CARLA with speed in the input | yes | worse (33.0% against 42.7%) | Wen 2021 (Table 1) |
| CARLA without speed | no (speed missing) | better (25.7% against 9.2%; 67.1 against 37.6) | Wen 2021; Chuang 2022 |
| MuJoCo without velocities | no | better, but copycat limits it | Wen 2020 (Tables 1, 2) |
| Human teleoperated manipulation (simulated) | possibly not (non-Markovian humans) | mostly better | robomimic (Table 1) |
| Image-based Diffusion Policy | partly (2 frames give motion) | best at 2, worse beyond | Chi 2024 (App. B.1) |
| Octo, zero-shot | partly | 1 extra frame helped, more did not | Octo 2024 (App. E) |
| RT-1, six frames | partly | history helped, mainly with distractors | RT-1 (Table 13) |
| Diffusion policies with 16-step history | mostly yes for Square, Push-T | much worse unless regularised | Torne Villasevil 2025 (Table 3) |

One more pattern: stops, pauses and small actions in the data made policies stall, whether or not they used history. This covers Codevilla's single frame plus speed, Diffusion Policy's LSTM-GMM and IBC, and the earlier methods in the ACT paper. Predicting chunks of future actions was the remedy used in ACT and Diffusion Policy.

## 5. What this means for the Ned2 pusher (judgement, not measured)

### 5.1 Do our data meet the two copycat conditions?

- (i) Consecutive actions alike. Along the straight sheet and the gentle arc, the same move name probably repeats for many 1 cm steps in a row. This is not measured yet but can be counted from the labels.
- (ii) Previous action easy to read off. The camera sits on the forearm. A tip move shifts the whole view and a forearm twist rotates it, and both pictures are taken with the arm still. Two pictures side by side would show the last step almost directly.
- If both hold, a two-picture model can score highly offline just by repeating the last move. It would then fail where the line bends, which is what the unseen exam tracks test.

### 5.2 Is one picture enough?

- In the papers, history helped when one input lacked speed or velocity. Our pictures are taken with the arm and the puck at rest, so there is no motion to recover. The puck, the line ahead and the magnet block should all be visible in the current picture.
- Where one picture may not be enough:
  - the crossing on the loop sheet;
  - the gaps in the dashed loop;
  - any moment when the puck leaves the view.
- A look-back of one step (1 cm) shows nearly the same scene and would not solve these cases. Torne Villasevil et al. needed long contexts plus a special training trick for real memory tasks. So check first whether the single-picture model actually fails there.

### 5.3 The "small motion before a real move" pattern

- This looks like the pauses and small actions in Diffusion Policy and ACT, where single-step policies overfitted and got stuck. It also looks like Codevilla's inertia: after a twist with the tip still, the next picture can look almost the same, so the model may choose "small motion" again and again.
- With history, a pair of pictures that looks still becomes a cue in its own right. Chuang et al. measured that a still-looking history made history policies stop far more often (40.88% of cases).
- A history model could learn the genuine pattern "after a twist comes a move". It would learn it through the same channel that makes copying possible.
- ACT and Diffusion Policy instead predict a chunk of future actions from the current observation. That captures "twist, then move" without feeding the past back in. Note that ACT's claim that this avoids causal confusion is an argument; the paper does not test it against an ACT with history.

### 5.4 Safe default (judgement)

1. Input: the current picture only. No stacked frames and no previous move as input; the course allows only the camera anyway, and Octo's experience with proprioception is a warning against feeding back the robot's own state.
2. Output: the move name and exact step for the next step, plus extra outputs for the following two or three steps (a small action chunk). The labels already exist, because each demonstration is a sequence.
3. At run time, execute step 1 of the chunk. Try on the robot whether also executing step 2 when step 1 is a small motion avoids stalls and saves time. This is untested.
4. Judge every model by closed-loop runs on held-out sheets, not by validation accuracy alone (de Haan Fig. 11 and Table 1; robomimic Sec. 4.5). Use two or three training seeds, because seeds alone moved success by up to 42% in Codevilla et al.
5. Log runs of consecutive small motions or stops on the robot. Long runs are the stall failure described above. The fix belongs in the data or the chunk outputs, not in a hand-written rule: check with the course rule that the motion must come from the model.

### 5.5 If we try history anyway

- Use at most one extra picture, as in Diffusion Policy and Octo. Compare it with the single-picture model at the same training budget and seeds.
- Copycat check, a simple version of Wen 2020 Table 1 and Torne Villasevil 2025 Sec. 3: compare how often the robot repeats its previous move name with how often the demonstrations do. Much more repetition on the robot means copying.
- If it copies, upweight changepoint pictures (Wen 2021: top 10% given 5 times the weight). History dropout gained little in Wen 2021 (34.0% against 33.0%).

### 5.6 Cheap checks on the existing 208 demonstrations (no robot time)

- The share of pictures whose move name equals the previous picture's, per sheet. This is condition (i).
- The accuracy of the trivial rule "repeat the previous move" on the validation split. Any model, and a history model in particular, should beat it clearly on changepoint pictures, not only on average.
- How often a small motion is followed by a real move, and how many small motions come in a row.

### 5.7 Cost on the laptop CPU

Two pictures roughly double the image-encoder work per training sample, unless features are cached. At run time the previous picture's features can be kept, so the 1.2 s step would hardly change. Cost is therefore not the deciding argument; the copycat risk is.

## 6. Claims list

Ids are shortened after the first row: -2 means history_and_causal_confusion-2. Kinds: measured = the paper ran it and reports the result; speculation = the authors' explanation, argument or untested claim; definition = what a term or method means. Status: checked = I read it in the source (HTML version, read through a fetch tool; key numbers re-read as verbatim table rows).

| id | claim (short) | source | location | status | kind |
|---|---|---|---|---|---|
| history_and_causal_confusion-1 | Causal misidentification: policy relies on correlates, not causes, and fails under the shift its own actions create | de Haan 2019 | Sec. 1, Sec. 3 | checked | definition |
| -2 | Adding the previous action made BC need many more demos or never reach expert reward | de Haan 2019 | Sec. 3.2, Fig. 3, Fig. 4 | checked | measured |
| -3 | Near-zero train and validation loss for both inputs, yet the confounded one earned clearly lower reward | de Haan 2019 | App. E, Fig. 11 | checked | measured |
| -4 | GTA-V: history gave better perplexity (0.834 against 0.989) but worse distance, interventions, collisions | de Haan 2019 (from Wang 2019) | Sec. 3.2, Table 1 | checked | measured |
| -5 | Authors say the problem is common, especially with history; their own tests only add the previous action | de Haan 2019 | Sec. 1, Sec. 3.2 | checked | speculation |
| -6 | Copycat: history policy predicts the previous action; needs correlated actions plus a recoverable previous action | Wen 2020 | Sec. 1, Sec. 4 | checked | definition |
| -7 | BC-OH actions far more predictable from previous actions than the expert's in all six tasks | Wen 2020 | Sec. 4, Table 1 | checked | measured |
| -8 | Without velocities, BC-OH beat BC-SO in 5 of 6 tasks and their fix beat both | Wen 2020 | Sec. 6.2, Table 2 | checked | measured |
| -9 | Authors report no significant gain yet in image-based experiments | Wen 2020 | Sec. 7 | checked | measured (no numbers) |
| -10 | Inertia problem: single image plus speed; stopped means stays stopped; fails to restart | Codevilla 2019 | Sec. 3.1, Sec. 3.2 | checked | definition |
| -11 | Share of inertia failures grew with more training data | Codevilla 2019 | Sec. 5.3, Fig. 4 | checked | measured |
| -12 | Speed-prediction branch is not a final solution; the problem stays open | Codevilla 2019 | Sec. 3.3, Sec. 5.3, Conclusion | checked | speculation |
| -13 | The random seed alone changed success by up to 42% (dense traffic) | Codevilla 2019 | Sec. 5.3, Table 3 | checked | measured |
| -14 | CARLA with speed: single frame 42.7%, history 33.0%, history dropout 34.0%, DAgger 35.2%, keyframes 43.4% | Wen 2021 | Table 1 | checked | measured |
| -15 | Without speed: single frame 9.2%, history 25.7%, keyframes 36.8%; image MuJoCo history helped | Wen 2021 | Table 1, Table 2 | checked | measured |
| -16 | Keyframe weighting: upweight samples a past-action-only model predicts badly | Wen 2021 | Sec. 4.3 | checked | definition |
| -17 | Speed withheld: history helped plain BC (37.6 to 67.1); residual method 78.1 | Chuang 2022 | Sec. 5.1, Table 2 | checked | measured |
| -18 | Still-looking history made the history policy stop in 40.88% of moving cases | Chuang 2022 | Table 5 | checked | measured |
| -19 | Past motion dropout (50%) because the net extrapolates past motion; no isolated ablation | Bansal 2019 | Sec. 4.2, Sec. 6.2 | checked | speculation |
| -20 | Adding the previous action (BC+H) hurt on Ant, Walker2d, HalfCheetah | Spencer 2021 | App. A.6, Table 3 | checked | measured |
| -21 | Causal confusion is feedback-driven covariate shift; a learner that brakes keeps braking | Spencer 2021 | Sec. 1, Fig. 1, Sec. 3 | checked | speculation |
| -22 | BC-RNN beat BC on most human datasets (Transport 71.3% against 17.3%), not on Tool Hang | Mandlekar 2021 | Sec. 4.1, Table 1 | checked | measured |
| -23 | Explanation: human demonstrations are non-Markovian | Mandlekar 2021 | Sec. 2 (C1) | checked | speculation |
| -24 | Sequence lengths 10, 30, 50 made no significant difference | Mandlekar 2021 | App. D.2 | checked | measured |
| -25 | Lowest-validation-loss checkpoint 10% to 100% worse than the best | Mandlekar 2021 | Sec. 4.5, Fig. 4a | checked | measured |
| -26 | ACT uses the current observation only, predicts chunks; 50 Hz, 50 demos per task | Zhao 2023 | Sec. III, IV, V-B | checked | definition |
| -27 | Chunking handles pauses without history's causal confusion (argued, not tested) | Zhao 2023 | Sec. I, Sec. IV-A | checked | speculation |
| -28 | Chunk size 1 to 100 raised success from 1% to 44%; earlier methods paused indefinitely | Zhao 2023 | Sec. VI-A, Fig. 6a, Sec. V-C | checked | measured |
| -29 | Image-based (CNN) Diffusion Policy got worse with longer observation horizon; 2 is a good compromise | Chi 2024 | App. B.1, Fig. 14, Tables 7-8 | checked | measured |
| -30 | Single-step policies overfit to pauses; LSTM-GMM and IBC got stuck on small actions in real Push-T | Chi 2024 | Sec. 4.3, Sec. 6.1, Sec. 5.3 | checked | measured |
| -31 | Octo: one extra frame helped zero-shot, more gave no clear benefit | Octo 2024 | Sec. III, App. E | checked | measured (no numbers) |
| -32 | Proprioception made Octo worse; authors guess causal confusion | Octo 2024 | App. E | checked | speculation |
| -33 | RT-1 without history: seen 97 to 82, unseen 76 to 62, distractors 83 to 50 | Brohan 2023 | App. D.4, Table 13 | checked | measured |
| -34 | 16-step history wrecked diffusion policies (Square 0.79 to 0.17); Push-T best without history | Torne Villasevil 2025 | App. C, Table 3, Fig. 9 | checked | measured |
| -35 | Regression policies copy past actions; diffusion policies under-use them | Torne Villasevil 2025 | Sec. 3, Fig. 2 | checked | measured |
| -36 | On memory-needing real tasks: fix 70%, no history 15%, plain history near zero on 3 of 4 | Torne Villasevil 2025 | Sec. 5.2 | checked | measured |
| -37 | Primary source of the GTA-V history numbers; authors blame BC overfitting the expert distribution | Wang 2019 | Table II | checked (row labels via de Haan) | measured |
| -38 | A NeurIPS reviewer suggested noise or mode averaging could explain the copycat results | NeurIPS 2020 review page | not opened | unchecked | speculation |

## 7. Limits of this note

- I read HTML versions (ar5iv and arXiv HTML) through a fetch tool that summarises pages, not the PDFs. Key numbers were re-read as verbatim table rows. Figures are images and could not be read: de Haan Fig. 4 and 11, Codevilla Fig. 4, Diffusion Policy Fig. 14, ACT Fig. 6.
- ar5iv may show an arXiv version that differs slightly from the proceedings version, so section numbers may be off by a little.
- Not opened:
  - the NeurIPS 2020 review page for Wen et al. (connection refused; claim -38 comes from a search snippet only);
  - OREO (Park et al. 2021);
  - Muller et al. 2006;
  - VPWEM (arXiv 2603.04910, 2026);
  - the 2025-2026 papers on memory in vision-language-action models.

  I did not survey 2026 work exhaustively.
- In Torne Villasevil et al. Table 3, the regression rows looked surprising and I could not confirm their row labels, so I did not use them.
- No paper here uses a stop-and-go loop like ours, with the robot at rest for every picture and about 1 cm steps. Every transfer to our robot in Section 5 is judgement.

## 8. Conclusion for the project (as returned to the orchestrator)

Evidence: in driving and simulated-control studies, adding earlier frames or the previous action helped behaviour cloning only when the current input lacked needed state such as speed; when it sufficed, history made closed-loop performance worse, sometimes while offline scores improved (de Haan 2019; Wen 2021 CARLA: 42.7% single frame vs 33.0% with history). History-trained policies tend to repeat the previous action when consecutive actions are alike and readable from the frames (Wen 2020). Stops, pauses and small actions made policies stall (Codevilla 2019; Diffusion Policy; ACT); predicting chunks of future actions helped. Diffusion Policy and Octo found one extra frame enough; longer histories hurt diffusion policies unless regularised (Torne Villasevil 2025), while RT-1 and robomimic gained from history in continuous control.

Judgement: in your stop-and-go loop the arm is still at every picture, so an extra frame mainly shows the last step, the copycat cue, and runs of identical moves make copying cheap. Safe default: current picture only, no stacking, predicting the next two or three steps so a twist is learnt with the following move. Test one extra frame separately, judged by closed-loop finishes on held-out sheets, several seeds, and move-repeat rates against the demonstrations.
