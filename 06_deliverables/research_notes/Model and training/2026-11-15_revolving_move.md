# The revolving move: what it is, why the models fail it, and what is built (15.11.26, evening)

Asked by Emranul after the robot test: "The robot has limits, so it has to reposition itself into a comfortable
position where it can see the line and puck and move it; we call it the revolving move. All the models are failing
on it; without it loops are impossible; somehow model E learned it."

## 1. What the revolving move is, measured

In the robot logs of sheet 05's loop (all laid as on 11.11) the forearm angle (joint 4) tells it at once:

| Run | Model | Forearm along the run (degrees, every 3 steps) |
|---|---|---|
| 11.11 run 1 | E | -2 ... -43 -62 -72 -68 -54 -53 -51 -40 -32 -17 -7 4 21 39 55 55 ... 57 |
| A4 (15.11) | E | -0 ... -40 -58 -67 -56 -44 -48 -34 -20 -2 19 35 49 62 63 ... 70 |
| A5 | 15 | -1 ... -49 -60 -65 -61 -53 -47 -46 -47 -55 -68 -83 -99 -115 -117 -117 -117 |
| B1 | 13 | 3 ... -44 -57 -64 -58 -44 -42 -47 -53 -60 -72 -84 -96 -108 -117 -117 -117 |
| C2 | 18 | -0 ... -45 -61 -72 -78 -76 -68 -64 -63 -69 -77 -88 -100 -112 -117 ... |

The revolving move is the forearm swinging back from one side (about -70) through the middle to the other side (+60)
while the rod goes round the puck. A loop needs a full turn of the pushing direction, the forearm can turn only 117
degrees either way, so without the swing every loop runs into the joint limit. Model E asks +6 to +7 degrees a step
during the swing; models 13, 15 and 18 ask +2 to +4, start to swing back, then turn the other way and wind on until
the limit (-117), where they are stuck. The tip paths (`evidence/robot_test_2026-11-15/run_paths.png`) show the same
runs going round the loop again or off the paper.

In the recordings the swing is common, not rare: 153 of 265 episodes have one (forearm from beyond 35 degrees on one
side to beyond 35 on the other): round 1 sheet 02 9 of 45, sheet 03 35 of 45, sheet 04 35 of 45, sheet 05 23 of 30,
sheet 06 21 of 25; round 2 30 episodes on 17 tracks. So the data holds the move; the models do not reproduce it.

## 2. Why the models fail it (measured on held-out course sheets)

- During swings the recipes turn the forearm the right way on 76 to 96 % of the pictures (sheet 05 held out: model
  E's recipe 76 %, the same recipe on all data 90 %, model 15's 80 %), but they ask too little: the mean asked turn is
  0.32 to 0.65 of the recorded one (sheet 05: 1.9 to 2.5 degrees against 5.9 recorded). This is the regression to the
  mean the literature describes for MSE policies: where the demonstrations hold two answers (keep turning, or swing
  back) the model answers the average, which is neither.
- Simply turning the twist up does not fix it: on the held-out pictures a gain of 1.5 leaves the error of the turn
  unchanged (3.34 against 3.37 degrees) and 2.0 to 2.5 makes it worse (3.9 to 4.4).
- One picture cannot show how far the forearm has wound: the camera turns with it. The decision to swing back depends
  on that (the demonstrators swing before the limit). Model E most likely does it on sheet 05 by having learnt that
  sheet's pictures (23 of its 30 sheet-05 episodes swing, in the same place); round 2 adds swings in other places and
  directions, which pulls the answer back to the average. This is an explanation, not tested.

## 3. What the literature says (summary of a search, links in the session's research report)

- MSE regression averages between modes (Chi and others, Diffusion Policy, RSS 2023). Fixes that model several
  answers: diffusion policies (best on Push-T, a pushing benchmark: 0.91 from pictures), discretised or mixture
  heads (Behavior Transformers; mixture density networks), action chunking with a committed chunk (Zhao and others,
  ACT, RSS 2023: chunking took success from 1 % to 44 %).
- Pushing with repositioning is treated as a separate contact mode in model-based work (Hogan and Rodriguez 2016;
  Xue and others 2023, who guide the switch with demonstrations); no paper was found that learns it end to end from
  pictures by behaviour cloning.
- History helps multi-stage tasks (robomimic: BC-RNN well ahead of BC) but invites copying the past (Wen and others,
  "copycat", NeurIPS 2020); keyframe weighting (Wen and others, ICML 2021) counts the change points more.
- "Weight-space programming" is most likely an MIT CSAIL blog post (Takehi and others, 2.11.26): a large language
  model edits another network's weights. Too early and too far from this problem to help here. MIT's liquid neural
  networks (Hasani and others) are a kind of recurrent cell; the GRU below does the same job in a plainer way.

## 4. What is built tonight (all tested on the laptop, nothing on the robot yet)

1. **Recurrent model** (`SequencePictureModel`): the backbone turns each of the last 12 camera pictures, 4 steps apart
   (about 44 cm of path in the recordings; on the robot 6 steps apart for its 0.65 cm steps), into features; a GRU
   reads them in order and the head answers from its last state and the newest picture. Only pictures are its input,
   as the course demands, and the sequence shows how the view has turned on the way, which one picture cannot.
   Checked: the sequences, that the driver reads them exactly as training does (difference 1e-8), that the backbone
   learns only from the newest two pictures (to fit the card), a short training, and the driver on 20 robot pictures.
   Robot models 21 (model 15's recipe, recurrent) and 22 (with swing weighting).
2. **Swing weighting** (`swing_weight`): the pictures inside a forearm swing count three times, with and without the
   recurrent model. Robot model 23 (model 15's recipe, swing weighting only).
3. **A bug fixed** in the direction memory: its own layer was left out of the optimiser, so the memory runs of the
   twelfth queue trained with a memory layer stuck at its random start. They are stopped and run again (model 19).

## 5. A question about the course rule (for Emranul)

The course demands the robot's one camera as the only input. The direction memory (models 19 and 20) feeds the model
the tip's past positions, which come from the arm's joints, not from the camera. Whether that is allowed is Emranul's
and Marshal's call (or the professor's); the recurrent model reads only pictures and is safe either way.

## 6. Next, if the recurrent model does not swing

A head that chooses between modes instead of averaging (a three-way forearm-turn class with a fixed 6-degree turn when
it says "turn", or a mixture head), and on the robot committing to a predicted chunk once a swing has begun. Diffusion
policy is the strongest known answer but is more than an evening's work.
