# The two-picture model (14.11.26)

Asked for by Emranul on 14.11.26 ("can u not make another model before marshal comes?"), before Marshal's own model
from the round 1 data is tested at the robot.

## What changed, and why

Every model so far saw one picture. One picture cannot show which way the puck came from (in a loop, at a crossing,
in a bend), so the model must guess its direction from the line alone. The two-picture model gets the camera's picture
**and the one taken just before it**, stacked as six colour channels. Everything else is model E's recipe (chunk
target, stop weighted 25, ResNet-18, 8 rounds, same augmentation, same data).

- Training (`train_policy.py`, `training_targets.previous_frame_indices`): the picture before is the camera's previous
  picture in the same episode, also where the cleaning dropped that row; an episode's first picture stands in for its
  own previous one. Both pictures of a row get the same augmentation (light, blur, noise, shift), so the only change
  between them is the robot's own move. timm repeats the pretrained first layer for the second picture and halves it.
- Driver (`model_mode.py`, `ModelPolicy`): keeps the picture of the step before; a new run forgets it, so the first
  picture of a run stands in for its own previous one, as in training. One-picture models behave exactly as before.
- On the robot the step before should be about as long as in the demonstrations (median 0.61 cm): so the model is
  meant for 0.65 cm steps, not 1 cm.

Settings: `TRAINING_RUNS["resnet18_two_pictures"]`, `..._seed_2` and `resnet18_model_7_two_pictures_all_sheets` in
`config.py`. Tests: `test_training.py` (previous pictures, equal augmentation, six-channel model) and
`test_model_mode.py` (the driver carries the picture before and forgets it between runs).

## Rule, fixed before the results

Kept only when **both** seeds beat model E's recipe (path within 45: 0.808 and 0.800, mean 0.804) by 1.5 points or more,
stop right no more than 8 episodes less often (103 and 105 of 209), **and** lose no more than 1 point of path within 45
on the turning pictures (path direction turning by 20 degrees or more from the step before; model E's recipe 0.737 and
0.735 on 3,784 such pictures). The last part guards against the trap of models with a history (de Haan and others
2019, causal confusion): they can score well offline by repeating the last move and then fail on the robot exactly
where the move must change. Scoring: `evidence/two_pictures_2026-11-14/score_two_pictures.py`.

## Results (ki-014, 14.11.26, 10:58 to 11:19; scores in `evidence/two_pictures_2026-11-14/two_pictures_scores.json`)

| Run | Path within 45 | Right stops of 209 (early, never) | Turning pictures | Steady pictures |
|---|---|---|---|---|
| Model E recipe, seed 1 | 0.808 | 103 (27, 71) | 0.737 | 0.821 |
| Model E recipe, seed 2 | 0.800 | 105 (39, 51) | 0.735 | 0.814 |
| Two pictures, seed 1 | 0.823 | 94 (31, 74) | 0.755 | 0.831 |
| Two pictures, seed 2 | 0.817 | 85 (39, 75) | 0.753 | 0.827 |

By the rule: **not kept.**
- Path: seed 1 is 1.9 points above model E's mean (0.804), seed 2 1.3 points: seed 2 misses the 1.5 points by 0.2.
  (Seed against seed, both gain 1.5 and 1.7 points; the rule as written compares with the mean.)
- Stops: 94 and 85 right, against 103 and 105: 9 to 20 fewer, more than the 8 allowed. This is the clear failure.
- Turning pictures: 1.8 points BETTER, more than on the steady pictures (1.0 to 1.2): no sign of the "repeat the last
  move" trap. The previous picture helps most where the line turns.

By sheet, path within 45 gains most on sheet 01 (0.894 and 0.881 against 0.864 and 0.847) and on the loop, sheet 05
(0.837 and 0.840 against 0.811 and 0.794); sheet 03 and sheet 04 (the sharp bend that failed on the robot) gain under
one point, sheet 06 none. The right stops fall mostly on sheets 03, 05 and 06.

Model 7 (the all-sheets copy for the robot, 18,032 pictures, 4.6 min): checkpoint sha256 537d4d2096ab, loads in the
driver, 15 ms per decision on the laptop (model E about 15). It has no offline score of its own.
