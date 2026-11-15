# Mid-way check of round 2 (14.11.26, during the break)

Asked for by Emranul ("can we pause halfway and train a model to see the difference"). Data: round 1 (209 episodes,
18,032 training rows) plus the 31 scored round 2 episodes recorded until 18:10 on 14.11.26 (T01 to T09, 2,358 rows),
built into `03_data/processed/v02_training_index` (round 1's index unchanged). Trained on ki-014, 18:40 to 18:55.
Recipe: model E's (chunk, stop weighted 25, ResNet-18, 8 rounds), one seed. Scores:
`evidence/midway_2026-11-14/midway_scores.json` (script beside it).

## The test

Each of the 9 new tracks held out once: trained on round 1 and the other 8 new tracks, tested on both prints of the
held-out track (a line the model has never seen). Against the same recipe trained on round 1 only, tested on the same
pictures. Rule, fixed in `config.py` before the results: the round 2 data helps when path within 45 on the held-out new
tracks rises by 2 points or more on average, without fewer episodes stopped right.

## Results

| Track | Path within 45, with round 2 | Round 1 only | Right stops, with round 2 | Round 1 only |
|---|---|---|---|---|
| T01 gentle S-curves | 0.869 | 0.804 | 1 of 3 | 2 of 3 |
| T02 two sharp bends | 0.766 | 0.731 | 1 of 3 | 3 of 3 |
| T03 loop | 0.830 | 0.808 | 0 of 3 | 0 of 3 |
| T04 hairpin and medium bend | 0.817 | 0.808 | 1 of 3 | 3 of 3 |
| T05 loop and corner | 0.783 | 0.759 | 0 of 3 | 0 of 3 |
| T06 hairpin and sharp bend | 0.748 | 0.748 | 1 of 4 | 0 of 4 |
| T07 two corners | 0.832 | 0.821 | 0 of 4 | 0 of 4 |
| T08 loop and hairpin | 0.847 | 0.844 | 0 of 4 | 0 of 4 |
| T09 short gentle | 0.895 | 0.851 | 3 of 4 | 1 of 4 |
| **Mean / total** | **0.821** | **0.797** | **7 of 31** | **9 of 31** |

By the rule: the steering part passes (+2.4 points, better on 8 of 9 tracks and worse on none); the stop part does not
(7 against 9 right stops), so the rule as written is NOT passed. The stop counts are small (31 episodes, one seed; a
difference of two can be chance) and low for both models on new lines (a quarter of the episodes): stopping on an
unseen line's end is the weak point either way.

Model 8 (round 1 plus all 31 round 2 episodes, all sheets, for the robot): 20,390 pictures, 3.3 min, checkpoint
`04_runs/gpu_ki014/2026-11-14_resnet18_model_8_round_2_part_all_sheets/checkpoints/all_sheets.pt` (sha256
74aa77af60b9), loads in the driver. It has no offline score of its own; the robot is its test.
