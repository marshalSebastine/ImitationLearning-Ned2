# Optuna search 2026-11-13_overnight

Model: factorised_classifier; 15 trials finished, 15 pruned or failed; 0.37 hours this time.

Best validation hard-sheet score in the search (20 epochs): 0.632 (trial 26)

Best settings:

```
{
  "score_1_weight": 0.7389009252814246,
  "learning_rate": 0.0003927855113769529,
  "weight_decay": 0.00014087448792374804,
  "class_weight_power": 0.8149812059935831,
  "colour_jitter": 0.18543517264939563,
  "batch_size": 128,
  "dropout": 0.4143544773543584
}
```

Trained in full with these settings: validation hard-sheet score 0.619.

| Trial | State | Score | Settings |
|---|---|---|---|
| 0 | complete | 0.620 | {"score_1_weight": 0.605242017352324, "learning_rate": 0.0008086838418059024, "weight_decay": 1.6609351855152178e-05, "class_weight_power": 0.8147284547191368, "colour_jitter": 0.02047928177497106, "batch_size": 256, "dropout": 0.2464714392810889} |
| 1 | complete | 0.600 | {"score_1_weight": 0.35427567145270245, "learning_rate": 1.8714977053738477e-05, "weight_decay": 0.003316005906578928, "class_weight_power": 0.7532885715774269, "colour_jitter": 0.07689896986486787, "batch_size": 256, "dropout": 0.23843010124686415} |
| 2 | complete | 0.622 | {"score_1_weight": 0.3515400501051982, "learning_rate": 5.322032647839836e-05, "weight_decay": 6.836276982807667e-05, "class_weight_power": 0.30151797991199936, "colour_jitter": 0.32399347582240823, "batch_size": 64, "dropout": 0.4844526530909537} |
| 3 | complete | 0.610 | {"score_1_weight": 0.972027936801666, "learning_rate": 2.994573509337349e-05, "weight_decay": 5.1937678514474675e-05, "class_weight_power": 0.7839235274862646, "colour_jitter": 0.06203012911718725, "batch_size": 128, "dropout": 0.4824177801773436} |
| 4 | complete | 0.620 | {"score_1_weight": 0.2655468142523338, "learning_rate": 0.00012572037940593758, "weight_decay": 0.00032936610432115083, "class_weight_power": 0.3804729253661757, "colour_jitter": 0.27387258674657994, "batch_size": 64, "dropout": 0.20368450762399892} |
| 5 | pruned | 0.507 | {"score_1_weight": 0.6161729653018138, "learning_rate": 2.334950993390276e-05, "weight_decay": 6.767859891237901e-05, "class_weight_power": 0.44567960708009013, "colour_jitter": 0.02978843217214832, "batch_size": 256, "dropout": 0.11967567319343164} |
| 6 | complete | 0.621 | {"score_1_weight": 0.952993938091102, "learning_rate": 0.0003354547502292508, "weight_decay": 1.604259627558811e-05, "class_weight_power": 0.6280091600387979, "colour_jitter": 0.12024574154582092, "batch_size": 128, "dropout": 0.48480289715410657} |
| 7 | pruned | 0.533 | {"score_1_weight": 0.45668693617427925, "learning_rate": 0.0006641580062284394, "weight_decay": 0.00038439000541888916, "class_weight_power": 0.06901189610915481, "colour_jitter": 0.07371522908940215, "batch_size": 128, "dropout": 0.2983728418305677} |
| 8 | pruned | 0.571 | {"score_1_weight": 0.3535013827241388, "learning_rate": 0.0005739822330203455, "weight_decay": 2.129156482991953e-06, "class_weight_power": 0.9846390496523194, "colour_jitter": 0.2506165187982376, "batch_size": 256, "dropout": 0.18587709404612196} |
| 9 | pruned | 0.555 | {"score_1_weight": 0.39304018169014043, "learning_rate": 0.0009939047069145769, "weight_decay": 6.770985654502958e-05, "class_weight_power": 0.08361983543822527, "colour_jitter": 0.022768668863318273, "batch_size": 64, "dropout": 0.27774763363281657} |
| 10 | pruned | 0.525 | {"score_1_weight": 0.57789902356253, "learning_rate": 7.369592309157535e-05, "weight_decay": 4.299884596941194e-05, "class_weight_power": 0.0661529089338104, "colour_jitter": 0.3149173530906727, "batch_size": 64, "dropout": 0.4312890456080242} |
| 11 | complete | 0.616 | {"score_1_weight": 0.8419008571550624, "learning_rate": 0.00018852857825025978, "weight_decay": 0.00011033021573630745, "class_weight_power": 0.4897007757636085, "colour_jitter": 0.23616801498481788, "batch_size": 128, "dropout": 0.4346165076856786} |
| 12 | complete | 0.626 | {"score_1_weight": 0.7431553003738726, "learning_rate": 0.0004169891364922881, "weight_decay": 1.7398565231622392e-06, "class_weight_power": 0.6408141943987183, "colour_jitter": 0.08134312801424642, "batch_size": 128, "dropout": 0.45773453780103485} |
| 13 | pruned | 0.555 | {"score_1_weight": 0.46003106685198847, "learning_rate": 0.00013523097992238745, "weight_decay": 4.04552543549019e-06, "class_weight_power": 0.041127719077350466, "colour_jitter": 0.07762064202984612, "batch_size": 64, "dropout": 0.44234246311623615} |
| 14 | pruned | 0.593 | {"score_1_weight": 0.8464553575010043, "learning_rate": 0.0009248627043138169, "weight_decay": 9.928837231784777e-06, "class_weight_power": 0.557113255236816, "colour_jitter": 0.10824325196198531, "batch_size": 128, "dropout": 0.31431688712510164} |
| 15 | pruned | 0.510 | {"score_1_weight": 0.3061498759489021, "learning_rate": 2.826066402815235e-05, "weight_decay": 7.50547451238245e-05, "class_weight_power": 0.3609600572802071, "colour_jitter": 0.3059518300736205, "batch_size": 64, "dropout": 0.4163319518932351} |
| 16 | complete | 0.621 | {"score_1_weight": 0.825565771044228, "learning_rate": 0.00021299404186774458, "weight_decay": 5.986654260052807e-06, "class_weight_power": 0.6855723976602232, "colour_jitter": 0.21862466817648746, "batch_size": 128, "dropout": 0.34075122842307337} |
| 17 | pruned | 0.526 | {"score_1_weight": 0.5441318887593938, "learning_rate": 4.224493690175561e-05, "weight_decay": 4.14619584789592e-05, "class_weight_power": 0.3812551927971811, "colour_jitter": 0.07904050088916567, "batch_size": 64, "dropout": 0.4973908978241134} |
| 18 | pruned | 0.571 | {"score_1_weight": 0.36489644561390966, "learning_rate": 0.0004994186229778463, "weight_decay": 4.67977939174589e-05, "class_weight_power": 0.5573058172913388, "colour_jitter": 0.22401112486694144, "batch_size": 64, "dropout": 0.4896633839055256} |
| 19 | pruned | 0.576 | {"score_1_weight": 0.5593253654844039, "learning_rate": 0.00043193162751641365, "weight_decay": 2.1036664138542568e-06, "class_weight_power": 0.37617742304558593, "colour_jitter": 0.05146560561333187, "batch_size": 128, "dropout": 0.47370507260910727} |
| 20 | complete | 0.617 | {"score_1_weight": 0.7335377361509727, "learning_rate": 0.0005094717755123455, "weight_decay": 2.6422653367690273e-06, "class_weight_power": 0.47594977256890475, "colour_jitter": 0.19908977246523113, "batch_size": 128, "dropout": 0.4315383196322564} |
| 21 | pruned | 0.579 | {"score_1_weight": 0.7110328341566787, "learning_rate": 0.0005320687346418298, "weight_decay": 1.4136538980406247e-06, "class_weight_power": 0.622314810545496, "colour_jitter": 0.08163950027057104, "batch_size": 128, "dropout": 0.4975749486428507} |
| 22 | pruned | 0.570 | {"score_1_weight": 0.48002025496681583, "learning_rate": 0.000405949088262311, "weight_decay": 1.5530857870702856e-05, "class_weight_power": 0.7809903292216922, "colour_jitter": 0.04447379493173658, "batch_size": 128, "dropout": 0.4752831919073519} |
| 23 | complete | 0.625 | {"score_1_weight": 0.8550290219381761, "learning_rate": 0.00038912604388019234, "weight_decay": 3.0345935285202186e-05, "class_weight_power": 0.6632595017607718, "colour_jitter": 0.10823572142039735, "batch_size": 128, "dropout": 0.42230299740318394} |
| 24 | pruned | 0.571 | {"score_1_weight": 0.7725305422783824, "learning_rate": 0.000422032623412823, "weight_decay": 1.1790475018999506e-06, "class_weight_power": 0.28617938561234046, "colour_jitter": 0.02571247728670198, "batch_size": 128, "dropout": 0.39210925076616077} |
| 25 | complete | 0.626 | {"score_1_weight": 0.8447238378373073, "learning_rate": 0.0004782657321012109, "weight_decay": 0.0001119512651568878, "class_weight_power": 0.7718548186498094, "colour_jitter": 0.14148788334257456, "batch_size": 128, "dropout": 0.38773510280055434} |
| 26 | complete | 0.632 | {"score_1_weight": 0.7389009252814246, "learning_rate": 0.0003927855113769529, "weight_decay": 0.00014087448792374804, "class_weight_power": 0.8149812059935831, "colour_jitter": 0.18543517264939563, "batch_size": 128, "dropout": 0.4143544773543584} |
| 27 | pruned | 0.609 | {"score_1_weight": 0.8562139332994719, "learning_rate": 0.0003962487108442416, "weight_decay": 0.0013440806334294553, "class_weight_power": 0.6253327329246731, "colour_jitter": 0.157233608555949, "batch_size": 128, "dropout": 0.3795944727758815} |
| 28 | complete | 0.619 | {"score_1_weight": 0.8194518560197332, "learning_rate": 0.0001392364821901727, "weight_decay": 4.121032473097285e-05, "class_weight_power": 0.8029387682067373, "colour_jitter": 0.11194314668396901, "batch_size": 128, "dropout": 0.36245263289409313} |
| 29 | complete | 0.630 | {"score_1_weight": 0.781576503206577, "learning_rate": 0.0005087219890054234, "weight_decay": 0.00014919483520753622, "class_weight_power": 0.9239437716368604, "colour_jitter": 0.15579746617674883, "batch_size": 128, "dropout": 0.41014668312404806} |
