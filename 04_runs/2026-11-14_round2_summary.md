# Training batch 2026-11-14_round2

Validation scores of the best epoch, mean ± spread over seeds. Hard-sheet score: mean over sheets 03, 04 and 06 of (move balanced accuracy + forearm accuracy) / 2. Test scores are for the report only; nothing is chosen by them.
Unseen-track score: the same mean of (move balanced accuracy + forearm accuracy) / 2 over the episodes of the track shapes never trained on (round 2: T01, T08, T11, T15), when the dataset has them.

| Policy | Seeds | Hard-sheet score | Move balanced accuracy | Forearm accuracy | Stop recall | Test hard-sheet score | Unseen-track score |
|---|---|---|---|---|---|---|---|
| factorised_classifier_tuned | 3 | 0.604 ± 0.002 | 0.469 ± 0.003 | 0.700 ± 0.008 | 0.785 ± 0.066 | 0.601 ± 0.005 | 0.483 ± 0.009 |
| factorised_classifier | 3 | 0.601 ± 0.006 | 0.462 ± 0.007 | 0.696 ± 0.012 | 0.763 ± 0.110 | 0.581 ± 0.009 | 0.476 ± 0.024 |
| history_classifier | 1 | 0.592 | 0.487 | 0.700 | 0.839 | 0.607 | 0.505 |
| exact_step_regressor | 3 | 0.583 ± 0.015 | 0.435 ± 0.010 | 0.706 ± 0.014 | 0.720 ± 0.076 | 0.577 ± 0.018 | 0.466 ± 0.009 |
| chunk_transformer | 3 | 0.562 ± 0.002 | 0.417 ± 0.015 | 0.691 ± 0.007 | 0.495 ± 0.135 | 0.564 ± 0.006 | 0.448 ± 0.007 |
| diffusion_policy | 3 | 0.452 ± 0.008 | 0.326 ± 0.007 | 0.547 ± 0.007 | 0.344 ± 0.030 | 0.444 ± 0.016 | 0.361 ± 0.013 |

Move balanced accuracy per sheet (validation):

| Policy | path_01 | path_02 | path_03 | path_04 | path_05 | path_06 |
|---|---|---|---|---|---|---|
| chunk_transformer | 0.257 ± 0.061 | 0.346 ± 0.026 | 0.360 ± 0.007 | 0.363 ± 0.021 | 0.537 ± 0.025 | 0.500 ± 0.021 |
| diffusion_policy | 0.284 ± 0.049 | 0.278 ± 0.004 | 0.281 ± 0.013 | 0.249 ± 0.019 | 0.428 ± 0.008 | 0.424 ± 0.040 |
| exact_step_regressor | 0.338 ± 0.030 | 0.351 ± 0.023 | 0.395 ± 0.024 | 0.391 ± 0.002 | 0.529 ± 0.015 | 0.523 ± 0.024 |
| factorised_classifier | 0.385 ± 0.068 | 0.362 ± 0.019 | 0.453 ± 0.027 | 0.420 ± 0.027 | 0.543 ± 0.041 | 0.557 ± 0.006 |
| factorised_classifier_tuned | 0.344 ± 0.033 | 0.385 ± 0.013 | 0.465 ± 0.015 | 0.449 ± 0.007 | 0.525 ± 0.025 | 0.526 ± 0.026 |
| history_classifier | 0.352 | 0.455 | 0.426 | 0.400 | 0.584 | 0.600 |

| Run | Best epoch | Epochs | Stopped because | Minutes | Device |
|---|---|---|---|---|---|
| chunk_transformer seed 1 | 25 | 35 | no gain for 10 epochs | 3.42 | NVIDIA GeForce RTX 2080 SUPER |
| chunk_transformer seed 2 | 35 | 45 | no gain for 10 epochs | 4.4 | NVIDIA GeForce RTX 2080 SUPER |
| chunk_transformer seed 3 | 39 | 49 | no gain for 10 epochs | 4.79 | NVIDIA GeForce RTX 2080 SUPER |
| diffusion_policy seed 1 | 140 | 150 | all epochs | 11.28 | NVIDIA GeForce RTX 2080 SUPER |
| diffusion_policy seed 2 | 149 | 150 | all epochs | 11.3 | NVIDIA GeForce RTX 2080 SUPER |
| diffusion_policy seed 3 | 130 | 150 | all epochs | 11.31 | NVIDIA GeForce RTX 2080 SUPER |
| exact_step_regressor seed 1 | 19 | 29 | no gain for 10 epochs | 2.08 | NVIDIA GeForce RTX 2080 SUPER |
| exact_step_regressor seed 2 | 30 | 40 | no gain for 10 epochs | 2.88 | NVIDIA GeForce RTX 2080 SUPER |
| exact_step_regressor seed 3 | 8 | 18 | no gain for 10 epochs | 1.3 | NVIDIA GeForce RTX 2080 SUPER |
| factorised_classifier seed 1 | 34 | 40 | all epochs | 2.72 | NVIDIA GeForce RTX 3060 Ti |
| factorised_classifier seed 2 | 6 | 16 | no gain for 10 epochs | 1.09 | NVIDIA GeForce RTX 3060 Ti |
| factorised_classifier seed 3 | 13 | 23 | no gain for 10 epochs | 3.31 | NVIDIA GeForce RTX 3060 Ti |
| factorised_classifier_tuned seed 1 | 18 | 28 | no gain for 10 epochs | 1.91 | NVIDIA GeForce RTX 3060 Ti |
| factorised_classifier_tuned seed 2 | 16 | 26 | no gain for 10 epochs | 2.25 | NVIDIA GeForce RTX 3060 Ti |
| factorised_classifier_tuned seed 3 | 23 | 33 | no gain for 10 epochs | 4.71 | NVIDIA GeForce RTX 3060 Ti |
| history_classifier seed 1 | 14 | 24 | no gain for 10 epochs | 4.23 | NVIDIA GeForce RTX 3060 Ti |
