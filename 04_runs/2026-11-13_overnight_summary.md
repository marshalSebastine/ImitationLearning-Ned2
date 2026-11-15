# Training batch 2026-11-13_overnight

Validation scores of the best epoch, mean ± spread over seeds. Hard-sheet score: mean over sheets 03, 04 and 06 of (move balanced accuracy + forearm accuracy) / 2. Test scores are for the report only; nothing is chosen by them.

| Policy | Seeds | Hard-sheet score | Move balanced accuracy | Forearm accuracy | Stop recall | Test hard-sheet score |
|---|---|---|---|---|---|---|
| optuna_trial_026 | 1 | 0.632 | 0.505 | 0.690 | 0.903 | 0.608 |
| optuna_trial_029 | 1 | 0.630 | 0.512 | 0.704 | 0.839 | 0.607 |
| optuna_trial_025 | 1 | 0.626 | 0.513 | 0.688 | 0.871 | 0.610 |
| optuna_trial_012 | 1 | 0.626 | 0.500 | 0.715 | 0.710 | 0.600 |
| optuna_trial_023 | 1 | 0.625 | 0.504 | 0.711 | 0.774 | 0.604 |
| optuna_trial_002 | 1 | 0.622 | 0.482 | 0.696 | 0.710 | 0.583 |
| optuna_trial_006 | 1 | 0.621 | 0.487 | 0.695 | 0.903 | 0.609 |
| optuna_trial_016 | 1 | 0.621 | 0.492 | 0.690 | 0.903 | 0.611 |
| optuna_trial_004 | 1 | 0.620 | 0.483 | 0.710 | 0.645 | 0.590 |
| optuna_trial_000 | 1 | 0.620 | 0.490 | 0.691 | 0.710 | 0.592 |
| factorised_classifier_tuned | 1 | 0.619 | 0.512 | 0.669 | 0.935 | 0.591 |
| optuna_trial_028 | 1 | 0.619 | 0.487 | 0.718 | 0.677 | 0.599 |
| history_classifier | 3 | 0.618 ± 0.001 | 0.499 ± 0.004 | 0.704 ± 0.006 | 0.710 ± 0.053 | 0.603 ± 0.006 |
| optuna_trial_020 | 1 | 0.617 | 0.506 | 0.696 | 0.903 | 0.621 |
| optuna_trial_011 | 1 | 0.616 | 0.493 | 0.693 | 0.871 | 0.597 |
| optuna_trial_003 | 1 | 0.610 | 0.463 | 0.701 | 0.806 | 0.600 |
| factorised_classifier | 3 | 0.609 ± 0.007 | 0.476 ± 0.012 | 0.679 ± 0.005 | 0.871 ± 0.091 | 0.606 ± 0.007 |
| optuna_trial_027 | 1 | 0.609 | 0.488 | 0.684 | 0.806 | 0.622 |
| optuna_trial_014 | 1 | 0.601 | 0.465 | 0.685 | 0.710 | 0.587 |
| optuna_trial_001 | 1 | 0.600 | 0.445 | 0.696 | 0.806 | 0.599 |
| exact_step_regressor | 3 | 0.598 ± 0.007 | 0.461 ± 0.005 | 0.710 ± 0.002 | 0.667 ± 0.055 | 0.592 ± 0.007 |
| optuna_trial_008 | 1 | 0.593 | 0.471 | 0.681 | 0.839 | 0.577 |
| optuna_trial_009 | 1 | 0.582 | 0.451 | 0.674 | 0.839 | 0.570 |
| optuna_trial_021 | 1 | 0.579 | 0.446 | 0.669 | 0.742 | 0.577 |
| chunk_transformer | 3 | 0.579 ± 0.003 | 0.419 ± 0.019 | 0.699 ± 0.008 | 0.452 ± 0.121 | 0.561 ± 0.008 |
| optuna_trial_019 | 1 | 0.576 | 0.424 | 0.693 | 0.581 | 0.561 |
| optuna_trial_018 | 1 | 0.571 | 0.467 | 0.650 | 0.968 | 0.580 |
| optuna_trial_024 | 1 | 0.571 | 0.397 | 0.695 | 0.258 | 0.570 |
| optuna_trial_022 | 1 | 0.570 | 0.432 | 0.682 | 0.548 | 0.579 |
| optuna_trial_013 | 1 | 0.555 | 0.401 | 0.668 | 0.613 | 0.553 |
| optuna_trial_007 | 1 | 0.533 | 0.375 | 0.690 | 0.161 | 0.557 |
| optuna_trial_017 | 1 | 0.526 | 0.358 | 0.665 | 0.774 | 0.523 |
| optuna_trial_010 | 1 | 0.525 | 0.316 | 0.669 | 0.323 | 0.526 |
| optuna_trial_015 | 1 | 0.510 | 0.339 | 0.649 | 0.742 | 0.505 |
| optuna_trial_005 | 1 | 0.507 | 0.343 | 0.646 | 0.871 | 0.497 |
| diffusion_policy | 3 | 0.476 ± 0.017 | 0.336 ± 0.020 | 0.572 ± 0.014 | 0.484 ± 0.046 | 0.457 ± 0.018 |

Move balanced accuracy per sheet (validation):

| Policy | path_01 | path_02 | path_03 | path_04 | path_05 | path_06 |
|---|---|---|---|---|---|---|
| chunk_transformer | 0.201 ± 0.042 | 0.338 ± 0.035 | 0.399 ± 0.015 | 0.398 ± 0.014 | 0.515 ± 0.063 | 0.509 ± 0.054 |
| diffusion_policy | 0.285 ± 0.043 | 0.265 ± 0.013 | 0.336 ± 0.018 | 0.302 ± 0.046 | 0.424 ± 0.029 | 0.401 ± 0.029 |
| exact_step_regressor | 0.265 ± 0.019 | 0.398 ± 0.007 | 0.416 ± 0.016 | 0.424 ± 0.019 | 0.581 ± 0.021 | 0.545 ± 0.028 |
| factorised_classifier | 0.312 ± 0.064 | 0.372 ± 0.025 | 0.482 ± 0.023 | 0.451 ± 0.023 | 0.554 ± 0.028 | 0.574 ± 0.035 |
| factorised_classifier_tuned | 0.435 | 0.436 | 0.491 | 0.511 | 0.538 | 0.605 |
| history_classifier | 0.289 ± 0.094 | 0.432 ± 0.014 | 0.491 ± 0.026 | 0.480 ± 0.023 | 0.573 ± 0.013 | 0.563 ± 0.020 |
| optuna_trial_000 | 0.278 | 0.401 | 0.457 | 0.508 | 0.583 | 0.567 |
| optuna_trial_001 | 0.289 | 0.311 | 0.465 | 0.421 | 0.491 | 0.540 |
| optuna_trial_002 | 0.235 | 0.381 | 0.482 | 0.458 | 0.588 | 0.556 |
| optuna_trial_003 | 0.280 | 0.338 | 0.491 | 0.440 | 0.483 | 0.554 |
| optuna_trial_004 | 0.314 | 0.382 | 0.484 | 0.492 | 0.537 | 0.516 |
| optuna_trial_005 | 0.272 | 0.282 | 0.326 | 0.326 | 0.333 | 0.440 |
| optuna_trial_006 | 0.465 | 0.400 | 0.492 | 0.471 | 0.519 | 0.563 |
| optuna_trial_007 | 0.366 | 0.319 | 0.308 | 0.353 | 0.447 | 0.420 |
| optuna_trial_008 | 0.399 | 0.387 | 0.444 | 0.455 | 0.538 | 0.546 |
| optuna_trial_009 | 0.480 | 0.389 | 0.458 | 0.412 | 0.442 | 0.506 |
| optuna_trial_010 | 0.175 | 0.239 | 0.327 | 0.342 | 0.264 | 0.381 |
| optuna_trial_011 | 0.387 | 0.398 | 0.487 | 0.475 | 0.549 | 0.547 |
| optuna_trial_012 | 0.311 | 0.392 | 0.460 | 0.488 | 0.627 | 0.573 |
| optuna_trial_013 | 0.383 | 0.298 | 0.408 | 0.371 | 0.390 | 0.493 |
| optuna_trial_014 | 0.363 | 0.361 | 0.456 | 0.452 | 0.527 | 0.544 |
| optuna_trial_015 | 0.272 | 0.258 | 0.341 | 0.348 | 0.319 | 0.403 |
| optuna_trial_016 | 0.376 | 0.396 | 0.538 | 0.444 | 0.508 | 0.574 |
| optuna_trial_017 | 0.275 | 0.258 | 0.342 | 0.347 | 0.386 | 0.449 |
| optuna_trial_018 | 0.482 | 0.409 | 0.417 | 0.400 | 0.542 | 0.584 |
| optuna_trial_019 | 0.338 | 0.323 | 0.404 | 0.402 | 0.498 | 0.504 |
| optuna_trial_020 | 0.405 | 0.447 | 0.518 | 0.458 | 0.535 | 0.560 |
| optuna_trial_021 | 0.488 | 0.313 | 0.430 | 0.399 | 0.511 | 0.563 |
| optuna_trial_022 | 0.299 | 0.386 | 0.416 | 0.400 | 0.455 | 0.496 |
| optuna_trial_023 | 0.288 | 0.423 | 0.446 | 0.526 | 0.602 | 0.521 |
| optuna_trial_024 | 0.305 | 0.319 | 0.368 | 0.345 | 0.492 | 0.541 |
| optuna_trial_025 | 0.333 | 0.409 | 0.507 | 0.507 | 0.597 | 0.595 |
| optuna_trial_026 | 0.409 | 0.390 | 0.550 | 0.484 | 0.543 | 0.593 |
| optuna_trial_027 | 0.309 | 0.428 | 0.530 | 0.428 | 0.546 | 0.558 |
| optuna_trial_028 | 0.323 | 0.365 | 0.483 | 0.474 | 0.597 | 0.532 |
| optuna_trial_029 | 0.398 | 0.405 | 0.449 | 0.518 | 0.607 | 0.567 |

| Run | Best epoch | Epochs | Stopped because | Minutes | Device |
|---|---|---|---|---|---|
| chunk_transformer seed 1 | 44 | 54 | no gain for 10 epochs | 4.26 | NVIDIA GeForce RTX 2080 SUPER |
| chunk_transformer seed 2 | 26 | 36 | no gain for 10 epochs | 2.89 | NVIDIA GeForce RTX 2080 SUPER |
| chunk_transformer seed 3 | 25 | 35 | no gain for 10 epochs | 2.83 | NVIDIA GeForce RTX 2080 SUPER |
| diffusion_policy seed 1 | 142 | 150 | all epochs | 9.17 | NVIDIA GeForce RTX 2080 SUPER |
| diffusion_policy seed 2 | 36 | 76 | no gain for 40 epochs | 4.66 | NVIDIA GeForce RTX 2080 SUPER |
| diffusion_policy seed 3 | 120 | 150 | all epochs | 9.18 | NVIDIA GeForce RTX 2080 SUPER |
| exact_step_regressor seed 1 | 19 | 29 | no gain for 10 epochs | 1.68 | NVIDIA GeForce RTX 2080 SUPER |
| exact_step_regressor seed 2 | 17 | 27 | no gain for 10 epochs | 1.57 | NVIDIA GeForce RTX 2080 SUPER |
| exact_step_regressor seed 3 | 21 | 31 | no gain for 10 epochs | 1.81 | NVIDIA GeForce RTX 2080 SUPER |
| factorised_classifier seed 1 | 5 | 15 | no gain for 10 epochs | 0.87 | NVIDIA GeForce RTX 2080 SUPER |
| factorised_classifier seed 2 | 9 | 19 | no gain for 10 epochs | 1.1 | NVIDIA GeForce RTX 2080 SUPER |
| factorised_classifier seed 3 | 13 | 23 | no gain for 10 epochs | 1.33 | NVIDIA GeForce RTX 2080 SUPER |
| factorised_classifier_tuned seed 1 | 5 | 15 | no gain for 10 epochs | 0.87 | NVIDIA GeForce RTX 2080 SUPER |
| history_classifier seed 1 | 7 | 17 | no gain for 10 epochs | 2.5 | NVIDIA GeForce RTX 2080 SUPER |
| history_classifier seed 2 | 9 | 19 | no gain for 10 epochs | 2.8 | NVIDIA GeForce RTX 2080 SUPER |
| history_classifier seed 3 | 9 | 19 | no gain for 10 epochs | 2.79 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_000 seed 1 | 18 | 20 | all epochs | 1.04 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_001 seed 1 | 19 | 20 | all epochs | 1.06 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_002 seed 1 | 19 | 20 | all epochs | 1.32 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_003 seed 1 | 15 | 20 | all epochs | 1.19 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_004 seed 1 | 9 | 20 | all epochs | 1.33 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_005 seed 1 | 3 | 3 | pruned by Optuna | 0.16 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_006 seed 1 | 6 | 20 | all epochs | 1.16 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_007 seed 1 | 3 | 3 | pruned by Optuna | 0.18 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_008 seed 1 | 5 | 6 | pruned by Optuna | 0.32 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_009 seed 1 | 4 | 5 | pruned by Optuna | 0.33 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_010 seed 1 | 3 | 3 | pruned by Optuna | 0.21 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_011 seed 1 | 6 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_012 seed 1 | 20 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_013 seed 1 | 3 | 3 | pruned by Optuna | 0.2 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_014 seed 1 | 3 | 5 | pruned by Optuna | 0.3 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_015 seed 1 | 3 | 3 | pruned by Optuna | 0.2 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_016 seed 1 | 5 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_017 seed 1 | 3 | 3 | pruned by Optuna | 0.2 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_018 seed 1 | 3 | 3 | pruned by Optuna | 0.21 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_019 seed 1 | 3 | 3 | pruned by Optuna | 0.18 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_020 seed 1 | 5 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_021 seed 1 | 3 | 3 | pruned by Optuna | 0.18 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_022 seed 1 | 3 | 3 | pruned by Optuna | 0.18 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_023 seed 1 | 13 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_024 seed 1 | 3 | 3 | pruned by Optuna | 0.18 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_025 seed 1 | 11 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_026 seed 1 | 5 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_027 seed 1 | 5 | 5 | pruned by Optuna | 0.3 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_028 seed 1 | 13 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
| optuna_trial_029 seed 1 | 13 | 20 | all epochs | 1.17 | NVIDIA GeForce RTX 2080 SUPER |
