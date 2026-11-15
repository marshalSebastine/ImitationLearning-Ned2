## How to compare models yourself

Everything runs from `02_code` with the project's Python. All settings are in `02_code/config.py`, in the dictionaries
whose names start with `TRAINING_`. Nothing touches the robot, and nothing in `03_data/raw` is changed.

| Step | Command (run from `02_code`) | What it does | Time on the laptop |
|---|---|---|---|
| 1 | `..\.venv\Scripts\python.exe training_dataset.py` | Builds one table of all kept pictures with their labels, plus picture caches, in `03_data/processed/v01_training_index/` | about 1 minute |
| 2 | `..\.venv\Scripts\python.exe training_features.py` | Turns every picture into numbers with six pretrained image models (frozen features), and times each model | about 55 minutes |
| 3 | `..\.venv\Scripts\python.exe training_probes.py backbones` | Trains small heads on the frozen features and scores them on held-out sheets; any experiment name of `TRAINING_PROBE_EXPERIMENTS`, `baselines`, or `all` | 2 to 40 minutes per experiment |
| 4 | `..\.venv\Scripts\python.exe train_policy.py resnet18_finetune_classes --folds without_path_03` | Trains a whole network on the pictures (end to end) for one named run of `TRAINING_RUNS`; without `--folds` it does all six held-out sheets | about 20 to 30 minutes per held-out sheet |
| 5 | `..\.venv\Scripts\python.exe compare_runs.py` | Collects every result into `04_runs/comparison.md` | seconds |
| 6 | `..\.venv\Scripts\python.exe training_error_analysis.py resnet18_imagenet_grid` | Shows where a model goes wrong: per sheet, bends against straight stretches, near the goal, light, camera view | under a minute |
| Test | `..\.venv\Scripts\python.exe test_training.py` | Checks the training code on small made-up examples and against the real labels | under a minute |

To try a new idea, add one entry to `TRAINING_RUNS` (or a variant to an experiment in `TRAINING_PROBE_EXPERIMENTS`)
that names only what differs from the defaults, run it, and run `compare_runs.py`. Every model is scored on the same
pictures with the same measures, so the numbers can be compared directly.

Quick check before a long run (one sheet, one epoch, 512 pictures, results in a folder ending in `_check` that git
ignores):

    ..\.venv\Scripts\python.exe train_policy.py resnet18_finetune_classes --folds without_path_04 --epochs 1 --train-row-limit 512

## Words used in this report

| Word | Meaning here |
|---|---|
| Held-out sheet (leave one sheet out) | Train on five sheets, test on the sixth, six times. It stands in for an unseen exam track. |
| Fold | One of those six train-and-test rounds. "Worst fold" is the sheet the model did worst on. |
| Backbone | The pretrained part of a network that turns a picture into numbers (for example ResNet-18, trained on ImageNet photos). |
| Frozen features | The backbone is used as it is; only a small head on top learns. Fast, but the backbone never learns about lines and pucks. |
| Fine-tuned (end to end) | The whole network, backbone included, learns from our pictures. Slower on a processor, much better here. |
| Head | The last small layers that turn the numbers into a move, a twist and a stop decision. |
| Move classes | One of 8 fixed directions in the picture (forward, forward-left, ...), or stop. |
| Exact step | The demonstrated move as numbers: how far forward and left in the picture, and how much the forearm turned. |
| Path direction | Where the tip is three steps later, seen from the current picture. Less noisy than one hand-guided step. |
| Within 45 degrees | The share of pictures where the predicted push direction is less than 45 degrees off the demonstrated one. |
| Stopped right / early / never | Per held-out episode: the model's first stop is within 2 cm of where the demonstration stopped / more than 3 cm before it / never comes. |
| Augmentation | Changing training pictures (light, blur, a small shift) so the model copes with changes it will meet, without changing what the right move is. |
| Always forward | A rule that always pushes straight up the picture. Any model must beat it clearly. |
