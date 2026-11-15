## The comparison set-up that now exists (02_code, branch emranul/training-analysis)

| File | What it does |
|---|---|
| `training_dataset.py` | Reads `03_data/raw/main_data` without changing it and builds `03_data/processed/v01_training_index/`: `frames.csv` (one row per picture with its labels, the episode's facts and measured picture statistics), `episodes.csv`, small grey copies of every picture, and two picture caches (224 x 224 and 160 x 120) |
| `training_targets.py` | The rows a model trains on (scores 1 to 3, mistake rows left out, the plan's start end), the targets (move class, exact step, chunk of future tip places, forearm twist, stop and near-goal), and the validation splits (one sheet held out; whole episodes held out) |
| `training_metrics.py` | The measures every model gets: push direction against the demonstrated step and against the path direction, class accuracy, stop per episode (right, early, never, with a threshold tuned on validation episodes), twist, lean and hand signs, chunk errors |
| `training_features.py` | Frozen features of every picture from six backbones (ResNet-18 pretrained and untrained, EfficientNet-B0, MobileNetV3, DINOv2-small, CLIP ViT-B/32), and the time per picture of each |
| `training_probes.py` | Quick comparisons: small heads on frozen features, one experiment per question, all folds, results in `04_runs/2026-11-09_frozen_feature_probes/`; also the rules without a model (`baselines`) |
| `train_policy.py` | Trains a whole network on the pictures (a small network from scratch, or a pretrained one fine-tuned), with label-aware augmentation, fold by fold; results in `04_runs/<date>_<run>/` |
| `training_error_analysis.py` | Where a model goes wrong (sheet, bends, near the goal, forearm turn, light, camera view, score), and what averaging the direction over the last pictures would give |
| `compare_runs.py` | One table of everything run so far: `04_runs/comparison.md` |
| `test_training.py` | Tests on small made-up examples, and a check that the direction rule gives the recording tool's own labels on all real moving rows |

Settings live in `config.py` only, in `TRAINING_DATA_PARAMETERS`, `TRAINING_FEATURE_PARAMETERS`,
`TRAINING_TARGET_PARAMETERS`, `TRAINING_METRIC_PARAMETERS`, `TRAINING_PROBE_PARAMETERS`, `TRAINING_PROBE_EXPERIMENTS`,
`TRAINING_RUN_PARAMETERS`, `TRAINING_RUNS`, `TRAINING_COMPARISON_PARAMETERS` and `TRAINING_ERROR_ANALYSIS_PARAMETERS`.
`config.main_checkout_folder()` lets the code in a git worktree find the data in the main folder. Model weights and
per-picture prediction files stay out of git (`.gitignore` and `04_runs/.gitignore`).

Installed into `.venv` on 9.11.26 (from PyPI, after a dry run showed that nothing already installed would change):
torch 2.15.0 (processor build), torchvision 0.29.2, timm 1.0.31, pandas 3.1.1, scikit-learn 1.9.1. Marshal needs the
same on his Mac (`pip install torch torchvision timm pandas scikit-learn`); on a Mac, PyTorch can use the Apple GPU
("mps"), which `train_policy.py` does not use yet.
