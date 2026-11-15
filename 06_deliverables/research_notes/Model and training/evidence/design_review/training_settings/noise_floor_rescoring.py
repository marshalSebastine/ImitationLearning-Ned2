"""
How far apart two runs of the SAME setting fall when only the seed changes, on the same test rows (read only).

Gives the noise floor for the paired comparisons in like_for_like_rescoring.py: the per-fold within-45-degree
difference and the count of episodes stopped right by only one of the two runs. Also compares the end-to-end mirror
run with the plain classes run on the fold both have finished, and gives exact two-sided sign-test p values for the
discordant episode counts. Run with python -B.
"""

import math
import sys

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
sys.path.insert(0, "C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/training_settings")

from pathlib import Path

import numpy

from config import TRAINING_TARGET_PARAMETERS
from like_for_like_rescoring import FOLDS, PREDICTION_FOLDER, scored
from training_targets import load_training_frames

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")


def sign_test_p(first, second):
    """Exact two-sided sign test on discordant counts."""

    total = first + second
    if total == 0:
        return 1.0
    smaller = min(first, second)
    tail = sum(math.comb(total, k) for k in range(0, smaller + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def paired(frames, position_of_frame, first, second, label):
    """Print per-fold paired differences between two prediction sets given as {fold: predictions}."""

    differences = []
    discordant = [0, 0]
    rights = [[], []]
    for fold in first:
        if fold not in second:
            continue
        common = numpy.intersect1d(first[fold]["frame_index"], second[fold]["frame_index"])
        base = scored(frames, position_of_frame, first[fold], common)
        other = scored(frames, position_of_frame, second[fold], common)
        differences.append(other["within_45_deg"] - base["within_45_deg"])
        discordant[0] += sum(1 for e in base["outcome"] if other["outcome"][e] == "right" and base["outcome"][e] != "right")
        discordant[1] += sum(1 for e in base["outcome"] if base["outcome"][e] == "right" and other["outcome"][e] != "right")
        rights[0].append(base["episodes_stopped_right"])
        rights[1].append(other["episodes_stopped_right"])
    print(f"{label}: within 45 differences per fold {[round(d, 3) for d in differences]}, mean {numpy.mean(differences):+.4f}, "
          f"largest |difference| {max(abs(d) for d in differences):.3f}; stopped right {numpy.mean(rights[0]):.3f} against {numpy.mean(rights[1]):.3f}; "
          f"discordant episodes {discordant[0]} against {discordant[1]} (sign test p {sign_test_p(*discordant):.3f})")


def load_probe(name):
    """Return {fold: predictions} of a saved probe variant."""

    return {fold: dict(numpy.load(PREDICTION_FOLDER / f"{name}_{fold}.npz")) for fold in FOLDS if (PREDICTION_FOLDER / f"{name}_{fold}.npz").exists()}


def load_run(run_folder):
    """Return {fold: predictions} of an end-to-end run."""

    return {path.stem.replace("predictions_", ""): dict(numpy.load(path)) for path in sorted((RUNS_FOLDER / run_folder).glob("predictions_*.npz"))}


def main():
    frames = load_training_frames(dict(TRAINING_TARGET_PARAMETERS))
    position_of_frame = {index: position for position, index in enumerate(frames["frame_index"].to_numpy())}
    print("=== Seed against seed, same setting (the noise floor)")
    classes = {"default": load_probe("scores_1_2_3")}
    for seed in (11, 12, 13):
        classes[f"seed {seed}"] = load_probe(f"classes_seed_{seed}")
    names = list(classes)
    for first_position in range(len(names)):
        for second_position in range(first_position + 1, len(names)):
            paired(frames, position_of_frame, classes[names[first_position]], classes[names[second_position]],
                   f"classes {names[first_position]} vs {names[second_position]}")
    exact = {"default": load_probe("exact_step")}
    for seed in (11, 12, 13):
        exact[f"seed {seed}"] = load_probe(f"exact_seed_{seed}")
    names = list(exact)
    for first_position in range(len(names)):
        for second_position in range(first_position + 1, len(names)):
            paired(frames, position_of_frame, exact[names[first_position]], exact[names[second_position]],
                   f"exact {names[first_position]} vs {names[second_position]}")
    print("\n=== Settings compared on the same rows")
    paired(frames, position_of_frame, classes["default"], load_probe("no_class_weights"), "classes square-root weights vs no class weights")
    paired(frames, position_of_frame, classes["default"], load_probe("scores_weighted"), "classes equal score weights vs 0.4/0.7/1.0")
    paired(frames, position_of_frame, classes["default"], load_probe("square_root_weights_label_smoothing"), "classes vs label smoothing 0.1")
    print("\n=== End to end, mirror run against the plain classes run (same seed, same folds finished so far)")
    paired(frames, position_of_frame, load_run("2026-11-09_resnet18_finetune_classes"), load_run("2026-11-09_resnet18_finetune_mirror"),
           "end-to-end classes vs classes with mirroring")
    paired(frames, position_of_frame, load_run("2026-11-09_resnet18_finetune_classes"), load_run("2026-11-09_small_cnn_scratch_classes"),
           "end-to-end ResNet-18 classes vs small CNN classes")
    print("\nsign test p values used in the section:")
    for first, second in ((24, 16), (14, 24), (17, 8), (14, 23), (23, 20), (21, 17)):
        print(f"   {first} against {second}: p {sign_test_p(first, second):.3f}")


if __name__ == "__main__":
    main()
