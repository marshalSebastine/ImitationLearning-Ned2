"""
Collect the per-fold evidence behind the training-settings section (read only; prints to standard output).

Reads the probe and end-to-end result files of the training-analysis worktree and prints, for every comparison the
section uses, the per-fold values, the paired per-fold differences against the default and how many folds agree in
sign. Also estimates the spread between seeds and the number of seeds needed to tell two settings apart.
Nothing under D:/AI_Project is written.
"""

import json
import math
import statistics
from pathlib import Path

RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
PROBE_FOLDER = RUNS_FOLDER / "2026-11-09_frozen_feature_probes"
MEASURES = ["within_45_deg", "path_within_45_deg", "class_balanced_accuracy", "episodes_stopped_right",
            "episodes_stopped_early", "episodes_never_stopped", "tuned_two_episodes_stopped_right",
            "tuned_two_episodes_stopped_early", "twist_sign_accuracy"]


def probe_results(experiment_name):
    """Return the variants of one probe experiment by name."""

    content = json.loads((PROBE_FOLDER / f"{experiment_name}.json").read_text(encoding = "utf-8"))
    return {result["name"]: result for result in content["results"]}


def fold_values(result, measure):
    """Return {fold name: value} of one measure."""

    return {fold["fold"]: fold.get(measure) for fold in result["folds"]}


def compare(label, baseline, variant, measures = MEASURES):
    """Print the paired per-fold differences of a variant against a baseline."""

    print(f"\n--- {label}: {variant['name']} minus {baseline['name']}")
    for measure in measures:
        base = fold_values(baseline, measure)
        other = fold_values(variant, measure)
        folds = [fold for fold in base if fold in other and base[fold] is not None and other[fold] is not None]
        if not folds:
            continue
        differences = [other[fold] - base[fold] for fold in folds]
        positive = sum(1 for value in differences if value > 0)
        print(f"{measure:36s} base mean {statistics.mean(base[f] for f in folds):.3f}  variant mean "
              f"{statistics.mean(other[f] for f in folds):.3f}  diff mean {statistics.mean(differences):+.3f}  "
              f"per fold [{', '.join(f'{value:+.3f}' for value in differences)}]  higher in {positive} of {len(folds)}")


def per_fold_table(label, results, measure):
    """Print one measure per fold for several variants."""

    print(f"\n=== {label}: {measure} per fold")
    folds = [fold["fold"] for fold in results[0]["folds"]]
    print("variant".ljust(42) + "  ".join(fold.replace("without_path_", "w")[:12].rjust(6) for fold in folds) + "   mean")
    for result in results:
        values = fold_values(result, measure)
        row = [values.get(fold) for fold in folds]
        print(result["name"][:40].ljust(42) + "  ".join(f"{value:6.3f}" if value is not None else "   n/a" for value in row)
              + f"  {statistics.mean(v for v in row if v is not None):.3f}")


def seed_spread():
    """Print the spread of the six-fold mean between seeds, and the seeds needed per setting."""

    seeds = probe_results("seeds")
    designs = probe_results("designs")
    balance = probe_results("balance")
    data_amount = probe_results("data_amount")
    groups = {
        "classes (default seed + 11, 12, 13)": [balance["square_root_weights"]] + [seeds[f"classes_seed_{s}"] for s in (11, 12, 13)],
        "exact (default seed + 11, 12, 13)": [designs["exact_step"]] + [seeds[f"exact_seed_{s}"] for s in (11, 12, 13)],
        "classes, all episodes, seeds 1, 2, 3": [data_amount[f"episode_share_1.0_seed_{s}"] for s in (1, 2, 3)],
    }
    print("\n=== Seed spread of the six-fold mean")
    spreads = {}
    for label, results in groups.items():
        for measure in ("within_45_deg", "path_within_45_deg", "episodes_stopped_right", "tuned_two_episodes_stopped_right"):
            means = [statistics.mean(fold_values(result, measure).values()) for result in results]
            spread = statistics.stdev(means)
            spreads[(label, measure)] = spread
            # per fold spread between seeds
            per_fold = {}
            for result in results:
                for fold, value in fold_values(result, measure).items():
                    per_fold.setdefault(fold, []).append(value)
            fold_spreads = {fold: statistics.stdev(values) for fold, values in per_fold.items()}
            print(f"{label:40s} {measure:34s} means {[round(m, 3) for m in means]}  range {min(means):.3f} to {max(means):.3f}  "
                  f"sd {spread:.4f}  per-fold sd {min(fold_spreads.values()):.3f} to {max(fold_spreads.values()):.3f}")
    return spreads


def seeds_needed(standard_deviation, difference, alpha = 0.05, power = 0.8):
    """Return the seeds per setting for a two-sided two-sample t-test (Welch with equal spreads), by search."""

    try:
        from scipy import stats
    except ImportError:
        z = 1.959964 + 0.841621
        return math.ceil(2 * (z * standard_deviation / difference) ** 2)
    for seeds in range(2, 200):
        degrees = 2 * seeds - 2
        noncentrality = difference / (standard_deviation * math.sqrt(2 / seeds))
        critical = stats.t.ppf(1 - alpha / 2, degrees)
        achieved = 1 - stats.nct.cdf(critical, degrees, noncentrality) + stats.nct.cdf(-critical, degrees, noncentrality)
        if achieved >= power:
            return seeds
    return None


def probability_right_pick(standard_deviation, difference, seeds):
    """Chance that the truly better setting shows the higher mean over the given seeds (normal approximation)."""

    from statistics import NormalDist
    return NormalDist().cdf(difference / (standard_deviation * math.sqrt(2 / seeds)))


def main():
    backbones = probe_results("backbones")
    balance = probe_results("balance")
    scores = probe_results("scores")
    cleaning = probe_results("cleaning")
    head_size = probe_results("head_size")
    designs = probe_results("designs")
    splits = probe_results("splits")
    stop = probe_results("stop")
    data_amount = probe_results("data_amount")

    default = balance["square_root_weights"]
    for name in ("no_class_weights", "full_inverse_weights", "square_root_weights_label_smoothing"):
        compare("class weights", default, balance[name])
    for name in ("scores_2_3", "score_3_only", "scores_weighted"):
        compare("scores", scores["scores_1_2_3"], scores[name])
    for name in ("mistakes_and_flags_dropped", "nothing_dropped", "mistakes_and_static_rows_dropped"):
        compare("cleaning", cleaning["mistakes_dropped_flags_kept"], cleaning[name])
    for name in ("linear", "hidden_1024"):
        compare("head size", head_size["hidden_256"], head_size[name])
    compare("design", designs["classes_small_motion_next_move"], designs["exact_step"])
    compare("small motion", designs["classes_small_motion_next_move"], designs["classes_small_motion_dropped"])
    compare("backbone untrained", backbones["resnet18_imagenet_grid"], backbones["resnet18_random_grid"])

    per_fold_table("class weights", [balance[n] for n in balance], "within_45_deg")
    per_fold_table("class weights", [balance[n] for n in balance], "class_balanced_accuracy")
    per_fold_table("class weights", [balance[n] for n in balance], "episodes_stopped_right")
    per_fold_table("scores", [scores[n] for n in scores], "within_45_deg")
    per_fold_table("scores", [scores[n] for n in scores], "episodes_stopped_right")
    per_fold_table("cleaning", [cleaning[n] for n in cleaning], "within_45_deg")
    per_fold_table("head size", [head_size[n] for n in head_size], "within_45_deg")

    spreads = seed_spread()
    print("\n=== Seeds per setting needed (two-sided 5 %, power 0.8), from the measured seed sd of the six-fold mean")
    for (label, measure), spread in spreads.items():
        if measure not in ("within_45_deg", "episodes_stopped_right"):
            continue
        for difference in (0.01, 0.02, 0.03, 0.05):
            print(f"{label:40s} {measure:24s} sd {spread:.4f} difference {difference:.2f}: seeds {seeds_needed(spread, difference)}  "
                  f"P(right pick with 1, 2, 3 seeds) {probability_right_pick(spread, difference, 1):.2f}, "
                  f"{probability_right_pick(spread, difference, 2):.2f}, {probability_right_pick(spread, difference, 3):.2f}")

    print("\n=== End-to-end runs, per fold")
    runs = {}
    for run_folder in sorted(RUNS_FOLDER.glob("2026-11-09_*")):
        metrics_path = run_folder / "metrics.json"
        if not metrics_path.exists():
            print(f"{run_folder.name}: no metrics.json yet")
            continue
        metrics = json.loads(metrics_path.read_text(encoding = "utf-8"))
        runs[metrics["name"]] = metrics
        print(f"\n{metrics['name']} settings: epochs {metrics['settings']['epochs']}, share per epoch {metrics['settings'].get('train_share_per_epoch')}, "
              f"lr {metrics['settings']['learning_rate']}, backbone factor {metrics['settings']['backbone_learning_rate_factor']}, "
              f"weight decay {metrics['settings']['weight_decay']}, batch {metrics['settings']['batch_size']}, "
              f"last checkpoint kept {metrics['settings'].get('is_last_checkpoint_kept')}, augmentation {metrics['settings']['augmentation']}")
        for fold in metrics["folds"]:
            print("   " + fold["fold"] + "  " + "  ".join(f"{measure}={fold.get(measure)}" for measure in MEASURES if fold.get(measure) is not None)
                  + f"  epochs_run={fold.get('epochs_run')} minutes={fold.get('minutes')}")
    if "resnet18_finetune_exact" in runs and "resnet18_finetune_classes" in runs:
        exact = {fold["fold"]: fold for fold in runs["resnet18_finetune_exact"]["folds"]}
        classes = {fold["fold"]: fold for fold in runs["resnet18_finetune_classes"]["folds"]}
        print("\nexact minus classes, end to end, per fold:")
        for measure in MEASURES:
            differences = [exact[f][measure] - classes[f][measure] for f in exact if f in classes and exact[f].get(measure) is not None and classes[f].get(measure) is not None]
            if differences:
                print(f"   {measure:36s} {[round(d, 3) for d in differences]}  mean {statistics.mean(differences):+.3f}")

    print("\n=== Validation loss history (held-out episodes of the training sheets), end-to-end")
    for run_folder in sorted(RUNS_FOLDER.glob("2026-11-09_*")):
        for fold_path in sorted(run_folder.glob("fold_*.json")):
            history = json.loads(fold_path.read_text(encoding = "utf-8"))["history"]
            losses = [epoch["validation_loss"] for epoch in history]
            best_epoch = 1 + losses.index(min(losses))
            print(f"{run_folder.name:45s} {fold_path.stem:24s} epochs {len(history)}  best validation epoch {best_epoch}  "
                  f"last minus best {losses[-1] - min(losses):+.4f}  train/validation last {history[-1]['train_loss']}/{losses[-1]}  "
                  f"seconds per epoch {statistics.median(epoch['seconds'] for epoch in history):.0f}")


if __name__ == "__main__":
    main()
