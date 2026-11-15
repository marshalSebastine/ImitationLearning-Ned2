"""
Per-fold tables of the data-amount experiment (frozen ResNet-18 grid probes, leave one sheet out).

Reads, never writes, D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes/
data_amount.json and prints, for every setting, the per-fold within-45-degree share of the direction rows, the
path within-45 share, the stop measures, and the mean and spread over the three seeds. Also fits how the fold mean
grows with the number of training sheets and with the share of episodes.

Usage: python data_amount_table.py
"""

import json
import math
import statistics

PROBE_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes/"

PARAMETERS = {
    "file_name": "data_amount.json",
    "measures": ["within_45_deg", "path_within_45_deg", "episodes_stopped_right", "episodes_never_stopped",
                 "tuned_two_episodes_stopped_right", "tuned_two_episodes_never_stopped"],
}


def load_results(parameters = None):
    """
    Load the list of results of one probe file.

    Parameters
    ----------
    parameters : dict, optional
        Merged over PARAMETERS.

    Returns
    -------
    list of dict
        One entry per named setting.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    with open(PROBE_FOLDER + settings["file_name"], "r", encoding = "utf-8") as handle:
        content = json.load(handle)
    return content["results"]


def print_fold_keys(results):
    """
    Print the keys of the first fold so the available fields are known.
    """
    first_fold = results[0]["folds"][0]
    print("fold keys:", sorted(first_fold.keys()))
    extra_keys = [key for key in results[0].keys() if key not in ("folds", "settings", "summary", "class_names")]
    print("result keys besides folds, settings, summary:", extra_keys)


def setting_label(result):
    """
    Return a short label (kind, level, seed) for a result.
    """
    settings = result["settings"]
    if settings.get("train_sheet_count") is not None:
        return ("sheets", settings["train_sheet_count"], settings["seed"])
    return ("episode_share", settings["train_episode_share"], settings["seed"])


def main():
    """
    Print the tables.
    """
    results = load_results()
    print_fold_keys(results)
    by_level = {}
    for result in results:
        kind, level, seed = setting_label(result)
        by_level.setdefault((kind, level), []).append((seed, result))
    fold_names = [fold["fold"] for fold in results[0]["folds"]]
    for measure in PARAMETERS["measures"]:
        print()
        print(f"=== {measure} (per fold, mean of three seeds; seed values in brackets) ===")
        print("setting".ljust(22) + "".join(name.replace("without_path_", "out ").ljust(26) for name in fold_names)
              + "fold mean (seed range)")
        for (kind, level), seed_results in sorted(by_level.items()):
            row = f"{kind} {level}".ljust(22)
            seed_means = []
            for fold_index, fold_name in enumerate(fold_names):
                values = []
                for seed, result in sorted(seed_results):
                    fold = result["folds"][fold_index]
                    assert fold["fold"] == fold_name
                    values.append(fold.get(measure))
                clean = [value for value in values if value is not None]
                if clean:
                    text = f"{statistics.mean(clean):.3f} (" + ", ".join(f"{value:.3f}" for value in clean) + ")"
                else:
                    text = "n/a"
                row += text.ljust(26)
            for seed, result in sorted(seed_results):
                values = [fold.get(measure) for fold in result["folds"] if fold.get(measure) is not None]
                if values:
                    seed_means.append(statistics.mean(values))
            if seed_means:
                row += f"{statistics.mean(seed_means):.3f} ({min(seed_means):.3f} to {max(seed_means):.3f})"
            print(row)
    print()
    print("=== test rows and episodes per fold (from the 100 percent setting) ===")
    for fold in results[0]["folds"]:
        print(fold["fold"], "test_rows", fold["test_rows"], "test_episodes", fold["test_episodes"],
              "direction_rows", fold["direction_rows"])
    print()
    print("=== growth per doubling (fold mean of within_45_deg, mean of seeds) ===")
    sheet_levels = sorted(level for kind, level in by_level if kind == "sheets")
    share_levels = sorted(level for kind, level in by_level if kind == "episode_share")
    level_means = {}
    for key, seed_results in by_level.items():
        seed_means = [statistics.mean(fold["within_45_deg"] for fold in result["folds"]) for seed, result in
                      seed_results]
        level_means[key] = statistics.mean(seed_means)
    level_means[("sheets", 5)] = level_means[("episode_share", 1.0)]
    for level in sheet_levels + [5]:
        print("sheets", level, f"{level_means[('sheets', level)]:.4f}")
    for level in share_levels:
        print("episode share", level, f"{level_means[('episode_share', level)]:.4f}")
    # least-squares slope of the mean against log2 of the sheet count and of the episode share
    for kind, levels in (("sheets", sheet_levels + [5]), ("episode_share", share_levels)):
        x_values = [math.log2(level) for level in levels]
        y_values = [level_means[(kind, level)] for level in levels]
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / sum(
            (x - x_mean) ** 2 for x in x_values)
        print(f"{kind}: slope {slope:.4f} per doubling (least squares over levels {levels})")


if __name__ == "__main__":
    main()
