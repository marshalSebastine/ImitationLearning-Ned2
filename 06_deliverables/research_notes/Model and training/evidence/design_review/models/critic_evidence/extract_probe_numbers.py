"""Read-only extraction of per-fold numbers from the frozen probe JSON files, for the evidence critique.

Nothing under D:/AI_Project is written. Output goes to standard output only.
"""

import json
import pathlib

PROBE_FOLDER = pathlib.Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes")


def load_json(file_name):
    with open(PROBE_FOLDER / file_name, encoding = "utf-8") as file_handle:
        return json.load(file_handle)


def fold_value(fold, key):
    value = fold.get(key)
    return "-" if value is None else value


def print_results(file_name, keys):
    content = load_json(file_name)
    print("=" * 30, file_name, "top-level keys:", list(content.keys()))
    results = content.get("results", [])
    for result in results:
        name = result.get("name")
        settings = result.get("settings", {})
        folds = result.get("folds", [])
        line_parts = []
        for key in keys:
            values = [fold_value(fold, key) for fold in folds]
            line_parts.append(key + "=" + str(values))
        extra = {
            "seconds": result.get("seconds"),
            "backbone": settings.get("backbone"),
            "feature_kind": settings.get("feature_kind"),
            "design": settings.get("design"),
            "seed": settings.get("seed"),
            "history_lags": settings.get("history_lags"),
            "is_difference_history": settings.get("is_difference_history"),
            "is_stop_separate": settings.get("is_stop_separate"),
            "stop_target": settings.get("stop_target"),
        }
        fold_names = [fold.get("fold") for fold in folds]
        print("--", name, extra)
        print("   folds:", fold_names)
        for part in line_parts:
            print("  ", part)


def main():
    keys = ["direction_rows", "within_45_deg", "path_within_45_deg", "class_balanced_accuracy",
            "episodes_stopped_right", "tuned_two_episodes_stopped_right", "twist_sign_accuracy"]
    for file_name in ["backbones.json", "seeds.json", "designs.json", "history.json"]:
        print_results(file_name, keys)
    baselines = load_json("baselines.json")
    print("=" * 30, "baselines.json")
    for split_name, split_content in baselines.items():
        for rule_name, rule_content in split_content.items():
            folds = rule_content.get("folds", [])
            print("--", split_name, rule_name)
            for key in ["direction_rows", "within_45_deg", "path_direction_rows", "path_within_45_deg"]:
                print("  ", key, [fold.get(key) for fold in folds])
            print("   folds:", [fold.get("fold") for fold in folds])


main()
