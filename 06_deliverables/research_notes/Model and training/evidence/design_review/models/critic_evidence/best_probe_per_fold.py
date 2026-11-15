"""Read-only: the best frozen probe per held-out sheet over every probe file, and per-sheet seed ranges.

Nothing under D:/AI_Project is written. Output goes to standard output only.
"""

import json
import pathlib

PROBE_FOLDER = pathlib.Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes")


def main():
    best = {}
    for file_path in sorted(PROBE_FOLDER.glob("*.json")):
        if file_path.name == "baselines.json":
            continue
        with open(file_path, encoding = "utf-8") as file_handle:
            content = json.load(file_handle)
        for result in content.get("results", []):
            for fold in result.get("folds", []):
                fold_name = fold.get("fold", "")
                if not fold_name.startswith("without_path_"):
                    continue
                for key in ["within_45_deg", "path_within_45_deg"]:
                    value = fold.get(key)
                    if value is None:
                        continue
                    entry = best.setdefault((fold_name, key), [])
                    entry.append((value, file_path.name, result.get("name"), fold.get("direction_rows"),
                                  result.get("settings", {}).get("is_audit_cleaning")))
    for (fold_name, key), entries in sorted(best.items()):
        entries.sort(reverse = True)
        print(fold_name, key, "top 4:", entries[:4])
    with open(PROBE_FOLDER / "seeds.json", encoding = "utf-8") as file_handle:
        seeds = json.load(file_handle)
    with open(PROBE_FOLDER / "designs.json", encoding = "utf-8") as file_handle:
        designs = json.load(file_handle)
    with open(PROBE_FOLDER / "backbones.json", encoding = "utf-8") as file_handle:
        backbones = json.load(file_handle)
    groups = {
        "classes": [r for r in backbones["results"] if r["name"] == "resnet18_imagenet_grid"]
                   + [r for r in seeds["results"] if r["name"].startswith("classes_seed_")],
        "exact": [r for r in designs["results"] if r["name"] == "exact_step"]
                 + [r for r in seeds["results"] if r["name"].startswith("exact_seed_")],
    }
    for group_name, results in groups.items():
        for key in ["within_45_deg", "path_within_45_deg", "tuned_two_episodes_stopped_right"]:
            fold_count = len(results[0]["folds"])
            ranges = []
            for index in range(fold_count):
                values = [r["folds"][index].get(key) for r in results]
                values = [v for v in values if v is not None]
                if values:
                    ranges.append((results[0]["folds"][index]["fold"][-2:], round(min(values), 4), round(max(values), 4),
                                   round(100 * (max(values) - min(values)), 1)))
            means = []
            for r in results:
                values = [f.get(key) for f in r["folds"]]
                if all(v is not None for v in values):
                    means.append(round(sum(values) / len(values), 4))
            print(group_name, key, "per-sheet (sheet, min, max, range points):", ranges, "means:", means)


main()
