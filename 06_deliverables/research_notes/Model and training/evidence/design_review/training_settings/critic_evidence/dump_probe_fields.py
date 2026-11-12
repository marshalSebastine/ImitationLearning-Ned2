"""Print the per-fold fields of the probe and run result files that the training-settings section cites.

Read only: opens the JSON files of 04_runs and prints numbers.
"""

import json
import pathlib
import statistics
import sys

RUNS_FOLDER = pathlib.Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
PROBE_FOLDER = RUNS_FOLDER / "2026-11-09_frozen_feature_probes"

FIELDS = [
    "within_45_deg",
    "path_within_45_deg",
    "episodes_stopped_right",
    "episodes_stopped_early",
    "episodes_never_stopped",
    "tuned_two_episodes_stopped_right",
    "tuned_two_episodes_stopped_early",
    "tuned_two_episodes_never_stopped",
    "test_rows",
    "test_episodes",
    "direction_rows",
]


def print_probe_file(file_name):
    """Print every result of one probe file with its per-fold fields and the fold means."""
    path = PROBE_FOLDER / file_name
    content = json.loads(path.read_text(encoding = "utf-8"))
    print("=" * 100)
    print(file_name, "|", content.get("question"))
    for result in content["results"]:
        settings = result.get("settings", {})
        folds = result.get("folds", [])
        extra_keys = [key for key in result.keys() if key not in ("settings", "folds", "class_names")]
        print("-" * 100)
        print(
            result.get("name"),
            "| seed", settings.get("seed"),
            "| train_episode_share", settings.get("train_episode_share"),
            "| train_sheet_count", settings.get("train_sheet_count"),
            "| kept_scores", settings.get("kept_scores"),
            "| score_weights", settings.get("score_weights"),
            "| flagged_dropped", settings.get("is_flagged_dropped"),
            "| static_dropped", settings.get("is_static_dropped"),
            "| mistake_dropped", settings.get("is_mistake_dropped"),
            "| label_smoothing", settings.get("label_smoothing"),
            "| hidden", settings.get("hidden_size"),
            "| dropout", settings.get("dropout"),
            "| class_weight_power", settings.get("class_weight_power"),
            "| extra keys", extra_keys,
        )
        for key in extra_keys:
            value = result[key]
            if not isinstance(value, (list, dict)) or len(str(value)) < 400:
                print("   ", key, "=", value)
        for field in FIELDS:
            values = [fold.get(field) for fold in folds]
            if all(value is None for value in values):
                continue
            numeric_values = [value for value in values if isinstance(value, (int, float))]
            mean_text = ""
            if numeric_values and len(numeric_values) == len(values):
                mean_text = " mean %.4f" % statistics.mean(numeric_values)
            print("   %-34s %s%s" % (field, values, mean_text))
        fold_extra = set()
        for fold in folds:
            for key in fold.keys():
                if "train" in key or "second" in key or "epoch" in key or "fold" in key or "sheet" in key:
                    fold_extra.add(key)
        for key in sorted(fold_extra):
            print("   %-34s %s" % (key, [fold.get(key) for fold in folds]))


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print_probe_file(name)
