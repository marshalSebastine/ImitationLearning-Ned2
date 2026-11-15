"""
Print the stop outcomes of the frozen-probe stop variants (stop.json, designs.json), fold by fold (read only).
Run with python -B.
"""

import json
from pathlib import Path

PROBE_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes")
FIELDS = ["episodes_stopped_right", "episodes_stopped_early", "episodes_never_stopped",
          "tuned_two_episodes_stopped_right", "tuned_two_episodes_stopped_early", "tuned_two_episodes_never_stopped",
          "tuned_two_stop_threshold", "within_45_deg", "test_episodes"]


def main():
    for file_name in ("stop.json", "designs.json"):
        content = json.loads((PROBE_FOLDER / file_name).read_text(encoding = "utf-8"))
        print(f"\n######## {file_name}: top-level type {type(content).__name__}")
        for variant in content["results"]:
            name = variant.get("name")
            folds = variant.get("folds", [])
            print(f"\n== {name} (stop_target {variant.get('settings', {}).get('stop_target')}, design {variant.get('settings', {}).get('design')})")
            totals = {}
            for fold in folds:
                values = " ".join(f"{field.replace('episodes_', '').replace('stopped_', '')}={fold.get(field)}" for field in FIELDS if field in fold)
                print(f"   {fold.get('fold')}: {values}")
                episodes = fold.get("test_episodes", 0)
                for field in FIELDS[:6]:
                    if field in fold:
                        totals[field] = totals.get(field, 0.0) + fold[field] * episodes
                totals["episodes"] = totals.get("episodes", 0) + episodes
            print("   pooled episode counts: " + " ".join(f"{field.replace('episodes_', '').replace('stopped_', '')}={round(value)}" for field, value in totals.items()))


if __name__ == "__main__":
    main()
