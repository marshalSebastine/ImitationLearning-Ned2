"""
Offline proxy for correction rounds: does the recovery-like data already in the demonstrations (the replays of the
239 hand corrections, 37.9 percent of the rows) help where the model is weakest, with the puck off the line?

Frozen ResNet-18 grid features, exact-step head, leave one sheet out (the project's own probe code,
02_code/training_probes.run_variant, read only). Three training-row rules, three seeds each:
    all                     every training row (the usual probe)
    no_corrections          without the correction-replay rows (rows after a hand correction started)
    no_matched_first_pass   without the first-pass rows of randomly chosen training episodes, as many rows as the
                            correction replays of that fold (controls for having less data)
Predictions and results are written to this script's folder only (output_folder overrides the probe folder).
The row rule is applied by replacing training_probes.subsampled_training_rows in memory (the file is not edited);
"train_episode_share" 0.99 only makes run_variant call it, the replacement keeps every episode.

Usage: python recovery_ablation.py
"""

import json
import sys
import time
from pathlib import Path

CODE_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/02_code"
sys.path.insert(0, CODE_FOLDER)

import numpy
import pandas

import training_probes

PARAMETERS = {
    "output_folder": Path(__file__).resolve().parent / "recovery_ablation",
    "seeds": [1, 2, 3],
    "row_rules": ["all", "no_corrections", "no_matched_first_pass"],
    "design": "exact",
}

CORRECTION_CACHE = {}


def correction_mask(frames):
    """
    True on every row that belongs to the replay of a hand correction (from the first row with waypoint 0 after row 0
    of its episode onwards), as the audit marks correction starts.
    """
    key = id(frames)
    if key not in CORRECTION_CACHE:
        is_start = ((frames["waypoint"].fillna(-1) == 0) & (frames["row_in_episode"] > 0)).astype(int)
        started = is_start.groupby(frames["episode_id"]).cumsum() > 0
        CORRECTION_CACHE[key] = started.to_numpy()
    return CORRECTION_CACHE[key]


def rule_rows(frames, train_rows, parameters, seed):
    """
    Replacement for training_probes.subsampled_training_rows: apply the row rule of this variant.
    """
    rule = parameters.get("row_rule", "all")
    in_correction = correction_mask(frames)
    if rule == "all":
        return train_rows
    if rule == "no_corrections":
        return train_rows[~in_correction[train_rows]]
    if rule == "no_matched_first_pass":
        correction_count = int(in_correction[train_rows].sum())
        randomness_generator = numpy.random.default_rng(seed)
        episodes = frames["episode_id"].to_numpy()
        training_episodes = numpy.unique(episodes[train_rows])
        randomness_generator.shuffle(training_episodes)
        is_dropped = numpy.zeros(len(train_rows), dtype = bool)
        dropped = 0
        for episode in training_episodes:
            if dropped >= correction_count:
                break
            first_pass = (episodes[train_rows] == episode) & ~in_correction[train_rows]
            is_dropped |= first_pass
            dropped += int(first_pass.sum())
        return train_rows[~is_dropped]
    raise ValueError(f"unknown row rule {rule}")


def main(parameters = None):
    """
    Run the nine variants and save their results.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    settings["output_folder"].mkdir(parents = True, exist_ok = True)
    training_probes.subsampled_training_rows = rule_rows
    frames_cache = {}
    features_cache = {}
    results = []
    for seed in settings["seeds"]:
        for rule in settings["row_rules"]:
            variant = {"name": f"recovery_{rule}_seed_{seed}", "design": settings["design"], "split": "sheets",
                       "seed": seed, "row_rule": rule, "train_episode_share": 0.99,
                       "output_folder": str(settings["output_folder"]), "is_prediction_saved": True}
            started = time.perf_counter()
            result = training_probes.run_variant(variant, frames_cache, features_cache)
            results.append(result)
            summary = result["summary"]
            print(f"{variant['name']}: {time.perf_counter() - started:.0f} s, within 45 {summary.get('within_45_deg_mean')}, "
                  f"path within 45 {summary.get('path_within_45_deg_mean')}, stopped right {summary.get('episodes_stopped_right_mean')}",
                  flush = True)
            (settings["output_folder"] / "recovery_ablation.json").write_text(json.dumps({"results": results}, indent = 1, default = str),
                                                                              encoding = "utf-8")
    frames = next(iter(frames_cache.values()))
    in_correction = correction_mask(frames)
    print("rows:", len(frames), "correction-replay rows:", int(in_correction.sum()))


if __name__ == "__main__":
    main()
