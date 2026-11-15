"""
Re-score saved probe predictions on the same test rows (read only; prints to standard output).

The score and flag variants of the probes filtered the held-out sheet's rows as well as the training rows, so their
summary numbers were computed on fewer, easier test rows than the default. Here every variant and the default are
scored on the rows both of them predicted, with the worktree's own measures (training_metrics), so the comparison is
like for like. Run with python -B so that no bytecode is written next to the project's code.
"""

import sys

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

from pathlib import Path

import numpy

from config import TRAINING_TARGET_PARAMETERS
from training_metrics import direction_metrics, stop_metrics
from training_targets import DIRECTION_NAMES, load_training_frames

PREDICTION_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes/predictions")
FOLDS = [f"without_path_0{number}" for number in range(1, 7)]
DEFAULT_VARIANT = "scores_1_2_3"
VARIANTS = ["score_3_only", "scores_2_3", "scores_weighted", "mistakes_and_flags_dropped", "mistakes_and_static_rows_dropped",
            "nothing_dropped", "mistakes_dropped_flags_kept", "classes_held_out_sheets"]


def scored(frames, position_of_frame, predictions, frame_indices):
    """Return the direction and stop measures of one set of predictions on the given frames."""

    order = numpy.sort(numpy.array([position_of_frame[index] for index in frame_indices]))
    rows = frames.iloc[order]
    lookup = {index: position for position, index in enumerate(predictions["frame_index"])}
    picked = numpy.array([lookup[index] for index in rows["frame_index"].to_numpy()])
    is_moving = rows["action"].isin(DIRECTION_NAMES).to_numpy()
    measures = direction_metrics(rows["move_angle_deg"].to_numpy(), predictions["angle_deg"][picked], is_moving)
    stop = stop_metrics((rows["action"] == "stop").to_numpy().astype(float), predictions["stop_probability"][picked],
                        rows["goal_distance_cm"].fillna(0.0).to_numpy(), rows["episode_id"].to_numpy())
    measures.update(stop)
    # per-episode outcome at the fixed threshold, for paired counts
    episode_ids = rows["episode_id"].to_numpy()
    is_predicted = predictions["stop_probability"][picked] >= 0.5
    goal = rows["goal_distance_cm"].fillna(0.0).to_numpy()
    outcome = {}
    for episode_id in numpy.unique(episode_ids):
        positions = numpy.flatnonzero(episode_ids == episode_id)
        passing = positions[is_predicted[positions]]
        outcome[episode_id] = "never" if len(passing) == 0 else ("right" if goal[passing[0]] <= 2.0 else ("early" if goal[passing[0]] > 3.0 else "near"))
    measures["outcome"] = outcome
    return measures


def main():
    frames = load_training_frames(dict(TRAINING_TARGET_PARAMETERS))
    position_of_frame = {index: position for position, index in enumerate(frames["frame_index"].to_numpy())}
    for variant in VARIANTS:
        print(f"\n=== {variant} against {DEFAULT_VARIANT}, scored on the rows both predicted")
        totals = {"within": [], "path": [], "right_variant": [], "right_default": [], "never_variant": [], "never_default": []}
        paired_right = [0, 0]
        for fold in FOLDS:
            default_path = PREDICTION_FOLDER / f"{DEFAULT_VARIANT}_{fold}.npz"
            variant_path = PREDICTION_FOLDER / f"{variant}_{fold}.npz"
            if not variant_path.exists():
                print(f"{fold}: no predictions")
                continue
            default = dict(numpy.load(default_path))
            other = dict(numpy.load(variant_path))
            common = numpy.intersect1d(default["frame_index"], other["frame_index"])
            common = numpy.array([index for index in common if index in position_of_frame])
            base = scored(frames, position_of_frame, default, common)
            alternative = scored(frames, position_of_frame, other, common)
            identical = (len(default["frame_index"]) == len(other["frame_index"])
                         and numpy.array_equal(default["frame_index"], other["frame_index"])
                         and numpy.allclose(default["angle_deg"], other["angle_deg"]))
            only_variant_right = sum(1 for episode in base["outcome"] if alternative["outcome"][episode] == "right" and base["outcome"][episode] != "right")
            only_default_right = sum(1 for episode in base["outcome"] if base["outcome"][episode] == "right" and alternative["outcome"][episode] != "right")
            paired_right[0] += only_variant_right
            paired_right[1] += only_default_right
            print(f"{fold}: common rows {len(common)} (default {len(default['frame_index'])}, variant {len(other['frame_index'])}), "
                  f"episodes {len(base['outcome'])}, direction rows {base['direction_rows']}; within 45 default {base['within_45_deg']:.3f} "
                  f"variant {alternative['within_45_deg']:.3f} ({alternative['within_45_deg'] - base['within_45_deg']:+.3f}); "
                  f"stopped right default {base['episodes_stopped_right']:.3f} variant {alternative['episodes_stopped_right']:.3f}; "
                  f"never default {base['episodes_never_stopped']:.3f} variant {alternative['episodes_never_stopped']:.3f}; "
                  f"episodes right only with variant {only_variant_right}, only with default {only_default_right}"
                  + ("; predictions identical" if identical else ""))
            totals["within"].append(alternative["within_45_deg"] - base["within_45_deg"])
            totals["right_variant"].append(alternative["episodes_stopped_right"])
            totals["right_default"].append(base["episodes_stopped_right"])
            totals["never_variant"].append(alternative["episodes_never_stopped"])
            totals["never_default"].append(base["episodes_never_stopped"])
        if totals["within"]:
            print(f"mean within 45 difference {numpy.mean(totals['within']):+.4f} (higher in {sum(1 for d in totals['within'] if d > 0)} of {len(totals['within'])} folds); "
                  f"stopped right default {numpy.mean(totals['right_default']):.3f} variant {numpy.mean(totals['right_variant']):.3f}; "
                  f"never default {numpy.mean(totals['never_default']):.3f} variant {numpy.mean(totals['never_variant']):.3f}; "
                  f"discordant episodes: right only with variant {paired_right[0]}, only with default {paired_right[1]}")


if __name__ == "__main__":
    main()
