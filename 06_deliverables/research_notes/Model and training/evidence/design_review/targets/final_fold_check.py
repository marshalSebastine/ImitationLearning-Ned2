"""
The same output checks as exact_analysis.py for the folds of the final candidate that exist (audit-cleaned training
rows), so that the length and twist shrinkage can be compared with the earlier exact run. Writes final_fold_check.out.txt.
"""

import sys

import numpy
import pandas

from common_frames import DIRECTION_NAMES, OUTPUT_FOLDER, read_frames

lines = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush = True)


def main(folds) -> None:
    frames = read_frames()
    frames["goal_distance_cm"] = frames["goal_distance_cm"].fillna(0.0)
    keep = ["frame_index", "episode_id", "row_in_episode", "goal_distance_cm", "action", "tip_forward_m", "tip_left_m", "forearm_change_deg"]
    for fold in folds:
        table = pandas.read_csv(OUTPUT_FOLDER / f"final_outputs_{fold}.csv").merge(frames[keep], on = "frame_index", how = "left")
        table["predicted_cm"] = numpy.hypot(table["forward_cm"], table["left_cm"])
        table["demonstrated_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"].fillna(0.0), table["tip_left_m"].fillna(0.0))
        moving = table["action"].isin(DIRECTION_NAMES)
        stop = table["action"] == "stop"
        twisted = (table["forearm_change_deg"].abs() >= 2.0) & ~stop
        away = (table["goal_distance_cm"] > 3.0) & ~stop
        still = away & (table["predicted_cm"] < 0.3) & (table["twist_deg"].abs() < 1.0)
        say(f"{fold}: pictures {len(table)}, episodes {table['episode_id'].nunique()}")
        say(f"  moving rows {int(moving.sum())}: median predicted {table.loc[moving, 'predicted_cm'].median():.3f} cm against demonstrated "
            f"{table.loc[moving, 'demonstrated_cm'].median():.3f} cm; stop rows {int(stop.sum())}: median predicted {table.loc[stop, 'predicted_cm'].median():.3f} cm")
        say(f"  twisted rows {int(twisted.sum())}: median |predicted| {table.loc[twisted, 'twist_deg'].abs().median():.2f} against "
            f"{table.loc[twisted, 'forearm_change_deg'].abs().median():.2f} deg; sign right {(numpy.sign(table.loc[twisted, 'twist_deg']) == numpy.sign(table.loc[twisted, 'forearm_change_deg'])).mean():.3f}")
        say(f"  standing still away from the goal: {int(still.sum())} of {int(away.sum())} pictures in {table.loc[still, 'episode_id'].nunique()} episodes; "
            f"predicted step under 0.3 cm: {int((away & (table['predicted_cm'] < 0.3)).sum())}")
    (OUTPUT_FOLDER / "final_fold_check.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["without_path_01"])
