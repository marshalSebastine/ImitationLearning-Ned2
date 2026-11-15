"""
Stop checks on the final candidate's finished folds (read only): where its early stops happen at the thresholds it chose
on validation episodes, and its stop probability on far pictures with and without a line ahead.
Writes final_stop_check.out.txt.
"""

import json
import sys

import numpy
import pandas

from common_frames import AUDIT_FOLDER, OUTPUT_FOLDER, RUNS_FOLDER, read_frames
from stop_analysis import episode_outcomes, shares

RUN_FOLDER = RUNS_FOLDER / "2026-11-09_resnet18_final_candidate"


def main(folds) -> None:
    frames = read_frames()
    view = pandas.read_csv(AUDIT_FOLDER / "view_and_tracks" / "s09_view_rows.csv")
    frames = frames.merge(view[["frame_index", "line_share_top_half"]], on = "frame_index", how = "left")
    frames["no_line_ahead"] = frames["line_share_top_half"].fillna(1.0) <= 0.002
    frames["goal_distance_cm"] = frames["goal_distance_cm"].fillna(0.0)
    lookup = frames[["frame_index", "episode_id", "sheet", "row_in_episode", "goal_distance_cm", "step_fraction", "no_line_ahead", "action"]].set_index("frame_index")
    lines = []
    for fold in folds:
        measures = json.loads((RUN_FOLDER / f"fold_{fold}.json").read_text(encoding = "utf-8"))["measures"]
        data = numpy.load(RUN_FOLDER / f"predictions_{fold}.npz")
        table = lookup.loc[data["frame_index"]].reset_index()
        table["stop_probability"] = data["stop_probability"]
        table = table.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
        for consecutive, key in ((1, "tuned_stop_threshold"), (2, "tuned_two_stop_threshold")):
            threshold = measures[key]
            outcomes = episode_outcomes(table, threshold, consecutive)
            early = outcomes[outcomes["outcome"] == "early"]
            result = shares(outcomes)
            lines.append(f"{fold}, {consecutive} in a row at {threshold}: right {result['right']:.2f}, early {result['early']:.2f}, never {result['never']:.2f}; "
                         f"early stops {len(early)}, no line ahead {int(early['no_line_ahead'].sum()) if len(early) else 0}, median distance "
                         f"{early['distance'].median() if len(early) else float('nan'):.1f} cm")
        far = table[table["goal_distance_cm"] > 5.0]
        lines.append(f"{fold}: over 5 cm from the goal, p90 stop probability with no line ahead {far.loc[far['no_line_ahead'], 'stop_probability'].quantile(0.9):.3f} "
                     f"(n {int(far['no_line_ahead'].sum())}) against {far.loc[~far['no_line_ahead'], 'stop_probability'].quantile(0.9):.3f} (n {int((~far['no_line_ahead']).sum())}); "
                     f"stop pictures median {table.loc[table['action'] == 'stop', 'stop_probability'].median():.3f}")
    print("\n".join(lines))
    (OUTPUT_FOLDER / "final_stop_check.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["without_path_01"])
