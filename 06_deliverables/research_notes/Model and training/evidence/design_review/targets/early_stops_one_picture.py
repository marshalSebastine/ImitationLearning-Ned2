"""
Where the end-to-end exact model stops early with one picture over the threshold chosen on the other held-out sheets
(thresholds from stop_analysis.out.txt: 0.4, 0.4 and 0.6 for sheets 03, 05 and 06). Read only; writes
early_stops_one_picture.out.txt.
"""

import numpy
import pandas

from common_frames import AUDIT_FOLDER, OUTPUT_FOLDER, RUNS_FOLDER, read_frames
from stop_analysis import episode_outcomes

THRESHOLDS = {"without_path_03": 0.4, "without_path_05": 0.4, "without_path_06": 0.6}


def main() -> None:
    frames = read_frames()
    view = pandas.read_csv(AUDIT_FOLDER / "view_and_tracks" / "s09_view_rows.csv")
    frames = frames.merge(view[["frame_index", "line_share_top_half"]], on = "frame_index", how = "left")
    frames["no_line_ahead"] = frames["line_share_top_half"].fillna(1.0) <= 0.002
    frames["goal_distance_cm"] = frames["goal_distance_cm"].fillna(0.0)
    lookup = frames[["frame_index", "episode_id", "sheet", "row_in_episode", "goal_distance_cm", "step_fraction", "no_line_ahead", "action"]].set_index("frame_index")
    early_parts = []
    lines = []
    for fold, threshold in THRESHOLDS.items():
        data = numpy.load(RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact" / f"predictions_{fold}.npz")
        table = lookup.loc[data["frame_index"]].reset_index()
        table["stop_probability"] = data["stop_probability"]
        table = table.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
        outcomes = episode_outcomes(table, threshold, 1)
        early = outcomes[outcomes["outcome"] == "early"]
        early_parts.append(early)
        lines.append(f"{fold}: threshold {threshold}, episodes {len(outcomes)}, early {len(early)}, of these no line ahead {int(early['no_line_ahead'].sum())}, "
                     f"median distance {early['distance'].median():.1f} cm")
    early = pandas.concat(early_parts)
    lines.append(f"all: early {len(early)}, no line ahead {int(early['no_line_ahead'].sum())} ({early['no_line_ahead'].mean():.2f}), "
                 f"median distance to the goal {early['distance'].median():.1f} cm (p25 {early['distance'].quantile(0.25):.1f}, p75 {early['distance'].quantile(0.75):.1f})")
    print("\n".join(lines))
    (OUTPUT_FOLDER / "early_stops_one_picture.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
