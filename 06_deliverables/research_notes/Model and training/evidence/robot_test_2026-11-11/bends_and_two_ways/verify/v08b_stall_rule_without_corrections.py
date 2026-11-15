"""
Verifier's follow-up to v08: the same stall rules on the demonstrations WITHOUT a hand correction only (after a hand
correction the tip is taken back, which a model run never does), and where on the line the net-travel rule fires.

Read-only. Prints to the screen and to v08b_stall_rule_without_corrections.out.txt.
"""

from pathlib import Path

import numpy
import pandas

from v08_stall_rule_on_demonstrations import STALL_PARAMETERS, first_firing

if __name__ == "__main__":
    frames = pandas.read_csv(STALL_PARAMETERS["frames_path"], low_memory = False).sort_values(["episode_id", "row_in_episode"])
    lines = []
    plain = frames[frames["hand_corrections"].fillna(0) == 0]
    lines.append(f"demonstrations without a hand correction: {plain['episode_id'].nunique()} of {frames['episode_id'].nunique()}")
    for label, rule in STALL_PARAMETERS["rules"].items():
        fired = []
        for episode_id, episode in plain.groupby("episode_id", sort = False):
            index = first_firing(100.0 * episode[["tip_x_m", "tip_y_m"]].to_numpy(), numpy.degrees(episode["joint_4"].to_numpy()), rule)
            if index is not None:
                row = episode.iloc[index]
                fired.append(f"{row['sheet'][-2:]} {row['start_kind']} row {int(row['row_in_episode'])} of {len(episode)}, {row['goal_distance_cm']:.1f} cm from the goal")
        lines.append(f"{label}: fires in {len(fired)}" + (": " + "; ".join(fired) if fired else ""))
    print("\n".join(lines))
    (Path(__file__).resolve().parent / "v08b_stall_rule_without_corrections.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")
