"""
Verifier's check of a risk the analyst did not test: would the stall rules end a run that moves at the DEMONSTRATED pace?

The analyst tried the rules on the ten robot runs, all driven with the fixed 1 cm step. The rule is meant for the next
session, where the step has the length the model asks for, which follows the demonstrations' pace. So each recorded
demonstration is treated here as if it were a run: its tip positions and forearm angles, picture by picture, are fed
to the rules. A rule that fires on many good demonstrations would also end good runs that imitate them.

Read-only (frames.csv). Prints to the screen and to v08_stall_rule_on_demonstrations.out.txt.
"""

from pathlib import Path
from typing import List, Optional

import numpy
import pandas

STALL_PARAMETERS = {
    "frames_path": Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv"),
    "rules": {"driver's rule: 30 pictures, diagonal 3 cm, forearm 15 deg": [30, 3.0, 15.0, "diagonal"],
              "analyst's rule 1: 15 pictures, diagonal 4 cm, forearm 15 deg": [15, 4.0, 15.0, "diagonal"],
              "analyst's rule 2: 12 pictures, net travel 1.5 cm, forearm 15 deg": [12, 1.5, 15.0, "net"]}
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


def first_firing(tips_cm: numpy.ndarray, forearm_deg: numpy.ndarray, rule: List) -> Optional[int]:
    """Return the index of the first picture at which the rule fires, or None."""

    count, extent_cm, twist_deg, kind = rule
    for last in range(count - 1, len(tips_cm)):
        window = tips_cm[last - count + 1:last + 1]
        if kind == "diagonal":
            extent = float(numpy.hypot(*(window.max(axis = 0) - window.min(axis = 0))))
        else:
            extent = float(numpy.linalg.norm(window[-1] - window[0]))
        if extent < extent_cm and float(numpy.ptp(forearm_deg[last - count + 1:last + 1])) < twist_deg:
            return last
    return None


if __name__ == "__main__":
    settings = dict(STALL_PARAMETERS)
    frames = pandas.read_csv(settings["frames_path"], low_memory = False).sort_values(["episode_id", "row_in_episode"])
    say(f"episodes {frames['episode_id'].nunique()}, rows {len(frames)}; every recorded row is used (also rows the training drops)")
    for label, rule in settings["rules"].items():
        fired = []
        for episode_id, episode in frames.groupby("episode_id", sort = False):
            tips = 100.0 * episode[["tip_x_m", "tip_y_m"]].to_numpy()
            forearm = numpy.degrees(episode["joint_4"].to_numpy())
            index = first_firing(tips, forearm, rule)
            if index is not None:
                row = episode.iloc[index]
                fired.append({"episode_id": episode_id, "sheet": row["sheet"], "row": int(row["row_in_episode"]), "rows": len(episode),
                              "steps_to_end": int(len(episode) - 1 - index), "goal_distance_cm": row["goal_distance_cm"],
                              "hand_corrections": row["hand_corrections"]})
        fired = pandas.DataFrame(fired)
        say("")
        say(f"== {label} ==")
        say(f"fires in {len(fired)} of {frames['episode_id'].nunique()} demonstrations")
        if len(fired):
            before_end = fired[fired["steps_to_end"] > 5]
            say(f"   of these, more than 5 pictures before the demonstration's end: {len(before_end)}; median pictures still to go at the firing "
                f"{fired['steps_to_end'].median():.0f}; median distance to the goal at the firing {fired['goal_distance_cm'].median():.1f} cm")
            say("   per sheet: " + ", ".join(f"{sheet[-2:]}: {count}" for sheet, count in fired.groupby("sheet").size().items()))
            say(f"   firing more than 5 cm from the goal: {int((fired['goal_distance_cm'] > 5.0).sum())}; of the firing episodes, with a hand correction: "
                f"{int((fired['hand_corrections'].fillna(0) > 0).sum())}")
    (Path(__file__).resolve().parent / "v08_stall_rule_on_demonstrations.out.txt").write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
