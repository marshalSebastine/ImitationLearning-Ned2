"""
The backup step-length rule on the end-to-end exact outputs (held-out sheets 03, 05, 06; read only): predicted length
times a gain, clamped to 0.5 to 1.5 cm. How often each clamp binds, and the executed median by situation.
Writes rescaled_length.out.txt.
"""

import numpy
import pandas

from common_frames import DIRECTION_NAMES, OUTPUT_FOLDER, read_frames


def main() -> None:
    frames = read_frames()[["frame_index", "action", "tip_forward_m", "tip_left_m", "forearm_change_deg", "goal_distance_cm"]]
    table = pandas.concat([pandas.read_csv(OUTPUT_FOLDER / f"exact_outputs_{fold}.csv") for fold in ("without_path_03", "without_path_05", "without_path_06")])
    table = table.merge(frames, on = "frame_index", how = "left")
    table = table[table["action"] != "stop"].copy()
    predicted = numpy.hypot(table["forward_cm"], table["left_cm"])
    moving = table["action"].isin(DIRECTION_NAMES)
    demonstrated = 100.0 * numpy.hypot(table["tip_forward_m"], table["tip_left_m"])
    gain = float(demonstrated[moving].median() / predicted[moving].median())
    executed = numpy.clip(predicted * gain, 0.5, 1.5)
    lines = [f"non-stop pictures {len(table)}; gain that matches the moving-row medians {gain:.2f}"]
    lines.append(f"executed with that gain and clamps 0.5 to 1.5 cm: median {numpy.median(executed):.2f} cm; at the lower clamp {(executed <= 0.5).mean():.3f}; "
                 f"at the upper clamp {(executed >= 1.5).mean():.3f}")
    twist_in_place = (table["action"] == "small_motion") & (table["forearm_change_deg"].abs() >= 3.0)
    near = table["goal_distance_cm"].fillna(99.0) <= 3.0
    for name, mask in (("moving rows", moving), ("twists in place", twist_in_place), ("within 3 cm of the goal", near)):
        lines.append(f"  {name}: n {int(mask.sum())}, executed median {numpy.median(executed[mask]):.2f} cm, demonstrated median {numpy.median(demonstrated[mask]):.2f} cm")
    print("\n".join(lines))
    (OUTPUT_FOLDER / "rescaled_length.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
