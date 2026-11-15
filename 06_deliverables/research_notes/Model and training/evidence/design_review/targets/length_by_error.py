"""
Is the short predicted step a sign of uncertainty? Predicted step length of the exact-step models per held-out sheet and
by the angle error of the prediction, on moving rows (read only). Writes length_by_error.out.txt.
"""

import numpy
import pandas

from common_frames import DIRECTION_NAMES, OUTPUT_FOLDER, read_frames

lines = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush = True)


def main() -> None:
    frames = read_frames()
    keep = ["frame_index", "action", "tip_forward_m", "tip_left_m", "move_angle_deg", "forearm_change_deg"]
    sources = [("exact run", fold, f"exact_outputs_{fold}.csv") for fold in ("without_path_03", "without_path_05", "without_path_06")]
    sources += [("final candidate", "without_path_01", "final_outputs_without_path_01.csv")]
    pooled = []
    for name, fold, file_name in sources:
        table = pandas.read_csv(OUTPUT_FOLDER / file_name).merge(frames[keep], on = "frame_index", how = "left")
        table = table[table["action"].isin(DIRECTION_NAMES)].copy()
        table["predicted_cm"] = numpy.hypot(table["forward_cm"], table["left_cm"])
        table["demonstrated_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"], table["tip_left_m"])
        angle = numpy.degrees(numpy.arctan2(table["left_cm"], table["forward_cm"]))
        table["angle_error_deg"] = numpy.abs((angle - table["move_angle_deg"] + 180.0) % 360.0 - 180.0)
        table["source"] = name
        say(f"{name} {fold}: moving rows {len(table)}, median predicted {table['predicted_cm'].median():.2f} cm, demonstrated "
            f"{table['demonstrated_cm'].median():.2f} cm, median angle error {table['angle_error_deg'].median():.1f} deg, "
            f"median |twist| predicted {table['twist_deg'].abs().median():.2f} against {table['forearm_change_deg'].abs().median():.2f}")
        if name == "exact run":
            pooled.append(table)
    pooled = pandas.concat(pooled)
    say("exact run, sheets 03, 05, 06 pooled, moving rows by angle error: n, median predicted cm, share of the demonstrated median")
    demonstrated = pooled["demonstrated_cm"].median()
    for low, high in ((0, 15), (15, 30), (30, 45), (45, 90), (90, 181)):
        part = pooled[(pooled["angle_error_deg"] >= low) & (pooled["angle_error_deg"] < high)]
        say(f"  {low:>3} to {high:>3} deg: {len(part):>5}, {part['predicted_cm'].median():.2f}, {part['predicted_cm'].median() / demonstrated:.2f}")
    (OUTPUT_FOLDER / "length_by_error.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
