"""
How the end-to-end exact model's predicted twist relates to the demonstrated twist on held-out sheets 03, 05 and 06
(read only): the least-squares gain, and the sign accuracy and demonstrated size by predicted size. Writes
twist_calibration.out.txt.
"""

import numpy
import pandas

from common_frames import OUTPUT_FOLDER, read_frames


def main() -> None:
    frames = read_frames()[["frame_index", "action", "forearm_change_deg"]]
    table = pandas.concat([pandas.read_csv(OUTPUT_FOLDER / f"exact_outputs_{fold}.csv") for fold in ("without_path_03", "without_path_05", "without_path_06")])
    table = table.merge(frames, on = "frame_index", how = "left")
    table = table[table["action"] != "stop"]
    predicted = table["twist_deg"].to_numpy()
    demonstrated = table["forearm_change_deg"].fillna(0.0).clip(-15, 15).to_numpy()
    gain = float((predicted * demonstrated).sum() / (predicted * predicted).sum())
    lines = [f"non-stop pictures {len(table)}; least-squares gain demonstrated on predicted (through zero) {gain:.2f}; "
             f"median |demonstrated| / median |predicted| {numpy.median(numpy.abs(demonstrated)) / numpy.median(numpy.abs(predicted)):.2f}"]
    lines.append("by |predicted twist|: rows, sign right where the demonstration twisted 2 deg or more, median |demonstrated|, median |predicted|")
    for low, high in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 20)):
        part = (numpy.abs(predicted) >= low) & (numpy.abs(predicted) < high)
        twisted = part & (numpy.abs(demonstrated) >= 2.0)
        sign = float((numpy.sign(predicted[twisted]) == numpy.sign(demonstrated[twisted])).mean()) if twisted.any() else float("nan")
        lines.append(f"  {low} to {high} deg: {int(part.sum())}, {sign:.3f}, {numpy.median(numpy.abs(demonstrated[part])):.2f}, {numpy.median(numpy.abs(predicted[part])):.2f}")
    print("\n".join(lines))
    (OUTPUT_FOLDER / "twist_calibration.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
