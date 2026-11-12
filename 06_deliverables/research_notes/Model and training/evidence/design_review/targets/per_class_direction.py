"""
Where the exact-step model and the move-name model differ: the share of moving pictures predicted within 45 deg, per
demonstrated direction class, for the end-to-end runs on held-out sheets 03, 05 and 06 (read only).
Writes per_class_direction.out.txt.
"""

import numpy
import pandas

from common_frames import DIRECTION_NAMES, OUTPUT_FOLDER, RUNS_FOLDER, read_frames

NOMINAL = {"forward": 0.0, "forward_left": 60.0, "left": 90.0, "back_left": 135.0, "back": 180.0, "back_right": -135.0,
           "right": -90.0, "forward_right": -60.0}
lines = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush = True)


def nearest(angles):
    names = list(NOMINAL)
    values = numpy.array([NOMINAL[name] for name in names])
    differences = numpy.abs((numpy.asarray(angles)[:, None] - values[None, :] + 180.0) % 360.0 - 180.0)
    return numpy.array(names)[numpy.argmin(differences, axis = 1)]


def main() -> None:
    frames = read_frames()[["frame_index", "action", "move_angle_deg"]]
    results = {}
    for run in ("2026-11-09_resnet18_finetune_exact", "2026-11-09_resnet18_finetune_classes"):
        parts = []
        for fold in ("without_path_03", "without_path_05", "without_path_06"):
            data = numpy.load(RUNS_FOLDER / run / f"predictions_{fold}.npz")
            part = pandas.DataFrame({"frame_index": data["frame_index"], "angle_deg": data["angle_deg"]})
            part["fold"] = fold
            parts.append(part)
        table = pandas.concat(parts).merge(frames, on = "frame_index", how = "left")
        table = table[table["action"].isin(DIRECTION_NAMES)].copy()
        table["true_class"] = nearest(table["move_angle_deg"].to_numpy())
        table["error"] = numpy.abs((table["angle_deg"] - table["move_angle_deg"] + 180.0) % 360.0 - 180.0)
        table["predicted_class"] = nearest(table["angle_deg"].to_numpy())
        results[run] = table
    say("share within 45 deg by demonstrated class (end-to-end, sheets 03, 05, 06 pooled): exact / names; rows; predicted-class share exact / names")
    exact = results["2026-11-09_resnet18_finetune_exact"]
    names = results["2026-11-09_resnet18_finetune_classes"]
    for name in NOMINAL:
        exact_part = exact[exact["true_class"] == name]
        names_part = names[names["true_class"] == name]
        say(f"  {name:>13}: {(exact_part['error'] <= 45).mean():.3f} / {(names_part['error'] <= 45).mean():.3f}; rows {len(exact_part)}; "
            f"predicted as this class {(exact['predicted_class'] == name).mean():.3f} / {(names['predicted_class'] == name).mean():.3f} "
            f"(demonstrated {(exact['true_class'] == name).mean():.3f})")
    back = ["back", "back_left", "back_right"]
    exact_back = exact[exact["true_class"].isin(back)]
    names_back = names[names["true_class"].isin(back)]
    say(f"  all back classes: within 45 {(exact_back['error'] <= 45).mean():.3f} / {(names_back['error'] <= 45).mean():.3f} over {len(exact_back)} rows; "
        f"other classes {(exact[~exact['true_class'].isin(back)]['error'] <= 45).mean():.3f} / {(names[~names['true_class'].isin(back)]['error'] <= 45).mean():.3f}")
    probe_folder = RUNS_FOLDER / "2026-11-09_frozen_feature_probes" / "predictions"
    say("frozen probes, six held-out sheets pooled: within 45 deg on back classes / on the other classes")
    for run, key in (("2026-11-09_resnet18_finetune_classes", "angle_expected_deg"), ("2026-11-09_resnet18_finetune_exact", "angle_deg")):
        parts = []
        for fold in ("without_path_03", "without_path_05", "without_path_06"):
            data = numpy.load(RUNS_FOLDER / run / f"predictions_{fold}.npz")
            parts.append(pandas.DataFrame({"frame_index": data["frame_index"], "angle": data[key]}))
        table = pandas.concat(parts).merge(frames, on = "frame_index", how = "left")
        table = table[table["action"].isin(DIRECTION_NAMES)].copy()
        true = nearest(table["move_angle_deg"].to_numpy())
        error = numpy.abs((table["angle"] - table["move_angle_deg"] + 180.0) % 360.0 - 180.0)
        is_back = numpy.isin(true, back)
        say(f"  end-to-end {run[11:]} ({key}): back {(error[is_back] <= 45).mean():.3f} / other {(error[~is_back] <= 45).mean():.3f} / all {(error <= 45).mean():.3f}")
    for variant, key in (("exact_held_out_sheets", "angle_deg"), ("classes_held_out_sheets", "angle_deg"), ("classes_held_out_sheets", "angle_expected_deg"),
                         ("chunk_of_5_steps", "angle_deg"), ("mixed_classes_and_exact", "angle_vector_deg"), ("mixed_classes_and_exact", "angle_deg"),
                         ("mixed_classes_and_exact", "angle_expected_deg"), ("no_class_weights", "angle_deg"), ("no_class_weights", "angle_expected_deg")):
        parts = []
        for number in range(1, 7):
            data = numpy.load(probe_folder / f"{variant}_without_path_0{number}.npz")
            parts.append(pandas.DataFrame({"frame_index": data["frame_index"], "angle": data[key]}))
        table = pandas.concat(parts).merge(frames, on = "frame_index", how = "left")
        table = table[table["action"].isin(DIRECTION_NAMES)].copy()
        table["true_class"] = nearest(table["move_angle_deg"].to_numpy())
        error = numpy.abs((table["angle"] - table["move_angle_deg"] + 180.0) % 360.0 - 180.0)
        is_back = table["true_class"].isin(back)
        say(f"  {variant} ({key}): {(error[is_back] <= 45).mean():.3f} / {(error[~is_back] <= 45).mean():.3f} "
            f"(back rows {int(is_back.sum())} of {len(table)}); predicted back share {nearest(table['angle'].to_numpy()).tolist().count('back') / len(table):.3f}")
    (OUTPUT_FOLDER / "per_class_direction.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
