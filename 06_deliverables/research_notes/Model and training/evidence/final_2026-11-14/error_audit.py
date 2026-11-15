"""
Where is the reference model wrong? An audit of its held-out predictions by situation, 15.11.26.

Asked by Emranul after he found the corrections by hand: "there are a lot of hidden things like this; you should find
them". Every held-out picture of the reference runs (two seeds, 16 new tracks) is put into groups by what the
recording knows about it, and the path measure (predicted direction within 45 degrees of where the tip goes) is
given per group, with the group's share of the pictures. A group far below the average is a candidate.

Run from 02_code:  python "<this file>"
"""

import sys
from pathlib import Path
from typing import Dict

import numpy
import pandas

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "02_code"))

from config import TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy_emranul import merged_run_settings
from training_probes import path_directions, training_arrays
from training_targets import load_training_frames


ERROR_AUDIT_PARAMETERS = {
    "runs_folder": Path(__file__).resolve().parents[5] / "04_runs" / "gpu_ki014",
    "runs": ["2026-11-14_resnet18_full_new_shapes", "2026-11-14_resnet18_full_new_shapes_seed_2"],
    "recipe_run_name": "resnet18_full_new_shapes",
    "smallest_group": 60,
    "output_path": Path(__file__).resolve().parent / "error_audit.csv"
}


def audited_rows(parameters: Dict) -> pandas.DataFrame:
    """Return one row per held-out picture and run, with the angle error and everything the recording knows."""

    settings = merged_run_settings(parameters["recipe_run_name"])
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    settings["chunk_steps"] = target_parameters["chunk_steps"]
    data_parameters = dict(TRAINING_DATA_PARAMETERS)
    data_parameters["processed_folder"] = settings["processed_folder"]
    frames = load_training_frames(target_parameters, data_parameters)
    arrays, _, _ = training_arrays(frames, settings, target_parameters)
    path_angle, is_path_row = path_directions(frames, numpy.arange(len(frames)), arrays)
    frames = frames.assign(path_angle = path_angle, is_path_row = is_path_row, row = numpy.arange(len(frames)))
    grouped = frames.groupby("episode_id", sort = False)
    frames["previous_path_angle"] = grouped["path_angle"].shift(1)
    frames["rows_kept_in_episode"] = grouped["row"].transform("count")
    frames["kept_position"] = grouped.cumcount()
    row_of_frame = {int(frame_index): row for row, frame_index in enumerate(frames["frame_index"].to_numpy())}
    parts = []
    for run in parameters["runs"]:
        for path in sorted((parameters["runs_folder"] / run).glob("predictions_without_T*.npz")):
            saved = numpy.load(path)
            rows = numpy.array([row_of_frame[int(frame_index)] for frame_index in saved["frame_index"]])
            part = frames.iloc[rows].copy()
            part["run"] = run
            part["predicted_angle"] = saved["angle_deg"]
            part["stop_probability"] = saved["stop_probability"]
            parts.append(part)
    table = pandas.concat(parts, ignore_index = True)
    table = table[table["is_path_row"]].copy()
    table["error_deg"] = numpy.abs((table["predicted_angle"] - table["path_angle"] + 180.0) % 360.0 - 180.0)
    table["is_within_45"] = table["error_deg"] <= 45.0
    return table


def grouped_table(table: pandas.DataFrame) -> Dict[str, pandas.Series]:
    """Return the groupings: a name and, per picture, the group it falls into."""

    reach_cm = 100.0 * numpy.hypot(table["tip_x_m"], table["tip_y_m"])
    path_angle = table["path_angle"]
    bend_deg = numpy.abs((path_angle - table["previous_path_angle"] + 180.0) % 360.0 - 180.0)
    left_cm = 100.0 * table["tip_y_m"]
    return {
        "where the path goes (picture frame)": pandas.cut(numpy.abs(path_angle), [-1, 15, 45, 90, 135, 181], labels = ["ahead 0-15", "15-45", "sideways 45-90", "90-135", "backwards 135-180"]),
        "path goes left or right": numpy.where(path_angle > 15, "left", numpy.where(path_angle < -15, "right", "ahead")),
        "how sharply the path turns at this step": pandas.cut(bend_deg.fillna(0), [-1, 10, 25, 45, 181], labels = ["0-10 deg", "10-25", "25-45", "over 45"]),
        "steps since the episode's start": pandas.cut(table["kept_position"], [-1, 2, 7, 20, 10000], labels = ["1-3", "4-8", "9-21", "later"]),
        "steps to the episode's end": pandas.cut(table["rows_kept_in_episode"] - table["kept_position"], [0, 3, 8, 20, 10000], labels = ["last 3", "4-8", "9-20", "earlier"]),
        "steps since an h press": pandas.cut(table["steps_since_correction"].fillna(9999), [0, 3, 8, 20, 9998, 10000], labels = ["1-3", "4-8", "9-20", "later", "no h before"]),
        "steps to the next h press": pandas.cut(table["steps_to_next_correction"].fillna(9999), [0, 2, 5, 10, 9998, 10000], labels = ["1-2", "3-5", "6-10", "earlier", "no h after"]),
        "reach of the rod tip": pandas.cut(reach_cm, [0, 25, 32, 38, 42, 100], labels = ["under 25 cm", "25-32", "32-38", "38-42", "over 42"]),
        "tip left or right of the robot's middle": pandas.cut(left_cm, [-100, -12, -4, 4, 12, 100], labels = ["right over 12 cm", "right 4-12", "middle", "left 4-12", "left over 12"]),
        "forearm twist now": pandas.cut(numpy.abs(table["forearm_turn_deg"]), [-1, 10, 30, 60, 200], labels = ["0-10 deg", "10-30", "30-60", "over 60"]),
        "forearm twist changes in this step": pandas.cut(numpy.abs(table["forearm_change_deg"].fillna(0)), [-1, 1, 4, 8, 100], labels = ["under 1 deg", "1-4", "4-8", "over 8"]),
        "rod lean": pandas.cut(table["rod_from_vertical_deg"], [-1, 20, 30, 40, 90], labels = ["under 20 deg", "20-30", "30-40", "over 40"]),
        "step length": pandas.cut(100.0 * table["move_length_m"], [-1, 0.4, 0.8, 1.2, 10], labels = ["under 0.4 cm", "0.4-0.8", "0.8-1.2", "over 1.2"]),
        "operator's wait before this step": pandas.cut(table["decision_after_capture_s"].fillna(0), [-1, 0.5, 2, 10, 100000], labels = ["under 0.5 s (running on)", "0.5-2", "2-10", "over 10 s"]),
        "picture sharpness": pandas.qcut(table["sharpness_measured"], 4, labels = ["least sharp quarter", "second", "third", "sharpest quarter"]),
        "picture brightness": pandas.qcut(table["brightness_mean"], 4, labels = ["darkest quarter", "second", "third", "brightest quarter"]),
        "episode score": table["score"].astype(int).astype(str),
        "line kind": numpy.where(table["path_id"].astype(str).str.contains("dashed"), "dashed", "solid"),
        "start kind": table["start_kind"].astype(str).str.slice(0, 3),
        "camera shift of the episode": pandas.cut(table["camera_shift_px"].fillna(0), [-1, 3, 6, 100], labels = ["under 3 px", "3-6", "over 6"]),
        "track": table["path_id"].astype(str).str.extract(r"(T\d\d)")[0],
    }


def main(parameters: Dict = ERROR_AUDIT_PARAMETERS) -> None:
    """Print every grouping and write the groups, worst first, to "output_path"."""

    table = audited_rows(parameters)
    overall = float(table["is_within_45"].mean())
    print(f"held-out path pictures (two seeds together): {len(table)}, within 45 degrees: {overall:.3f}")
    records = []
    for name, groups in grouped_table(table).items():
        summary = table.groupby(numpy.asarray(groups, dtype = object), dropna = True)["is_within_45"].agg(["mean", "count"])
        print(f"\n{name}")
        for group, line in summary.iterrows():
            share = line["count"] / len(table)
            print(f"   {str(group):28s} {line['mean']:.3f}   {int(line['count']):6d} pictures ({100 * share:4.1f} %)")
            if line["count"] >= parameters["smallest_group"]:
                records.append({"grouping": name, "group": group, "within_45": round(line["mean"], 3), "pictures": int(line["count"]),
                                "share": round(share, 3), "lost_points": round(100 * share * (overall - line["mean"]), 2)})
    ranked = pandas.DataFrame(records).sort_values("lost_points", ascending = False)
    ranked.to_csv(parameters["output_path"], index = False)
    print("\ngroups that cost the most (points of the overall measure lost against the average):")
    print(ranked.head(22).to_string(index = False))


if __name__ == "__main__":
    main()
