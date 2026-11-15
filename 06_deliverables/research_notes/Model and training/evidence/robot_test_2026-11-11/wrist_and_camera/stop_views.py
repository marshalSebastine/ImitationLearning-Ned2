"""
Which wrist did the demonstrations have when they stopped, and would the models still stop there with the wrist held?
(key: wrist_and_camera, part 7)

The stop decision is learnt from the stop rows (the last row of every demonstration and the rows the audit cleaning
relabelled as stops). If many of them were recorded with a wrist far from the driver's 13.5 degrees, the driver shows
the model a stop view moved up or down the picture by 0.70 px at 160 per degree. Part 1 counts the stop rows by wrist
(measured). Part 2 imitates the held-wrist view on the stop rows whose wrist was more than 5.7 degrees away, exactly as
shift_sensitivity.py does (the fork kept, everything else moved; a crude imitation), and reads the stop chance of the
fold model that never saw the sheet and of model E (calculated).

Run from this folder (about two minutes on two threads):
    D:/AI_Project/.venv/Scripts/python.exe stop_views.py
"""

from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas
import torch

import wrist_geometry
import model_mode
import shift_sensitivity
import training_targets


STOP_PARAMETERS = {
    "stop_threshold": 0.5,
    "stop_plain_shifts_px": [[4, 0], [-4, 0], [8, 0], [-8, 0], [0, 4], [0, -4], [0, 8], [0, -8], [0, 12], [0, -12]]
}



def main(parameters: Optional[Dict] = None) -> None:
    """Print the wrist at the stop rows and the stop chance as demonstrated and in the imitated held-wrist view."""

    settings = dict(shift_sensitivity.SENSITIVITY_PARAMETERS)
    settings.update(STOP_PARAMETERS)
    settings.update(parameters or {})
    torch.set_num_threads(settings["threads"])
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    held = wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]

    frames = training_targets.load_training_frames()
    frames["joint_5_deg"] = numpy.degrees(frames["joint_5"])
    frames["wrist_offset_deg"] = frames["joint_5_deg"] - held
    frames["is_on_mark"] = frames["sheet_orientation"].str.contains("START MARK")
    stops = frames[frames["action"] == "stop"].copy()
    is_off = stops["wrist_offset_deg"].abs() > settings["wrist_range_deg"]
    print(f"training rows after the audit cleaning: {len(frames)}; stop rows: {len(stops)} in {stops['episode_id'].nunique()} episodes")
    print(f"stop rows whose wrist is more than {settings['wrist_range_deg']:.1f} degrees from the held {held}: {int(is_off.sum())} ({100.0 * is_off.mean():.1f} %); "
          f"wrist at the stop rows: 5th percentile {stops['joint_5_deg'].quantile(0.05):.1f}, median {stops['joint_5_deg'].median():.1f}, 95th {stops['joint_5_deg'].quantile(0.95):.1f} degrees")
    by_sheet = stops.groupby("sheet").agg(stop_rows = ("wrist_offset_deg", "size"), wrist_median_deg = ("joint_5_deg", "median"),
                                          wrist_p10_deg = ("joint_5_deg", lambda values: values.quantile(0.1)), wrist_p90_deg = ("joint_5_deg", lambda values: values.quantile(0.9)),
                                          off_range_share = ("wrist_offset_deg", lambda values: (values.abs() > settings["wrist_range_deg"]).mean()),
                                          view_shift_mean_abs_160_px = ("wrist_offset_deg", lambda values: settings["pixels_per_wrist_degree_160"] * values.abs().mean()))
    print(by_sheet.round(2).to_string())
    by_start = stops.groupby("is_on_mark").agg(stop_rows = ("wrist_offset_deg", "size"), wrist_median_deg = ("joint_5_deg", "median"),
                                               off_range_share = ("wrist_offset_deg", lambda values: (values.abs() > settings["wrist_range_deg"]).mean()))
    print("by start (True: start mark, False: taught start):")
    print(by_start.round(2).to_string())
    all_rows_off = (frames["wrist_offset_deg"].abs() > settings["wrist_range_deg"]).mean()
    print(f"for comparison, all training rows: {100.0 * all_rows_off:.1f} % are more than {settings['wrist_range_deg']:.1f} degrees from {held}")

    images = numpy.load(settings["cache_file"], mmap_mode = "r")
    mask = shift_sensitivity.fork_mask(images, settings)
    pictures = numpy.stack([images[index] for index in stops["frame_index"]])
    shifts = numpy.rint(settings["pixels_per_wrist_degree_160"] * stops["wrist_offset_deg"].to_numpy()).astype(int)
    # two controls beside the held-wrist view: the whole picture moved by the same amount (no fork kept, nothing
    # painted over), and the content moved the other way by the same amount (as far from the recorded view, but not
    # the driver's view), which shows whether any move of this size costs stops or only the move towards 13.5 degrees
    changed_pictures = {
        "as demonstrated": pictures,
        "held-wrist view": shift_sensitivity.held_wrist_pictures(pictures, shifts, mask),
        "whole picture moved the same": numpy.stack([shift_sensitivity.shifted_pictures(picture[None], 0, int(shift))[0] for picture, shift in zip(pictures, shifts)]),
        "moved the other way (fork kept)": shift_sensitivity.held_wrist_pictures(pictures, -shifts, mask)
    }
    # plain shifts of the whole picture on every stop row: how fragile is the stop decision to a view that sits a few
    # pixels elsewhere, sideways as the camera's step on 8.11.26 moved it, or up and down as the wrist moves it
    for shift_x, shift_y in settings["stop_plain_shifts_px"]:
        changed_pictures[f"plain shift x {shift_x:+d}, y {shift_y:+d}"] = shift_sensitivity.shifted_pictures(pictures, shift_x, shift_y)

    all_sheets_policy = model_mode.ModelPolicy(settings["all_sheets_run_name"], Path(settings["all_sheets_run_folder"]) / "checkpoints" / "all_sheets.pt",
                                               Path(settings["all_sheets_run_folder"]) / "metrics.json")
    answers = {"model E (all sheets)": {name: shift_sensitivity.model_answers(all_sheets_policy, changed, settings)["stop_probability"] for name, changed in changed_pictures.items()}}
    fold = {name: numpy.zeros(len(stops)) for name in changed_pictures}
    sheets = stops["sheet"].to_numpy()
    for sheet in settings["sheets"]:
        positions = numpy.flatnonzero(sheets == sheet)
        policy = model_mode.ModelPolicy(settings["fold_run_name"], Path(settings["fold_run_folder"]) / "checkpoints" / f"without_{sheet}.pt",
                                        Path(settings["fold_run_folder"]) / "metrics.json")
        for name, changed in changed_pictures.items():
            fold[name][positions] = shift_sensitivity.model_answers(policy, changed[positions], settings)["stop_probability"]
    answers["fold models (sheet never seen)"] = fold
    torch.set_num_threads(settings["threads"])

    print(f"\n=== stop chance at the demonstrated stop rows (a stop needs {settings['stop_threshold']} or more) ===")
    rows = []
    groups = [("wrist within range of 13.5 (view unchanged)", ~is_off.to_numpy()),
              ("wrist out of range, all", is_off.to_numpy()),
              ("wrist out of range, below 13.5 (content moves up)", is_off.to_numpy() & (shifts < 0)),
              ("wrist out of range, above 13.5 (content moves down)", is_off.to_numpy() & (shifts > 0))]
    for who, views in answers.items():
        for label, is_group in groups:
            if is_group.sum() == 0:
                continue
            rows.append({"model": who, "stop rows": label, "count": int(is_group.sum()),
                         "content_shift_mean_abs_160_px": float(numpy.abs(shifts[is_group]).mean()),
                         "stop_chance_mean_demonstrated": float(views["as demonstrated"][is_group].mean()),
                         "stop_chance_mean_held_view": float(views["held-wrist view"][is_group].mean()),
                         "would_stop_share_demonstrated": float((views["as demonstrated"][is_group] >= settings["stop_threshold"]).mean()),
                         "would_stop_share_held_view": float((views["held-wrist view"][is_group] >= settings["stop_threshold"]).mean()),
                         "would_stop_share_whole_picture_moved": float((views["whole picture moved the same"][is_group] >= settings["stop_threshold"]).mean()),
                         "would_stop_share_moved_other_way": float((views["moved the other way (fork kept)"][is_group] >= settings["stop_threshold"]).mean())})
    table = pandas.DataFrame(rows)
    print(table.round(3).to_string(index = False))
    table.to_csv(output_folder / "stop_views.csv", index = False)

    generator = numpy.random.default_rng(settings["seed"])
    for who, views in answers.items():
        for label, is_group in [("out-of-range stop rows", is_off.to_numpy()), ("in-range stop rows (moved by 4 px or less)", ~is_off.to_numpy())]:
            for name in ["held-wrist view", "whole picture moved the same", "moved the other way (fork kept)"]:
                differences = (views[name][is_group] >= settings["stop_threshold"]).astype(float) - (views["as demonstrated"][is_group] >= settings["stop_threshold"]).astype(float)
                draws = generator.integers(0, len(differences), size = (2000, len(differences)))
                means = differences[draws].mean(axis = 1)
                print(f"{who}, {label}: share with a stop chance of {settings['stop_threshold']} or more, {name} against as demonstrated: "
                      f"{100.0 * differences.mean():+.1f} points (95 % interval {100.0 * numpy.quantile(means, 0.025):+.1f} to {100.0 * numpy.quantile(means, 0.975):+.1f})")
    print("\n=== every stop row under plain shifts of the whole picture (px at 160): share with a stop chance of the threshold or more ===")
    plain_rows = []
    for who, views in answers.items():
        row = {"model": who, "unchanged": float((views["as demonstrated"] >= settings["stop_threshold"]).mean())}
        for shift_x, shift_y in settings["stop_plain_shifts_px"]:
            row[f"x{shift_x:+d} y{shift_y:+d}"] = float((views[f"plain shift x {shift_x:+d}, y {shift_y:+d}"] >= settings["stop_threshold"]).mean())
        plain_rows.append(row)
    plain_table = pandas.DataFrame(plain_rows)
    print(plain_table.round(3).to_string(index = False))
    plain_table.to_csv(output_folder / "stop_views_plain_shifts.csv", index = False)
    print(f"\nsaved {output_folder / 'stop_views.csv'} and stop_views_plain_shifts.csv")



if __name__ == "__main__":
    main()
