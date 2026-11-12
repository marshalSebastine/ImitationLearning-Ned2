"""
How much do the models' answers change when the view is shifted as the camera's creep and the held wrist shift it?
(key: wrist_and_camera, part 6)

Two view changes were found: the camera sat about 16 px (at 640; 4 px at the model's 160) further right on 11.11.26
than when sheets 01 to 04 were demonstrated (camera_creep.py), and a held wrist moves the hand, the block and the
table up or down the picture by 2.8 px at 640 (0.70 px at 160) per degree of difference from the demonstrated wrist
while the blue fork stays where it is (held_wrist_views.py). Both are imitated here on the cached 160 x 120 training
pictures, and the saved models answer the changed pictures:

    plain shifts      the whole picture moved by whole pixels with its edge repeated, as the training augmentation does
                      (it used up to 4 px); positive x = to the right, positive y = downwards
    held-wrist view   for demonstration rows whose wrist was more than 5.7 degrees from 13.5: everything except the
                      fork moved up or down by the row's own amount, the fork kept, the fork's ghost painted over
                      (an imitation, not a rendering: the table's perspective and the rod's turn are ignored)

Models: the fold models of model E's recipe on the sheet each never saw (the exam's situation), and model E itself,
trained on all sheets, on the same rows (the situation of 11.11.26). No training; about three minutes on two threads.
Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe shift_sensitivity.py
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy
import pandas
import torch

from matplotlib import pyplot

import wrist_geometry
import model_mode
from train_policy import augmented_pictures
from training_probes import predictions_from_outputs


SENSITIVITY_PARAMETERS = {
    "cache_file": Path("D:/AI_Project/03_data/processed/v01_training_index/images_wide_160.npy"),
    "fold_run_folder": wrist_geometry.ROOT_FOLDER / "04_runs" / "gpu_ki014" / "2026-11-10_resnet18_chunk_stop_weight_25",
    "fold_run_name": "resnet18_chunk_stop_weight_25",
    "all_sheets_run_folder": wrist_geometry.ROOT_FOLDER / "04_runs" / "gpu_ki014" / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets",
    "all_sheets_run_name": "resnet18_model_4_chunk_stop_25_all_sheets",
    "sheets": ["path_01", "path_02", "path_03", "path_04", "path_05", "path_06"],
    "rows_per_sheet": 100,
    "wrist_rows_per_sheet": 100,
    "seed": 11_11_26,
    "threads": 2,
    "batch_size": 100,
    "plain_shifts_px": [[0, 0], [4, 0], [-4, 0], [8, 0], [-8, 0], [0, 4], [0, -4], [0, 8], [0, -8]],
    # 2.803 px at 640 per degree of wrist is 0.70 px at 160
    "pixels_per_wrist_degree_160": 2.803 / 4.0,
    "wrist_range_deg": 16.0 / 2.803,
    "smallest_step_m": 0.005,
    "smallest_twist_deg": 2.0,
    "fork_blue_share": 0.85,
    "fork_sample_pictures": 800,
    "figure_dpi": 130
}



def shifted_pictures(
        pictures: numpy.ndarray,
        shift_x: int,
        shift_y: int
) -> numpy.ndarray:
    """Move every picture's content by whole pixels (x to the right, y downwards), repeating the edge pixels."""

    reach = max(abs(shift_x), abs(shift_y))
    if reach == 0:
        return pictures.copy()
    padded = numpy.pad(pictures, ((0, 0), (reach, reach), (reach, reach), (0, 0)), mode = "edge")
    height, width = pictures.shape[1:3]
    return padded[:, reach - shift_y:reach - shift_y + height, reach - shift_x:reach - shift_x + width]



def fork_mask(
        images: numpy.ndarray,
        parameters: Dict
) -> numpy.ndarray:
    """
    The pixels the blue fork fills in nearly every picture (the fork is part of the forearm, as the camera is, so it
    does not move in the picture): saturated blue in at least "fork_blue_share" of a random sample, grown by one pixel.
    """

    chosen = numpy.sort(numpy.random.default_rng(parameters["seed"]).choice(len(images), parameters["fork_sample_pictures"], replace = False))
    blue_count = numpy.zeros(images.shape[1:3])
    for index in chosen:
        hsv = cv2.cvtColor(numpy.ascontiguousarray(images[index]), cv2.COLOR_RGB2HSV)
        blue_count += (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 125) & (hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 60)
    mask = (blue_count / len(chosen) >= parameters["fork_blue_share"]).astype(numpy.uint8)
    return cv2.dilate(mask, numpy.ones((3, 3), numpy.uint8)).astype(bool)



def held_wrist_pictures(
        pictures: numpy.ndarray,
        shifts_y: numpy.ndarray,
        mask: numpy.ndarray
) -> numpy.ndarray:
    """
    Imitate the view with the wrist held: each picture's content moved up or down by its own whole number of pixels,
    the fork kept where it is, and the moved copy of the fork painted over from its surroundings.
    """

    results = numpy.empty_like(pictures)
    for index, (picture, shift_y) in enumerate(zip(pictures, shifts_y)):
        moved = shifted_pictures(picture[None], 0, int(shift_y))[0].copy()
        ghost = shifted_pictures(mask[None, :, :, None].astype(numpy.uint8), 0, int(shift_y))[0, :, :, 0].astype(bool) & ~mask
        if ghost.any():
            moved = cv2.inpaint(numpy.ascontiguousarray(moved), ghost.astype(numpy.uint8), 2, cv2.INPAINT_TELEA)
        moved[mask] = picture[mask]
        results[index] = moved
    return results



def model_answers(
        policy: model_mode.ModelPolicy,
        pictures: numpy.ndarray,
        parameters: Dict
) -> Dict[str, numpy.ndarray]:
    """
    The model's answers to cached pictures, prepared exactly as in training without augmentation: the exact step's
    direction (what the robot driver carries out), the chunk's first-step direction and path direction (what the
    training runs scored), the twist and the stop chance.
    """

    collected = {"exact_angle_deg": [], "chunk_angle_deg": [], "path_angle_deg": [], "twist_deg": [], "stop_probability": [], "length_cm": []}
    with torch.inference_mode():
        for start in range(0, len(pictures), parameters["batch_size"]):
            batch = torch.from_numpy(numpy.ascontiguousarray(pictures[start:start + parameters["batch_size"]]))
            inputs, _ = augmented_pictures(batch, policy.settings, numpy.random.default_rng(0), is_training = False)
            outputs = policy.models[0](inputs)
            predictions = predictions_from_outputs(outputs, policy.class_names, policy.settings["stop_target"])
            exact = outputs["exact"].numpy()
            collected["exact_angle_deg"].append(numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0])))
            collected["length_cm"].append(numpy.hypot(exact[:, 0], exact[:, 1]))
            collected["chunk_angle_deg"].append(numpy.asarray(predictions["angle_deg"]))
            collected["path_angle_deg"].append(numpy.asarray(predictions.get("angle_vector_deg", predictions["angle_deg"])))
            collected["twist_deg"].append(exact[:, 2])
            collected["stop_probability"].append(numpy.asarray(predictions["stop_probability"]))
    return {name: numpy.concatenate(values) for name, values in collected.items()}



def angle_gap(first: numpy.ndarray, second: numpy.ndarray) -> numpy.ndarray:
    """The absolute difference of two directions in degrees, at most 180."""

    return numpy.abs((numpy.asarray(first) - numpy.asarray(second) + 180.0) % 360.0 - 180.0)



def scores(
        answers: Dict[str, numpy.ndarray],
        base: Dict[str, numpy.ndarray],
        rows: pandas.DataFrame,
        parameters: Dict
) -> Dict[str, float]:
    """The numbers printed for one model and one view change, against the demonstrated step and the unchanged view."""

    demonstrated = rows["demonstrated_angle_deg"].to_numpy()
    twist = rows["forearm_change_deg"].to_numpy(dtype = float)
    is_twisting = numpy.abs(twist) >= parameters["smallest_twist_deg"]
    return {
        "exact_within_45": float((angle_gap(answers["exact_angle_deg"], demonstrated) <= 45.0).mean()),
        "chunk_within_45": float((angle_gap(answers["chunk_angle_deg"], demonstrated) <= 45.0).mean()),
        "exact_error_median_deg": float(numpy.median(angle_gap(answers["exact_angle_deg"], demonstrated))),
        "exact_turned_median_deg": float(numpy.median(angle_gap(answers["exact_angle_deg"], base["exact_angle_deg"]))),
        "exact_turned_p90_deg": float(numpy.quantile(angle_gap(answers["exact_angle_deg"], base["exact_angle_deg"]), 0.9)),
        "exact_drift_mean_deg": float(numpy.mean((answers["exact_angle_deg"] - base["exact_angle_deg"] + 180.0) % 360.0 - 180.0)),
        "length_mean_cm": float(answers["length_cm"].mean()),
        "twist_sign_right": float((numpy.sign(answers["twist_deg"][is_twisting]) == numpy.sign(twist[is_twisting])).mean()),
        "twist_size_ratio": float(numpy.abs(answers["twist_deg"][is_twisting]).mean() / numpy.abs(twist[is_twisting]).mean()),
        "twist_change_mean_abs_deg": float(numpy.abs(answers["twist_deg"] - base["twist_deg"]).mean()),
        "stop_chance_mean": float(answers["stop_probability"].mean())
    }



def chosen_rows(
        frames: pandas.DataFrame,
        fold_run_folder: Path,
        parameters: Dict
) -> Tuple[pandas.DataFrame, pandas.DataFrame, Dict[int, Dict[str, float]]]:
    """
    The rows the test uses: per sheet a random sample of the fold's scored rows that move the tip, and a second sample
    restricted to rows whose wrist was outside the range around 13.5 degrees; also the fold's saved predictions by
    frame index, to check the inference path.
    """

    generator = numpy.random.default_rng(parameters["seed"])
    plain, wrist, saved_by_frame = [], [], {}
    for sheet in parameters["sheets"]:
        saved = numpy.load(Path(fold_run_folder) / f"predictions_without_{sheet}.npz")
        for position, frame_index in enumerate(saved["frame_index"]):
            saved_by_frame[int(frame_index)] = {"chunk_angle_deg": float(saved["angle_deg"][position]), "twist_deg": float(saved["twist_deg"][position])}
        of_sheet = frames[frames["frame_index"].isin(saved["frame_index"]) & (frames["action"] != "stop")
                          & (numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0)) >= parameters["smallest_step_m"])]
        plain.append(of_sheet.loc[generator.choice(of_sheet.index, size = min(len(of_sheet), parameters["rows_per_sheet"]), replace = False)])
        off_wrist = of_sheet[(of_sheet["joint_5_deg"] - wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]).abs() > parameters["wrist_range_deg"]]
        wrist.append(off_wrist.loc[generator.choice(off_wrist.index, size = min(len(off_wrist), parameters["wrist_rows_per_sheet"]), replace = False)])
    return pandas.concat(plain).sort_values("frame_index"), pandas.concat(wrist).sort_values("frame_index"), saved_by_frame



def main(parameters: Optional[Dict] = None) -> None:
    """Run both view changes through the fold models and model E, print the tables and save two figures."""

    settings = dict(SENSITIVITY_PARAMETERS)
    settings.update(parameters or {})
    torch.set_num_threads(settings["threads"])
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 200)
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]
    images = numpy.load(settings["cache_file"], mmap_mode = "r")
    frames = wrist_geometry.load_demonstration_frames()
    frames["demonstrated_angle_deg"] = numpy.degrees(numpy.arctan2(frames["tip_left_m"].fillna(0.0), frames["tip_forward_m"].fillna(0.0)))
    plain_rows, wrist_rows, saved_by_frame = chosen_rows(frames, settings["fold_run_folder"], settings)
    mask = fork_mask(images, settings)
    print(f"rows: {len(plain_rows)} for the plain shifts, {len(wrist_rows)} with the wrist more than {settings['wrist_range_deg']:.1f} degrees from 13.5 for the held-wrist view; "
          f"fork mask: {int(mask.sum())} of {mask.size} pixels")

    wrist_offsets = wrist_rows["joint_5_deg"].to_numpy() - wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]
    wrist_shifts = numpy.rint(settings["pixels_per_wrist_degree_160"] * wrist_offsets).astype(int)
    print(f"held-wrist view: the content moves by {numpy.abs(wrist_shifts).mean():.1f} px at 160 on average (5th to 95th percentile "
          f"{numpy.quantile(wrist_shifts, 0.05):.0f} to {numpy.quantile(wrist_shifts, 0.95):.0f}; positive = downwards)")

    plain_pictures = numpy.stack([images[index] for index in plain_rows["frame_index"]])
    wrist_pictures = numpy.stack([images[index] for index in wrist_rows["frame_index"]])
    wrist_changed = held_wrist_pictures(wrist_pictures, wrist_shifts, mask)
    wrist_plain_shift = numpy.stack([shifted_pictures(picture[None], 0, int(shift))[0] for picture, shift in zip(wrist_pictures, wrist_shifts)])

    policies = {}
    for sheet in settings["sheets"]:
        policies[sheet] = model_mode.ModelPolicy(settings["fold_run_name"], Path(settings["fold_run_folder"]) / "checkpoints" / f"without_{sheet}.pt",
                                                 Path(settings["fold_run_folder"]) / "metrics.json")
    all_sheets_policy = model_mode.ModelPolicy(settings["all_sheets_run_name"], Path(settings["all_sheets_run_folder"]) / "checkpoints" / "all_sheets.pt",
                                               Path(settings["all_sheets_run_folder"]) / "metrics.json")
    torch.set_num_threads(settings["threads"])

    def answers_of(
            who: str,
            rows: pandas.DataFrame,
            pictures: numpy.ndarray
    ) -> Dict[str, numpy.ndarray]:
        """The fold models' answers (each on the rows of the sheet it never saw) or model E's, in the rows' order."""

        if who == "model E (all sheets)":
            return model_answers(all_sheets_policy, pictures, settings)
        parts = {}
        sheets = rows["sheet"].to_numpy()
        for sheet in settings["sheets"]:
            positions = numpy.flatnonzero(sheets == sheet)
            if len(positions) == 0:
                continue
            answers = model_answers(policies[sheet], pictures[positions], settings)
            for name, values in answers.items():
                parts.setdefault(name, numpy.zeros(len(rows)))[positions] = values
        return parts

    models = ["fold models (sheet never seen)", "model E (all sheets)"]
    plain_table = []
    base_answers = {}
    for who in models:
        base_answers[who] = answers_of(who, plain_rows, plain_pictures)
        if who == models[0]:
            saved_angle = numpy.array([saved_by_frame[int(index)]["chunk_angle_deg"] for index in plain_rows["frame_index"]])
            saved_twist = numpy.array([saved_by_frame[int(index)]["twist_deg"] for index in plain_rows["frame_index"]])
            print(f"check of the inference path (fold models, unchanged pictures, against the predictions saved by training): direction differs by at most "
                  f"{angle_gap(base_answers[who]['chunk_angle_deg'], saved_angle).max():.4f} degrees, twist by at most {numpy.abs(base_answers[who]['twist_deg'] - saved_twist).max():.5f}")
            head_gap = angle_gap(base_answers[who]["exact_angle_deg"], base_answers[who]["chunk_angle_deg"])
            print(f"the exact step's direction (carried out by the driver) against the chunk's first step (scored by training): they differ by "
                  f"{numpy.median(head_gap):.1f} degrees (median), {numpy.quantile(head_gap, 0.9):.1f} (90th percentile)")
        for shift_x, shift_y in settings["plain_shifts_px"]:
            answers = answers_of(who, plain_rows, shifted_pictures(plain_pictures, shift_x, shift_y))
            plain_table.append(dict({"model": who, "shift_x_160": shift_x, "shift_y_160": shift_y}, **scores(answers, base_answers[who], plain_rows, settings)))
    plain_table = pandas.DataFrame(plain_table)
    print("\n=== plain shifts of the whole picture (px at 160; 4 px = 16 px at 640, the camera's creep and the training augmentation's range) ===")
    print(plain_table.round(3).to_string(index = False))
    plain_table.to_csv(output_folder / "shift_sensitivity_plain.csv", index = False)

    print("\n=== plain shifts by sheet, fold models: share of exact steps within 45 degrees of the demonstrated step ===")
    by_sheet = []
    sheets_of_rows = plain_rows["sheet"].to_numpy()
    for shift_x, shift_y in [[0, 0], [4, 0], [8, 0], [-4, 0], [-8, 0]]:
        answers = answers_of(models[0], plain_rows, shifted_pictures(plain_pictures, shift_x, shift_y))
        gaps = angle_gap(answers["exact_angle_deg"], plain_rows["demonstrated_angle_deg"].to_numpy())
        row = {"shift_x_160": shift_x}
        for sheet in settings["sheets"]:
            row[sheet[-2:]] = float((gaps[sheets_of_rows == sheet] <= 45.0).mean())
        row["twist_mean_abs_deg"] = float(numpy.abs(answers["twist_deg"]).mean())
        by_sheet.append(row)
    print(pandas.DataFrame(by_sheet).round(2).to_string(index = False))

    def paired_change(
            is_right_before: numpy.ndarray,
            is_right_after: numpy.ndarray
    ) -> str:
        """The change of a share between two views of the same rows, in points, with a 95 % bootstrap interval over the rows."""

        generator = numpy.random.default_rng(settings["seed"])
        differences = is_right_after.astype(float) - is_right_before.astype(float)
        draws = generator.integers(0, len(differences), size = (2000, len(differences)))
        means = differences[draws].mean(axis = 1)
        return f"{100.0 * differences.mean():+.1f} points (95 % interval {100.0 * numpy.quantile(means, 0.025):+.1f} to {100.0 * numpy.quantile(means, 0.975):+.1f})"

    creep_lines = []
    for who in models:
        demonstrated = plain_rows["demonstrated_angle_deg"].to_numpy()
        crept = answers_of(who, plain_rows, shifted_pictures(plain_pictures, 4, 0))
        creep_lines.append(f"  {who}: exact steps within 45 degrees, the picture 4 px to the right against unchanged: "
                           + paired_change(angle_gap(base_answers[who]["exact_angle_deg"], demonstrated) <= 45.0, angle_gap(crept["exact_angle_deg"], demonstrated) <= 45.0))
    print("\n=== the camera's creep imitated (the whole picture 4 px to the right at 160), paired on the same rows ===")
    print("\n".join(creep_lines))

    wrist_table = []
    wrist_lines = []
    groups = [("all rows", numpy.ones(len(wrist_rows), dtype = bool)),
              ("content moved up (demonstrated wrist below 13.5)", wrist_shifts < 0),
              ("content moved down (demonstrated wrist above 13.5)", wrist_shifts > 0)]
    for who in models:
        base = answers_of(who, wrist_rows, wrist_pictures)
        demonstrated = wrist_rows["demonstrated_angle_deg"].to_numpy()
        for name, pictures in [("as demonstrated", wrist_pictures), ("held-wrist view (fork kept)", wrist_changed), ("whole picture shifted by the same amount", wrist_plain_shift)]:
            answers = answers_of(who, wrist_rows, pictures)
            for label, is_part in groups:
                part_answers = {key: values[is_part] for key, values in answers.items()}
                part_base = {key: values[is_part] for key, values in base.items()}
                wrist_table.append(dict({"model": who, "view": name, "rows": label, "count": int(is_part.sum())}, **scores(part_answers, part_base, wrist_rows[is_part], settings)))
                if name == "held-wrist view (fork kept)":
                    wrist_lines.append(f"  {who}, {label} ({int(is_part.sum())} rows): exact steps within 45 degrees, held-wrist view against as demonstrated: "
                                       + paired_change(angle_gap(base["exact_angle_deg"][is_part], demonstrated[is_part]) <= 45.0,
                                                       angle_gap(answers["exact_angle_deg"][is_part], demonstrated[is_part]) <= 45.0))
    wrist_table = pandas.DataFrame(wrist_table)
    print(f"\n=== the held-wrist view on {len(wrist_rows)} demonstration rows whose wrist was outside 13.5 plus or minus {settings['wrist_range_deg']:.1f} degrees ===")
    print(wrist_table.round(3).to_string(index = False))
    print("\npaired on the same rows:")
    print("\n".join(wrist_lines))
    wrist_table.to_csv(output_folder / "shift_sensitivity_held_wrist.csv", index = False)

    figure, axes = pyplot.subplots(2, 4, figsize = (13, 5.2))
    order = numpy.argsort(wrist_shifts)
    examples = [order[2], order[len(order) // 6], order[-len(order) // 6], order[-3]]
    for column, example in enumerate(examples):
        axes[0, column].imshow(wrist_pictures[example])
        axes[0, column].set_title(f"demonstrated: wrist {wrist_rows['joint_5_deg'].iloc[example]:.1f}", fontsize = 9)
        axes[1, column].imshow(wrist_changed[example])
        axes[1, column].set_title(f"imitated wrist 13.5: content {wrist_shifts[example]:+d} px", fontsize = 9)
        for row in range(2):
            axes[row, column].axis("off")
    figure.suptitle("The held-wrist view imitated on cached 160 x 120 pictures (the fork kept, everything else moved)", fontsize = 10)
    figure.tight_layout()
    figure.savefig(output_folder / "shift_sensitivity_examples.png", dpi = settings["figure_dpi"])
    pyplot.close(figure)

    figure, axis = pyplot.subplots(1, 1, figsize = (7.5, 4.6))
    for who, colour in zip(models, ["tab:blue", "tab:red"]):
        along_x = plain_table[(plain_table["model"] == who) & (plain_table["shift_y_160"] == 0)].sort_values("shift_x_160")
        along_y = plain_table[(plain_table["model"] == who) & (plain_table["shift_x_160"] == 0)].sort_values("shift_y_160")
        axis.plot(along_x["shift_x_160"], 100.0 * along_x["exact_within_45"], color = colour, marker = "o", label = f"{who}, shifted sideways")
        axis.plot(along_y["shift_y_160"], 100.0 * along_y["exact_within_45"], color = colour, marker = "s", linestyle = "--", label = f"{who}, shifted up or down")
    axis.axvspan(-4.0, 4.0, color = "grey", alpha = 0.15)
    axis.text(-3.8, axis.get_ylim()[0] + 1.0, "training augmentation", fontsize = 8)
    axis.axvline(4.0, color = "black", linestyle = ":")
    axis.text(4.1, axis.get_ylim()[0] + 4.0, "camera on 11.11 against\nsheets 01 to 04 (sideways)", fontsize = 8)
    axis.set_xlabel("shift of the whole picture (px at 160 x 120)")
    axis.set_ylabel("exact steps within 45 degrees of the demonstrated step (%)")
    axis.set_title(f"Direction under shifts of the view ({len(plain_rows)} rows)")
    axis.legend(fontsize = 8)
    axis.grid(alpha = 0.3)
    figure.tight_layout()
    figure.savefig(output_folder / "shift_sensitivity.png", dpi = settings["figure_dpi"])
    pyplot.close(figure)
    print(f"\nsaved {output_folder / 'shift_sensitivity.png'}, shift_sensitivity_examples.png and two csv tables")



if __name__ == "__main__":
    main()
