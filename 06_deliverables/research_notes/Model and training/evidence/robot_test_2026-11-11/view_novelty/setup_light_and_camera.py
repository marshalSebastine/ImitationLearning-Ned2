"""
Set-up checks of the robot test of 11.11.26 that do not need a model: light and colour, sharpness, the camera's place
on its mount, the picture files themselves, and where each run started.

    light and colour    the brightness, contrast and sharpness of every robot picture, measured exactly as the
                        training index measured them (training_dataset.picture_statistics), and the colour of the
                        white paper (the median colour of the bright, nearly grey pixels), against the training
                        pictures by lighting note and recording day;
    camera on its mount the shift of the picture's bottom strip (the robot's own hand and block, which look the same
                        whenever the wrist and hand have the same angles) against the recording tool's reference
                        picture, by the recording tool's own phase-correlation check; for the demonstrations'
                        first pictures and for every robot picture (the wrist and hand were held at the same angles);
    picture files       size and JPEG quantisation tables of the robot pictures against training pictures;
    where a run started the tip's place, the forearm twist and the picture's heading at step 0 against the
                        demonstrations' fixed start.

Nothing connects to the robot. Run from this folder (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe setup_light_and_camera.py
Output: setup_light_and_camera.out.txt, setup_light_and_camera_per_picture.csv, figures/setup_light_and_colour.png,
figures/setup_camera_strips.png
"""

import math
from typing import Dict, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot
from PIL import Image

from common import (EVIDENCE_FOLDER, PLOT_STYLE, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, run_picture_paths, run_steps,
                    save_montage, small_rgb_picture, styled_axes, training_frames, training_pictures)
from config import RECORDING_PARAMETERS
from neighbours import percentile_rank
from record_episodes import camera_shift_px
from training_dataset import merged_parameters, picture_statistics

SETUP_PARAMETERS = {
    # paper pixels of a 160 x 120 RGB picture: bright and nearly grey (the table is darker, the robot's blue and the
    # puck's red are strongly coloured)
    "paper_value_at_least": 150,
    "paper_spread_at_most": 80,
    "chunk_pictures": 1000,
    "camera_reference": "forearm_cover_tape_bare_camera_x159_y-15mm.jpg",
    "standard_start_kinds": ["STD", "OL1", "OL1.5", "OR1", "OR1.5", "TL10", "TL15", "TR10", "TR15", "SQP", "SQL"],
    "demonstration_start": {"tip_x_m": 0.140, "tip_y_m": -0.015, "joint_4_deg": 0.0},
    # the strips drawn one above the other: guide lines at these columns of the 320 px wide strip, and these runs
    "strip_guides_px": [110, 210],
    "strip_runs": [2, 5, 7, 9]
}



def colour_statistics(
        small_pictures: numpy.ndarray,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Measure the overall colour and the colour of the white paper in a stack of 160 x 120 pictures.

    Parameters
    ----------
    small_pictures : numpy.ndarray
        (pictures, 120, 160, 3), uint8 RGB.
    parameters : dict
        "paper_value_at_least" and "paper_spread_at_most".

    Returns
    -------
    pandas.DataFrame
        Per picture: "mean_red", "mean_green", "mean_blue" (whole picture), "paper_red", "paper_green", "paper_blue"
        (median over the paper pixels), "paper_share" and "paper_red_over_blue".
    """

    pictures = numpy.asarray(small_pictures).astype(numpy.int16)
    largest, smallest = pictures.max(axis = 3), pictures.min(axis = 3)
    is_paper = (largest >= parameters["paper_value_at_least"]) & (largest - smallest <= parameters["paper_spread_at_most"])
    rows = []
    for picture, mask in zip(pictures, is_paper):
        paper = picture[mask] if mask.sum() >= 50 else numpy.full((1, 3), numpy.nan)
        red, green, blue = numpy.median(paper, axis = 0)
        rows.append({"mean_red": picture[:, :, 0].mean(), "mean_green": picture[:, :, 1].mean(), "mean_blue": picture[:, :, 2].mean(),
                     "paper_red": red, "paper_green": green, "paper_blue": blue, "paper_share": mask.mean(),
                     "paper_red_over_blue": red / blue if blue else numpy.nan})
    return pandas.DataFrame(rows)



def robot_picture_table(parameters: Dict) -> pandas.DataFrame:
    """Measure every robot picture: the training index's statistics, the colours and the bottom strip's shift."""

    data_parameters = merged_parameters()
    reference_bytes = (VIEW_NOVELTY_PARAMETERS["raw_folder"] / "camera_check" / "camera_reference" / parameters["camera_reference"]).read_bytes()
    region = RECORDING_PARAMETERS["camera_check_region"]
    tables = []
    for run_number in ROBOT_RUNS:
        rows, small = [], []
        for step, path in enumerate(run_picture_paths(run_number)):
            picture_bytes = path.read_bytes()
            statistics = picture_statistics(cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR), data_parameters)
            shift_x, shift_y = camera_shift_px(reference_bytes, picture_bytes, region)
            rows.append({"run": run_number, "step": step, **statistics, "strip_shift_x_px": shift_x, "strip_shift_y_px": shift_y, "file_bytes": len(picture_bytes)})
            small.append(small_rgb_picture(picture_bytes))
        tables.append(pandas.concat([pandas.DataFrame(rows), colour_statistics(numpy.stack(small), parameters)], axis = 1))
    return pandas.concat(tables).reset_index(drop = True)



def training_picture_table(parameters: Dict) -> pandas.DataFrame:
    """Join frames.csv (brightness, sharpness, lighting) with the colours measured from the 160 x 120 cache."""

    frames = training_frames()
    cache = training_pictures()
    colours = []
    for start in range(0, len(cache), parameters["chunk_pictures"]):
        colours.append(colour_statistics(cache[start:start + parameters["chunk_pictures"]], parameters))
    return pandas.concat([frames[["frame_index", "episode_id", "sheet", "session_date", "lighting", "row_in_episode", "start_kind", "brightness_mean",
                                  "brightness_std", "red_share", "dark_share_outside_robot", "sharpness_measured", "frame_path", "joint_5", "joint_6"]],
                          pandas.concat(colours).reset_index(drop = True)], axis = 1)



def say_light_and_colour(
        robot: pandas.DataFrame,
        training: pandas.DataFrame,
        log: OutputLog
) -> None:
    """Say the robot pictures' light, colour and sharpness against the training pictures."""

    measures = [("brightness_mean", "brightness (grey 0 to 255)"), ("brightness_std", "contrast (spread of grey)"), ("sharpness_measured", "sharpness (variance of the Laplacian)"),
                ("paper_red", "paper colour, red"), ("paper_green", "paper colour, green"), ("paper_blue", "paper colour, blue"),
                ("paper_red_over_blue", "paper red over blue (1 = neutral white, lower = bluer)")]
    log.say("LIGHT, COLOUR AND SHARPNESS (measured: robot pictures from their JPEGs, training pictures from frames.csv and the 160 x 120 cache)")
    log.say("")
    log.say("Training pictures by recording day and lighting note (median; 5 % to 95 % in brackets)")
    header = "group                          pictures  " + "  ".join(f"{name:>24s}" for name, _ in measures)
    log.say(header)
    groups = [(f"{day} {lighting}", group) for (day, lighting), group in training.groupby(["session_date", "lighting"])] + [("ALL TRAINING PICTURES", training)]
    for label, group in groups:
        cells = [f"{group[name].median():7.2f} [{group[name].quantile(0.05):6.2f},{group[name].quantile(0.95):7.2f}]" for name, _ in measures]
        log.say(f"{label:30s} {len(group):8d}  " + "  ".join(f"{cell:>24s}" for cell in cells))
    log.say("")
    log.say("Robot runs of 11.11.26 (median of the run; in brackets the place of that median among ALL training pictures, percentile 0 to 100)")
    log.say(header)
    for run_number in ROBOT_RUNS:
        group = robot[robot["run"] == run_number]
        cells = [f"{group[name].median():7.2f} [pct {percentile_rank(numpy.array([group[name].median()]), training[name].dropna().to_numpy())[0]:3.0f}]" for name, _ in measures]
        log.say(f"run {run_number:2d} ({ROBOT_RUNS[run_number]['sheet']}){'':16s} {len(group):8d}  " + "  ".join(f"{cell:>24s}" for cell in cells))
    log.say("")
    for name, label in measures:
        low, high = training[name].quantile(0.005), training[name].quantile(0.995)
        outside = robot[(robot[name] < low) | (robot[name] > high)]
        log.say(f"{label}: training pictures span {training[name].min():.2f} to {training[name].max():.2f} (middle 99 %: {low:.2f} to {high:.2f}); "
                f"robot pictures span {robot[name].min():.2f} to {robot[name].max():.2f}; {len(outside)} of {len(robot)} robot pictures lie outside the training pictures' middle 99 %"
                + (f" (runs {sorted(outside['run'].unique())})" if len(outside) else ""))



def say_camera_shift(
        robot: pandas.DataFrame,
        training: pandas.DataFrame,
        parameters: Dict,
        log: OutputLog
) -> None:
    """Say how far the picture's bottom strip (the robot's own hand) sits from the reference, for demonstrations and robot runs."""

    reference_path = VIEW_NOVELTY_PARAMETERS["raw_folder"] / "camera_check" / "camera_reference" / parameters["camera_reference"]
    reference_bytes = reference_path.read_bytes()
    region = RECORDING_PARAMETERS["camera_check_region"]
    first = training[(training["row_in_episode"] == 0) & training["start_kind"].isin(parameters["standard_start_kinds"])].copy()
    shifts = [camera_shift_px(reference_bytes, (VIEW_NOVELTY_PARAMETERS["raw_folder"] / path).read_bytes(), region) for path in first["frame_path"]]
    first["strip_shift_x_px"] = [shift[0] for shift in shifts]
    first["strip_shift_y_px"] = [shift[1] for shift in shifts]

    log.say("")
    log.say("THE CAMERA ON ITS MOUNT (shift of the bottom strip, the robot's own hand and block, against the recording tool's reference picture")
    log.say(f"{reference_path.name} of 6.11.26, in pixels of the 640 x 480 picture: x to the right, y downwards; the recording tool warned above "
            f"{RECORDING_PARAMETERS['camera_shift_warning_px']} px; training augmentation shifts by up to 4 px at 160 px wide = 16 px here)")
    log.say("")
    log.say("Demonstrations, first picture of every standard-start episode (wrist 13.5, hand -6.1 degrees, as held in the robot runs)")
    log.say("recording day   episodes   shift x: median  [min, max]     shift y: median  [min, max]")
    for day, group in first.groupby("session_date"):
        log.say(f"{day}  {len(group):9d}   {group['strip_shift_x_px'].median():15.1f}  [{group['strip_shift_x_px'].min():5.1f},{group['strip_shift_x_px'].max():5.1f}]"
                f"   {group['strip_shift_y_px'].median():15.1f}  [{group['strip_shift_y_px'].min():5.1f},{group['strip_shift_y_px'].max():5.1f}]")
    last_day = first[first["session_date"] == first["session_date"].max()]
    log.say("")
    log.say("Robot runs of 11.11.26, every picture (the wrist and hand were held, so the strip shows the same parts in every picture)")
    log.say("run   pictures   shift x: median  [5 %, 95 %]     shift y: median  [5 %, 95 %]    against the last recording day's median: x, y")
    for run_number in ROBOT_RUNS:
        group = robot[robot["run"] == run_number]
        log.say(f"{run_number:3d}  {len(group):9d}   {group['strip_shift_x_px'].median():15.1f}  [{group['strip_shift_x_px'].quantile(0.05):5.1f},{group['strip_shift_x_px'].quantile(0.95):5.1f}]"
                f"   {group['strip_shift_y_px'].median():15.1f}  [{group['strip_shift_y_px'].quantile(0.05):5.1f},{group['strip_shift_y_px'].quantile(0.95):5.1f}]"
                f"    {group['strip_shift_x_px'].median() - last_day['strip_shift_x_px'].median():+6.1f}, {group['strip_shift_y_px'].median() - last_day['strip_shift_y_px'].median():+6.1f}")
    log.say(f"all robot pictures: shift x median {robot['strip_shift_x_px'].median():.1f}, shift y median {robot['strip_shift_y_px'].median():.1f}; "
            f"last recording day ({last_day['session_date'].iloc[0]}): x {last_day['strip_shift_x_px'].median():.1f}, y {last_day['strip_shift_y_px'].median():.1f}")



def say_picture_files(
        training: pandas.DataFrame,
        log: OutputLog
) -> None:
    """Say whether the robot's JPEGs are the same kind of file as the training JPEGs (size and quantisation tables)."""

    def described(path) -> tuple:
        with Image.open(path) as picture:
            tables = tuple(tuple(table) for _, table in sorted(picture.quantization.items()))
            return picture.size, picture.mode, hash(tables), sum(tables[0]) if tables else None

    log.say("")
    log.say("THE PICTURE FILES (size, colour mode, and the JPEG quantisation tables, which fix the compression strength)")
    sample = training.groupby("session_date").head(40)
    training_kinds = pandas.Series([described(VIEW_NOVELTY_PARAMETERS["raw_folder"] / path) for path in sample["frame_path"]]).value_counts()
    robot_kinds = pandas.Series([described(path) for run_number in ROBOT_RUNS for path in run_picture_paths(run_number)]).value_counts()
    for label, kinds in (("training pictures (40 per recording day)", training_kinds), ("robot pictures (all)", robot_kinds)):
        for (size, mode, table_hash, table_sum), count in kinds.items():
            log.say(f"  {label}: {count} files of {size[0]} x {size[1]}, mode {mode}, luminance table sum {table_sum}, tables id {table_hash % 100000}")
    log.say(f"  same kinds of file in both: {set(training_kinds.index) == set(robot_kinds.index)}")



def say_starts(
        parameters: Dict,
        log: OutputLog
) -> None:
    """Say where each run started against the demonstrations' fixed start (measured from steps.csv and frames.csv)."""

    start = parameters["demonstration_start"]
    log.say("")
    log.say("WHERE EACH RUN STARTED (step 0 of steps.csv; the demonstrations' standard starts all began with the tip at "
            f"x {100 * start['tip_x_m']:.1f} cm, y {100 * start['tip_y_m']:.1f} cm, forearm twist 0 degrees: measured, frames.csv row 0)")
    log.say("run  sheet     tip x cm  tip y cm   from the demonstrations' start: forward cm, left cm, distance cm   forearm twist deg   picture heading deg")
    for run_number in ROBOT_RUNS:
        row = run_steps(run_number).iloc[0]
        forward, left = 100 * (row["tip_x"] - start["tip_x_m"]), 100 * (row["tip_y"] - start["tip_y_m"])
        log.say(f"{run_number:3d}  {ROBOT_RUNS[run_number]['sheet']:8s}  {100 * row['tip_x']:8.1f}  {100 * row['tip_y']:8.1f}   {forward:+38.1f}, {left:+5.1f}, {math.hypot(forward, left):5.1f}"
                f"   {math.degrees(row['joint_4']):+17.1f}   {row['picture_heading_deg']:+19.1f}")



def draw_light_and_colour(
        robot: pandas.DataFrame,
        training: pandas.DataFrame,
        figure_path
) -> None:
    """Draw brightness and paper colour: the training pictures' spread as a grey band, each robot run as a dot with its range."""

    measures = [("brightness_mean", "brightness (grey, 0 to 255)"), ("paper_red_over_blue", "paper colour: red over blue\n(1 = neutral white, lower = bluer)"),
                ("sharpness_measured", "sharpness\n(variance of the Laplacian)")]
    figure, grid = pyplot.subplots(1, 3, figsize = (13, 4.2), facecolor = PLOT_STYLE["surface"])
    for axes, (name, label) in zip(grid, measures):
        styled_axes(axes)
        axes.axhspan(training[name].quantile(0.05), training[name].quantile(0.95), color = PLOT_STYLE["neutral"], alpha = 0.18, linewidth = 0)
        axes.axhline(training[name].median(), color = PLOT_STYLE["neutral"], linewidth = 1.0)
        for run_number in ROBOT_RUNS:
            values = robot.loc[robot["run"] == run_number, name]
            axes.vlines(run_number, values.quantile(0.05), values.quantile(0.95), color = PLOT_STYLE["series"][0], linewidth = 2.0)
            axes.plot([run_number], [values.median()], marker = "o", markersize = 6.5, color = PLOT_STYLE["series"][0], markeredgecolor = PLOT_STYLE["surface"], markeredgewidth = 1.0)
        axes.set_xticks(list(ROBOT_RUNS))
        axes.set_xlabel("robot run", fontsize = 8, color = PLOT_STYLE["ink_secondary"])
        axes.set_title(label, fontsize = 9, color = PLOT_STYLE["ink"], loc = "left")
        axes.text(10.6, training[name].median(), "training\nmedian", fontsize = 7, color = PLOT_STYLE["ink_secondary"], va = "center", ha = "left")
        axes.set_xlim(0.3, 12.2)
    figure.suptitle("Light, colour and sharpness of the robot pictures (blue: median and 5 % to 95 % of each run) against the training pictures (grey band: their 5 % to 95 %)",
                    fontsize = 10, color = PLOT_STYLE["ink"], x = 0.01, ha = "left")
    figure.tight_layout(rect = (0, 0, 1, 0.94))
    figure.savefig(figure_path, dpi = 110, facecolor = PLOT_STYLE["surface"])
    pyplot.close(figure)



def draw_camera_strips(
        training: pandas.DataFrame,
        parameters: Dict,
        figure_path
) -> None:
    """
    Draw the picture's bottom strip (the robot's own hand and block) for the reference picture, one first picture of
    every recording day and the first pictures of four robot runs, one above the other with the same guide lines, so
    the camera's creep on its mount can be seen directly.
    """

    top, bottom, left, right = RECORDING_PARAMETERS["camera_check_region"]

    def strip(picture: numpy.ndarray, label: str) -> numpy.ndarray:
        height, width = picture.shape[:2]
        crop = picture[int(top * height):int(bottom * height), int(left * width):int(right * width)]
        crop = cv2.resize(crop, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        for guide in parameters["strip_guides_px"]:
            cv2.line(crop, (2 * guide, 0), (2 * guide, crop.shape[0]), (0, 255, 255), 1)
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 18), (0, 0, 0), -1)
        cv2.putText(crop, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return crop

    reference_path = VIEW_NOVELTY_PARAMETERS["raw_folder"] / "camera_check" / "camera_reference" / parameters["camera_reference"]
    strips = [strip(cv2.imread(str(reference_path)), "reference picture of the recording tool, 6.11.26")]
    first = training[(training["row_in_episode"] == 0) & training["start_kind"].isin(parameters["standard_start_kinds"])]
    for day, group in first.groupby("session_date"):
        row = group.iloc[len(group) // 2]
        strips.append(strip(cv2.imread(str(VIEW_NOVELTY_PARAMETERS["raw_folder"] / row["frame_path"])), f"demonstration, first picture, {day} ({row['sheet']})"))
    for run_number in parameters["strip_runs"]:
        strips.append(strip(cv2.imread(str(run_picture_paths(run_number)[0])), f"robot run {run_number}, step 0, 11.11.26"))
    save_montage(figure_path, numpy.vstack(strips))



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(SETUP_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "setup_light_and_camera.out.txt")
    robot = robot_picture_table(settings)
    training = training_picture_table(settings)
    say_light_and_colour(robot, training, log)
    say_camera_shift(robot, training, settings, log)
    say_picture_files(training, log)
    say_starts(settings, log)
    robot.round(4).to_csv(EVIDENCE_FOLDER / "setup_light_and_camera_per_picture.csv", index = False)
    figure_folder = EVIDENCE_FOLDER / "figures"
    figure_folder.mkdir(exist_ok = True)
    draw_light_and_colour(robot, training, figure_folder / "setup_light_and_colour.png")
    draw_camera_strips(training, settings, figure_folder / "setup_camera_strips.png")
    log.say("")
    log.say("written: setup_light_and_camera_per_picture.csv, figures/setup_light_and_colour.png, figures/setup_camera_strips.png")
    log.save()



if __name__ == "__main__":
    main()
