"""
Shared helpers of the view-novelty check of the first robot test (11.11.26): where the files are, the ten robot runs,
how a logged robot picture becomes network input (exactly as the robot driver does it), and how a model's picture
descriptors are read out.

Nothing here connects to the robot: only saved pictures, saved logs and saved models are read.

The question of this folder: were the robot's pictures like the training pictures? A picture descriptor is taken from
the model itself (what the prediction head sees), and every robot picture is compared with its nearest training
pictures; the same distance among the demonstrations themselves is the yardstick.

Run any script from this folder with the project's Python:
    D:/AI_Project/.venv/Scripts/python.exe <script>.py
The large feature tables (about 2 GB) are kept outside git, in the folder named by the environment variable
VIEW_NOVELTY_CACHE (the system's temporary folder when it is not set); extract_features.py writes them and the other
scripts say so when one is missing.

The scripts, in the order they build on each other (each prints to <script>.out.txt; figures go to figures/):
    contact_sheets.py, failure_strips.py    every picture of every run; the enlarged strips the failure steps were read
                                            off (FAILURE_STEPS below)
    extract_features.py                     model_e, then fold without_path_01 to 06 with --all-pictures (the slow part,
                                            two to five minutes each on two threads)
    novelty_model_e.py                      the distance of every robot picture to the training pictures, model E
    novelty_fold_models.py                  the same with models that never saw the sheet (the fair yardstick)
    key_frames.py                           the nearest demonstration pictures and what they did against what was asked
    setup_light_and_camera.py               light, colour, sharpness, the camera's creep on its mount, where runs started
    camera_shift_counterfactual.py          what the camera's creep does to model E's distances and answers
    camera_pose_check.py                    the camera's height, tilt and roll in the runs against the demonstrations
    hoodie_check.py                         the blue hoodie: where it was, painted out, painted in
    sheet02_square_check.py                 run 3: the sheet lying square
    run1_start_offset.py                    runs 1 and 2: the puck beside the line
    puck_beside_tip.py                      where the puck sat beside the tip's line of travel
    stale_pictures.py, picture_lag_check.py were consecutive pictures identical, or late
    per_run_summary.py                      one block of numbers per run, gathered from the tables above
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy
import pandas
import torch

EVIDENCE_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = EVIDENCE_FOLDER.parents[5]
CODE_FOLDER = ROOT_FOLDER / "02_code"
if str(CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(CODE_FOLDER))

from config import MODEL_MODE_PARAMETERS, TRAINING_DATA_PARAMETERS

VIEW_NOVELTY_PARAMETERS = {
    "cache_folder": Path(os.environ.get("VIEW_NOVELTY_CACHE", Path(tempfile.gettempdir()) / "view_novelty_cache")),
    "model_runs_folder": Path(MODEL_MODE_PARAMETERS["run_folder"]),
    "processed_folder": Path(TRAINING_DATA_PARAMETERS["processed_folder"]),
    "raw_folder": Path(TRAINING_DATA_PARAMETERS["raw_folder"]),
    "trained_models_folder": ROOT_FOLDER / "04_runs" / "gpu_ki014",
    "model_e_folder": "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets",
    "model_e_run_name": "resnet18_model_4_chunk_stop_25_all_sheets",
    "fold_folder": "2026-11-10_resnet18_chunk_stop_weight_25",
    "fold_run_name": "resnet18_chunk_stop_weight_25",
    "threads": 2,
    "batch_size": 128
}

# the ten runs of 11.11.26 as the brief lists them
ROBOT_RUNS = {
    1: {"folder": "2026-11-11_172021_check_no_puck", "model": "A", "sheet": "path_01", "note": "start 1.5 cm left; reached the end"},
    2: {"folder": "2026-11-11_172421_E_sheet01_run1", "model": "E", "sheet": "path_01", "note": "on the line; stopped at the end"},
    3: {"folder": "2026-11-11_172743_E_sheet02_run1", "model": "E", "sheet": "path_02", "note": "sheet lying straight; steered right; lost"},
    4: {"folder": "2026-11-11_173036_E_sheet02_run2", "model": "E", "sheet": "path_02", "note": "whole arc; stopped at the end"},
    5: {"folder": "2026-11-11_173549_E_sheet03_run1", "model": "E", "sheet": "path_03", "note": "overshot the first bend (hoodie?)"},
    6: {"folder": "2026-11-11_174108_E_sheet04_run1", "model": "E", "sheet": "path_04", "note": "stuck at the hairpin (hoodie)"},
    7: {"folder": "2026-11-11_174448_E_sheet05_run1", "model": "E", "sheet": "path_05", "note": "whole alpha loop; stopped at the end"},
    8: {"folder": "2026-11-11_174936_E_sheet06_run1", "model": "E", "sheet": "path_06", "note": "left the dashed loop early"},
    9: {"folder": "2026-11-11_175203_E_sheet03_run2_no_hoodie", "model": "E", "sheet": "path_03", "note": "as run 5, nothing blue in view"},
    10: {"folder": "2026-11-11_175544_B_team5_sheet03", "model": "B", "sheet": "path_03", "note": "team of five; same overshoot"}
}



# Where each failed run went wrong, read by eye off the enlarged strips (failure_strips.py, figures/failure_strip_run_NN.png),
# good to about one step: "first_wrong_decision" is the first picture whose asked direction clearly leaves the line
# ahead, "puck_off_line" the first picture in which the puck no longer touches the line.
FAILURE_STEPS = {
    3: {"first_wrong_decision": 0, "puck_off_line": 5, "reason": "asked 28 degrees right at step 0 while the arc leaves to the upper left"},
    5: {"first_wrong_decision": 2, "puck_off_line": 5, "reason": "asked 22 then 54 degrees right from step 2 while the line bends left"},
    6: {"first_wrong_decision": 14, "puck_off_line": 16, "reason": "asked 40 to 62 degrees right from step 14 at the sharp corner; the tip slid past the puck"},
    8: {"first_wrong_decision": 7, "puck_off_line": 11, "reason": "asked 19 to 38 degrees left from step 7, past the crossing, while the loop goes on straight and then right"},
    9: {"first_wrong_decision": 2, "puck_off_line": 5, "reason": "asked 31 to 44 degrees right from step 2 while the line bends left"},
    10: {"first_wrong_decision": 2, "puck_off_line": 5, "reason": "asked 44 to 58 degrees right from step 2 while the line bends left"}
}



class OutputLog:

    def __init__(self, output_path: Path):
        """
        Collect every printed line so that the script's output is also saved as a text file beside it.

        Parameters
        ----------
        output_path : pathlib.Path
            The text file written by ``save``.
        """

        self.output_path = Path(output_path)
        self.lines = []


    def say(self, text: str = "") -> None:
        """Print ``text`` and keep it for the text file."""

        print(text, flush = True)
        self.lines.append(str(text))


    def save(self) -> None:
        """Write everything said so far to the text file (UTF-8)."""

        self.output_path.write_text("\n".join(self.lines) + "\n", encoding = "utf-8")



def cache_folder() -> Path:
    """Return the folder of the large feature tables, made when missing."""

    folder = Path(VIEW_NOVELTY_PARAMETERS["cache_folder"])
    folder.mkdir(parents = True, exist_ok = True)
    return folder



def run_folder(run_number: int) -> Path:
    """Return the log folder of one of the ten robot runs."""

    return VIEW_NOVELTY_PARAMETERS["model_runs_folder"] / ROBOT_RUNS[run_number]["folder"]



def run_steps(run_number: int) -> pandas.DataFrame:
    """Return steps.csv of a robot run, one row per picture."""

    return pandas.read_csv(run_folder(run_number) / "steps.csv")



def run_picture_paths(run_number: int) -> List[Path]:
    """Return the pictures of a robot run in step order."""

    steps = run_steps(run_number)
    return [run_folder(run_number) / frame for frame in steps["frame"]]



def small_rgb_picture(
        picture_bytes: bytes,
        picture_size: Optional[List[int]] = None
) -> numpy.ndarray:
    """
    Shrink a camera JPEG exactly as the training cache and the robot driver do.

    The picture is decoded, turned from BGR to RGB and shrunk with area averaging (``training_dataset.py`` and
    ``model_mode.ModelPolicy.picture_input`` use the same three steps).

    Parameters
    ----------
    picture_bytes : bytes
        The camera's JPEG.
    picture_size : list of int, optional
        ``[width, height]``; 160 x 120 by default.

    Returns
    -------
    numpy.ndarray
        (height, width, 3), uint8 RGB.
    """

    width, height = picture_size or [160, 120]
    picture = cv2.imdecode(numpy.frombuffer(picture_bytes, dtype = numpy.uint8), cv2.IMREAD_COLOR)
    if picture is None:
        raise ValueError("the picture could not be decoded")
    return cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (width, height), interpolation = cv2.INTER_AREA)



def run_small_pictures(run_number: int) -> numpy.ndarray:
    """Return every picture of a robot run at 160 x 120, RGB, in step order."""

    return numpy.stack([small_rgb_picture(path.read_bytes()) for path in run_picture_paths(run_number)])



def training_pictures() -> numpy.ndarray:
    """Return the 160 x 120 training cache (all 18,522 pictures in ``frame_index`` order), memory-mapped."""

    return numpy.load(VIEW_NOVELTY_PARAMETERS["processed_folder"] / "images_wide_160.npy", mmap_mode = "r")



def training_frames() -> pandas.DataFrame:
    """Return frames.csv, one row per training picture, in ``frame_index`` order."""

    frames = pandas.read_csv(VIEW_NOVELTY_PARAMETERS["processed_folder"] / "frames.csv", low_memory = False)
    return frames.sort_values("frame_index").reset_index(drop = True)



def load_policy(
        run_name: str,
        model_folder: str,
        checkpoint_name: str = "all_sheets"
):
    """
    Load a saved model through the robot driver's own loader, so the pictures are prepared as on the robot.

    Parameters
    ----------
    run_name : str
        The recipe's name in ``TRAINING_RUNS``.
    model_folder : str
        The folder of the trained model inside "trained_models_folder".
    checkpoint_name : str, optional
        "all_sheets" or a fold name such as "without_path_03".

    Returns
    -------
    model_mode.ModelPolicy
        The loaded model; ``policy.models[0]`` is the network.
    """

    # importing model_mode does not connect to anything: the robot link is only built inside robot_run
    from model_mode import ModelPolicy

    folder = VIEW_NOVELTY_PARAMETERS["trained_models_folder"] / model_folder
    policy = ModelPolicy(run_name, folder / "checkpoints" / f"{checkpoint_name}.pt", folder / "metrics.json")
    torch.set_num_threads(VIEW_NOVELTY_PARAMETERS["threads"])
    return policy



def picture_descriptors(
        policy,
        small_pictures: numpy.ndarray,
        log: Optional[OutputLog] = None
) -> Dict[str, numpy.ndarray]:
    """
    Read a model's descriptors and answers for a stack of 160 x 120 pictures.

    Three descriptors are taken from the network itself: "grid" is the backbone's feature map averaged over the
    3 x 4 cells the prediction head sees (6,144 numbers, it keeps where things are in the picture), "mean" is the
    feature map averaged over the whole picture (512 numbers, it forgets where things are) and "hidden" is the
    head's shared hidden layer (256 numbers, the last stage before the answers).

    Parameters
    ----------
    policy : model_mode.ModelPolicy
        From ``load_policy``.
    small_pictures : numpy.ndarray
        (pictures, 120, 160, 3), uint8 RGB.
    log : OutputLog, optional
        Progress is said here when given.

    Returns
    -------
    dict of str to numpy.ndarray
        "grid", "mean", "hidden" (float32), "exact" (forward cm, left cm, twist, lean, hand), "chunk" (when the
        model has one) and "stop_probability".
    """

    from train_policy import augmented_pictures

    model = policy.models[0]
    batch_size = VIEW_NOVELTY_PARAMETERS["batch_size"]
    collected = {"grid": [], "mean": [], "hidden": [], "exact": [], "chunk": [], "stop_probability": []}
    with torch.inference_mode():
        for start in range(0, len(small_pictures), batch_size):
            # a copy, because the training cache is a read-only memory map
            batch = torch.from_numpy(numpy.array(small_pictures[start:start + batch_size]))
            inputs, _ = augmented_pictures(batch, policy.settings, numpy.random.default_rng(0), is_training = False)
            feature_map = model.backbone.forward_features(inputs)
            grid = torch.nn.functional.adaptive_avg_pool2d(feature_map, model.pooling_grid).flatten(1)
            head = model.head
            hidden = head.shared((grid.float() - head.input_mean) / head.input_standard_deviation)
            collected["grid"].append(grid.numpy().astype(numpy.float32))
            collected["mean"].append(feature_map.mean(dim = (2, 3)).numpy().astype(numpy.float32))
            collected["hidden"].append(hidden.numpy().astype(numpy.float32))
            collected["exact"].append(head.outputs["exact"](hidden).numpy())
            if "chunk" in head.outputs:
                collected["chunk"].append(head.outputs["chunk"](hidden).numpy())
            collected["stop_probability"].append(torch.sigmoid(head.outputs["stop"](hidden)).numpy()[:, 0])
            if log is not None and (start // batch_size) % 20 == 0:
                log.say(f"    pictures {start} of {len(small_pictures)}")
    return {name: numpy.concatenate(parts) for name, parts in collected.items() if parts}



def unit_rows(features: numpy.ndarray) -> numpy.ndarray:
    """Return the rows of ``features`` scaled to length 1, so a dot product is the cosine similarity."""

    features = numpy.asarray(features, dtype = numpy.float32)
    return features / numpy.maximum(numpy.linalg.norm(features, axis = 1, keepdims = True), 1e-12)



def trained_frames(run_name: str) -> pandas.DataFrame:
    """
    Return the rows a named run trained on, with the labels as it saw them (after the audit's cleaning).

    Parameters
    ----------
    run_name : str
        A key of ``TRAINING_RUNS``, for example the recipe of model E.

    Returns
    -------
    pandas.DataFrame
        From ``training_targets.load_training_frames`` with the run's own target settings.
    """

    from config import TRAINING_TARGET_PARAMETERS
    from train_policy import merged_run_settings
    from training_targets import load_training_frames

    settings = merged_run_settings(run_name)
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    return load_training_frames(target_parameters)



def load_descriptors(name: str) -> Dict[str, numpy.ndarray]:
    """
    Read a descriptor table written by extract_features.py; the half-precision "grid" table comes back as float32.

    Parameters
    ----------
    name : str
        The file's name in the cache folder without ending, for example "model_e_training".

    Returns
    -------
    dict of str to numpy.ndarray
        The saved arrays.
    """

    path = cache_folder() / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run extract_features.py first (see its docstring)")
    with numpy.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["grid"] = arrays["grid"].astype(numpy.float32)
    return arrays



# Colours of every figure in this folder (a validated palette: blue, orange and aqua stay apart for colour-blind readers;
# red is kept for "went wrong"); text is never drawn in a data colour.
PLOT_STYLE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "went_wrong": "#d03b3b",
    "neutral": "#898781"
}



def styled_axes(axes) -> None:
    """Give a matplotlib axes the quiet look used in this folder: hairline grid, no box, muted ticks."""

    axes.set_facecolor(PLOT_STYLE["surface"])
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(PLOT_STYLE["axis"])
        axes.spines[side].set_linewidth(0.8)
    axes.grid(True, color = PLOT_STYLE["grid"], linewidth = 0.6)
    axes.set_axisbelow(True)
    axes.tick_params(colors = PLOT_STYLE["ink_secondary"], labelsize = 8, length = 3, width = 0.6)



def save_montage(
        figure_path,
        picture_bgr: numpy.ndarray
) -> None:
    """
    Save a montage of camera pictures as a PNG with a palette of 256 colours.

    The camera pictures hold few colours (paper, table, line, puck, the robot's blue), so the palette loses nothing
    that matters to the eye and the file is about a quarter of the full-colour PNG.

    Parameters
    ----------
    figure_path : pathlib.Path
        Where the PNG is written.
    picture_bgr : numpy.ndarray
        The montage as OpenCV holds it (BGR).
    """

    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2RGB))
    image.quantize(colors = 256, method = Image.Quantize.MEDIANCUT, dither = Image.Dither.NONE).save(str(figure_path), optimize = True)
