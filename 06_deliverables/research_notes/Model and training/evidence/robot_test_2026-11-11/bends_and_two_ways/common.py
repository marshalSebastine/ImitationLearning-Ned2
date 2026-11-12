"""
Shared helpers of the analysis "sharp bends, hairpins and two ways to go" (robot test of 11.11.26, key bends_and_two_ways).

Read-only on the project: nothing here connects to the robot, trains a model or writes outside this folder (the cache of
model outputs goes to the computer's temporary folder and is rebuilt when missing). The demonstrations are read through
``training_targets.load_training_frames`` with the audit's cleaning, so the rows are exactly the 18,032 pictures the
models of 10.11.26 were trained and scored on.

The kind of place of a picture comes from the LINE, not from the demonstrator's motion: the audit of 9.11.26 fitted every
episode's puck positions to the traced line (copied to inputs/audit_line_positions.csv, good to about 1 to 2 cm), and
the line's own turning within "place_half_window_cm" before and after the puck's place gives the kind:
    straight      under 15 degrees of turning within 6 cm of line (the absolute turning, so an S-bend counts in full);
    gentle bend   15 to 60 degrees;
    sharp bend    60 degrees or more, without being a hairpin;
    hairpin       the line's direction itself changes by 150 degrees or more within the 6 cm (the audit's rule for a
                  U-turn; only the two hairpins of sheet 04 reach it);
    line end      the puck within "line_end_cm" of the end cap it is pushed to, whatever the line does there.

Settings: ANALYSIS_PARAMETERS below; every script merges its own overrides over them.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
import pandas
import torch

ANALYSIS_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = ANALYSIS_FOLDER.parents[5]
CODE_FOLDER = ROOT_FOLDER / "02_code"
sys.path.insert(0, str(CODE_FOLDER))

from config import MODEL_MODE_PARAMETERS, TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import PictureToMoveModel, augmented_pictures, merged_run_settings
from training_probes import design_outputs
from training_targets import chunk_targets, load_training_frames

PLACE_NAMES = ["straight", "gentle bend", "sharp bend", "hairpin", "line end"]
MOVING_ACTIONS = ["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"]

ANALYSIS_PARAMETERS = {
    "fold_runs_folder": ROOT_FOLDER / "04_runs" / "gpu_ki014",
    "model_runs_folder": Path(MODEL_MODE_PARAMETERS["run_folder"]),
    "cache_folder": Path(tempfile.gettempdir()) / "bends_and_two_ways_cache",
    "line_positions_path": ANALYSIS_FOLDER / "inputs" / "audit_line_positions.csv",
    "line_profiles_path": ANALYSIS_FOLDER / "inputs" / "audit_line_profiles.npz",
    "place_half_window_cm": 3.0,
    "heading_smoothing_cm": 1.0,
    "place_limits_deg": [15.0, 60.0, 150.0],
    "line_end_cm": 2.0,
    "short_step_cm": 0.3,
    "twist_in_place_deg": 3.0,
    "jitter_angle_deg": 90.0,
    "path_steps": 3,
    "path_minimum_cm": 1.5,
    "threads": 2,
    "batch_size": 256
}



def merged_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``ANALYSIS_PARAMETERS`` with any overrides merged over it."""

    merged = dict(ANALYSIS_PARAMETERS)
    merged.update(parameters or {})
    return merged



def angle_difference_deg(
        first_deg: numpy.ndarray,
        second_deg: numpy.ndarray
) -> numpy.ndarray:
    """Return the size of the angle between two directions in degrees, 0 to 180."""

    return numpy.abs((numpy.asarray(first_deg, dtype = float) - numpy.asarray(second_deg, dtype = float) + 180.0) % 360.0 - 180.0)



def line_turning_within_window(
        points: numpy.ndarray,
        half_window_cm: float,
        smoothing_cm: float
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return, for every sample of a traced line, how much the line turns within a window centred on it.

    The heading of the line is smoothed first, because the traced points carry about a millimetre of noise. Two values
    are returned: the sum of the absolute heading changes (an S-bend counts in full) and the net change of heading from
    the window's start to its end, with its sign (a U-turn gives about 180 degrees, an S-bend about none; positive when
    the line turns to the left on the way from end A to end B).

    Parameters
    ----------
    points : numpy.ndarray
        (samples, 2), the line in cm, one sample every 0.1 cm of arc length.
    half_window_cm : float
        Half the window's length along the line.
    smoothing_cm : float
        Length of the moving average on the heading.

    Returns
    -------
    tuple of numpy.ndarray
        The absolute and the net turning within the window, (samples,) each, in degrees (the window is shorter at the
        line's two ends).
    """

    steps = numpy.diff(points, axis = 0)
    sample_cm = float(numpy.median(numpy.linalg.norm(steps, axis = 1)))
    heading = numpy.unwrap(numpy.arctan2(steps[:, 1], steps[:, 0]))
    width = max(1, int(round(smoothing_cm / sample_cm)))
    kernel = numpy.ones(width) / width
    smoothed = numpy.convolve(numpy.pad(heading, width, mode = "edge"), kernel, mode = "same")[width:-width]
    absolute_change = numpy.abs(numpy.diff(smoothed, prepend = smoothed[0]))
    cumulative = numpy.concatenate([[0.0], numpy.cumsum(absolute_change)])
    half = int(round(half_window_cm / sample_cm))
    positions = numpy.arange(len(points))
    lower = numpy.clip(positions - half, 0, len(absolute_change))
    upper = numpy.clip(positions + half, 0, len(absolute_change))
    absolute_turning = numpy.degrees(cumulative[upper] - cumulative[lower])
    last = len(smoothed) - 1
    # signed: positive when the line turns to the left (counter-clockwise seen from above) on the way from end A to end B
    net_turning = numpy.degrees(smoothed[numpy.clip(upper, 0, last)] - smoothed[numpy.clip(lower, 0, last)])
    return absolute_turning, net_turning



def place_of_turning(
        absolute_turning_deg: numpy.ndarray,
        net_turning_deg: numpy.ndarray,
        limits_deg: List[float]
) -> numpy.ndarray:
    """
    Return the kind of place of every pair of turning values.

    Parameters
    ----------
    absolute_turning_deg, net_turning_deg : numpy.ndarray
        From ``line_turning_within_window``, looked up at the puck's place (missing where the episode has no fit).
    limits_deg : list of float
        Where a gentle bend and a sharp bend begin (absolute turning) and where a hairpin begins (net turning).

    Returns
    -------
    numpy.ndarray
        A name of ``PLACE_NAMES`` per value, or "no fit" for a missing value.
    """

    absolute_turning = numpy.asarray(absolute_turning_deg, dtype = float)
    net_turning = numpy.abs(numpy.asarray(net_turning_deg, dtype = float))
    kinds = numpy.full(len(absolute_turning), "straight", dtype = object)
    kinds[absolute_turning >= limits_deg[0]] = "gentle bend"
    kinds[absolute_turning >= limits_deg[1]] = "sharp bend"
    kinds[net_turning >= limits_deg[2]] = "hairpin"
    kinds[numpy.isnan(absolute_turning)] = "no fit"
    return kinds



def load_demonstration_frames(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    Read the cleaned demonstration rows and add the step, the line position and the kind of place of every picture.

    Parameters
    ----------
    parameters : dict, optional
        Overrides for ``ANALYSIS_PARAMETERS``.

    Returns
    -------
    pandas.DataFrame
        The rows of ``load_training_frames`` (audit cleaning on), sorted by episode and step, with the added columns
        "step_forward_cm", "step_left_cm", "step_cm" (the labelled tip move after the picture, in the picture's frame),
        "step_picture_deg" and "step_table_deg" (its direction in the picture and on the table), "twist_deg",
        "is_moving", "is_stop", "puck_arc_cm" (from end A), "travel_arc_cm" (from the end the episode started at),
        "puck_to_line_cm", "travels_towards_A", "turning_deg" and "net_turning_deg" (the line's absolute and net turning
        within the window at the puck), "bend_turn_deg" (the net turning seen in the episode's direction of travel,
        positive to the left) and "place".
    """

    settings = merged_parameters(parameters)
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters["is_audit_cleaning"] = True
    frames = load_training_frames(target_parameters)
    frames["step_forward_cm"] = 100.0 * frames["tip_forward_m"].fillna(0.0)
    frames["step_left_cm"] = 100.0 * frames["tip_left_m"].fillna(0.0)
    frames["step_cm"] = numpy.hypot(frames["step_forward_cm"], frames["step_left_cm"])
    frames["step_picture_deg"] = numpy.degrees(numpy.arctan2(frames["step_left_cm"], frames["step_forward_cm"]))
    frames["step_table_deg"] = (frames["picture_heading_deg"] + frames["step_picture_deg"] + 180.0) % 360.0 - 180.0
    frames["twist_deg"] = frames["forearm_change_deg"].fillna(0.0)
    frames["is_stop"] = frames["action"] == "stop"
    frames["is_moving"] = frames["action"].isin(MOVING_ACTIONS)

    positions = pandas.read_csv(settings["line_positions_path"])
    positions = positions.rename(columns = {"puck_line_arc_from_A_cm": "puck_arc_cm"})
    frames = frames.merge(positions, on = "frame_index", how = "left")
    profiles = numpy.load(settings["line_profiles_path"])
    frames["turning_deg"] = numpy.nan
    frames["net_turning_deg"] = numpy.nan
    frames["line_cm"] = numpy.nan
    for sheet, group in frames.groupby("sheet"):
        number = sheet[-2:]
        arc = profiles[f"arc_{number}"]
        absolute_turning, net_turning = line_turning_within_window(profiles[f"points_{number}"], settings["place_half_window_cm"],
                                                                   settings["heading_smoothing_cm"])
        has_position = group["puck_arc_cm"].notna().to_numpy()
        nearest = numpy.clip(numpy.searchsorted(arc, group["puck_arc_cm"].fillna(0.0).to_numpy()), 0, len(arc) - 1)
        frames.loc[group.index, "turning_deg"] = numpy.where(has_position, absolute_turning[nearest], numpy.nan)
        frames.loc[group.index, "net_turning_deg"] = numpy.where(has_position, net_turning[nearest], numpy.nan)
        frames.loc[group.index, "line_cm"] = float(arc[-1])
    frames["place"] = place_of_turning(frames["turning_deg"].to_numpy(), frames["net_turning_deg"].to_numpy(), settings["place_limits_deg"])
    # the puck's place counted along the direction of travel: 0 at the end the episode started from
    is_towards_a = frames["travels_towards_A"].fillna(False).astype(bool)
    frames["travel_arc_cm"] = numpy.where(is_towards_a, frames["line_cm"] - frames["puck_arc_cm"], frames["puck_arc_cm"])
    # which way the line bends for this episode: positive = to the left of its direction of travel (a line that turns
    # left on the way from A to B turns right on the way back)
    frames["bend_turn_deg"] = numpy.where(is_towards_a, -frames["net_turning_deg"], frames["net_turning_deg"])
    # the last centimetres before the end cap are their own kind of place: the demonstrators fiddle the puck onto the
    # cap there, which is the stop's question and would blur the numbers of the straight stretches
    is_line_end = (frames["line_cm"] - frames["travel_arc_cm"]) <= settings["line_end_cm"]
    frames.loc[is_line_end & (frames["place"] != "no fit"), "place"] = "line end"
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    return frames



def path_direction(
        frames: pandas.DataFrame,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    Return where the demonstration's tip went over the next steps of every row, in the picture's frame.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_demonstration_frames``.
    parameters : dict, optional
        Supplies "path_steps" and "path_minimum_cm" (the report's path direction: 3 steps ahead, scored from 1.5 cm).

    Returns
    -------
    pandas.DataFrame
        Columns "path_deg" (direction to the tip's place "path_steps" steps later), "path_cm" (its distance),
        "is_path_row" (not a stop row and at least "path_minimum_cm"), and "ahead_K_forward_cm" / "ahead_K_left_cm"
        for K = 1 to 5 (the demonstrated tip place K steps later).
    """

    settings = merged_parameters(parameters)
    chunk = chunk_targets(frames, 5)
    vector = chunk[:, settings["path_steps"] - 1, :]
    result = pandas.DataFrame(index = frames.index)
    result["path_deg"] = numpy.degrees(numpy.arctan2(vector[:, 1], vector[:, 0]))
    result["path_cm"] = numpy.linalg.norm(vector, axis = 1)
    result["is_path_row"] = (~frames["is_stop"].to_numpy()) & (result["path_cm"].to_numpy() >= settings["path_minimum_cm"])
    for ahead in range(1, 6):
        result[f"ahead_{ahead}_forward_cm"] = chunk[:, ahead - 1, 0]
        result[f"ahead_{ahead}_left_cm"] = chunk[:, ahead - 1, 1]
    return result



def load_model(
        run_name: str,
        run_folder: Path,
        checkpoint_name: str
) -> Dict:
    """
    Load one saved model for inference on the processor.

    Parameters
    ----------
    run_name : str
        The run in ``TRAINING_RUNS`` whose recipe built the model.
    run_folder : pathlib.Path
        The folder with metrics.json and checkpoints/.
    checkpoint_name : str
        For example "without_path_04" or "all_sheets".

    Returns
    -------
    dict
        "model", "settings" (the run settings with the training normalisation) and "class_names".
    """

    metrics = json.loads((Path(run_folder) / "metrics.json").read_text(encoding = "utf-8"))
    settings = merged_run_settings(run_name)
    settings["chunk_steps"] = TRAINING_TARGET_PARAMETERS["chunk_steps"]
    settings["normalisation"] = metrics["settings"]["normalisation"]
    # the checkpoint holds every weight, so the ImageNet start weights need not be fetched
    settings["is_pretrained"] = False
    class_names = metrics["class_names"]
    output_sizes = design_outputs(settings["design"], len(class_names), settings["chunk_steps"], settings.get("is_stop_separate", False))
    model = PictureToMoveModel(settings, output_sizes)
    model.load_state_dict(torch.load(Path(run_folder) / "checkpoints" / f"{checkpoint_name}.pt", map_location = "cpu"))
    model.eval()
    torch.set_num_threads(ANALYSIS_PARAMETERS["threads"])
    return {"model": model, "settings": settings, "class_names": class_names}



def outputs_of_pictures(
        loaded: Dict,
        pictures: numpy.ndarray,
        is_hidden_kept: bool = False,
        parameters: Optional[Dict] = None
) -> Dict[str, numpy.ndarray]:
    """
    Run a loaded model over cached pictures and return its raw outputs.

    Parameters
    ----------
    loaded : dict
        From ``load_model``.
    pictures : numpy.ndarray
        (pictures, 120, 160, 3) uint8 RGB, as in images_wide_160.npy.
    is_hidden_kept : bool, optional
        Also return the head's shared hidden layer ("hidden", 256 numbers per picture), the model's own description of
        the picture, used to find look-alike pictures.
    parameters : dict, optional
        Supplies "batch_size" and "threads".

    Returns
    -------
    dict of str to numpy.ndarray
        One array per output layer ("exact", "chunk", "stop", "direction" as the design has them), raw values.
    """

    settings = merged_parameters(parameters)
    torch.set_num_threads(settings["threads"])
    model = loaded["model"]
    parts = []
    with torch.inference_mode():
        for start in range(0, len(pictures), settings["batch_size"]):
            batch = torch.from_numpy(numpy.ascontiguousarray(pictures[start:start + settings["batch_size"]]))
            inputs, _ = augmented_pictures(batch, loaded["settings"], numpy.random.default_rng(0), is_training = False)
            feature_map = model.backbone.forward_features(inputs)
            if model.pooling == "grid":
                pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, model.pooling_grid).flatten(1)
            else:
                pooled = feature_map.mean(dim = (2, 3))
            standardised = (pooled.float() - model.head.input_mean) / model.head.input_standard_deviation
            shared = model.head.shared(standardised)
            part = {name: layer(shared).numpy() for name, layer in model.head.outputs.items()}
            if is_hidden_kept:
                part["hidden"] = shared.numpy().astype(numpy.float16)
            parts.append(part)
    return {name: numpy.concatenate([part[name] for part in parts]) for name in parts[0]}



def cached_pictures(frame_indices: numpy.ndarray) -> numpy.ndarray:
    """Return the 160 x 120 training pictures of the given frame indices, in that order."""

    cache = numpy.load(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "images_wide_160.npy", mmap_mode = "r")
    order = numpy.argsort(frame_indices)
    pictures = numpy.asarray(cache[numpy.asarray(frame_indices)[order]])
    return pictures[numpy.argsort(order)]



def robot_run_pictures(run_folder: Path) -> numpy.ndarray:
    """
    Return the logged pictures of one robot run, shrunk exactly as ``ModelPolicy.picture_input`` shrinks them.

    Parameters
    ----------
    run_folder : pathlib.Path
        A folder of 03_data/model_runs with frames/NNN.jpg and steps.csv.

    Returns
    -------
    numpy.ndarray
        (pictures, 120, 160, 3) uint8 RGB, in the order of steps.csv.
    """

    import cv2

    steps = pandas.read_csv(Path(run_folder) / "steps.csv")
    pictures = []
    for frame in steps["frame"]:
        picture = cv2.imread(str(Path(run_folder) / frame), cv2.IMREAD_COLOR)
        pictures.append(cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (160, 120), interpolation = cv2.INTER_AREA))
    return numpy.stack(pictures)



class TeeOutput:

    def __init__(self, output_path: Path):
        """Print to the screen and into a text file at once, so every script leaves its printed output beside it."""

        self.output_file = open(output_path, "w", encoding = "utf-8")


    def __call__(self, *values) -> None:

        text = " ".join(str(value) for value in values)
        print(text, flush = True)
        self.output_file.write(text + "\n")
        self.output_file.flush()
