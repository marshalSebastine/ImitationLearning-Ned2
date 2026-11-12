"""
Verifier's own helpers: the demonstration rows with a kind of place from the verifier's own turning measure, the
demonstrated tip places ahead, and a plain forward pass of a saved model.

Nothing here is taken from the analyst's common.py; only the analyst's two input files (the audit's traced lines and
puck positions along them) are read, because no other source of line positions exists in the project. Read-only on the
project; model outputs are cached in the computer's temporary folder.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas
import torch

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from config import RECORDING_PARAMETERS, TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import PictureToMoveModel, augmented_pictures, merged_run_settings
from training_probes import design_outputs
from training_targets import load_training_frames

VERIFY_PARAMETERS = {
    "positions_path": VERIFY_FOLDER.parent / "inputs" / "audit_line_positions.csv",
    "profiles_path": VERIFY_FOLDER.parent / "inputs" / "audit_line_profiles.npz",
    "fold_runs_folder": ROOT_FOLDER / "04_runs" / "gpu_ki014",
    "cache_folder": Path(tempfile.gettempdir()) / "bends_and_two_ways_verify_cache",
    "half_window_cm": 3.0,
    "chord_samples": 5,
    "gentle_from_deg": 15.0,
    "sharp_from_deg": 60.0,
    "hairpin_from_deg": 150.0,
    "line_end_cm": 2.0,
    "threads": 2,
    "batch_size": 128
}

PLACES = ["straight", "gentle bend", "sharp bend", "hairpin", "line end"]


def own_turning(points: numpy.ndarray, half_window_cm: float, chord_samples: int) -> tuple:
    """
    Return the line's absolute and net turning within a window round every sample.

    The heading comes from chords of 2 * chord_samples * 0.1 cm, which removes the tracing noise.

    Parameters
    ----------
    points : numpy.ndarray
        (samples, 2) in cm, one sample every 0.1 cm.
    half_window_cm : float
        Half the window along the line.
    chord_samples : int
        Half the chord's length in samples.

    Returns
    -------
    tuple of numpy.ndarray
        Absolute and net turning in degrees per sample.
    """

    count = len(points)
    ahead = numpy.clip(numpy.arange(count) + chord_samples, 0, count - 1)
    behind = numpy.clip(numpy.arange(count) - chord_samples, 0, count - 1)
    vector = points[ahead] - points[behind]
    heading = numpy.unwrap(numpy.arctan2(vector[:, 1], vector[:, 0]))
    cumulative = numpy.cumsum(numpy.abs(numpy.diff(heading, prepend = heading[0])))
    half = int(round(half_window_cm / 0.1))
    low = numpy.clip(numpy.arange(count) - half, 0, count - 1)
    high = numpy.clip(numpy.arange(count) + half, 0, count - 1)
    return numpy.degrees(cumulative[high] - cumulative[low]), numpy.degrees(heading[high] - heading[low])


def demonstration_rows(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    Return the cleaned demonstration rows with the step, the puck's place along the line and the kind of place.

    Parameters
    ----------
    parameters : dict, optional
        Overrides of VERIFY_PARAMETERS.

    Returns
    -------
    pandas.DataFrame
        The rows of load_training_frames (audit cleaning), sorted by episode and step, with "step_cm",
        "step_picture_deg", "step_table_deg", "twist_deg", "is_stop", "arc_cm" (puck along the line from end A),
        "travel_sign" (+1 from A to B, -1 from B to A, from the episode's own net progress along the line),
        "absolute_turning_deg", "net_turning_deg", "bend_turn_deg" (net turning seen in the direction of travel) and
        "place".
    """

    settings = dict(VERIFY_PARAMETERS)
    settings.update(parameters or {})
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters["is_audit_cleaning"] = True
    frames = load_training_frames(target_parameters)
    frames["step_forward_cm"] = 100.0 * frames["tip_forward_m"].fillna(0.0)
    frames["step_left_cm"] = 100.0 * frames["tip_left_m"].fillna(0.0)
    frames["step_cm"] = numpy.hypot(frames["step_forward_cm"], frames["step_left_cm"])
    frames["step_picture_deg"] = numpy.degrees(numpy.arctan2(frames["step_left_cm"], frames["step_forward_cm"]))
    frames["step_table_deg"] = (frames["picture_heading_deg"] + RECORDING_PARAMETERS["picture_left_sign"] * frames["step_picture_deg"] + 180.0) % 360.0 - 180.0
    frames["twist_deg"] = frames["forearm_change_deg"].fillna(0.0)
    frames["is_stop"] = frames["action"] == "stop"

    positions = pandas.read_csv(settings["positions_path"])[["frame_index", "puck_line_arc_from_A_cm", "puck_to_line_cm"]]
    frames = frames.merge(positions.rename(columns = {"puck_line_arc_from_A_cm": "arc_cm"}), on = "frame_index", how = "left")
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    profiles = numpy.load(settings["profiles_path"])
    frames["absolute_turning_deg"] = numpy.nan
    frames["net_turning_deg"] = numpy.nan
    frames["line_cm"] = numpy.nan
    for sheet, group in frames.groupby("sheet"):
        number = sheet[-2:]
        absolute, net = own_turning(profiles[f"points_{number}"], settings["half_window_cm"], settings["chord_samples"])
        arc = profiles[f"arc_{number}"]
        has_arc = group["arc_cm"].notna().to_numpy()
        index = numpy.clip(numpy.rint(group["arc_cm"].fillna(0.0).to_numpy() / 0.1).astype(int), 0, len(arc) - 1)
        frames.loc[group.index, "absolute_turning_deg"] = numpy.where(has_arc, absolute[index], numpy.nan)
        frames.loc[group.index, "net_turning_deg"] = numpy.where(has_arc, net[index], numpy.nan)
        frames.loc[group.index, "line_cm"] = float(arc[-1])
    # direction of travel from the episode's own progress along the line (not from the analyst's column)
    progress = frames.groupby("episode_id")["arc_cm"].agg(lambda values: values.dropna().iloc[-1] - values.dropna().iloc[0] if values.notna().sum() > 1 else numpy.nan)
    frames["travel_sign"] = numpy.sign(frames["episode_id"].map(progress)).replace(0.0, numpy.nan)
    frames["bend_turn_deg"] = frames["travel_sign"] * frames["net_turning_deg"]
    to_go = numpy.where(frames["travel_sign"] > 0, frames["line_cm"] - frames["arc_cm"], frames["arc_cm"])
    place = numpy.full(len(frames), "straight", dtype = object)
    place[frames["absolute_turning_deg"].to_numpy() >= settings["gentle_from_deg"]] = "gentle bend"
    place[frames["absolute_turning_deg"].to_numpy() >= settings["sharp_from_deg"]] = "sharp bend"
    place[numpy.abs(frames["net_turning_deg"].to_numpy()) >= settings["hairpin_from_deg"]] = "hairpin"
    place[to_go <= settings["line_end_cm"]] = "line end"
    place[frames["arc_cm"].isna().to_numpy() | frames["travel_sign"].isna().to_numpy()] = "no place"
    frames["place"] = place
    return frames


def tips_ahead(frames: pandas.DataFrame, steps_ahead: int) -> numpy.ndarray:
    """
    Return the demonstrated tip place 1 to steps_ahead steps later in the current picture's frame (forward, left; cm).

    Written anew (vectorised per episode); after the episode's last row the tip stays where it stopped, as in training.
    """

    sign = RECORDING_PARAMETERS["picture_left_sign"]
    result = numpy.zeros((len(frames), steps_ahead, 2))
    for _, group in frames.groupby("episode_id", sort = False):
        positions = group.index.to_numpy()
        tip = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
        heading = numpy.radians(group["picture_heading_deg"].to_numpy())
        forward = numpy.stack([numpy.cos(heading), numpy.sin(heading)], axis = 1)
        left = numpy.stack([numpy.cos(heading + sign * numpy.pi / 2), numpy.sin(heading + sign * numpy.pi / 2)], axis = 1)
        for ahead in range(1, steps_ahead + 1):
            future = numpy.minimum(numpy.arange(len(group)) + ahead, len(group) - 1)
            moved = tip[future] - tip
            result[positions, ahead - 1, 0] = (moved * forward).sum(axis = 1)
            result[positions, ahead - 1, 1] = (moved * left).sum(axis = 1)
    return result


def load_saved_model(run_name: str, run_folder: Path, checkpoint_name: str) -> Dict:
    """Load one saved model for a forward pass on the processor (the steps of ModelPolicy, without the robot code)."""

    metrics = json.loads((Path(run_folder) / "metrics.json").read_text(encoding = "utf-8"))
    settings = merged_run_settings(run_name)
    settings["chunk_steps"] = TRAINING_TARGET_PARAMETERS["chunk_steps"]
    settings["normalisation"] = metrics["settings"]["normalisation"]
    settings["is_pretrained"] = False
    output_sizes = design_outputs(settings["design"], len(metrics["class_names"]), settings["chunk_steps"], settings.get("is_stop_separate", False))
    model = PictureToMoveModel(settings, output_sizes)
    model.load_state_dict(torch.load(Path(run_folder) / "checkpoints" / f"{checkpoint_name}.pt", map_location = "cpu"))
    model.eval()
    return {"model": model, "settings": settings, "class_names": metrics["class_names"]}


def forward_pass(loaded: Dict, pictures: numpy.ndarray, is_feature_kept: bool = False, parameters: Optional[Dict] = None) -> Dict[str, numpy.ndarray]:
    """
    Run a model over (pictures, 120, 160, 3) uint8 RGB pictures with its own forward function.

    With is_feature_kept the pooled backbone features (before the head) are returned too, as "pooled": a second,
    head-independent description of the picture for the look-alike check.
    """

    settings = dict(VERIFY_PARAMETERS)
    settings.update(parameters or {})
    torch.set_num_threads(settings["threads"])
    model = loaded["model"]
    parts = []
    with torch.inference_mode():
        for start in range(0, len(pictures), settings["batch_size"]):
            batch = torch.from_numpy(numpy.ascontiguousarray(pictures[start:start + settings["batch_size"]]))
            inputs, _ = augmented_pictures(batch, loaded["settings"], numpy.random.default_rng(0), is_training = False)
            part = {name: value.numpy() for name, value in model(inputs).items()}
            if is_feature_kept:
                feature_map = model.backbone.forward_features(inputs)
                part["pooled"] = feature_map.mean(dim = (2, 3)).numpy().astype(numpy.float32)
            parts.append(part)
    return {name: numpy.concatenate([part[name] for part in parts]) for name in parts[0]}


def cached_training_pictures(frame_indices: numpy.ndarray) -> numpy.ndarray:
    """Return the cached 160 x 120 pictures of the given frame indices, in the given order."""

    cache = numpy.load(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "images_wide_160.npy", mmap_mode = "r")
    indices = numpy.asarray(frame_indices)
    order = numpy.argsort(indices)
    pictures = numpy.asarray(cache[indices[order]])
    return pictures[numpy.argsort(order)]


def angle_between(first_deg, second_deg) -> numpy.ndarray:
    """Return the size of the angle between two directions, 0 to 180 degrees."""

    return numpy.abs((numpy.asarray(first_deg, dtype = float) - numpy.asarray(second_deg, dtype = float) + 180.0) % 360.0 - 180.0)
