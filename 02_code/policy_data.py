"""
The training data of the policies, read from a dataset that build_dataset.py wrote (Marshal, 13.11.26).

``PolicyData`` reads samples.csv, shrinks every picture once into a cache file beside it, and turns the labels into
arrays: the move, the forearm and lean labels, the at-goal flag and the exact step. It also knows, for every picture,
the pictures before it in the same episode (for the history model) and the steps after it (for the chunk models).
``picture_batch`` cuts and colours a batch of pictures on the training device.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy
import torch

from config import TRAINING_PARAMETERS


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
SPLIT_NAMES = ["train", "validation", "test", "held_out"]


def merged_training_parameters(training_parameters: Optional[Dict] = None) -> Dict:
    """``TRAINING_PARAMETERS`` with any overrides; relative folders are resolved against the project folder."""

    parameters = dict(TRAINING_PARAMETERS)
    parameters.update(training_parameters or {})
    for key in ("processed_folder", "raw_folder", "runs_folder"):
        folder = Path(parameters[key])
        parameters[key] = folder if folder.is_absolute() else PROJECT_FOLDER / folder
    return parameters


def load_picture_cache(
        samples: List[Dict],
        dataset_folder: Path,
        raw_folder: Path,
        cache_size: List[int]
) -> numpy.ndarray:
    """
    All pictures of the dataset, shrunk to ``cache_size`` (width, height) as RGB, in the order of ``samples``.

    The first call reads every JPEG and writes ``pictures_<width>x<height>.npy`` beside samples.csv; later calls load
    that file, unless the list of pictures has changed since.

    Parameters
    ----------
    samples : list of dict
        The rows of samples.csv, each with "picture" (path relative to the raw folder).
    dataset_folder : Path
        The dataset's folder in the processed folder, where the cache is kept.
    raw_folder : Path
        The folder the picture paths are relative to.
    cache_size : list of int
        Width and height of the shrunk pictures.

    Returns
    -------
    numpy.ndarray
        uint8 array of shape (pictures, height, width, 3).
    """

    width, height = cache_size
    cache_path = dataset_folder / f"pictures_{width}x{height}.npy"
    key_path = dataset_folder / f"pictures_{width}x{height}.json"
    pictures_key = hashlib.sha256("\n".join(sample["picture"] for sample in samples).encode("utf-8")).hexdigest()
    if cache_path.exists() and key_path.exists():
        if json.loads(key_path.read_text(encoding = "utf-8")).get("pictures_key") == pictures_key:
            # Read from disk only as the kept pictures are copied out, so the file is not held twice in memory.
            return numpy.load(cache_path, mmap_mode = "r")

    pictures = numpy.empty((len(samples), height, width, 3), dtype = numpy.uint8)
    for index, sample in enumerate(samples):
        picture = cv2.imread(str(raw_folder / sample["picture"]))
        if picture is None:
            raise FileNotFoundError(f"cannot read {raw_folder / sample['picture']}")
        # INTER_AREA is the right filter for shrinking; the robot must use the same one on its live pictures.
        pictures[index] = cv2.cvtColor(cv2.resize(picture, (width, height), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        if index and index % 2_000 == 0:
            print(f"  picture cache: {index} of {len(samples)}", flush = True)
    numpy.save(cache_path, pictures)
    key_path.write_text(json.dumps({"pictures_key": pictures_key, "count": len(samples), "size": cache_size}), encoding = "utf-8")
    return pictures


def number_or_zero(text: str) -> float:
    """A number from a CSV cell; an empty cell counts as zero."""

    return float(text) if text not in (None, "") else 0.0


class PolicyData:
    """
    The pictures and labels of one dataset, as arrays ready for training.

    Parameters
    ----------
    training_parameters : dict, optional
        Overrides for ``TRAINING_PARAMETERS``.
    chunk_length : int, default 8
        How many steps, from this one on, the chunk targets hold.
    history_length : int, default 3
        How many pictures, this one included, the history indices hold.
    """

    def __init__(
            self,
            training_parameters: Optional[Dict] = None,
            chunk_length: int = 8,
            history_length: int = 3
    ):

        self.parameters = merged_training_parameters(training_parameters)
        dataset_folder = self.parameters["processed_folder"] / self.parameters["dataset_name"]
        with open(dataset_folder / "samples.csv", encoding = "utf-8", newline = "") as file:
            all_samples = list(csv.DictReader(file))
        if not all_samples:
            raise ValueError(f"{dataset_folder / 'samples.csv'} holds no pictures")
        all_pictures = load_picture_cache(all_samples, dataset_folder, self.parameters["raw_folder"], self.parameters["cache_size"])

        move_index = {move: index for index, move in enumerate(self.parameters["move_classes"])}
        kept = [index for index, sample in enumerate(all_samples) if sample["action"] in move_index]
        self.samples = [all_samples[index] for index in kept]
        self.pictures = numpy.ascontiguousarray(all_pictures[kept])
        del all_pictures
        self.count = len(self.samples)
        self.__device_pictures: Dict[str, torch.Tensor] = {}

        forearm_index = {label: index for index, label in enumerate(self.parameters["forearm_classes"])}
        lean_index = {label: index for index, label in enumerate(self.parameters["lean_classes"])}
        self.moves = numpy.array([move_index[sample["action"]] for sample in self.samples], dtype = numpy.int64)
        # The tool leaves the forearm and lean labels empty on stop rows: nothing turns there.
        self.forearm_labels = numpy.array([forearm_index.get(sample["forearm_label"] or "none", forearm_index["none"])
                                           for sample in self.samples], dtype = numpy.int64)
        self.lean_labels = numpy.array([lean_index.get(sample["lean_label"] or "none", lean_index["none"])
                                        for sample in self.samples], dtype = numpy.int64)
        self.at_goal = numpy.array([number_or_zero(sample["is_at_goal"]) for sample in self.samples], dtype = numpy.float32)
        self.exact = numpy.array([[number_or_zero(sample[column]) for column in self.parameters["exact_columns"]]
                                  for sample in self.samples], dtype = numpy.float32)
        self.is_stop = self.moves == move_index["stop"]
        self.sheets = [sample["sheet"] for sample in self.samples]
        self.scores = [str(sample["score"]) for sample in self.samples]

        held_out = self.parameters["held_out_sheet"]
        self.splits = numpy.array(["held_out" if held_out and sample["sheet"] == held_out else sample["split"]
                                   for sample in self.samples])

        self.__link_episodes(chunk_length, history_length)
        self.__normalise_exact()


    def __link_episodes(
            self,
            chunk_length: int,
            history_length: int
    ) -> None:
        """Index, for every picture, the earlier pictures of its episode and the steps after it."""

        by_episode: Dict[str, List[int]] = {}
        for index, sample in enumerate(self.samples):
            by_episode.setdefault(sample["episode_key"], []).append(index)
        self.history = numpy.zeros((self.count, history_length), dtype = numpy.int64)
        # -1 marks a step after the episode's end: the target there is "stopped".
        self.chunk = numpy.full((self.count, chunk_length), -1, dtype = numpy.int64)
        self.previous_move = numpy.full(self.count, len(self.parameters["move_classes"]), dtype = numpy.int64)
        for indices in by_episode.values():
            indices = sorted(indices, key = lambda index: int(self.samples[index]["step"]))
            for position, index in enumerate(indices):
                # Column 0 is this picture, then one step back each; the first picture of an episode repeats itself.
                for back in range(history_length):
                    self.history[index, back] = indices[max(0, position - back)]
                following = indices[position:position + chunk_length]
                self.chunk[index, :len(following)] = following
                if position > 0:
                    self.previous_move[index] = self.moves[indices[position - 1]]


    def __normalise_exact(self) -> None:
        """Mean and spread of the exact step over the training pictures that move, used to scale the regression targets."""

        moving = (self.splits == "train") & ~self.is_stop
        reference = self.exact[moving] if moving.any() else self.exact
        self.exact_mean = reference.mean(axis = 0)
        self.exact_std = numpy.maximum(reference.std(axis = 0), 1e-6)
        self.exact_scaled = ((self.exact - self.exact_mean) / self.exact_std).astype(numpy.float32)


    def indices(self, split: str) -> numpy.ndarray:
        """The picture indices of one split ("train", "validation", "test" or "held_out")."""

        return numpy.flatnonzero(self.splits == split)


    def move_class_weights(self) -> numpy.ndarray:
        """
        Loss weights for the move classes: (1 / share) ** "class_weight_power" over the training pictures, stop
        multiplied by "stop_weight_multiplier", scaled to a mean of 1 over the training pictures.
        """

        train_moves = self.moves[self.indices("train")]
        counts = numpy.bincount(train_moves, minlength = len(self.parameters["move_classes"])).astype(numpy.float64)
        shares = numpy.maximum(counts, 1.0) / max(1, len(train_moves))
        weights = shares ** -self.parameters["class_weight_power"]
        weights[self.parameters["move_classes"].index("stop")] *= self.parameters["stop_weight_multiplier"]
        mean_weight = (weights * counts).sum() / max(1.0, counts.sum())
        return (weights / mean_weight).astype(numpy.float32)


    def sampling_weights(self, indices: numpy.ndarray) -> numpy.ndarray:
        """How likely each training picture is drawn: every sheet equally likely when "is_sheet_balanced", else uniform."""

        if not self.parameters["is_sheet_balanced"]:
            return numpy.ones(len(indices), dtype = numpy.float64)
        sheets = [self.sheets[index] for index in indices]
        per_sheet = {sheet: sheets.count(sheet) for sheet in set(sheets)}
        return numpy.array([1.0 / per_sheet[sheet] for sheet in sheets], dtype = numpy.float64)


    def score_weights(self) -> numpy.ndarray:
        """Each picture's loss weight from its episode's score ("score_weights"; 1 for a score not named there)."""

        return numpy.array([self.parameters["score_weights"].get(score, 1.0) for score in self.scores], dtype = numpy.float32)


    def tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        The label arrays as tensors on ``device``, and the pictures there too when "is_pictures_on_device".

        The pictures are sent to a device once and kept there for later runs in the same process; the small arrays are
        made anew each time, as the score weights may change between Optuna trials.
        """

        pictures = torch.from_numpy(self.pictures)
        if self.parameters["is_pictures_on_device"]:
            if str(device) not in self.__device_pictures:
                self.__device_pictures[str(device)] = pictures.to(device)
            pictures = self.__device_pictures[str(device)]
        tensors = {
            "pictures": pictures,
            "moves": torch.from_numpy(self.moves).to(device),
            "forearm_labels": torch.from_numpy(self.forearm_labels).to(device),
            "lean_labels": torch.from_numpy(self.lean_labels).to(device),
            "at_goal": torch.from_numpy(self.at_goal).to(device),
            "exact": torch.from_numpy(self.exact).to(device),
            "exact_scaled": torch.from_numpy(self.exact_scaled).to(device),
            "is_stop": torch.from_numpy(self.is_stop).to(device),
            "history": torch.from_numpy(self.history).to(device),
            "chunk": torch.from_numpy(self.chunk).to(device),
            "previous_move": torch.from_numpy(self.previous_move).to(device),
            "score_weights": torch.from_numpy(self.score_weights()).to(device)
        }
        return tensors


def picture_batch(
        pictures: torch.Tensor,
        indices: torch.Tensor,
        is_training: bool,
        parameters: Dict,
        device: torch.device
) -> torch.Tensor:
    """
    Cut, colour and normalise a batch of cached pictures.

    Training cuts a random "crop_size" window per sample (the same window for all pictures of one sample's history)
    and scales brightness, contrast and saturation by up to "colour_jitter"; evaluation cuts the centre and changes
    nothing else.

    Parameters
    ----------
    pictures : torch.Tensor
        uint8 cache of shape (pictures, height, width, 3), on ``device`` or on the CPU.
    indices : torch.Tensor
        Picture indices, shape (batch,) or (batch, history).
    is_training : bool
        Random crop and colour jitter when True.
    parameters : dict
        Merged ``TRAINING_PARAMETERS``.
    device : torch.device
        Where the batch goes.

    Returns
    -------
    torch.Tensor
        float tensor of shape (batch, 3, crop height, crop width), or (batch, history, 3, crop height, crop width).
    """

    has_history = indices.dim() == 2
    batch_size = indices.shape[0]
    flat_indices = indices.reshape(-1)
    if pictures.device != flat_indices.device:
        batch = pictures[flat_indices.cpu()].to(device)
    else:
        batch = pictures[flat_indices]
    per_sample = indices.shape[1] if has_history else 1
    _, height, width, _ = batch.shape
    crop_width, crop_height = parameters["crop_size"]

    if is_training:
        offset_y = torch.randint(0, height - crop_height + 1, (batch_size,), device = device).repeat_interleave(per_sample)
        offset_x = torch.randint(0, width - crop_width + 1, (batch_size,), device = device).repeat_interleave(per_sample)
    else:
        offset_y = torch.full((batch_size * per_sample,), (height - crop_height) // 2, device = device)
        offset_x = torch.full((batch_size * per_sample,), (width - crop_width) // 2, device = device)
    rows = offset_y[:, None] + torch.arange(crop_height, device = device)[None, :]
    columns = offset_x[:, None] + torch.arange(crop_width, device = device)[None, :]
    samples = torch.arange(batch.shape[0], device = device)
    batch = batch[samples[:, None, None], rows[:, :, None], columns[:, None, :]]
    batch = batch.permute(0, 3, 1, 2).float() / 255.0

    jitter = parameters["colour_jitter"]
    if is_training and jitter > 0:
        def factors() -> torch.Tensor:
            return (1.0 + (torch.rand(batch_size, device = device) * 2.0 - 1.0) * jitter).repeat_interleave(per_sample)[:, None, None, None]
        batch = batch * factors()
        grey = batch.mean(dim = (1, 2, 3), keepdim = True)
        batch = (batch - grey) * factors() + grey
        luminance = (0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3])
        batch = (batch - luminance) * factors() + luminance
        batch = batch.clamp(0.0, 1.0)

    mean = torch.tensor(parameters["imagenet_mean"], device = device)[None, :, None, None]
    std = torch.tensor(parameters["imagenet_std"], device = device)[None, :, None, None]
    batch = (batch - mean) / std
    if has_history:
        batch = batch.reshape(batch_size, per_sample, *batch.shape[1:])
    return batch
