"""
Train a picture-to-move model end to end and score it on held-out sheets or episodes (9.11.26, Model and training).

Unlike training_probes.py, the whole network learns here, from the pictures themselves, with augmentation. A run is a
named entry of TRAINING_RUNS in config.py: the network (a small network from scratch, or a pretrained one that is
fine-tuned), the target design (the same heads and losses as the probes), the augmentation and the training schedule.
Every run is scored with the shared offline measures, fold by fold, so its numbers sit beside the probes' numbers.

Augmentation, and why each kind keeps the label right:
    colour      brightness, contrast, saturation and gamma: the light changes, the right move does not;
    blur, noise the camera's focus and the dim room change the picture, not the move;
    shift       a few pixels only: the taped camera creeps a few pixels between days (camera_shift_px), while a large
                shift would move the line relative to the puck and change the right move;
    mirror      left and right swapped in the picture AND in the label (left moves become right moves, twists change
                sign); off by default, because the camera does not sit on the rod's axis, so a mirrored picture shows
                a view the camera can never have (tested as its own run).

Run from 02_code, after training_dataset.py:
    ..\\.venv\\Scripts\\python.exe train_policy_emranul.py resnet18_finetune_classes
    ..\\.venv\\Scripts\\python.exe train_policy_emranul.py resnet18_finetune_classes --folds without_path_04 without_path_05
Results: 04_runs/<date>_<run name>/metrics.json and fold_<name>.json; weights in checkpoints/ (not in git).
Settings: TRAINING_RUN_PARAMETERS and TRAINING_RUNS in config.py.
"""

import argparse
import copy
import datetime
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
import pandas
import timm
import torch

from config import TRAINING_DATA_PARAMETERS, TRAINING_RUN_PARAMETERS, TRAINING_RUNS, TRAINING_TARGET_PARAMETERS
from training_metrics import summarised_folds
from training_probes import ProbeHead, design_outputs, fold_measures, head_losses, predictions_from_outputs, training_arrays
from training_targets import (MIRRORED_DIRECTION, class_weights, episode_folds, inner_validation_mask, leave_one_sheet_out_folds,
                              load_training_frames, new_shape_folds, previous_frame_indices, round_1_to_new_folds)



class SmallConvolutionalBackbone(torch.nn.Module):

    def __init__(
            self,
            channels: List[int],
            input_channels: int = 3
    ):
        """
        Initialise a small network trained from scratch: one block per entry of ``channels``.

        Each block is a 3 x 3 convolution with stride 2, batch normalisation and a ReLU, followed by a second 3 x 3
        convolution, so five blocks shrink a 160 x 120 picture to 5 x 4 cells.

        Parameters
        ----------
        channels : list of int
            The width of each block, for example ``[32, 64, 96, 128, 192]``.
        input_channels : int, optional
            Three per picture shown at once.
        """

        super().__init__()
        layers = []
        width = input_channels
        for block_width in channels:
            layers += [
                torch.nn.Conv2d(width, block_width, 3, stride = 2, padding = 1, bias = False),
                torch.nn.BatchNorm2d(block_width),
                torch.nn.ReLU(inplace = True),
                torch.nn.Conv2d(block_width, block_width, 3, padding = 1, bias = False),
                torch.nn.BatchNorm2d(block_width),
                torch.nn.ReLU(inplace = True)
            ]
            width = block_width
        self.blocks = torch.nn.Sequential(*layers)
        self.num_features = width


    def forward_features(self, pictures: torch.Tensor) -> torch.Tensor:

        return self.blocks(pictures)



class PictureToMoveModel(torch.nn.Module):

    def __init__(
            self,
            run_settings: Dict,
            output_sizes: Dict[str, int]
    ):
        """
        Initialise a backbone with the shared prediction head on top.

        Parameters
        ----------
        run_settings : dict
            "backbone" ("small_cnn" or a timm model name), "is_pretrained", "small_cnn_channels", "pooling"
            ("average" for one summary, or "grid" for the cells of "pooling_grid", which keeps where things are),
            "hidden_size", "dropout" and "picture_count" (1, or 2 for the picture and the one before it, stacked as six
            colour channels).
        output_sizes : dict of str to int
            From ``design_outputs``.
        """

        super().__init__()
        input_channels = 3 * run_settings.get("picture_count", 1)
        if run_settings["backbone"] == "small_cnn":
            self.backbone = SmallConvolutionalBackbone(run_settings["small_cnn_channels"], input_channels)
        else:
            # with six channels timm repeats the pretrained first layer's weights for the second picture and halves
            # them, so at the start two equal pictures give exactly the pretrained network's answer
            self.backbone = timm.create_model(run_settings["backbone"], pretrained = run_settings["is_pretrained"], num_classes = 0,
                                              in_chans = input_channels)
        self.pooling = run_settings["pooling"]
        self.pooling_grid = tuple(run_settings["pooling_grid"])
        cells = self.pooling_grid[0] * self.pooling_grid[1] if self.pooling == "grid" else 1
        # 15.11.26: the direction memory ("memory_distances_cm"): where the tip was some cm of path ago, as arrows in
        # the picture's frame with a known-flag each; a small layer turns them into "memory_hidden_size" numbers that
        # join the picture's features before the head
        self.memory_size = 3 * len(run_settings.get("memory_distances_cm") or [])
        self.memory_scale_cm = float(run_settings.get("memory_scale_cm", 10.0))
        memory_width = 0
        if self.memory_size:
            memory_width = int(run_settings["memory_hidden_size"])
            self.memory_layer = torch.nn.Sequential(torch.nn.Linear(self.memory_size, memory_width), torch.nn.ReLU())
        self.head = ProbeHead(self.backbone.num_features * cells + memory_width, output_sizes, run_settings["hidden_size"], run_settings["dropout"])


    def forward(
            self,
            pictures: torch.Tensor,
            memory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """The outputs for a batch of pictures; a model with a direction memory takes it too (none = all unknown)."""

        feature_map = self.backbone.forward_features(pictures)
        if self.pooling == "grid":
            pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, self.pooling_grid).flatten(1)
        else:
            pooled = feature_map.mean(dim = (2, 3))
        if self.memory_size:
            if memory is None:
                memory = torch.zeros(len(pooled), self.memory_size, device = pooled.device)
            arrow_count = 2 * self.memory_size // 3
            scaled = torch.cat([memory[:, :arrow_count] / self.memory_scale_cm, memory[:, arrow_count:]], dim = 1).to(pooled.dtype)
            pooled = torch.cat([pooled, self.memory_layer(scaled)], dim = 1)
        return self.head(pooled)



class SequencePictureModel(torch.nn.Module):

    def __init__(
            self,
            run_settings: Dict,
            output_sizes: Dict[str, int]
    ):
        """
        The recurrent model (15.11.26): the backbone turns each of the last "sequence_frames" camera pictures (taken
        "sequence_stride" steps apart, the newest last) into features, a GRU of "gru_size" reads them in order, and the
        head answers from the GRU's last state together with the newest picture's own features. Only pictures are its
        input, as the course demands; the sequence lets it see how the view has turned on the way (the forearm's
        winding) and where it came from (the first or second pass of a loop).

        Parameters
        ----------
        run_settings : dict
            As for PictureToMoveModel, plus "sequence_frames", "sequence_gradient_frames" (the newest pictures the
            backbone learns from; the older ones pass through it without gradient, to fit the card) and "gru_size".
        output_sizes : dict of str to int
            From ``design_outputs``.
        """

        super().__init__()
        if run_settings["backbone"] == "small_cnn":
            self.backbone = SmallConvolutionalBackbone(run_settings["small_cnn_channels"], 3)
        else:
            self.backbone = timm.create_model(run_settings["backbone"], pretrained = run_settings["is_pretrained"], num_classes = 0)
        self.frame_count = int(run_settings["sequence_frames"])
        self.gradient_frames = int(run_settings["sequence_gradient_frames"])
        self.memory_size = 0
        self.gru = torch.nn.GRU(self.backbone.num_features, int(run_settings["gru_size"]), batch_first = True)
        self.head = ProbeHead(int(run_settings["gru_size"]) + self.backbone.num_features, output_sizes, run_settings["hidden_size"], run_settings["dropout"])


    def picture_features(self, pictures: torch.Tensor) -> torch.Tensor:
        """(pictures, 3, height, width) normalised input to (pictures, features), averaged over the feature map."""

        return self.backbone.forward_features(pictures).mean(dim = (2, 3))


    def answer_from_features(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """(batch, frames, features), the oldest picture first, to the outputs at the newest picture."""

        states, _ = self.gru(features)
        return self.head(torch.cat([states[:, -1], features[:, -1]], dim = 1))


    def forward(
            self,
            pictures: torch.Tensor,
            memory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """(batch, 3 x frames, height, width) with the newest picture in the first three channels (as stacked by
        ``SequencePictures`` and augmented alike by ``augmented_pictures``)."""

        batch, channels, height, width = pictures.shape
        frames = pictures.view(batch, channels // 3, 3, height, width)
        newest = self.picture_features(frames[:, :self.gradient_frames].reshape(-1, 3, height, width)).view(batch, self.gradient_frames, -1)
        if frames.shape[1] > self.gradient_frames:
            with torch.no_grad():
                older = self.picture_features(frames[:, self.gradient_frames:].reshape(-1, 3, height, width)).view(batch, frames.shape[1] - self.gradient_frames, -1)
            features = torch.cat([newest, older], dim = 1)
        else:
            features = newest
        return self.answer_from_features(torch.flip(features, dims = [1]))



class SequencePictures:

    def __init__(
            self,
            cache: numpy.ndarray,
            frames: pandas.DataFrame,
            frame_count: int,
            stride: int
    ):
        """
        The picture sequences of the recurrent model, built when a batch asks for them: for each row the camera's
        pictures 0, stride, 2 x stride, ... steps back (an episode's first picture repeats before it began), stacked as
        channels with the newest first, like the two-picture model's input.
        """

        self.cache = torch.from_numpy(numpy.ascontiguousarray(cache))
        lags = [stride * number for number in range(frame_count)]
        self.indices = numpy.stack([previous_frame_indices(frames, lag) for lag in lags], axis = 1)


    def __len__(self) -> int:

        return len(self.indices)


    def __getitem__(self, rows) -> torch.Tensor:

        rows = numpy.atleast_1d(numpy.asarray(rows))
        stacked = self.cache[torch.from_numpy(self.indices[rows].reshape(-1))]
        count, frame_count = len(rows), self.indices.shape[1]
        height, width = stacked.shape[1], stacked.shape[2]
        return stacked.view(count, frame_count, height, width, 3).permute(0, 2, 3, 1, 4).reshape(count, height, width, 3 * frame_count)



def merged_run_settings(run_name: str) -> Dict:
    """Return ``TRAINING_RUN_PARAMETERS`` with the named run's settings merged over it."""

    settings = dict(TRAINING_RUN_PARAMETERS)
    settings.update(TRAINING_RUNS[run_name])
    settings["name"] = run_name
    return settings



def gaussian_kernel(sigma: float) -> torch.Tensor:
    """Return a normalised one-dimensional Gaussian kernel of about three sigmas each way."""

    radius = max(1, int(math.ceil(3.0 * sigma)))
    positions = torch.arange(-radius, radius + 1, dtype = torch.float32)
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / kernel.sum()



def augmented_pictures(
        pictures: torch.Tensor,
        settings: Dict,
        randomness_generator: numpy.random.Generator,
        is_training: bool
) -> Tuple[torch.Tensor, numpy.ndarray]:
    """
    Turn a batch of cached pictures, one or several per row, into normalised network input, augmented when training.

    With several pictures per row (the picture and the one before it, 14.11.26), every picture of a row gets the same
    augmentation (the same light, blur, noise level, shift and mirror), so the only change between them is the robot's
    own move.

    Parameters
    ----------
    pictures : torch.Tensor
        (batch, height, width, 3 x pictures per row), uint8 RGB, the newest picture first.
    settings : dict
        The run settings: "augmentation" (the strength of each kind) and "normalisation".
    randomness_generator : numpy.random.Generator
        Source of every random choice, so a run can be repeated.
    is_training : bool
        Augment only when True.

    Returns
    -------
    tuple of torch.Tensor and numpy.ndarray
        (batch, 3 x pictures per row, height, width) float input, and which rows were mirrored.
    """

    picture_count = pictures.shape[-1] // 3
    if picture_count == 1:
        return augmented_single_pictures(pictures, settings, randomness_generator, is_training)
    generator_before = copy.deepcopy(randomness_generator)
    parts = []
    is_mirrored = None
    for picture_number in range(picture_count):
        # the first picture draws from the run's own generator (so it moves on as with one picture); the others repeat
        # exactly the same draws from a copy taken before
        generator = randomness_generator if picture_number == 0 else copy.deepcopy(generator_before)
        part, part_is_mirrored = augmented_single_pictures(pictures[..., 3 * picture_number:3 * picture_number + 3], settings, generator, is_training)
        parts.append(part)
        is_mirrored = part_is_mirrored if is_mirrored is None else is_mirrored
    return torch.cat(parts, dim = 1), is_mirrored



def table_as_paper(
        batch: torch.Tensor,
        is_repainted: numpy.ndarray,
        parameters: Dict
) -> torch.Tensor:
    """
    Repaint the table in the chosen pictures in the picture's own paper colour (14.11.26).

    In round 1 every line ends 2 to 4 cm from the paper's edge, and the model learned the edge as its stop cue
    (2026-11-14_stop_check.md). With the table painted like paper, an end near the edge looks like an end in the middle
    of a sheet, so the stop has to be read from the line itself. A pixel is table when it is not bluish like the paper
    (red minus blue above a limit), not red like the puck, not dark, and outside the robot's own area; it takes the
    mean colour of the picture's paper pixels, shaded by its own brightness against the table's mean, so the table's
    light and shadow stay.

    Parameters
    ----------
    batch : torch.Tensor
        (batch, 3, height, width) float RGB between 0 and 1, before any other augmentation.
    is_repainted : numpy.ndarray
        Which pictures of the batch are repainted.
    parameters : dict
        The colour limits ("table_as_paper" of the run settings).

    Returns
    -------
    torch.Tensor
        The batch with the table of the chosen pictures repainted.
    """

    red, green, blue = batch[:, 0], batch[:, 1], batch[:, 2]
    brightness = batch.mean(dim = 1)
    is_table = (((red - blue) > parameters["table_red_over_blue_least"]) & ((red - green) < parameters["table_red_over_green_most"])
                & (brightness > parameters["table_brightness_least"]))
    left, top, right, bottom = parameters["robot_area_px"]
    is_table[:, top:bottom, left:right] = False
    is_table = is_table & torch.from_numpy(is_repainted).view(-1, 1, 1)
    is_paper = ((red - blue) < parameters["paper_red_over_blue_most"]) & (brightness > parameters["paper_brightness_least"])

    paper_count = is_paper.sum(dim = (1, 2)).float()
    paper_colour = (batch * is_paper.unsqueeze(1)).sum(dim = (2, 3)) / paper_count.clamp(min = 1.0).view(-1, 1)
    usual_colour = torch.tensor(parameters["paper_colour"], dtype = batch.dtype).view(1, 3)
    paper_colour = torch.where((paper_count >= parameters["paper_pixels_least"]).view(-1, 1), paper_colour, usual_colour)
    table_brightness = (brightness * is_table).sum(dim = (1, 2)) / is_table.sum(dim = (1, 2)).float().clamp(min = 1.0)
    shade = (brightness / table_brightness.clamp(min = 0.05).view(-1, 1, 1)).clamp(parameters["shade_least"], parameters["shade_most"]) ** parameters["shade_power"]
    repainted = (paper_colour.view(-1, 3, 1, 1) * shade.unsqueeze(1)).clamp(0.0, 1.0)
    return torch.where(is_table.unsqueeze(1), repainted, batch)



def grown(
        mask: torch.Tensor,
        size: int
) -> torch.Tensor:
    """Return a (height, width) boolean mask grown by a square of ``size`` pixels (an odd number)."""

    return torch.nn.functional.max_pool2d(mask.float().view(1, 1, *mask.shape), size, stride = 1, padding = size // 2).view(*mask.shape) > 0



def made_up_end(
        picture: torch.Tensor,
        reach_forward_left_cm: numpy.ndarray,
        parameters: Dict
) -> Optional[torch.Tensor]:
    """
    Erase the line that leads on from the puck in one picture, so that it shows a line ending under the puck (14.11.26).

    The model had 19 demonstrated stops on lines that end in the middle of the paper and read the paper's edge as its
    stop cue (2026-11-14_stop_check.md). Any picture of the way gives one more end: the puck is found by its red, the
    line by its dark blue; the ink within "seed_radius_px" of the puck on the side the demonstration goes on to
    (``reach_forward_left_cm``) and all ink that hangs together with it (dashes are joined first) is painted over with
    the paper's colour nearby. The line behind the puck, and other parts of the track that do not hang together with
    it, stay.

    Parameters
    ----------
    picture : torch.Tensor
        (3, height, width) float RGB between 0 and 1, as cached.
    reach_forward_left_cm : numpy.ndarray
        Where the demonstration's tip is five steps later, forward and left in the picture's frame.
    parameters : dict
        The colour and size limits ("made_up_end" of the run settings); sizes are in pixels of a 160-pixel-wide picture.

    Returns
    -------
    torch.Tensor or None
        The picture with its made-up end, or None when no puck is seen, the demonstration hardly moves on, or no ink
        lies ahead of the puck.
    """

    red, green, blue = picture[0], picture[1], picture[2]
    brightness = picture.mean(dim = 0)
    height, width = brightness.shape
    scale = width / 160.0
    is_puck = (((red - green) > parameters["puck_red_over_green_least"]) & ((red - blue) > parameters["puck_red_over_blue_least"])
               & (red > parameters["puck_red_least"]))
    reach = float(numpy.linalg.norm(reach_forward_left_cm))
    if int(is_puck.sum()) < parameters["puck_pixels_least"] * scale ** 2 or reach < parameters["reach_least_cm"]:
        return None
    rows, columns = torch.meshgrid(torch.arange(height, dtype = torch.float32), torch.arange(width, dtype = torch.float32), indexing = "ij")
    puck_row, puck_column = rows[is_puck].mean(), columns[is_puck].mean()
    # forward is up in the picture (smaller rows), left is to its left (smaller columns)
    ahead_row, ahead_column = -float(reach_forward_left_cm[0]) / reach, -float(reach_forward_left_cm[1]) / reach
    along = (rows - puck_row) * ahead_row + (columns - puck_column) * ahead_column
    distance = torch.sqrt((rows - puck_row) ** 2 + (columns - puck_column) ** 2)

    def odd(pixels: float) -> int:
        return 2 * int(round(pixels * scale / 2)) + 1

    blue_over_red = blue - red
    # the line is thin, so its ink is darker than what lies around it; a broad shadow is as dark as its surroundings
    window = odd(parameters["ink_contrast_window_px"] - 1)
    brightness_nearby = torch.nn.functional.avg_pool2d(brightness.view(1, 1, height, width), window, stride = 1, padding = window // 2,
                                                       count_include_pad = False)[0, 0]
    is_ink = ((brightness > parameters["ink_brightness_least"]) & (brightness < parameters["ink_brightness_most"])
              & (blue_over_red > parameters["ink_blue_over_red_least"]) & (blue_over_red < parameters["ink_blue_over_red_most"])
              & ((brightness_nearby - brightness) > parameters["ink_darker_than_nearby_least"]) & ~grown(is_puck, odd(2)))
    left, top, right, bottom = [int(round(value * scale)) for value in parameters["robot_area_px"]]
    is_ink[top:bottom, left:right] = False
    # dashes are joined into one line; then everything that hangs together with the ink just ahead of the puck goes
    is_joined = grown(is_ink, odd(parameters["joining_px"] - 1))
    is_flooded = is_joined & (along > -parameters["behind_margin_px"] * scale) & (distance < parameters["seed_radius_px"] * scale)
    if not bool((is_flooded & is_ink).any()):
        return None
    for _ in range(int(parameters["flood_steps"] * scale)):
        larger = grown(is_flooded, 3) & is_joined
        if bool((larger == is_flooded).all()):
            break
        is_flooded = larger
    is_erased = grown(is_ink & is_flooded, odd(parameters["erase_margin_px"] - 1)) & ~is_puck
    is_paper = ((blue_over_red > -parameters["paper_red_over_blue_most"]) & (brightness > parameters["paper_brightness_least"]) & ~is_erased)
    window = odd(parameters["fill_window_px"] - 1)
    paper_sum = torch.nn.functional.avg_pool2d((picture * is_paper).unsqueeze(0), window, stride = 1, padding = window // 2,
                                               count_include_pad = False)[0]
    paper_share = torch.nn.functional.avg_pool2d(is_paper.float().view(1, 1, height, width), window, stride = 1, padding = window // 2,
                                                 count_include_pad = False)[0, 0]
    usual_colour = torch.tensor(parameters["paper_colour"], dtype = picture.dtype).view(3, 1, 1).expand_as(picture)
    paper_nearby = torch.where(paper_share > 0.05, paper_sum / paper_share.clamp(min = 0.05), usual_colour)
    return torch.where(is_erased.unsqueeze(0), paper_nearby, picture)



def made_up_ends(
        pictures: torch.Tensor,
        arrays: Dict[str, numpy.ndarray],
        rows: numpy.ndarray,
        settings: Dict,
        randomness_generator: numpy.random.Generator
) -> Tuple[torch.Tensor, numpy.ndarray]:
    """
    Give a random share of a training batch's pictures a made-up line end.

    Parameters
    ----------
    pictures : torch.Tensor
        (batch, height, width, 3), uint8 RGB, the batch's cached pictures.
    arrays : dict
        The targets of all rows; "chunk" gives where the demonstration goes on to, "stop" which rows are stops already.
    rows : numpy.ndarray
        The batch's rows.
    settings : dict
        The run settings: "made_up_end_probability" and "made_up_end".
    randomness_generator : numpy.random.Generator
        Source of the draw.

    Returns
    -------
    tuple of torch.Tensor and numpy.ndarray
        The batch's pictures, and which of them now show a made-up end (their targets become a stop).
    """

    if pictures.shape[-1] != 3:
        raise ValueError("made-up ends are written for one picture per row")
    is_drawn = randomness_generator.random(len(rows)) < settings["made_up_end_probability"]
    is_ended = numpy.zeros(len(rows), dtype = bool)
    reach = arrays["chunk"][rows].reshape(len(rows), -1, 2)[:, -1, :]
    pictures = pictures.clone()
    for position in numpy.flatnonzero(is_drawn & (arrays["stop"][rows] < 0.5)):
        ended = made_up_end(pictures[position].permute(2, 0, 1).float().div(255.0), reach[position], settings["made_up_end"])
        if ended is not None:
            pictures[position] = (ended.permute(1, 2, 0) * 255.0).round().clamp(0, 255).to(torch.uint8)
            is_ended[position] = True
    return pictures, is_ended



def ended_targets(
        targets: Dict[str, torch.Tensor],
        is_ended: numpy.ndarray
) -> Dict[str, torch.Tensor]:
    """Return a batch's targets with the rows of made-up ends turned into stops: no move, no turn, stop."""

    ended = torch.from_numpy(is_ended)
    targets = dict(targets)
    for name in ("exact", "chunk"):
        targets[name] = targets[name].clone()
        targets[name][ended] = 0.0
    targets["stop"] = targets["stop"].clone()
    targets["stop"][ended] = 1.0
    targets["twist"] = targets["twist"].clone()
    targets["twist"][ended] = 1
    targets["direction"] = targets["direction"].clone()
    targets["direction"][ended] = -1
    return targets



def turned_pictures(
        pictures: torch.Tensor,
        settings: Dict,
        randomness_generator: numpy.random.Generator
) -> Tuple[torch.Tensor, numpy.ndarray]:
    """
    Turn every picture of a training batch by its own small random angle about the place of the rod's tip (15.11.26,
    Emranul's idea: use the same pictures turned, so that the model meets more line directions without new recordings).

    A positive angle turns what the picture shows counter-clockwise as seen on the screen: a line that pointed up now
    points up and to the left. The step that belongs to the picture has to be turned the same way
    (``turned_targets``). The robot's own parts turn with the picture, which never happens at the robot, so the angles
    are kept small ("turn_most_deg" of "augmentation").

    Parameters
    ----------
    pictures : torch.Tensor
        (batch, height, width, 3 x pictures per row), uint8 RGB.
    settings : dict
        The run settings: "augmentation" with "turn_most_deg" and "turn_centre_px" (column and row of the rod's tip in
        a 160-pixel-wide picture).
    randomness_generator : numpy.random.Generator
        Source of the angles.

    Returns
    -------
    tuple of torch.Tensor and numpy.ndarray
        The turned pictures (uint8, the same shape) and each picture's angle in degrees.
    """

    augmentation = settings["augmentation"]
    count, height, width = pictures.shape[0], pictures.shape[1], pictures.shape[2]
    angles_deg = randomness_generator.uniform(-augmentation["turn_most_deg"], augmentation["turn_most_deg"], count)
    angles = torch.from_numpy(numpy.radians(angles_deg)).float()
    cosine, sine = torch.cos(angles), torch.sin(angles)
    scale = width / 160.0
    # the centre in the grid's own units (-1 to 1 across the picture, columns first)
    centre_x = 2.0 * augmentation["turn_centre_px"][0] * scale / width - 1.0
    centre_y = 2.0 * augmentation["turn_centre_px"][1] * scale / height - 1.0
    # where each pixel of the turned picture comes from: the inverse turn about the centre, with the picture's
    # width-to-height ratio, because the grid's units are not square
    matrix = torch.zeros(count, 2, 3)
    matrix[:, 0, 0] = cosine
    matrix[:, 0, 1] = -sine * height / width
    matrix[:, 1, 0] = sine * width / height
    matrix[:, 1, 1] = cosine
    matrix[:, 0, 2] = centre_x - (matrix[:, 0, 0] * centre_x + matrix[:, 0, 1] * centre_y)
    matrix[:, 1, 2] = centre_y - (matrix[:, 1, 0] * centre_x + matrix[:, 1, 1] * centre_y)
    batch = pictures.permute(0, 3, 1, 2).float()
    grid = torch.nn.functional.affine_grid(matrix, list(batch.shape), align_corners = False)
    turned = torch.nn.functional.grid_sample(batch, grid, mode = "bilinear", padding_mode = "border", align_corners = False)
    return turned.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1), angles_deg



def turned_targets(
        targets: Dict[str, torch.Tensor],
        angles_deg: numpy.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Return a batch's targets with every step turned by its picture's angle (see ``turned_pictures``).

    Forward and left of the exact step and of every step of the chunk turn together (positive angle: towards the
    left); the forearm, lean and hand changes are the arm's own and stay. The move's name is not used for a turned
    picture (the eight names are 45 degrees apart).
    """

    angles = torch.from_numpy(numpy.radians(angles_deg)).float()
    cosine, sine = torch.cos(angles), torch.sin(angles)
    targets = dict(targets)
    exact = targets["exact"].clone()
    exact[:, 0] = targets["exact"][:, 0] * cosine - targets["exact"][:, 1] * sine
    exact[:, 1] = targets["exact"][:, 0] * sine + targets["exact"][:, 1] * cosine
    targets["exact"] = exact
    chunk = targets["chunk"].clone().view(len(angles), -1, 2)
    forward, left = chunk[:, :, 0].clone(), chunk[:, :, 1].clone()
    chunk[:, :, 0] = forward * cosine.view(-1, 1) - left * sine.view(-1, 1)
    chunk[:, :, 1] = forward * sine.view(-1, 1) + left * cosine.view(-1, 1)
    targets["chunk"] = chunk.view(len(angles), -1)
    targets["direction"] = torch.full_like(targets["direction"], -1)
    return targets



def mirrored_batch(
        targets: Dict[str, torch.Tensor],
        is_mirrored: numpy.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Return a batch's turned targets with left and right swapped where the picture was mirrored after its turn: the
    same swaps as ``batch_targets`` makes, on targets whose move names are not used.
    """

    if not is_mirrored.any():
        return targets
    mirrored = torch.from_numpy(is_mirrored)
    targets = dict(targets)
    twist = targets["twist"].clone()
    twist[mirrored] = 2 - twist[mirrored]
    targets["twist"] = twist
    exact = targets["exact"].clone()
    for column in (1, 2, 4):
        exact[mirrored, column] = -exact[mirrored, column]
    targets["exact"] = exact
    chunk = targets["chunk"].clone()
    chunk[mirrored, 1::2] = -chunk[mirrored, 1::2]
    targets["chunk"] = chunk
    return targets



def augmented_single_pictures(
        pictures: torch.Tensor,
        settings: Dict,
        randomness_generator: numpy.random.Generator,
        is_training: bool
) -> Tuple[torch.Tensor, numpy.ndarray]:
    """
    Turn a batch of cached pictures, one per row, into normalised network input, augmented when training.

    Parameters
    ----------
    pictures : torch.Tensor
        (batch, height, width, 3), uint8 RGB.
    settings : dict
        The run settings: "augmentation" (the strength of each kind) and "normalisation".
    randomness_generator : numpy.random.Generator
        Source of every random choice, so a run can be repeated.
    is_training : bool
        Augment only when True.

    Returns
    -------
    tuple of torch.Tensor and numpy.ndarray
        (batch, 3, height, width) float input, and which pictures were mirrored.
    """

    batch = pictures.permute(0, 3, 1, 2).float().div(255.0)
    count = batch.shape[0]
    is_mirrored = numpy.zeros(count, dtype = bool)
    augmentation = settings["augmentation"]
    if is_training:
        if augmentation.get("table_as_paper_probability", 0.0) > 0:
            # drawn only when switched on, so that every other run keeps its random numbers
            is_repainted = randomness_generator.random(count) < augmentation["table_as_paper_probability"]
            batch = table_as_paper(batch, is_repainted, settings["table_as_paper"])
        brightness = torch.from_numpy(randomness_generator.uniform(1 - augmentation["brightness"], 1 + augmentation["brightness"], count)).float().view(-1, 1, 1, 1)
        contrast = torch.from_numpy(randomness_generator.uniform(1 - augmentation["contrast"], 1 + augmentation["contrast"], count)).float().view(-1, 1, 1, 1)
        saturation = torch.from_numpy(randomness_generator.uniform(1 - augmentation["saturation"], 1 + augmentation["saturation"], count)).float().view(-1, 1, 1, 1)
        gamma = torch.from_numpy(numpy.exp(randomness_generator.uniform(-augmentation["gamma_log"], augmentation["gamma_log"], count))).float().view(-1, 1, 1, 1)
        batch = batch * brightness
        if augmentation["channel_gain"] > 0:
            # a small gain per colour channel imitates the bluish daylight cast and the violet puck in shadow
            gains = randomness_generator.uniform(1 - augmentation["channel_gain"], 1 + augmentation["channel_gain"], (count, 3))
            batch = batch * torch.from_numpy(gains).float().view(count, 3, 1, 1)
        mean_grey = batch.mean(dim = (1, 2, 3), keepdim = True)
        batch = (batch - mean_grey) * contrast + mean_grey
        grey = batch.mean(dim = 1, keepdim = True)
        batch = (batch - grey) * saturation + grey
        batch = batch.clamp(0.0, 1.0) ** gamma
        for index in numpy.flatnonzero(randomness_generator.random(count) < augmentation["blur_probability"]):
            kernel = gaussian_kernel(float(randomness_generator.uniform(0.3, augmentation["blur_sigma_most"])))
            size = len(kernel)
            picture = batch[index:index + 1].reshape(3, 1, *batch.shape[2:])
            picture = torch.nn.functional.conv2d(torch.nn.functional.pad(picture, (size // 2, size // 2, 0, 0), mode = "replicate"), kernel.view(1, 1, 1, -1))
            picture = torch.nn.functional.conv2d(torch.nn.functional.pad(picture, (0, 0, size // 2, size // 2), mode = "replicate"), kernel.view(1, 1, -1, 1))
            batch[index] = picture.reshape(3, *batch.shape[2:])
        if augmentation["noise_most"] > 0:
            noise_level = torch.from_numpy(randomness_generator.uniform(0, augmentation["noise_most"], count)).float().view(-1, 1, 1, 1)
            batch = batch + noise_level * torch.from_numpy(randomness_generator.standard_normal(batch.shape).astype(numpy.float32))
        shift = augmentation["shift_px"]
        if shift > 0:
            padded = torch.nn.functional.pad(batch, (shift, shift, shift, shift), mode = "replicate")
            height, width = batch.shape[2:]
            offsets = randomness_generator.integers(0, 2 * shift + 1, size = (count, 2))
            batch = torch.stack([padded[index, :, offsets[index, 0]:offsets[index, 0] + height, offsets[index, 1]:offsets[index, 1] + width]
                                 for index in range(count)])
        if augmentation["mirror_probability"] > 0:
            is_mirrored = randomness_generator.random(count) < augmentation["mirror_probability"]
            if is_mirrored.any():
                batch[torch.from_numpy(is_mirrored)] = batch[torch.from_numpy(is_mirrored)].flip(-1)
        batch = batch.clamp(0.0, 1.0)
    if settings["input_channels"] == "line_and_puck":
        return line_and_puck_maps(batch), is_mirrored
    mean = torch.tensor(settings["normalisation"]["mean"]).view(1, 3, 1, 1)
    standard_deviation = torch.tensor(settings["normalisation"]["std"]).view(1, 3, 1, 1)
    return (batch - mean) / standard_deviation, is_mirrored



def line_and_puck_maps(batch: torch.Tensor) -> torch.Tensor:
    """
    Turn RGB pictures into three maps that keep only what the push depends on: the dark line, the red puck and grey.

    "dark" is how much darker a pixel is than its picture's median grey (so the same line looks the same in bright
    and dim light); "red" is how much the red channel exceeds the larger of green and blue; "grey" is the grey
    picture with its own median taken off. All three are centred and scaled to about unit spread.

    Parameters
    ----------
    batch : torch.Tensor
        (batch, 3, height, width), RGB in 0 to 1.

    Returns
    -------
    torch.Tensor
        (batch, 3, height, width).
    """

    grey = batch.mean(dim = 1, keepdim = True)
    median = grey.flatten(1).median(dim = 1).values.view(-1, 1, 1, 1)
    dark = torch.relu(median - grey) / (median + 1e-3)
    red = torch.relu(batch[:, 0:1] - torch.maximum(batch[:, 1:2], batch[:, 2:3]))
    centred_grey = (grey - median) / (median + 1e-3)
    return torch.cat([dark * 4.0 - 0.5, red * 6.0 - 0.3, centred_grey * 2.0], dim = 1)



def batch_targets(
        arrays: Dict[str, numpy.ndarray],
        rows: numpy.ndarray,
        is_mirrored: numpy.ndarray,
        mirror_class_index: numpy.ndarray,
        direction_class_weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Return the targets of a batch, mirrored where the picture was mirrored.

    A mirrored picture swaps left and right: the left and right classes swap, the left component of the step, the
    forearm and hand changes and the left components of the chunk change sign, and a twist to the left becomes a twist
    to the right.
    """

    direction = arrays["direction"][rows].copy()
    twist = arrays["twist"][rows].copy()
    exact = arrays["exact"][rows].copy()
    chunk = arrays["chunk"][rows].copy()
    if is_mirrored.any():
        is_labelled = is_mirrored & (direction >= 0)
        direction[is_labelled] = mirror_class_index[direction[is_labelled]]
        twist[is_mirrored] = 2 - twist[is_mirrored]
        exact[is_mirrored, 1] *= -1.0
        exact[is_mirrored, 2] *= -1.0
        exact[is_mirrored, 4] *= -1.0
        chunk[is_mirrored, 1::2] *= -1.0
    return {
        "direction": torch.from_numpy(direction),
        "twist": torch.from_numpy(twist),
        "exact": torch.from_numpy(exact),
        "chunk": torch.from_numpy(chunk),
        "stop": torch.from_numpy(arrays["stop"][rows]),
        "sample_weight": torch.from_numpy(arrays["sample_weight"][rows]),
        "direction_class_weights": direction_class_weights
    }



def memory_batch(
        arrays: Dict[str, numpy.ndarray],
        rows: numpy.ndarray,
        is_mirrored: numpy.ndarray,
        settings: Dict,
        randomness_generator: Optional[numpy.random.Generator] = None
) -> torch.Tensor:
    """
    The direction memory of a batch (15.11.26): mirrored where the picture was (the left components change sign); in
    training ("randomness_generator" given) hidden for a share "memory_hide_probability" of the pictures (all zero, as
    at a run's start), so that the picture keeps doing the work and the model cannot only repeat its past, and the known
    arrows shaken by "memory_noise_cm", as the robot's own path is less tidy than a demonstration.
    """

    memory = arrays["memory"][rows].astype(numpy.float32).copy()
    arrow_count = 2 * memory.shape[1] // 3
    if is_mirrored.any():
        memory[numpy.ix_(numpy.flatnonzero(is_mirrored), numpy.arange(1, arrow_count, 2))] *= -1.0
    if randomness_generator is not None:
        flags = numpy.repeat(memory[:, arrow_count:], 2, axis = 1)
        memory[:, :arrow_count] += (randomness_generator.normal(0.0, settings["memory_noise_cm"], (len(rows), arrow_count)) * flags).astype(numpy.float32)
        memory[randomness_generator.random(len(rows)) < settings["memory_hide_probability"]] = 0.0
    return torch.from_numpy(memory)



def model_outputs(
        model: PictureToMoveModel,
        pictures: torch.Tensor,
        rows: numpy.ndarray,
        settings: Dict,
        arrays: Optional[Dict[str, numpy.ndarray]] = None,
        is_memory_hidden: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Return the model's outputs for the given rows, without augmentation, in batches, on the processor. A model with a
    direction memory gets the rows' memory from ``arrays`` (none when ``is_memory_hidden``, the copycat check).
    """

    model.eval()
    device = next(model.parameters()).device
    parts = []
    with torch.inference_mode():
        for start in range(0, len(rows), settings["evaluation_batch_size"]):
            batch_rows = rows[start:start + settings["evaluation_batch_size"]]
            inputs, _ = augmented_pictures(pictures[batch_rows], settings, numpy.random.default_rng(0), is_training = False)
            memory = None
            if model.memory_size and not is_memory_hidden:
                memory = memory_batch(arrays, batch_rows, numpy.zeros(len(batch_rows), dtype = bool), settings).to(device)
            parts.append({name: value.cpu() for name, value in model(inputs.to(device), memory).items()})
    return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}



def training_device(settings: Dict) -> torch.device:
    """Return the device to train on: with "device" "auto", the graphics card when PyTorch sees one, else the processor."""

    if settings["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(settings["device"])



def train_fold(
        settings: Dict,
        pictures: torch.Tensor,
        arrays: Dict[str, numpy.ndarray],
        train_rows: numpy.ndarray,
        validation_rows: numpy.ndarray,
        class_names: List[str],
        seed: int
) -> Tuple[PictureToMoveModel, List[Dict]]:
    """
    Train one model on one fold, keeping the weights of the epoch with the lowest validation loss.

    Returns
    -------
    tuple of PictureToMoveModel and list of dict
        The model and the loss per epoch.
    """

    torch.manual_seed(seed)
    randomness_generator = numpy.random.default_rng(seed)
    device = training_device(settings)
    output_sizes = design_outputs(settings["design"], len(class_names), settings["chunk_steps"], settings.get("is_stop_separate", False))
    model_class = SequencePictureModel if settings.get("sequence_frames") else PictureToMoveModel
    model = model_class(settings, output_sizes).to(device)
    backbone_parameters = list(model.backbone.parameters())
    # 15.11.26: every part outside the backbone (head, memory layer, recurrent layer) learns at the head's rate; before,
    # only model.head was handed to the optimiser and the direction memory's own layer stayed at its random start
    head_parameters = [value for name, value in model.named_parameters() if not name.startswith("backbone.")]
    backbone_rate = settings["learning_rate"] * (settings["backbone_learning_rate_factor"] if settings["is_pretrained"] else 1.0)
    optimiser = torch.optim.AdamW([{"params": backbone_parameters, "lr": backbone_rate}, {"params": head_parameters, "lr": settings["learning_rate"]}],
                                  weight_decay = settings["weight_decay"])
    rows_per_epoch = max(settings["batch_size"], int(round(settings["train_share_per_epoch"] * len(train_rows))))
    steps_per_epoch = math.ceil(rows_per_epoch / settings["batch_size"])
    total_steps = steps_per_epoch * settings["epochs"]
    warm_up_steps = steps_per_epoch * settings["warm_up_epochs"]
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / max(1, warm_up_steps))
                                                 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(1, total_steps)))))
    class_index = {name: index for index, name in enumerate(class_names)}
    mirror_class_index = numpy.array([class_index[MIRRORED_DIRECTION[name]] for name in class_names])
    direction_class_weights = torch.from_numpy(class_weights(arrays["direction"][train_rows], len(class_names), settings["class_weight_power"]))
    head_settings = dict(settings)

    history = []
    best_loss = math.inf
    best_state = None
    epochs_without_gain = 0
    for epoch in range(settings["epochs"]):
        model.train()
        started = time.perf_counter()
        # neighbouring pictures are about 1 cm apart and look alike, so an epoch may use a random share of them
        order = randomness_generator.permutation(train_rows)[:rows_per_epoch]
        epoch_loss = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            batch_pictures = pictures[rows]
            is_ended = None
            if settings.get("made_up_end_probability", 0.0) > 0:
                # drawn only when switched on, so that every other run keeps its random numbers
                batch_pictures, is_ended = made_up_ends(batch_pictures, arrays, rows, settings, randomness_generator)
            turn_angles_deg = None
            if settings["augmentation"].get("turn_most_deg", 0.0) > 0:
                # also drawn only when switched on; the turn comes before the mirror, the targets follow in that order
                batch_pictures, turn_angles_deg = turned_pictures(batch_pictures, settings, randomness_generator)
            inputs, is_mirrored = augmented_pictures(batch_pictures, settings, randomness_generator, is_training = True)
            if turn_angles_deg is None:
                batch = batch_targets(arrays, rows, is_mirrored, mirror_class_index, direction_class_weights)
            else:
                unmirrored = batch_targets(arrays, rows, numpy.zeros(len(rows), dtype = bool), mirror_class_index, direction_class_weights)
                batch = mirrored_batch(turned_targets(unmirrored, turn_angles_deg), is_mirrored)
            if is_ended is not None and is_ended.any():
                batch = ended_targets(batch, is_ended)
            targets = {name: value.to(device) for name, value in batch.items()}
            memory = memory_batch(arrays, rows, is_mirrored, settings, randomness_generator).to(device) if model.memory_size else None
            loss = head_losses(model(inputs.to(device), memory), targets, head_settings)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            epoch_loss += float(loss) * len(rows)
        # the exam model ("split" "all") keeps no check pictures, so it has no validation loss and keeps its last weights
        if len(validation_rows) == 0:
            history.append({"epoch": epoch + 1, "train_loss": round(epoch_loss / len(order), 4), "validation_loss": None,
                            "seconds": round(time.perf_counter() - started, 1)})
            print(f"{settings['name']} | epoch {epoch + 1} | train {history[-1]['train_loss']} | {history[-1]['seconds']} s", flush = True)
            continue
        outputs = model_outputs(model, pictures, validation_rows, settings, arrays)
        validation_targets = batch_targets(arrays, validation_rows, numpy.zeros(len(validation_rows), dtype = bool), mirror_class_index, direction_class_weights)
        validation_loss = float(head_losses(outputs, validation_targets, head_settings))
        history.append({"epoch": epoch + 1, "train_loss": round(epoch_loss / len(order), 4), "validation_loss": round(validation_loss, 4),
                        "seconds": round(time.perf_counter() - started, 1)})
        print(f"{settings['name']} | epoch {epoch + 1} | train {history[-1]['train_loss']} | validation {history[-1]['validation_loss']} | {history[-1]['seconds']} s", flush = True)
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= settings["patience"]:
                break
    # validation loss is a poor guide to robot success (robomimic, Mandlekar and others 2021), so a fixed number of
    # epochs with the last weights is the default for the final model; the best-validation weights stay an option
    if not settings.get("is_last_checkpoint_kept") and best_state is not None:
        model.load_state_dict(best_state)
    return model, history



def milliseconds_per_picture(
        model: PictureToMoveModel,
        settings: Dict
) -> float:
    """Return the median time of one picture through the model on this laptop, as the robot loop would run it."""

    model.eval()
    height, width = settings["picture_size"][1], settings["picture_size"][0]
    picture = torch.rand(1, 3 * (settings.get("sequence_frames") or settings.get("picture_count", 1)), height, width)
    timings = []
    with torch.inference_mode():
        for repeat in range(23):
            started = time.perf_counter()
            model(picture)
            if repeat >= 3:
                timings.append(1000.0 * (time.perf_counter() - started))
    return round(float(numpy.median(timings)), 1)



def run_training(
        run_name: str,
        fold_names: Optional[List[str]] = None,
        overrides: Optional[Dict] = None
) -> Dict:
    """
    Train and score one named run on every fold of its split (or on the named folds), and save the results.

    Parameters
    ----------
    run_name : str
        A key of ``TRAINING_RUNS``.
    fold_names : list of str, optional
        Only these folds, for example ``["without_path_04"]``.
    overrides : dict, optional
        Settings changed for this call only, for a quick check (for example "epochs" 1 and "train_row_limit" 512);
        the results then go to a folder whose name ends in "_check".

    Returns
    -------
    dict
        The settings, the measures per fold and their summary.
    """

    settings = merged_run_settings(run_name)
    settings.update(overrides or {})
    settings["device"] = str(training_device(settings))
    torch.set_num_threads(settings["threads"])
    # the pictures always have one size, so the graphics card may pick its fastest convolution once (not bit-for-bit
    # repeatable, unlike the processor runs)
    torch.backends.cudnn.benchmark = settings["device"] == "cuda"
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    settings["chunk_steps"] = target_parameters["chunk_steps"]
    # 14.11.26: a run may name its own training index ("processed_folder"), so a model on round 1 plus part of round 2
    # can be trained while the round 1 index stays as it is
    data_parameters = dict(TRAINING_DATA_PARAMETERS)
    if settings.get("processed_folder"):
        data_parameters["processed_folder"] = settings["processed_folder"]
    frames = load_training_frames(target_parameters, data_parameters)
    cache = numpy.load(Path(data_parameters["processed_folder"]) / f"images_{settings['picture_cache']}.npy", mmap_mode = "r")
    if settings.get("sequence_frames"):
        # 15.11.26: the recurrent model reads the whole cache once (1.3 GB) and stacks each batch's sequences on demand
        if settings.get("memory_distances_cm") or settings.get("picture_count", 1) > 1:
            raise ValueError("the recurrent model takes neither the direction memory nor a second stacked picture")
        pictures = SequencePictures(numpy.array(cache), frames, settings["sequence_frames"], settings["sequence_stride"])
    else:
        pictures = torch.from_numpy(numpy.array(cache[frames["frame_index"].to_numpy()]))
    if settings.get("picture_count", 1) > 1 and not settings.get("sequence_frames"):
        # 14.11.26: the picture before each one (the camera's previous picture in the same episode, also where the
        # cleaning dropped that row; an episode's first picture is repeated), stacked behind it as three more channels
        earlier_pictures = [torch.from_numpy(numpy.array(cache[previous_frame_indices(frames, lag)]))
                            for lag in range(1, settings["picture_count"])]
        pictures = torch.cat([pictures] + earlier_pictures, dim = -1)
    if settings["backbone"] != "small_cnn":
        data_configuration = timm.data.resolve_data_config({}, model = timm.create_model(settings["backbone"], pretrained = False, num_classes = 0))
        settings["normalisation"] = {"mean": list(data_configuration["mean"]), "std": list(data_configuration["std"])}

    arrays, class_names, is_moving = training_arrays(frames, settings, target_parameters)
    if settings.get("memory_distances_cm") and settings["augmentation"].get("turn_most_deg", 0.0) > 0:
        raise ValueError("the direction memory is not turned with turned pictures; switch one of them off")
    is_exam_model = settings["split"] == "all"
    if is_exam_model:
        # the exam model learns from every sheet and every episode; its test is the robot, not a held-out sheet
        folds = [{"name": "all_sheets", "train": numpy.ones(len(frames), dtype = bool), "test": numpy.zeros(len(frames), dtype = bool)}]
    elif settings["split"] == "sheets":
        folds = leave_one_sheet_out_folds(frames)
    elif settings["split"] == "new_shapes":
        folds = new_shape_folds(frames, settings.get("new_track_train_count"), settings["seed"])
    elif settings["split"] == "round_1_to_new":
        folds = round_1_to_new_folds(frames)
    else:
        folds = episode_folds(frames, target_parameters["episode_fold_count"], target_parameters["seed"])
    if fold_names:
        folds = [fold for fold in folds if fold["name"] in fold_names]

    run_date = settings["date"] or datetime.date.today().isoformat()
    run_folder = Path(settings["output_root"]) / f"{run_date}_{run_name}{'_check' if overrides else ''}"
    (run_folder / "checkpoints").mkdir(parents = True, exist_ok = True)
    fold_results = []
    for fold_number, fold in enumerate(folds):
        is_validation = (numpy.zeros(len(frames), dtype = bool) if is_exam_model
                         else inner_validation_mask(frames, fold["train"], target_parameters["inner_validation_share"], target_parameters["seed"] + fold_number))
        train_rows = numpy.flatnonzero(fold["train"] & ~is_validation)
        validation_rows = numpy.flatnonzero(is_validation)
        test_rows = numpy.flatnonzero(fold["test"])
        if settings.get("train_row_limit"):
            limit_generator = numpy.random.default_rng(settings["seed"])
            train_rows = numpy.sort(limit_generator.choice(train_rows, size = min(len(train_rows), settings["train_row_limit"]), replace = False))
            validation_rows = validation_rows[:settings["train_row_limit"] // 4]
            test_rows = test_rows[:settings["train_row_limit"]]
        started = time.perf_counter()
        model, history = train_fold(settings, pictures, arrays, train_rows, validation_rows, class_names, settings["seed"] + fold_number)
        if is_exam_model:
            measures = {"fold": fold["name"], "train_rows": len(train_rows), "epochs_run": len(history), "minutes": round((time.perf_counter() - started) / 60.0, 1)}
            fold_results.append(measures)
            (run_folder / f"fold_{fold['name']}.json").write_text(json.dumps({"measures": measures, "history": history}, indent = 1), encoding = "utf-8")
            model = model.to("cpu")
            torch.save(model.state_dict(), run_folder / "checkpoints" / f"{fold['name']}.pt")
            print(f"{run_name} | {fold['name']} | trained on {len(train_rows)} pictures in {measures['minutes']} min", flush = True)
            continue
        predictions = predictions_from_outputs(model_outputs(model, pictures, test_rows, settings, arrays), class_names, settings["stop_target"])
        validation_predictions = predictions_from_outputs(model_outputs(model, pictures, validation_rows, settings, arrays), class_names, settings["stop_target"])
        measures = fold_measures(predictions, frames, test_rows, arrays, is_moving, validation_predictions, validation_rows)
        if model.memory_size:
            # the copycat check (15.11.26): the same test pictures with the memory hidden; a model that leans on its
            # memory alone falls far below its score with the memory
            hidden = predictions_from_outputs(model_outputs(model, pictures, test_rows, settings, arrays, is_memory_hidden = True), class_names, settings["stop_target"])
            hidden_measures = fold_measures(hidden, frames, test_rows, arrays, is_moving)
            measures.update({f"memory_hidden_{name}": hidden_measures.get(name) for name in ("path_within_45_deg", "within_45_deg", "fixed_two_episodes_stopped_right")})
        measures.update({"fold": fold["name"], "epochs_run": len(history), "minutes": round((time.perf_counter() - started) / 60.0, 1)})
        fold_results.append(measures)
        (run_folder / f"fold_{fold['name']}.json").write_text(json.dumps({"measures": measures, "history": history}, indent = 1), encoding = "utf-8")
        numpy.savez_compressed(run_folder / f"predictions_{fold['name']}.npz", frame_index = frames["frame_index"].to_numpy()[test_rows], **predictions)
        # saved and timed on the processor, as the robot loop on the laptop runs it
        model = model.to("cpu")
        torch.save(model.state_dict(), run_folder / "checkpoints" / f"{fold['name']}.pt")
        print(f"{run_name} | {fold['name']} | {json.dumps({key: measures.get(key) for key in ('within_45_deg', 'class_accuracy', 'episodes_stopped_right', 'episodes_stopped_early', 'twist_sign_accuracy')})}", flush = True)

    result = {
        "name": run_name,
        "settings": {key: (str(value) if isinstance(value, Path) else value) for key, value in settings.items()},
        "class_names": class_names,
        "milliseconds_per_picture": milliseconds_per_picture(model, settings),
        "folds": fold_results,
        "summary": {} if is_exam_model else summarised_folds(fold_results)
    }
    (run_folder / "metrics.json").write_text(json.dumps(result, indent = 1), encoding = "utf-8")
    return result



def main() -> None:

    parser = argparse.ArgumentParser(description = "Train one named run of TRAINING_RUNS and score it fold by fold.")
    parser.add_argument("run_name", choices = sorted(TRAINING_RUNS))
    parser.add_argument("--folds", nargs = "*", default = None, help = "only these folds, for example without_path_04")
    parser.add_argument("--epochs", type = int, default = None, help = "quick check: this many epochs")
    parser.add_argument("--train-row-limit", type = int, default = None, help = "quick check: train on this many pictures")
    arguments = parser.parse_args()
    overrides = {key: value for key, value in (("epochs", arguments.epochs), ("train_row_limit", arguments.train_row_limit)) if value is not None}
    result = run_training(arguments.run_name, arguments.folds, overrides)
    print(json.dumps(result["summary"], indent = 1))



if __name__ == "__main__":
    main()
