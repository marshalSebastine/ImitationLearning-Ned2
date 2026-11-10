"""
Frozen picture features for quick model comparisons (9.11.26, Model and training session).

A pretrained image model ("backbone") turns every cached picture into a list of numbers once; small prediction heads
are then trained on these numbers in seconds instead of hours (training_probes.py). Two features per backbone:
    global   the backbone's own summary of the whole picture (average of the last feature map, or the class token);
    grid     the last feature map averaged into a 4 x 4 grid of cells, so the head still knows WHERE things are
             (where the line runs relative to the puck), which a single summary can lose.
The script also times one picture through each backbone on this laptop, the number that matters in the robot loop.

Reads images_square_224.npy from the processed folder (training_dataset.py) and writes
features/<backbone>_global.npy, features/<backbone>_grid.npy (float16, frames.csv order) and features/speed.json.

Run from 02_code (downloads the pretrained weights from the Hugging Face hub on first use):
    ..\\.venv\\Scripts\\python.exe training_features.py
    ..\\.venv\\Scripts\\python.exe training_features.py resnet18_imagenet dinov2_small
Settings: TRAINING_FEATURE_PARAMETERS in config.py.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
import timm
import torch

from config import TRAINING_DATA_PARAMETERS, TRAINING_FEATURE_PARAMETERS



def merged_feature_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``TRAINING_FEATURE_PARAMETERS`` with any overrides merged over it."""

    merged = dict(TRAINING_FEATURE_PARAMETERS)
    merged.update(parameters or {})
    return merged



def build_backbone(
        backbone_settings: Dict,
        seed: int
) -> Tuple[torch.nn.Module, Dict]:
    """
    Create a backbone without its classification layer, in evaluation mode.

    Parameters
    ----------
    backbone_settings : dict
        "timm_name", "is_pretrained" and optional "creation_arguments" (for example the input size of a vision
        transformer).
    seed : int
        Seed for the random weights of a backbone that is not pretrained.

    Returns
    -------
    tuple of torch.nn.Module and dict
        The model and its input normalisation ("mean" and "std" per colour channel).
    """

    torch.manual_seed(seed)
    model = timm.create_model(
        backbone_settings["timm_name"],
        pretrained = backbone_settings["is_pretrained"],
        num_classes = 0,
        **backbone_settings.get("creation_arguments", {})
    )
    model.eval()
    data_configuration = timm.data.resolve_data_config({}, model = model)
    return model, {"mean": list(data_configuration["mean"]), "std": list(data_configuration["std"])}



def pooled_features(
        model: torch.nn.Module,
        pictures: torch.Tensor,
        grid_cells: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the global and the grid features of a batch of normalised pictures.

    A convolutional backbone returns a feature map (batch, channels, height, width); a vision transformer returns
    tokens (batch, prefix tokens plus patches, width), whose patch tokens are put back on their grid first.

    Parameters
    ----------
    model : torch.nn.Module
        A timm backbone without its classification layer.
    pictures : torch.Tensor
        (batch, 3, height, width), normalised.
    grid_cells : int
        Cells per side of the grid feature.

    Returns
    -------
    tuple of torch.Tensor
        (batch, channels) and (batch, channels * grid_cells * grid_cells).
    """

    feature_output = model.forward_features(pictures)
    global_feature = model.forward_head(feature_output, pre_logits = True)
    if feature_output.dim() == 4:
        feature_map = feature_output
    else:
        prefix_count = getattr(model, "num_prefix_tokens", 1)
        patch_tokens = feature_output[:, prefix_count:, :]
        side_height, side_width = model.patch_embed.dynamic_feat_size(pictures.shape[-2:]) \
            if hasattr(model.patch_embed, "dynamic_feat_size") else model.patch_embed.grid_size
        feature_map = patch_tokens.transpose(1, 2).reshape(pictures.shape[0], -1, side_height, side_width)
    grid_feature = torch.nn.functional.adaptive_avg_pool2d(feature_map, grid_cells).flatten(1)
    return global_feature, grid_feature



def extract_backbone_features(
        backbone_name: str,
        parameters: Dict,
        data_parameters: Dict
) -> Dict:
    """
    Compute and save the global and grid features of every cached picture for one backbone.

    Parameters
    ----------
    backbone_name : str
        A key of the "backbones" setting.
    parameters : dict
        The feature settings.
    data_parameters : dict
        Supplies "processed_folder".

    Returns
    -------
    dict
        The feature sizes and the time the extraction took.
    """

    processed_folder = Path(data_parameters["processed_folder"])
    feature_folder = processed_folder / "features"
    feature_folder.mkdir(parents = True, exist_ok = True)
    global_path = feature_folder / f"{backbone_name}_global.npy"
    grid_path = feature_folder / f"{backbone_name}_grid.npy"
    if global_path.exists() and grid_path.exists() and not parameters["is_overwritten"]:
        return {"backbone": backbone_name, "skipped": True}

    pictures = numpy.load(processed_folder / f"images_{parameters['picture_cache']}.npy", mmap_mode = "r")
    model, normalisation = build_backbone(parameters["backbones"][backbone_name], parameters["seed"])
    mean = torch.tensor(normalisation["mean"]).view(1, 3, 1, 1)
    standard_deviation = torch.tensor(normalisation["std"]).view(1, 3, 1, 1)
    batch_size = parameters["batch_size"]
    global_parts = []
    grid_parts = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(pictures), batch_size):
            batch = torch.from_numpy(numpy.array(pictures[start:start + batch_size]))
            batch = batch.permute(0, 3, 1, 2).float().div(255.0)
            batch = (batch - mean) / standard_deviation
            global_feature, grid_feature = pooled_features(model, batch, parameters["grid_cells"])
            global_parts.append(global_feature.to(torch.float16).numpy())
            grid_parts.append(grid_feature.to(torch.float16).numpy())
            if (start // batch_size) % 40 == 0:
                print(f"{backbone_name}: {start + len(batch)} of {len(pictures)} pictures, {time.perf_counter() - started:.0f} s", flush = True)
    numpy.save(global_path, numpy.concatenate(global_parts))
    numpy.save(grid_path, numpy.concatenate(grid_parts))
    return {
        "backbone": backbone_name,
        "global_size": int(global_parts[0].shape[1]),
        "grid_size": int(grid_parts[0].shape[1]),
        "seconds": round(time.perf_counter() - started, 1),
        "normalisation": normalisation
    }



def time_one_picture(
        backbone_name: str,
        parameters: Dict
) -> Dict:
    """
    Time one picture through a backbone, as the robot loop would run it.

    Parameters
    ----------
    backbone_name : str
        A key of the "backbones" setting.
    parameters : dict
        Supplies "backbones", "speed_input_sizes", "speed_repeats", "speed_threads" and "seed".

    Returns
    -------
    dict
        The parameter count and the median and 90th-percentile milliseconds per picture for each input size.
    """

    model, _ = build_backbone(parameters["backbones"][backbone_name], parameters["seed"])
    torch.set_num_threads(parameters["speed_threads"])
    result = {"backbone": backbone_name, "parameters_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
    for size_name, (width, height) in parameters["speed_input_sizes"].items():
        if parameters["backbones"][backbone_name].get("is_fixed_input") and size_name != "square_224":
            continue
        picture = torch.rand(1, 3, height, width)
        timings = []
        with torch.inference_mode():
            for repeat in range(parameters["speed_repeats"] + 3):
                started = time.perf_counter()
                model.forward_features(picture)
                if repeat >= 3:
                    timings.append(1000 * (time.perf_counter() - started))
        result[f"milliseconds_{size_name}_median"] = round(float(numpy.median(timings)), 1)
        result[f"milliseconds_{size_name}_90th"] = round(float(numpy.percentile(timings, 90)), 1)
    torch.set_num_threads(parameters["extraction_threads"])
    return result



def main(backbone_names: Optional[List[str]] = None) -> None:

    parameters = merged_feature_parameters()
    torch.set_num_threads(parameters["extraction_threads"])
    names = backbone_names or list(parameters["backbones"])
    feature_folder = Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "features"
    feature_folder.mkdir(parents = True, exist_ok = True)
    summary_path = feature_folder / "extraction_summary.json"
    summary = json.loads(summary_path.read_text(encoding = "utf-8")) if summary_path.exists() else {}
    for name in names:
        result = extract_backbone_features(name, parameters, TRAINING_DATA_PARAMETERS)
        if not result.get("skipped"):
            summary[name] = result
            summary_path.write_text(json.dumps(summary, indent = 2), encoding = "utf-8")
        print(json.dumps(result), flush = True)

    speed_path = feature_folder / "speed.json"
    speeds = json.loads(speed_path.read_text(encoding = "utf-8")) if speed_path.exists() else {}
    for name in names:
        speeds[name] = time_one_picture(name, parameters)
        speed_path.write_text(json.dumps(speeds, indent = 2), encoding = "utf-8")
        print(json.dumps(speeds[name]), flush = True)



if __name__ == "__main__":
    main(sys.argv[1:] or None)
