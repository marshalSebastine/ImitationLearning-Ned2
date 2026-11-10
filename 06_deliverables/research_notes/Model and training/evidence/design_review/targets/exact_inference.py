"""
Re-run the saved end-to-end exact-step models (2026-11-09_resnet18_finetune_exact, held-out sheets 03, 05, 06) on their
own test pictures, read only, to get the outputs the saved predictions leave out: the predicted tip move (forward and
left, cm) and twist, lean and hand changes. The network is re-built here exactly as train_policy.PictureToMoveModel
builds it (ResNet-18 from timm, a 3 x 4 grid of the last feature map, one hidden layer of 256), so no project module
is imported. The stop probability and the move angle are checked against the saved predictions.
Writes exact_outputs_<fold>.csv next to this script.
"""

import sys
import time
from pathlib import Path

import numpy
import pandas
import timm
import torch

from common_frames import OUTPUT_FOLDER, PROCESSED_FOLDER, RUNS_FOLDER

RUN_FOLDER = RUNS_FOLDER / "2026-11-09_resnet18_finetune_exact"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STANDARD_DEVIATION = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Head(torch.nn.Module):

    def __init__(self, input_size: int, output_sizes: dict, hidden_size: int, dropout: float):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_size))
        self.register_buffer("input_standard_deviation", torch.ones(input_size))
        self.shared = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(input_size, hidden_size), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.outputs = torch.nn.ModuleDict({name: torch.nn.Linear(hidden_size, size) for name, size in output_sizes.items()})

    def forward(self, features: torch.Tensor) -> dict:
        features = (features.float() - self.input_mean) / self.input_standard_deviation
        shared = self.shared(features)
        return {name: layer(shared) for name, layer in self.outputs.items()}


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("resnet18.tv_in1k", pretrained = False, num_classes = 0)
        self.head = Head(self.backbone.num_features * 12, {"exact": 5, "stop": 1}, 256, 0.2)

    def forward(self, pictures: torch.Tensor) -> dict:
        feature_map = self.backbone.forward_features(pictures)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, (3, 4)).flatten(1)
        return self.head(pooled)


def main(folds, run_folder: Path = RUN_FOLDER, prefix: str = "exact_outputs") -> None:
    torch.set_num_threads(3)
    cache = numpy.load(PROCESSED_FOLDER / "images_wide_160.npy", mmap_mode = "r")
    for fold in folds:
        started = time.perf_counter()
        saved = numpy.load(run_folder / f"predictions_{fold}.npz")
        model = Model()
        state = torch.load(run_folder / "checkpoints" / f"{fold}.pt", map_location = "cpu", weights_only = True)
        missing, unexpected = model.load_state_dict(state, strict = True), None
        model.eval()
        frame_index = saved["frame_index"]
        parts = []
        with torch.inference_mode():
            for start in range(0, len(frame_index), 128):
                batch = torch.from_numpy(numpy.array(cache[frame_index[start:start + 128]])).permute(0, 3, 1, 2).float().div(255.0)
                outputs = model((batch - MEAN) / STANDARD_DEVIATION)
                parts.append(torch.cat([outputs["exact"], torch.sigmoid(outputs["stop"])], dim = 1).numpy())
        values = numpy.concatenate(parts)
        table = pandas.DataFrame({"frame_index": frame_index, "forward_cm": values[:, 0], "left_cm": values[:, 1], "twist_deg": values[:, 2],
                                  "lean_deg": values[:, 3], "hand_deg": values[:, 4], "stop_probability": values[:, 5]})
        angle = numpy.degrees(numpy.arctan2(table["left_cm"], table["forward_cm"]))
        angle_gap = numpy.abs((angle - saved["angle_deg"] + 180.0) % 360.0 - 180.0)
        stop_gap = numpy.abs(table["stop_probability"] - saved["stop_probability"])
        print(f"{fold}: {len(table)} pictures in {time.perf_counter() - started:.0f} s; largest stop-probability gap to the saved "
              f"predictions {stop_gap.max():.2e}; largest angle gap {angle_gap.max():.2e} deg; twist gap {numpy.abs(table['twist_deg'] - saved['twist_deg']).max():.2e}", flush = True)
        table.to_csv(OUTPUT_FOLDER / f"{prefix}_{fold}.csv", index = False)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--final"]:
        main(sys.argv[2:], RUNS_FOLDER / "2026-11-09_resnet18_final_candidate", "final_outputs")
    else:
        main(sys.argv[1:] or ["without_path_03", "without_path_05", "without_path_06"])
