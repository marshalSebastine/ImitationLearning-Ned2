"""
Count the parameters and the multiply-adds per picture of the end-to-end models compared on 9.11.26.

The small network and the head are re-built here from their definitions in train_policy.py and training_probes.py
(read, not imported, so nothing is written into the project); the ResNet-18 comes from timm without weights.
Nothing is trained and nothing is timed.
"""

import json
import sys

import timm
import torch


SMALL_NETWORK_CHANNELS = [32, 64, 96, 128, 192]
POOLING_GRID = (3, 4)
HIDDEN_SIZE = 256
PICTURE_HEIGHT, PICTURE_WIDTH = 120, 160
DESIGN_OUTPUTS = {"classes": {"direction": 9, "twist": 3}, "exact": {"exact": 5, "stop": 1}, "chunk": {"chunk": 10, "exact": 5, "stop": 1}}


class SmallNetwork(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        layers = []
        width = 3
        for block_width in channels:
            layers += [torch.nn.Conv2d(width, block_width, 3, stride = 2, padding = 1, bias = False),
                       torch.nn.BatchNorm2d(block_width), torch.nn.ReLU(inplace = True),
                       torch.nn.Conv2d(block_width, block_width, 3, padding = 1, bias = False),
                       torch.nn.BatchNorm2d(block_width), torch.nn.ReLU(inplace = True)]
            width = block_width
        self.blocks = torch.nn.Sequential(*layers)
        self.num_features = width

    def forward_features(self, pictures):
        return self.blocks(pictures)


class Head(torch.nn.Module):

    def __init__(self, input_size, output_sizes):
        super().__init__()
        self.shared = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(input_size, HIDDEN_SIZE), torch.nn.GELU(), torch.nn.Dropout(0.2))
        self.outputs = torch.nn.ModuleDict({name: torch.nn.Linear(HIDDEN_SIZE, size) for name, size in output_sizes.items()})

    def forward(self, features):
        shared = self.shared(features)
        return {name: layer(shared) for name, layer in self.outputs.items()}


def multiply_adds(model, picture):
    """Return the multiply-adds of all convolutions and linear layers for one picture (forward hooks)."""

    total = {"value": 0}

    def convolution_hook(module, inputs, output):
        kernel = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total["value"] += output.numel() * kernel

    def linear_hook(module, inputs, output):
        total["value"] += module.in_features * module.out_features

    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(picture)
    for handle in handles:
        handle.remove()
    return total["value"]


class WholeModel(torch.nn.Module):

    def __init__(self, backbone, output_sizes):
        super().__init__()
        self.backbone = backbone
        self.head = Head(backbone.num_features * POOLING_GRID[0] * POOLING_GRID[1], output_sizes)

    def forward(self, pictures):
        feature_map = self.backbone.forward_features(pictures)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, POOLING_GRID).flatten(1)
        return self.head(pooled)


def main():
    results = {}
    picture = torch.rand(1, 3, PICTURE_HEIGHT, PICTURE_WIDTH)
    for backbone_name in ["small_cnn", "resnet18.tv_in1k", "mobilenetv3_large_100.ra_in1k", "efficientnet_b0.ra_in1k"]:
        for design, output_sizes in DESIGN_OUTPUTS.items():
            backbone = SmallNetwork(SMALL_NETWORK_CHANNELS) if backbone_name == "small_cnn" else timm.create_model(backbone_name, pretrained = False, num_classes = 0)
            model = WholeModel(backbone, output_sizes).eval()
            with torch.inference_mode():
                feature_map = backbone.forward_features(picture)
            results[f"{backbone_name} | {design}"] = {
                "backbone_parameters": sum(parameter.numel() for parameter in backbone.parameters()),
                "head_parameters": sum(parameter.numel() for parameter in model.head.parameters()),
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "feature_map_at_160_by_120": list(feature_map.shape[1:]),
                "multiply_adds_per_picture_160_by_120": multiply_adds(model, picture)
            }
    resnet = timm.create_model("resnet18.tv_in1k", pretrained = False, num_classes = 0).eval()
    results["resnet18 backbone at 224 x 224 (check against torchvision's 1.81 G)"] = {"multiply_adds_per_picture": multiply_adds(resnet, torch.rand(1, 3, 224, 224))}
    json.dump(results, sys.stdout, indent = 1)


if __name__ == "__main__":
    main()
