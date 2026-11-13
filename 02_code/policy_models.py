"""
The five policies trained by train_policy.py (Marshal, 13.11.26). All see the forearm picture through the same
ResNet-18; they differ in what they predict.

1. ``FactorisedClassifier``: the move name, the forearm label, the lean label and "at goal", each its own output.
2. ``ExactStepRegressor``: the exact step as numbers (tip forward and left, forearm and lean change) and a stop chance.
3. ``HistoryClassifier``: like 1, from this picture, the ones before it and the previous move.
4. ``ChunkTransformer``: the next steps at once, ACT-style (a transformer decoder reads the picture's feature map).
5. ``DiffusionPolicy``: the next steps drawn by a denoising diffusion model, so two valid answers are not averaged.
"""

import math
from typing import Dict, Optional

import torch
import torchvision
from torch import nn


def resnet_trunk(is_pretrained: bool) -> nn.Module:
    """
    ResNet-18 up to its last convolution block, giving a feature map of 512 channels.

    ImageNet weights are downloaded by torchvision on first use (``train_policy.py --check`` does this before a night).
    """

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if is_pretrained else None
    network = torchvision.models.resnet18(weights = weights)
    return nn.Sequential(*list(network.children())[:-2])


class PictureEncoder(nn.Module):
    """One picture to one 512-number feature: ResNet-18 then the average over the feature map."""

    def __init__(self, is_pretrained: bool):

        super().__init__()
        self.trunk = resnet_trunk(is_pretrained)
        self.width = 512


    def forward(self, pictures: torch.Tensor) -> torch.Tensor:
        return self.trunk(pictures).mean(dim = (2, 3))


class ClassHeads(nn.Module):
    """The four outputs of the classifiers: move, forearm label, lean label and the at-goal logit."""

    def __init__(
            self,
            feature_width: int,
            class_counts: Dict[str, int],
            dropout: float
    ):

        super().__init__()
        self.shared = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_width, 256), nn.ReLU(), nn.Dropout(dropout))
        self.move = nn.Linear(256, class_counts["move"])
        self.forearm = nn.Linear(256, class_counts["forearm"])
        self.lean = nn.Linear(256, class_counts["lean"])
        self.at_goal = nn.Linear(256, 1)


    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(features)
        return {"move": self.move(shared), "forearm": self.forearm(shared), "lean": self.lean(shared),
                "at_goal": self.at_goal(shared).squeeze(-1)}


class FactorisedClassifier(nn.Module):
    """Model 1: one picture, four separate class outputs."""

    kind = "classifier"

    def __init__(
            self,
            class_counts: Dict[str, int],
            model_parameters: Dict,
            is_pretrained: bool
    ):

        super().__init__()
        self.encoder = PictureEncoder(is_pretrained)
        self.heads = ClassHeads(self.encoder.width, class_counts, model_parameters["dropout"])


    def forward(self, pictures: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.heads(self.encoder(pictures))


class ExactStepRegressor(nn.Module):
    """Model 2: one picture, the exact step (scaled) and a stop logit."""

    kind = "regressor"

    def __init__(
            self,
            class_counts: Dict[str, int],
            model_parameters: Dict,
            is_pretrained: bool
    ):

        super().__init__()
        self.encoder = PictureEncoder(is_pretrained)
        self.huber_delta = float(model_parameters["huber_delta"])
        dropout = model_parameters["dropout"]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.width, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, class_counts["exact"] + 1))


    def forward(self, pictures: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.head(self.encoder(pictures))
        return {"exact": output[:, :-1], "stop": output[:, -1]}


class HistoryClassifier(nn.Module):
    """
    Model 3: this picture and the ones before it, each through the same encoder, plus the previous move.

    The previous move has one extra "unknown" value: the first picture of an episode has none, and training hides it
    at random ("previous_move_dropout") so the model learns from the pictures rather than copying the last move.
    """

    kind = "classifier"

    def __init__(
            self,
            class_counts: Dict[str, int],
            model_parameters: Dict,
            is_pretrained: bool
    ):

        super().__init__()
        self.encoder = PictureEncoder(is_pretrained)
        self.history_length = model_parameters["history_length"]
        self.previous_move_dropout = model_parameters["previous_move_dropout"]
        self.unknown_move = class_counts["move"]
        self.move_embedding = nn.Embedding(class_counts["move"] + 1, 32)
        self.heads = ClassHeads(self.encoder.width * self.history_length + 32, class_counts, model_parameters["dropout"])


    def forward(
            self,
            pictures: torch.Tensor,
            previous_move: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        batch_size = pictures.shape[0]
        features = self.encoder(pictures.flatten(0, 1)).reshape(batch_size, -1)
        if self.training and self.previous_move_dropout > 0:
            hidden = torch.rand(batch_size, device = pictures.device) < self.previous_move_dropout
            previous_move = torch.where(hidden, torch.full_like(previous_move, self.unknown_move), previous_move)
        return self.heads(torch.cat([features, self.move_embedding(previous_move)], dim = 1))


class ChunkTransformer(nn.Module):
    """
    Model 4: the next "chunk_length" steps from one picture, ACT-style.

    The feature map's cells become tokens with a learned position; one learned query per future step reads them
    through a transformer decoder and gives that step's exact numbers, stop logit and move name.
    """

    kind = "chunk"

    def __init__(
            self,
            class_counts: Dict[str, int],
            model_parameters: Dict,
            is_pretrained: bool
    ):

        super().__init__()
        width = model_parameters["model_width"]
        self.chunk_length = model_parameters["chunk_length"]
        self.trunk = resnet_trunk(is_pretrained)
        self.project = nn.Conv2d(512, width, kernel_size = 1)
        # Enough positions for any feature map up to 16 x 16 cells; the pictures here give 5 x 4.
        self.positions = nn.Parameter(torch.randn(256, width) * 0.02)
        self.queries = nn.Parameter(torch.randn(self.chunk_length, width) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model = width, nhead = model_parameters["heads"], dim_feedforward = width * 4,
                                           dropout = model_parameters["dropout"], batch_first = True)
        self.decoder = nn.TransformerDecoder(layer, num_layers = model_parameters["decoder_layers"])
        self.exact = nn.Linear(width, class_counts["exact"])
        self.stop = nn.Linear(width, 1)
        self.move = nn.Linear(width, class_counts["move"])


    def forward(self, pictures: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.project(self.trunk(pictures)).flatten(2).transpose(1, 2)
        tokens = tokens + self.positions[:tokens.shape[1]][None]
        queries = self.queries[None].expand(pictures.shape[0], -1, -1)
        decoded = self.decoder(queries, tokens)
        return {"exact": self.exact(decoded), "stop": self.stop(decoded).squeeze(-1), "move": self.move(decoded)}


def timestep_embedding(
        timesteps: torch.Tensor,
        width: int
) -> torch.Tensor:
    """The usual sine and cosine embedding of the diffusion step number."""

    half = width // 2
    frequencies = torch.exp(-math.log(10_000.0) * torch.arange(half, device = timesteps.device) / half)
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim = 1)


class ResidualBlock(nn.Module):
    """A two-layer residual block with layer normalisation, for the noise predictor."""

    def __init__(self, width: int):

        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))


    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(values)


class DiffusionPolicy(nn.Module):
    """
    Model 5: a denoising diffusion model over the next "chunk_length" steps, each the scaled exact step plus a stop
    value (+1 stop, -1 move), conditioned on the picture.

    Training adds noise at a random step of a cosine schedule and learns to predict that noise; sampling runs
    "sampling_steps" deterministic DDIM steps from pure noise.

    # Reference
    # Source: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023; Song et al.,
    #         "Denoising Diffusion Implicit Models", ICLR 2021; Nichol and Dhariwal, "Improved Denoising Diffusion
    #         Probabilistic Models", ICML 2021 (cosine schedule).
    # Link: https://arxiv.org/abs/2303.04137, https://arxiv.org/abs/2010.02502, https://arxiv.org/abs/2102.09672
    """

    kind = "diffusion"

    def __init__(
            self,
            class_counts: Dict[str, int],
            model_parameters: Dict,
            is_pretrained: bool
    ):

        super().__init__()
        self.chunk_length = model_parameters["chunk_length"]
        self.step_width = class_counts["exact"] + 1
        self.diffusion_steps = model_parameters["diffusion_steps"]
        self.sampling_steps = model_parameters["sampling_steps"]
        width = model_parameters["hidden_width"]
        self.encoder = PictureEncoder(is_pretrained)
        self.condition = nn.Sequential(nn.Linear(self.encoder.width + 128, width), nn.SiLU())
        self.input = nn.Linear(self.chunk_length * self.step_width, width)
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(model_parameters["residual_blocks"])])
        self.film = nn.ModuleList([nn.Linear(width, 2 * width) for _ in range(model_parameters["residual_blocks"])])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.chunk_length * self.step_width))

        steps = torch.arange(self.diffusion_steps + 1, dtype = torch.float64) / self.diffusion_steps
        alphas_cumulative = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumulative = (alphas_cumulative / alphas_cumulative[0]).clamp(1e-5, 1.0)
        self.register_buffer("alphas_cumulative", alphas_cumulative[1:].float())


    def predict_noise(
            self,
            noisy: torch.Tensor,
            timesteps: torch.Tensor,
            picture_features: torch.Tensor
    ) -> torch.Tensor:
        """The noise in ``noisy`` (batch, chunk, step width) at ``timesteps``, given the picture's features."""

        condition = self.condition(torch.cat([picture_features, timestep_embedding(timesteps, 128)], dim = 1))
        values = self.input(noisy.flatten(1))
        for block, film in zip(self.blocks, self.film):
            scale, shift = film(condition).chunk(2, dim = 1)
            values = block(values * (1 + scale) + shift)
        return self.output(values).reshape(noisy.shape)


    def training_loss(
            self,
            pictures: torch.Tensor,
            targets: torch.Tensor,
            weights: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared error of the predicted noise, each sample weighted by ``weights``."""

        features = self.encoder(pictures)
        timesteps = torch.randint(0, self.diffusion_steps, (targets.shape[0],), device = targets.device)
        noise = torch.randn_like(targets)
        alpha = self.alphas_cumulative[timesteps][:, None, None]
        noisy = alpha.sqrt() * targets + (1 - alpha).sqrt() * noise
        error = (self.predict_noise(noisy, timesteps, features) - noise).pow(2).mean(dim = (1, 2))
        return (error * weights).sum() / weights.sum()


    @torch.no_grad()
    def sample(
            self,
            pictures: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Draw one chunk per picture with deterministic DDIM steps; shape (batch, chunk, step width)."""

        features = self.encoder(pictures)
        shape = (pictures.shape[0], self.chunk_length, self.step_width)
        chunk = torch.randn(shape, device = pictures.device, generator = generator)
        timesteps = torch.linspace(self.diffusion_steps - 1, 0, self.sampling_steps, device = pictures.device).round().long()
        for index, timestep in enumerate(timesteps):
            alpha = self.alphas_cumulative[timestep]
            alpha_next = self.alphas_cumulative[timesteps[index + 1]] if index + 1 < len(timesteps) else torch.tensor(1.0, device = pictures.device)
            noise = self.predict_noise(chunk, timestep.repeat(shape[0]), features)
            clean = ((chunk - (1 - alpha).sqrt() * noise) / alpha.sqrt()).clamp(-6.0, 6.0)
            chunk = alpha_next.sqrt() * clean + (1 - alpha_next).sqrt() * noise
        return chunk


POLICY_CLASSES = {
    "factorised_classifier": FactorisedClassifier,
    "exact_step_regressor": ExactStepRegressor,
    "history_classifier": HistoryClassifier,
    "chunk_transformer": ChunkTransformer,
    "diffusion_policy": DiffusionPolicy
}


def build_policy_model(
        model_name: str,
        class_counts: Dict[str, int],
        model_parameters: Dict,
        is_pretrained: bool
) -> nn.Module:
    """One of the five policies by name ("factorised_classifier", ..., "diffusion_policy")."""

    if model_name not in POLICY_CLASSES:
        raise ValueError(f"unknown model {model_name}; choose from {sorted(POLICY_CLASSES)}")
    return POLICY_CLASSES[model_name](class_counts, model_parameters, is_pretrained)
