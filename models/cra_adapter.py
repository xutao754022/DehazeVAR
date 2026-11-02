from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from utils.haze_operators import (
    fourier_amplitude_map,
    structural_feature_maps,
)


TARGET_DOMAIN_NAMES = ("hazy", "clean")


def _build_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "silu":
        return nn.SiLU(inplace=False)
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported decoder-adapter activation: {name!r}")


class ZeroInitializedResidualExpert(nn.Module):
    """A zero-opening bottleneck residual branch for one target domain."""

    def __init__(
        self,
        channels: int,
        reduction_channels: int,
        activation: str,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0 < reduction_channels <= channels:
            raise ValueError(
                "reduction_channels must be in [1, channels], got "
                f"{reduction_channels} for {channels} channels"
            )
        self.channels = int(channels)
        self.reduction_channels = int(reduction_channels)
        self.activation_name = activation.strip().lower()
        self.reduction = nn.Conv2d(
            self.channels,
            self.reduction_channels,
            kernel_size=1,
        )
        self.depthwise = nn.Conv2d(
            self.reduction_channels,
            self.reduction_channels,
            kernel_size=3,
            padding=1,
            groups=self.reduction_channels,
        )
        self.activation = _build_activation(self.activation_name)
        self.restoration = nn.Conv2d(
            self.reduction_channels,
            self.channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.restoration.weight)
        nn.init.zeros_(self.restoration.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                "Expected decoder-adapter input shaped BxCxHxW with "
                f"C={self.channels}, got {tuple(features.shape)}"
            )
        hidden = self.reduction(features)
        hidden = self.depthwise(hidden)
        hidden = self.activation(hidden)
        return self.restoration(hidden)


class TargetDomainResidualAdapter(nn.Module):
    """Route a residual branch through independent hazy and clean experts."""

    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        activation: str,
        domain_names: Sequence[str] = TARGET_DOMAIN_NAMES,
    ) -> None:
        super().__init__()
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive")
        names = tuple(str(name) for name in domain_names)
        if not names or len(set(names)) != len(names):
            raise ValueError(f"domain_names must be unique and non-empty: {names}")
        self.channels = int(channels)
        self.reduction_ratio = int(reduction_ratio)
        self.reduction_channels = max(1, self.channels // self.reduction_ratio)
        self.activation_name = activation.strip().lower()
        self.domain_names = names
        self.experts = nn.ModuleDict(
            {
                name: ZeroInitializedResidualExpert(
                    channels=self.channels,
                    reduction_channels=self.reduction_channels,
                    activation=self.activation_name,
                )
                for name in self.domain_names
            }
        )

    def forward(
        self,
        features: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        if target_labels.ndim != 1 or target_labels.shape[0] != features.shape[0]:
            raise ValueError(
                "Target-domain labels must be a length-B vector, got "
                f"labels={tuple(target_labels.shape)}, features={tuple(features.shape)}"
            )
        labels = target_labels.to(device=features.device, dtype=torch.long)
        if bool(((labels < 0) | (labels >= len(self.domain_names))).any()):
            raise ValueError(
                f"Target-domain labels must be in [0, {len(self.domain_names) - 1}]"
            )
        expert_outputs = torch.stack(
            [self.experts[name](features) for name in self.domain_names],
            dim=1,
        )
        selector = nn.functional.one_hot(
            labels,
            num_classes=len(self.domain_names),
        ).to(dtype=expert_outputs.dtype)
        selector = selector.view(features.shape[0], len(self.domain_names), 1, 1, 1)
        return (expert_outputs * selector).sum(dim=1)

    def signature(self) -> dict[str, object]:
        return {
            "domain_names": list(self.domain_names),
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio,
            "reduction_channels": self.reduction_channels,
            "activation": self.activation_name,
            "architecture": "1x1-reduce_depthwise-3x3_activation_1x1-zero-restore",
            "zero_initialized_restoration": True,
        }


@dataclass(frozen=True)
class HazeFactorizedRepresentation:
    global_embedding: torch.Tensor
    haze_density_map: torch.Tensor
    atmospheric_light: torch.Tensor
    amplitude_embedding: torch.Tensor
    structure_embedding: torch.Tensor
    severity_logit: torch.Tensor


def _convolution_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, output_channels),
        nn.SiLU(inplace=True),
    )


class HazeFactorizedSideEncoder(nn.Module):
    """Encode haze factors without changing the frozen VAR tokenizer or codebook."""

    def __init__(
        self,
        *,
        base_channels: int,
        global_dimension: int,
        amplitude_dimension: int,
        structure_dimension: int,
        latent_size: int,
    ) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("hfse_base_channels must be divisible by 8")
        channels = (
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4,
        )
        self.latent_size = int(latent_size)
        self.image_encoder = nn.Sequential(
            _convolution_block(3, channels[0]),
            _convolution_block(channels[0], channels[1]),
            _convolution_block(channels[1], channels[2]),
            _convolution_block(channels[2], channels[3]),
        )
        self.haze_map_head = nn.Sequential(
            nn.Conv2d(channels[3], channels[1], kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[1], 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_head = nn.Sequential(
            nn.Linear(channels[3], global_dimension),
            nn.SiLU(inplace=True),
        )
        self.atmospheric_light_head = nn.Sequential(
            nn.Linear(channels[3], 3),
            nn.Sigmoid(),
        )
        self.severity_head = nn.Linear(global_dimension, 1)
        self.amplitude_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, amplitude_dimension),
            nn.SiLU(inplace=True),
        )
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(6, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels * 2, structure_dimension, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> HazeFactorizedRepresentation:
        image_features = self.image_encoder(image)
        if image_features.shape[-2:] != (self.latent_size, self.latent_size):
            raise ValueError(
                "HFSE image encoder must align with the VAR latent grid: "
                f"actual={image_features.shape[-2:]}, expected={self.latent_size}"
            )
        pooled = F.adaptive_avg_pool2d(image_features, 1).flatten(1)
        global_embedding = self.global_head(pooled)
        with torch.no_grad():
            amplitude = fourier_amplitude_map(image.detach(), self.latent_size)
            structure = F.interpolate(
                structural_feature_maps(image.detach()),
                size=(self.latent_size, self.latent_size),
                mode="bilinear",
                align_corners=False,
            )
        return HazeFactorizedRepresentation(
            global_embedding=global_embedding,
            haze_density_map=self.haze_map_head(image_features),
            atmospheric_light=self.atmospheric_light_head(pooled),
            amplitude_embedding=self.amplitude_encoder(amplitude),
            structure_embedding=self.structure_encoder(structure),
            severity_logit=self.severity_head(global_embedding).squeeze(-1),
        )


class AmplitudeResidualTokenEncoder(nn.Module):
    """Inject RGB amplitude residual context into the first three VAR scales."""

    def __init__(
        self,
        *,
        channels: int,
        token_dimension: int,
        coarse_patch_nums: tuple[int, ...],
    ) -> None:
        super().__init__()
        if channels % 8 != 0:
            raise ValueError("frequency_encoder_channels must be divisible by eight")
        self.coarse_patch_nums = coarse_patch_nums
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )
        self.scale_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, token_dimension))
            for _ in coarse_patch_nums
        )
        for projection in self.scale_projections:
            nn.init.zeros_(projection[-1].weight)
            nn.init.zeros_(projection[-1].bias)

    def forward(self, amplitude: torch.Tensor, *, total_tokens: int) -> torch.Tensor:
        features = self.encoder(amplitude)
        conditions = []
        for patch_num, projection in zip(
            self.coarse_patch_nums, self.scale_projections, strict=True
        ):
            scale_features = (
                F.adaptive_avg_pool2d(features, (patch_num, patch_num))
                .flatten(2)
                .transpose(1, 2)
            )
            conditions.append(projection(scale_features))
        coarse = torch.cat(conditions, dim=1)
        if coarse.shape[1] > total_tokens:
            raise ValueError("Amplitude condition has more tokens than CycleVAR")
        return F.pad(coarse, (0, 0, 0, total_tokens - coarse.shape[1]))


class PhaseCorrectionTokenEncoder(nn.Module):
    """Inject a zero-initialized phase correction into middle VAR scales."""

    def __init__(
        self,
        *,
        channels: int,
        token_dimension: int,
        middle_patch_nums: tuple[int, ...],
        prefix_tokens: int,
    ) -> None:
        super().__init__()
        if channels % 8 != 0:
            raise ValueError(
                "frequency_phase_encoder_channels must be divisible by eight"
            )
        self.middle_patch_nums = middle_patch_nums
        self.prefix_tokens = int(prefix_tokens)
        self.encoder = nn.Sequential(
            nn.Conv2d(6, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )
        self.scale_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, token_dimension))
            for _ in middle_patch_nums
        )
        for projection in self.scale_projections:
            nn.init.zeros_(projection[-1].weight)
            nn.init.zeros_(projection[-1].bias)

    def forward(self, phase: torch.Tensor, *, total_tokens: int) -> torch.Tensor:
        features = self.encoder(phase)
        middle = []
        for patch_num, projection in zip(
            self.middle_patch_nums, self.scale_projections, strict=True
        ):
            scale_features = (
                F.adaptive_avg_pool2d(features, (patch_num, patch_num))
                .flatten(2)
                .transpose(1, 2)
            )
            middle.append(projection(scale_features))
        middle_tokens = torch.cat(middle, dim=1)
        suffix = total_tokens - self.prefix_tokens - middle_tokens.shape[1]
        if suffix < 0:
            raise ValueError("Phase condition has more tokens than CycleVAR")
        return F.pad(middle_tokens, (0, 0, self.prefix_tokens, suffix))


def _downsample_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, output_channels),
        nn.SiLU(inplace=True),
    )


def _upsample_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.GroupNorm(8, output_channels),
        nn.SiLU(inplace=True),
    )


class HazeAwareSharedIDDetailAdapter(nn.Module):
    """Predict a bounded detail residual while retaining the original token IDs."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        code_dimension: int,
        domain_count: int,
        domain_dimension: int,
        base_channels: int,
        detail_initialization_standard_deviation: float,
        gamma_maximum: float,
    ) -> None:
        super().__init__()
        if base_channels % 8 != 0:
            raise ValueError("hsda_base_channels must be divisible by 8")
        self.gamma_maximum = float(gamma_maximum)
        self.base_channels = int(base_channels)
        self.domain_dimension = int(domain_dimension)
        self.detail_codebook = nn.Embedding(vocabulary_size, code_dimension)
        nn.init.normal_(
            self.detail_codebook.weight,
            mean=0.0,
            std=float(detail_initialization_standard_deviation),
        )
        self.domain_embedding = nn.Embedding(domain_count, domain_dimension)
        self.structure_encoder = nn.Sequential(
            _downsample_block(6, base_channels),
            _downsample_block(base_channels, base_channels * 2),
            _downsample_block(base_channels * 2, base_channels * 2),
            _downsample_block(base_channels * 2, base_channels * 2),
        )
        decoder_input_channels = (
            code_dimension * 2 + base_channels * 2 + 1 + domain_dimension
        )
        self.residual_decoder = nn.Sequential(
            nn.Conv2d(
                decoder_input_channels,
                base_channels * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(inplace=True),
            _upsample_block(base_channels * 4, base_channels * 4),
            _upsample_block(base_channels * 4, base_channels * 2),
            _upsample_block(base_channels * 2, base_channels),
            _upsample_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.gamma_parameter = nn.Parameter(torch.zeros(()))

    @property
    def effective_gamma(self) -> torch.Tensor:
        return self.gamma_maximum * torch.tanh(self.gamma_parameter)

    def forward(
        self,
        *,
        base_latent: torch.Tensor,
        detail_latent: torch.Tensor,
        target_labels: torch.Tensor,
        source_image: torch.Tensor,
        visibility_map: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            structure_maps = structural_feature_maps(source_image.detach())
        structure = self.structure_encoder(structure_maps.to(base_latent.dtype))
        visibility = F.interpolate(
            visibility_map.float(),
            size=base_latent.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(base_latent.dtype)
        structure = structure * visibility
        domain = self.domain_embedding(target_labels).to(base_latent.dtype)
        domain = domain[:, :, None, None].expand(
            -1, -1, base_latent.shape[-2], base_latent.shape[-1]
        )
        features = torch.cat(
            (base_latent, detail_latent, structure, visibility, domain), dim=1
        )
        return self.residual_decoder(features)


CrossScaleResidualAggregation = HazeAwareSharedIDDetailAdapter
DomainAdaptiveDecoderExpert = TargetDomainResidualAdapter
ConditionalResidualAdapter = TargetDomainResidualAdapter

__all__ = [
    "AmplitudeResidualTokenEncoder",
    "ConditionalResidualAdapter",
    "CrossScaleResidualAggregation",
    "DomainAdaptiveDecoderExpert",
    "HazeAwareSharedIDDetailAdapter",
    "HazeFactorizedRepresentation",
    "HazeFactorizedSideEncoder",
    "PhaseCorrectionTokenEncoder",
    "TARGET_DOMAIN_NAMES",
    "TargetDomainResidualAdapter",
    "ZeroInitializedResidualExpert",
]
