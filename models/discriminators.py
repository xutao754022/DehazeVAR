from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
from transformers import SwinConfig, SwinModel, ViTConfig, ViTModel

from utils.io_utils import resolve_project_path


DISCRIMINATOR_TYPES = (
    "dino_only",
    "dino_spectral",
)


class _FeaturePatchHead(nn.Module):
    def __init__(self, channels: int = 768, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, hidden, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(hidden, 1, kernel_size=1)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class _FeatureGlobalHead(nn.Module):
    def __init__(self, channels: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(channels, hidden)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Linear(hidden, 1)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _DINOFeatureDomainHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_early = _FeaturePatchHead()
        self.patch_late = _FeaturePatchHead()
        self.global_head = _FeatureGlobalHead(channels=768)

    def forward(
        self,
        early: torch.Tensor,
        late: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [
            self.patch_early(early),
            self.patch_late(late),
            self.global_head(global_feature),
        ]


def patchwise_fourier_power_spectrum(
    images: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    """Apply the SpecFormer per-patch 2D Fourier power transform."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected Bx3xHxW images, got {images.shape}")
    height, width = images.shape[-2:]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image size {(height, width)} must be divisible by patch_size={patch_size}"
        )
    batch, channels = images.shape[:2]
    patches_high = height // patch_size
    patches_wide = width // patch_size
    patches = images.float().unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        batch * patches_high * patches_wide,
        channels,
        patch_size,
        patch_size,
    )
    with torch.autocast(device_type=images.device.type, enabled=False):
        frequency = torch.fft.fftshift(
            torch.fft.fft2(patches.float(), dim=(-2, -1)), dim=(-2, -1)
        )
        power = frequency.real.square() + frequency.imag.square()
        minimum = power.amin(dim=(1, 2, 3), keepdim=True)
        maximum = power.amax(dim=(1, 2, 3), keepdim=True)
        power = (power - minimum) / (maximum - minimum + 1e-5)
    return power.reshape(
        batch,
        patches_high,
        patches_wide,
        channels,
        patch_size,
        patch_size,
    ).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)


class _SpectralTransformerBackbone(nn.Module):
    """SpecFormer: patch Fourier power followed by a two-stage Swin encoder."""

    def __init__(
        self,
        *,
        image_size: int,
        fourier_patch_size: int,
        embed_dimension: int,
        depths: tuple[int, int],
        heads: tuple[int, int],
        window_size: int,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.fourier_patch_size = int(fourier_patch_size)
        self.embed_dimension = int(embed_dimension)
        self.depths = tuple(int(value) for value in depths)
        self.heads = tuple(int(value) for value in heads)
        self.window_size = int(window_size)
        config = SwinConfig(
            image_size=self.image_size,
            patch_size=self.fourier_patch_size,
            num_channels=3,
            embed_dim=self.embed_dimension,
            depths=list(self.depths),
            num_heads=list(self.heads),
            window_size=self.window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.1,
        )
        self.transformer = SwinModel(config, add_pooling_layer=True)
        self.output_dimension = self.embed_dimension * 2 ** (len(self.depths) - 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spectrum = patchwise_fourier_power_spectrum(
            images, patch_size=self.fourier_patch_size
        )
        output = self.transformer(pixel_values=spectrum, return_dict=True)
        pooled = output.pooler_output
        if pooled is None:
            pooled = output.last_hidden_state.mean(dim=1)
        return pooled.float()

    def signature(self) -> dict[str, Any]:
        return {
            "type": "iccv2023_specformer",
            "paper": "On the Effectiveness of Spectral Discriminators for Perceptual Quality Improvement",
            "image_size": self.image_size,
            "fourier": {
                "patch_size": self.fourier_patch_size,
                "transform": "two_dimensional_fftshift_power",
                "reduction": "none_rgb",
                "normalization": "fixed_per_patch_minmax",
            },
            "aggregator": {
                "type": "swin_transformer",
                "embed_dimension": self.embed_dimension,
                "depths": list(self.depths),
                "heads": list(self.heads),
                "window_size": self.window_size,
            },
        }


class _SpectralDomainHeads(nn.Module):
    def __init__(self, channels: int, domains: tuple[str, ...]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {domain: spectral_norm(nn.Linear(channels, 1)) for domain in domains}
        )

    def forward(self, features: torch.Tensor, domain: str) -> torch.Tensor:
        return self.heads[domain](features)


def _binary_cross_entropy_losses(
    logits: list[torch.Tensor],
    target_value: float,
) -> list[torch.Tensor]:
    losses = []
    for branch_logits in logits:
        target = torch.full_like(branch_logits, target_value)
        losses.append(F.binary_cross_entropy_with_logits(branch_logits, target))
    if not losses:
        raise RuntimeError("A discriminator must expose at least one logit branch")
    return losses


def _reference_scaled_mean(
    losses: list[torch.Tensor],
    reference_branch_count: int,
) -> torch.Tensor:
    if reference_branch_count <= 0:
        raise ValueError("reference_branch_count must be positive")
    return torch.stack(losses).mean() * float(reference_branch_count)


class UnpairedDehazingDINOFeatureDiscriminator(nn.Module):
    """Frozen DINO ViT-B/8 with spatial and global domain heads."""

    domains = ("hazy", "clean")

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        real_label: float,
        feature_layers: tuple[int, int],
        reference_branch_count: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        config = ViTConfig.from_pretrained(str(checkpoint_path), local_files_only=True)
        config.layer_norm_eps = 1e-6
        self.backbone = ViTModel.from_pretrained(
            str(checkpoint_path),
            config=config,
            local_files_only=True,
            add_pooling_layer=False,
        )
        self.backbone.eval().requires_grad_(False)
        self.feature_layers = tuple(int(index) for index in feature_layers)
        if len(self.feature_layers) != 2:
            raise ValueError("DINO discriminator requires exactly two feature layers")
        if not all(
            0 < index <= int(config.num_hidden_layers)
            for index in self.feature_layers
        ):
            raise ValueError(
                f"DINO feature layers {self.feature_layers} exceed "
                f"the {config.num_hidden_layers}-layer backbone"
            )
        self.real_label = float(real_label)
        self.reference_branch_count = int(reference_branch_count)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.heads = nn.ModuleDict(
            {domain: _DINOFeatureDomainHead() for domain in self.domains}
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def set_heads_trainable(self, enabled: bool) -> None:
        for parameter in self.heads.parameters():
            parameter.requires_grad_(enabled)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.heads.parameters())

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(
            images.add(1.0).mul(0.5),
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return (images - self.image_mean) / self.image_std

    def _selected_hidden_states(
        self,
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            pixel_values=pixels,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("DINO did not return hidden states")
        early = hidden_states[self.feature_layers[0]]
        late = hidden_states[self.feature_layers[1]]
        return early, late, outputs.last_hidden_state

    @staticmethod
    def _patches_to_map(tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, 1:, :]
        side = round(patches.shape[1] ** 0.5)
        if side * side != patches.shape[1]:
            raise RuntimeError(f"DINO patch count is not square: {patches.shape}")
        return patches.transpose(1, 2).reshape(
            patches.shape[0], patches.shape[2], side, side
        )

    def _features(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixels = self._preprocess(images)
        if self.activation_checkpointing and pixels.requires_grad:
            early_tokens, late_tokens, final_tokens = checkpoint(
                self._selected_hidden_states,
                pixels,
                use_reentrant=False,
            )
        else:
            early_tokens, late_tokens, final_tokens = self._selected_hidden_states(
                pixels
            )
        return (
            self._patches_to_map(early_tokens).float(),
            self._patches_to_map(late_tokens).float(),
            final_tokens[:, 0, :].float(),
        )

    def logits(self, images: torch.Tensor, domain: str) -> list[torch.Tensor]:
        if domain not in self.heads:
            raise ValueError(f"Unknown domain {domain!r}; expected {self.domains}")
        return list(self.heads[domain](*self._features(images)))

    def loss(
        self,
        images: torch.Tensor,
        domain: str,
        for_real: bool,
        for_generator: bool = False,
    ) -> torch.Tensor:
        target_value = self.real_label if (for_real or for_generator) else 0.0
        return _reference_scaled_mean(
            _binary_cross_entropy_losses(self.logits(images, domain), target_value),
            self.reference_branch_count,
        )

    def discriminator_signature(self) -> dict[str, Any]:
        return {
            "type": "dino_only",
            "feature_layers": list(self.feature_layers),
            "reference_branch_count": self.reference_branch_count,
            "activation_checkpointing": self.activation_checkpointing,
        }


class UnpairedDehazingDINOWithSpectralDiscriminator(
    UnpairedDehazingDINOFeatureDiscriminator
):
    """Frozen DINO spatial heads combined with the 2D FFT SpecFormer discriminator."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        real_label: float,
        feature_layers: tuple[int, int],
        reference_branch_count: int,
        activation_checkpointing: bool,
        spectral_image_size: int,
        spectral_patch_size: int,
        spectral_embed_dimension: int,
        spectral_depths: tuple[int, int],
        spectral_heads: tuple[int, int],
        spectral_window_size: int,
        spectral_family_weight: float,
    ) -> None:
        super().__init__(
            checkpoint_path,
            real_label=real_label,
            feature_layers=feature_layers,
            reference_branch_count=reference_branch_count,
            activation_checkpointing=activation_checkpointing,
        )
        self.spectral_family_weight = float(spectral_family_weight)
        if not 0.0 <= self.spectral_family_weight <= 1.0:
            raise ValueError("spectral_family_weight must be between zero and one")
        self.heads["spectral_backbone"] = _SpectralTransformerBackbone(
            image_size=spectral_image_size,
            fourier_patch_size=spectral_patch_size,
            embed_dimension=spectral_embed_dimension,
            depths=spectral_depths,
            heads=spectral_heads,
            window_size=spectral_window_size,
        )
        self.heads["spectral_domain_heads"] = _SpectralDomainHeads(
            self.heads["spectral_backbone"].output_dimension,
            self.domains,
        )

    def family_logits(
        self,
        images: torch.Tensor,
        domain: str,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if domain not in self.domains:
            raise ValueError(f"Unknown domain {domain!r}; expected {self.domains}")
        dino_logits = list(self.heads[domain](*self._features(images)))
        spectral_features = self.heads["spectral_backbone"](images)
        spectral_logits = self.heads["spectral_domain_heads"](
            spectral_features, domain
        )
        return dino_logits, spectral_logits

    def logits(self, images: torch.Tensor, domain: str) -> list[torch.Tensor]:
        dino_logits, spectral_logits = self.family_logits(images, domain)
        return [*dino_logits, spectral_logits]

    def loss(
        self,
        images: torch.Tensor,
        domain: str,
        for_real: bool,
        for_generator: bool = False,
    ) -> torch.Tensor:
        target_value = self.real_label if (for_real or for_generator) else 0.0
        dino_logits, spectral_logits = self.family_logits(images, domain)
        dino_loss = torch.stack(
            _binary_cross_entropy_losses(dino_logits, target_value)
        ).mean()
        spectral_loss = _binary_cross_entropy_losses(
            [spectral_logits], target_value
        )[0]
        family_mean = (
            (1.0 - self.spectral_family_weight) * dino_loss
            + self.spectral_family_weight * spectral_loss
        )
        return family_mean * float(self.reference_branch_count)

    def discriminator_signature(self) -> dict[str, Any]:
        return {
            "type": "dino_spectral",
            "dino": {
                "feature_layers": list(self.feature_layers),
                "reference_branch_count": self.reference_branch_count,
                "activation_checkpointing": self.activation_checkpointing,
            },
            "spectral": self.heads["spectral_backbone"].signature(),
            "spectral_family_weight": self.spectral_family_weight,
            "family_combination": "weighted_mean_then_reference_three_branch_scale",
        }


def build_unpaired_dehazing_discriminator(
    model_config: dict[str, Any],
    loss_config: dict[str, Any],
) -> nn.Module:
    """Build the configured DehazeVAR spatial or spatial-spectral discriminator."""
    discriminator_type = str(
        model_config.get("discriminator_type", "dino_only")
    )
    if discriminator_type not in DISCRIMINATOR_TYPES:
        raise ValueError(
            f"Unknown discriminator_type {discriminator_type!r}; "
            f"expected one of {DISCRIMINATOR_TYPES}"
        )
    real_label = float(loss_config["real_label_smoothing"])
    reference_branch_count = int(
        model_config.get("discriminator_reference_branch_count", 3)
    )
    common_dino_arguments = {
        "checkpoint_path": resolve_project_path(model_config["dino_checkpoint"]),
        "real_label": real_label,
        "feature_layers": tuple(
            int(index)
            for index in model_config.get("discriminator_feature_layers", [4, 8])
        ),
        "reference_branch_count": reference_branch_count,
        "activation_checkpointing": bool(
            model_config.get("dino_activation_checkpointing", True)
        ),
    }
    if discriminator_type == "dino_only":
        return UnpairedDehazingDINOFeatureDiscriminator(**common_dino_arguments)
    return UnpairedDehazingDINOWithSpectralDiscriminator(
        **common_dino_arguments,
        spectral_image_size=int(model_config.get("spectral_image_size", 256)),
        spectral_patch_size=int(model_config.get("spectral_patch_size", 32)),
        spectral_embed_dimension=int(
            model_config.get("spectral_embed_dimension", 96)
        ),
        spectral_depths=tuple(
            int(value) for value in model_config.get("spectral_depths", [8, 2])
        ),
        spectral_heads=tuple(
            int(value) for value in model_config.get("spectral_heads", [3, 6])
        ),
        spectral_window_size=int(
            model_config.get("spectral_window_size", 4)
        ),
        spectral_family_weight=float(
            model_config.get("spectral_family_weight", 0.5)
        ),
    )


DINOUnpairedDehazingDiscriminator = UnpairedDehazingDINOFeatureDiscriminator
DINOSpectralUnpairedDehazingDiscriminator = (
    UnpairedDehazingDINOWithSpectralDiscriminator
)
build_dehazevar_discriminator = build_unpaired_dehazing_discriminator

__all__ = [
    "DINOSpectralUnpairedDehazingDiscriminator",
    "DINOUnpairedDehazingDiscriminator",
    "DISCRIMINATOR_TYPES",
    "UnpairedDehazingDINOFeatureDiscriminator",
    "UnpairedDehazingDINOWithSpectralDiscriminator",
    "build_dehazevar_discriminator",
    "build_unpaired_dehazing_discriminator",
    "patchwise_fourier_power_spectrum",
]
