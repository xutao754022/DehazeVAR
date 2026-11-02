from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from models.discriminators import (
    UnpairedDehazingDINOFeatureDiscriminator,
    _binary_cross_entropy_losses,
)
from utils.losses import (
    canonical_fingerprint,
    haar_high_band_loss,
    sha256_file,
    ssim_loss,
)


DISEN_VAE_CHECKPOINT_FORMAT = "cyclevar_v4_disen_vae_v2"
DISEN_VAE_CHECKPOINT_VERSION = 2
DISEN_VAE_PRETRAINING_STEPS = 10_000


class ZeroInitializedRGBResidualAdapter(nn.Module):
    """A small image-space residual branch that is exactly closed at step zero."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("disen_vae_adapter_channels must be positive")
        self.channels = int(channels)
        self.input = nn.Conv2d(3, self.channels, kernel_size=3, padding=1)
        self.depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            groups=self.channels,
        )
        self.output = nn.Conv2d(self.channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.input(image), inplace=False)
        hidden = F.silu(self.depthwise(hidden), inplace=False)
        return self.output(hidden)

    def signature(self) -> dict[str, object]:
        return {
            "type": "zero_initialized_rgb_residual_adapter",
            "version": 1,
            "channels": self.channels,
            "layers": ["conv3x3", "depthwise3x3", "conv3x3_zero_init"],
            "activation": "silu",
        }


class DisentangledVAEDecoderBranches(nn.Module):
    """Two trainable decoder copies followed by domain-specific RGB adapters."""

    domains = ("hazy", "clean")

    def __init__(self, base_decoder: nn.Module, adapter_channels: int) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict(
            {domain: copy.deepcopy(base_decoder) for domain in self.domains}
        )
        self.adapters = nn.ModuleDict(
            {
                domain: ZeroInitializedRGBResidualAdapter(adapter_channels)
                for domain in self.domains
            }
        )

    def decode(
        self, decoder_input: torch.Tensor, domain: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if domain not in self.domains:
            raise ValueError(f"Unknown decoder branch {domain!r}")
        base = self.decoders[domain](decoder_input).clamp(-1.0, 1.0)
        residual = self.adapters[domain](base)
        return (base + residual).clamp(-1.0, 1.0), residual

    def decode_labels(
        self,
        decoder_input: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels.ndim != 1 or labels.shape[0] != decoder_input.shape[0]:
            raise ValueError(
                f"Decoder labels must be B, got labels={labels.shape}, input={decoder_input.shape}"
            )
        id_to_domain = {0: "hazy", 1: "clean"}
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError(
                f"Unexpected decoder labels: {labels.detach().cpu().tolist()}"
            )
        unique = torch.unique(labels)
        if unique.numel() == 1:
            return self.decode(decoder_input, id_to_domain[int(unique.item())])
        hazy_output, hazy_residual = self.decode(decoder_input, "hazy")
        clean_output, clean_residual = self.decode(decoder_input, "clean")
        clean_mask = labels[:, None, None, None] == 1
        return (
            torch.where(clean_mask, clean_output, hazy_output),
            torch.where(clean_mask, clean_residual, hazy_residual),
        )

    def branch_parameters(self) -> list[nn.Parameter]:
        return list(self.decoders.parameters()) + list(self.adapters.parameters())

    def branch_state_dict(self) -> dict[str, object]:
        return {
            "decoders": self.decoders.state_dict(),
            "adapters": self.adapters.state_dict(),
        }

    def load_branch_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"decoders", "adapters"}:
            raise ValueError(
                "Disentangled VAE state must contain decoder and adapter weights"
            )
        if not isinstance(state["decoders"], Mapping) or not isinstance(
            state["adapters"], Mapping
        ):
            raise TypeError(
                "Disentangled VAE decoder and adapter states must be mappings"
            )
        self.decoders.load_state_dict(state["decoders"], strict=True)
        self.adapters.load_state_dict(state["adapters"], strict=True)

    def signature(self) -> dict[str, object]:
        return {
            "type": "two_trainable_cyclevar_decoder_branches",
            "version": 2,
            "domains": list(self.domains),
            "decoder_initialization": "deepcopy_official_cyclevar_vae_decoder",
            "decoder_trainable_during_pretraining": True,
            "frozen_during_cyclevar_joint_training": True,
            "adapter": self.adapters["clean"].signature(),
        }


def _gaussian_kernel(
    sigma: torch.Tensor,
    *,
    kernel_size: int,
    channels: int,
) -> torch.Tensor:
    radius = kernel_size // 2
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=sigma.device,
        dtype=torch.float32,
    )
    one_dimensional = torch.exp(-(coordinates.square()) / (2.0 * sigma.square()))
    one_dimensional = one_dimensional / one_dimensional.sum()
    kernel = one_dimensional[:, None] * one_dimensional[None, :]
    return kernel.expand(channels, 1, kernel_size, kernel_size)


class SyntheticHazePipeline(nn.Module):
    """Seeded pseudo-paired haze synthesis with physical and camera degradations."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.homogeneous_probability = float(config["homogeneous_probability"])
        self.transmission_range = tuple(float(v) for v in config["transmission_range"])
        self.atmospheric_light_range = tuple(
            float(v) for v in config["atmospheric_light_range"]
        )
        self.temperature_shift = float(config["temperature_shift"])
        self.noise_standard_deviation = tuple(
            float(v) for v in config["noise_standard_deviation"]
        )
        self.blur_sigma = tuple(float(v) for v in config["blur_sigma"])
        self.compression_levels = tuple(int(v) for v in config["compression_levels"])
        self.smooth_grid_size = int(config["smooth_grid_size"])
        if not 0.0 <= self.homogeneous_probability <= 1.0:
            raise ValueError("homogeneous_probability must be in [0, 1]")
        if self.smooth_grid_size < 2:
            raise ValueError("smooth_grid_size must be at least two")

    @staticmethod
    def _uniform(
        batch: int,
        low: float,
        high: float,
        *,
        device: torch.device,
        shape: tuple[int, ...] = (),
    ) -> torch.Tensor:
        return low + (high - low) * torch.rand(batch, *shape, device=device)

    def _transmission(self, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = clean.shape
        scalar = self._uniform(
            batch,
            *self.transmission_range,
            device=clean.device,
            shape=(1, 1, 1),
        )
        low_resolution = torch.rand(
            batch,
            1,
            self.smooth_grid_size,
            self.smooth_grid_size,
            device=clean.device,
        )
        smooth = F.interpolate(
            low_resolution,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        varying = self.transmission_range[0] + (
            self.transmission_range[1] - self.transmission_range[0]
        ) * smooth
        homogeneous = (
            torch.rand(batch, 1, 1, 1, device=clean.device)
            < self.homogeneous_probability
        )
        return torch.where(homogeneous, scalar, varying), homogeneous.float()

    @staticmethod
    def _compression(image: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        red, green, blue = image.unbind(dim=1)
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        cb = (blue - luminance) * 0.564 + 0.5
        cr = (red - luminance) * 0.713 + 0.5
        chroma = torch.stack((cb, cr), dim=1)
        chroma = F.avg_pool2d(chroma, kernel_size=2, stride=2)
        chroma = F.interpolate(
            chroma, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        quantized_y = torch.round(luminance[:, None] * levels) / levels
        quantized_c = torch.round(chroma * levels) / levels
        cb, cr = quantized_c.unbind(dim=1)
        red = quantized_y[:, 0] + 1.403 * (cr - 0.5)
        blue = quantized_y[:, 0] + 1.773 * (cb - 0.5)
        green = quantized_y[:, 0] - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
        return torch.stack((red, green, blue), dim=1).clamp(0.0, 1.0)

    @torch.no_grad()
    def forward(self, clean: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean_01 = clean.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        batch = clean.shape[0]
        transmission, homogeneous = self._transmission(clean_01)
        atmospheric_light = self._uniform(
            batch,
            *self.atmospheric_light_range,
            device=clean.device,
            shape=(3, 1, 1),
        )
        hazy = clean_01 * transmission + atmospheric_light * (1.0 - transmission)

        temperature = self._uniform(
            batch,
            -self.temperature_shift,
            self.temperature_shift,
            device=clean.device,
            shape=(1, 1, 1),
        )
        gains = torch.cat(
            (1.0 + temperature, torch.ones_like(temperature), 1.0 - temperature),
            dim=1,
        )
        hazy = (hazy * gains).clamp(0.0, 1.0)

        sigma = self._uniform(
            1, *self.blur_sigma, device=clean.device
        ).clamp_min(0.05)
        kernel = _gaussian_kernel(sigma, kernel_size=7, channels=3)
        hazy = F.conv2d(hazy, kernel, padding=3, groups=3)

        noise_sigma = self._uniform(
            batch,
            *self.noise_standard_deviation,
            device=clean.device,
            shape=(1, 1, 1),
        )
        hazy = (hazy + torch.randn_like(hazy) * noise_sigma).clamp(0.0, 1.0)
        minimum_levels, maximum_levels = self.compression_levels
        levels = torch.randint(
            minimum_levels,
            maximum_levels + 1,
            (batch, 1, 1, 1),
            device=clean.device,
        ).float()
        hazy = self._compression(hazy, levels)
        return hazy.mul(2.0).sub(1.0), {
            "homogeneous_fraction": homogeneous.mean(),
            "transmission_mean": transmission.mean(),
            "atmospheric_light_mean": atmospheric_light.mean(),
            "temperature_shift_mean": temperature.abs().mean(),
            "noise_sigma_mean": noise_sigma.mean(),
            "compression_levels_mean": levels.mean(),
            "blur_sigma": sigma.squeeze(),
        }

    def signature(self) -> dict[str, object]:
        return {
            "type": "pseudo_paired_clean_to_haze",
            "version": 1,
            "physical": {
                "models": ["spatially_varying_asm", "homogeneous_asm"],
                "homogeneous_probability": self.homogeneous_probability,
                "transmission_range": list(self.transmission_range),
                "atmospheric_light_range": list(self.atmospheric_light_range),
                "smooth_grid_size": self.smooth_grid_size,
            },
            "camera": {
                "temperature_shift": self.temperature_shift,
                "noise_standard_deviation": list(self.noise_standard_deviation),
                "blur_sigma": list(self.blur_sigma),
                "compression": "chroma_420_plus_uniform_quantization",
                "compression_levels": list(self.compression_levels),
            },
        }


class DINOAndDomainPretrainingDiscriminator(nn.Module):
    """Frozen DINO features with real/fake domain heads and a domain classifier."""

    domains = ("hazy", "clean")

    def __init__(self, config: Mapping[str, Any], *, real_label: float) -> None:
        super().__init__()
        self.dino = UnpairedDehazingDINOFeatureDiscriminator(
            config["dino_checkpoint"],
            real_label=real_label,
            feature_layers=tuple(int(v) for v in config["discriminator_feature_layers"]),
            reference_branch_count=int(config["discriminator_reference_branch_count"]),
            activation_checkpointing=bool(config["dino_activation_checkpointing"]),
        )
        self.domain_classifier = spectral_norm(nn.Linear(768, len(self.domains)))
        self.real_label = float(real_label)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.backbone.eval()
        return self

    def set_trainable(self, enabled: bool) -> None:
        self.dino.set_heads_trainable(enabled)
        for parameter in self.domain_classifier.parameters():
            parameter.requires_grad_(enabled)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [*self.dino.head_parameters(), *self.domain_classifier.parameters()]

    def trainable_state_dict(self) -> dict[str, object]:
        return {
            "dino_heads": self.dino.heads.state_dict(),
            "domain_classifier": self.domain_classifier.state_dict(),
        }

    def load_trainable_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state) != {"dino_heads", "domain_classifier"}:
            raise ValueError(
                f"Pretraining discriminator state mismatch: {sorted(state)}"
            )
        self.dino.heads.load_state_dict(state["dino_heads"], strict=True)
        self.domain_classifier.load_state_dict(
            state["domain_classifier"], strict=True
        )

    def logits(
        self,
        images: torch.Tensor,
        domain: str,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if domain not in self.domains:
            raise ValueError(f"Unknown pretraining domain {domain!r}")
        early, late, global_feature = self.dino._features(images)
        adversarial = list(self.dino.heads[domain](early, late, global_feature))
        return adversarial, self.domain_classifier(global_feature)

    def generator_loss(
        self, images: torch.Tensor, domain: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adversarial_logits, domain_logits = self.logits(images, domain)
        adversarial = torch.stack(
            _binary_cross_entropy_losses(adversarial_logits, self.real_label)
        ).mean()
        domain_id = self.domains.index(domain)
        labels = torch.full(
            (images.shape[0],), domain_id, device=images.device, dtype=torch.long
        )
        return adversarial, F.cross_entropy(domain_logits, labels)

    def discriminator_loss(
        self,
        *,
        real: torch.Tensor,
        fake: torch.Tensor,
        domain: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_adversarial, real_domain = self.logits(real, domain)
        fake_adversarial, fake_domain = self.logits(fake, domain)
        adversarial = 0.5 * (
            torch.stack(
                _binary_cross_entropy_losses(real_adversarial, self.real_label)
            ).mean()
            + torch.stack(_binary_cross_entropy_losses(fake_adversarial, 0.0)).mean()
        )
        domain_id = self.domains.index(domain)
        real_labels = torch.full(
            (real.shape[0],), domain_id, device=real.device, dtype=torch.long
        )
        fake_labels = torch.full(
            (fake.shape[0],), domain_id, device=fake.device, dtype=torch.long
        )
        domain_classification = 0.5 * (
            F.cross_entropy(real_domain, real_labels)
            + F.cross_entropy(fake_domain, fake_labels)
        )
        return adversarial, domain_classification

    def signature(self) -> dict[str, object]:
        return {
            "type": "frozen_dino_real_fake_plus_domain_classifier",
            "version": 1,
            "dino": self.dino.discriminator_signature(),
            "domain_classifier": {
                "input_dimension": 768,
                "classes": list(self.domains),
                "normalization": "spectral_norm",
                "loss": "cross_entropy",
            },
        }


class DisentangledVAEPretrainingObjective(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.l1_weight = float(config["l1_weight"])
        self.ssim_weight = float(config["ssim_weight"])
        self.wavelet_weight = float(config["wavelet_weight"])
        self.dino_adversarial_weight = float(config["dino_adversarial_weight"])
        self.domain_adversarial_weight = float(config["domain_adversarial_weight"])
        self.wavelet_level_weights = tuple(
            float(v) for v in config["wavelet_level_weights"]
        )
        self.ssim_window_size = int(config["ssim_window_size"])
        self.ssim_sigma = float(config["ssim_sigma"])

    def reconstruction(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "l1": F.l1_loss(prediction.float(), target.float()),
            "ssim": ssim_loss(
                prediction.float(),
                target.float(),
                window_size=self.ssim_window_size,
                sigma=self.ssim_sigma,
                data_range=2.0,
            ),
            "wavelet": haar_high_band_loss(
                prediction.float(),
                target.float(),
                level_weights=self.wavelet_level_weights,
            ),
        }

    def generator_total(
        self,
        *,
        clean_components: Mapping[str, torch.Tensor],
        hazy_components: Mapping[str, torch.Tensor],
        clean_adversarial: torch.Tensor,
        hazy_adversarial: torch.Tensor,
        clean_domain: torch.Tensor,
        hazy_domain: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction = 0.5 * sum(
            self.l1_weight * components["l1"]
            + self.ssim_weight * components["ssim"]
            + self.wavelet_weight * components["wavelet"]
            for components in (clean_components, hazy_components)
        )
        adversarial = 0.5 * (clean_adversarial + hazy_adversarial)
        domain = 0.5 * (clean_domain + hazy_domain)
        return (
            reconstruction
            + self.dino_adversarial_weight * adversarial
            + self.domain_adversarial_weight * domain
        )

    def signature(self) -> dict[str, object]:
        return {
            "type": "paired_clean_synthetic_haze_reconstruction_plus_adversarial",
            "version": 1,
            "weights": {
                "l1": self.l1_weight,
                "ssim": self.ssim_weight,
                "wavelet": self.wavelet_weight,
                "dino_adversarial": self.dino_adversarial_weight,
                "domain_adversarial": self.domain_adversarial_weight,
            },
            "ssim": {
                "window_size": self.ssim_window_size,
                "sigma": self.ssim_sigma,
                "data_range": 2.0,
            },
            "wavelet_level_weights": list(self.wavelet_level_weights),
        }


def validate_disen_vae_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_step: int = DISEN_VAE_PRETRAINING_STEPS,
) -> dict[str, Any]:
    if payload.get("format") != DISEN_VAE_CHECKPOINT_FORMAT:
        raise ValueError("Not a PRT disentangled-VAE checkpoint")
    if int(payload.get("format_version", -1)) != DISEN_VAE_CHECKPOINT_VERSION:
        raise ValueError("Unsupported PRT disentangled-VAE checkpoint version")
    if int(payload.get("step", -1)) != expected_step:
        raise ValueError(
            f"Expected PRT disentangled-VAE step {expected_step}, got {payload.get('step')}"
        )
    signature = payload.get("pretraining_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("PRT disentangled-VAE checkpoint lacks a signature")
    if canonical_fingerprint(signature) != payload.get("pretraining_signature_sha256"):
        raise ValueError("PRT disentangled-VAE signature fingerprint mismatch")
    if not isinstance(payload.get("decoder_branch_state_dict"), Mapping):
        raise ValueError("PRT disentangled-VAE checkpoint lacks decoder branch state")
    return dict(payload)


def load_disen_vae_checkpoint(
    checkpoint: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_step: int = DISEN_VAE_PRETRAINING_STEPS,
) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing PRT disentangled-VAE checkpoint: {path}")
    if expected_sha256:
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise ValueError(
                f"PRT checkpoint SHA256 mismatch: expected={expected_sha256}, observed={observed}"
            )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    return validate_disen_vae_checkpoint(payload, expected_step=expected_step)


ParallaxResilientTokenizer = DisentangledVAEDecoderBranches
PRTPretrainingDiscriminator = DINOAndDomainPretrainingDiscriminator
PRTPretrainingObjective = DisentangledVAEPretrainingObjective
validate_prt_checkpoint = validate_disen_vae_checkpoint
load_prt_checkpoint = load_disen_vae_checkpoint

__all__ = [
    "DISEN_VAE_CHECKPOINT_FORMAT",
    "DISEN_VAE_CHECKPOINT_VERSION",
    "DISEN_VAE_PRETRAINING_STEPS",
    "DINOAndDomainPretrainingDiscriminator",
    "DisentangledVAEDecoderBranches",
    "DisentangledVAEPretrainingObjective",
    "PRTPretrainingDiscriminator",
    "PRTPretrainingObjective",
    "ParallaxResilientTokenizer",
    "SyntheticHazePipeline",
    "ZeroInitializedRGBResidualAdapter",
    "load_disen_vae_checkpoint",
    "load_prt_checkpoint",
    "validate_disen_vae_checkpoint",
    "validate_prt_checkpoint",
]
