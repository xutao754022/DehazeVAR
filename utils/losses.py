from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import lpips
import torch
from torch import nn
from torch.nn import functional as F


LEGACY_OBJECTIVE_PROFILE = "legacy_v2"
_SUPPORTED_OBJECTIVE_PROFILES = {
    LEGACY_OBJECTIVE_PROFILE,
}
_IMAGE_PATHS = {"cycle", "identity"}
_V3_HAAR_LEVEL_WEIGHTS = (1.0, 0.5, 0.25)
_AUXILIARY_PREFIXES = {
    "scale_ce_loss": "scale_ce_",
    "latent_cycle": "latent_cycle_",
    "asm_loss": "asm_",
    "freq_loss": "frequency_",
    "repre_align": "repa_",
}


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lpips = lpips.LPIPS(net="vgg", verbose=False)
        self.lpips.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.lpips.eval()
        return self

    def components(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        l1 = F.l1_loss(prediction.float(), target.float())
        perceptual = self.lpips(prediction.float(), target.float()).mean()
        return l1, perceptual


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _resolve_profile(objective_type: str | None) -> str:
    profile = (
        LEGACY_OBJECTIVE_PROFILE if objective_type is None else str(objective_type)
    )
    if profile not in _SUPPORTED_OBJECTIVE_PROFILES:
        raise ValueError(
            f"Unknown image objective profile {profile!r}; expected one of "
            f"{sorted(_SUPPORTED_OBJECTIVE_PROFILES)}"
        )
    return profile


def _finite_nonnegative(value: Any, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return resolved


def _legacy_path_weights(
    loss_config: Mapping[str, Any], path: str
) -> dict[str, float]:
    if path not in _IMAGE_PATHS:
        raise ValueError(
            f"Unknown image objective path {path!r}; expected cycle or identity"
        )
    return {
        "l1": _finite_nonnegative(loss_config[f"{path}_l1"], f"{path}_l1"),
        "lpips": _finite_nonnegative(
            loss_config[f"{path}_lpips"], f"{path}_lpips"
        ),
    }


def resolve_adversarial_weights(
    objective_type: str | None,
    loss_config: Mapping[str, Any],
) -> dict[str, float]:
    """Resolve separate generator and discriminator adversarial coefficients."""
    profile = _resolve_profile(objective_type)
    if profile != LEGACY_OBJECTIVE_PROFILE:
        raise AssertionError(f"Unreachable objective profile: {profile}")
    legacy_weight = _finite_nonnegative(loss_config["gan"], "gan")
    return {"generator": legacy_weight, "discriminator": legacy_weight}


def objective_signature(
    objective_type: str | None,
    loss_config: Mapping[str, Any],
) -> dict[str, object]:
    """Return a model-free signature suitable for strict checkpoint matching."""
    profile = _resolve_profile(objective_type)
    adversarial = resolve_adversarial_weights(profile, loss_config)
    if profile != LEGACY_OBJECTIVE_PROFILE:
        raise AssertionError(f"Unreachable objective profile: {profile}")
    signature: dict[str, object] = {
        "type": LEGACY_OBJECTIVE_PROFILE,
        "version": 1,
        "cycle": _legacy_path_weights(loss_config, "cycle"),
        "identity": _legacy_path_weights(loss_config, "identity"),
        "lpips": {"network": "vgg", "reduction": "mean"},
        "adversarial": adversarial,
    }
    auxiliary_type = loss_config.get("auxiliary_cycle_type")
    if auxiliary_type is not None:
        auxiliary_type = str(auxiliary_type)
        if auxiliary_type not in _AUXILIARY_PREFIXES:
            raise ValueError(
                f"Unknown auxiliary cycle objective {auxiliary_type!r}; expected one "
                f"of {sorted(_AUXILIARY_PREFIXES)}"
            )
        prefix = _AUXILIARY_PREFIXES[auxiliary_type]
        signature["auxiliary_cycle"] = {
            "type": auxiliary_type,
            "version": 1,
            "parameters": {
                key: deepcopy(value)
                for key, value in sorted(loss_config.items())
                if key.startswith(prefix)
            },
        }
    return signature


def _validate_image_pair(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim != 4 or prediction.shape != target.shape:
        raise ValueError(
            "Image objective expects equal BxCxHxW tensors, got "
            f"prediction={tuple(prediction.shape)} and target={tuple(target.shape)}"
        )
    if prediction.shape[-2] < 8 or prediction.shape[-1] < 8:
        raise ValueError("Three-level Haar loss requires image height and width >= 8")
    return prediction.float(), target.float()


def _gaussian_window(
    *,
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-coordinates.square() / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Compute differentiable ``1 - mean SSIM`` with a Gaussian window."""
    prediction, target = _validate_image_pair(prediction, target)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return _ssim_loss_fp32(
            prediction,
            target,
            window_size=window_size,
            sigma=sigma,
            data_range=data_range,
        )


def _ssim_loss_fp32(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int,
    sigma: float,
    data_range: float,
) -> torch.Tensor:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(
            f"window_size must be a positive odd integer, got {window_size}"
        )
    padding = window_size // 2
    if prediction.shape[-2] <= padding or prediction.shape[-1] <= padding:
        raise ValueError(
            f"Reflect-padded SSIM({window_size}) requires spatial dimensions "
            f"> {padding}"
        )
    channels = prediction.shape[1]
    window = _gaussian_window(
        window_size=window_size,
        sigma=float(sigma),
        channels=channels,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    prediction = F.pad(prediction, (padding,) * 4, mode="reflect")
    target = F.pad(target, (padding,) * 4, mode="reflect")
    mean_prediction = F.conv2d(prediction, window, groups=channels)
    mean_target = F.conv2d(target, window, groups=channels)
    variance_prediction = (
        F.conv2d(prediction.square(), window, groups=channels)
        - mean_prediction.square()
    )
    variance_target = (
        F.conv2d(target.square(), window, groups=channels) - mean_target.square()
    )
    covariance = (
        F.conv2d(prediction * target, window, groups=channels)
        - mean_prediction * mean_target
    )
    constant_1 = (0.01 * float(data_range)) ** 2
    constant_2 = (0.03 * float(data_range)) ** 2
    numerator = (2.0 * mean_prediction * mean_target + constant_1) * (
        2.0 * covariance + constant_2
    )
    denominator = (
        mean_prediction.square() + mean_target.square() + constant_1
    ) * (variance_prediction + variance_target + constant_2)
    mean_ssim = (numerator / denominator.clamp_min(1e-12)).mean()
    return 1.0 - mean_ssim.clamp(min=-1.0, max=1.0)


def _pad_even(image: torch.Tensor) -> torch.Tensor:
    pad_right = image.shape[-1] % 2
    pad_bottom = image.shape[-2] % 2
    if pad_right or pad_bottom:
        image = F.pad(image, (0, pad_right, 0, pad_bottom), mode="reflect")
    return image


def _haar_decompose(
    image: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    image = _pad_even(image)
    top_left = image[..., 0::2, 0::2]
    top_right = image[..., 0::2, 1::2]
    bottom_left = image[..., 1::2, 0::2]
    bottom_right = image[..., 1::2, 1::2]
    low_low = 0.5 * (top_left + top_right + bottom_left + bottom_right)
    low_high = 0.5 * (-top_left - top_right + bottom_left + bottom_right)
    high_low = 0.5 * (-top_left + top_right - bottom_left + bottom_right)
    high_high = 0.5 * (top_left - top_right - bottom_left + bottom_right)
    return low_low, (low_high, high_low, high_high)


def haar_high_band_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    level_weights: tuple[float, ...] = _V3_HAAR_LEVEL_WEIGHTS,
) -> torch.Tensor:
    """Compare three Haar high bands at each level with normalized weights."""
    prediction, target = _validate_image_pair(prediction, target)
    if not level_weights:
        raise ValueError("Haar loss requires at least one level")
    weights = tuple(
        _finite_nonnegative(weight, f"level_weights[{index}]")
        for index, weight in enumerate(level_weights)
    )
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("At least one Haar level weight must be positive")
    if min(prediction.shape[-2:]) < 2 ** len(weights):
        raise ValueError(
            f"{len(weights)}-level Haar loss requires each spatial dimension >= "
            f"{2 ** len(weights)}"
        )

    total = prediction.new_zeros(())
    prediction_low = prediction
    target_low = target
    for weight in weights:
        prediction_low, prediction_high = _haar_decompose(prediction_low)
        target_low, target_high = _haar_decompose(target_low)
        band_loss = torch.stack(
            [
                F.l1_loss(prediction_band, target_band)
                for prediction_band, target_band in zip(
                    prediction_high, target_high
                )
            ]
        ).mean()
        total = total + (weight / weight_sum) * band_loss
    return total


class LegacyV2ImageObjective(nn.Module):
    """Preserve the L1 + LPIPS cycle and identity reconstruction objective."""

    def __init__(
        self,
        loss_config: Mapping[str, Any],
        reconstruction_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if reconstruction_loss is None:
            reconstruction_loss = ReconstructionLoss()
        if not callable(getattr(reconstruction_loss, "components", None)):
            raise TypeError(
                "reconstruction_loss must provide components(prediction, target)"
            )
        self.reconstruction_loss = reconstruction_loss
        self.loss_config = dict(loss_config)
        self._signature = objective_signature(LEGACY_OBJECTIVE_PROFILE, loss_config)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        path: str = "cycle",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = _legacy_path_weights(self.loss_config, path)
        l1, lpips_value = self.reconstruction_loss.components(prediction, target)
        total = weights["l1"] * l1 + weights["lpips"] * lpips_value
        return total, {"l1": l1, "lpips": lpips_value}

    def objective_signature(self) -> dict[str, object]:
        return deepcopy(self._signature)


def build_image_objective(
    objective_type: str | None,
    loss_config: Mapping[str, Any],
    reconstruction_loss: nn.Module | None = None,
) -> nn.Module:
    profile = _resolve_profile(objective_type)
    if profile != LEGACY_OBJECTIVE_PROFILE:
        raise AssertionError(f"Unreachable image-objective profile: {profile}")
    return LegacyV2ImageObjective(loss_config, reconstruction_loss)


__all__ = [
    "LEGACY_OBJECTIVE_PROFILE",
    "LegacyV2ImageObjective",
    "ReconstructionLoss",
    "build_image_objective",
    "canonical_fingerprint",
    "haar_high_band_loss",
    "objective_signature",
    "resolve_adversarial_weights",
    "sha256_file",
    "ssim_loss",
]
