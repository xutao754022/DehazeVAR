from __future__ import annotations

import torch
from torch.nn import functional as F


def image_to_unit_range(image: torch.Tensor) -> torch.Tensor:
    """Convert the project image convention from [-1, 1] to [0, 1]."""
    return image.float().add(1.0).mul(0.5).clamp(0.0, 1.0)


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    image_unit = image_to_unit_range(image)
    weights = image_unit.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image_unit * weights).sum(dim=1, keepdim=True)


def _fixed_kernel(
    image: torch.Tensor, values: tuple[tuple[float, ...], ...]
) -> torch.Tensor:
    return image.new_tensor(values).view(1, 1, len(values), len(values[0]))


def sobel_feature_maps(image: torch.Tensor) -> torch.Tensor:
    grayscale = rgb_to_grayscale(image)
    sobel_x_kernel = (
        _fixed_kernel(
            grayscale,
            ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
        )
        / 8.0
    )
    sobel_y_kernel = sobel_x_kernel.transpose(-1, -2)
    sobel_x = F.conv2d(grayscale, sobel_x_kernel, padding=1)
    sobel_y = F.conv2d(grayscale, sobel_y_kernel, padding=1)
    gradient_magnitude = torch.sqrt(sobel_x.square() + sobel_y.square() + 1e-8)
    return torch.cat((sobel_x, sobel_y, gradient_magnitude), dim=1)


def wavelet_high_frequency_map(image: torch.Tensor) -> torch.Tensor:
    grayscale = rgb_to_grayscale(image)
    low_frequency = F.interpolate(
        F.avg_pool2d(grayscale, kernel_size=2, stride=2),
        size=grayscale.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return grayscale - low_frequency


def structural_feature_maps(image: torch.Tensor) -> torch.Tensor:
    """Return fixed edge, wavelet, and Fourier-phase maps without raw RGB."""
    grayscale = rgb_to_grayscale(image)
    laplacian_kernel = _fixed_kernel(
        grayscale,
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)),
    )
    sobel = sobel_feature_maps(image)
    laplacian = F.conv2d(grayscale, laplacian_kernel, padding=1)
    wavelet_high_frequency = wavelet_high_frequency_map(image)

    with torch.autocast(device_type=image.device.type, enabled=False):
        spectrum = torch.fft.rfft2(grayscale.float(), norm="ortho")
        phase = torch.angle(spectrum)
        phase_only = torch.polar(torch.ones_like(phase), phase)
        phase_map = torch.fft.irfft2(
            phase_only,
            s=grayscale.shape[-2:],
            norm="ortho",
        )
        phase_scale = phase_map.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6)
        phase_map = phase_map / phase_scale.view(-1, 1, 1, 1)

    return torch.cat(
        (
            sobel,
            laplacian,
            wavelet_high_frequency,
            phase_map.to(grayscale.dtype),
        ),
        dim=1,
    )


def fourier_amplitude_map(image: torch.Tensor, output_size: int) -> torch.Tensor:
    grayscale = rgb_to_grayscale(image)
    with torch.autocast(device_type=image.device.type, enabled=False):
        amplitude = torch.fft.rfft2(grayscale.float(), norm="ortho").abs().log1p()
        amplitude = torch.fft.fftshift(amplitude, dim=(-2,))
        mean = amplitude.flatten(1).mean(dim=1, keepdim=True)
        standard_deviation = (
            amplitude.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6)
        )
        amplitude = (amplitude - mean.view(-1, 1, 1, 1)) / standard_deviation.view(
            -1, 1, 1, 1
        )
    return F.interpolate(
        amplitude,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )


def rgb_log_amplitude_map(image: torch.Tensor, output_size: int) -> torch.Tensor:
    """Return absolute RGB log-amplitude without per-image standardization."""
    image_unit = image_to_unit_range(image)
    with torch.autocast(device_type=image.device.type, enabled=False):
        spectrum = torch.fft.fft2(image_unit.float(), norm="ortho")
        amplitude = torch.fft.fftshift(spectrum.abs().log1p(), dim=(-2, -1))
    return F.interpolate(
        amplitude,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )


@torch.no_grad()
def dcp_rgb_log_amplitude_residual_map(
    image: torch.Tensor,
    output_size: int,
    *,
    patch_size: int,
    omega: float,
) -> torch.Tensor:
    """Estimate a signed hazy-to-clear RGB log-amplitude residual with fixed DCP."""
    transmission, atmosphere = dark_channel_prior_targets(
        image, patch_size=patch_size, omega=omega
    )
    image_unit = image_to_unit_range(image)
    atmosphere = atmosphere.view(image.shape[0], 3, 1, 1)
    pseudo_clear = (
        (image_unit - atmosphere * (1.0 - transmission))
        / transmission.clamp_min(0.05)
    ).clamp(0.0, 1.0)
    with torch.autocast(device_type=image.device.type, enabled=False):
        hazy_spectrum = torch.fft.fft2(image_unit.float(), norm="ortho")
        clear_spectrum = torch.fft.fft2(pseudo_clear.float(), norm="ortho")
        residual = torch.fft.fftshift(
            clear_spectrum.abs().log1p() - hazy_spectrum.abs().log1p(),
            dim=(-2, -1),
        )
    return F.interpolate(
        residual,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )


def rgb_phase_unit_map(image: torch.Tensor, output_size: int) -> torch.Tensor:
    """Represent RGB FFT phase as stable real/imaginary unit vectors."""
    image_unit = image_to_unit_range(image)
    with torch.autocast(device_type=image.device.type, enabled=False):
        spectrum = torch.fft.fft2(image_unit.float(), norm="ortho")
        unit = spectrum / spectrum.abs().clamp_min(1e-8)
        phase = torch.cat((unit.real, unit.imag), dim=1)
        phase = torch.fft.fftshift(phase, dim=(-2, -1))
    return F.interpolate(
        phase,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )


def radial_frequency_mask(
    height: int,
    width: int,
    *,
    minimum_radius: float,
    maximum_radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a centered radial mask in cycles per pixel after ``fftshift``."""
    if not 0.0 <= minimum_radius < maximum_radius <= 0.75:
        raise ValueError(
            "Expected 0 <= minimum_radius < maximum_radius <= 0.75, got "
            f"{minimum_radius}, {maximum_radius}"
        )
    frequencies_y = torch.fft.fftshift(
        torch.fft.fftfreq(height, device=device, dtype=dtype)
    )
    frequencies_x = torch.fft.fftshift(
        torch.fft.fftfreq(width, device=device, dtype=dtype)
    )
    radius = torch.sqrt(
        frequencies_y[:, None].square() + frequencies_x[None, :].square()
    )
    return ((radius >= minimum_radius) & (radius < maximum_radius)).to(dtype)


def masked_log_amplitude_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    minimum_radius: float,
    maximum_radius: float,
) -> torch.Tensor:
    """Compute the requested log-amplitude L1 inside one radial band."""
    prediction_unit = image_to_unit_range(prediction)
    target_unit = image_to_unit_range(target)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction_spectrum = torch.fft.fftshift(
            torch.fft.fft2(prediction_unit.float(), norm="ortho"), dim=(-2, -1)
        )
        target_spectrum = torch.fft.fftshift(
            torch.fft.fft2(target_unit.float(), norm="ortho"), dim=(-2, -1)
        )
        mask = radial_frequency_mask(
            prediction.shape[-2],
            prediction.shape[-1],
            minimum_radius=minimum_radius,
            maximum_radius=maximum_radius,
            device=prediction.device,
            dtype=torch.float32,
        ).view(1, 1, prediction.shape[-2], prediction.shape[-1])
        difference = (
            prediction_spectrum.abs().log1p()
            - target_spectrum.abs().log1p()
        ).abs()
        normalizer = mask.sum() * prediction.shape[0] * prediction.shape[1]
        return (difference * mask).sum() / normalizer.clamp_min(1.0)


def masked_phase_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    minimum_radius: float,
    maximum_radius: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute a stable amplitude-confident cosine loss between FFT phases."""
    prediction_unit = image_to_unit_range(prediction)
    target_unit = image_to_unit_range(target)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction_spectrum = torch.fft.fftshift(
            torch.fft.fft2(prediction_unit.float(), norm="ortho"), dim=(-2, -1)
        )
        target_spectrum = torch.fft.fftshift(
            torch.fft.fft2(target_unit.float(), norm="ortho"), dim=(-2, -1)
        )
        magnitude_product = prediction_spectrum.abs() * target_spectrum.abs()
        phase_cosine = (
            prediction_spectrum * target_spectrum.conj()
        ).real / magnitude_product.clamp_min(epsilon)
        phase_distance = 1.0 - phase_cosine.clamp(-1.0, 1.0)
        radial_mask = radial_frequency_mask(
            prediction.shape[-2],
            prediction.shape[-1],
            minimum_radius=minimum_radius,
            maximum_radius=maximum_radius,
            device=prediction.device,
            dtype=torch.float32,
        ).view(1, 1, prediction.shape[-2], prediction.shape[-1])
        confidence = magnitude_product.detach().sqrt()
        weights = radial_mask * confidence
        return (phase_distance * weights).sum() / weights.sum().clamp_min(epsilon)


@torch.no_grad()
def dark_channel_prior_targets(
    image: torch.Tensor,
    *,
    patch_size: int = 15,
    omega: float = 0.95,
    top_fraction: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate soft transmission and atmospheric light with a fixed DCP operator."""
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(
            f"DCP patch_size must be a positive odd integer, got {patch_size}"
        )
    image_unit = image_to_unit_range(image)
    dark = image_unit.amin(dim=1, keepdim=True)
    dark = -F.max_pool2d(
        -dark,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
    )

    batch, _, height, width = dark.shape
    candidate_count = max(1, int(height * width * top_fraction))
    candidate_indices = dark.flatten(1).topk(candidate_count, dim=1).indices
    rgb_flat = image_unit.flatten(2).transpose(1, 2)
    candidates = torch.gather(
        rgb_flat,
        dim=1,
        index=candidate_indices.unsqueeze(-1).expand(-1, -1, 3),
    )
    candidate_luminance = candidates.mean(dim=-1)
    atmosphere_index = candidate_luminance.argmax(dim=1)
    atmosphere = candidates[
        torch.arange(batch, device=image.device), atmosphere_index
    ].clamp_min(0.05)

    normalized = image_unit / atmosphere.view(batch, 3, 1, 1)
    normalized_dark = normalized.amin(dim=1, keepdim=True)
    normalized_dark = -F.max_pool2d(
        -normalized_dark,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
    )
    transmission = (1.0 - float(omega) * normalized_dark).clamp(0.05, 1.0)
    return transmission, atmosphere


def synthesize_ranked_haze_pair(
    clear_image: torch.Tensor,
    *,
    light_transmission: float,
    heavy_transmission: float,
    atmospheric_light: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < heavy_transmission < light_transmission <= 1.0:
        raise ValueError(
            "Expected 0 < heavy_transmission < light_transmission <= 1, got "
            f"{heavy_transmission}, {light_transmission}"
        )
    clear_unit = image_to_unit_range(clear_image)
    atmosphere = clear_unit.new_full((clear_unit.shape[0], 3, 1, 1), atmospheric_light)
    light = clear_unit * light_transmission + atmosphere * (1.0 - light_transmission)
    heavy = clear_unit * heavy_transmission + atmosphere * (1.0 - heavy_transmission)
    return light.mul(2.0).sub(1.0), heavy.mul(2.0).sub(1.0)


def edge_aware_smoothness_loss(
    density_map: torch.Tensor,
    image: torch.Tensor,
    *,
    edge_weight: float,
) -> torch.Tensor:
    grayscale = F.interpolate(
        rgb_to_grayscale(image),
        size=density_map.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    density_dx = density_map[..., :, 1:] - density_map[..., :, :-1]
    density_dy = density_map[..., 1:, :] - density_map[..., :-1, :]
    image_dx = grayscale[..., :, 1:] - grayscale[..., :, :-1]
    image_dy = grayscale[..., 1:, :] - grayscale[..., :-1, :]
    loss_x = (density_dx.abs() * torch.exp(-edge_weight * image_dx.abs())).mean()
    loss_y = (density_dy.abs() * torch.exp(-edge_weight * image_dy.abs())).mean()
    return loss_x + loss_y


def sobel_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction_features = sobel_feature_maps(prediction)
    target_features = sobel_feature_maps(target)
    return F.l1_loss(prediction_features, target_features)


def wavelet_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction_wavelet = wavelet_high_frequency_map(prediction)
    target_wavelet = wavelet_high_frequency_map(target)
    return F.l1_loss(prediction_wavelet, target_wavelet)


__all__ = [
    "dcp_rgb_log_amplitude_residual_map",
    "dark_channel_prior_targets",
    "edge_aware_smoothness_loss",
    "fourier_amplitude_map",
    "image_to_unit_range",
    "masked_log_amplitude_loss",
    "masked_phase_cosine_loss",
    "radial_frequency_mask",
    "rgb_to_grayscale",
    "rgb_log_amplitude_map",
    "rgb_phase_unit_map",
    "sobel_feature_maps",
    "sobel_reconstruction_loss",
    "structural_feature_maps",
    "synthesize_ranked_haze_pair",
    "wavelet_high_frequency_map",
    "wavelet_reconstruction_loss",
]
