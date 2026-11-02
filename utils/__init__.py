from __future__ import annotations

from utils.haze_operators import (
    dark_channel_prior_targets,
    dcp_rgb_log_amplitude_residual_map,
    edge_aware_smoothness_loss,
    fourier_amplitude_map,
    image_to_unit_range,
    masked_log_amplitude_loss,
    masked_phase_cosine_loss,
    radial_frequency_mask,
    rgb_log_amplitude_map,
    rgb_phase_unit_map,
    rgb_to_grayscale,
    sobel_feature_maps,
    sobel_reconstruction_loss,
    structural_feature_maps,
    synthesize_ranked_haze_pair,
    wavelet_high_frequency_map,
    wavelet_reconstruction_loss,
)
from utils.io_utils import (
    PROJECT_ROOT,
    append_jsonl,
    atomic_write_json,
    finite_scalar_dict,
    linear_warmup_multiplier,
    load_config,
    project_root,
    resolve_project_path,
    seed_everything,
    seed_worker,
    set_requires_grad,
    trainable_parameters,
)
from utils.losses import (
    LEGACY_OBJECTIVE_PROFILE,
    LegacyV2ImageObjective,
    ReconstructionLoss,
    build_image_objective,
    canonical_fingerprint,
    haar_high_band_loss,
    objective_signature,
    resolve_adversarial_weights,
    sha256_file,
    ssim_loss,
)
from utils.srq import (
    softmax_relaxed_quantize,
    straight_through_codebook,
)
