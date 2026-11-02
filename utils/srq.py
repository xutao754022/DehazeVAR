from __future__ import annotations

import torch
from torch.nn import functional as F


def softmax_relaxed_quantize(
    logits: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float,
    use_gumbel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ``[..., V]`` logits to differentiable ``[..., C]`` code vectors."""
    if logits.ndim < 2:
        raise ValueError(f"Expected logits with a vocabulary axis, got {logits.shape}")
    if codebook.ndim != 2 or logits.shape[-1] != codebook.shape[0]:
        raise ValueError(
            f"Incompatible logits/codebook shapes: {logits.shape} vs {codebook.shape}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    logits_fp32 = logits.float()
    if use_gumbel:
        probabilities = F.gumbel_softmax(
            logits_fp32, tau=float(temperature), hard=False, dim=-1
        )
    else:
        probabilities = torch.softmax(logits_fp32 / float(temperature), dim=-1)
    embeddings = probabilities @ codebook.float()
    return embeddings, probabilities


def straight_through_codebook(
    residual: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour code vectors with identity gradients to ``residual``."""
    if residual.shape[-1] != codebook.shape[-1]:
        raise ValueError(f"Channel mismatch: {residual.shape} vs {codebook.shape}")
    with torch.no_grad():
        flat = residual.detach().reshape(-1, residual.shape[-1]).float()
        weight = codebook.detach().float()
        # Preserve the upstream VAR quantizer's operation order.  The nearest
        # two code vectors can be extremely close, so a mathematically
        # equivalent rearrangement can change argmin through FP32 rounding.
        distances = flat.square().sum(dim=1, keepdim=True) + weight.square().sum(
            dim=1
        )
        distances.addmm_(flat, weight.t(), beta=1.0, alpha=-2.0)
        indices = distances.argmin(dim=1).view(residual.shape[:-1])
        hard = F.embedding(indices, codebook)
    quantized = residual + (hard.to(residual.dtype) - residual).detach()
    return quantized, indices
