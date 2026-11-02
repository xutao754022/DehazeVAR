from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.cra_adapter import (
    AmplitudeResidualTokenEncoder,
    HazeAwareSharedIDDetailAdapter,
    HazeFactorizedRepresentation,
    HazeFactorizedSideEncoder,
    PhaseCorrectionTokenEncoder,
    TargetDomainResidualAdapter,
)
from models.prt_tokenizer import (
    DISEN_VAE_PRETRAINING_STEPS,
    DisentangledVAEDecoderBranches,
    load_disen_vae_checkpoint,
)
from utils.haze_operators import (
    dark_channel_prior_targets,
    dcp_rgb_log_amplitude_residual_map,
    edge_aware_smoothness_loss,
    image_to_unit_range,
    masked_log_amplitude_loss,
    masked_phase_cosine_loss,
    rgb_phase_unit_map,
    sobel_reconstruction_loss,
    synthesize_ranked_haze_pair,
    wavelet_reconstruction_loss,
)
from utils.io_utils import project_root, resolve_project_path
from utils.losses import sha256_file
from utils.srq import softmax_relaxed_quantize, straight_through_codebook


VAR_ROOT = project_root() / "third_party" / "VAR"
if str(VAR_ROOT) not in sys.path:
    sys.path.insert(0, str(VAR_ROOT))

from third_party.VAR.models import build_vae_var  # noqa: E402


_RESET_CLASSES = (
    nn.Linear,
    nn.LayerNorm,
    nn.BatchNorm2d,
    nn.SyncBatchNorm,
    nn.Conv1d,
    nn.Conv2d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
)

DOMAIN_TO_ID = {"hazy": 0, "clean": 1}


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CycleVARGenerator(nn.Module):
    """Parallel one-step CycleVAR generator built on the official VAR-d16."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        model_config = config["model"]
        self.patch_nums = tuple(int(value) for value in model_config["patch_nums"])
        self.temperature = float(model_config["srq_temperature"])
        self.use_gumbel = bool(model_config["srq_gumbel"])
        self.blend_alpha = float(model_config["blend_alpha"])
        self.gradient_checkpointing = bool(
            model_config.get("gradient_checkpointing", True)
        )
        if not 0.0 <= self.blend_alpha <= 1.0:
            raise ValueError(f"blend_alpha must be in [0, 1], got {self.blend_alpha}")

        reset_methods = {klass: klass.reset_parameters for klass in _RESET_CLASSES}
        try:
            vae, var = build_vae_var(
                device=device,
                patch_nums=self.patch_nums,
                depth=int(model_config["depth"]),
                flash_if_available=True,
                fused_if_available=False,
            )
        finally:
            for klass, reset_method in reset_methods.items():
                klass.reset_parameters = reset_method

        vae_path = resolve_project_path(model_config["vae_checkpoint"])
        var_path = resolve_project_path(model_config["var_checkpoint"])
        if not vae_path.is_file() or not var_path.is_file():
            raise FileNotFoundError(
                f"Missing pretrained weights: vae={vae_path}, var={var_path}"
            )
        vae.load_state_dict(_load_state_dict(vae_path), strict=True)
        var.load_state_dict(_load_state_dict(var_path), strict=True)

        vae.eval()
        vae.requires_grad_(False)
        var.cond_drop_rate = 0.0
        self.vae = vae
        self.var = var

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.var.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    def _source_context(
        self,
        image: torch.Tensor,
        *,
        return_continuous: bool = False,
        return_prefixes: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """STE tokenize, sum all residuals, then downsample to every scale."""
        quantizer = self.vae.quantize
        feature = self.vae.quant_conv(self.vae.encoder(image))
        batch, _, height, width = feature.shape
        if (height, width) != (self.patch_nums[-1], self.patch_nums[-1]):
            raise ValueError(
                f"VAE latent is {height}x{width}; expected {self.patch_nums[-1]}x"
                f"{self.patch_nums[-1]} for 256x256 input"
            )

        with torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
            residual = feature.detach().float()
            hard_fhat = torch.zeros_like(residual)
            all_indices: list[torch.Tensor] = []
            hard_prefixes: list[torch.Tensor] = []
            scale_count = len(self.patch_nums)
            for scale_index, patch_num in enumerate(self.patch_nums):
                scale_residual = (
                    F.interpolate(residual, size=(patch_num, patch_num), mode="area")
                    if scale_index != scale_count - 1
                    else residual
                )
                scale_bhwc = scale_residual.permute(0, 2, 3, 1).contiguous()
                hard_bhwc, indices = straight_through_codebook(
                    scale_bhwc, quantizer.embedding.weight
                )
                hard_map = hard_bhwc.permute(0, 3, 1, 2).contiguous()
                if scale_index != scale_count - 1:
                    hard_map = F.interpolate(
                        hard_map,
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                    )
                ratio = scale_index / (scale_count - 1)
                residual_map = quantizer.quant_resi[ratio](hard_map)
                hard_fhat = hard_fhat + residual_map
                residual = residual - residual_map
                all_indices.append(indices.reshape(batch, -1))
                if return_prefixes:
                    hard_prefixes.append(hard_fhat.clone())

        feature_fp32 = feature.float()
        full_fhat = feature_fp32 + (hard_fhat - feature_fp32).detach()

        contexts = []
        for patch_num in self.patch_nums:
            scale_feature = (
                F.interpolate(full_fhat, size=(patch_num, patch_num), mode="area")
                if patch_num != self.patch_nums[-1]
                else full_fhat
            )
            contexts.append(scale_feature.flatten(2).transpose(1, 2))
        context = torch.cat(contexts, dim=1)
        indices = torch.cat(all_indices, dim=1)
        outputs: list[object] = [context, full_fhat, indices]
        if return_continuous:
            outputs.extend((feature_fp32, hard_fhat.detach()))
        if return_prefixes:
            outputs.append(tuple(prefix.detach() for prefix in hard_prefixes))
        return tuple(outputs)

    def _parallel_var_forward(
        self,
        target_labels: torch.Tensor,
        context: torch.Tensor,
        token_condition: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """All source scales enter one causal VAR forward pass."""
        var = self.var
        batch = context.shape[0]
        if context.shape[1] != var.L:
            raise ValueError(f"Expected context length {var.L}, got {context.shape}")

        device_type = context.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            condition = var.class_emb(target_labels)
            x = var.word_embed(context.float())
            if token_condition is not None:
                if token_condition.shape != x.shape:
                    raise ValueError(
                        "VAR token condition must match the embedded source context: "
                        f"condition={token_condition.shape}, context={x.shape}"
                    )
                x = x + token_condition.float()
            x = x + var.lvl_embed(var.lvl_1L.expand(batch, -1)) + var.pos_1LC
            x = x + F.pad(var.pos_start, (0, 0, 0, var.L - var.first_l))
            attention_bias = var.attn_bias_for_masking
            conditioned = var.shared_ada_lin(condition)

        probe = x.new_ones(8, 8)
        main_dtype = torch.matmul(probe, probe).dtype
        x = x.to(dtype=main_dtype)
        conditioned = conditioned.to(dtype=main_dtype)
        attention_bias = attention_bias.to(dtype=main_dtype)

        for block in var.blocks:
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(
                    lambda x_, cond_, bias_, current_block=block: current_block(
                        x=x_, cond_BD=cond_, attn_bias=bias_
                    ),
                    x,
                    conditioned,
                    attention_bias,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                x = block(x=x, cond_BD=conditioned, attn_bias=attention_bias)
        hidden = x.float()
        logits = var.get_logits(hidden, condition)
        if return_hidden:
            return logits, hidden
        return logits

    def _srq_latent(
        self,
        logits: torch.Tensor,
        source_fhat: torch.Tensor,
        *,
        return_prefixes: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Build the blended SRQ latent once, before any decoder-side extension."""
        quantizer = self.vae.quantize
        batch = logits.shape[0]
        begin = 0
        predicted_fhat = torch.zeros_like(source_fhat, dtype=torch.float32)
        normalized_entropies: list[torch.Tensor] = []
        predicted_prefixes: list[torch.Tensor] = []
        scale_count = len(self.patch_nums)
        use_gumbel = self.training and self.use_gumbel

        for scale_index, patch_num in enumerate(self.patch_nums):
            length = patch_num * patch_num
            scale_logits = logits[:, begin : begin + length]
            embeddings, probabilities = softmax_relaxed_quantize(
                scale_logits,
                quantizer.embedding.weight,
                temperature=self.temperature,
                use_gumbel=use_gumbel,
            )
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            normalized_entropies.append(
                entropy.mean() / torch.log(entropy.new_tensor(probabilities.shape[-1]))
            )
            embedding_map = embeddings.transpose(1, 2).reshape(
                batch, self.vae.Cvae, patch_num, patch_num
            )
            if scale_index != scale_count - 1:
                embedding_map = F.interpolate(
                    embedding_map,
                    size=source_fhat.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
            ratio = scale_index / (scale_count - 1)
            predicted_fhat = predicted_fhat + quantizer.quant_resi[ratio](
                embedding_map
            ).float()
            if return_prefixes:
                predicted_prefixes.append(predicted_fhat.clone())
            begin += length

        blended = (
            self.blend_alpha * predicted_fhat
            + (1.0 - self.blend_alpha) * source_fhat.float()
        )
        mean_entropy = torch.stack(normalized_entropies).mean()
        if return_prefixes:
            return blended, mean_entropy, tuple(predicted_prefixes)
        return blended, mean_entropy

    def _srq_decode(
        self,
        logits: torch.Tensor,
        source_fhat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blended, mean_entropy = self._srq_latent(logits, source_fhat)
        image = self.vae.decoder(self.vae.post_quant_conv(blended)).clamp(-1.0, 1.0)
        return image, mean_entropy

    def forward(
        self,
        source: torch.Tensor,
        target_class_id: int | torch.Tensor,
        return_aux: bool = False,
    ):
        if source.ndim != 4 or source.shape[1:] != (3, 256, 256):
            raise ValueError(f"Expected Bx3x256x256 source images, got {source.shape}")
        if isinstance(target_class_id, int):
            labels = torch.full(
                (source.shape[0],),
                target_class_id,
                dtype=torch.long,
                device=source.device,
            )
        else:
            labels = target_class_id.to(device=source.device, dtype=torch.long)

        context, source_fhat, source_indices = self._source_context(source)
        logits = self._parallel_var_forward(labels, context)
        translated, entropy = self._srq_decode(logits, source_fhat)
        if not return_aux:
            return translated
        unique_fraction = torch.unique(source_indices.detach()).numel() / float(
            self.vae.V
        )
        return translated, {
            "srq_entropy": float(entropy.detach()),
            "source_code_unique_fraction": float(unique_fraction),
        }


class CycleVARUnpairedDehazingGenerator(CycleVARGenerator):
    """Adapt the pretrained class condition to two trainable restoration domains."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device=device)
        model_config = config["model"]
        initialization = str(model_config["domain_embedding_initialization"])
        if initialization != "imagenet-mean-plus-noise":
            raise ValueError(
                "domain_embedding_initialization must be "
                f"'imagenet-mean-plus-noise', got {initialization!r}"
            )

        pretrained_class_count = int(model_config["pretrained_class_count"])
        pretrained_embedding = self.var.class_emb.weight.detach().float()
        if not 1 <= pretrained_class_count <= pretrained_embedding.shape[0]:
            raise ValueError(
                "pretrained_class_count is incompatible with the VAR class embedding: "
                f"{pretrained_class_count} vs {pretrained_embedding.shape[0]}"
            )

        class_embedding = pretrained_embedding[:pretrained_class_count]
        mean_embedding = class_embedding.mean(dim=0, keepdim=True)
        mean_standard_deviation = class_embedding.std(dim=0).mean()
        noise_scale = float(model_config["domain_embedding_noise_scale"])
        domain_embedding = nn.Embedding(
            len(DOMAIN_TO_ID),
            pretrained_embedding.shape[1],
            device=device,
            dtype=self.var.class_emb.weight.dtype,
        )
        with torch.no_grad():
            domain_embedding.weight.copy_(mean_embedding.repeat(len(DOMAIN_TO_ID), 1))
            domain_embedding.weight.add_(
                torch.randn_like(domain_embedding.weight)
                * mean_standard_deviation.to(domain_embedding.weight.dtype)
                * noise_scale
            )
        self.var.class_emb = domain_embedding
        self.var.cond_drop_rate = 0.0

    @staticmethod
    def domain_id(domain: str | int) -> int:
        if isinstance(domain, int):
            if domain not in DOMAIN_TO_ID.values():
                raise ValueError(f"Unknown restoration domain id: {domain}")
            return domain
        if domain not in DOMAIN_TO_ID:
            raise ValueError(
                f"Unknown restoration domain {domain!r}; expected {tuple(DOMAIN_TO_ID)}"
            )
        return DOMAIN_TO_ID[domain]

    def forward(
        self,
        source: torch.Tensor,
        target_domain: str | int | torch.Tensor,
        return_aux: bool = False,
    ):
        if isinstance(target_domain, str):
            target_domain = self.domain_id(target_domain)
        return super().forward(source, target_domain, return_aux=return_aux)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def additional_trainable_parameters(self) -> list[nn.Parameter]:
        return []

    def extension_signature(self) -> dict[str, object]:
        return {"type": "none"}

    def extension_state_dict(self) -> dict[str, object]:
        return {}

    def load_extension_state_dict(self, state: dict[str, object]) -> None:
        if state:
            raise ValueError("Baseline generator received unexpected extension state")

    def adapter_training_loss(
        self,
        *,
        hazy: torch.Tensor,
        clean: torch.Tensor,
        step: int,
        reconstruction: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del hazy, clean, step, reconstruction
        zero = self.var.class_emb.weight.new_zeros(())
        return zero, {}

    def cycle_auxiliary_loss(
        self,
        *,
        source: torch.Tensor,
        source_domain: str,
        first_auxiliary: dict[str, object],
        second_auxiliary: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del source, source_domain, first_auxiliary, second_auxiliary
        zero = self.var.class_emb.weight.new_zeros(())
        return zero, {}

    def identity_auxiliary_loss(
        self,
        *,
        source: torch.Tensor,
        source_domain: str,
        auxiliary: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del source, source_domain, auxiliary
        zero = self.var.class_emb.weight.new_zeros(())
        return zero, {}

    @property
    def adapter_pretraining_steps(self) -> int:
        return 0

    def adapter_pretraining_loss(
        self,
        clean: torch.Tensor,
        reconstruction: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del clean, reconstruction
        raise RuntimeError("This generator does not define adapter pretraining")

    @property
    def adapter_identity_pretraining_steps(self) -> int:
        return 0

    def adapter_identity_pretraining_loss(
        self,
        clean: torch.Tensor,
        reconstruction: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del clean, reconstruction
        raise RuntimeError(
            "This generator does not define identity-branch adapter pretraining"
        )

    def finalize_adapter_pretraining(self) -> None:
        """Restore any output gate required for strict joint step-zero parity."""


class CycleVARCRAUnpairedDehazingGenerator(CycleVARUnpairedDehazingGenerator):
    """CycleVAR with target-domain residual experts at the frozen decoder input."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device=device)
        model_config = config["model"]
        self.cra_reduction_ratio = int(model_config["cra_reduction_ratio"])
        self.cra_activation = str(model_config["cra_activation"])
        adapter_seed = int(model_config["vae_adapter_initialization_seed"])
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(adapter_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(adapter_seed)
            self.conditional_residual_adapter = TargetDomainResidualAdapter(
                channels=int(self.vae.Cvae),
                reduction_ratio=self.cra_reduction_ratio,
                activation=self.cra_activation,
            )

    @property
    def decoder_adapter(self) -> TargetDomainResidualAdapter:
        return self.conditional_residual_adapter

    def forward(
        self,
        source: torch.Tensor,
        target_domain: str | int | torch.Tensor,
        return_aux: bool = False,
    ):
        if source.ndim != 4 or source.shape[1:] != (3, 256, 256):
            raise ValueError(f"Expected Bx3x256x256 source images, got {source.shape}")
        if isinstance(target_domain, str):
            target_domain = self.domain_id(target_domain)
        if isinstance(target_domain, int):
            labels = torch.full(
                (source.shape[0],),
                target_domain,
                dtype=torch.long,
                device=source.device,
            )
        else:
            labels = target_domain.to(device=source.device, dtype=torch.long)

        context, source_fhat, source_indices = self._source_context(source)
        logits = self._parallel_var_forward(labels, context)
        blended_latent, entropy = self._srq_latent(logits, source_fhat)
        decoder_input = self.vae.post_quant_conv(blended_latent)
        decoder_residual = self.decoder_adapter(decoder_input, labels)
        translated = self.vae.decoder(decoder_input + decoder_residual).clamp(-1.0, 1.0)
        if not return_aux:
            return translated
        unique_fraction = torch.unique(source_indices.detach()).numel() / float(
            self.vae.V
        )
        return translated, {
            "srq_entropy": float(entropy.detach()),
            "source_code_unique_fraction": float(unique_fraction),
            "cra_decoder_residual_absolute_mean": float(
                decoder_residual.detach().float().abs().mean()
            ),
        }

    def additional_trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.decoder_adapter.parameters())

    def extension_signature(self) -> dict[str, object]:
        return {
            "type": "cra",
            "version": 1,
            "injection_point": "post_quant_conv_pre_decoder",
            "routing": "target_domain",
            "adapter": self.decoder_adapter.signature(),
            "vae_frozen": True,
            "token_vocabulary_unchanged": True,
        }

    def extension_state_dict(self) -> dict[str, object]:
        return {"decoder_adapter": self.decoder_adapter.state_dict()}

    def load_extension_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"decoder_adapter"}:
            raise ValueError(
                "CRA extension state must contain only decoder_adapter, got "
                f"{sorted(state)}"
            )
        self.decoder_adapter.load_state_dict(state["decoder_adapter"], strict=True)


class CycleVARHFSEUnpairedDehazingGenerator(CycleVARUnpairedDehazingGenerator):
    """CycleVAR with a factorized haze side encoder and zero-initialized injection."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device=device)
        model_config = config["model"]
        self.hfse_base_channels = int(model_config["hfse_base_channels"])
        self.hfse_global_dimension = int(model_config["hfse_global_dimension"])
        self.hfse_amplitude_dimension = int(model_config["hfse_amplitude_dimension"])
        self.hfse_structure_dimension = int(model_config["hfse_structure_dimension"])
        self.hfse_dcp_patch_size = int(model_config["hfse_dcp_patch_size"])
        self.hfse_dcp_omega = float(model_config["hfse_dcp_omega"])
        self.hfse_light_transmission = float(model_config["hfse_light_transmission"])
        self.hfse_heavy_transmission = float(model_config["hfse_heavy_transmission"])
        self.hfse_synthetic_atmospheric_light = float(
            model_config["hfse_synthetic_atmospheric_light"]
        )
        self.hfse_ranking_margin = float(model_config["hfse_ranking_margin"])
        self.hfse_edge_weight = float(model_config["hfse_edge_weight"])
        self.hfse_loss_config = dict(config["loss"])
        adapter_seed = int(model_config["vae_adapter_initialization_seed"])
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(adapter_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(adapter_seed)
            self.haze_factorized_side_encoder = HazeFactorizedSideEncoder(
                base_channels=self.hfse_base_channels,
                global_dimension=self.hfse_global_dimension,
                amplitude_dimension=self.hfse_amplitude_dimension,
                structure_dimension=self.hfse_structure_dimension,
                latent_size=self.patch_nums[-1],
            )
            token_factor_dimension = (
                self.hfse_global_dimension
                + 3
                + self.hfse_amplitude_dimension
                + self.hfse_structure_dimension
            )
            var_embedding_dimension = int(self.var.word_embed.out_features)
            self.hfse_scale_projections = nn.ModuleList(
                nn.Sequential(
                    nn.LayerNorm(token_factor_dimension + 1),
                    nn.Linear(token_factor_dimension + 1, var_embedding_dimension),
                )
                for _ in self.patch_nums
            )
            for projection in self.hfse_scale_projections:
                nn.init.zeros_(projection[-1].weight)
                nn.init.zeros_(projection[-1].bias)

    def _token_condition(
        self,
        factors: HazeFactorizedRepresentation,
    ) -> torch.Tensor:
        global_factors = torch.cat(
            (
                factors.global_embedding,
                factors.atmospheric_light,
                factors.amplitude_embedding,
            ),
            dim=1,
        )
        conditions = []
        for patch_num, projection in zip(self.patch_nums, self.hfse_scale_projections):
            density = (
                F.interpolate(
                    factors.haze_density_map,
                    size=(patch_num, patch_num),
                    mode="bilinear",
                    align_corners=False,
                )
                .flatten(2)
                .transpose(1, 2)
            )
            repeated_global = global_factors.unsqueeze(1).expand(
                -1, patch_num * patch_num, -1
            )
            structure = (
                F.interpolate(
                    factors.structure_embedding,
                    size=(patch_num, patch_num),
                    mode="bilinear",
                    align_corners=False,
                )
                .flatten(2)
                .transpose(1, 2)
            )
            conditions.append(
                projection(torch.cat((repeated_global, density, structure), dim=-1))
            )
        return torch.cat(conditions, dim=1)

    def forward(
        self,
        source: torch.Tensor,
        target_domain: str | int | torch.Tensor,
        return_aux: bool = False,
    ):
        if source.ndim != 4 or source.shape[1:] != (3, 256, 256):
            raise ValueError(f"Expected Bx3x256x256 source images, got {source.shape}")
        if isinstance(target_domain, str):
            target_domain = self.domain_id(target_domain)
        if isinstance(target_domain, int):
            labels = torch.full(
                (source.shape[0],),
                target_domain,
                dtype=torch.long,
                device=source.device,
            )
        else:
            labels = target_domain.to(device=source.device, dtype=torch.long)

        context, source_fhat, source_indices = self._source_context(source)
        factors = self.haze_factorized_side_encoder(source)
        logits = self._parallel_var_forward(
            labels,
            context,
            token_condition=self._token_condition(factors),
        )
        translated, entropy = self._srq_decode(logits, source_fhat)
        if not return_aux:
            return translated
        unique_fraction = torch.unique(source_indices.detach()).numel() / float(
            self.vae.V
        )
        return translated, {
            "srq_entropy": float(entropy.detach()),
            "source_code_unique_fraction": float(unique_fraction),
            "hfse_haze_density_mean": float(factors.haze_density_map.detach().mean()),
            "hfse_severity_score": float(factors.severity_logit.detach().mean()),
            "hfse_atmospheric_light_mean": float(
                factors.atmospheric_light.detach().mean()
            ),
        }

    def additional_trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.haze_factorized_side_encoder.parameters()) + list(
            self.hfse_scale_projections.parameters()
        )

    def extension_signature(self) -> dict[str, object]:
        return {
            "type": "hfse",
            "version": 1,
            "patch_nums": list(self.patch_nums),
            "base_channels": self.hfse_base_channels,
            "global_dimension": self.hfse_global_dimension,
            "amplitude_dimension": self.hfse_amplitude_dimension,
            "structure_dimension": self.hfse_structure_dimension,
            "dcp_patch_size": self.hfse_dcp_patch_size,
            "dcp_omega": self.hfse_dcp_omega,
            "light_transmission": self.hfse_light_transmission,
            "heavy_transmission": self.hfse_heavy_transmission,
            "synthetic_atmospheric_light": self.hfse_synthetic_atmospheric_light,
            "ranking_margin": self.hfse_ranking_margin,
            "edge_weight": self.hfse_edge_weight,
            "loss_weights": {
                key: float(value)
                for key, value in sorted(self.hfse_loss_config.items())
                if key.startswith("hfse_")
            },
            "density_map_convention": "one_minus_transmission",
            "token_vocabulary_unchanged": True,
        }

    def extension_state_dict(self) -> dict[str, object]:
        return {
            "side_encoder": self.haze_factorized_side_encoder.state_dict(),
            "scale_projections": self.hfse_scale_projections.state_dict(),
        }

    def load_extension_state_dict(self, state: dict[str, object]) -> None:
        self.haze_factorized_side_encoder.load_state_dict(
            state["side_encoder"], strict=True
        )
        self.hfse_scale_projections.load_state_dict(
            state["scale_projections"], strict=True
        )

    def adapter_training_loss(
        self,
        *,
        hazy: torch.Tensor,
        clean: torch.Tensor,
        step: int,
        reconstruction: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del step, reconstruction
        hazy_factors = self.haze_factorized_side_encoder(hazy)
        clean_factors = self.haze_factorized_side_encoder(clean)
        transmission_target, atmospheric_light_target = dark_channel_prior_targets(
            hazy,
            patch_size=self.hfse_dcp_patch_size,
            omega=self.hfse_dcp_omega,
        )
        density_target = 1.0 - F.interpolate(
            transmission_target,
            size=hazy_factors.haze_density_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        classification_logits = torch.cat(
            (hazy_factors.severity_logit, clean_factors.severity_logit), dim=0
        )
        classification_targets = torch.cat(
            (
                torch.ones_like(hazy_factors.severity_logit),
                torch.zeros_like(clean_factors.severity_logit),
            ),
            dim=0,
        )
        classification_loss = F.binary_cross_entropy_with_logits(
            classification_logits,
            classification_targets,
        )
        transmission_loss = F.smooth_l1_loss(
            hazy_factors.haze_density_map.float(), density_target.float()
        ) + 0.25 * F.smooth_l1_loss(
            clean_factors.haze_density_map.float(),
            torch.zeros_like(clean_factors.haze_density_map).float(),
        )
        atmospheric_light_loss = F.smooth_l1_loss(
            hazy_factors.atmospheric_light.float(), atmospheric_light_target.float()
        )

        light_haze, heavy_haze = synthesize_ranked_haze_pair(
            clean,
            light_transmission=self.hfse_light_transmission,
            heavy_transmission=self.hfse_heavy_transmission,
            atmospheric_light=self.hfse_synthetic_atmospheric_light,
        )
        light_factors = self.haze_factorized_side_encoder(light_haze)
        heavy_factors = self.haze_factorized_side_encoder(heavy_haze)
        ranking_loss = F.relu(
            self.hfse_ranking_margin
            - (heavy_factors.severity_logit - light_factors.severity_logit)
        ).mean()
        invariance_loss = (
            1.0
            - F.cosine_similarity(
                clean_factors.structure_embedding.float(),
                heavy_factors.structure_embedding.float(),
                dim=1,
            )
        ).mean()
        smoothness_loss = 0.5 * (
            edge_aware_smoothness_loss(
                hazy_factors.haze_density_map,
                hazy,
                edge_weight=self.hfse_edge_weight,
            )
            + edge_aware_smoothness_loss(
                clean_factors.haze_density_map,
                clean,
                edge_weight=self.hfse_edge_weight,
            )
        )

        losses = {
            "hfse_classification": classification_loss,
            "hfse_transmission": transmission_loss,
            "hfse_atmospheric_light": atmospheric_light_loss,
            "hfse_ranking": ranking_loss,
            "hfse_invariance": invariance_loss,
            "hfse_smoothness": smoothness_loss,
        }
        total = sum(
            float(self.hfse_loss_config[f"{name}_weight"]) * value
            for name, value in losses.items()
        )
        total = float(self.hfse_loss_config["hfse_total_weight"]) * total
        return total, losses


class CycleVARFrequencyLossUnpairedDehazingGenerator(
    CycleVARUnpairedDehazingGenerator
):
    """Coarse amplitude compensation and middle-band phase consistency."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device=device)
        model_config = config["model"]
        loss_config = config["loss"]
        self.frequency_encoder_channels = int(
            model_config["frequency_encoder_channels"]
        )
        self.frequency_phase_encoder_channels = int(
            model_config["frequency_phase_encoder_channels"]
        )
        self.frequency_low_cutoff = float(loss_config["frequency_low_cutoff"])
        self.frequency_middle_cutoff = float(
            loss_config["frequency_middle_cutoff"]
        )
        self.frequency_amplitude_weight = float(
            loss_config["frequency_amplitude_weight"]
        )
        self.frequency_phase_weight = float(loss_config["frequency_phase_weight"])
        self.frequency_dcp_patch_size = int(loss_config["frequency_dcp_patch_size"])
        self.frequency_dcp_omega = float(loss_config["frequency_dcp_omega"])
        adapter_seed = int(model_config["vae_adapter_initialization_seed"])
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(adapter_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(adapter_seed)
            self.amplitude_residual_encoder = AmplitudeResidualTokenEncoder(
                channels=self.frequency_encoder_channels,
                token_dimension=int(self.var.word_embed.out_features),
                coarse_patch_nums=self.patch_nums[:3],
            )
            self.phase_correction_encoder = PhaseCorrectionTokenEncoder(
                channels=self.frequency_phase_encoder_channels,
                token_dimension=int(self.var.word_embed.out_features),
                middle_patch_nums=self.patch_nums[3:6],
                prefix_tokens=sum(patch_num**2 for patch_num in self.patch_nums[:3]),
            )

    def _token_condition(
        self, source: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        total_tokens = sum(patch_num**2 for patch_num in self.patch_nums)
        token_dimension = int(self.var.word_embed.out_features)
        if not bool(torch.all(labels == self.domain_id("clean")).item()):
            return source.new_zeros(
                (source.shape[0], total_tokens, token_dimension), dtype=torch.float32
            )
        with torch.no_grad():
            amplitude_residual = dcp_rgb_log_amplitude_residual_map(
                source.detach(),
                output_size=self.patch_nums[-1],
                patch_size=self.frequency_dcp_patch_size,
                omega=self.frequency_dcp_omega,
            )
            phase = rgb_phase_unit_map(
                source.detach(), output_size=self.patch_nums[-1]
            )
        return self.amplitude_residual_encoder(
            amplitude_residual, total_tokens=total_tokens
        ) + self.phase_correction_encoder(
            phase, total_tokens=total_tokens
        )

    def forward(
        self,
        source: torch.Tensor,
        target_domain: str | int | torch.Tensor,
        return_aux: bool = False,
    ):
        if source.ndim != 4 or source.shape[1:] != (3, 256, 256):
            raise ValueError(f"Expected Bx3x256x256 source images, got {source.shape}")
        if isinstance(target_domain, str):
            target_domain = self.domain_id(target_domain)
        if isinstance(target_domain, int):
            labels = torch.full(
                (source.shape[0],), target_domain, dtype=torch.long, device=source.device
            )
        else:
            labels = target_domain.to(device=source.device, dtype=torch.long)
        context, source_fhat, source_indices = self._source_context(source)
        logits = self._parallel_var_forward(
            labels, context, token_condition=self._token_condition(source, labels)
        )
        translated, entropy = self._srq_decode(logits, source_fhat)
        if not return_aux:
            return translated
        unique_fraction = torch.unique(source_indices.detach()).numel() / float(
            self.vae.V
        )
        return translated, {
            "srq_entropy": float(entropy.detach()),
            "source_code_unique_fraction": float(unique_fraction),
            "_translated": translated,
        }

    def additional_trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.amplitude_residual_encoder.parameters()) + list(
            self.phase_correction_encoder.parameters()
        )

    def extension_signature(self) -> dict[str, object]:
        return {
            "type": "freq_loss",
            "version": 1,
            "patch_nums": list(self.patch_nums),
            "amplitude_residual_encoder": {
                "input": "fixed_dcp_pseudoclear_minus_hazy_rgb_log_fft_amplitude",
                "channels": self.frequency_encoder_channels,
                "injected_scale_ordinals": [1, 2, 3],
                "zero_initialized_output": True,
            },
            "phase_correction_encoder": {
                "input": "rgb_fft_phase_unit_vectors_16x16",
                "channels": self.frequency_phase_encoder_channels,
                "injected_scale_ordinals": [4, 5, 6],
                "zero_initialized_output": True,
            },
            "amplitude_loss": {
                "source": "fixed_dcp_asm_render_of_hazy_to_clean_output",
                "radial_band": [0.0, self.frequency_low_cutoff],
                "weight": self.frequency_amplitude_weight,
            },
            "phase_loss": {
                "source": "hazy_to_clean_output_vs_source_hazy",
                "radial_band": [
                    self.frequency_low_cutoff,
                    self.frequency_middle_cutoff,
                ],
                "amplitude_confidence": True,
                "weight": self.frequency_phase_weight,
            },
            "active_direction": "hazy_to_clean_only",
        }

    def extension_state_dict(self) -> dict[str, object]:
        return {
            "amplitude_residual_encoder": self.amplitude_residual_encoder.state_dict(),
            "phase_correction_encoder": self.phase_correction_encoder.state_dict(),
        }

    def load_extension_state_dict(self, state: dict[str, object]) -> None:
        self.amplitude_residual_encoder.load_state_dict(
            state["amplitude_residual_encoder"], strict=True
        )
        self.phase_correction_encoder.load_state_dict(
            state["phase_correction_encoder"], strict=True
        )

    def cycle_auxiliary_loss(
        self,
        *,
        source: torch.Tensor,
        source_domain: str,
        first_auxiliary: dict[str, object],
        second_auxiliary: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del second_auxiliary
        if source_domain != "hazy":
            zero = self.var.class_emb.weight.new_zeros(())
            return zero, {}
        translated = first_auxiliary.get("_translated")
        if not isinstance(translated, torch.Tensor):
            raise RuntimeError("Frequency loss requires the hazy-to-clean image")
        transmission, atmosphere = dark_channel_prior_targets(
            source.detach(),
            patch_size=self.frequency_dcp_patch_size,
            omega=self.frequency_dcp_omega,
        )
        clear_unit = image_to_unit_range(translated)
        atmosphere = atmosphere.view(source.shape[0], 3, 1, 1)
        asm_image_unit = clear_unit * transmission + atmosphere * (1.0 - transmission)
        asm_image = asm_image_unit.mul(2.0).sub(1.0)
        amplitude = masked_log_amplitude_loss(
            asm_image,
            source,
            minimum_radius=0.0,
            maximum_radius=self.frequency_low_cutoff,
        )
        phase = masked_phase_cosine_loss(
            translated,
            source,
            minimum_radius=self.frequency_low_cutoff,
            maximum_radius=self.frequency_middle_cutoff,
        )
        total = (
            self.frequency_amplitude_weight * amplitude
            + self.frequency_phase_weight * phase
        )
        return total, {"frequency_amplitude": amplitude, "frequency_phase": phase}


class CycleVARHSDAUnpairedDehazingGenerator(CycleVARUnpairedDehazingGenerator):
    """CycleVAR with a shared-ID detail codebook and residual decoder."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device=device)
        model_config = config["model"]
        training_config = config["training"]
        self.hsda_loss_config = dict(config["loss"])
        self.hsda_dcp_patch_size = int(model_config["hsda_dcp_patch_size"])
        self.hsda_dcp_omega = float(model_config["hsda_dcp_omega"])
        self.hsda_base_channels = int(model_config["hsda_base_channels"])
        self.hsda_domain_dimension = int(model_config["hsda_domain_dimension"])
        self.hsda_gamma_maximum = float(model_config["hsda_gamma_maximum"])
        self._adapter_pretraining_steps = int(
            training_config["adapter_pretraining_steps"]
        )
        self._adapter_identity_pretraining_steps = int(
            training_config["adapter_identity_pretraining_steps"]
        )
        adapter_seed = int(model_config["vae_adapter_initialization_seed"])
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(adapter_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(adapter_seed)
            self.haze_aware_shared_id_detail_adapter = HazeAwareSharedIDDetailAdapter(
                vocabulary_size=int(self.vae.V),
                code_dimension=int(self.vae.Cvae),
                domain_count=2,
                domain_dimension=self.hsda_domain_dimension,
                base_channels=self.hsda_base_channels,
                detail_initialization_standard_deviation=float(
                    model_config["hsda_detail_initialization_standard_deviation"]
                ),
                gamma_maximum=self.hsda_gamma_maximum,
            )

    @property
    def detail_adapter(self) -> HazeAwareSharedIDDetailAdapter:
        return self.haze_aware_shared_id_detail_adapter

    def _detail_fhat_from_probabilities(
        self,
        scale_probabilities: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch = scale_probabilities[0].shape[0]
        detail_fhat = self.detail_adapter.detail_codebook.weight.new_zeros(
            batch,
            self.vae.Cvae,
            output_size[0],
            output_size[1],
        )
        for patch_num, probabilities in zip(self.patch_nums, scale_probabilities):
            embeddings = (
                probabilities.float()
                @ self.detail_adapter.detail_codebook.weight.float()
            )
            detail_map = embeddings.transpose(1, 2).reshape(
                batch, self.vae.Cvae, patch_num, patch_num
            )
            if detail_map.shape[-2:] != output_size:
                detail_map = F.interpolate(
                    detail_map,
                    size=output_size,
                    mode="bicubic",
                    align_corners=False,
                )
            detail_fhat = detail_fhat + detail_map
        return detail_fhat

    def _detail_fhat_from_indices(
        self,
        source_indices: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        probabilities = []
        begin = 0
        vocabulary_size = self.detail_adapter.detail_codebook.num_embeddings
        for patch_num in self.patch_nums:
            length = patch_num * patch_num
            indices = source_indices[:, begin : begin + length]
            probabilities.append(
                F.one_hot(indices, num_classes=vocabulary_size).to(torch.float32)
            )
            begin += length
        return self._detail_fhat_from_probabilities(probabilities, output_size)

    def _visibility_map(self, source: torch.Tensor) -> torch.Tensor:
        transmission, _ = dark_channel_prior_targets(
            source,
            patch_size=self.hsda_dcp_patch_size,
            omega=self.hsda_dcp_omega,
        )
        return transmission

    def _decode_with_detail_adapter(
        self,
        *,
        logits: torch.Tensor,
        source_fhat: torch.Tensor,
        source_indices: torch.Tensor,
        source_image: torch.Tensor,
        target_labels: torch.Tensor,
        residual_scale_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantizer = self.vae.quantize
        batch = logits.shape[0]
        begin = 0
        predicted_fhat = torch.zeros_like(source_fhat, dtype=torch.float32)
        normalized_entropies: list[torch.Tensor] = []
        scale_probabilities: list[torch.Tensor] = []
        scale_count = len(self.patch_nums)
        use_gumbel = self.training and self.use_gumbel

        for scale_index, patch_num in enumerate(self.patch_nums):
            length = patch_num * patch_num
            scale_logits = logits[:, begin : begin + length]
            embeddings, probabilities = softmax_relaxed_quantize(
                scale_logits,
                quantizer.embedding.weight,
                temperature=self.temperature,
                use_gumbel=use_gumbel,
            )
            scale_probabilities.append(probabilities)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            normalized_entropies.append(
                entropy.mean() / torch.log(entropy.new_tensor(probabilities.shape[-1]))
            )
            embedding_map = embeddings.transpose(1, 2).reshape(
                batch, self.vae.Cvae, patch_num, patch_num
            )
            if scale_index != scale_count - 1:
                embedding_map = F.interpolate(
                    embedding_map,
                    size=source_fhat.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
            ratio = scale_index / (scale_count - 1)
            predicted_fhat = (
                predicted_fhat + quantizer.quant_resi[ratio](embedding_map).float()
            )
            begin += length

        base_latent = (
            self.blend_alpha * predicted_fhat
            + (1.0 - self.blend_alpha) * source_fhat.float()
        )
        predicted_detail = self._detail_fhat_from_probabilities(
            scale_probabilities, source_fhat.shape[-2:]
        )
        source_detail = self._detail_fhat_from_indices(
            source_indices, source_fhat.shape[-2:]
        )
        detail_latent = (
            self.blend_alpha * predicted_detail
            + (1.0 - self.blend_alpha) * source_detail
        )
        base_image = self.vae.decoder(self.vae.post_quant_conv(base_latent)).clamp(
            -1.0, 1.0
        )
        residual = self.detail_adapter(
            base_latent=base_latent,
            detail_latent=detail_latent,
            target_labels=target_labels,
            source_image=source_image,
            visibility_map=self._visibility_map(source_image),
        )
        residual_scale = (
            self.detail_adapter.effective_gamma
            if residual_scale_override is None
            else residual.new_tensor(float(residual_scale_override))
        )
        image = (base_image + residual_scale * residual).clamp(-1.0, 1.0)
        return image, torch.stack(normalized_entropies).mean(), residual

    def forward(
        self,
        source: torch.Tensor,
        target_domain: str | int | torch.Tensor,
        return_aux: bool = False,
    ):
        if source.ndim != 4 or source.shape[1:] != (3, 256, 256):
            raise ValueError(f"Expected Bx3x256x256 source images, got {source.shape}")
        if isinstance(target_domain, str):
            target_domain = self.domain_id(target_domain)
        if isinstance(target_domain, int):
            labels = torch.full(
                (source.shape[0],),
                target_domain,
                dtype=torch.long,
                device=source.device,
            )
        else:
            labels = target_domain.to(device=source.device, dtype=torch.long)

        context, source_fhat, source_indices = self._source_context(source)
        logits = self._parallel_var_forward(labels, context)
        translated, entropy, residual = self._decode_with_detail_adapter(
            logits=logits,
            source_fhat=source_fhat,
            source_indices=source_indices,
            source_image=source,
            target_labels=labels,
        )
        if not return_aux:
            return translated
        unique_fraction = torch.unique(source_indices.detach()).numel() / float(
            self.vae.V
        )
        return translated, {
            "srq_entropy": float(entropy.detach()),
            "source_code_unique_fraction": float(unique_fraction),
            "hsda_gamma": float(self.detail_adapter.effective_gamma.detach()),
            "hsda_residual_absolute_mean": float(residual.detach().abs().mean()),
            "hsda_detail_codebook_norm": float(
                self.detail_adapter.detail_codebook.weight.detach().norm(dim=1).mean()
            ),
        }

    def additional_trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.detail_adapter.parameters())

    def extension_signature(self) -> dict[str, object]:
        return {
            "type": "hsda",
            "version": 1,
            "patch_nums": list(self.patch_nums),
            "vocabulary_size": int(self.vae.V),
            "code_dimension": int(self.vae.Cvae),
            "base_channels": self.hsda_base_channels,
            "domain_dimension": self.hsda_domain_dimension,
            "gamma_maximum": self.hsda_gamma_maximum,
            "dcp_patch_size": self.hsda_dcp_patch_size,
            "dcp_omega": self.hsda_dcp_omega,
            "structure_channels": 6,
            "visibility_gate": "transmission_times_structure",
            "loss_weights": {
                key: float(value)
                for key, value in sorted(self.hsda_loss_config.items())
                if key.startswith("hsda_")
            },
            "shared_token_probabilities": True,
            "token_vocabulary_unchanged": True,
        }

    def extension_state_dict(self) -> dict[str, object]:
        return {"detail_adapter": self.detail_adapter.state_dict()}

    def load_extension_state_dict(self, state: dict[str, object]) -> None:
        self.detail_adapter.load_state_dict(state["detail_adapter"], strict=True)

    @property
    def adapter_pretraining_steps(self) -> int:
        return self._adapter_pretraining_steps

    def _clear_detail_loss(
        self,
        clean: torch.Tensor,
        reconstruction: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, source_fhat, source_indices = self._source_context(clean)
        with torch.no_grad():
            base_image = self.vae.decoder(
                self.vae.post_quant_conv(source_fhat.float())
            ).clamp(-1.0, 1.0)
            visibility = self._visibility_map(clean)
        detail_latent = self._detail_fhat_from_indices(
            source_indices, source_fhat.shape[-2:]
        )
        labels = torch.full(
            (clean.shape[0],),
            self.domain_id("clean"),
            device=clean.device,
            dtype=torch.long,
        )
        predicted_residual = self.detail_adapter(
            base_latent=source_fhat.float(),
            detail_latent=detail_latent,
            target_labels=labels,
            source_image=clean,
            visibility_map=visibility,
        )
        target_residual = (clean.float() - base_image.float()).detach()
        refined = (base_image.float() + predicted_residual.float()).clamp(-1.0, 1.0)
        detail_l1 = F.l1_loss(predicted_residual.float(), target_residual)
        detail_sobel = sobel_reconstruction_loss(refined, clean)
        detail_wavelet = wavelet_reconstruction_loss(refined, clean)
        _, detail_lpips = reconstruction.components(refined, clean)
        losses = {
            "hsda_detail_l1": detail_l1,
            "hsda_detail_sobel": detail_sobel,
            "hsda_detail_wavelet": detail_wavelet,
            "hsda_detail_lpips": detail_lpips,
        }
        total = sum(
            float(self.hsda_loss_config[f"{name}_weight"]) * value
            for name, value in losses.items()
        )
        return total, losses

    def adapter_pretraining_loss(
        self,
        clean: torch.Tensor,
        reconstruction: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._clear_detail_loss(clean, reconstruction)

    @property
    def adapter_identity_pretraining_steps(self) -> int:
        return self._adapter_identity_pretraining_steps

    def adapter_identity_pretraining_loss(
        self,
        clean: torch.Tensor,
        reconstruction: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = torch.full(
            (clean.shape[0],),
            self.domain_id("clean"),
            device=clean.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            context, source_fhat, source_indices = self._source_context(clean)
            logits = self._parallel_var_forward(labels, context)
        identity, _, residual = self._decode_with_detail_adapter(
            logits=logits,
            source_fhat=source_fhat,
            source_indices=source_indices,
            source_image=clean,
            target_labels=labels,
            residual_scale_override=1.0,
        )
        identity_l1, identity_lpips = reconstruction.components(identity, clean)
        total = (
            float(self.hsda_loss_config["identity_l1"]) * identity_l1
            + float(self.hsda_loss_config["identity_lpips"]) * identity_lpips
        )
        return total, {
            "hsda_identity_l1": identity_l1,
            "hsda_identity_lpips": identity_lpips,
            "hsda_identity_residual_absolute_mean": residual.float().abs().mean(),
            "hsda_identity_gamma": self.detail_adapter.effective_gamma,
        }

    def finalize_adapter_pretraining(self) -> None:
        with torch.no_grad():
            self.detail_adapter.gamma_parameter.zero_()
        if float(self.detail_adapter.effective_gamma.detach()) != 0.0:
            raise RuntimeError("HSDA gamma must be exactly zero at joint step zero")

    def adapter_training_loss(
        self,
        *,
        hazy: torch.Tensor,
        clean: torch.Tensor,
        step: int,
        reconstruction: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del hazy, step
        total, losses = self._clear_detail_loss(clean, reconstruction)
        joint_weight = float(self.hsda_loss_config["hsda_joint_detail_weight"])
        return joint_weight * total, {
            **losses,
            "hsda_training_stage": total.new_tensor(3.0),
        }


def _target_labels(
    generator: CycleVARHFSEUnpairedDehazingGenerator,
    source: torch.Tensor,
    target_domain: str | int | torch.Tensor,
) -> torch.Tensor:
    if source.ndim != 4 or source.shape[1:] != (3, 256, 256):
        raise ValueError(f"Expected Bx3x256x256 source images, got {source.shape}")
    if isinstance(target_domain, str):
        target_domain = generator.domain_id(target_domain)
    if isinstance(target_domain, int):
        return torch.full(
            (source.shape[0],),
            target_domain,
            dtype=torch.long,
            device=source.device,
        )
    labels = target_domain.to(device=source.device, dtype=torch.long)
    if labels.shape != (source.shape[0],):
        raise ValueError(
            "Target-domain labels must have one value per image: "
            f"actual={tuple(labels.shape)}, batch={source.shape[0]}"
        )
    return labels


def _hfse_auxiliary(
    *,
    generator: CycleVARHFSEUnpairedDehazingGenerator,
    source_indices: torch.Tensor,
    entropy: torch.Tensor,
    factors: HazeFactorizedRepresentation,
) -> dict[str, object]:
    unique_fraction = torch.unique(source_indices.detach()).numel() / float(
        generator.vae.V
    )
    return {
        "srq_entropy": float(entropy.detach()),
        "source_code_unique_fraction": float(unique_fraction),
        "hfse_haze_density_mean": float(factors.haze_density_map.detach().mean()),
        "hfse_severity_score": float(factors.severity_logit.detach().mean()),
        "hfse_atmospheric_light_mean": float(
            factors.atmospheric_light.detach().mean()
        ),
    }


def _unique_parameters(*groups: list[nn.Parameter]) -> list[nn.Parameter]:
    parameters = [parameter for group in groups for parameter in group]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("Combined extension parameter groups overlap")
    return parameters


def _combined_token_condition(
    *,
    generator: CycleVARUnpairedDehazingGenerator,
    source: torch.Tensor,
    labels: torch.Tensor,
    factors: object,
) -> torch.Tensor:
    hfse_condition = CycleVARHFSEUnpairedDehazingGenerator._token_condition(
        generator, factors
    )
    frequency_condition = (
        CycleVARFrequencyLossUnpairedDehazingGenerator._token_condition(
            generator, source, labels
        )
    )
    if hfse_condition.shape != frequency_condition.shape:
        raise ValueError(
            "HFSE and frequency token conditions must share one VAR layout: "
            f"hfse={tuple(hfse_condition.shape)}, "
            f"frequency={tuple(frequency_condition.shape)}"
        )
    return hfse_condition + frequency_condition


def _hsda_shared_id_latents(
    *,
    generator: CycleVARHSDAUnpairedDehazingGenerator,
    logits: torch.Tensor,
    source_fhat: torch.Tensor,
    source_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantizer = generator.vae.quantize
    batch = logits.shape[0]
    begin = 0
    predicted_fhat = torch.zeros_like(source_fhat, dtype=torch.float32)
    normalized_entropies: list[torch.Tensor] = []
    scale_probabilities: list[torch.Tensor] = []
    scale_count = len(generator.patch_nums)
    use_gumbel = generator.training and generator.use_gumbel

    for scale_index, patch_num in enumerate(generator.patch_nums):
        length = patch_num * patch_num
        scale_logits = logits[:, begin : begin + length]
        embeddings, probabilities = softmax_relaxed_quantize(
            scale_logits,
            quantizer.embedding.weight,
            temperature=generator.temperature,
            use_gumbel=use_gumbel,
        )
        scale_probabilities.append(probabilities)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        normalized_entropies.append(
            entropy.mean() / torch.log(entropy.new_tensor(probabilities.shape[-1]))
        )
        embedding_map = embeddings.transpose(1, 2).reshape(
            batch, generator.vae.Cvae, patch_num, patch_num
        )
        if scale_index != scale_count - 1:
            embedding_map = F.interpolate(
                embedding_map,
                size=source_fhat.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
        ratio = scale_index / (scale_count - 1)
        predicted_fhat = (
            predicted_fhat + quantizer.quant_resi[ratio](embedding_map).float()
        )
        begin += length

    base_latent = (
        generator.blend_alpha * predicted_fhat
        + (1.0 - generator.blend_alpha) * source_fhat.float()
    )
    predicted_detail = generator._detail_fhat_from_probabilities(
        scale_probabilities, source_fhat.shape[-2:]
    )
    source_detail = generator._detail_fhat_from_indices(
        source_indices, source_fhat.shape[-2:]
    )
    detail_latent = (
        generator.blend_alpha * predicted_detail
        + (1.0 - generator.blend_alpha) * source_detail
    )
    return base_latent, detail_latent, torch.stack(normalized_entropies).mean()


def _decode_cra_hsda(
    *,
    generator: CycleVARHSDAUnpairedDehazingGenerator,
    base_latent: torch.Tensor,
    detail_latent: torch.Tensor,
    source: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoder_input = generator.vae.post_quant_conv(base_latent)
    decoder_residual = generator.decoder_adapter(decoder_input, labels)
    base_image = generator.vae.decoder(decoder_input + decoder_residual).clamp(
        -1.0, 1.0
    )
    detail_residual = generator.detail_adapter(
        base_latent=base_latent,
        detail_latent=detail_latent,
        target_labels=labels,
        source_image=source,
        visibility_map=generator._visibility_map(source),
    )
    translated = (
        base_image + generator.detail_adapter.effective_gamma * detail_residual
    ).clamp(-1.0, 1.0)
    return translated, decoder_residual, detail_residual


def _hsda_auxiliary(
    *,
    generator: CycleVARHSDAUnpairedDehazingGenerator,
    residual: torch.Tensor,
) -> dict[str, object]:
    return {
        "hsda_gamma": float(generator.detail_adapter.effective_gamma.detach()),
        "hsda_residual_absolute_mean": float(residual.detach().float().abs().mean()),
        "hsda_detail_codebook_norm": float(
            generator.detail_adapter.detail_codebook.weight.detach()
            .float()
            .norm(dim=1)
            .mean()
        ),
    }


def _hfse_hsda_adapter_training_loss(
    *,
    generator: CycleVARHSDAUnpairedDehazingGenerator,
    hazy: torch.Tensor,
    clean: torch.Tensor,
    step: int,
    reconstruction: nn.Module,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hfse_total, hfse_losses = (
        CycleVARHFSEUnpairedDehazingGenerator.adapter_training_loss(
            generator,
            hazy=hazy,
            clean=clean,
            step=step,
            reconstruction=reconstruction,
        )
    )
    hsda_total, hsda_losses = (
        CycleVARHSDAUnpairedDehazingGenerator.adapter_training_loss(
            generator,
            hazy=hazy,
            clean=clean,
            step=step,
            reconstruction=reconstruction,
        )
    )
    return hfse_total + hsda_total, {
        **hfse_losses,
        **hsda_losses,
        "hfse_weighted_total": hfse_total,
        "hsda_weighted_total": hsda_total,
    }


class CycleVARHFSECRAFrequencyLossHSDAUnpairedDehazingGenerator(
    CycleVARCRAUnpairedDehazingGenerator,
    CycleVARFrequencyLossUnpairedDehazingGenerator,
    CycleVARHFSEUnpairedDehazingGenerator,
    CycleVARHSDAUnpairedDehazingGenerator,
):
    """HFSE/frequency conditioning and CRA followed by shared-ID detail decoding."""

    def forward(
        self,
        source: torch.Tensor,
        target_domain: str | int | torch.Tensor,
        return_aux: bool = False,
    ):
        labels = _target_labels(self, source, target_domain)
        context, source_fhat, source_indices = self._source_context(source)
        factors = self.haze_factorized_side_encoder(source)
        logits = self._parallel_var_forward(
            labels,
            context,
            token_condition=_combined_token_condition(
                generator=self,
                source=source,
                labels=labels,
                factors=factors,
            ),
        )
        base_latent, detail_latent, entropy = _hsda_shared_id_latents(
            generator=self,
            logits=logits,
            source_fhat=source_fhat,
            source_indices=source_indices,
        )
        translated, decoder_residual, detail_residual = _decode_cra_hsda(
            generator=self,
            base_latent=base_latent,
            detail_latent=detail_latent,
            source=source,
            labels=labels,
        )
        if not return_aux:
            return translated
        auxiliary = _hfse_auxiliary(
            generator=self,
            source_indices=source_indices,
            entropy=entropy,
            factors=factors,
        )
        auxiliary.update(
            _hsda_auxiliary(generator=self, residual=detail_residual)
        )
        auxiliary.update(
            {
                "cra_decoder_residual_absolute_mean": float(
                    decoder_residual.detach().float().abs().mean()
                ),
                "_translated": translated,
            }
        )
        return translated, auxiliary

    def additional_trainable_parameters(self) -> list[nn.Parameter]:
        return _unique_parameters(
            CycleVARHFSEUnpairedDehazingGenerator.additional_trainable_parameters(
                self
            ),
            CycleVARCRAUnpairedDehazingGenerator.additional_trainable_parameters(
                self
            ),
            CycleVARFrequencyLossUnpairedDehazingGenerator.additional_trainable_parameters(
                self
            ),
            CycleVARHSDAUnpairedDehazingGenerator.additional_trainable_parameters(
                self
            ),
        )

    @property
    def adapter_pretraining_steps(self) -> int:
        return CycleVARHSDAUnpairedDehazingGenerator.adapter_pretraining_steps.fget(
            self
        )

    def adapter_pretraining_loss(self, *args, **kwargs):
        return CycleVARHSDAUnpairedDehazingGenerator.adapter_pretraining_loss(
            self, *args, **kwargs
        )

    @property
    def adapter_identity_pretraining_steps(self) -> int:
        return CycleVARHSDAUnpairedDehazingGenerator.adapter_identity_pretraining_steps.fget(
            self
        )

    def adapter_identity_pretraining_loss(self, *args, **kwargs):
        return CycleVARHSDAUnpairedDehazingGenerator.adapter_identity_pretraining_loss(
            self, *args, **kwargs
        )

    def finalize_adapter_pretraining(self) -> None:
        CycleVARHSDAUnpairedDehazingGenerator.finalize_adapter_pretraining(self)

    def adapter_training_loss(self, **kwargs):
        return _hfse_hsda_adapter_training_loss(generator=self, **kwargs)

    def extension_signature(self) -> dict[str, object]:
        return {
            "type": "hfse_cra_freq_loss_hsda",
            "version": 1,
            "parent_baseline": "v3_CRA_freq_loss",
            "initialization": "fresh_step_zero_from_var_d16",
            "composition": (
                "hfse_plus_frequency_condition_then_single_var_then_cra_"
                "decoder_then_hsda_shared_id_residual"
            ),
            "components": {
                "hfse": CycleVARHFSEUnpairedDehazingGenerator.extension_signature(self),
                "cra": CycleVARCRAUnpairedDehazingGenerator.extension_signature(self),
                "freq_loss": CycleVARFrequencyLossUnpairedDehazingGenerator.extension_signature(
                    self
                ),
                "hsda": CycleVARHSDAUnpairedDehazingGenerator.extension_signature(self),
            },
            "single_var_forward": True,
            "single_srq_probability_draw": True,
            "token_vocabulary_unchanged": True,
        }

    def extension_state_dict(self) -> dict[str, object]:
        return {
            "hfse": CycleVARHFSEUnpairedDehazingGenerator.extension_state_dict(
                self
            ),
            "cra": CycleVARCRAUnpairedDehazingGenerator.extension_state_dict(self),
            "freq_loss": CycleVARFrequencyLossUnpairedDehazingGenerator.extension_state_dict(
                self
            ),
            "hsda": CycleVARHSDAUnpairedDehazingGenerator.extension_state_dict(self),
        }

    def load_extension_state_dict(self, state: dict[str, object]) -> None:
        names = ("hfse", "cra", "freq_loss", "hsda")
        if set(state) != set(names):
            raise ValueError(
                "Combined HFSE/HSDA extension state mismatch: "
                f"observed={sorted(state)}, expected={sorted(names)}"
            )
        for name in names:
            if not isinstance(state[name], dict):
                raise TypeError(
                    "Every combined extension state component must be a mapping"
                )
        CycleVARHFSEUnpairedDehazingGenerator.load_extension_state_dict(
            self, state["hfse"]
        )
        CycleVARCRAUnpairedDehazingGenerator.load_extension_state_dict(
            self, state["cra"]
        )
        CycleVARFrequencyLossUnpairedDehazingGenerator.load_extension_state_dict(
            self, state["freq_loss"]
        )
        CycleVARHSDAUnpairedDehazingGenerator.load_extension_state_dict(
            self, state["hsda"]
        )

    def cycle_auxiliary_loss(self, **kwargs):
        return CycleVARFrequencyLossUnpairedDehazingGenerator.cycle_auxiliary_loss(
            self, **kwargs
        )


class CycleVARHFSECRAFrequencyLossHSDADisentangledVAEUnpairedDehazingGenerator(
    CycleVARHFSECRAFrequencyLossHSDAUnpairedDehazingGenerator
):
    """DehazeVAR generator integrating VAR-d16, PRT, CRA, SRQ, and frequency conditioning."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device=device)
        model_config = config["model"]
        adapter_seed = int(model_config["vae_adapter_initialization_seed"]) + 505
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(adapter_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(adapter_seed)
            self.disentangled_vae_branches = DisentangledVAEDecoderBranches(
                self.vae.decoder,
                int(model_config["disen_vae_adapter_channels"]),
            )
        checkpoint = model_config.get("disen_vae_pretrained_checkpoint")
        if not checkpoint:
            raise ValueError(
                "DehazeVAR requires model.disen_vae_pretrained_checkpoint"
            )
        checkpoint_path = resolve_project_path(checkpoint)
        checkpoint_sha256 = model_config.get("disen_vae_pretrained_checkpoint_sha256")
        expected_pretraining_step = int(
            model_config.get(
                "disen_vae_pretraining_step", DISEN_VAE_PRETRAINING_STEPS
            )
        )
        allow_smoke = bool(
            model_config.get("disen_vae_allow_smoke", False)
            or model_config.get("allow_non_production_disen_vae", False)
        )
        if expected_pretraining_step != DISEN_VAE_PRETRAINING_STEPS and not allow_smoke:
            raise ValueError(
                "A non-10k PRT checkpoint is allowed only when --allow-non-production-prt is set"
            )
        payload = load_disen_vae_checkpoint(
            checkpoint_path,
            expected_sha256=str(checkpoint_sha256) if checkpoint_sha256 else None,
            expected_step=expected_pretraining_step,
        )
        expected_signature_sha256 = model_config.get(
            "disen_vae_pretraining_signature_sha256"
        )
        if (
            expected_signature_sha256
            and payload["pretraining_signature_sha256"] != expected_signature_sha256
        ):
            raise ValueError(
                "PRT pretraining signature differs from the configured signature SHA256"
            )
        official_vae_sha256 = sha256_file(
            resolve_project_path(model_config["vae_checkpoint"])
        )
        if (
            payload["pretraining_signature"].get("base_vae_checkpoint_sha256")
            != official_vae_sha256
        ):
            raise ValueError(
                "PRT checkpoint was not initialized from the configured official VAE"
            )
        self.disentangled_vae_branches.load_branch_state_dict(
            payload["decoder_branch_state_dict"]
        )
        self.disentangled_vae_branches.eval().requires_grad_(False)
        self.disen_vae_checkpoint = str(checkpoint)
        self.disen_vae_checkpoint_sha256 = str(
            checkpoint_sha256 or sha256_file(checkpoint_path)
        )
        self.disen_vae_pretraining_signature = dict(payload["pretraining_signature"])
        self.disen_vae_pretraining_signature_sha256 = str(
            payload["pretraining_signature_sha256"]
        )
        self.disen_vae_pretraining_step = expected_pretraining_step
        self.disen_vae_smoke_only = allow_smoke

    def train(self, mode: bool = True):
        super().train(mode)
        self.disentangled_vae_branches.eval()
        return self

    def forward(
        self,
        source: torch.Tensor,
        target_domain: str | int | torch.Tensor,
        return_aux: bool = False,
    ):
        labels = _target_labels(self, source, target_domain)
        context, source_fhat, source_indices = self._source_context(source)
        factors = self.haze_factorized_side_encoder(source)
        logits = self._parallel_var_forward(
            labels,
            context,
            token_condition=_combined_token_condition(
                generator=self,
                source=source,
                labels=labels,
                factors=factors,
            ),
        )
        base_latent, detail_latent, entropy = _hsda_shared_id_latents(
            generator=self,
            logits=logits,
            source_fhat=source_fhat,
            source_indices=source_indices,
        )
        decoder_input = self.vae.post_quant_conv(base_latent)
        decoder_residual = self.decoder_adapter(decoder_input, labels)
        base_image, branch_residual = self.disentangled_vae_branches.decode_labels(
            decoder_input + decoder_residual, labels
        )
        detail_residual = self.detail_adapter(
            base_latent=base_latent,
            detail_latent=detail_latent,
            target_labels=labels,
            source_image=source,
            visibility_map=self._visibility_map(source),
        )
        translated = (
            base_image + self.detail_adapter.effective_gamma * detail_residual
        ).clamp(-1.0, 1.0)
        if not return_aux:
            return translated
        auxiliary = _hfse_auxiliary(
            generator=self,
            source_indices=source_indices,
            entropy=entropy,
            factors=factors,
        )
        auxiliary.update(_hsda_auxiliary(generator=self, residual=detail_residual))
        auxiliary.update(
            {
                "cra_decoder_residual_absolute_mean": float(
                    decoder_residual.detach().float().abs().mean()
                ),
                "disen_vae_branch_residual_absolute_mean": float(
                    branch_residual.detach().float().abs().mean()
                ),
                "_translated": translated,
            }
        )
        return translated, auxiliary

    def additional_trainable_parameters(self) -> list[nn.Parameter]:
        return super().additional_trainable_parameters()

    def extension_signature(self) -> dict[str, object]:
        return {
            "type": "hfse_cra_freq_loss_hsda_disen_vae",
            "version": 1,
            "parent_baseline": "v3_CRA_freq_loss_HSDA",
            "parent": super().extension_signature(),
            "disen_vae": {
                "branches": self.disentangled_vae_branches.signature(),
                "checkpoint": self.disen_vae_checkpoint,
                "checkpoint_sha256": self.disen_vae_checkpoint_sha256,
                "pretraining_signature_sha256": (
                    self.disen_vae_pretraining_signature_sha256
                ),
                "pretraining_step": self.disen_vae_pretraining_step,
                "smoke_only": self.disen_vae_smoke_only,
                "frozen_during_cyclevar": True,
            },
        }

    def extension_state_dict(self) -> dict[str, object]:
        return {
            "parent": super().extension_state_dict(),
            "disen_vae": self.disentangled_vae_branches.branch_state_dict(),
        }

    def load_extension_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"parent", "disen_vae"}:
            raise ValueError(f"PRT extension state mismatch: {sorted(state)}")
        if not isinstance(state["parent"], dict) or not isinstance(
            state["disen_vae"], Mapping
        ):
            raise TypeError("PRT extension components must be mappings")
        super().load_extension_state_dict(state["parent"])
        self.disentangled_vae_branches.load_branch_state_dict(state["disen_vae"])

    def adapter_training_loss(self, **kwargs):
        return _hfse_hsda_adapter_training_loss(generator=self, **kwargs)


DehazeVARGenerator = (
    CycleVARHFSECRAFrequencyLossHSDADisentangledVAEUnpairedDehazingGenerator
)


def build_unpaired_dehazing_generator(
    config: Mapping[str, Any],
    device: torch.device,
) -> CycleVARUnpairedDehazingGenerator:
    adapter_type = str(
        config.get("model", {}).get(
            "vae_adapter_type", "hfse_cra_freq_loss_hsda_disen_vae"
        )
    )
    if adapter_type in {
        "dehazevar",
        "hfse_cra_freq_loss_hsda_disen_vae",
        "cyclevar_hfse_cra_freq_loss_hsda_disen_vae_unpaired_dehazing",
        "v4_disen_vae",
        "disen_vae",
    }:
        return DehazeVARGenerator(dict(config), device=device)
    if adapter_type == "hfse_cra_freq_loss_hsda":
        return CycleVARHFSECRAFrequencyLossHSDAUnpairedDehazingGenerator(
            dict(config), device=device
        )
    raise ValueError(
        f"Unsupported model.vae_adapter_type={adapter_type!r}. "
        "Expected 'dehazevar' or 'hfse_cra_freq_loss_hsda_disen_vae'."
    )


build_dehazevar_generator = build_unpaired_dehazing_generator

__all__ = [
    "CycleVARCRAUnpairedDehazingGenerator",
    "CycleVARFrequencyLossUnpairedDehazingGenerator",
    "CycleVARGenerator",
    "CycleVARHFSECRAFrequencyLossHSDADisentangledVAEUnpairedDehazingGenerator",
    "CycleVARHFSECRAFrequencyLossHSDAUnpairedDehazingGenerator",
    "CycleVARHFSEUnpairedDehazingGenerator",
    "CycleVARHSDAUnpairedDehazingGenerator",
    "CycleVARUnpairedDehazingGenerator",
    "DOMAIN_TO_ID",
    "DehazeVARGenerator",
    "VAR_ROOT",
    "build_dehazevar_generator",
    "build_unpaired_dehazing_generator",
]
