from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import shutil
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.unpaired_dehazing import (
    TrainTransform,
    image_files_recursive,
    inference_transform,
)
from models.prt_tokenizer import (
    DINOAndDomainPretrainingDiscriminator,
    DISEN_VAE_CHECKPOINT_FORMAT,
    DISEN_VAE_CHECKPOINT_VERSION,
    DISEN_VAE_PRETRAINING_STEPS,
    DisentangledVAEDecoderBranches,
    DisentangledVAEPretrainingObjective,
    SyntheticHazePipeline,
)
from utils.io_utils import (
    PROJECT_ROOT,
    append_jsonl,
    atomic_write_json,
    load_config,
    resolve_project_path,
    seed_everything,
    seed_worker,
)
from utils.losses import canonical_fingerprint, sha256_file, ssim_loss


VAR_ROOT = PROJECT_ROOT / "third_party" / "VAR"
if str(VAR_ROOT) not in sys.path:
    sys.path.insert(0, str(VAR_ROOT))

from third_party.VAR.models.vqvae import VQVAE  # noqa: E402


class _CleanImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform: Any) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = self.transform(image)
        return {"image": tensor, "path": str(path)}


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _amp_context(precision: str):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp32":
        return nullcontext()
    raise ValueError(f"Unsupported precision: {precision!r}")


def _index_fingerprint(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _split_domain(
    *,
    domain: str,
    root: Path,
    paths: list[Path],
    heldout_count: int,
    seed: int,
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    if not 0 < heldout_count < len(paths):
        raise ValueError(
            f"heldout_count for {domain} must be in [1, {len(paths) - 1}], "
            f"got {heldout_count}"
        )

    def ranking_key(path: Path) -> tuple[bytes, str]:
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(
            f"{seed}\0{domain}\0{relative}".encode("utf-8")
        ).digest()
        return digest, relative

    ranked = sorted(paths, key=ranking_key)
    heldout = ranked[:heldout_count]
    training = ranked[heldout_count:]
    heldout_records = []
    for path in heldout:
        relative = path.relative_to(root).as_posix()
        heldout_records.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "domain": domain,
        "root": str(root),
        "selection": "sha256(seed + NUL + domain + NUL + relative_path)",
        "seed": seed,
        "source_count": len(paths),
        "training_count": len(training),
        "heldout_count": len(heldout),
        "training_index_sha256": _index_fingerprint(root, training),
        "heldout": heldout_records,
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    return training, heldout, manifest


def _build_vae(config: dict[str, Any], device: torch.device) -> tuple[VQVAE, Path]:
    model_config = config["model"]
    patch_nums = tuple(int(value) for value in model_config["patch_nums"])
    vae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=patch_nums,
    )
    checkpoint_path = resolve_project_path(model_config["vae_checkpoint"]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing official VAR VAE checkpoint: {checkpoint_path}")
    vae.load_state_dict(_load_state_dict(checkpoint_path), strict=True)
    vae.to(device)
    return vae, checkpoint_path


@torch.no_grad()
def _hard_latent(
    vae: VQVAE,
    images: torch.Tensor,
    patch_nums: tuple[int, ...],
) -> torch.Tensor:
    with torch.autocast(device_type=images.device.type, enabled=False):
        continuous = vae.quant_conv(vae.encoder(images.float())).float()
        hard_scales = vae.quantize.f_to_idxBl_or_fhat(
            continuous,
            to_fhat=True,
            v_patch_nums=patch_nums,
        )
        hard = hard_scales[-1].float()
    return hard


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
    target = target.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
    mse = F.mse_loss(prediction, target)
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def _ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        1.0
        - ssim_loss(
            prediction.float(),
            target.float(),
            window_size=11,
            sigma=1.5,
            data_range=2.0,
        )
    )


@torch.no_grad()
def _evaluate_heldout(
    *,
    vae: torch.nn.Module,
    branches: DisentangledVAEDecoderBranches,
    pipeline: SyntheticHazePipeline,
    paths: list[Path],
    run_directory: Path,
    step: int,
    inference_size: int,
    patch_nums: tuple[int, ...],
    workers: int,
    seed: int,
) -> None:
    branches.eval()
    dataset = _CleanImageDataset(paths, inference_transform(inference_size))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    values: dict[str, dict[str, list[float]]] = {
        "clean": {"psnr": [], "ssim": []},
        "synthetic_hazy": {"psnr": [], "ssim": []},
    }
    device = next(vae.parameters()).device
    fork_devices = [device.index or 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed + 300_000)
        torch.cuda.manual_seed(seed + 300_000)
        for batch in tqdm(loader, desc=f"prt-heldout:{step}"):
            clean = batch["image"].to(device, non_blocking=True)
            synthetic_hazy, _ = pipeline(clean)
            hard = _hard_latent(vae, clean, patch_nums)
            decoder_input = vae.post_quant_conv(hard.float())
            clean_prediction, _ = branches.decode(decoder_input, "clean")
            hazy_prediction, _ = branches.decode(decoder_input, "hazy")
            for domain, prediction, target in (
                ("clean", clean_prediction, clean),
                ("synthetic_hazy", hazy_prediction, synthetic_hazy),
            ):
                values[domain]["psnr"].append(_psnr(prediction, target))
                values[domain]["ssim"].append(_ssim(prediction, target))
    summary = {
        domain: {
            "count": len(metrics["psnr"]),
            "psnr": sum(metrics["psnr"]) / len(metrics["psnr"]),
            "ssim": sum(metrics["ssim"]) / len(metrics["ssim"]),
        }
        for domain, metrics in values.items()
    }
    atomic_write_json(
        run_directory / f"heldout_step_{step:07d}.json",
        {"step": step, "domains": summary},
    )
    branches.train()


def _save_checkpoint(
    *,
    run_directory: Path,
    step: int,
    branches: DisentangledVAEDecoderBranches,
    discriminator: DINOAndDomainPretrainingDiscriminator,
    optimizer_branches: Adam,
    optimizer_discriminator: Adam,
    signature: dict[str, Any],
    config: dict[str, Any],
    minimum_free_gib: float,
) -> Path:
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(checkpoint_directory).free < int(minimum_free_gib * 2**30):
        raise OSError(
            f"Refusing PRT checkpoint with less than {minimum_free_gib} GiB free"
        )
    output = checkpoint_directory / f"step_{step:07d}.pt"
    temporary = checkpoint_directory / f".step_{step:07d}.pt.tmp"
    payload = {
        "format": DISEN_VAE_CHECKPOINT_FORMAT,
        "format_version": DISEN_VAE_CHECKPOINT_VERSION,
        "step": step,
        "decoder_branch_state_dict": branches.branch_state_dict(),
        "discriminator_state_dict": discriminator.trainable_state_dict(),
        "pretraining_signature": signature,
        "pretraining_signature_sha256": canonical_fingerprint(signature),
        "optimizer_branches": optimizer_branches.state_dict(),
        "optimizer_discriminator": optimizer_discriminator.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": __import__("numpy").random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "config": {key: value for key, value in config.items() if key != "_config_path"},
    }
    torch.save(payload, temporary)
    os.replace(temporary, output)
    atomic_write_json(
        output.with_suffix(".ready.json"),
        {
            "status": "ready",
            "step": step,
            "checkpoint": str(output),
            "checkpoint_sha256": sha256_file(output),
            "pretraining_signature_sha256": canonical_fingerprint(signature),
        },
    )
    return output


def run_prt_training(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    data_config = config["data"]
    model_config = config["model"]
    pretrain_config = config["disen_vae_pretraining"]
    run_directory = Path(args.run_directory).resolve()
    if run_directory.exists() and args.resume is None:
        raise FileExistsError(f"Refusing to reuse pretraining run: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PRT disentangled-VAE pretraining requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(pretrain_config["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(pretrain_config["allow_tf32"])

    clean_root = resolve_project_path(data_config["clean_image_directory"]).resolve()
    clean_paths = image_files_recursive(clean_root)
    expected_clean = data_config.get("expected_clean_image_count")
    if expected_clean is not None and len(clean_paths) != int(expected_clean):
        raise RuntimeError(
            f"Clean-domain count changed: {len(clean_paths)} != {int(expected_clean)}"
        )
    heldout_count = int(
        args.heldout_clean_count
        if args.heldout_clean_count is not None
        else pretrain_config["heldout_clean_count"]
    )
    training_paths, heldout_paths, split_manifest = _split_domain(
        domain="clean",
        root=clean_root,
        paths=clean_paths,
        heldout_count=heldout_count,
        seed=seed,
    )
    atomic_write_json(run_directory / "clean_split_manifest.json", split_manifest)

    vae, base_checkpoint = _build_vae(config, device)
    vae.eval().requires_grad_(False)
    adapter_seed = int(model_config["vae_adapter_initialization_seed"]) + 505
    fork_devices = [device.index or 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(adapter_seed)
        torch.cuda.manual_seed(adapter_seed)
        branches = DisentangledVAEDecoderBranches(
            vae.decoder,
            int(model_config["disen_vae_adapter_channels"]),
        ).to(device)
    branch_parameters = branches.branch_parameters()
    discriminator_config = dict(model_config)
    discriminator_config["dino_checkpoint"] = str(
        resolve_project_path(model_config["dino_checkpoint"])
    )
    discriminator = DINOAndDomainPretrainingDiscriminator(
        discriminator_config,
        real_label=float(config["loss"]["real_label_smoothing"]),
    ).to(device)
    pipeline = SyntheticHazePipeline(pretrain_config["synthetic_haze"]).to(device)
    objective = DisentangledVAEPretrainingObjective(pretrain_config).to(device)
    optimizer_branches = Adam(
        branch_parameters,
        lr=float(pretrain_config["learning_rate"]),
        betas=tuple(float(value) for value in pretrain_config["adam_betas"]),
        eps=float(pretrain_config["adam_epsilon"]),
        weight_decay=float(pretrain_config["weight_decay"]),
    )
    discriminator_parameters = discriminator.trainable_parameters()
    optimizer_discriminator = Adam(
        discriminator_parameters,
        lr=float(pretrain_config["discriminator_learning_rate"]),
        betas=tuple(float(value) for value in pretrain_config["adam_betas"]),
        eps=float(pretrain_config["adam_epsilon"]),
        weight_decay=float(pretrain_config["weight_decay"]),
    )
    max_steps = int(
        args.max_steps if args.max_steps is not None else pretrain_config["max_steps"]
    )
    production = max_steps == DISEN_VAE_PRETRAINING_STEPS
    if max_steps <= 0 or (not production and args.max_steps is None):
        raise ValueError("Production PRT pretraining must be 10k")
    precision = str(pretrain_config["mixed_precision"])
    patch_nums = tuple(int(value) for value in model_config["patch_nums"])
    signature = {
        "type": "v4_disen_vae_pseudo_paired_pretraining",
        "version": 2,
        "max_steps": max_steps,
        "production": production,
        "base_vae_checkpoint": str(base_checkpoint),
        "base_vae_checkpoint_sha256": sha256_file(base_checkpoint),
        "frozen": [
            "encoder",
            "quant_conv",
            "quantizer_and_codebook",
            "post_quant_conv",
        ],
        "trainable": [
            "clean_decoder_copy",
            "hazy_decoder_copy",
            "clean_rgb_residual_adapter",
            "hazy_rgb_residual_adapter",
        ],
        "decoder_branches": branches.signature(),
        "synthetic_pipeline": pipeline.signature(),
        "objective": objective.signature(),
        "discriminator": discriminator.signature(),
        "clean_split_fingerprint": split_manifest["fingerprint"],
        "seed": seed,
        "adapter_initialization_seed": adapter_seed,
    }
    atomic_write_json(
        run_directory / "pretraining_signature.json",
        {"signature": signature, "sha256": canonical_fingerprint(signature)},
    )
    atomic_write_json(
        run_directory / "resolved_config_upaired_dehazing.json",
        {key: value for key, value in config.items() if key != "_config_path"},
    )

    start_step = 0
    if args.resume is not None:
        payload = torch.load(
            Path(args.resume).resolve(), map_location="cpu", weights_only=False
        )
        if payload.get("pretraining_signature_sha256") != canonical_fingerprint(
            signature
        ):
            raise ValueError("Refusing PRT resume with a different signature")
        branches.load_branch_state_dict(payload["decoder_branch_state_dict"])
        discriminator.load_trainable_state_dict(payload["discriminator_state_dict"])
        optimizer_branches.load_state_dict(payload["optimizer_branches"])
        optimizer_discriminator.load_state_dict(payload["optimizer_discriminator"])
        random.setstate(payload["python_rng"])
        __import__("numpy").random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        start_step = int(payload["step"])

    dataset = _CleanImageDataset(
        training_paths,
        TrainTransform(
            resize=int(data_config["train_resize"]),
            crop=int(data_config["train_crop"]),
            horizontal_flip=bool(data_config["train_horizontal_flip"]),
            preserve_aspect_ratio=True,
        ),
    )
    workers = int(args.workers if args.workers is not None else data_config["workers"])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = iter(loader)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    process_path = run_directory / "process.json"
    atomic_write_json(
        process_path,
        {"pid": os.getpid(), "alive": True, "status": "training", "step": start_step},
    )
    if start_step == 0 and not getattr(args, "skip_initial_eval", False):
        _evaluate_heldout(
            vae=vae,
            branches=branches,
            pipeline=pipeline,
            paths=heldout_paths,
            run_directory=run_directory,
            step=0,
            inference_size=int(data_config["inference_size"]),
            patch_nums=patch_nums,
            workers=workers,
            seed=seed,
        )

    progress = tqdm(total=max_steps, initial=start_step, desc="prt-pretrain")
    step = start_step
    max_grad_norm = float(pretrain_config["max_grad_norm"])
    while step < max_steps and not stop_requested:
        started = time.time()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        clean = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            synthetic_hazy, synthetic_stats = pipeline(clean)
            hard = _hard_latent(vae, clean, patch_nums)
            decoder_input = vae.post_quant_conv(hard.float())

        discriminator.set_trainable(False)
        optimizer_branches.zero_grad(set_to_none=True)
        with _amp_context(precision):
            clean_prediction, clean_residual = branches.decode(
                decoder_input, "clean"
            )
            hazy_prediction, hazy_residual = branches.decode(decoder_input, "hazy")
            clean_components = objective.reconstruction(clean_prediction, clean)
            hazy_components = objective.reconstruction(
                hazy_prediction, synthetic_hazy
            )
            clean_adversarial, clean_domain = discriminator.generator_loss(
                clean_prediction, "clean"
            )
            hazy_adversarial, hazy_domain = discriminator.generator_loss(
                hazy_prediction, "hazy"
            )
            generator_loss = objective.generator_total(
                clean_components=clean_components,
                hazy_components=hazy_components,
                clean_adversarial=clean_adversarial,
                hazy_adversarial=hazy_adversarial,
                clean_domain=clean_domain,
                hazy_domain=hazy_domain,
            )
        generator_loss.backward()
        branch_grad_norm = clip_grad_norm_(branch_parameters, max_grad_norm)
        if not torch.isfinite(branch_grad_norm):
            raise FloatingPointError("Non-finite PRT decoder-branch gradient")
        optimizer_branches.step()

        discriminator.set_trainable(True)
        optimizer_discriminator.zero_grad(set_to_none=True)
        with _amp_context(precision):
            d_clean_adversarial, d_clean_domain = discriminator.discriminator_loss(
                real=clean,
                fake=clean_prediction.detach(),
                domain="clean",
            )
            d_hazy_adversarial, d_hazy_domain = discriminator.discriminator_loss(
                real=synthetic_hazy,
                fake=hazy_prediction.detach(),
                domain="hazy",
            )
            discriminator_loss = 0.5 * (
                d_clean_adversarial
                + d_hazy_adversarial
                + d_clean_domain
                + d_hazy_domain
            )
        discriminator_loss.backward()
        discriminator_grad_norm = clip_grad_norm_(
            discriminator_parameters, max_grad_norm
        )
        if not torch.isfinite(discriminator_grad_norm):
            raise FloatingPointError("Non-finite PRT discriminator gradient")
        optimizer_discriminator.step()

        step += 1
        record = {
            "step": step,
            "time": time.time(),
            "elapsed_seconds": time.time() - started,
            "generator_total": float(generator_loss.detach()),
            "clean_l1": float(clean_components["l1"].detach()),
            "clean_ssim_loss": float(clean_components["ssim"].detach()),
            "clean_wavelet": float(clean_components["wavelet"].detach()),
            "hazy_l1": float(hazy_components["l1"].detach()),
            "hazy_ssim_loss": float(hazy_components["ssim"].detach()),
            "hazy_wavelet": float(hazy_components["wavelet"].detach()),
            "clean_dino_generator": float(clean_adversarial.detach()),
            "hazy_dino_generator": float(hazy_adversarial.detach()),
            "clean_domain_generator": float(clean_domain.detach()),
            "hazy_domain_generator": float(hazy_domain.detach()),
            "discriminator_total": float(discriminator_loss.detach()),
            "decoder_branch_grad_norm": float(branch_grad_norm),
            "discriminator_grad_norm": float(discriminator_grad_norm),
            "clean_residual_absolute_mean": float(
                clean_residual.detach().abs().mean()
            ),
            "hazy_residual_absolute_mean": float(
                hazy_residual.detach().abs().mean()
            ),
            "gpu_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            **{
                f"synthetic_{key}": float(value.detach())
                for key, value in synthetic_stats.items()
            },
        }
        if not all(
            math.isfinite(value)
            for value in record.values()
            if isinstance(value, float)
        ):
            raise FloatingPointError(f"Non-finite PRT metric: {record}")
        append_jsonl(run_directory / "pretraining_metrics.jsonl", record)
        atomic_write_json(run_directory / "latest_metrics.json", record)
        atomic_write_json(
            process_path,
            {"pid": os.getpid(), "alive": True, "status": "training", "step": step},
        )
        progress.update(1)
        progress.set_postfix(
            clean=f"{record['clean_l1']:.3f}",
            hazy=f"{record['hazy_l1']:.3f}",
            d=f"{record['discriminator_total']:.3f}",
        )

    progress.close()
    if stop_requested:
        atomic_write_json(
            process_path,
            {"pid": os.getpid(), "alive": False, "status": "stopped", "step": step},
        )
        raise InterruptedError(f"PRT pretraining stopped at step {step}")
    checkpoint = _save_checkpoint(
        run_directory=run_directory,
        step=step,
        branches=branches,
        discriminator=discriminator,
        optimizer_branches=optimizer_branches,
        optimizer_discriminator=optimizer_discriminator,
        signature=signature,
        config=config,
        minimum_free_gib=float(pretrain_config["minimum_checkpoint_free_gib"]),
    )
    _evaluate_heldout(
        vae=vae,
        branches=branches,
        pipeline=pipeline,
        paths=heldout_paths,
        run_directory=run_directory,
        step=step,
        inference_size=int(data_config["inference_size"]),
        patch_nums=patch_nums,
        workers=workers,
        seed=seed,
    )
    atomic_write_json(
        process_path,
        {
            "pid": os.getpid(),
            "alive": False,
            "status": "complete",
            "step": step,
            "checkpoint": str(checkpoint),
        },
    )
