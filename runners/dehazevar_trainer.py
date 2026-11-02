from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.unpaired_dehazing import (
    UnpairedDehazingDataset,
    denormalize,
    inference_transform,
)
from models.dehazevar_generator import (
    CycleVARUnpairedDehazingGenerator,
    build_unpaired_dehazing_generator,
)
from models.discriminators import build_unpaired_dehazing_discriminator
from utils.io_utils import (
    PROJECT_ROOT,
    append_jsonl,
    atomic_write_json,
    finite_scalar_dict,
    linear_warmup_multiplier,
    load_config,
    resolve_project_path,
    seed_everything,
    seed_worker,
)
from utils.losses import (
    ReconstructionLoss,
    build_image_objective,
    objective_signature,
    resolve_adversarial_weights,
)


def amp_context(precision: str):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        raise ValueError("FP16 training is unsupported without GradScaler; use BF16")
    if precision == "fp32":
        return nullcontext()
    raise ValueError(f"Unsupported mixed_precision={precision!r}")


def parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
    return float(squared.sqrt())


def clip_finite_gradients(
    parameters: list[torch.nn.Parameter],
    max_norm: float,
    *,
    run_dir: Path,
    step: int,
    component: str,
) -> float:
    """Clip gradients and abort before optimizer state can be contaminated."""
    total_norm = clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(total_norm):
        crash = {
            "step": step,
            "reason": "non_finite_gradient",
            "component": component,
            "gradient_norm": float(total_norm),
        }
        atomic_write_json(run_dir / "crash.json", crash)
        raise FloatingPointError(json.dumps(crash, sort_keys=True))
    return float(total_norm)


def save_checkpoint(
    run_dir: Path,
    step: int,
    generator: CycleVARUnpairedDehazingGenerator,
    discriminator: nn.Module,
    optimizer_g: Optimizer,
    optimizer_d: Optimizer,
    scheduler_g: LambdaLR,
    scheduler_d: LambdaLR,
    config: dict[str, Any],
    training_objective_signature: dict[str, object],
    max_to_keep: int,
    minimum_free_gib: float,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(checkpoint_dir).free
    required_free_bytes = int(minimum_free_gib * 2**30)
    if free_bytes < required_free_bytes:
        raise OSError(
            f"Refusing checkpoint with only {free_bytes / 2**30:.2f} GiB free; "
            f"minimum_checkpoint_free_gib={minimum_free_gib}"
        )
    output = checkpoint_dir / f"step_{step:07d}.pt"
    temporary = checkpoint_dir / f".step_{step:07d}.pt.tmp"
    payload = {
        "format_version": 3,
        "task": "upaired_dehazing",
        "step": step,
        "generator_var": generator.var.state_dict(),
        "generator_extension_signature": generator.extension_signature(),
        "generator_extension": generator.extension_state_dict(),
        "training_objective_signature": training_objective_signature,
        "discriminator_heads": discriminator.heads.state_dict(),
        "discriminator_signature": discriminator.discriminator_signature(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict(),
        "scheduler_d": scheduler_d.state_dict(),
        "python_rng": __import__("random").getstate(),
        "numpy_rng": __import__("numpy").random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "config": {key: value for key, value in config.items() if key != "_config_path"},
    }
    torch.save(payload, temporary)
    os.replace(temporary, output)
    latest_tmp = checkpoint_dir / ".latest.pt.tmp"
    latest = checkpoint_dir / "latest.pt"
    try:
        latest_tmp.unlink()
    except FileNotFoundError:
        pass
    latest_tmp.symlink_to(output.name)
    os.replace(latest_tmp, latest)
    ready_marker = checkpoint_dir / f"step_{step:07d}.ready.json"
    atomic_write_json(
        ready_marker,
        {
            "checkpoint": str(output),
            "step": step,
            "status": "ready",
            "time": time.time(),
        },
    )
    if max_to_keep > 0:
        checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
        for stale in checkpoints[:-max_to_keep]:
            stale.unlink()
            stale.with_suffix(".ready.json").unlink(missing_ok=True)
    return output


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def canonical_json_fingerprint(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def image_index_fingerprint(root: Path, images: list[Path]) -> str:
    """Fingerprint the audited file index without hashing all image payloads."""
    digest = hashlib.sha256()
    for image_path in images:
        relative_path = image_path.relative_to(root).as_posix()
        stat = image_path.stat()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    path: Path,
    generator: CycleVARUnpairedDehazingGenerator,
    discriminator: nn.Module,
    optimizer_g: Optimizer,
    optimizer_d: Optimizer,
    scheduler_g: LambdaLR,
    scheduler_d: LambdaLR,
    training_objective_signature: dict[str, object],
    *,
    strict_signature: bool = True,
) -> int:
    checkpoint_payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_format = checkpoint_payload.get("format_version")
    if checkpoint_format not in {2, 3}:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    checkpoint_discriminator_signature = checkpoint_payload.get(
        "discriminator_signature",
        {"type": "dino_only", "feature_layers": [4, 8]},
    )
    current_discriminator_signature = discriminator.discriminator_signature()
    if strict_signature and (
        checkpoint_discriminator_signature != current_discriminator_signature
    ):
        raise ValueError(
            "Refusing to resume with a different discriminator: "
            f"checkpoint={checkpoint_discriminator_signature}, "
            f"current={current_discriminator_signature}"
        )
    generator.var.load_state_dict(checkpoint_payload["generator_var"], strict=True)
    checkpoint_extension_signature = checkpoint_payload.get(
        "generator_extension_signature", {"type": "none"}
    )
    if checkpoint_format == 2 and checkpoint_extension_signature != {"type": "none"}:
        raise ValueError(
            "Checkpoint format 2 cannot contain a VAE-adapter extension; "
            "use format 3"
        )
    current_extension_signature = generator.extension_signature()
    if strict_signature and (
        _normalize_extension_signature(checkpoint_extension_signature)
        != _normalize_extension_signature(current_extension_signature)
    ):
        raise ValueError(
            "Refusing to resume with a different VAE adapter: "
            f"checkpoint={checkpoint_extension_signature}, "
            f"current={current_extension_signature}"
        )
    generator.load_extension_state_dict(
        checkpoint_payload.get("generator_extension", {})
    )
    checkpoint_objective_signature = checkpoint_payload.get(
        "training_objective_signature",
        None,
    )
    if checkpoint_objective_signature is None:
        checkpoint_loss_config = checkpoint_payload.get("config", {}).get("loss")
        if not isinstance(checkpoint_loss_config, dict):
            checkpoint_loss_config = {
                "gan": 0.5,
                "cycle_l1": 1.0,
                "cycle_lpips": 10.0,
                "identity_l1": 1.0,
                "identity_lpips": 1.0,
            }
        checkpoint_objective_signature = objective_signature(
            "legacy_v2", checkpoint_loss_config
        )
    if strict_signature and (
        checkpoint_objective_signature != training_objective_signature
    ):
        raise ValueError(
            "Refusing to resume with a different training objective: "
            f"checkpoint={checkpoint_objective_signature}, "
            f"current={training_objective_signature}"
        )
    discriminator.heads.load_state_dict(
        checkpoint_payload["discriminator_heads"], strict=True
    )
    optimizer_g.load_state_dict(checkpoint_payload["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint_payload["optimizer_d"])
    scheduler_g.load_state_dict(checkpoint_payload["scheduler_g"])
    scheduler_d.load_state_dict(checkpoint_payload["scheduler_d"])
    __import__("random").setstate(checkpoint_payload["python_rng"])
    __import__("numpy").random.set_state(checkpoint_payload["numpy_rng"])
    torch.set_rng_state(checkpoint_payload["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint_payload["cuda_rng"])
    return int(checkpoint_payload["step"])


@torch.no_grad()
def save_sample_grid(
    run_dir: Path,
    step: int,
    generator: CycleVARUnpairedDehazingGenerator,
    hazy: torch.Tensor,
    clean: torch.Tensor,
    hazy_label: int,
    clean_label: int,
    precision: str,
) -> None:
    was_training = generator.training
    generator.eval()
    device = next(generator.parameters()).device
    rows = []
    for sample_index in range(hazy.shape[0]):
        hazy_sample = hazy[sample_index : sample_index + 1].to(device)
        clean_sample = clean[sample_index : sample_index + 1].to(device)
        with amp_context(precision):
            fake_clean = generator(hazy_sample, clean_label)
            fake_hazy = generator(clean_sample, hazy_label)
        rows.extend([hazy_sample, fake_clean, clean_sample, fake_hazy])
    images = torch.cat(rows, dim=0)
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images), sample_dir / f"step_{step:07d}.png", nrow=4)
    generator.train(was_training)


def load_fixed_samples(
    dataset: UnpairedDehazingDataset,
    *,
    sample_count: int,
    inference_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[str]]]:
    if sample_count <= 0:
        raise ValueError("fixed_sample_count must be positive")
    sample_count = min(
        sample_count, len(dataset.hazy_images), len(dataset.clean_images)
    )
    transform = inference_transform(inference_size)
    hazy_tensors = []
    clean_tensors = []
    for hazy_path, clean_path in zip(
        dataset.hazy_images[:sample_count], dataset.clean_images[:sample_count]
    ):
        with Image.open(hazy_path) as image:
            hazy_tensors.append(
                transform(ImageOps.exif_transpose(image).convert("RGB"))
            )
        with Image.open(clean_path) as image:
            clean_tensors.append(
                transform(ImageOps.exif_transpose(image).convert("RGB"))
            )
    paths = {
        "hazy": [str(path) for path in dataset.hazy_images[:sample_count]],
        "clean": [str(path) for path in dataset.clean_images[:sample_count]],
    }
    return torch.stack(hazy_tensors), torch.stack(clean_tensors), paths


def run_dehazevar_training(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    train_config = config["training"]
    model_config = config["model"]
    loss_config = config["loss"]
    data_config = config["data"]
    if getattr(args, "prt_checkpoint", None) is not None:
        model_config["disen_vae_pretrained_checkpoint"] = str(
            Path(args.prt_checkpoint).resolve()
        )
    if getattr(args, "allow_non_production_prt", False):
        model_config["allow_non_production_disen_vae"] = True
    if "dino_checkpoint" in model_config:
        model_config["dino_checkpoint"] = str(
            resolve_project_path(model_config["dino_checkpoint"])
        )

    run_dir = Path(args.run_dir).resolve()
    if (run_dir / "train_metrics.jsonl").exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to append a fresh run to existing metrics: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("DehazeVAR full-backbone training requires CUDA")
    device = torch.device("cuda", 0)
    precision = str(train_config["mixed_precision"])
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16, but the selected GPU does not support it")

    seed = int(train_config["seed"])
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(train_config["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(train_config["allow_tf32"])
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(device)

    max_steps = int(args.max_steps or train_config["max_steps"])
    checkpoint_every = (
        int(args.checkpoint_every)
        if args.checkpoint_every is not None
        else int(train_config["checkpoint_every"])
    )
    checkpoints_to_keep = int(train_config.get("checkpoints_to_keep", 3))
    minimum_checkpoint_free_gib = float(
        train_config.get("minimum_checkpoint_free_gib", 8.0)
    )
    sample_every = (
        int(args.sample_every)
        if args.sample_every is not None
        else int(train_config["sample_every"])
    )
    workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(data_config["workers"])
    )

    dataset = UnpairedDehazingDataset(
        hazy_image_directory=resolve_project_path(data_config["hazy_image_directory"]),
        clean_image_directory=resolve_project_path(data_config["clean_image_directory"]),
        resize=int(data_config["train_resize"]),
        crop=int(data_config["train_crop"]),
        horizontal_flip=bool(data_config["train_horizontal_flip"]),
        overfit_single_pair=args.overfit_single_pair,
        additional_clean_image_directory=(
            resolve_project_path(data_config["additional_clean_image_directory"])
            if data_config.get("additional_clean_image_directory")
            else None
        ),
        additional_clean_image_manifest=(
            resolve_project_path(data_config["additional_clean_image_manifest"])
            if data_config.get("additional_clean_image_manifest")
            else None
        ),
        additional_clean_sampling_probability=float(
            data_config.get("additional_clean_sampling_probability", 0.0)
        ),
    )
    observed_counts = {
        "hazy": len(dataset.hazy_images),
        "clean": len(dataset.primary_clean_images),
        "additional_clean": len(dataset.additional_clean_images),
        "clean_total": len(dataset.clean_images),
    }
    if (
        "expected_hazy_image_count" in data_config
        and "expected_clean_image_count" in data_config
    ):
        expected_counts = {
            "hazy": int(data_config["expected_hazy_image_count"]),
            "clean": int(data_config["expected_clean_image_count"]),
            "additional_clean": int(
                data_config.get("expected_additional_clean_image_count", 0)
            ),
            "clean_total": int(data_config["expected_clean_image_count"])
            + int(data_config.get("expected_additional_clean_image_count", 0)),
        }
        if observed_counts != expected_counts:
            raise RuntimeError(
                "Training data count changed; update the config explicitly: "
                f"observed={observed_counts}, expected={expected_counts}"
            )
    data_manifest = {
        "hazy_root": str(dataset.hazy_image_directory),
        "hazy_index_sha256": image_index_fingerprint(
            dataset.hazy_image_directory,
            dataset.hazy_images,
        ),
        "primary_clean_root": str(dataset.clean_image_directory),
        "primary_clean_index_sha256": image_index_fingerprint(
            dataset.clean_image_directory,
            dataset.primary_clean_images,
        ),
        "additional_clean_root": (
            str(dataset.additional_clean_image_directory)
            if dataset.additional_clean_image_directory is not None
            else None
        ),
        "additional_clean_manifest": (
            str(dataset.additional_clean_image_manifest)
            if dataset.additional_clean_image_manifest is not None
            else None
        ),
        "additional_clean_manifest_sha256": file_sha256(
            dataset.additional_clean_image_manifest
        ),
        "additional_clean_index_sha256": (
            image_index_fingerprint(
                dataset.additional_clean_image_directory,
                dataset.additional_clean_images,
            )
            if dataset.additional_clean_image_directory is not None
            else None
        ),
        "additional_clean_sampling_probability": (
            dataset.additional_clean_sampling_probability
        ),
        "counts": observed_counts,
    }
    data_manifest_fingerprint = canonical_json_fingerprint(data_manifest)
    evaluation_protocol_fingerprint = canonical_json_fingerprint(
        config["evaluation"]
    )
    atomic_write_json(
        run_dir / "resolved_config_upaired_dehazing.json",
        {key: value for key, value in config.items() if key != "_config_path"},
    )
    atomic_write_json(
        run_dir / "data_manifest_upaired_dehazing.json",
        {
            **data_manifest,
            "fingerprint": data_manifest_fingerprint,
        },
    )
    fixed_hazy, fixed_clean, fixed_sample_paths = load_fixed_samples(
        dataset,
        sample_count=int(train_config["fixed_sample_count"]),
        inference_size=int(data_config["inference_size"]),
    )
    batch_size = int(args.batch_size or train_config["batch_size"])
    generator = build_unpaired_dehazing_generator(config, device=device).to(device)
    post_generator_python_rng = __import__("random").getstate()
    post_generator_numpy_rng = __import__("numpy").random.get_state()
    post_generator_torch_rng = torch.get_rng_state()
    post_generator_cuda_rng = torch.cuda.get_rng_state_all()
    objective_profile = str(loss_config.get("objective_profile", "legacy_v2"))
    reconstruction: nn.Module = ReconstructionLoss().to(device)
    image_objective = build_image_objective(
        objective_profile,
        loss_config,
        reconstruction_loss=reconstruction,
    ).to(device)
    training_objective_signature = objective_signature(
        objective_profile,
        loss_config,
    )
    adversarial_weights = resolve_adversarial_weights(
        objective_profile,
        loss_config,
    )
    generator_adversarial_weight = adversarial_weights["generator"]
    discriminator_adversarial_weight = adversarial_weights["discriminator"]
    adapter_pretraining_steps = (
        int(args.adapter_pretraining_steps)
        if args.adapter_pretraining_steps is not None
        else int(generator.adapter_pretraining_steps)
    )
    if adapter_pretraining_steps < 0:
        raise ValueError("adapter_pretraining_steps must be non-negative")
    completed_adapter_pretraining_steps = 0
    if adapter_pretraining_steps > 0 and not args.resume:
        adapter_parameters = generator.additional_trainable_parameters()
        if not adapter_parameters:
            raise ValueError(
                "Adapter pretraining was requested, but the generator has no adapter parameters"
            )
        pretraining_optimizer = Adam(
            adapter_parameters,
            lr=float(
                train_config.get(
                    "adapter_pretraining_learning_rate",
                    train_config["learning_rate"],
                )
            ),
            betas=(
                float(train_config["adam_beta1"]),
                float(train_config["adam_beta2"]),
            ),
            eps=float(train_config["adam_epsilon"]),
            weight_decay=float(train_config["adam_weight_decay"]),
        )
        pretraining_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed + 10_000),
        )
        pretraining_iterator = iter(pretraining_loader)
        pretraining_progress = tqdm(
            total=adapter_pretraining_steps,
            dynamic_ncols=True,
            desc="cra-adapter-pretrain",
        )
        for pretraining_step in range(1, adapter_pretraining_steps + 1):
            try:
                pretraining_batch = next(pretraining_iterator)
            except StopIteration:
                pretraining_iterator = iter(pretraining_loader)
                pretraining_batch = next(pretraining_iterator)
            clean_pretraining = pretraining_batch["clean"].to(
                device, non_blocking=True
            )
            generator.train()
            pretraining_optimizer.zero_grad(set_to_none=True)
            with amp_context(precision):
                pretraining_loss, pretraining_components = (
                    generator.adapter_pretraining_loss(
                        clean_pretraining,
                        reconstruction,
                    )
                )
            pretraining_loss.backward()
            pretraining_grad_norm = clip_finite_gradients(
                adapter_parameters,
                float(train_config["max_grad_norm"]),
                run_dir=run_dir,
                step=pretraining_step,
                component="vae_adapter_pretraining",
            )
            pretraining_optimizer.step()
            pretraining_record = {
                "pretraining_step": pretraining_step,
                "time": time.time(),
                "adapter_pretraining_total": float(pretraining_loss.detach()),
                "adapter_pretraining_grad_norm": pretraining_grad_norm,
                **{
                    key: float(value.detach())
                    for key, value in pretraining_components.items()
                },
            }
            if not finite_scalar_dict(pretraining_record):
                raise FloatingPointError(
                    f"Non-finite adapter pretraining metrics: {pretraining_record}"
                )
            append_jsonl(
                run_dir / "adapter_pretraining_metrics.jsonl",
                pretraining_record,
            )
            atomic_write_json(
                run_dir / "latest_adapter_pretraining_metrics.json",
                pretraining_record,
            )
            pretraining_progress.update(1)
            pretraining_progress.set_postfix(
                loss=f"{pretraining_record['adapter_pretraining_total']:.4f}"
            )
            completed_adapter_pretraining_steps = pretraining_step
        pretraining_progress.close()
        atomic_write_json(
            run_dir / "adapter_pretraining.complete.json",
            {
                "status": "complete",
                "steps": completed_adapter_pretraining_steps,
                "extension_signature": generator.extension_signature(),
                "time": time.time(),
            },
        )
        del pretraining_iterator, pretraining_loader, pretraining_optimizer

    adapter_identity_pretraining_steps = (
        int(args.adapter_identity_pretraining_steps)
        if args.adapter_identity_pretraining_steps is not None
        else int(generator.adapter_identity_pretraining_steps)
    )
    if adapter_identity_pretraining_steps < 0:
        raise ValueError("adapter_identity_pretraining_steps must be non-negative")
    completed_adapter_identity_pretraining_steps = 0
    if not args.resume:
        generator.finalize_adapter_pretraining()

    __import__("random").setstate(post_generator_python_rng)
    __import__("numpy").random.set_state(post_generator_numpy_rng)
    torch.set_rng_state(post_generator_torch_rng)
    torch.cuda.set_rng_state_all(post_generator_cuda_rng)
    discriminator = build_unpaired_dehazing_discriminator(
        model_config,
        loss_config,
    ).to(device)
    seed_everything(seed)

    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    generator_parameters = list(generator.var.parameters()) + list(
        generator.additional_trainable_parameters()
    )
    discriminator_parameters = discriminator.head_parameters()
    if str(train_config["optimizer"]).lower() != "adam":
        raise ValueError("The audited dehazing configuration requires optimizer=adam")
    optimizer_g = Adam(
        generator_parameters,
        lr=float(train_config["learning_rate"]),
        betas=(float(train_config["adam_beta1"]), float(train_config["adam_beta2"])),
        eps=float(train_config["adam_epsilon"]),
        weight_decay=float(train_config["adam_weight_decay"]),
    )
    optimizer_d = Adam(
        discriminator_parameters,
        lr=float(train_config["discriminator_learning_rate"]),
        betas=(float(train_config["adam_beta1"]), float(train_config["adam_beta2"])),
        eps=float(train_config["adam_epsilon"]),
        weight_decay=float(train_config["adam_weight_decay"]),
    )
    warmup_steps = int(train_config["warmup_steps"])
    scheduler_g = LambdaLR(
        optimizer_g,
        lr_lambda=lambda update: linear_warmup_multiplier(update, warmup_steps),
    )
    scheduler_d = LambdaLR(
        optimizer_d,
        lr_lambda=lambda update: linear_warmup_multiplier(update, warmup_steps),
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            Path(args.resume).resolve(),
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d,
            training_objective_signature,
        )

    metadata = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_visible_index": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": current_git_commit(),
        "trainable_generator_parameters": generator.trainable_parameter_count,
        "trainable_discriminator_parameters": sum(
            parameter.numel() for parameter in discriminator_parameters
        ),
        "dataset_counts": observed_counts,
        "data_manifest_fingerprint": data_manifest_fingerprint,
        "evaluation_protocol_fingerprint": evaluation_protocol_fingerprint,
        "discriminator_signature": discriminator.discriminator_signature(),
        "generator_extension_signature": generator.extension_signature(),
        "training_objective_signature": training_objective_signature,
        "adapter_pretraining_steps": completed_adapter_pretraining_steps,
        "adapter_identity_pretraining_steps": (
            completed_adapter_identity_pretraining_steps
        ),
        "batch_size": batch_size,
        "fixed_sample_paths": fixed_sample_paths,
        "start_step": start_step,
        "max_steps": max_steps,
        "config": {key: value for key, value in config.items() if key != "_config_path"},
    }
    atomic_write_json(run_dir / "run_metadata.json", metadata)
    atomic_write_json(run_dir / "process.json", {"pid": os.getpid(), "alive": True})

    stop_requested = False

    def handle_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        append_jsonl(
            run_dir / "events.jsonl",
            {"time": time.time(), "event": "signal", "signum": signum},
        )

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    hazy_label = generator.domain_id("hazy")
    clean_label = generator.domain_id("clean")
    max_grad_norm = float(train_config["max_grad_norm"])
    metrics_path = run_dir / "train_metrics.jsonl"
    progress = tqdm(total=max_steps, initial=start_step, dynamic_ncols=True)
    data_iterator = iter(loader)
    step = start_step

    if (
        start_step == 0
        and bool(train_config.get("save_initial_checkpoint", False))
        and not args.skip_initial_checkpoint
    ):
        initial_checkpoint = save_checkpoint(
            run_dir,
            0,
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d,
            config,
            training_objective_signature,
            checkpoints_to_keep,
            minimum_checkpoint_free_gib,
        )
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "time": time.time(),
                "event": "initial_checkpoint",
                "path": str(initial_checkpoint),
            },
        )
        if sample_every > 0:
            save_sample_grid(
                run_dir,
                0,
                generator,
                fixed_hazy,
                fixed_clean,
                hazy_label,
                clean_label,
                precision,
            )

    while step < max_steps and not stop_requested:
        step_started = time.time()
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)
        hazy = batch["hazy"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        clean_source_batch = batch["clean_source"]
        clean_source = (
            str(clean_source_batch[0])
            if isinstance(clean_source_batch, (list, tuple))
            and len(clean_source_batch) == 1
            else list(clean_source_batch)
            if isinstance(clean_source_batch, (list, tuple))
            else str(clean_source_batch)
        )
        torch.cuda.reset_peak_memory_stats(device)
        generator.train()
        discriminator.train()

        # 1) Bidirectional cycle objective, accumulated direction-by-direction.
        optimizer_g.zero_grad(set_to_none=True)
        with amp_context(precision):
            fake_clean, aux_hazy_to_clean = generator(
                hazy, clean_label, return_aux=True
            )
            reconstructed_hazy, aux_clean_to_hazy_cycle = generator(
                fake_clean, hazy_label, return_aux=True
            )
            cycle_hazy, cycle_hazy_components = image_objective(
                reconstructed_hazy, hazy, path="cycle"
            )
            cycle_hazy_auxiliary, cycle_hazy_auxiliary_components = (
                generator.cycle_auxiliary_loss(
                    source=hazy,
                    source_domain="hazy",
                    first_auxiliary=aux_hazy_to_clean,
                    second_auxiliary=aux_clean_to_hazy_cycle,
                )
            )
        (0.5 * (cycle_hazy + cycle_hazy_auxiliary)).backward()
        aux_hazy_to_clean = {
            key: value
            for key, value in aux_hazy_to_clean.items()
            if not key.startswith("_")
        }
        del aux_clean_to_hazy_cycle
        del reconstructed_hazy, fake_clean

        with amp_context(precision):
            fake_hazy, aux_clean_to_hazy = generator(
                clean, hazy_label, return_aux=True
            )
            reconstructed_clean, aux_hazy_to_clean_cycle = generator(
                fake_hazy, clean_label, return_aux=True
            )
            cycle_clean, cycle_clean_components = image_objective(
                reconstructed_clean, clean, path="cycle"
            )
            cycle_clean_auxiliary, cycle_clean_auxiliary_components = (
                generator.cycle_auxiliary_loss(
                    source=clean,
                    source_domain="clean",
                    first_auxiliary=aux_clean_to_hazy,
                    second_auxiliary=aux_hazy_to_clean_cycle,
                )
            )
        (0.5 * (cycle_clean + cycle_clean_auxiliary)).backward()
        aux_clean_to_hazy = {
            key: value
            for key, value in aux_clean_to_hazy.items()
            if not key.startswith("_")
        }
        del aux_hazy_to_clean_cycle
        del reconstructed_clean, fake_hazy
        grad_cycle = parameter_grad_norm(generator_parameters)

        # 2) Generator adversarial objective. Keep detached fakes for D.
        discriminator.set_heads_trainable(False)
        with amp_context(precision):
            fake_clean = generator(hazy, clean_label)
            gan_hazy_to_clean = generator_adversarial_weight * discriminator.loss(
                fake_clean, "clean", for_real=True, for_generator=True
            )
        (0.5 * gan_hazy_to_clean).backward()
        fake_clean_for_d = fake_clean.detach()
        del fake_clean

        with amp_context(precision):
            fake_hazy = generator(clean, hazy_label)
            gan_clean_to_hazy = generator_adversarial_weight * discriminator.loss(
                fake_hazy, "hazy", for_real=True, for_generator=True
            )
        (0.5 * gan_clean_to_hazy).backward()
        fake_hazy_for_d = fake_hazy.detach()
        del fake_hazy
        grad_gan = parameter_grad_norm(generator_parameters)
        discriminator.set_heads_trainable(True)

        # 3) Identity objective, direction-by-direction.
        with amp_context(precision):
            identity_hazy, aux_identity_hazy = generator(
                hazy, hazy_label, return_aux=True
            )
            identity_hazy_loss, identity_hazy_components = image_objective(
                identity_hazy, hazy, path="identity"
            )
            identity_hazy_auxiliary, identity_hazy_auxiliary_components = (
                generator.identity_auxiliary_loss(
                    source=hazy,
                    source_domain="hazy",
                    auxiliary=aux_identity_hazy,
                )
            )
        (0.5 * (identity_hazy_loss + identity_hazy_auxiliary)).backward()
        del aux_identity_hazy
        del identity_hazy

        with amp_context(precision):
            identity_clean, aux_identity_clean = generator(
                clean, clean_label, return_aux=True
            )
            identity_clean_loss, identity_clean_components = image_objective(
                identity_clean, clean, path="identity"
            )
            identity_clean_auxiliary, identity_clean_auxiliary_components = (
                generator.identity_auxiliary_loss(
                    source=clean,
                    source_domain="clean",
                    auxiliary=aux_identity_clean,
                )
            )
        (0.5 * (identity_clean_loss + identity_clean_auxiliary)).backward()
        del aux_identity_clean
        del identity_clean

        # 4) Adapter-specific supervision.
        with amp_context(precision):
            adapter_training_loss, adapter_training_components = (
                generator.adapter_training_loss(
                    hazy=hazy,
                    clean=clean,
                    step=step + 1,
                    reconstruction=reconstruction,
                )
            )
        if adapter_training_loss.requires_grad:
            adapter_training_loss.backward()
        grad_identity = clip_finite_gradients(
            generator_parameters,
            max_grad_norm,
            run_dir=run_dir,
            step=step + 1,
            component="generator",
        )
        optimizer_g.step()
        scheduler_g.step()
        optimizer_g.zero_grad(set_to_none=True)

        # 5) Two domain discriminator heads, using the latest cached fake images.
        optimizer_d.zero_grad(set_to_none=True)
        with amp_context(precision):
            d_clean_real = discriminator.loss(
                clean, "clean", for_real=True, for_generator=False
            )
            d_clean_fake = discriminator.loss(
                fake_clean_for_d, "clean", for_real=False, for_generator=False
            )
            d_hazy_real = discriminator.loss(
                hazy, "hazy", for_real=True, for_generator=False
            )
            d_hazy_fake = discriminator.loss(
                fake_hazy_for_d, "hazy", for_real=False, for_generator=False
            )
            discriminator_loss = discriminator_adversarial_weight * 0.25 * (
                d_clean_real + d_clean_fake + d_hazy_real + d_hazy_fake
            )
        discriminator_loss.backward()
        grad_discriminator = clip_finite_gradients(
            discriminator_parameters,
            max_grad_norm,
            run_dir=run_dir,
            step=step + 1,
            component="discriminator",
        )
        optimizer_d.step()
        scheduler_d.step()
        optimizer_d.zero_grad(set_to_none=True)
        del fake_clean_for_d, fake_hazy_for_d

        step += 1
        objective_metrics = {
            **{
                f"cycle_hazy_{key}": float(value.detach())
                for key, value in cycle_hazy_components.items()
            },
            **{
                f"cycle_clean_{key}": float(value.detach())
                for key, value in cycle_clean_components.items()
            },
            **{
                f"identity_hazy_{key}": float(value.detach())
                for key, value in identity_hazy_components.items()
            },
            **{
                f"identity_clean_{key}": float(value.detach())
                for key, value in identity_clean_components.items()
            },
        }
        auxiliary_metrics = {
            "cycle_hazy_auxiliary": float(cycle_hazy_auxiliary.detach()),
            "cycle_clean_auxiliary": float(cycle_clean_auxiliary.detach()),
            "identity_hazy_auxiliary": float(identity_hazy_auxiliary.detach()),
            "identity_clean_auxiliary": float(identity_clean_auxiliary.detach()),
            **{
                f"cycle_hazy_auxiliary_{key}": float(value.detach())
                for key, value in cycle_hazy_auxiliary_components.items()
            },
            **{
                f"cycle_clean_auxiliary_{key}": float(value.detach())
                for key, value in cycle_clean_auxiliary_components.items()
            },
            **{
                f"identity_hazy_auxiliary_{key}": float(value.detach())
                for key, value in identity_hazy_auxiliary_components.items()
            },
            **{
                f"identity_clean_auxiliary_{key}": float(value.detach())
                for key, value in identity_clean_auxiliary_components.items()
            },
        }
        scalar_metrics = {
            "cycle_hazy": float(cycle_hazy.detach()),
            "cycle_clean": float(cycle_clean.detach()),
            "gan_hazy_to_clean": float(gan_hazy_to_clean.detach()),
            "gan_clean_to_hazy": float(gan_clean_to_hazy.detach()),
            "identity_hazy": float(identity_hazy_loss.detach()),
            "identity_clean": float(identity_clean_loss.detach()),
            **objective_metrics,
            **auxiliary_metrics,
            "discriminator": float(discriminator_loss.detach()),
            "d_clean_real": float(d_clean_real.detach()),
            "d_clean_fake": float(d_clean_fake.detach()),
            "d_hazy_real": float(d_hazy_real.detach()),
            "d_hazy_fake": float(d_hazy_fake.detach()),
            "generator_total": float(
                0.5
                * (
                    cycle_hazy.detach()
                    + cycle_clean.detach()
                    + cycle_hazy_auxiliary.detach()
                    + cycle_clean_auxiliary.detach()
                    + gan_hazy_to_clean.detach()
                    + gan_clean_to_hazy.detach()
                    + identity_hazy_loss.detach()
                    + identity_clean_loss.detach()
                    + identity_hazy_auxiliary.detach()
                    + identity_clean_auxiliary.detach()
                    + 2.0 * adapter_training_loss.detach()
                )
            ),
            "adapter_training_total": float(adapter_training_loss.detach()),
            **{
                key: float(value.detach())
                for key, value in adapter_training_components.items()
            },
            "grad_after_cycle": float(grad_cycle),
            "grad_after_cycle_gan": float(grad_gan),
            "grad_generator_preclip": float(grad_identity),
            "grad_discriminator_preclip": float(grad_discriminator),
        }
        if not finite_scalar_dict(scalar_metrics):
            crash = {"step": step, "reason": "non_finite_metric", **scalar_metrics}
            atomic_write_json(run_dir / "crash.json", crash)
            raise FloatingPointError(json.dumps(crash, sort_keys=True))

        record = {
            "step": step,
            "time": time.time(),
            "elapsed_seconds": time.time() - step_started,
            "clean_source": clean_source,
            "lr_generator": optimizer_g.param_groups[0]["lr"],
            "lr_discriminator": optimizer_d.param_groups[0]["lr"],
            "srq_entropy_hazy_to_clean": aux_hazy_to_clean["srq_entropy"],
            "srq_entropy_clean_to_hazy": aux_clean_to_hazy["srq_entropy"],
            "source_code_unique_hazy": aux_hazy_to_clean[
                "source_code_unique_fraction"
            ],
            "source_code_unique_clean": aux_clean_to_hazy[
                "source_code_unique_fraction"
            ],
            "gpu_memory_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "gpu_memory_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
            "gpu_peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            **scalar_metrics,
        }
        for direction, auxiliary in (
            ("hazy_to_clean", aux_hazy_to_clean),
            ("clean_to_hazy", aux_clean_to_hazy),
        ):
            for key, value in auxiliary.items():
                if (
                    key in {"srq_entropy", "source_code_unique_fraction"}
                    or key.startswith("_")
                ):
                    continue
                if isinstance(value, (int, float)):
                    record[f"{direction}_{key}"] = value
        append_jsonl(metrics_path, record)
        atomic_write_json(run_dir / "latest_metrics.json", record)
        progress.update(1)
        progress.set_postfix(
            cycle=f"{0.5 * (record['cycle_hazy'] + record['cycle_clean']):.3f}",
            identity=f"{0.5 * (record['identity_hazy'] + record['identity_clean']):.3f}",
            d=f"{record['discriminator']:.3f}",
            mem=f"{record['gpu_peak_memory_gib']:.1f}G",
        )

        if sample_every > 0 and step % sample_every == 0:
            save_sample_grid(
                run_dir,
                step,
                generator,
                fixed_hazy,
                fixed_clean,
                hazy_label,
                clean_label,
                precision,
            )
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            checkpoint_path = save_checkpoint(
                run_dir,
                step,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                scheduler_g,
                scheduler_d,
                config,
                training_objective_signature,
                checkpoints_to_keep,
                minimum_checkpoint_free_gib,
            )
            append_jsonl(
                run_dir / "events.jsonl",
                {
                    "time": time.time(),
                    "event": "checkpoint",
                    "path": str(checkpoint_path),
                },
            )

    if (stop_requested or step == max_steps) and not args.skip_final_checkpoint:
        final_path = run_dir / "checkpoints" / f"step_{step:07d}.pt"
        if not final_path.is_file():
            save_checkpoint(
                run_dir,
                step,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                scheduler_g,
                scheduler_d,
                config,
                training_objective_signature,
                checkpoints_to_keep,
                minimum_checkpoint_free_gib,
            )
    progress.close()
    status = "stopped" if stop_requested else "complete"
    atomic_write_json(
        run_dir / "process.json",
        {"pid": os.getpid(), "alive": False, "status": status, "step": step},
    )


def _normalize_extension_signature(sig: Any) -> Any:
    import copy
    if not isinstance(sig, dict):
        return sig
    out = copy.deepcopy(sig)
    if isinstance(out.get("disen_vae"), dict) and out["disen_vae"].get("checkpoint_sha256"):
        out["disen_vae"].pop("checkpoint", None)
    return out
