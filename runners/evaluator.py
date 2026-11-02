from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm

from datasets.unpaired_dehazing import (
    denormalize,
    image_files_recursive,
    inference_transform,
)
from models.dehazevar_generator import build_unpaired_dehazing_generator
from utils.io_utils import (
    append_jsonl,
    atomic_write_json,
    load_config,
    resolve_project_path,
)
from utils.losses import objective_signature, ssim_loss

if TYPE_CHECKING:
    from models.dehazevar_generator import CycleVARUnpairedDehazingGenerator


def resolve_test_sets(
    evaluation_config: dict[str, Any], requested_names: list[str] | None
) -> dict[str, dict[str, Any]]:
    """Resolve an ordered, duplicate-free subset from the configured test-set registry."""
    registry = evaluation_config["test_sets"]
    names = list(
        requested_names
        if requested_names is not None
        else evaluation_config["production_test_sets"]
    )
    if not names:
        raise ValueError("At least one evaluation dataset must be selected")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate evaluation dataset names: {names}")
    missing = [name for name in names if name not in registry]
    if missing:
        raise ValueError(f"Unregistered evaluation datasets: {missing}")
    return {name: registry[name] for name in names}


def amp_context(precision: str):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp32":
        return nullcontext()
    raise ValueError(f"Unsupported evaluation precision: {precision!r}")


def output_name_by_source_stem(sources: list[Path]) -> dict[Path, str]:
    names: dict[Path, str] = {}
    owners: dict[str, Path] = {}
    for source in sources:
        output_name = f"{source.stem}.png"
        if output_name in owners:
            raise RuntimeError(
                "Duplicate source stems cannot be paired unambiguously: "
                f"{owners[output_name]} and {source}"
            )
        names[source] = output_name
        owners[output_name] = source
    return names


def validate_directory_contents(directory: Path, expected_names: set[str]) -> None:
    for temporary in directory.glob(".*.tmp.png"):
        temporary.unlink()
    actual_names = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    stale_names = actual_names - expected_names
    if stale_names:
        raise RuntimeError(
            f"Refusing stale images in {directory}: {sorted(stale_names)[:5]}"
        )


def image_is_valid(path: Path, expected_size: int) -> bool:
    try:
        with Image.open(path) as image:
            if image.size != (expected_size, expected_size):
                return False
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def atomic_save_png(image: Image.Image, output_path: Path) -> None:
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.png"
    )
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_evaluation_lock(path: Path):
    """Hold one advisory lock for an evaluation run; process exit releases it safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluator already holds {path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "time": time.time()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_resized_reference(
    source_directory: Path,
    output_directory: Path,
    size: int,
) -> Path:
    """Create a resized reference derivative directory without altering source images."""
    sources = image_files_recursive(source_directory)
    output_names = output_name_by_source_stem(sources)
    expected_names = set(output_names.values())
    output_directory.mkdir(parents=True, exist_ok=True)
    validate_directory_contents(output_directory, expected_names)
    for source, output_name in tqdm(
        output_names.items(), desc=f"reference:{output_directory.name}"
    ):
        output_path = output_directory / output_name
        if output_path.is_file() and image_is_valid(output_path, size):
            continue
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((size, size), Image.Resampling.LANCZOS)
            atomic_save_png(image, output_path)
    validate_directory_contents(output_directory, expected_names)
    invalid_names = sorted(
        name
        for name in expected_names
        if not image_is_valid(output_directory / name, size)
    )
    if invalid_names:
        raise RuntimeError(
            f"Reference materialization is incomplete in {output_directory}: "
            f"{invalid_names[:5]}"
        )
    return output_directory


def _normalize_extension_signature(sig: Any) -> Any:
    if not isinstance(sig, dict):
        return sig
    out = copy.deepcopy(sig)
    if isinstance(out.get("disen_vae"), dict) and out["disen_vae"].get("checkpoint_sha256"):
        out["disen_vae"].pop("checkpoint", None)
    return out


def load_generator(
    config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[CycleVARUnpairedDehazingGenerator, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_format = payload.get("format_version")
    if checkpoint_format not in {2, 3} or payload.get("task") != "upaired_dehazing":
        raise ValueError(f"Not an unpaired-dehazing checkpoint: {checkpoint_path}")
    generator = build_unpaired_dehazing_generator(config, device=device).to(device)
    generator.var.load_state_dict(payload["generator_var"], strict=True)
    checkpoint_extension_signature = payload.get(
        "generator_extension_signature", {"type": "none"}
    )
    if checkpoint_format == 2 and checkpoint_extension_signature != {"type": "none"}:
        raise ValueError(
            "Checkpoint format 2 cannot contain a VAE-adapter extension; "
            "use format 3"
        )
    current_extension_signature = generator.extension_signature()
    if _normalize_extension_signature(
        checkpoint_extension_signature
    ) != _normalize_extension_signature(current_extension_signature):
        raise ValueError(
            "Checkpoint VAE-adapter signature does not match the evaluation config: "
            f"checkpoint={checkpoint_extension_signature}, "
            f"config={current_extension_signature}"
        )
    generator.load_extension_state_dict(payload.get("generator_extension", {}))

    loss_config = config["loss"]
    objective_profile = str(loss_config.get("objective_profile", "legacy_v2"))
    configured_objective_signature = objective_signature(
        objective_profile,
        loss_config,
    )
    checkpoint_objective_signature = payload.get(
        "training_objective_signature",
        None,
    )
    if checkpoint_objective_signature is None:
        checkpoint_loss_config = payload.get("config", {}).get("loss")
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
    if checkpoint_objective_signature != configured_objective_signature:
        raise ValueError(
            "Checkpoint training objective does not match the evaluation config: "
            f"checkpoint={checkpoint_objective_signature}, "
            f"config={configured_objective_signature}"
        )
    generator.eval()
    return generator, int(payload["step"])


@torch.inference_mode()
def generate_dataset_predictions(
    *,
    dataset_name: str,
    source_directory: Path,
    output_directory: Path,
    generator: CycleVARUnpairedDehazingGenerator,
    transform,
    precision: str,
    device: torch.device,
    max_images: int,
    target_domain: str = "clean",
) -> dict[str, Any]:
    sources = image_files_recursive(source_directory)
    full_count = len(sources)
    if max_images > 0:
        sources = sources[:max_images]
    output_names = output_name_by_source_stem(sources)
    expected_names = set(output_names.values())
    output_directory.mkdir(parents=True, exist_ok=True)
    validate_directory_contents(output_directory, expected_names)

    total_seconds = 0.0
    generated_count = 0
    for source_path, output_name in tqdm(
        output_names.items(), desc=f"infer:{dataset_name}"
    ):
        output_path = output_directory / output_name
        if output_path.is_file() and image_is_valid(
            output_path, int(transform.transforms[0].size[0])
        ):
            continue
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            source = transform(image).unsqueeze(0).to(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with amp_context(precision):
            translated = generator(source, target_domain)
        torch.cuda.synchronize(device)
        total_seconds += time.perf_counter() - started
        atomic_save_png(
            TF.to_pil_image(denormalize(translated)[0].cpu()), output_path
        )
        generated_count += 1
    validate_directory_contents(output_directory, expected_names)
    invalid_names = sorted(
        name
        for name in expected_names
        if not image_is_valid(
            output_directory / name, int(transform.transforms[0].size[0])
        )
    )
    actual_count = len(expected_names) - len(invalid_names)
    if invalid_names or actual_count != len(sources):
        raise RuntimeError(
            f"Prediction count mismatch for {dataset_name}: "
            f"expected={len(sources)}, actual={actual_count}, "
            f"invalid={invalid_names[:5]}"
        )
    return {
        "dataset": dataset_name,
        "source_directory": str(source_directory),
        "prediction_directory": str(output_directory),
        "source_count": full_count,
        "evaluated_count": actual_count,
        "newly_generated_count": generated_count,
        "mean_generation_seconds_for_new_images": (
            total_seconds / generated_count if generated_count else 0.0
        ),
        "inference_preprocessing": f"exact_{transform.transforms[0].size[0]}_lanczos",
        "complete_dataset": max_images <= 0,
    }


def _pair_predictions_with_ground_truth(
    prediction_directory: Path,
    ground_truth_directory: Path,
    pairing_rule: str,
) -> list[tuple[Path, Path]]:
    predictions = image_files_recursive(prediction_directory)
    ground_truths = {
        path.stem: path for path in image_files_recursive(ground_truth_directory)
    }
    pairs: list[tuple[Path, Path]] = []
    for pred_path in predictions:
        if pairing_rule == "same_stem":
            gt_key = pred_path.stem
        elif pairing_rule == "prefix_underscore":
            gt_key = pred_path.stem.split("_")[0]
        else:
            raise ValueError(f"Unsupported pairing_rule={pairing_rule!r}")
        if gt_key not in ground_truths:
            raise FileNotFoundError(
                f"Missing ground-truth image for prediction {pred_path.name} "
                f"(key={gt_key!r}) in {ground_truth_directory}"
            )
        pairs.append((pred_path, ground_truths[gt_key]))
    return pairs


def _load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return TF.to_tensor(rgb).unsqueeze(0).to(device)


@torch.inference_mode()
def compute_dataset_metrics(
    *,
    dataset_name: str,
    dataset_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    prediction_directory: Path,
    result_directory: Path,
    reference_root: Path,
    inference_size: int,
) -> dict[str, float]:
    mode = str(dataset_config["evaluation_mode"])
    metrics = [str(m) for m in dataset_config.get("metrics", [])]
    device = torch.device(str(evaluation_config.get("metric_device", "cuda:0")))
    try:
        import pyiqa  # type: ignore[import-not-found]
    except ImportError:
        pyiqa = None

    results: dict[str, float] = {}
    if mode == "full-reference":
        gt_source = resolve_project_path(dataset_config["ground_truth_directory"])
        gt_directory = prepare_resized_reference(
            gt_source,
            reference_root / f"{dataset_name}_ground_truth",
            inference_size,
        )
        pairs = _pair_predictions_with_ground_truth(
            prediction_directory,
            gt_directory,
            str(dataset_config.get("pairing_rule", "same_stem")),
        )
        iqa_models: dict[str, Any] = {}
        lpips_fallback = None
        for metric_name in metrics:
            if pyiqa is not None:
                iqa_models[metric_name] = pyiqa.create_metric(
                    metric_name, device=device
                )
            elif metric_name == "lpips":
                import lpips

                lpips_fallback = lpips.LPIPS(net="alex").to(device).eval()
            elif metric_name not in {"psnr", "ssim"}:
                raise RuntimeError(
                    f"Metric {metric_name!r} requires the 'pyiqa' package to be installed"
                )

        accumulators: dict[str, list[float]] = {m: [] for m in metrics}
        for pred_path, gt_path in tqdm(pairs, desc=f"metrics:{dataset_name}"):
            pred_tensor = _load_rgb_tensor(pred_path, device)
            gt_tensor = _load_rgb_tensor(gt_path, device)
            for metric_name in metrics:
                if metric_name in iqa_models:
                    score = float(iqa_models[metric_name](pred_tensor, gt_tensor).item())
                elif metric_name == "psnr":
                    mse = float(F.mse_loss(pred_tensor, gt_tensor).item())
                    score = 100.0 if mse == 0.0 else -10.0 * math.log10(mse)
                elif metric_name == "ssim":
                    score = float(
                        (1.0 - ssim_loss(pred_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0)).item()
                    )
                elif metric_name == "lpips" and lpips_fallback is not None:
                    score = float(
                        lpips_fallback(pred_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0)
                        .mean()
                        .item()
                    )
                else:
                    raise RuntimeError(f"Unsupported metric: {metric_name}")
                accumulators[metric_name].append(score)
        for metric_name, scores in accumulators.items():
            results[metric_name] = float(sum(scores) / max(len(scores), 1))

    elif mode == "no-reference":
        if pyiqa is None:
            raise RuntimeError(
                "No-reference evaluation (FID, NIQE, MUSIQ, MANIQA) requires the 'pyiqa' package"
            )
        prediction_paths = image_files_recursive(prediction_directory)
        for metric_name in metrics:
            metric_model = pyiqa.create_metric(metric_name, device=device)
            if metric_name == "fid":
                fid_ref_source = resolve_project_path(
                    evaluation_config["fid_reference_directory"]
                )
                fid_preprocess = str(
                    evaluation_config.get(
                        "fid_reference_preprocess", "resize_to_inference_size"
                    )
                )
                if fid_preprocess == "resize_to_inference_size":
                    fid_ref_dir = prepare_resized_reference(
                        fid_ref_source,
                        reference_root / "fid_reference",
                        inference_size,
                    )
                else:
                    fid_ref_dir = fid_ref_source
                results[metric_name] = float(
                    metric_model(str(prediction_directory), str(fid_ref_dir))
                )
            else:
                scores = [
                    float(metric_model(_load_rgb_tensor(path, device)).item())
                    for path in tqdm(
                        prediction_paths, desc=f"metric:{dataset_name}:{metric_name}"
                    )
                ]
                results[metric_name] = float(sum(scores) / max(len(scores), 1))
    else:
        raise ValueError(f"Unsupported evaluation_mode: {mode!r}")

    atomic_write_json(
        result_directory / "metrics.json",
        {"dataset": dataset_name, "metrics": results},
    )
    return results


def evaluate_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    evaluation_root: Path,
    max_images: int,
    dataset_names: list[str] | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Dehazing inference and evaluation require CUDA")
    evaluation_config = config["evaluation"]
    test_sets = resolve_test_sets(evaluation_config, dataset_names)
    selected_dataset_names = list(test_sets)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator, step = load_generator(config, checkpoint_path, device)
    step_directory = evaluation_root / f"step_{step:07d}"
    completion_path = step_directory / "evaluation.complete.json"
    if completion_path.is_file():
        existing = json.loads(completion_path.read_text(encoding="utf-8"))
        existing_dataset_names = list(
            existing.get("dataset_names", existing.get("dataset_results", {}))
        )
        if existing_dataset_names == selected_dataset_names and (
            max_images > 0 or existing.get("status") == "complete"
        ):
            return existing
        raise RuntimeError(
            f"Existing evaluation at {completion_path} used datasets "
            f"{existing_dataset_names}, requested {selected_dataset_names}; "
            "use a distinct --output-directory"
        )
    step_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        step_directory / "evaluation.in_progress.json",
        {
            "status": "running",
            "step": step,
            "checkpoint": str(checkpoint_path),
            "pid": os.getpid(),
            "dataset_names": selected_dataset_names,
            "time": time.time(),
        },
    )

    inference_size = int(config["data"]["inference_size"])
    transform = inference_transform(inference_size)
    precision = str(config["training"]["mixed_precision"])
    dataset_results: dict[str, dict[str, Any]] = {}
    current_dataset_name: str | None = None
    try:
        for dataset_name, dataset_config in test_sets.items():
            current_dataset_name = dataset_name
            dataset_directory = step_directory / dataset_name
            predictions = dataset_directory / "predictions"
            source_dir = resolve_project_path(dataset_config["source_directory"])
            inference_result = generate_dataset_predictions(
                dataset_name=dataset_name,
                source_directory=source_dir,
                output_directory=predictions,
                generator=generator,
                transform=transform,
                precision=precision,
                device=device,
                max_images=max_images,
            )
            dataset_results[dataset_name] = {
                **inference_result,
                "metrics": {},
            }
            atomic_write_json(
                dataset_directory / "inference.json",
                dataset_results[dataset_name],
            )

        del generator
        torch.cuda.empty_cache()

        reference_root = (
            evaluation_root.parent / f"evaluation_references_{inference_size}"
        )
        for dataset_name, dataset_config in test_sets.items():
            current_dataset_name = dataset_name
            dataset_directory = step_directory / dataset_name
            result = dataset_results[dataset_name]
            if dataset_config["evaluation_mode"] == "qualitative-only":
                result["metric_status"] = "qualitative_only"
                atomic_write_json(dataset_directory / "result.json", result)
                continue
            metric_values = compute_dataset_metrics(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                evaluation_config=evaluation_config,
                prediction_directory=dataset_directory / "predictions",
                result_directory=dataset_directory,
                reference_root=reference_root,
                inference_size=inference_size,
            )
            result["metrics"] = metric_values
            result["metric_result_json"] = str(dataset_directory / "metrics.json")
            result["metric_status"] = (
                "subset_complete" if max_images > 0 else "complete"
            )
            atomic_write_json(dataset_directory / "result.json", result)

        payload = {
            "schema_version": 1,
            "status": "complete" if max_images <= 0 else "subset_complete",
            "step": step,
            "checkpoint": str(checkpoint_path),
            "dataset_names": selected_dataset_names,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "inference_size": inference_size,
            "dataset_results": dataset_results,
        }
        atomic_write_json(completion_path, payload)
        current_dataset_name = None
        (step_directory / "evaluation.in_progress.json").unlink(missing_ok=True)
        append_jsonl(evaluation_root.parent / "evaluation_metrics.jsonl", payload)
        return payload
    except BaseException as error:
        failure = {
            "status": "failed",
            "step": step,
            "checkpoint": str(checkpoint_path),
            "dataset": current_dataset_name,
            "exception_type": type(error).__name__,
            "exception": str(error),
            "time": time.time(),
        }
        atomic_write_json(step_directory / "evaluation.failed.json", failure)
        raise


def run_evaluation(args: argparse.Namespace) -> None:
    if args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    config = load_config(args.config)
    dataset_names = list(resolve_test_sets(config["evaluation"], args.dataset_names))
    checkpoint_path = Path(args.checkpoint).resolve()
    if args.output_directory:
        evaluation_root = Path(args.output_directory).resolve()
    else:
        evaluation_root = checkpoint_path.parent.parent / "evaluations"
    with exclusive_evaluation_lock(evaluation_root.parent / "evaluation.lock"):
        payload = evaluate_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            evaluation_root=evaluation_root,
            max_images=args.max_images,
            dataset_names=dataset_names,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_inference(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("DehazeVAR inference requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint_path = Path(args.checkpoint).resolve()
    generator, step = load_generator(config, checkpoint_path, device)
    inference_size = int(config["data"]["inference_size"])
    transform = inference_transform(inference_size)
    precision = str(config["training"]["mixed_precision"])
    summary = generate_dataset_predictions(
        dataset_name="custom_inference",
        source_directory=Path(args.input_dir).resolve(),
        output_directory=Path(args.output_dir).resolve(),
        generator=generator,
        transform=transform,
        precision=precision,
        device=device,
        max_images=int(args.max_images),
        target_domain=str(args.target_domain),
    )
    summary["step"] = step
    summary["checkpoint"] = str(checkpoint_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
