from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config["_config_path"] = str(path)
    return config


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def trainable_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def linear_warmup_multiplier(update: int, warmup_updates: int) -> float:
    if warmup_updates <= 0:
        return 1.0
    return min(1.0, float(update + 1) / float(warmup_updates))


def finite_scalar_dict(values: dict[str, float]) -> bool:
    return all(np.isfinite(float(value)) for value in values.values())


PROJECT_ROOT = project_root()
