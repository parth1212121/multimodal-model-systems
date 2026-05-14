from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def safe_torch_save(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def cosine_warmup_lambda(
    current_step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    if total_steps <= 0:
        return 1.0
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(item, device) for item in batch)
    return batch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def build_export_metadata(
    source: str,
    model_family: str,
    config: dict[str, Any],
    feature_dim: int,
    normalization: dict[str, Any],
    pooling: str,
) -> dict[str, Any]:
    return {
        "bundle_type": "vision_encoder",
        "source": source,
        "model_family": model_family,
        "config": config,
        "feature_dim": feature_dim,
        "normalization": normalization,
        "pooling": pooling,
    }
