from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)


def build_optimizer(parameters: Iterable[torch.nn.Parameter], cfg: dict) -> torch.optim.Optimizer:
    optimizer_name = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    eps = float(cfg.get("eps", 1.0e-8))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_scale: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(total_steps, 1)
    warmup_steps = max(min(warmup_steps, total_steps - 1), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    Image.fromarray(image).save(output_path)


def save_image_grid(tensors: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    grid = make_grid(tensors.detach().cpu().clamp(-1.0, 1.0), nrow=nrow, normalize=True, value_range=(-1, 1))
    save_tensor_image(grid * 2.0 - 1.0, path)
