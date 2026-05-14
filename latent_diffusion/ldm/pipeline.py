from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ldm.checkpoints import load_checkpoint
from ldm.config import load_json
from ldm.models.unet import LatentDiffusionModel
from ldm.models.vae import ConvVAE


def build_vae_from_config(config: dict[str, Any]) -> ConvVAE:
    model_cfg = config["model"]
    return ConvVAE(
        input_channels=int(model_cfg.get("input_channels", 3)),
        latent_channels=int(model_cfg.get("latent_channels", 4)),
        base_channels=int(model_cfg.get("base_channels", 32)),
    )


def build_ldm_from_config(config: dict[str, Any]) -> LatentDiffusionModel:
    model_cfg = config["model"]
    return LatentDiffusionModel(
        latent_channels=int(model_cfg.get("latent_channels", 4)),
        base_channels=int(model_cfg.get("base_channels", 128)),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4])),
        time_dim=int(model_cfg.get("time_embedding_dim", 512)),
        context_dim=int(model_cfg.get("context_dim", 512)),
        num_heads=int(model_cfg.get("attention_heads", 8)),
        text_seq_len=int(model_cfg.get("text_seq_len", 77)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
    )


def load_vae_from_checkpoint(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> ConvVAE:
    model = build_vae_from_config(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_ldm_from_checkpoint(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> LatentDiffusionModel:
    model = build_ldm_from_config(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_latent_stats(path: str | Path) -> dict[str, list[float]]:
    return load_json(path)


def latent_stats_to_tensors(stats: dict[str, list[float]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["mean"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    return mean, std


def denormalize_latents(latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return latents * std + mean
