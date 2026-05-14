from __future__ import annotations

from pathlib import Path

import torch

from .vit import VisionTransformer


def build_vision_backbone(config: dict) -> VisionTransformer:
    return VisionTransformer(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        in_channels=config.get("in_channels", 3),
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        dropout=config.get("dropout", 0.0),
    )


def load_vision_encoder_bundle(path: str | Path, device: str | torch.device = "cpu") -> tuple[VisionTransformer, dict]:
    bundle = torch.load(path, map_location=device, weights_only=False)
    if bundle.get("bundle_type") != "vision_encoder":
        raise ValueError(f"{path} is not a vision encoder bundle")
    model = build_vision_backbone(bundle["config"])
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    return model, bundle
