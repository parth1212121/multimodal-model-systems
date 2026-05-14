from __future__ import annotations

from pathlib import Path

import torch

from .clip_model import CLIPModel
from .text import TextTransformer
from .tokenization import SimpleTokenizer
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


def build_text_backbone(config: dict, vocab_size: int) -> TextTransformer:
    return TextTransformer(
        vocab_size=vocab_size,
        context_length=config["context_length"],
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


def load_clip_full_model_bundle(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CLIPModel, dict, SimpleTokenizer]:
    bundle = torch.load(path, map_location=device, weights_only=False)
    if bundle.get("bundle_type") != "clip_multimodal":
        raise ValueError(f"{path} is not a CLIP multimodal bundle")
    config = bundle["config"]
    tokenizer = SimpleTokenizer.from_dict(bundle["tokenizer"])
    vision = build_vision_backbone(config["model"]["vision"])
    text = build_text_backbone(config["model"]["text"], vocab_size=len(tokenizer.token_to_id))
    model = CLIPModel(
        vision_backbone=vision,
        text_backbone=text,
        embed_dim=config["model"]["embed_dim"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    return model, bundle, tokenizer
