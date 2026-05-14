from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .text import TextTransformer
from .vit import VisionTransformer


class CLIPModel(nn.Module):
    def __init__(
        self,
        vision_backbone: VisionTransformer,
        text_backbone: TextTransformer,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.text_backbone = text_backbone
        self.vision_projection = nn.Linear(vision_backbone.hidden_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_backbone.hidden_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_image(self, images: torch.Tensor, pooling: str = "cls") -> dict[str, torch.Tensor]:
        features = self.vision_backbone.forward_features(images)
        if pooling == "cls":
            pooled = features["cls"]
        elif pooling == "mean_patches":
            pooled = features["patches"].mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        projected = F.normalize(self.vision_projection(pooled), dim=-1)
        return {"features": pooled, "embeddings": projected}

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.text_backbone(input_ids, attention_mask)
        projected = F.normalize(self.text_projection(features["pooled"]), dim=-1)
        return {"features": features["pooled"], "embeddings": projected}

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        image_outputs = self.encode_image(images)
        text_outputs = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_outputs["embeddings"] @ text_outputs["embeddings"].t()
        logits_per_text = logits_per_image.t()
        return {
            "image_embeddings": image_outputs["embeddings"],
            "text_embeddings": text_outputs["embeddings"],
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "logit_scale": logit_scale,
        }

    @staticmethod
    def compute_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = outputs["logits_per_image"].size(0)
        targets = torch.arange(batch_size, device=outputs["logits_per_image"].device)
        loss_i = F.cross_entropy(outputs["logits_per_image"], targets)
        loss_t = F.cross_entropy(outputs["logits_per_text"], targets)
        return 0.5 * (loss_i + loss_t)
