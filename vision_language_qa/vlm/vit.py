from __future__ import annotations

import torch
from torch import nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def _interpolate_pos_embed(self, height: int, width: int) -> torch.Tensor:
        if height == self.image_size and width == self.image_size:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        base_size = int(self.num_patches**0.5)
        patch_pos = patch_pos.reshape(1, base_size, base_size, self.hidden_dim).permute(0, 3, 1, 2)
        new_h = height // self.patch_size
        new_w = width // self.patch_size
        patch_pos = nn.functional.interpolate(
            patch_pos,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, self.hidden_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos_embed = self._interpolate_pos_embed(images.size(-2), images.size(-1))
        x = self.dropout(x + pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return {"cls": x[:, 0], "patches": x[:, 1:], "tokens": x}

    def forward(self, images: torch.Tensor, pooling: str = "cls") -> torch.Tensor:
        features = self.forward_features(images)
        if pooling == "cls":
            return features["cls"]
        if pooling == "mean_patches":
            return features["patches"].mean(dim=1)
        raise ValueError(f"Unsupported pooling mode: {pooling}")
