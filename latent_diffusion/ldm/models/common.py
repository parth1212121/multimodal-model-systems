from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = make_group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = make_group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels) if time_dim is not None else None
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        if self.time_proj is not None:
            if time_embedding is None:
                raise ValueError("time_embedding must be provided for time-conditioned ResBlock2D")
            x = x + self.time_proj(F.silu(time_embedding))[:, :, None, None]
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual


class Downsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        arguments = timesteps.float()[:, None] * frequencies[None, :]
        embeddings = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self_attended, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self_attended
        cross_attended, _ = self.cross_attn(self.norm2(x), context, context, need_weights=False)
        x = x + cross_attended
        x = x + self.ffn(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int,
        depth: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.norm = make_group_norm(channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList([BasicTransformerBlock(channels, context_dim, num_heads) for _ in range(depth)])
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.gradient_checkpointing = gradient_checkpointing

    def _checkpointed_forward(self, block: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return checkpoint(block, x, context, use_reentrant=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj_in(self.norm(x))
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width).transpose(1, 2)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = self._checkpointed_forward(block, x, context)
            else:
                x = block(x, context)
        x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
        return residual + self.proj_out(x)
