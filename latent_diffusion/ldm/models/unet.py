from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.models.common import (
    Downsample2D,
    ResBlock2D,
    SinusoidalTimeEmbedding,
    SpatialTransformer,
    Upsample2D,
    make_group_norm,
)


class DownStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        context_dim: int,
        num_heads: int,
        add_downsample: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResBlock2D(in_channels, out_channels, time_dim=time_dim),
                ResBlock2D(out_channels, out_channels, time_dim=time_dim),
            ]
        )
        self.transformer = SpatialTransformer(
            out_channels,
            context_dim=context_dim,
            num_heads=num_heads,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.downsample = Downsample2D(out_channels, out_channels) if add_downsample else None

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.resblocks:
            x = block(x, time_embedding)
        x = self.transformer(x, context)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class UpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        context_dim: int,
        num_heads: int,
        add_upsample: bool,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResBlock2D(in_channels + skip_channels, out_channels, time_dim=time_dim),
                ResBlock2D(out_channels, out_channels, time_dim=time_dim),
            ]
        )
        self.transformer = SpatialTransformer(
            out_channels,
            context_dim=context_dim,
            num_heads=num_heads,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.upsample = Upsample2D(out_channels, out_channels // 2) if add_upsample else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_embedding: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        for block in self.resblocks:
            x = block(x, time_embedding)
        x = self.transformer(x, context)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class LatentUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: tuple[int, int, int] = (1, 2, 4),
        time_dim: int = 512,
        context_dim: int = 512,
        num_heads: int = 8,
        text_seq_len: int = 77,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        widths = [base_channels * multiplier for multiplier in channel_multipliers]
        self.text_seq_len = text_seq_len
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(latent_channels, widths[0], kernel_size=3, padding=1)

        self.down_stages = nn.ModuleList(
            [
                DownStage(widths[0], widths[0], time_dim, context_dim, num_heads, add_downsample=True, gradient_checkpointing=gradient_checkpointing),
                DownStage(widths[0], widths[1], time_dim, context_dim, num_heads, add_downsample=True, gradient_checkpointing=gradient_checkpointing),
                DownStage(widths[1], widths[2], time_dim, context_dim, num_heads, add_downsample=False, gradient_checkpointing=gradient_checkpointing),
            ]
        )

        self.mid_block1 = ResBlock2D(widths[2], widths[2], time_dim=time_dim)
        self.mid_transformer = SpatialTransformer(
            widths[2],
            context_dim=context_dim,
            num_heads=num_heads,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.mid_block2 = ResBlock2D(widths[2], widths[2], time_dim=time_dim)

        self.up_stages = nn.ModuleList(
            [
                UpStage(widths[2], widths[2], widths[2], time_dim, context_dim, num_heads, add_upsample=True, gradient_checkpointing=gradient_checkpointing),
                UpStage(widths[1], widths[1], widths[1], time_dim, context_dim, num_heads, add_upsample=True, gradient_checkpointing=gradient_checkpointing),
                UpStage(widths[0], widths[0], widths[0], time_dim, context_dim, num_heads, add_upsample=False, gradient_checkpointing=gradient_checkpointing),
            ]
        )
        self.output_norm = make_group_norm(widths[0])
        self.output_conv = nn.Conv2d(widths[0], latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embed(timesteps)
        x = self.input_conv(x)
        skips = []
        for stage in self.down_stages:
            x, skip = stage(x, time_embedding, context)
            skips.append(skip)

        x = self.mid_block1(x, time_embedding)
        x = self.mid_transformer(x, context)
        x = self.mid_block2(x, time_embedding)

        x = self.up_stages[0](x, skips[-1], time_embedding, context)
        x = self.up_stages[1](x, skips[-2], time_embedding, context)
        x = self.up_stages[2](x, skips[-3], time_embedding, context)
        return self.output_conv(F.silu(self.output_norm(x)))


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: tuple[int, int, int] = (1, 2, 4),
        time_dim: int = 512,
        context_dim: int = 512,
        num_heads: int = 8,
        text_seq_len: int = 77,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.unet = LatentUNet(
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            time_dim=time_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            text_seq_len=text_seq_len,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.null_context = nn.Parameter(torch.zeros(1, text_seq_len, context_dim))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timesteps, context)

    def expanded_null_context(self, batch_size: int) -> torch.Tensor:
        return self.null_context.expand(batch_size, -1, -1)
