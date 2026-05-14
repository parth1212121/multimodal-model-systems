from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.models.common import Downsample2D, ResBlock2D, Upsample2D, make_group_norm


class ConvVAE(nn.Module):
    def __init__(self, input_channels: int = 3, latent_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)

        self.encoder_stage_32 = nn.ModuleList(
            [ResBlock2D(base_channels, base_channels), ResBlock2D(base_channels, base_channels)]
        )
        self.down_32_to_64 = Downsample2D(base_channels, base_channels * 2)

        self.encoder_stage_64 = nn.ModuleList(
            [ResBlock2D(base_channels * 2, base_channels * 2), ResBlock2D(base_channels * 2, base_channels * 2)]
        )
        self.down_64_to_128 = Downsample2D(base_channels * 2, base_channels * 4)

        self.encoder_stage_128 = nn.ModuleList(
            [ResBlock2D(base_channels * 4, base_channels * 4), ResBlock2D(base_channels * 4, base_channels * 4)]
        )
        self.down_128_to_mid = Downsample2D(base_channels * 4, base_channels * 4)

        self.encoder_mid = nn.ModuleList(
            [ResBlock2D(base_channels * 4, base_channels * 4), ResBlock2D(base_channels * 4, base_channels * 4)]
        )
        self.to_moments = nn.Conv2d(base_channels * 4, latent_channels * 2, kernel_size=3, padding=1)

        self.from_latent = nn.Conv2d(latent_channels, base_channels * 4, kernel_size=3, padding=1)
        self.decoder_mid = nn.ModuleList(
            [ResBlock2D(base_channels * 4, base_channels * 4), ResBlock2D(base_channels * 4, base_channels * 4)]
        )

        self.decoder_stage_128 = nn.ModuleList(
            [ResBlock2D(base_channels * 4, base_channels * 4), ResBlock2D(base_channels * 4, base_channels * 4)]
        )
        self.up_128_to_64 = Upsample2D(base_channels * 4, base_channels * 2)

        self.decoder_stage_64 = nn.ModuleList(
            [ResBlock2D(base_channels * 2, base_channels * 2), ResBlock2D(base_channels * 2, base_channels * 2)]
        )
        self.up_64_to_32 = Upsample2D(base_channels * 2, base_channels)

        self.decoder_stage_32 = nn.ModuleList(
            [ResBlock2D(base_channels, base_channels), ResBlock2D(base_channels, base_channels)]
        )
        self.up_32_to_out = Upsample2D(base_channels, base_channels)

        self.output_norm = make_group_norm(base_channels)
        self.output_conv = nn.Conv2d(base_channels, input_channels, kernel_size=3, padding=1)

    def encode_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        for block in self.encoder_stage_32:
            x = block(x)
        x = self.down_32_to_64(x)
        for block in self.encoder_stage_64:
            x = block(x)
        x = self.down_64_to_128(x)
        for block in self.encoder_stage_128:
            x = block(x)
        x = self.down_128_to_mid(x)
        for block in self.encoder_mid:
            x = block(x)
        moments = self.to_moments(x)
        return torch.chunk(moments, chunks=2, dim=1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def encode(self, x: torch.Tensor, sample_posterior: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_stats(x)
        latent = self.reparameterize(mu, logvar) if sample_posterior else mu
        return latent, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        for block in self.decoder_mid:
            x = block(x)
        for block in self.decoder_stage_128:
            x = block(x)
        x = self.up_128_to_64(x)
        for block in self.decoder_stage_64:
            x = block(x)
        x = self.up_64_to_32(x)
        for block in self.decoder_stage_32:
            x = block(x)
        x = self.up_32_to_out(x)
        x = self.output_conv(F.silu(self.output_norm(x)))
        return torch.tanh(x)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> dict[str, torch.Tensor]:
        latent, mu, logvar = self.encode(x, sample_posterior=sample_posterior)
        reconstruction = self.decode(latent)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
        }

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
