from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1.0e-5, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 500, schedule: str = "cosine") -> None:
        super().__init__()
        if schedule != "cosine":
            raise ValueError(f"Unsupported diffusion schedule: {schedule}")

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1.0e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def _extract(tensor: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        values = tensor.gather(0, timesteps)
        return values.view(timesteps.shape[0], *((1,) * (len(target_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def p_mean_variance(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise_prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.predict_x0_from_noise(x_t, timesteps, noise_prediction)
        mean = self._extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x0 + self._extract(
            self.posterior_mean_coef2, timesteps, x_t.shape
        ) * x_t
        log_variance = self._extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return mean, log_variance

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        null_context: torch.Tensor,
        guidance_scale: float = 4.0,
    ) -> torch.Tensor:
        noise_uncond = model(x_t, timesteps, null_context)
        noise_cond = model(x_t, timesteps, context)
        noise_prediction = (1.0 + guidance_scale) * noise_cond - guidance_scale * noise_uncond
        mean, log_variance = self.p_mean_variance(x_t, timesteps, noise_prediction)
        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().view(x_t.shape[0], *((1,) * (x_t.dim() - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        context: torch.Tensor,
        null_context: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 4.0,
    ) -> torch.Tensor:
        latents = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            latents = self.p_sample(model, latents, timesteps, context, null_context, guidance_scale=guidance_scale)
        return latents
