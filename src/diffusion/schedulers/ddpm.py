from __future__ import annotations
"""
DDPM noise scheduler with linear and cosine beta schedules.
Shared between image diffusion (LDM) and world model decoder.
"""

import torch
from typing import Tuple


class DDPMScheduler:
    """
    DDPM noise schedule (Ho et al., 2020) + cosine variant (Nichol & Dhariwal, 2021).

    Args:
        num_train_timesteps: total diffusion steps T
        beta_start:          min noise (linear schedule)
        beta_end:            max noise (linear schedule)
        beta_schedule:       'cosine' or 'linear'
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
    ):
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        elif beta_schedule == "cosine":
            steps = num_train_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float64) / num_train_timesteps
            f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_cumprod = f / f[0]
            betas = (1 - alpha_cumprod[1:] / alpha_cumprod[:-1]).clamp(0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule!r}")

        self.betas = betas.float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

    def _at(self, coef: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return coef.to(t.device).gather(0, t).reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """q(x_t | x_0): sample noisy x_t and return (x_t, noise)."""
        noise = torch.randn_like(x0)
        a = self._at(self.sqrt_alphas_cumprod, t, x0.shape)
        b = self._at(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return a * x0 + b * noise, noise

    def step(self, noise_pred: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        """DDPM ancestral sampling step: p(x_{t-1} | x_t)."""
        dev = x_t.device
        beta_t = self.betas[t].to(dev)
        alpha_t = self.alphas[t].to(dev)
        alpha_bar_t = self.alphas_cumprod[t].to(dev)

        mean = (x_t - beta_t / (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_t.sqrt()
        if t == 0:
            return mean
        return mean + self.posterior_variance[t].to(dev).sqrt() * torch.randn_like(x_t)

    def ddim_step(
        self,
        noise_pred: torch.Tensor,
        t: int,
        t_prev: int,
        x_t: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM reverse step (Song et al., 2020). eta=0 is deterministic."""
        dev = x_t.device
        ab = self.alphas_cumprod[t].to(dev)
        ab_prev = self.alphas_cumprod[t_prev].to(dev) if t_prev >= 0 else torch.ones(1, device=dev)

        x0 = ((x_t - (1 - ab).sqrt() * noise_pred) / ab.sqrt()).clamp(-1, 1)
        sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt().clamp(min=0)
        direction = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * noise_pred
        noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
        return ab_prev.sqrt() * x0 + direction + noise
