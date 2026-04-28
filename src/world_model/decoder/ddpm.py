from __future__ import annotations
import torch
from typing import Tuple


class DDPMScheduler:
    """
    DDPM noise scheduler (Ho et al., 2020) with cosine or linear beta schedule.

    Manages the forward (noising) and reverse (denoising) processes.
    The reverse step is used during training loss computation and DDIM sampling.

    Args:
        num_train_timesteps: total diffusion steps T
        beta_start:          smallest noise level (linear schedule only)
        beta_end:            largest noise level (linear schedule only)
        beta_schedule:       'cosine' (Nichol & Dhariwal, 2021) or 'linear'
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
            alphas_cumprod = f / f[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = betas.clamp(0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        self.betas = betas.float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # Derived quantities used in forward and reverse steps
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _at(self, coef: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Gather coefficient at timestep t and broadcast to x_shape."""
        b = t.shape[0]
        out = coef.to(t.device).gather(0, t)
        return out.reshape(b, *([1] * (len(x_shape) - 1)))

    # ------------------------------------------------------------------
    # Forward process  q(x_t | x_0)
    # ------------------------------------------------------------------

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from q(x_t | x_0) = N(sqrt(ā_t)*x_0, (1-ā_t)*I).

        Args:
            x0: (B, ...) clean data
            t:  (B,)     integer timesteps in [0, T-1]
        Returns:
            x_t:   noisy sample
            noise: the noise that was added (prediction target)
        """
        noise = torch.randn_like(x0)
        sqrt_ab = self._at(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1m_ab = self._at(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

    # ------------------------------------------------------------------
    # Reverse step  p(x_{t-1} | x_t)   (DDPM ancestral sampling)
    # ------------------------------------------------------------------

    def step(
        self, noise_pred: torch.Tensor, t: int, x_t: torch.Tensor
    ) -> torch.Tensor:
        """
        One DDPM reverse step.

        Args:
            noise_pred: predicted noise from denoising network
            t:          current integer timestep (scalar)
            x_t:        noisy sample at step t
        Returns:
            x_{t-1}: denoised sample
        """
        device = x_t.device
        beta_t = self.betas[t].to(device)
        alpha_t = self.alphas[t].to(device)
        alpha_bar_t = self.alphas_cumprod[t].to(device)

        coef = beta_t / (1.0 - alpha_bar_t).sqrt()
        mean = (x_t - coef * noise_pred) / alpha_t.sqrt()

        if t == 0:
            return mean

        var = self.posterior_variance[t].to(device)
        return mean + var.sqrt() * torch.randn_like(x_t)

    # ------------------------------------------------------------------
    # DDIM step  (deterministic, fewer steps at inference)
    # ------------------------------------------------------------------

    def ddim_step(
        self,
        noise_pred: torch.Tensor,
        t: int,
        t_prev: int,
        x_t: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        One DDIM reverse step (Song et al., 2020).

        Args:
            noise_pred: predicted noise from denoising network
            t:          current timestep index
            t_prev:     previous timestep index (< t)
            x_t:        noisy sample at t
            eta:        stochasticity: 0 = deterministic DDIM, 1 = DDPM
        Returns:
            x_{t_prev}
        """
        device = x_t.device
        ab = self.alphas_cumprod[t].to(device)
        ab_prev = self.alphas_cumprod[t_prev].to(device) if t_prev >= 0 else torch.ones(1, device=device)

        x0_pred = (x_t - (1 - ab).sqrt() * noise_pred) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
        direction = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * noise_pred

        noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
        return ab_prev.sqrt() * x0_pred + direction + noise
