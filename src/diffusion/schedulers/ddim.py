from __future__ import annotations
"""
DDIM sampler (Song et al., 2020) — deterministic, ~50× fewer steps than DDPM.

Given a trained DDPMScheduler and a denoising UNet, this sampler runs
denoising in ~50 steps instead of 1000, producing the same quality.
"""

import torch
from typing import Callable, Optional

from .ddpm import DDPMScheduler


class DDIMSampler:
    """
    DDIM deterministic sampler built on top of a DDPMScheduler.

    Args:
        scheduler: DDPMScheduler with trained betas
        num_inference_steps: number of DDIM steps (20–100 is typical)
        eta: stochasticity (0 = fully deterministic DDIM, 1 ≈ DDPM)
    """

    def __init__(
        self,
        scheduler: DDPMScheduler,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        T = scheduler.num_train_timesteps
        step_ratio = T // num_inference_steps
        self.timesteps = (
            torch.arange(num_inference_steps - 1, -1, -1) * step_ratio
        ).long()

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        shape: tuple,
        device: torch.device,
        conditioning: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        uncond_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the full DDIM reverse chain from pure noise to a clean sample.

        Args:
            model:               callable (x_t, t, cond) → noise_pred
            shape:               output tensor shape (B, C, H, W)
            device:              torch device
            conditioning:        (B, N, D) conditioning tokens (text, VQ, etc.)
            guidance_scale:      classifier-free guidance scale (1 = no guidance)
            uncond_conditioning: uncond tokens for CFG; required if scale > 1
        Returns:
            x0: (B, C, H, W) generated sample in [-1, 1]
        """
        x = torch.randn(shape, device=device)
        ts = self.timesteps.to(device)

        for i, t in enumerate(ts):
            t_prev_idx = i + 1
            t_prev = ts[t_prev_idx].item() if t_prev_idx < len(ts) else -1

            t_batch = t.expand(shape[0])
            noise_pred = model(x, t_batch, conditioning)

            # Classifier-free guidance
            if guidance_scale > 1.0 and uncond_conditioning is not None:
                noise_uncond = model(x, t_batch, uncond_conditioning)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            x = self.scheduler.ddim_step(noise_pred, t.item(), t_prev, x, self.eta)

        return x
