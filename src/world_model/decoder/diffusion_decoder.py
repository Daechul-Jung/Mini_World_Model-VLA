"""Stage-D diffusion decoder: latents -> sharp pixels.

Optional. The stage-A tokenizer already has a decoder; this one trades ~25 extra
network evaluations per frame for sharpness, by treating the frame latents as
cross-attention conditioning for a DDPM UNet (the DIAMOND / latent-diffusion
pattern).

When to use which renderer:

| Caller                        | Renderer      | Why                                   |
|-------------------------------|---------------|---------------------------------------|
| RL loop (`WorldModelEnv`)     | `tokenizer`   | 1 forward/frame; the policy sees a    |
|                               |               | blurrier but 25x cheaper frame        |
| Qualitative videos, papers    | `decoder`     | sharpness is the deliverable          |
| Reward models on imagined obs | *measure it*  | a reward model tuned on real frames   |
|                               |               | may be sensitive to tokenizer blur    |

That last row is a real open question in this project, not a settled choice --
see `research/012_reward_on_imagined_frames.md`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.core.base import Decoder
from world_model.core.registry import DECODERS

from .ddpm import DDPMScheduler
from .diffusion_unet import UNet


@DECODERS.register(
    "diffusion_unet",
    paper="Ho et al. 2020 / DIAMOND",
    status="baseline",
    note="DDPM UNet cross-attending to frame latents",
)
class DiffusionDecoder(Decoder):
    """DDPM UNet conditioned on tokenizer latents via cross-attention.

    Args:
        context_dim: must equal the tokenizer's `latent_spec.dim`; the frame's
            (D, h, w) latent map is flattened to (h*w, D) conditioning tokens.
        num_train_timesteps / beta_schedule: DDPM noise schedule.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        attn_at_levels: Tuple[bool, ...] = (False, False, True, True),
        context_dim: int = 256,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ):
        super().__init__()
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mults=tuple(channel_mults),
            attn_at_levels=tuple(attn_at_levels),
            context_dim=context_dim,
            n_res_blocks=n_res_blocks,
            dropout=dropout,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule
        )
        self.context_dim = context_dim

    # ------------------------------------------------------------------ helper

    @staticmethod
    def _to_context(latents: torch.Tensor) -> torch.Tensor:
        """(B, D, h, w) -> (B, h*w, D) cross-attention tokens."""
        b, d, h, w = latents.shape
        return latents.reshape(b, d, h * w).transpose(1, 2)

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """(B, T, ...) -> (B*T, ...) plus the shape needed to undo it."""
        b, t = x.shape[:2]
        return x.flatten(0, 1), b, t

    # ----------------------------------------------------------------- stage D

    def forward(
        self, frames: torch.Tensor, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Denoising score-matching loss.

        frames:  (B, T, 3, H, W) or (B, 3, H, W) -- the clean targets
        latents: matching (B, T, D, h, w) or (B, D, h, w) from the *frozen* stage-A
                 tokenizer. Training on the tokenizer's own latents (rather than
                 dynamics predictions) is deliberate: stage D learns to render a
                 latent well, and is not asked to also absorb dynamics error.
        """
        if frames.ndim == 5:
            frames, _, _ = self._flatten_time(frames)
            latents, _, _ = self._flatten_time(latents)

        b = frames.shape[0]
        t = torch.randint(0, self.scheduler.num_train_timesteps, (b,), device=frames.device)
        noisy, noise = self.scheduler.add_noise(frames, t)
        pred = self.unet(noisy, t, self._to_context(latents))
        loss = F.mse_loss(pred, noise)
        return loss, {"denoise_mse": loss.detach().item()}

    # ---------------------------------------------------------------- sampling

    @torch.no_grad()
    def render(
        self,
        latents: torch.Tensor,
        steps: int = 25,
        eta: float = 0.0,
        image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """DDIM-sample frames from latents. (B, T, D, h, w) -> (B, T, 3, H, W)."""
        squeeze_time = latents.ndim == 4
        if squeeze_time:
            latents = latents.unsqueeze(1)
        flat, b, t = self._flatten_time(latents)

        h_lat, w_lat = flat.shape[-2:]
        # The UNet is fully convolutional; infer pixel size from the tokenizer's
        # downsample factor unless the caller pins it.
        size = image_size or h_lat * kwargs.get("downsample", 8)

        context = self._to_context(flat)
        x = torch.randn(flat.shape[0], 3, size, size, device=flat.device, dtype=flat.dtype)

        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1, 0, steps, dtype=torch.long, device=flat.device
        )
        for i, t_cur in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < steps else torch.tensor(-1, device=flat.device)
            noise_pred = self.unet(x, t_cur.expand(x.shape[0]), context)
            x = self.scheduler.ddim_step(noise_pred, int(t_cur), int(t_prev), x, eta)

        frames = x.clamp(-1, 1).unflatten(0, (b, t))
        return frames.squeeze(1) if squeeze_time else frames
