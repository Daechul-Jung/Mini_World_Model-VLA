from __future__ import annotations
"""
Latent Diffusion Model (LDM) for image generation.

Reference: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models"
           (CVPR 2022) — the paper behind Stable Diffusion.

Architecture:
  - VAE compresses images to a compact latent space (e.g. 32×32×4 for 256px input)
  - UNet operates in latent space (8× smaller than pixel space → huge compute savings)
  - Conditioning via cross-attention (text embeddings, class labels, or unconditional)
  - Trained with score-matching objective in latent space

For room image generation, train on LSUN bedroom/living_room or personal images.
Use DDIM sampling (50 steps) at inference for fast generation.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae.autoencoder import VAE
from .schedulers.ddpm import DDPMScheduler
from .schedulers.ddim import DDIMSampler


# ---------------------------------------------------------------------------
# UNet for latent diffusion
# (Same design as world model decoder but configured for latent space)
# ---------------------------------------------------------------------------

def _norm(ch: int) -> nn.GroupNorm:
    g = min(32, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -freq)
        emb = t.float()[:, None] * emb[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n1 = _norm(in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.tp = nn.Linear(t_dim, out_ch)
        self.n2 = _norm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.sc = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.tp(F.silu(t))[:, :, None, None]
        return self.c2(self.drop(F.silu(self.n2(h)))) + self.sc(x)


class CrossAttn(nn.Module):
    """Cross-attention for conditioning on text/label embeddings."""

    def __init__(self, ch: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = _norm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True,
                                          kdim=context_dim, vdim=context_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, ctx, ctx)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class SelfAttn(nn.Module):
    def __init__(self, ch: int, num_heads: int = 8):
        super().__init__()
        self.norm = _norm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class LDMUNet(nn.Module):
    """
    UNet that denoises in VAE latent space, conditioned on text/class context.

    Args:
        in_channels:   latent channels (z_channels from VAE)
        out_channels:  same as in_channels (predicts noise)
        base_channels: channel width at finest latent resolution
        channel_mults: width multiplier per level
        attn_at_levels: apply self+cross attention at this level?
        context_dim:   conditioning embedding dimension (e.g. 768 for CLIP)
        n_res_blocks:  ResBlocks per level
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 256,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        attn_at_levels: Tuple[bool, ...] = (False, True, True, True),
        context_dim: int = 512,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        t_dim = base_channels * 4
        self.time_emb = SinusoidalEmb(base_channels)
        channels = [base_channels * m for m in channel_mults]

        self.in_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        self._skips = []
        ch = base_channels
        for i, out_ch in enumerate(channels):
            res = nn.ModuleList([
                ResBlock(ch if j == 0 else out_ch, out_ch, t_dim, dropout)
                for j in range(n_res_blocks)
            ])
            attn = nn.ModuleList([
                nn.ModuleList([SelfAttn(out_ch), CrossAttn(out_ch, context_dim)])
                for _ in range(n_res_blocks)
            ]) if attn_at_levels[i] else None
            down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1) if i < len(channels) - 1 else None
            self.downs.append(nn.ModuleDict({"res": res, "attn": attn, "down": down}))
            self._skips.append(out_ch)
            ch = out_ch

        # Mid
        self.mid_res1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_sa = SelfAttn(ch)
        self.mid_ca = CrossAttn(ch, context_dim)
        self.mid_res2 = ResBlock(ch, ch, t_dim, dropout)

        # Decoder
        self.ups = nn.ModuleList()
        for i, (out_ch, skip_ch, use_attn) in enumerate(
            zip(reversed(channels[:-1]) , reversed(self._skips[:-1]), reversed(attn_at_levels[:-1]))
        ):
            res = nn.ModuleList([
                ResBlock(ch + skip_ch if j == 0 else out_ch, out_ch, t_dim, dropout)
                for j in range(n_res_blocks)
            ])
            attn = nn.ModuleList([
                nn.ModuleList([SelfAttn(out_ch), CrossAttn(out_ch, context_dim)])
                for _ in range(n_res_blocks)
            ]) if use_attn else None
            self.ups.append(nn.ModuleDict({
                "up": nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
                "res": res, "attn": attn,
            }))
            ch = out_ch

        self.out_norm = _norm(ch)
        self.out_proj = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, ctx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = self.in_proj(x)

        skips = []
        for level in self.downs:
            for i, res in enumerate(level["res"]):
                h = res(h, t_emb)
                if level["attn"] is not None:
                    sa, ca = level["attn"][i]
                    h = sa(h)
                    if ctx is not None:
                        h = ca(h, ctx)
            skips.append(h)
            if level["down"] is not None:
                h = level["down"](h)

        h = self.mid_res1(h, t_emb)
        h = self.mid_sa(h)
        if ctx is not None:
            h = self.mid_ca(h, ctx)
        h = self.mid_res2(h, t_emb)

        for level, skip in zip(self.ups, reversed(skips[:-1])):
            h = level["up"](h)
            h = torch.cat([h, skip], dim=1)
            for i, res in enumerate(level["res"]):
                h = res(h, t_emb)
                if level["attn"] is not None:
                    sa, ca = level["attn"][i]
                    h = sa(h)
                    if ctx is not None:
                        h = ca(h, ctx)

        return self.out_proj(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# LDM wrapper
# ---------------------------------------------------------------------------

class LatentDiffusionModel(nn.Module):
    """
    Full Latent Diffusion Model: VAE + UNet in latent space.

    Conditioning options:
      - Unconditional:       pass conditioning=None
      - Class-conditional:   pass a class-label embedding  (B, 1, context_dim)
      - Text-conditional:    pass CLIP text embeddings      (B, N_tokens, context_dim)

    Training (two-phase recommended):
      Phase 1: train VAE alone with reconstruction + KL loss.
      Phase 2: freeze VAE, train UNet with diffusion loss on frozen latents.

    Args:
        vae:            trained VAE (can be frozen during diffusion training)
        unet:           LDMUNet denoiser
        scheduler:      DDPMScheduler
        latent_scale:   scale factor applied to VAE latents before UNet
                        (0.18215 for SD-style VAEs to normalize std≈1)
    """

    def __init__(
        self,
        vae: VAE,
        unet: LDMUNet,
        scheduler: DDPMScheduler,
        latent_scale: float = 0.18215,
    ):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.latent_scale = latent_scale

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _, _ = self.vae.encode(x, sample=False)  # deterministic at inference
        return z * self.latent_scale

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.latent_scale)

    def diffusion_loss(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the LDM training loss (DDPM noise prediction in latent space).

        Args:
            x:            (B, C, H, W) images in [-1, 1]
            conditioning: (B, N, context_dim) or None
        Returns:
            scalar MSE loss
        """
        with torch.no_grad():
            z, _, _ = self.vae.encode(x, sample=True)
            z = z * self.latent_scale

        B = z.shape[0]
        t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=z.device)
        z_t, noise = self.scheduler.add_noise(z, t)
        noise_pred = self.unet(z_t, t, conditioning)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: Optional[torch.Tensor] = None,
        guidance_scale: float = 7.5,
        uncond_conditioning: Optional[torch.Tensor] = None,
        ddim_steps: int = 50,
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate images using DDIM sampling.

        Args:
            batch_size:          number of images to generate
            device:              target device
            conditioning:        (B, N, D) conditioning tokens or None
            guidance_scale:      CFG scale (1 = no guidance, 7.5 = typical)
            uncond_conditioning: (B, N, D) null conditioning for CFG
            ddim_steps:          number of denoising steps
        Returns:
            images: (B, 3, H, W) in [-1, 1]
        """
        sampler = DDIMSampler(self.scheduler, num_inference_steps=ddim_steps)

        # Infer latent spatial size from VAE downsample factor
        # For channel_mults=(1,2,4,4) → 3 downsamples → f=8; 256px → 32×32 latent
        h_lat = (image_size or 256) // 8
        z_channels = self.vae.encoder.net[-1].out_channels // 2  # mean only

        def model_fn(x, t, ctx):
            return self.unet(x, t, ctx)

        z = sampler.sample(
            model=model_fn,
            shape=(batch_size, 4, h_lat, h_lat),
            device=device,
            conditioning=conditioning,
            guidance_scale=guidance_scale,
            uncond_conditioning=uncond_conditioning,
        )
        return self.decode(z)

    @classmethod
    def create_medium(cls, context_dim: int = 512) -> "LatentDiffusionModel":
        """
        Medium LDM (~200M params) for room image generation.
        Trains on LSUN bedroom/living_room or personal indoor images.

        VAE:   ~84M  (base=128, mults=(1,2,4,4), z=4)
        UNet:  ~116M (base=256, mults=(1,2,4,4), context_dim=512)
        Total: ~200M
        """
        vae = VAE(
            in_channels=3,
            base_channels=128,
            channel_mults=(1, 2, 4, 4),
            z_channels=4,
            n_res_blocks=2,
            kl_weight=1e-6,
        )
        unet = LDMUNet(
            in_channels=4,
            out_channels=4,
            base_channels=256,
            channel_mults=(1, 2, 4, 4),
            attn_at_levels=(False, True, True, True),
            context_dim=context_dim,
            n_res_blocks=2,
        )
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="cosine")
        return cls(vae, unet, scheduler)
