from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional encoding → MLP for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -freq)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Core blocks
# ---------------------------------------------------------------------------

def _norm(ch: int) -> nn.GroupNorm:
    groups = min(32, ch)
    while ch % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    """ResBlock with time-step conditioning via additive scale-shift."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = _norm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class SelfAttn2D(nn.Module):
    """Spatial self-attention (flatten H×W → sequence)."""

    def __init__(self, ch: int, num_heads: int = 8):
        super().__init__()
        self.norm = _norm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class CrossAttn2D(nn.Module):
    """
    Cross-attention conditioning on external context (e.g. VQ latent tokens).

    Args:
        ch:          spatial feature channels
        context_dim: context token dimension
        num_heads:   attention heads
    """

    def __init__(self, ch: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = _norm(ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True,
                                          kdim=context_dim, vdim=context_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, ctx, ctx)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Encoder / Decoder levels
# ---------------------------------------------------------------------------

class DownLevel(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        use_attn: bool,
        context_dim: int,
        n_res: int,
        dropout: float,
    ):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResBlock(in_ch if i == 0 else out_ch, out_ch, time_dim, dropout)
            for i in range(n_res)
        ])
        self.self_attn = SelfAttn2D(out_ch) if use_attn else None
        self.cross_attn = CrossAttn2D(out_ch, context_dim) if (use_attn and context_dim > 0) else None
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for res in self.resnets:
            x = res(x, t_emb)
        if self.self_attn is not None:
            x = self.self_attn(x)
        if self.cross_attn is not None and ctx is not None:
            x = self.cross_attn(x, ctx)
        skip = x
        return self.downsample(x), skip


class UpLevel(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        time_dim: int,
        use_attn: bool,
        context_dim: int,
        n_res: int,
        dropout: float,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.resnets = nn.ModuleList([
            ResBlock(in_ch + skip_ch if i == 0 else out_ch, out_ch, time_dim, dropout)
            for i in range(n_res)
        ])
        self.self_attn = SelfAttn2D(out_ch) if use_attn else None
        self.cross_attn = CrossAttn2D(out_ch, context_dim) if (use_attn and context_dim > 0) else None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = torch.cat([self.upsample(x), skip], dim=1)
        for res in self.resnets:
            x = res(x, t_emb)
        if self.self_attn is not None:
            x = self.self_attn(x)
        if self.cross_attn is not None and ctx is not None:
            x = self.cross_attn(x, ctx)
        return x


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net denoiser for the world model's diffusion decoder.

    Conditioned on:
      - Diffusion timestep t   (sinusoidal + MLP)
      - VQ-VAE latent context  (cross-attention at lower resolutions)

    Args:
        in_channels:     noisy image channels (3 for RGB)
        out_channels:    predicted noise channels
        base_channels:   channel count at the finest resolution
        channel_mults:   multiplier at each downsampling level
        attn_at_levels:  whether to apply self + cross attention per level
        context_dim:     VQ latent embedding dimension (0 = no conditioning)
        n_res_blocks:    number of ResBlocks per level
        dropout:         dropout rate
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        attn_at_levels: Tuple[bool, ...] = (False, False, True, True),
        context_dim: int = 256,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_emb = SinusoidalTimeEmb(base_channels)

        channels = [base_channels * m for m in channel_mults]
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (down)
        self.down_levels = nn.ModuleList()
        in_ch = base_channels
        self._skip_channels: List[int] = []
        for i, out_ch in enumerate(channels):
            self.down_levels.append(
                DownLevel(in_ch, out_ch, time_dim, attn_at_levels[i], context_dim, n_res_blocks, dropout)
            )
            self._skip_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        self.mid_res1 = ResBlock(in_ch, in_ch, time_dim, dropout)
        self.mid_self_attn = SelfAttn2D(in_ch)
        self.mid_cross_attn = CrossAttn2D(in_ch, context_dim) if context_dim > 0 else None
        self.mid_res2 = ResBlock(in_ch, in_ch, time_dim, dropout)

        # Decoder (up)
        self.up_levels = nn.ModuleList()
        up_channels = list(reversed(channels))
        skip_channels = list(reversed(self._skip_channels))
        attn_rev = list(reversed(attn_at_levels))
        for i in range(len(up_channels) - 1):
            out_ch = up_channels[i + 1]
            self.up_levels.append(
                UpLevel(in_ch, skip_channels[i], out_ch, time_dim, attn_rev[i], context_dim, n_res_blocks, dropout)
            )
            in_ch = out_ch
        # Final up to base_channels
        self.up_levels.append(
            UpLevel(in_ch, skip_channels[-1], base_channels, time_dim, attn_rev[-1], context_dim, n_res_blocks, dropout)
        )

        self.out_norm = _norm(base_channels)
        self.out_proj = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, C, H, W) noisy input
            t:       (B,)         integer timestep indices
            context: (B, N, context_dim) conditioning tokens or None
        Returns:
            noise_pred: (B, C, H, W)
        """
        t_emb = self.time_emb(t)    # (B, time_dim)
        h = self.input_proj(x)

        skips = []
        for level in self.down_levels:
            h, skip = level(h, t_emb, context)
            skips.append(skip)

        h = self.mid_res1(h, t_emb)
        h = self.mid_self_attn(h)
        if self.mid_cross_attn is not None and context is not None:
            h = self.mid_cross_attn(h, context)
        h = self.mid_res2(h, t_emb)

        for level, skip in zip(self.up_levels, reversed(skips)):
            h = level(h, skip, t_emb, context)

        return self.out_proj(F.silu(self.out_norm(h)))
