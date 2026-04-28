from __future__ import annotations
"""
Variational Autoencoder (VAE) for Latent Diffusion Models.

Compresses high-res RGB images to a compact latent space where the diffusion
UNet operates, dramatically reducing compute.

Typical configuration (matching LDM / Stable Diffusion f=8):
  Input:  256×256×3
  Latent: 32×32×4   (8× spatial downsample, 4-channel latent)

The encoder uses the reparameterization trick to sample z ~ N(μ, σ²).
The KL loss is small (weight 1e-6) so the autoencoder behaves close to
a deterministic encoder at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _norm(ch: int) -> nn.GroupNorm:
    g = min(32, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            _norm(ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            _norm(ch), nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttnBlock(nn.Module):
    """Single-head spatial self-attention (at lowest resolution only)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = _norm(ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        scale = C ** -0.5
        q = self.q(h).reshape(B, C, -1).permute(0, 2, 1)   # (B, HW, C)
        k = self.k(h).reshape(B, C, -1).permute(0, 2, 1)
        v = self.v(h).reshape(B, C, -1).permute(0, 2, 1)
        attn = F.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.out(out)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """
    Progressive downsampling CNN encoder.

    Args:
        in_channels:   RGB channels (3)
        base_channels: width at finest resolution
        channel_mults: multiplier per downsampling level
        latent_dim:    output channels (mean + logvar → 2 * z_channels)
        n_res_blocks:  ResBlocks per level
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        z_channels: int = 4,
        n_res_blocks: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, base_channels, 3, padding=1)]
        ch = base_channels

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            if out_ch != ch:
                layers.append(nn.Conv2d(ch, out_ch, 1))
                ch = out_ch
            for _ in range(n_res_blocks):
                layers.append(ResBlock(ch))
            if i != len(channel_mults) - 1:
                layers.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))  # downsample

        # Bottleneck with self-attention
        layers += [ResBlock(ch), AttnBlock(ch), ResBlock(ch)]
        layers += [_norm(ch), nn.SiLU(), nn.Conv2d(ch, 2 * z_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, logvar) both of shape (B, z_channels, h, w)."""
        out = self.net(x)
        mean, logvar = out.chunk(2, dim=1)
        logvar = logvar.clamp(-30, 20)
        return mean, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class VAEDecoder(nn.Module):
    """Mirror of VAEEncoder."""

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        z_channels: int = 4,
        n_res_blocks: int = 2,
    ):
        super().__init__()
        reversed_mults = list(reversed(channel_mults))
        start_ch = base_channels * reversed_mults[0]

        layers: list[nn.Module] = [
            nn.Conv2d(z_channels, start_ch, 3, padding=1),
            ResBlock(start_ch), AttnBlock(start_ch), ResBlock(start_ch),
        ]
        ch = start_ch

        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            if out_ch != ch:
                layers.append(nn.Conv2d(ch, out_ch, 1))
                ch = out_ch
            for _ in range(n_res_blocks):
                layers.append(ResBlock(ch))
            if i != len(reversed_mults) - 1:
                layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))  # upsample

        layers += [_norm(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    Variational Autoencoder used in Latent Diffusion Models.

    The encoder produces a diagonal Gaussian q(z|x) = N(μ, σ²).
    During training, z is sampled via the reparameterization trick.
    During inference, use z = mean for deterministic encoding.

    Args:
        in_channels:  image channels (3)
        base_channels: encoder/decoder channel width at finest resolution
        channel_mults: channel multiplier per downsampling level
        z_channels:   latent channel depth (4 → latent is 4×H/f×W/f)
        n_res_blocks: ResBlocks per level
        kl_weight:    weight on KL divergence term in loss
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        z_channels: int = 4,
        n_res_blocks: int = 2,
        kl_weight: float = 1e-6,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.encoder = VAEEncoder(in_channels, base_channels, channel_mults, z_channels, n_res_blocks)
        self.decoder = VAEDecoder(in_channels, base_channels, channel_mults, z_channels, n_res_blocks)

    def encode(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      (B, C, H, W) image in [-1, 1]
            sample: if True, sample z; if False, return mean
        Returns:
            z:      (B, z_channels, h, w) latent
            mean:   (B, z_channels, h, w)
            logvar: (B, z_channels, h, w)
        """
        mean, logvar = self.encoder(x)
        if sample:
            std = (0.5 * logvar).exp()
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean
        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_rec: (B, C, H, W) reconstruction
            loss:  scalar (rec_loss + kl_weight * kl_loss)
        """
        z, mean, logvar = self.encode(x, sample=True)
        x_rec = self.decode(z)

        rec_loss = F.mse_loss(x_rec, x)
        kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()
        loss = rec_loss + self.kl_weight * kl_loss

        return x_rec, loss
