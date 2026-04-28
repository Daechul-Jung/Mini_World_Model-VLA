from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .quantizer import VectorQuantizer


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _norm(ch: int) -> nn.GroupNorm:
    groups = min(32, ch)
    while ch % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, ch)


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


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Convolutional encoder with progressive downsampling.

    For a 256×256 input with channel_mults=(1,2,4), this produces
    a 32×32 spatial feature map (8× downsample).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_mults: Tuple[int, ...],
        latent_dim: int,
        n_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = base_channels
        layers: list[nn.Module] = [nn.Conv2d(in_channels, ch, 3, padding=1)]

        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(n_res_blocks):
                if ch != out_ch:
                    layers.append(nn.Conv2d(ch, out_ch, 1))
                    ch = out_ch
                layers.append(ResBlock(ch, dropout))
            layers.append(Downsample(ch))

        layers += [_norm(ch), nn.SiLU(), nn.Conv2d(ch, latent_dim, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Mirror of Encoder with transposed convolutions for upsampling."""

    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        channel_mults: Tuple[int, ...],
        latent_dim: int,
        n_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        reversed_mults = list(reversed(channel_mults))
        start_ch = base_channels * reversed_mults[0]

        layers: list[nn.Module] = [nn.Conv2d(latent_dim, start_ch, 3, padding=1)]
        ch = start_ch

        for mult in reversed_mults:
            out_ch = base_channels * mult
            layers.append(Upsample(ch))
            for _ in range(n_res_blocks):
                if ch != out_ch:
                    layers.append(nn.Conv2d(ch, out_ch, 1))
                    ch = out_ch
                layers.append(ResBlock(ch, dropout))

        layers += [_norm(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# VQ-VAE
# ---------------------------------------------------------------------------

class VQVAE(nn.Module):
    """
    VQ-VAE for video frame tokenization.

    Encodes each frame into a grid of discrete tokens from a learned codebook.
    These tokens serve as input to the DynamicsTransformer.

    Args:
        in_channels:    image channels (3 for RGB)
        base_channels:  channel width before any multiplier
        channel_mults:  channel multiplier at each downsampling stage
        num_embeddings: codebook vocabulary size K
        latent_dim:     per-token embedding dimension D (= codebook dim)
        n_res_blocks:   ResBlocks applied per stage
        beta:           VQ commitment loss weight
        dropout:        spatial dropout in ResBlocks
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_embeddings: int = 1024,
        latent_dim: int = 256,
        n_res_blocks: int = 2,
        beta: float = 0.25,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder(in_channels, base_channels, channel_mults, latent_dim, n_res_blocks, dropout)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, beta)
        self.decoder = Decoder(in_channels, base_channels, channel_mults, latent_dim, n_res_blocks, dropout)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) images in [-1, 1]
        Returns:
            z_q:    (B, D, h, w)  quantized latent map
            vq_loss: scalar
            indices:(B, h, w)     discrete code indices
        """
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z)
        return z_q, vq_loss, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode directly from discrete index map."""
        z_q = self.quantizer.decode_indices(indices)
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Full VQ-VAE forward pass.

        Returns:
            x_rec:   (B, C, H, W) reconstructed image
            loss:    scalar total loss
            metrics: dict with 'rec_loss' and 'vq_loss'
        """
        z_q, vq_loss, _ = self.encode(x)
        x_rec = self.decode(z_q)
        rec_loss = F.mse_loss(x_rec, x)
        loss = rec_loss + vq_loss
        return x_rec, loss, {"rec_loss": rec_loss.item(), "vq_loss": vq_loss.item()}
