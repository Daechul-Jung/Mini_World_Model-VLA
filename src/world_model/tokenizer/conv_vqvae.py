"""Per-frame convolutional VQ-VAE tokenizer (stage A, baseline).

This is the simplest thing that satisfies `VideoTokenizer`: every frame is
tokenised independently, so tokens carry no temporal context. Genie instead uses
an ST-transformer VQ-VAE whose tokens see neighbouring frames -- see
`st_vqvae.py` and `research/002_st_transformer_tokenizer.md`.

Keep this one around regardless: it trains on *static images* (LSUN rooms), so it
is the only stage-A option when no video is available, and it is the fastest
baseline to A/B a new quantizer against.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.core.base import LatentSpec, VideoTokenizer
from world_model.core.registry import QUANTIZERS, TOKENIZERS

from .quantizers.vq import VectorQuantizer


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
# Tokenizer (satisfies world_model.core.base.VideoTokenizer)
# ---------------------------------------------------------------------------

@TOKENIZERS.register(
    "conv_vqvae",
    status="baseline",
    note="per-frame, no temporal context; trainable on static images",
)
class ConvVQVAETokenizer(VideoTokenizer):
    """Frame-independent VQ-VAE.

    The quantizer is itself pluggable via `quantizer={"name": "fsq", ...}` --
    swapping VQ for FSQ/LFQ is the cheapest fix for codebook collapse and needs
    no change here.

    Args:
        in_channels: image channels.
        base_channels: width before multipliers.
        channel_mults: one entry per 2x downsample stage. len(mults) sets the
            token grid: 256px with 3 stages -> 32x32 = 1024 tokens/frame.
        latent_dim: per-token embedding width D.
        quantizer: registry config; defaults to plain VQ with `codebook_size`.
        codebook_size: convenience shorthand for the default quantizer.
        image_size: only used to precompute `latent_spec.grid`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        latent_dim: int = 256,
        n_res_blocks: int = 2,
        dropout: float = 0.0,
        image_size: int = 256,
        codebook_size: int = 1024,
        quantizer: Dict | None = None,
        perceptual_weight: float = 0.0,
    ):
        super().__init__()
        channel_mults = tuple(channel_mults)
        self.encoder = Encoder(in_channels, base_channels, channel_mults, latent_dim, n_res_blocks, dropout)
        self.decoder = Decoder(in_channels, base_channels, channel_mults, latent_dim, n_res_blocks, dropout)

        quantizer = dict(quantizer or {"name": "vq", "beta": 0.25})
        quantizer.setdefault("num_embeddings", codebook_size)
        quantizer.setdefault("embedding_dim", latent_dim)
        self.quantizer = QUANTIZERS.build(quantizer)

        self.perceptual_weight = perceptual_weight
        downsample = 2 ** len(channel_mults)
        grid = image_size // downsample
        self._spec = LatentSpec(
            grid=(grid, grid),
            dim=latent_dim,
            discrete=True,
            vocab_size=getattr(self.quantizer, "num_embeddings", codebook_size),
        )

    @property
    def latent_spec(self) -> LatentSpec:
        return self._spec

    # ------------------------------------------------------------------ encode

    def encode(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        squeeze_time = frames.ndim == 4
        if squeeze_time:                       # (B, 3, H, W) -> (B, 1, 3, H, W)
            frames = frames.unsqueeze(1)
        b, t = frames.shape[:2]
        flat = frames.flatten(0, 1)
        z_q, aux_loss, indices = self.quantizer(self.encoder(flat))
        out = {
            "latents": z_q.unflatten(0, (b, t)),
            "indices": indices.unflatten(0, (b, t)),
            "aux_loss": aux_loss,
        }
        if squeeze_time:
            out["latents"] = out["latents"].squeeze(1)
            out["indices"] = out["indices"].squeeze(1)
        return out

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        squeeze_time = latents.ndim == 4
        if squeeze_time:
            latents = latents.unsqueeze(1)
        b, t = latents.shape[:2]
        frames = self.decoder(latents.flatten(0, 1)).unflatten(0, (b, t))
        return frames.squeeze(1) if squeeze_time else frames

    def indices_to_latents(self, indices: torch.Tensor) -> torch.Tensor:
        squeeze_time = indices.ndim == 3
        if squeeze_time:
            indices = indices.unsqueeze(1)
        b, t = indices.shape[:2]
        z_q = self.quantizer.decode_indices(indices.flatten(0, 1)).unflatten(0, (b, t))
        return z_q.squeeze(1) if squeeze_time else z_q

    # ----------------------------------------------------------------- stage A

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Reconstruction pass. Accepts (B, 3, H, W) or (B, T, 3, H, W)."""
        enc = self.encode(frames)
        recon = self.decode(enc["latents"])
        rec_loss = F.mse_loss(recon, frames)
        loss = rec_loss + enc["aux_loss"]
        metrics = {"rec_loss": rec_loss.detach().item(), "vq_loss": float(enc["aux_loss"].detach())}
        return recon, loss, metrics
