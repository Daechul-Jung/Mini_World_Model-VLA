"""Finite Scalar Quantization -- a codebook that cannot collapse.

Mentzer et al., 2023 (arXiv:2309.15505). Implements
`src/world_model/research/003_quantizer_alternatives.md`.

The idea: drop the learned codebook entirely. Project to a handful of channels
(`len(levels)`, typically 4-6), squash each into a bounded range, and round each
independently to one of `L` levels. The "codebook" is the implicit product grid
`prod(levels)`, and every entry is reachable by construction -- so there are no
dead codes, no commitment/codebook loss, no EMA, and no restart heuristic.

Trade-off versus a learned codebook: the code geometry is a fixed axis-aligned
grid rather than something fitted to the data, so at equal vocabulary size FSQ
usually reconstructs slightly worse than a *healthy* VQ. It reliably beats a
*collapsed* one, which is the situation that actually arises.

`num_embeddings` is derived from `levels`, not chosen freely -- the tokenizer
reads it back through `LatentSpec.vocab_size`, so the dynamics model's vocabulary
follows automatically. Some useful settings:

    [8, 8, 8, 5, 5]  -> 8000    (paper's recommendation for ~2^13)
    [8, 5, 5, 5]     -> 1000    (close to a 1024 codebook)
    [4, 4, 4, 4, 4]  -> 1024    (exactly 1024)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from world_model.core.registry import QUANTIZERS


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient."""
    return z + (torch.round(z) - z).detach()


@QUANTIZERS.register(
    "fsq",
    paper="Mentzer et al., 2023 (arXiv:2309.15505)",
    status="recommended",
    note="no learned codebook, so no collapse; vocab = prod(levels)",
)
class FSQuantizer(nn.Module):
    """Finite scalar quantization with 1x1 projections to and from `embedding_dim`.

    Args:
        embedding_dim: the tokenizer's latent width D. FSQ operates on a much
            smaller `len(levels)`-dim space, so 1x1 convolutions bracket it.
        levels: quantization levels per FSQ channel.
        num_embeddings: ignored if given; `prod(levels)` wins. Accepted only so
            that a config written for `vq` can switch quantizer without also
            having to delete the key.
    """

    def __init__(
        self,
        embedding_dim: int,
        levels: Sequence[int] = (8, 5, 5, 5),
        num_embeddings: int | None = None,
        **_ignored,
    ):
        super().__init__()
        levels = list(levels)
        self.embedding_dim = embedding_dim
        self.n_channels = len(levels)
        self.num_embeddings = int(torch.tensor(levels).prod())

        if num_embeddings is not None and num_embeddings != self.num_embeddings:
            print(
                f"[fsq] ignoring codebook_size={num_embeddings}; the implicit "
                f"codebook is prod({levels}) = {self.num_embeddings}"
            )

        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        basis = torch.cat([torch.ones(1), torch.tensor(levels[:-1], dtype=torch.float32).cumprod(0)])
        self.register_buffer("basis", basis)

        self.proj_in = nn.Conv2d(embedding_dim, self.n_channels, 1)
        self.proj_out = nn.Conv2d(self.n_channels, embedding_dim, 1)

    # ---------------------------------------------------------------- quantize

    def _bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Squash each channel into the open interval its level grid spans."""
        half_l = (self.levels - 1) * (1 - eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        shaped = (1, -1, 1, 1)
        return torch.tanh(z + shift.view(shaped)) * half_l.view(shaped) - offset.view(shaped)

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        """-> quantized values renormalised to roughly [-1, 1]."""
        quantized = round_ste(self._bound(z))
        half_width = (self.levels // 2).view(1, -1, 1, 1)
        return quantized / half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, n_channels, H, W) normalised codes -> (B, H, W) integer indices."""
        half_width = (self.levels // 2).view(1, -1, 1, 1)
        digits = (codes * half_width) + half_width                # -> [0, L-1]
        return (digits * self.basis.view(1, -1, 1, 1)).sum(1).round().long()

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> (B, n_channels, H, W) normalised codes."""
        idx = indices.unsqueeze(1).float()
        basis = self.basis.view(1, -1, 1, 1)
        levels = self.levels.view(1, -1, 1, 1)
        digits = torch.floor(idx / basis) % levels                # mixed radix
        half_width = self.levels.div(2, rounding_mode="floor").view(1, -1, 1, 1)
        return (digits - half_width) / half_width

    # ----------------------------------------------------------------- forward

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z: (B, D, H, W) -> (z_q (B, D, H, W), loss, indices (B, H, W)).

        The returned loss is exactly zero: FSQ needs no commitment term, which is
        one fewer weight to tune and one fewer way for stage A to go wrong.
        """
        codes = self._quantize(self.proj_in(z))
        indices = self.codes_to_indices(codes)
        z_q = self.proj_out(codes)
        return z_q, torch.zeros((), device=z.device, dtype=z.dtype), indices

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> (B, D, H, W)."""
        squeeze = indices.ndim == 2
        if squeeze:                                   # (B, N) -> treat as 1 x N
            indices = indices.unsqueeze(1)
        z_q = self.proj_out(self.indices_to_codes(indices))
        return z_q.squeeze(2).transpose(1, 2) if squeeze else z_q
