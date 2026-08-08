"""EMA vector quantizer with data-dependent init and dead-code restart.

This is the fix for the collapse that plain `vq` exhibits on this data (LSUN
rooms: 13 of 1024 codes alive after 2 epochs, perplexity 4.5, and *falling*).

Three changes, each addressing a distinct cause:

1. **Data-dependent initialisation.** `vq` initialises the codebook to
   `uniform(-1/K, 1/K)` -- with K = 1024 that is a range of +/-0.001, so every
   code starts essentially at the origin while encoder outputs have unit-ish
   scale. Nearest-neighbour assignment among 1024 near-identical vectors is
   arbitrary, and the handful that drift out are the only ones that ever get
   used. Here the codebook is instead seeded from the first batch's actual
   encoder outputs, so codes start spread across the data.

2. **EMA codebook updates** (van den Oord et al., appendix A.1). The codebook is
   a buffer updated by an exponential moving average of the encoder vectors
   assigned to it, not a parameter trained by gradient descent. This removes the
   codebook loss term and is markedly more stable -- it is what most modern
   VQ-VAEs use.

3. **Dead-code restart.** A code that is never selected receives no gradient and
   no EMA update, so it can never come back on its own -- collapse is an
   absorbing state. Every `restart_every` steps, codes whose usage has decayed
   below `restart_threshold` are re-seeded to randomly chosen encoder vectors
   from the current batch. This is the single most effective of the three.

Keeps `num_embeddings` and the `(z_q, loss, indices)` interface identical to
`vq`, so swapping is one config line and nothing downstream changes.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.core.registry import QUANTIZERS


@QUANTIZERS.register(
    "vq_ema",
    paper="van den Oord et al., 2017 (appendix A.1)",
    status="recommended",
    note="EMA updates + data-dependent init + dead-code restart",
)
class EMAVectorQuantizer(nn.Module):
    """Vector quantization with an EMA-updated codebook.

    Args:
        num_embeddings: codebook size K.
        embedding_dim: code width D.
        beta: commitment loss weight. Only the commitment term remains -- the
            codebook term is replaced by the EMA update.
        decay: EMA rate for codebook and cluster sizes. 0.99 is standard; lower
            adapts faster but is noisier.
        epsilon: Laplace smoothing on cluster sizes, guarding a divide-by-zero
            for a code claimed by nothing this batch.
        restart_every: steps between dead-code sweeps. 0 disables restarts.
        restart_threshold: EMA cluster size below which a code counts as dead.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        restart_every: int = 100,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon
        self.restart_every = restart_every
        self.restart_threshold = restart_threshold

        # Buffers, not parameters: the codebook is updated by EMA, not by the
        # optimiser. Registering them as buffers also means they are saved in and
        # restored from the checkpoint, which a plain attribute would not be.
        self.register_buffer("codebook", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------- setup

    @torch.no_grad()
    def _data_init(self, flat: torch.Tensor) -> None:
        """Seed the codebook from real encoder outputs on the first batch."""
        flat = flat.to(self.codebook.dtype)
        n = flat.shape[0]
        if n >= self.num_embeddings:
            pick = torch.randperm(n, device=flat.device)[: self.num_embeddings]
            codes = flat[pick]
        else:
            # Fewer vectors than codes: tile and jitter so codes stay distinct.
            reps = self.num_embeddings // n + 1
            codes = flat.repeat(reps, 1)[: self.num_embeddings]
            codes = codes + 0.01 * torch.randn_like(codes)

        self.codebook.copy_(codes)
        self.embed_avg.copy_(codes)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _restart_dead(self, flat: torch.Tensor) -> int:
        """Re-seed unused codes from the current batch. Returns how many."""
        flat = flat.to(self.codebook.dtype)
        dead = self.cluster_size < self.restart_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return 0
        pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        fresh = flat[pick]
        self.codebook[dead] = fresh
        self.embed_avg[dead] = fresh
        self.cluster_size[dead] = 1.0
        return n_dead

    # ----------------------------------------------------------------- forward

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z: (B, C, H, W) -> (z_q (B, C, H, W), loss, indices (B, H, W)).

        Runs under bf16 autocast in training, so `z` arrives in bf16 while the
        codebook and its EMA statistics are kept in fp32. That split is
        deliberate -- an EMA with decay 0.99 accumulates over hundreds of steps
        and bf16 has ~8 bits of mantissa, so the running averages would quantise
        away. Every write into an fp32 buffer therefore casts explicitly; an
        index-put with mismatched dtypes is a hard error, not a silent promotion.
        """
        b, c, h, w = z.shape
        z_bhwc = z.permute(0, 2, 3, 1).contiguous()
        flat = z_bhwc.view(-1, self.embedding_dim)

        if self.training and not bool(self.initialized):
            self._data_init(flat.detach())

        codebook = self.codebook.to(flat.dtype)
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(1)
        )
        indices = distances.argmin(1)
        z_q = codebook[indices].view(z_bhwc.shape)

        if self.training:
            self._ema_update(flat.detach(), indices)

        # Commitment only -- the EMA update replaces the codebook loss term.
        loss = self.beta * F.mse_loss(z_q.detach(), z_bhwc)

        z_q = z_bhwc + (z_q - z_bhwc).detach()          # straight-through
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, indices.view(b, h, w)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        flat = flat.to(self.codebook.dtype)
        onehot = F.one_hot(indices, self.num_embeddings).to(self.codebook.dtype)

        counts = onehot.sum(0)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)

        # Laplace smoothing keeps a momentarily-unclaimed code from dividing by 0.
        n = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )
        self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))

        self.step += 1
        if self.restart_every and int(self.step) % self.restart_every == 0:
            self._restart_dead(flat)

    # ------------------------------------------------------------------ decode

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """(B, H, W) -> (B, C, H, W), or (B, N) -> (B, N, C)."""
        z_q = self.codebook[indices.reshape(-1)]
        if indices.ndim == 3:
            b, h, w = indices.shape
            return z_q.view(b, h, w, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
        if indices.ndim == 2:
            return z_q.view(*indices.shape, self.embedding_dim)
        return z_q
