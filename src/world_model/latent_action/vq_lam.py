"""Latent Action Model (stage B) -- Genie's unsupervised action discovery.

The idea in one sentence: force the transition x_t -> x_{t+1} through a bottleneck
so narrow that only the *controllable* part of the change survives, then call
whatever survives "the action".

    encoder(x_{<=t}, x_{t+1}) -> a_t          (VQ bottleneck, |A| = 8 codes)
    decoder(x_{<=t}, a_t)     -> x_{t+1}      (training-only head)

Two details from the paper (arXiv:2402.15391) that are easy to get wrong:

* **Pixel input, not tokens.** Table 2 ablates this: a token-input LAM scores
  1.33 controllability vs 1.91 for pixel-input, because tokenisation discards
  the fine motion cues the LAM needs. That is why this module owns its own
  patch embedding rather than reusing the stage-A tokenizer.
* **Discarded at inference.** The encoder exists only to produce training targets
  for the dynamics model. At rollout time the caller supplies an integer in
  [0, |A|). Anything wanting to *drive* the world from robot actions therefore
  needs a translation layer -- `src/bridge/action_space.py`, not this file.

|A| = 8 is small on purpose: it keeps the codes interpretable (you can enumerate
all eight and see what each one does) and it keeps the dynamics model's action
vocabulary trivially learnable. It is also the main limitation for robotics --
8 discrete codes cannot express a 7-DoF continuous action. See
`research/006_continuous_latent_actions.md`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.core.base import ActionSpaceSpec, LatentActionModel
from world_model.core.registry import LATENT_ACTIONS, QUANTIZERS


class PatchEmbed(nn.Module):
    """Non-overlapping patch embedding. Genie uses patch size 16 for the LAM."""

    def __init__(self, in_channels: int = 3, patch_size: int = 16, dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) -> (B, T, n_patches, dim)."""
        b, t = x.shape[:2]
        z = self.proj(x.flatten(0, 1))                    # (B*T, dim, h, w)
        return z.flatten(2).transpose(1, 2).unflatten(0, (b, t))


class STBlock(nn.Module):
    """One spatiotemporal block: spatial attention, then temporal attention, then MLP.

    Spatial attention is bidirectional over the `n` patches within a frame.
    Temporal attention is causal over the `t` frames at a fixed patch position.
    This is the memory trick that makes video transformers affordable: cost goes
    from O((T*N)^2) to O(T*N^2 + T^2*N).
    """

    def __init__(self, dim: int, num_heads: int, causal_temporal: bool = True, dropout: float = 0.0):
        super().__init__()
        self.causal_temporal = causal_temporal
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N, D)."""
        b, t, n, d = x.shape

        h = self.norm_s(x).reshape(b * t, n, d)
        x = x + self.attn_s(h, h, h, need_weights=False)[0].reshape(b, t, n, d)

        h = self.norm_t(x).transpose(1, 2).reshape(b * n, t, d)
        mask = (
            torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            if self.causal_temporal
            else None
        )
        h = self.attn_t(h, h, h, attn_mask=mask, need_weights=False)[0]
        x = x + h.reshape(b, n, t, d).transpose(1, 2)

        return x + self.mlp(self.norm_m(x))


@LATENT_ACTIONS.register(
    "vq_lam",
    paper="arXiv:2402.15391",
    status="baseline",
    note="Genie LAM: pixel input, VQ bottleneck, discarded at inference",
)
class VQLatentActionModel(LatentActionModel):
    """Genie-style latent action model.

    Args:
        num_actions: |A|, the size of the action codebook. Genie uses 8.
        patch_size: LAM patch size (16 in the paper).
        dim / depth / num_heads: encoder and decoder trunk size. Both trunks are
            small -- the LAM's job is discovering *which* transitions exist, not
            rendering; stage A and stage D own image quality.
        action_dim: width of the action embedding handed to the dynamics model.
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        patch_size: int = 16,
        num_actions: int = 8,
        dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        action_dim: int = 128,
        dropout: float = 0.0,
        quantizer: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.n_patches = self.grid**2
        self.dim = dim
        self._spec = ActionSpaceSpec(kind="latent", num_actions=num_actions)

        # --- encoder: sees x_{<=t} AND x_{t+1} (non-causal in time on purpose) ---
        self.embed = PatchEmbed(in_channels, patch_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, 1, self.n_patches, dim))
        self.time_pos = nn.Parameter(torch.zeros(1, 64, 1, dim))
        self.encoder = nn.ModuleList(
            [STBlock(dim, num_heads, causal_temporal=False, dropout=dropout) for _ in range(depth)]
        )
        self.to_action = nn.Linear(dim, action_dim)

        # Default to `vq_ema`, not `vq`. The action codebook is the whole point
        # of this module and plain VQ has no way to revive a code once it stops
        # being selected -- collapse is an absorbing state. With |A| = 8 there is
        # very little margin: losing six codes leaves a world model that cannot
        # be steered, while `rec_loss` keeps falling because the decoder learns
        # to predict the next frame from the past alone. Dead-code restart is
        # what keeps the action channel alive.
        quantizer = dict(quantizer or {"name": "vq_ema", "beta": 0.25, "restart_every": 20})
        quantizer.setdefault("num_embeddings", num_actions)
        quantizer.setdefault("embedding_dim", action_dim)
        self.quantizer = QUANTIZERS.build(quantizer)

        # --- decoder: training-only head, predicts x_{t+1} from x_{<=t} + a_t ---
        self.dec_blocks = nn.ModuleList(
            [STBlock(dim, num_heads, causal_temporal=True, dropout=dropout) for _ in range(depth)]
        )
        self.action_to_dim = nn.Linear(action_dim, dim)
        self.to_pixels = nn.Linear(dim, patch_size * patch_size * in_channels)
        self.in_channels = in_channels

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)

    @property
    def action_spec(self) -> ActionSpaceSpec:
        return self._spec

    # ------------------------------------------------------------------ shared

    def _embed(self, frames: torch.Tensor) -> torch.Tensor:
        t = frames.shape[1]
        return self.embed(frames) + self.pos + self.time_pos[:, :t]

    # ---------------------------------------------------------------- encoding

    def infer_actions(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """frames: (B, T, 3, H, W) -> latent action for each of the T-1 transitions.

        The encoder is *non-causal* in time: predicting a_t is allowed to look at
        x_{t+1}, which is the whole point -- a_t is defined as what distinguishes
        x_{t+1} from what the past alone would predict.
        """
        x = self._embed(frames)
        for block in self.encoder:
            x = block(x)

        # One action vector per transition, pooled over patches.
        per_frame = self.to_action(x.mean(dim=2))           # (B, T, action_dim)
        transitions = per_frame[:, 1:]                       # a_t describes t -> t+1

        # The quantizer works on (B, C, H, W); present transitions as a 1-D grid.
        z = transitions.transpose(1, 2).unsqueeze(-1)        # (B, A, T-1, 1)
        z_q, aux_loss, indices = self.quantizer(z)
        return {
            "embeddings": z_q.squeeze(-1).transpose(1, 2),   # (B, T-1, action_dim)
            "indices": indices.squeeze(-1),                  # (B, T-1)
            "aux_loss": aux_loss,
        }

    # ----------------------------------------------------------------- stage B

    def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Predict x_{1..T-1} from x_{0..T-2} and the inferred latent actions.

        Reconstruction quality here is not the goal -- a LAM that reconstructs
        perfectly has probably widened the bottleneck and smuggled appearance
        through the action code. Watch `action_perplexity` instead: if it
        collapses toward 1, every transition is being labelled the same and the
        dynamics model will learn to ignore actions entirely.
        """
        b, t = frames.shape[:2]
        if t < 2:
            raise ValueError("latent action model needs at least 2 frames")

        act = self.infer_actions(frames)
        past = self._embed(frames[:, :-1])                            # (B, T-1, N, D)
        cond = self.action_to_dim(act["embeddings"]).unsqueeze(2)     # (B, T-1, 1, D)

        h = past + cond
        for block in self.dec_blocks:
            h = block(h)

        patches = self.to_pixels(h)                                   # (B, T-1, N, p*p*C)
        pred = self._unpatchify(patches)                              # (B, T-1, C, H, W)
        target = frames[:, 1:]

        rec_loss = F.mse_loss(pred, target)
        loss = rec_loss + act["aux_loss"]

        with torch.no_grad():
            counts = torch.bincount(
                act["indices"].flatten(), minlength=self._spec.num_actions
            ).float()
            probs = counts / counts.sum().clamp_min(1)
            perplexity = float((-(probs * (probs + 1e-10).log()).sum()).exp())

        return pred, loss, {
            "rec_loss": rec_loss.detach().item(),
            "vq_loss": float(act["aux_loss"].detach()),
            "action_perplexity": perplexity,
            "actions_used": float((counts > 0).sum()),
        }

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, T, N, p*p*C) -> (B, T, C, H, W)."""
        b, t, n, _ = patches.shape
        p, c, g = self.patch_size, self.in_channels, self.grid
        x = patches.reshape(b, t, g, g, p, p, c)
        x = x.permute(0, 1, 6, 2, 4, 3, 5)               # b t c g p g p
        return x.reshape(b, t, c, g * p, g * p)

    # -------------------------------------------------------------- inspection

    @torch.no_grad()
    def action_codebook(self) -> torch.Tensor:
        """The |A| action embeddings. Sweep these to see what each code *does*:
        hold a context clip fixed, roll the dynamics model forward under each
        code, and watch which direction the camera or gripper moves.

        Quantizers store the codebook differently -- `vq` in an `nn.Embedding`,
        `vq_ema` in a buffer -- so read whichever exists rather than assuming.
        """
        for attr in ("codebook", "embedding"):
            book = getattr(self.quantizer, attr, None)
            if book is not None:
                return (book.weight if hasattr(book, "weight") else book).detach()
        raise AttributeError(
            f"{type(self.quantizer).__name__} has no explicit codebook to sweep "
            "(FSQ's grid is implicit). Enumerate indices 0..num_actions-1 instead."
        )
