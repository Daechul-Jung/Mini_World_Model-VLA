"""Bottleneck adapters and gated residuals -- the reference `PolicyModule`s.

Not research contributions; they are the two shapes almost every new idea ends up
needing, and they demonstrate the identity-at-init rule concretely. Copy one when
starting a new module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..core.registry import MODULES
from .base import PolicyModule


@MODULES.register("bottleneck_adapter", paper="Houlsby et al., 2019", status="baseline")
class BottleneckAdapter(PolicyModule):
    """Down-project, non-linearity, up-project, residual.

    The up-projection is zero-initialised, so at step 0 this is exactly the
    identity and a frozen pretrained backbone keeps its behaviour. That is what
    makes it safe to bolt onto OpenVLA or pi0 without destroying the pretraining.

    Cost: `2 * dim * bottleneck` parameters per instance -- roughly 0.1% of a 7B
    backbone at `bottleneck=64`, which is why this is the affordable way to
    adapt a large VLA on an 8 GB card.
    """

    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.0, scale: float = 1.0):
        super().__init__(dim)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)
        self.scale = scale

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        h = self.up(self.drop(self.act(self.down(self.norm(features)))))
        return features + self.scale * h


@MODULES.register("gated_residual", status="baseline")
class GatedResidual(PolicyModule):
    """A transformer block whose contribution is gated by a learned scalar.

    `features + tanh(gate) * block(features)`, with `gate` initialised to 0.
    Use this when the new idea needs real capacity rather than a bottleneck, and
    you still want the identity-at-init guarantee. Watch the gate value during
    training: if it stays near zero, the module is not earning its parameters --
    that is a kill signal, and a cheap one.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__(dim)
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        t = features.shape[1]
        mask = (
            torch.triu(torch.ones(t, t, device=features.device, dtype=torch.bool), diagonal=1)
            if self.causal
            else None
        )
        h = self.norm1(features)
        h = self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = h + self.mlp(self.norm2(features + h))
        return features + torch.tanh(self.gate) * h

    @torch.no_grad()
    def gate_value(self) -> float:
        """Log this. A gate that never leaves ~0 means the idea is not helping."""
        return float(torch.tanh(self.gate))
