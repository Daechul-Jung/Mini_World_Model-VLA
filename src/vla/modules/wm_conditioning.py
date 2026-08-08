"""World-model conditioning -- the first VLA <- world-model idea slot.

The project's stated hypothesis in both directions is that the two models help
each other. This module is the *world model helps the VLA* direction: give the
policy access to what the world model predicts will happen, so it can act on a
short lookahead rather than on the current frame alone.

    obs_t --> world model --> imagined latents for t+1..t+k
                                     |
    obs_t --> VLA backbone --> features --> [ this module ] --> head --> action
                                    cross-attends to the imagination

Three variants worth separating, because they make different claims:

1. **Passive lookahead** (implemented here). Roll the world model forward under
   the policy's *current* action and cross-attend to the result. Claim: features
   that encode consequences beat features that encode only appearance.
2. **Counterfactual lookahead**. Roll forward under several candidate actions and
   attend to all of them -- a learned, differentiable one-step planner. More
   expensive, stronger claim.
3. **Representation transfer**. Skip rollouts entirely; use the world model's
   *tokenizer* as a frozen visual encoder. Cheapest, and the honest baseline that
   variants 1 and 2 must beat before the rollout cost is justified.

Run variant 3 first. If a frozen tokenizer as an encoder already captures the
gain, the imagination machinery is not what is helping.

**Cost warning.** Every forward pass now includes `k` world-model steps. At
`k=4` with a 32x32 grid and tokenizer rendering, that is roughly 4 extra
transformer forwards per training step -- acceptable. With diffusion rendering it
is ~100, which is not. Keep `render="tokenizer"` here.

See `research/010_world_model_conditioned_vla.md`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..core.registry import MODULES
from .base import PolicyModule


@MODULES.register(
    "wm_conditioning",
    status="idea",
    note="cross-attend policy features to world-model imagination",
)
class WorldModelConditioning(PolicyModule):
    """Cross-attention from policy features to imagined world-model latents.

    The world model is supplied through `context["wm_latents"]` rather than held
    as a submodule, so this module never owns or trains it and the same instance
    works with any world-model checkpoint. `src/bridge/` is what populates that
    key.

    Args:
        dim: policy feature width.
        latent_dim: world-model latent width (`GenieWorldModel.latent_spec.dim`).
        num_heads: cross-attention heads.

    Identity at init: the output projection is zero-initialised, so an untrained
    module leaves the pretrained policy untouched.
    """

    required_context = ("wm_latents",)

    def __init__(self, dim: int, latent_dim: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__(dim)
        self.to_kv = nn.Linear(latent_dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        ctx = self.check_context(context)
        latents = ctx["wm_latents"]                      # (B, k, D_lat, h, w)

        b = latents.shape[0]
        kv = latents.flatten(3).permute(0, 1, 3, 2).reshape(b, -1, latents.shape[2])
        kv = self.norm_kv(self.to_kv(kv))                # (B, k*h*w, D)

        q = self.norm_q(features)
        attended = self.attn(q, kv, kv, need_weights=False)[0]
        return features + torch.tanh(self.gate) * self.out(attended)

    @torch.no_grad()
    def gate_value(self) -> float:
        return float(torch.tanh(self.gate))
