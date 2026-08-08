"""Weighted combination of reward models, with disagreement as uncertainty.

Two reasons this exists rather than a single reward:

1. **Reward hacking becomes visible.** A policy exploiting one reward's blind
   spot usually does not exploit another's. When rewards that normally agree
   start diverging, that is the signature -- `disagreement` is logged for exactly
   this, and `disagreement_penalty` lets you subtract it from the total so the
   policy is actively discouraged from finding those regions.

2. **Uncertainty comes for free.** Ensemble variance is the standard cheap
   epistemic uncertainty estimate, and the world-model env uses it to decide how
   far to trust a rollout.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from common.types import Observation

from .base import REWARDS, RewardModel


@REWARDS.register("ensemble", status="baseline")
class EnsembleReward(RewardModel):
    """Weighted sum of component rewards.

    Args:
        components: list of registry configs, each built into a `RewardModel`.
        weights: one per component; defaults to uniform.
        disagreement_penalty: subtract `penalty * std` from the total. Set > 0
            when optimising against learned rewards inside a learned world model
            -- it makes the policy prefer regions where the reward models agree,
            which are the regions where they were trained.
        normalize: z-score each component using a running estimate before
            weighting. Without it, one component with a naturally larger scale
            silently dominates regardless of its weight.
    """

    def __init__(
        self,
        components: Sequence[Dict[str, Any]],
        weights: Optional[Sequence[float]] = None,
        disagreement_penalty: float = 0.0,
        normalize: bool = True,
        instruction: Optional[str] = None,
    ):
        super().__init__()
        self.components = nn.ModuleList([REWARDS.build(dict(c)) for c in components])
        n = len(self.components)
        self.register_buffer("weights", torch.tensor(list(weights or [1.0 / n] * n)))
        self.disagreement_penalty = disagreement_penalty
        self.normalize = normalize
        self.instruction = instruction

        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.momentum = 0.99

    def reset(self, context_frames: torch.Tensor) -> None:
        for component in self.components:
            component.reset(context_frames)

    def forward(
        self, obs: Observation, latents: Optional[torch.Tensor] = None, step: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rewards, infos = [], {}
        for i, component in enumerate(self.components):
            r, info = component(obs, latents=latents, step=step)
            rewards.append(r)
            for k, v in info.items():
                infos[f"{type(component).__name__}/{k}"] = v
        stacked = torch.stack(rewards, dim=0)                # (n, B)

        if self.normalize and self.training:
            with torch.no_grad():
                batch_mean, batch_var = stacked.mean(1), stacked.var(1, unbiased=False)
                self.running_mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
                self.running_var.mul_(self.momentum).add_(batch_var, alpha=1 - self.momentum)
        if self.normalize:
            stacked = (stacked - self.running_mean[:, None]) / self.running_var.sqrt()[:, None].clamp_min(1e-6)

        total = (self.weights[:, None] * stacked).sum(0)
        disagreement = stacked.std(0) if len(self.components) > 1 else torch.zeros_like(total)
        if self.disagreement_penalty:
            total = total - self.disagreement_penalty * disagreement

        return total, {
            "disagreement": float(disagreement.mean()),
            "uncertainty": float(disagreement.mean()),
            **infos,
        }
