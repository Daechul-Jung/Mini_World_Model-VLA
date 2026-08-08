"""Action heads: features -> actions, and the loss that trains them.

The head owns the loss, not the trainer. That is deliberate -- these four
parameterisations need four different objectives over the same data:

| Head             | Output           | Loss                  | Used by      |
|------------------|------------------|-----------------------|--------------|
| `continuous_mse` | mean action      | MSE / L1 / Huber      | Octo-small   |
| `discrete_bins`  | per-dim logits   | cross-entropy         | RT-1, OpenVLA|
| `diffusion`      | denoiser         | score matching        | Octo-base    |
| `flow_matching`  | velocity field   | flow matching         | pi0          |

Which one matters more than it looks. Multi-modal demonstrations (two valid ways
to grasp a cup) average to an invalid action under MSE -- the classic "reaches
between the two grasp points" failure. Discrete bins and diffusion both handle
that; MSE does not. On the other hand MSE trains in minutes and diffusion needs a
sampling loop at inference. Start with MSE, and when the policy stalls at a
plausible-but-wrong action, suspect the head before the backbone.

`sample()` is separate from `forward()` because RL post-training needs a
stochastic action *with a log-probability*, which a deterministic regression head
cannot give without an explicit distribution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class ActionHead(nn.Module, ABC):
    """Maps (B, T, D) features to an action chunk, and scores targets."""

    def __init__(self, dim: int, action_dim: int, action_chunk: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.action_chunk = action_chunk

    @abstractmethod
    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """features (B, T, D) -> actions (B, chunk, action_dim), normalised space.

        Only the last timestep's features drive the prediction; earlier ones are
        history. Heads that need the full sequence take it anyway.
        """

    @abstractmethod
    def loss(
        self,
        features: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """target_actions: (B, chunk, action_dim) already normalised."""

    def sample(
        self, features: torch.Tensor, temperature: float = 1.0, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Stochastic action plus log-probability, for RL.

        Default: deterministic, no log-prob. Any head used with a policy-gradient
        algorithm must override this -- `stage_rl` refuses to run otherwise,
        rather than silently optimising a constant.
        """
        return self.forward(features, **kwargs), None

    @property
    def supports_rl(self) -> bool:
        """True when `sample` returns a usable log-probability."""
        return False
