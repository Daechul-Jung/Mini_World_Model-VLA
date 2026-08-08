"""Automatic reward functions for imagined rollouts.

The open problem you flagged: *how do you get a reward signal when there is no
simulator and no ground truth?* Inside a world model there is no object pose, no
contact sensor, no success flag -- only generated pixels. Every reward here is
therefore an approximation, and the honest framing is that this package is a
menu of approximations with different failure modes, not a solution.

Four families, cheapest first:

| Reward           | Signal                            | Fails when                    |
|------------------|-----------------------------------|-------------------------------|
| `goal_image`     | embedding distance to a goal frame| goal frames are unavailable, or the embedding is fooled by background |
| `progress`       | time-contrastive value (VIP/LIV)  | trained on data unlike the imagined frames |
| `vlm_judge`      | a VLM scores "did it pick it up?" | slow; VLMs are unreliable on blurry generated frames |
| `dynamics_prior` | world-model likelihood            | rewards *predictable*, not *successful*, behaviour |

Two rules the contract enforces:

1. **Every reward reports uncertainty.** A reward model queried far outside its
   training distribution -- which is exactly where an RL policy will drive it --
   must say so. `ensemble.py` gets this from disagreement; single models from a
   density estimate or a fixed constant.

2. **Rewards are composed, not chosen.** `EnsembleReward` combines several with
   weights. A single learned reward on generated pixels is the most reward-
   hackable object in this repo; two disagreeing rewards at least make hacking
   visible.

See `research/015_automatic_rewards.md` for the design space and what to try
first (short answer: `goal_image` with a frozen encoder, because it needs no
extra training and gives a baseline the others must beat).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from common.registry import Registry
from common.types import Observation

REWARDS = Registry("bridge.reward")


class RewardModel(nn.Module, ABC):
    """Scores an imagined observation."""

    #: human-readable task description, forwarded to the policy as instruction
    instruction: Optional[str] = None

    @abstractmethod
    def forward(
        self, obs: Observation, latents: Optional[torch.Tensor] = None, step: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return `(reward (B,), info)`.

        `info` **must** carry an `uncertainty` key -- a per-batch scalar or tensor
        that grows as the input leaves the reward model's training distribution.
        The RL loop uses it to down-weight or terminate untrustworthy rollouts.
        """

    def reset(self, context_frames: torch.Tensor) -> None:
        """Called at `env.reset` with the real prompt frames.

        Goal-conditioned rewards use this to fix the goal; progress rewards use
        it to anchor step 0.
        """

    @property
    def is_learned(self) -> bool:
        """True if this reward has trained parameters that could be hacked."""
        return any(p.requires_grad for p in self.parameters())
