"""Goal-image reward: embedding similarity to a target frame.

The cheapest automatic reward that is not obviously wrong, and the baseline every
fancier idea has to beat. No training: freeze a visual encoder, embed the goal
frame once, and reward the negative distance to it at each step.

Where the goal frame comes from, in descending order of how much it costs you:

1. **The last frame of a demonstration.** Free -- `data/openx/` episodes end in
   the success state. Works for the task the demo shows, and nothing else.
2. **A held-out success image** you provide.
3. **A generated goal**: prompt the world model itself, or an image model, for
   "what success looks like". Interesting and circular -- the reward then depends
   on the same generative model being optimised against.

**The known failure mode**, and it is severe: a frozen ImageNet/CLIP encoder is
dominated by global scene appearance. A policy that never touches the object but
moves the camera so the scene *looks* like the goal frame scores well. Two
partial defences, both implemented here: crop to a region of interest, and use
`delta` mode so only *improvement* is rewarded rather than absolute similarity.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.types import Observation

from .base import REWARDS, RewardModel


@REWARDS.register("goal_image", status="baseline", note="no training required")
class GoalImageReward(RewardModel):
    """Negative embedding distance to a goal image.

    Args:
        encoder: any module mapping (B, 3, H, W) -> (B, D). Defaults to a frozen
            ResNet-18 trunk. Pass the world model's *tokenizer* encoder instead
            to reward proximity in the same space the dynamics model operates in
            -- a cheap and interesting variant, because it removes the
            distribution gap between real and imagined frames entirely.
        mode: `"absolute"` rewards similarity each step; `"delta"` rewards the
            improvement over the previous step, which is denser and less prone to
            the standing-still-in-a-good-looking-pose failure.
        goal_image: (1, 3, H, W) or (B, 3, H, W). If None, `reset` uses the last
            context frame, which is only correct when the context ends at success.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        mode: str = "delta",
        scale: float = 1.0,
        goal_image: Optional[torch.Tensor] = None,
        instruction: Optional[str] = None,
        crop: Optional[Tuple[int, int, int, int]] = None,
    ):
        super().__init__()
        self.encoder = encoder or self._default_encoder()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        self.mode = mode
        self.scale = scale
        self.crop = crop
        self.instruction = instruction
        self.register_buffer("goal_embedding", torch.zeros(1, 1), persistent=False)
        self._prev: Optional[torch.Tensor] = None
        if goal_image is not None:
            self.set_goal(goal_image)

    @staticmethod
    def _default_encoder() -> nn.Module:
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        return model.eval()

    # ------------------------------------------------------------------- goals

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        if self.crop:
            top, left, height, width = self.crop
            images = images[..., top : top + height, left : left + width]
        images = (images.clamp(-1, 1) + 1) / 2                # encoder expects [0, 1]
        return F.normalize(self.encoder(images), dim=-1)

    @torch.no_grad()
    def set_goal(self, goal_image: torch.Tensor) -> None:
        self.goal_embedding = self._embed(goal_image)

    def reset(self, context_frames: torch.Tensor) -> None:
        """Default goal: the last context frame. Override by calling `set_goal`."""
        if self.goal_embedding.numel() <= 1:
            self.set_goal(context_frames[:, -1])
        self._prev = None

    # ------------------------------------------------------------------ reward

    @torch.no_grad()
    def forward(
        self, obs: Observation, latents: Optional[torch.Tensor] = None, step: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        current = self._embed(obs.image[:, -1])
        similarity = (current * self.goal_embedding).sum(-1)   # cosine, in [-1, 1]

        if self.mode == "delta":
            reward = (
                torch.zeros_like(similarity)
                if self._prev is None
                else (similarity - self._prev)
            )
            self._prev = similarity
        else:
            reward = similarity

        # Distance from the goal manifold as a crude confidence proxy: a frame
        # that resembles nothing in the goal set is one this reward is guessing on.
        uncertainty = (1.0 - similarity.abs()).mean()

        return reward * self.scale, {
            "similarity": float(similarity.mean()),
            "uncertainty": float(uncertainty),
        }
