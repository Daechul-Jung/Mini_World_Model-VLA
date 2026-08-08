"""Translating between a VLA's action space and a world model's.

**This is the hard part of the whole VLA x world-model plan, and it is worth
being explicit about why.**

A Genie-style world model is driven by |A| = 8 discrete latent codes, discovered
without supervision, that mean whatever the video data made them mean -- typically
things like "camera pans left" or "the visible arm moves down". A VLA emits a
continuous 4-to-7-DoF robot action in physical units. These are not the same
space, they do not have the same dimensionality, and nothing in either model's
training connects them.

So "use the world model as an environment for the VLA" requires a translation
layer, and which one you pick is a research decision, not plumbing. Three
options, in increasing order of cost and fidelity:

1. **`LearnedActionProjector`** -- train a small map from robot actions to latent
   codes on paired data. Needs a dataset with both video and real actions, which
   `data/openx/` has. Cheap, and honest about what it is: a lookup from action to
   "which of 8 things this looked like".

2. **`RobotConditionedDynamics`** (the better answer, and not in this file) --
   skip latent actions entirely. Train stage C with `action_kind="robot"` on
   OpenX episodes so the dynamics model is conditioned on true robot actions from
   the start. No translation needed, the action space is exact, and the cost is
   that you need action-labelled video, so it cannot use LSUN or unlabelled room
   footage. **For the robotics half of this project, prefer this.** Latent actions
   are the right tool for the room-video half, where no action labels exist.

3. **Both, in one model** -- a dynamics model conditioned on the concatenation of
   a latent code and a robot action, trained on a mix, with the missing modality
   dropped out. Lets unlabelled room video and labelled robot video train one
   model. This is the interesting version and it is unexplored here; see
   `research/011_unified_action_conditioning.md`.

The `ActionTranslator` contract below covers option 1 and makes option 3
expressible. Option 2 needs no translator at all, which is exactly why it is the
recommendation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.registry import Registry
from common.types import ActionSpec

ACTION_TRANSLATORS = Registry("bridge.action_translator")


class ActionTranslator(nn.Module, ABC):
    """Maps VLA actions into whatever the world model's dynamics accepts."""

    @abstractmethod
    def forward(self, robot_actions: torch.Tensor) -> torch.Tensor:
        """(B, action_dim) physical/normalised robot action -> world-model action.

        For a latent action space, returns (B,) int64 code indices.
        For a robot action space, this is the identity.
        """

    @property
    @abstractmethod
    def target_kind(self) -> str:
        """`"latent"` or `"robot"` -- must match `Dynamics.action_spec.kind`."""


@ACTION_TRANSLATORS.register("identity")
class IdentityTranslator(ActionTranslator):
    """For a dynamics model already conditioned on robot actions.

    Use this whenever stage C was trained with `action_kind="robot"`. It is the
    no-translation path, and the one to prefer for robotics.
    """

    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim

    @property
    def target_kind(self) -> str:
        return "robot"

    def forward(self, robot_actions: torch.Tensor) -> torch.Tensor:
        return robot_actions


@ACTION_TRANSLATORS.register("learned_projector")
class LearnedActionProjector(ActionTranslator):
    """Robot action -> latent code, as a small learned classifier.

    Trained by `fit()` on paired (robot action, LAM-inferred latent code) data
    from action-labelled video. Straight-through Gumbel-softmax at training time
    keeps it differentiable, so the projector can also be trained *through* the
    world model if you later want end-to-end gradients.

    **Expect information loss and measure it.** `fit()` reports the accuracy of
    predicting the LAM's code from the robot action. If that accuracy is near
    chance, the two action spaces are not aligned at all and option 2 above is
    the only sound route -- not a bigger projector.
    """

    def __init__(self, action_dim: int = 7, num_actions: int = 8, hidden: int = 128):
        super().__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, num_actions)
        )

    @property
    def target_kind(self) -> str:
        return "latent"

    def logits(self, robot_actions: torch.Tensor) -> torch.Tensor:
        return self.net(robot_actions)

    def forward(self, robot_actions: torch.Tensor) -> torch.Tensor:
        return self.logits(robot_actions).argmax(-1)

    def soft_forward(self, robot_actions: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Straight-through one-hot, for gradients through the world model."""
        return F.gumbel_softmax(self.logits(robot_actions), tau=tau, hard=True)

    def fit(
        self,
        robot_actions: torch.Tensor,
        latent_codes: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """Supervised fit against LAM-inferred codes on paired data.

        Args:
            robot_actions: (N, action_dim) normalised robot actions.
            latent_codes: (N,) int64, from `LatentActionModel.infer_actions` on
                the *same* transitions.
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(self.logits(robot_actions), latent_codes)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = self.forward(robot_actions)
            acc = float((pred == latent_codes).float().mean())
            chance = 1.0 / self.num_actions
        return {
            "fit_loss": loss.detach().item(),
            "code_accuracy": acc,
            "chance": chance,
            # The number that decides whether this approach is viable at all.
            "lift_over_chance": acc - chance,
        }


def build_translator(cfg: Dict[str, Any], world_model: Any) -> ActionTranslator:
    """Build a translator and check it matches the world model's action space."""
    translator = ACTION_TRANSLATORS.build(cfg)
    expected = world_model.action_spec.kind
    if translator.target_kind != expected:
        raise ValueError(
            f"translator produces '{translator.target_kind}' actions but the "
            f"dynamics model expects '{expected}'. Either retrain stage C with "
            f"action_kind='{translator.target_kind}' or pick a different translator."
        )
    return translator
