"""Physics and scene-context priors: the `PhysicsHead` slot.

Genie 3 uses no physics engine -- physical plausibility is learned from video
alone. That works at DeepMind's data scale. At 5k LSUN images and a few TUM RGB-D
sequences it will not: pure next-token prediction on that much data learns
texture statistics long before it learns that objects are solid.

So this slot exists to *inject* the structure the data cannot supply, as
auxiliary heads on the dynamics model's features. They cost training compute and
zero inference compute, because they are dropped at rollout time.

Candidates, cheapest first:

* **Depth** -- TUM RGB-D ships ground-truth depth for every frame. Predicting it
  from dynamics features is free supervision for 3D structure, and it is the
  single highest-value auxiliary task available with the data already on disk.
* **Camera pose / ego-motion** -- TUM also ships ground-truth poses. Predicting
  the relative pose between consecutive frames grounds the latent actions in
  actual camera motion, which is directly useful when latent actions later have
  to be translated to robot actions.
* **Optical flow or scene flow** -- no labels needed (compute with an off-the-
  shelf estimator once, cache it), and it targets exactly the motion information
  the LAM is trying to isolate.
* **Contact / occupancy** -- needs a simulator, so it belongs after a sim
  environment is wired up rather than now.

The honest framing to keep in the docs: none of these make the model "know
physics". They make its features carry geometry, which is a prerequisite.
See `research/008_physics_auxiliary_heads.md`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class PhysicsHead(nn.Module, ABC):
    """An auxiliary prediction head attached to dynamics features.

    Contract rules:
      * Never changes the dynamics model's output -- only adds a loss term.
      * Declares `required_keys` so the dataset can refuse to run a stage whose
        supervision is missing, instead of silently training on zeros.
      * Weighted by `loss_weight` in the stage-C config; setting it to 0 must be
        equivalent to not having the head at all.
    """

    #: batch keys this head needs (e.g. ("depth",), ("pose",))
    required_keys: Tuple[str, ...] = ()

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def forward(
        self, features: torch.Tensor, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """features: (B, T, N, D) dynamics hidden states.

        Returns `(loss, metrics)`. The loss is already scaled by `loss_weight`.
        """

    def check_batch(self, batch: Dict[str, Any]) -> None:
        missing = [k for k in self.required_keys if k not in batch]
        if missing:
            raise KeyError(
                f"{type(self).__name__} needs batch keys {missing}; "
                "the dataset does not provide them -- either pick a dataset that "
                "does (TUM RGB-D has depth and poses) or drop this head"
            )
