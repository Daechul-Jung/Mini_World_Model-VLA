"""Shared data contracts.

These structs are the seam between the two projects. The VLA emits an `Action`;
the world model and the simulators both accept one; a reward model reads a
`Transition`. Keeping them here means neither project imports the other's model
code just to agree on a tensor layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class Observation:
    """What a policy sees at one timestep.

    `image` is the canonical modality; everything else is optional so a policy can
    declare which keys it consumes via `VLAPolicy.observation_keys`.
    """

    image: torch.Tensor                          # (B, T, 3, H, W) float in [-1, 1]
    instruction: Optional[list[str]] = None      # length B
    proprio: Optional[torch.Tensor] = None       # (B, T, D_proprio)
    wrist_image: Optional[torch.Tensor] = None   # (B, T, 3, H, W)
    pad_mask: Optional[torch.Tensor] = None      # (B, T) bool, True = real frame
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.image.shape[0]

    @property
    def horizon(self) -> int:
        return self.image.shape[1]

    def to(self, device: torch.device) -> "Observation":
        def _mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return Observation(
            image=_mv(self.image),
            instruction=self.instruction,
            proprio=_mv(self.proprio),
            wrist_image=_mv(self.wrist_image),
            pad_mask=_mv(self.pad_mask),
            extras={k: _mv(v) for k, v in self.extras.items()},
        )


@dataclass
class Action:
    """A predicted action chunk.

    `continuous` is the native robot action, already de-normalised into physical
    units. `logp` and `value` are populated only by RL-capable policies.
    """

    continuous: torch.Tensor                     # (B, chunk, action_dim)
    logp: Optional[torch.Tensor] = None          # (B,)
    value: Optional[torch.Tensor] = None         # (B,)
    latent: Optional[torch.Tensor] = None        # (B, D) pre-head features
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def first(self) -> torch.Tensor:
        """The action to actually execute this step: (B, action_dim)."""
        return self.continuous[:, 0]


@dataclass
class Transition:
    """One env step, the unit a reward model and a replay buffer consume."""

    observation: Observation
    action: torch.Tensor                         # (B, action_dim)
    next_observation: Observation
    reward: Optional[torch.Tensor] = None        # (B,)
    done: Optional[torch.Tensor] = None          # (B,) bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionSpec:
    """Action-space description, carried alongside dataset normalisation stats.

    Required whenever weights move between datasets or between a real dataset and
    a simulator: an Octo checkpoint trained on 4-DoF UCSD deltas means nothing
    applied to a 7-DoF Bridge action without these numbers.
    """

    dim: int
    low: Optional[torch.Tensor] = None           # (dim,)
    high: Optional[torch.Tensor] = None          # (dim,)
    mean: Optional[torch.Tensor] = None          # (dim,) dataset mean
    std: Optional[torch.Tensor] = None           # (dim,) dataset std
    q01: Optional[torch.Tensor] = None           # (dim,) 1st percentile
    q99: Optional[torch.Tensor] = None           # (dim,) 99th percentile
    gripper_index: Optional[int] = None          # index of the binary gripper dim
    relative: bool = True                        # deltas vs absolute poses
    name: str = "unnamed"

    def normalize(self, actions: torch.Tensor, scheme: str = "q99") -> torch.Tensor:
        """Map physical actions into the model's training range."""
        if scheme == "q99" and self.q01 is not None and self.q99 is not None:
            lo, hi = self.q01.to(actions), self.q99.to(actions)
            return (2 * (actions - lo) / (hi - lo + 1e-8) - 1).clamp(-1, 1)
        if scheme == "gaussian" and self.mean is not None and self.std is not None:
            return (actions - self.mean.to(actions)) / (self.std.to(actions) + 1e-8)
        return actions

    def denormalize(self, actions: torch.Tensor, scheme: str = "q99") -> torch.Tensor:
        """Inverse of `normalize` -- always call before sending to a robot/sim."""
        if scheme == "q99" and self.q01 is not None and self.q99 is not None:
            lo, hi = self.q01.to(actions), self.q99.to(actions)
            return (actions + 1) / 2 * (hi - lo) + lo
        if scheme == "gaussian" and self.mean is not None and self.std is not None:
            return actions * (self.std.to(actions) + 1e-8) + self.mean.to(actions)
        return actions
