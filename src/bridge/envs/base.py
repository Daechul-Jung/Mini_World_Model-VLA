"""Environment contract shared by the world model and real simulators.

Deliberately Gymnasium-shaped but batched and tensor-native, because the world
model is inherently batched (rolling 32 imagined episodes costs almost the same
as one) while MuJoCo is not. Wrapping both behind this means an RL algorithm
never knows which it is talking to -- which is the whole point, since the plan is
to train in imagination and validate in simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from common.registry import Registry
from common.types import Observation

ENVS = Registry("bridge.env")


@dataclass
class StepResult:
    """Batched Gymnasium step tuple."""

    observation: Observation
    reward: torch.Tensor                       # (B,)
    done: torch.Tensor                         # (B,) bool
    info: Dict[str, Any] = field(default_factory=dict)


class BaseEnv(ABC):
    """A batched environment."""

    @abstractmethod
    def reset(self, batch_size: int = 1, seed: Optional[int] = None) -> Observation: ...

    @abstractmethod
    def step(self, action: torch.Tensor) -> StepResult:
        """`action`: (B, action_dim) in physical units."""

    def close(self) -> None: ...

    @property
    def is_imagined(self) -> bool:
        """True for the world-model env.

        Anything logging results must record this. A success rate measured
        inside a learned world model and one measured in a simulator are not
        comparable numbers, and conflating them is the easiest way to fool
        yourself in this project.
        """
        return False
