"""The `VLAPolicy` contract -- one interface for every vision-language-action model.

Three very different things must satisfy this: a 27M Octo trained from scratch, a
7B OpenVLA loaded 4-bit and frozen, and a pi0 flow-matching policy. They differ
in tokenizer, in action parameterisation, and in whether their weights are
trainable at all. What they share, and all this interface commits to, is:

    observation (images + instruction) -> action chunk

Everything downstream -- the BC trainer, the RL post-training loop, the simulator
adapters, the world-model environment -- is written against this and nothing else.

**The composition story.** A VLA here is three replaceable pieces:

    Observation -> [ Backbone ] -> features -> [ Modules ] -> [ Head ] -> Action
                     frozen or        (B,T,D)   your new       action
                     trainable                  layers         param.

* `backbones/` holds the big pretrained trunk (Octo / OpenVLA / pi0). Swapping it
  is a config line.
* `modules/` is where a new research layer goes -- adapters, memory, world-model
  conditioning. Modules see features and return features, so they stack.
* `heads/` owns how features become actions (regression, discrete bins,
  diffusion, flow matching). Swapping the head changes the loss too, which is
  why the head owns `loss()` rather than the trainer.

That split is the answer to "add my idea as part of a pretrained VLA": load the
backbone frozen, insert a module, train the module and head.

**On this machine -- an RTX 4070 Laptop with 7.7 GiB** -- be explicit about what is
actually reachable, because it is less than a desktop 4070:

| Model    | Params | Full FT | LoRA FT   | 4-bit frozen + adapter | 4-bit inference |
|----------|--------|---------|-----------|------------------------|-----------------|
| Octo-S   | ~30M   | yes     | n/a       | yes                    | n/a             |
| Octo-M   | ~90M   | yes, grad accum | n/a | yes                   | n/a             |
| pi0      | 3.3B   | no      | no (~22G) | yes (~2.5 GB weights)  | yes             |
| OpenVLA  | 7B     | no      | no (~27G) | very tight (~6-7 GB)   | tight           |

So the realistic large-model path here is *frozen backbone + trainable module and
head*, and pi0 is the more comfortable of the two pretrained options. That is
exactly what this interface is shaped for. See ADR-004.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from common.types import Action, ActionSpec, Observation


@dataclass(frozen=True)
class PolicySpec:
    """What a policy needs and produces -- checked before a dataset is built.

    Catches the mismatches that otherwise surface as a shape error 40 minutes
    into training: a 4-DoF UCSD dataset fed to a 7-DoF policy, a wrist camera the
    backbone does not consume, a chunk size the head cannot emit.
    """

    action_dim: int
    action_chunk: int = 1                       # actions predicted per step
    obs_horizon: int = 2                        # frames of history consumed
    image_size: int = 256
    observation_keys: Tuple[str, ...] = ("image", "instruction")
    trainable: bool = True
    name: str = "unnamed"


class VLAPolicy(nn.Module, ABC):
    """A vision-language-action policy."""

    # ------------------------------------------------------------------ contract

    @property
    @abstractmethod
    def spec(self) -> PolicySpec: ...

    @abstractmethod
    def encode(self, obs: Observation) -> torch.Tensor:
        """Observation -> backbone features (B, T, D).

        Separated from `forward` on purpose: this is the tensor `modules/` hook
        into, the tensor an RL critic reads, and the tensor a world model can be
        conditioned on. A frozen backbone runs this under `torch.no_grad`.
        """

    @abstractmethod
    def forward(self, obs: Observation) -> Action:
        """Full pass: features -> modules -> head -> `Action`.

        The returned `continuous` tensor is in the *normalised* training space.
        Call `act()` for physical units.
        """

    @abstractmethod
    def loss(self, obs: Observation, target_actions: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Behaviour-cloning loss for one batch.

        Owned by the policy (which delegates to the head) rather than the
        trainer, because a diffusion head and a discrete-bin head need entirely
        different losses over the same data.
        """

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def act(self, obs: Observation, action_spec: Optional[ActionSpec] = None) -> torch.Tensor:
        """One environment step: (B, action_dim) in *physical* units.

        `action_spec` carries the dataset normalisation statistics. Skipping the
        de-normalisation is the single most common reason a checkpoint that
        trains fine does nothing on a robot or in a simulator.
        """
        self.eval()
        action = self.forward(obs)
        out = action.first
        return action_spec.denormalize(out) if action_spec is not None else out

    def reset(self) -> None:
        """Clear per-episode state (observation history, KV caches, action queue)."""

    # ------------------------------------------------------------------ training

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Only the parameters the current configuration actually trains.

        A frozen-backbone setup returns just the modules and head. The trainer
        uses this rather than `parameters()`, so `freeze_backbone: true` needs no
        cooperation from the training loop.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Default: no-op. Backbones with a distinct trunk should override."""

    def param_summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{type(self).__name__}: {total/1e6:.1f}M params, "
            f"{train/1e6:.1f}M trainable ({100*train/max(total,1):.1f}%)"
        )
