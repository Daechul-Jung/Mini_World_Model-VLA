"""The idea slot: layers you insert into an existing VLA.

This is the mechanism for "load pretrained weights, then add my idea as part of
the VLA". A `PolicyModule` sits between the backbone and the action head:

    obs --> backbone --> features --> [module_1] --> [module_2] --> head --> action
             (frozen)     (B,T,D)      (trained)      (trained)    (trained)

Contract rules that make modules stackable and removable:

1. **Shape-preserving by default.** `forward` takes (B, T, D) and returns
   (B, T, D). A module that changes D must declare `out_dim`, and only the last
   module in a stack may do so.
2. **Identity at initialisation.** A newly built module must leave features
   essentially unchanged -- zero-initialised output projections, gates starting
   closed. This is what makes "frozen backbone + new module" trainable at all:
   at step 0 the policy behaves exactly like the pretrained one, so the loss
   starts where the pretrained model left off instead of at random.
   `test_module_identity_at_init` in `tests/` enforces it.
3. **Extra context via `context`, never via the constructor.** A module that
   conditions on a world-model latent, a goal image, or a reward signal reads it
   from the `context` dict, so the policy never has to know what the module
   needs.

Adding an idea:
    1. Write `research/NNN_your_idea.md` (hypothesis, what to measure, kill criterion)
    2. Implement `YourModule(PolicyModule)` in `modules/your_idea.py`
    3. `@MODULES.register("your_idea")`
    4. Add it to the config's `modules:` list
    5. `pytest tests/test_vla_modules.py`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .. core.registry import MODULES


class PolicyModule(nn.Module, ABC):
    """A shape-preserving transformation of backbone features."""

    #: keys this module expects in `context`; checked once at first forward
    required_context: tuple[str, ...] = ()

    def __init__(self, dim: int, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim

    @abstractmethod
    def forward(
        self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """features: (B, T, D) -> (B, T, out_dim)."""

    def check_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        context = context or {}
        missing = [k for k in self.required_context if k not in context]
        if missing:
            raise KeyError(
                f"{type(self).__name__} needs context keys {missing}. "
                "Pass them through `Observation.extras`, which the policy forwards."
            )
        return context


class ModuleStack(nn.Module):
    """Runs a list of `PolicyModule`s in order, validating dimensions.

    Built by `build_modules(cfg, dim)`; a policy holds one and calls it between
    `encode()` and the head. An empty stack is the identity, so the same policy
    code path serves the no-modules baseline.
    """

    def __init__(self, modules: List[PolicyModule], dim: int) -> None:
        super().__init__()
        current = dim
        for i, mod in enumerate(modules):
            if mod.dim != current:
                raise ValueError(
                    f"module {i} ({type(mod).__name__}) expects dim {mod.dim}, "
                    f"but the previous stage outputs {current}"
                )
            if mod.out_dim != current and i != len(modules) - 1:
                raise ValueError(
                    f"module {i} ({type(mod).__name__}) changes dim "
                    f"{current}->{mod.out_dim} but is not last in the stack"
                )
            current = mod.out_dim
        self.mods = nn.ModuleList(modules)
        self.out_dim = current

    def forward(
        self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        for mod in self.mods:
            features = mod(features, context)
        return features

    def __len__(self) -> int:
        return len(self.mods)


def build_modules(configs: Optional[List[Dict[str, Any]]], dim: int) -> ModuleStack:
    """Build a `ModuleStack` from a config list, threading `dim` through."""
    built: List[PolicyModule] = []
    current = dim
    for cfg in configs or []:
        cfg = dict(cfg)
        cfg.setdefault("dim", current)
        mod = MODULES.build(cfg)
        built.append(mod)
        current = mod.out_dim
    return ModuleStack(built, dim)
