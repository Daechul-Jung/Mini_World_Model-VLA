"""pi0 (3.3B) as a frozen feature extractor with a trainable adapter + head.

pi0 (Physical Intelligence, arXiv:2410.24164) is PaliGemma-3B plus a ~300M
"action expert" that emits continuous actions by **flow matching** rather than by
discretising them into language tokens. Two consequences for this project:

* It is the natural comparison against OpenVLA, because the two differ mainly in
  action parameterisation. Running both frozen, with the same adapter and the
  same head, isolates that variable -- a cheap and genuinely informative
  experiment on one GPU.
* Its native action space is continuous and chunked (50 Hz action chunks), which
  matches this repo's `Action.continuous` contract more directly than OpenVLA's
  token-based one.

**On an 8 GB 4070 Laptop**: 3.3B at 4-bit is roughly 2.5 GB of weights, so frozen
inference and frozen-plus-adapter training both fit with room to spare -- notably
easier than OpenVLA. LoRA fine-tuning is reported at ~22 GB and is out of reach.

**Weights.** The openpi release is JAX-first; PyTorch ports are available on the
Hub (e.g. under `lerobot`). Which checkpoint you load changes the expected
observation format, so `model_name` and `variant` are config, not constants.
This class is written against the LeRobot-style PyTorch port; expect to adjust
`encode()` if you load a different one.

Status: the contract and the loading path are in place; this has not been run
against downloaded weights in this repo. Treat it as a scaffold to fill in when
you get to the pi0 comparison, and see `research/002_pi0_vs_openvla_frozen.md`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common.types import Action, Observation
from vla.core.base import PolicySpec, VLAPolicy
from vla.core.registry import HEADS, POLICIES
from vla.modules import build_modules

DEFAULT_MODEL = "lerobot/pi0"


@POLICIES.register(
    "pi0",
    paper="Black et al., 2024 (arXiv:2410.24164)",
    status="scaffold",
    note="3.3B flow-matching VLA; 4-bit frozen + adapter fits 8 GB",
)
class Pi0Policy(VLAPolicy):
    """Frozen pi0 backbone + trainable modules + trainable head."""

    def __init__(
        self,
        action_dim: int = 7,
        action_chunk: int = 8,
        model_name: str = DEFAULT_MODEL,
        load_in_4bit: bool = True,
        freeze_backbone: bool = True,
        feature_layer: int = -1,
        image_size: int = 224,
        head: Optional[Dict[str, Any]] = None,
        modules: Optional[List[Dict[str, Any]]] = None,
        device_map: str = "auto",
    ):
        super().__init__()
        from .loader import load_pi0

        self.processor, self.backbone, hidden_dim = load_pi0(
            model_name, load_in_4bit=load_in_4bit, device_map=device_map
        )
        self.feature_layer = feature_layer
        if freeze_backbone:
            self.freeze_backbone(True)

        self.modules_stack = build_modules(modules, hidden_dim)
        head_cfg = dict(head or {"name": "continuous_mse"})
        head_cfg.update(
            dim=self.modules_stack.out_dim, action_dim=action_dim, action_chunk=action_chunk
        )
        self.head = HEADS.build(head_cfg)

        self._spec = PolicySpec(
            action_dim=action_dim,
            action_chunk=action_chunk,
            obs_horizon=1,
            image_size=image_size,
            observation_keys=("image", "instruction", "proprio"),
            trainable=not freeze_backbone,
            name="pi0",
        )

    @property
    def spec(self) -> PolicySpec:
        return self._spec

    def freeze_backbone(self, freeze: bool = True) -> None:
        self.backbone.eval() if freeze else self.backbone.train()
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)

    def encode(self, obs: Observation) -> torch.Tensor:
        """-> (B, 1, hidden_dim).

        pi0 conditions on image + instruction + proprioceptive state. The exact
        input plumbing depends on the checkpoint's port; adapt here rather than
        anywhere downstream.
        """
        raise NotImplementedError(
            "pi0 feature extraction depends on which PyTorch port you load. "
            "Implement `encode` against your checkpoint's forward signature -- "
            "everything downstream (modules, head, trainers, envs) already works "
            "once this returns (B, 1, hidden_dim)."
        )

    def forward(self, obs: Observation) -> Action:
        features = self.modules_stack(self.encode(obs), {"observation": obs, **obs.extras})
        return Action(continuous=self.head(features), latent=features[:, -1])

    def loss(
        self, obs: Observation, target_actions: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        features = self.modules_stack(self.encode(obs), {"observation": obs, **obs.extras})
        return self.head.loss(features, target_actions, mask=mask)
