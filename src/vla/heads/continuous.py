"""Continuous action heads: deterministic regression and a Gaussian policy."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import HEADS
from .base import ActionHead


def _mlp(dim: int, hidden: int, out: int, depth: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(dim)]
    d = dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


@HEADS.register("continuous_mse", status="baseline")
class ContinuousMSEHead(ActionHead):
    """Deterministic regression to the action chunk.

    Fast, stable, and the right first thing to try. Its known failure is
    multi-modality: when the demonstrations contain two valid actions for the
    same observation, the MSE optimum is their average, which is often invalid.
    If offline action error is low but rollouts fail at grasp points, that is
    this head's signature -- switch to `discrete_bins` or `diffusion`.

    `loss_type="huber"` is a cheap partial mitigation: it down-weights the
    outliers that pull the mean between modes.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        action_chunk: int = 1,
        hidden: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
        loss_type: str = "huber",
        gripper_index: Optional[int] = -1,
        gripper_weight: float = 1.0,
    ):
        super().__init__(dim, action_dim, action_chunk)
        self.net = _mlp(dim, hidden, action_chunk * action_dim, depth, dropout)
        self.loss_type = loss_type
        self.gripper_index = gripper_index
        self.gripper_weight = gripper_weight

    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        last = features[:, -1] if features.ndim == 3 else features
        out = self.net(last)
        return out.reshape(-1, self.action_chunk, self.action_dim).tanh()

    def loss(
        self,
        features: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = self.forward(features)
        fn = {"mse": F.mse_loss, "l1": F.l1_loss, "huber": F.smooth_l1_loss}[self.loss_type]
        per_dim = fn(pred, target_actions, reduction="none")

        if self.gripper_index is not None and self.gripper_weight != 1.0:
            # The gripper dim decides task success but is one of N dims in the
            # mean, so it is easy to under-weight into irrelevance.
            weights = torch.ones(self.action_dim, device=pred.device)
            weights[self.gripper_index] = self.gripper_weight
            per_dim = per_dim * weights

        if mask is not None:
            per_dim = per_dim * mask.unsqueeze(-1)
            loss = per_dim.sum() / mask.sum().clamp_min(1) / self.action_dim
        else:
            loss = per_dim.mean()

        with torch.no_grad():
            metrics = {"action_l1": float((pred - target_actions).abs().mean())}
            if self.gripper_index is not None:
                g_pred = pred[..., self.gripper_index]
                g_true = target_actions[..., self.gripper_index]
                metrics["gripper_acc"] = float(((g_pred > 0) == (g_true > 0)).float().mean())
        return loss, metrics


@HEADS.register("gaussian", status="baseline", note="RL-capable continuous head")
class GaussianHead(ActionHead):
    """Tanh-squashed diagonal Gaussian -- the head to use for RL post-training.

    `ContinuousMSEHead` cannot be used with a policy gradient because it has no
    density. This one emits mean and log-std, so `sample()` returns an action
    together with its log-probability (with the tanh Jacobian correction).

    Behaviour cloning with this head is just Gaussian NLL, so a single checkpoint
    can be BC-pretrained and then RL-finetuned without swapping heads -- which is
    exactly the pretrain-then-post-train path this project is built around.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        action_chunk: int = 1,
        hidden: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        state_dependent_std: bool = True,
    ):
        super().__init__(dim, action_dim, action_chunk)
        out = action_chunk * action_dim
        self.net = _mlp(dim, hidden, out * (2 if state_dependent_std else 1), depth, dropout)
        self.state_dependent_std = state_dependent_std
        if not state_dependent_std:
            self.log_std = nn.Parameter(torch.zeros(action_chunk, action_dim))
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    @property
    def supports_rl(self) -> bool:
        return True

    def _dist_params(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        last = features[:, -1] if features.ndim == 3 else features
        out = self.net(last)
        shape = (-1, self.action_chunk, self.action_dim)
        if self.state_dependent_std:
            mean, log_std = out.chunk(2, dim=-1)
            mean, log_std = mean.reshape(shape), log_std.reshape(shape)
        else:
            mean = out.reshape(shape)
            log_std = self.log_std.expand_as(mean)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        mean, _ = self._dist_params(features)
        return mean.tanh()

    def sample(
        self, features: torch.Tensor, temperature: float = 1.0, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._dist_params(features)
        std = log_std.exp() * temperature
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        action = pre_tanh.tanh()

        logp = (-0.5 * noise.pow(2) - log_std - 0.5 * torch.log(torch.tensor(2 * torch.pi)))
        logp = logp - torch.log1p(-action.pow(2) + 1e-6)   # tanh change of variables
        return action, logp.sum(dim=(-1, -2))

    def loss(
        self,
        features: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Gaussian NLL in pre-tanh space."""
        mean, log_std = self._dist_params(features)
        target = target_actions.clamp(-0.999, 0.999).atanh()
        nll = log_std + 0.5 * ((target - mean) / log_std.exp()).pow(2)

        if mask is not None:
            nll = nll * mask.unsqueeze(-1)
            loss = nll.sum() / mask.sum().clamp_min(1) / self.action_dim
        else:
            loss = nll.mean()

        with torch.no_grad():
            metrics = {
                "action_l1": float((mean.tanh() - target_actions).abs().mean()),
                "mean_std": float(log_std.exp().mean()),
            }
        return loss, metrics
