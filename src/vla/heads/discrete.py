"""Discrete action head: per-dimension binning + cross-entropy.

The RT-1 / OpenVLA parameterisation. Each action dimension is discretised into
`n_bins` uniform buckets over the dataset's [q01, q99] range and predicted as a
classification problem.

Why this often beats regression on real robot data: the distribution over
actions at a given observation is genuinely multi-modal, and a softmax over bins
can represent "either +0.3 or -0.3, not 0" while a regression head cannot. It is
also the parameterisation the pretrained OpenVLA weights were trained with, so
matching it is what lets a frozen OpenVLA backbone be used without retraining
its notion of what an action is.

The cost is quantisation error: with 256 bins over [q01, q99], resolution is
about 0.8% of the action range per bin. That is fine for deltas and coarse for
absolute joint positions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import HEADS
from .base import ActionHead


@HEADS.register("discrete_bins", paper="RT-1 / OpenVLA", status="baseline")
class DiscreteBinsHead(ActionHead):
    """Per-dimension uniform binning over [-1, 1] in normalised action space.

    Actions must already be normalised (`ActionSpec.normalize`) before binning,
    which is why the bin edges here are fixed at [-1, 1] rather than learned --
    the dataset statistics live in one place, `ActionSpec`, instead of being
    duplicated in the head.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int,
        action_chunk: int = 1,
        n_bins: int = 256,
        hidden: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__(dim, action_dim, action_chunk)
        self.n_bins = n_bins
        self.label_smoothing = label_smoothing

        layers: list[nn.Module] = [nn.LayerNorm(dim)]
        d = dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, action_chunk * action_dim * n_bins))
        self.net = nn.Sequential(*layers)

        centers = torch.linspace(-1, 1, n_bins)
        self.register_buffer("bin_centers", centers)

    @property
    def supports_rl(self) -> bool:
        return True

    # ------------------------------------------------------------------ binning

    def to_bins(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, chunk, A) in [-1, 1] -> int64 bin indices."""
        scaled = (actions.clamp(-1, 1) + 1) / 2 * (self.n_bins - 1)
        return scaled.round().long().clamp(0, self.n_bins - 1)

    def from_bins(self, indices: torch.Tensor) -> torch.Tensor:
        return self.bin_centers[indices]

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        last = features[:, -1] if features.ndim == 3 else features
        out = self.net(last)
        return out.reshape(-1, self.action_chunk, self.action_dim, self.n_bins)

    # ------------------------------------------------------------------ contract

    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.from_bins(self.logits(features).argmax(-1))

    def sample(
        self, features: torch.Tensor, temperature: float = 1.0, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(features) / max(temperature, 1e-5)
        dist = torch.distributions.Categorical(logits=logits)
        indices = dist.sample()
        return self.from_bins(indices), dist.log_prob(indices).sum(dim=(-1, -2))

    def loss(
        self,
        features: torch.Tensor,
        target_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = self.logits(features)
        targets = self.to_bins(target_actions)
        ce = F.cross_entropy(
            logits.reshape(-1, self.n_bins),
            targets.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape(targets.shape)

        if mask is not None:
            ce = ce * mask.unsqueeze(-1)
            loss = ce.sum() / mask.sum().clamp_min(1) / self.action_dim
        else:
            loss = ce.mean()

        with torch.no_grad():
            pred = self.from_bins(logits.argmax(-1))
            metrics = {
                "bin_acc": float((logits.argmax(-1) == targets).float().mean()),
                "action_l1": float((pred - target_actions).abs().mean()),
                "gripper_acc": float(((pred[..., -1] > 0) == (target_actions[..., -1] > 0)).float().mean()),
            }
        return loss, metrics
