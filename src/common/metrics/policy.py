"""Policy metrics.

Offline action error is the only cheap signal available before a simulator is
wired up, but it is a weak proxy: two policies with identical MSE can differ
wildly in success rate because the errors that matter are concentrated at grasp
transitions. `gripper_accuracy` is tracked separately for exactly that reason.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def action_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    err = (pred - target) ** 2
    if mask is not None:
        err = err * mask.unsqueeze(-1)
        return float(err.sum() / mask.sum().clamp_min(1) / pred.shape[-1])
    return float(err.mean())


def action_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    err = (pred - target).abs()
    if mask is not None:
        err = err * mask.unsqueeze(-1)
        return float(err.sum() / mask.sum().clamp_min(1) / pred.shape[-1])
    return float(err.mean())


def gripper_accuracy(
    pred: torch.Tensor, target: torch.Tensor, index: int = -1, threshold: float = 0.0
) -> float:
    """Binary agreement on the gripper dimension -- the dim that decides success."""
    return float(((pred[..., index] > threshold) == (target[..., index] > threshold)).float().mean())


def success_rate(successes: Sequence[bool]) -> float:
    return float(sum(bool(s) for s in successes)) / max(len(successes), 1)
