"""Offline policy evaluation on held-out episodes.

The only evaluation available before a simulator is wired up, and the one to be
most careful with. Offline action error measures whether the policy reproduces
the demonstrator's actions on states the demonstrator visited. It says nothing
about recovery from states the demonstrator never entered -- which is where
imitation policies actually fail.

Report per-dimension error, not just the mean. On a 7-DoF action the mean hides
the two facts that matter: whether rotation is worse than translation, and
whether the gripper is right at the moments it changes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from common.types import Observation


@torch.no_grad()
def evaluate_offline(
    policy, loader: DataLoader, device: torch.device, max_batches: int = 100
) -> Dict[str, Any]:
    """Action error on held-out windows."""
    policy.eval().to(device)
    errors: List[torch.Tensor] = []
    gripper_hits, gripper_total = 0, 0
    transition_hits, transition_total = 0, 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        obs = Observation(
            image=batch["image"].to(device),
            instruction=batch.get("instruction"),
            pad_mask=batch.get("pad_mask", None),
        )
        target = batch["actions"].to(device)
        pred = policy(obs).continuous

        errors.append((pred - target).abs().mean(dim=1).cpu())     # (B, action_dim)

        g_pred, g_true = pred[..., -1] > 0, target[..., -1] > 0
        gripper_hits += int((g_pred == g_true).sum())
        gripper_total += g_true.numel()

        # Accuracy restricted to timesteps where the gripper state *changes* --
        # a handful of frames per episode that decide success, and which a
        # whole-episode average completely hides.
        if g_true.shape[1] > 1:
            changes = g_true[:, 1:] != g_true[:, :-1]
            if changes.any():
                transition_hits += int(((g_pred[:, 1:] == g_true[:, 1:]) & changes).sum())
                transition_total += int(changes.sum())

    per_dim = torch.cat(errors).mean(0)
    return {
        "action_l1": float(per_dim.mean()),
        "action_l1_per_dim": [round(float(v), 5) for v in per_dim],
        "gripper_acc": gripper_hits / max(gripper_total, 1),
        "gripper_transition_acc": transition_hits / max(transition_total, 1),
        "n_batches": min(i + 1, max_batches),
    }
