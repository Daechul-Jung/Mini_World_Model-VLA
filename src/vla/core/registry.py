"""Registries for the VLA project."""

from __future__ import annotations

from common.registry import Registry

POLICIES = Registry("vla.policy")        # whole VLA policies (Octo, OpenVLA, pi0, custom)
BACKBONES = Registry("vla.backbone")     # feature trunks, when reused across policies
HEADS = Registry("vla.head")             # features -> actions, and the matching loss
MODULES = Registry("vla.module")         # <- the idea slot: layers inserted into a policy
VLA_DATASETS = Registry("vla.dataset")
RL_ALGORITHMS = Registry("vla.rl")

__all__ = ["POLICIES", "BACKBONES", "HEADS", "MODULES", "VLA_DATASETS", "RL_ALGORITHMS"]
