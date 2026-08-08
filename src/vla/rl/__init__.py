"""RL post-training algorithms.

Implemented: the `RLAlgorithm` / `Rollout` contracts.
Specified, each with a research note: `ppo.py`, `grpo.py`, `awr.py`.
Read `base.py` first -- it states the three constraints (stochastic head,
KL anchor, on-policy default) that any algorithm here has to respect.
"""

from common.registry import autodiscover

autodiscover(__name__, skip=("base",))

from .base import RLAlgorithm, Rollout  # noqa: E402

__all__ = ["RLAlgorithm", "Rollout"]
