"""Automatic reward models for imagined rollouts.

Implemented: `goal_image`, `ensemble`.
Specified but not implemented, each with a research note:

* `progress`       -- VIP/LIV time-contrastive value; `research/016_progress_reward.md`
* `vlm_judge`      -- a VLM scores the imagined frame; `research/017_vlm_reward.md`
* `dynamics_prior` -- world-model likelihood as a plausibility term
"""

from common.registry import autodiscover

autodiscover(__name__, skip=("base",))

from .base import REWARDS, RewardModel  # noqa: E402

__all__ = ["RewardModel", "REWARDS"]
