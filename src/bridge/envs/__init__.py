"""Environments the VLA can be post-trained in.

* `world_model_env.py` -- rollouts inside `GenieWorldModel`. Fast, cheap, and
  approximate. This is the "RL in generated environments" path.
* `sim_env.py`         -- SimplerEnv / LIBERO / MuJoCo wrappers. Slow, and the
  only place a success rate means anything.

Both satisfy `BaseEnv`, so the same RL algorithm runs against either. Train in
imagination, validate in simulation -- and always log `env.is_imagined` next to
any number, because the two are not comparable.
"""

from .base import ENVS, BaseEnv, StepResult
from .world_model_env import WorldModelEnv

__all__ = ["BaseEnv", "StepResult", "ENVS", "WorldModelEnv"]
