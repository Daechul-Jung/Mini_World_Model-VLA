"""Where the VLA and the world model meet.

Neither project imports the other; both import `common`. This package is the only
place that knows about both, which keeps the two research tracks independently
runnable.

Contents:

* `action_space.py` -- translating VLA actions into world-model actions. Read the
  module docstring before planning around this: it is the genuinely hard part.
* `envs/`           -- the world model as an RL environment, plus real simulators.
* `rewards/`        -- automatic reward functions over imagined observations.
* `training/`       -- the RL-in-imagination stage.
"""

from .action_space import ACTION_TRANSLATORS, ActionTranslator, build_translator

__all__ = ["ActionTranslator", "ACTION_TRANSLATORS", "build_translator"]
