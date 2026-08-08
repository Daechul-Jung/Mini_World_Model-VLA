"""VLA training stages.

    stage_bc       behaviour cloning (from scratch, or frozen backbone + adapter)
    stage_rl       RL post-training in a simulator or the world model  [scaffold]
"""

from common.registry import autodiscover

autodiscover(__name__, skip=("data",))
