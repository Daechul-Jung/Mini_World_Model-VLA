"""World-model training stages.

Importing this package registers every stage with `common.stages.STAGES`, which
is how `scripts/train/train_world_model.py` dispatches `--stage`.

Order and dependencies:

    A  tokenizer      images or clips          -- no dependencies
    B  latent_action  clips (>= 2 frames)      -- no dependencies (pixel input)
    C  dynamics       clips                    -- needs A, and B if action_kind=latent
    D  decoder        clips                    -- needs A  (optional stage)
"""

from common.registry import autodiscover

autodiscover(__name__, skip=("data",))
