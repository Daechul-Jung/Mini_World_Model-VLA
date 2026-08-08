"""Genie-style world model, trained one component at a time.

Import order matters: quantizers must register before tokenizers are built, and
every component must register before `GenieWorldModel.from_config` runs. Each
sub-package's `__init__` calls `autodiscover`, so importing this package is
enough to make every registered component reachable by name.

    from world_model import GenieWorldModel, TOKENIZERS
    print(TOKENIZERS.describe())
"""

from . import tokenizer, latent_action, dynamics, decoder, memory, physics, data  # noqa: F401
from .core import (
    ActionSpaceSpec,
    Decoder,
    Dynamics,
    GenieWorldModel,
    LatentActionModel,
    LatentSpec,
    RolloutResult,
    VideoTokenizer,
    DECODERS,
    DYNAMICS,
    LATENT_ACTIONS,
    MEMORY,
    PHYSICS_HEADS,
    QUANTIZERS,
    TOKENIZERS,
    WM_DATASETS,
)

__all__ = [
    "GenieWorldModel",
    "RolloutResult",
    "VideoTokenizer",
    "LatentActionModel",
    "Dynamics",
    "Decoder",
    "LatentSpec",
    "ActionSpaceSpec",
    "TOKENIZERS",
    "QUANTIZERS",
    "LATENT_ACTIONS",
    "DYNAMICS",
    "DECODERS",
    "MEMORY",
    "PHYSICS_HEADS",
    "WM_DATASETS",
]
