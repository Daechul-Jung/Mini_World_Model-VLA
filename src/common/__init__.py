"""Framework layer shared by `src/vla`, `src/world_model` and `src/bridge`.

Contains no model code. Everything here is machinery that a new research idea
should be able to reuse without modification: registries, config loading,
checkpoint lineage, the stage contract, and the training loop.
"""

from .config import apply_overrides, config_hash, deep_merge, load_config
from .checkpoint import (
    CheckpointManager,
    CheckpointMeta,
    load_component,
    resolve_ckpt,
    resolve_lineage,
)
from .registry import Registry, autodiscover
from .seeding import get_device, seed_everything
from .stages import STAGES, Stage, StageContext
from .trainer import train_stage
from .types import Action, ActionSpec, Observation, Transition

__all__ = [
    "load_config",
    "apply_overrides",
    "deep_merge",
    "config_hash",
    "CheckpointManager",
    "CheckpointMeta",
    "load_component",
    "resolve_ckpt",
    "resolve_lineage",
    "Registry",
    "autodiscover",
    "seed_everything",
    "get_device",
    "Stage",
    "StageContext",
    "STAGES",
    "train_stage",
    "Observation",
    "Action",
    "Transition",
    "ActionSpec",
]
