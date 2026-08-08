"""Vision-language-action policies, composed from swappable parts.

    Observation -> [backbone] -> features -> [modules] -> [head] -> Action

Importing this package registers every backbone, head, module and dataset, so
anything reachable from a config is reachable by name:

    import vla
    print(vla.POLICIES.describe())
    policy = vla.POLICIES.build({"name": "octo_small", "action_dim": 4})
"""

from . import backbones, data, heads, modules  # noqa: F401
from .core import (
    BACKBONES,
    HEADS,
    MODULES,
    POLICIES,
    RL_ALGORITHMS,
    VLA_DATASETS,
    PolicySpec,
    VLAPolicy,
)

__all__ = [
    "VLAPolicy",
    "PolicySpec",
    "POLICIES",
    "BACKBONES",
    "HEADS",
    "MODULES",
    "VLA_DATASETS",
    "RL_ALGORITHMS",
]
