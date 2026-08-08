"""One registry per world-model component category.

Adding an idea = one new file + one `@register` line + one config line.
Nothing in `training/` or `eval/` ever needs to change.

    from world_model.core.registry import DYNAMICS

    @DYNAMICS.register("maskgit_st", paper="arXiv:2402.15391", status="planned")
    class MaskGITSTDynamics(Dynamics):
        ...
"""

from __future__ import annotations

from common.registry import Registry

TOKENIZERS = Registry("world_model.tokenizer")
QUANTIZERS = Registry("world_model.quantizer")
LATENT_ACTIONS = Registry("world_model.latent_action")
DYNAMICS = Registry("world_model.dynamics")
DECODERS = Registry("world_model.decoder")
MEMORY = Registry("world_model.memory")
PHYSICS_HEADS = Registry("world_model.physics")
WM_DATASETS = Registry("world_model.dataset")

__all__ = [
    "TOKENIZERS",
    "QUANTIZERS",
    "LATENT_ACTIONS",
    "DYNAMICS",
    "DECODERS",
    "MEMORY",
    "PHYSICS_HEADS",
    "WM_DATASETS",
]
