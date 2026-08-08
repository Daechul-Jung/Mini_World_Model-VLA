"""Insertable policy layers -- the VLA idea slot.

See `base.py` for the contract and the identity-at-init rule that makes a new
module safe to attach to frozen pretrained weights.
"""

from common.registry import autodiscover

autodiscover(__name__, skip=("base",))

from .base import ModuleStack, PolicyModule, build_modules  # noqa: E402

__all__ = ["PolicyModule", "ModuleStack", "build_modules"]
