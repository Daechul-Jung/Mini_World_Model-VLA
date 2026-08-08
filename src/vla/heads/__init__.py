"""Action heads. Importing this package registers every head by name.

Not yet implemented, and each has a research note describing what it would buy:

* `diffusion`      -- Octo-base's DDPM action head; `research/013_diffusion_head.md`
* `flow_matching`  -- pi0's parameterisation;       `research/014_flow_matching_head.md`
"""

from common.registry import autodiscover

autodiscover(__name__, skip=("base",))

from .base import ActionHead  # noqa: E402

__all__ = ["ActionHead"]
