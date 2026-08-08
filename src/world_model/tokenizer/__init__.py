"""Stage A: video tokenizers."""

from common.registry import autodiscover

from . import quantizers  # noqa: F401  -- must register before tokenizers build

autodiscover(__name__, skip=("quantizers",))
