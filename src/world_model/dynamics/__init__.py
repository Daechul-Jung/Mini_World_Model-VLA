"""Auto-registers every implementation in this package."""

from common.registry import autodiscover

autodiscover(__name__)
