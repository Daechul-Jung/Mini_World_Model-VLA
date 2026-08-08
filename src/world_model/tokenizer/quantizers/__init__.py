"""Pluggable quantizers for the stage-A tokenizer.

Registered here so `tokenizer.quantizer.name` in a config resolves without the
tokenizer needing to import any specific quantizer.
"""

from common.registry import autodiscover

autodiscover(__name__)
