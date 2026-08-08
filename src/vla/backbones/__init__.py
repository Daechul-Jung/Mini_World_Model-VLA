"""VLA backbones. Importing this registers every available policy by name.

| Registry name  | Params | Weights          | On an 8 GB 4070 Laptop             |
|----------------|--------|------------------|-----------------------------|
| `octo_small`   | ~30M   | from scratch     | full training                |
| `octo_medium`  | ~90M   | from scratch     | full training, grad accum    |
| `openvla`      | 7B     | HuggingFace      | frozen + adapter, 4-bit infer|
| `pi0`          | 3.3B   | HuggingFace      | frozen + adapter, 4-bit infer|

Large backbones import lazily: `openvla` and `pi0` need `transformers`, and
4-bit needs `bitsandbytes`, neither of which should be a hard dependency of the
Octo path.
"""

from . import octo  # noqa: F401

try:  # optional heavy backbones
    from . import openvla  # noqa: F401
except Exception:  # pragma: no cover - missing transformers/peft
    pass

try:
    from . import pi0  # noqa: F401
except Exception:  # pragma: no cover
    pass
