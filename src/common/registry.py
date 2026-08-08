"""Generic name -> factory registry.

This is the single mechanism that makes every component in this repo swappable.
A component category (tokenizer, dynamics, VLA backbone, reward, RL algorithm...)
owns one `Registry`. Implementations register themselves by name; configs refer to
components by that name and nothing else.

    TOKENIZERS = Registry("tokenizer")

    @TOKENIZERS.register("conv_vqvae")
    class ConvVQVAE(VideoTokenizer):
        ...

    tok = TOKENIZERS.build({"name": "conv_vqvae", "base_channels": 128})

Swapping an idea therefore never edits a training loop -- it edits one config line.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Dict, Iterable, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A named collection of factories, keyed by string."""

    def __init__(self, category: str) -> None:
        self.category = category
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ register

    def register(self, name: str, **meta: Any) -> Callable[[Type[T]], Type[T]]:
        """Class/function decorator that adds `name` to this registry.

        `meta` is free-form documentation (e.g. `paper=...`, `status="idea"`)
        surfaced by `describe()` so `scripts/tools/list_components.py` can print
        what is available without importing docs.
        """

        def _decorator(obj: Type[T]) -> Type[T]:
            if name in self._factories and self._factories[name] is not obj:
                raise KeyError(
                    f"{self.category} '{name}' is already registered to "
                    f"{self._factories[name]!r}"
                )
            self._factories[name] = obj
            self._meta[name] = meta
            return obj

        return _decorator

    # --------------------------------------------------------------------- build

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._factories:
            raise KeyError(
                f"unknown {self.category} '{name}'. "
                f"registered: {sorted(self._factories)}"
            )
        return self._factories[name]

    def build(self, cfg: Dict[str, Any] | None = None, **overrides: Any) -> Any:
        """Instantiate from a config dict containing a `name` key.

        Every other key is passed to the constructor as a keyword argument.
        `overrides` win over `cfg` -- used to inject runtime-only values such as
        an already-built sub-component.
        """
        cfg = dict(cfg or {})
        cfg.update(overrides)
        try:
            name = cfg.pop("name")
        except KeyError as exc:  # pragma: no cover - config error path
            raise KeyError(
                f"{self.category} config needs a 'name' key; got keys {sorted(cfg)}"
            ) from exc
        return self.get(name)(**cfg)

    # ---------------------------------------------------------------- inspection

    def names(self) -> list[str]:
        return sorted(self._factories)

    def describe(self) -> Dict[str, Dict[str, Any]]:
        return {n: dict(self._meta[n]) for n in self.names()}

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Registry({self.category!r}, {self.names()})"


def autodiscover(package: str, skip: Iterable[str] = ()) -> None:
    """Import every submodule of `package` so its `@register` decorators run.

    Called from each package's `__init__.py`. Without this, a component only
    exists once something has imported its file, which makes config-driven
    construction order-dependent.
    """
    skip = set(skip)
    mod = importlib.import_module(package)
    for info in pkgutil.iter_modules(mod.__path__):
        if info.name.startswith("_") or info.name in skip:
            continue
        importlib.import_module(f"{package}.{info.name}")
