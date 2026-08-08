"""YAML config loading with inheritance, CLI overrides, and content hashing.

Config rules used everywhere in this repo:

* A config may declare `_base_: path/to/other.yaml` (or a list) and is deep-merged
  on top of it. This keeps `octo_medium.yaml` a ten-line diff against
  `octo_small.yaml`.
* Any component sub-dict carries a `name` key consumed by a `Registry`.
* `config_hash()` is stamped into every checkpoint so a loaded weight file can be
  matched back to the exact config that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict.

    A sub-dict containing `_replace_: true` is substituted wholesale instead of
    merged. Needed whenever a child config switches a component to one that takes
    different constructor arguments -- merging would otherwise leave the parent's
    keys behind and the registry would pass them to a constructor that has never
    heard of them.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            if value.pop("_replace_", False):
                out[key] = copy.deepcopy(value)
            else:
                out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    for candidate in (REPO_ROOT / p, CONFIG_ROOT / p):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"config not found: {path}")


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> Dict[str, Any]:
    """Load a YAML config, resolving `_base_` inheritance."""
    resolved = _resolve(path)
    if resolved in _seen:
        raise ValueError(f"circular _base_ chain at {resolved}")

    with resolved.open() as fh:
        cfg = yaml.safe_load(fh) or {}

    bases = cfg.pop("_base_", None)
    if bases:
        if isinstance(bases, (str, Path)):
            bases = [bases]
        merged: Dict[str, Any] = {}
        for b in bases:
            merged = deep_merge(merged, load_config(b, _seen | {resolved}))
        cfg = deep_merge(merged, cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply `a.b.c=value` CLI overrides. Values are parsed as YAML scalars."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        node: Dict[str, Any] = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot descend into non-dict at {part!r} in {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def config_hash(cfg: Dict[str, Any], length: int = 10) -> str:
    """Stable short hash of a config. Stamped into checkpoints and run dirs."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def flatten(cfg: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten to `a.b.c -> value`, for logging to TensorBoard/W&B."""
    flat: Dict[str, Any] = {}
    for key, value in cfg.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{full}."))
        else:
            flat[full] = value
    return flat


def require(cfg: Dict[str, Any], keys: Iterable[str]) -> None:
    """Fail loudly and early on a missing config key, not deep inside a model."""
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"config is missing required keys: {missing}")
