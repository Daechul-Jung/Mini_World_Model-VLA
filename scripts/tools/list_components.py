"""Print every registered, swappable component in both projects.

    python scripts/tools/list_components.py

Use this instead of grepping when you want to know what a config's `name:` field
can be set to.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import bridge  # noqa: F401
import vla
import vla.training  # noqa: F401
import world_model as wm
import world_model.training  # noqa: F401
from bridge.action_space import ACTION_TRANSLATORS
from bridge.rewards import REWARDS
from common.stages import STAGES


def show(registry) -> None:
    print(f"\n{registry.category}")
    print("-" * len(registry.category))
    described = registry.describe()
    if not described:
        print("  (none registered)")
    for name, meta in described.items():
        bits = []
        if meta.get("status"):
            bits.append(f"[{meta['status']}]")
        if meta.get("paper"):
            bits.append(meta["paper"])
        if meta.get("note"):
            bits.append(f"-- {meta['note']}")
        print(f"  {name:<22} {' '.join(bits)}")


def main() -> int:
    for registry in (
        wm.TOKENIZERS,
        wm.QUANTIZERS,
        wm.LATENT_ACTIONS,
        wm.DYNAMICS,
        wm.DECODERS,
        wm.WM_DATASETS,
        vla.POLICIES,
        vla.HEADS,
        vla.MODULES,
        vla.VLA_DATASETS,
        REWARDS,
        ACTION_TRANSLATORS,
        STAGES,
    ):
        show(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
