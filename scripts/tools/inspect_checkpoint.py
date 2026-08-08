"""Inspect a checkpoint: what it is, what produced it, and what it was trained on.

    python scripts/tools/inspect_checkpoint.py stage_c_dynamics:best
    python scripts/tools/inspect_checkpoint.py checkpoints/vla/stage_bc/run/best.pt --project vla

The lineage section answers the question that silently ruins staged training:
"which tokenizer produced the tokens this dynamics model was trained on?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from common.checkpoint import load_checkpoint, resolve_ckpt, resolve_lineage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="path, or stage:best / stage:last shorthand")
    parser.add_argument("--project", default="world_model", choices=["world_model", "vla"])
    parser.add_argument("--keys", action="store_true", help="list state_dict keys and shapes")
    args = parser.parse_args()

    path = resolve_ckpt(args.checkpoint, project=args.project)
    state, meta = load_checkpoint(path)

    print(f"path         {path}")
    print(f"component    {meta.component}")
    print(f"stage        {meta.stage}")
    print(f"step/epoch   {meta.step} / {meta.epoch}")
    print(f"config_hash  {meta.config_hash}")

    if meta.metrics:
        print("\nmetrics")
        for k, v in sorted(meta.metrics.items()):
            print(f"  {k:<28} {v:.4f}")

    if "model" in state:
        params = sum(v.numel() for v in state["model"].values() if torch.is_tensor(v))
        print(f"\nparameters   {params/1e6:.2f}M")
    print(f"state keys   {sorted(state.keys())}")

    if "action_spec" in state:
        spec = state["action_spec"]
        print(f"\naction spec  dim={spec['dim']} name={spec.get('name')} "
              f"gripper_index={spec.get('gripper_index')}")
        print("  (a policy checkpoint without this is not deployable)")

    if meta.frozen_parents:
        print("\nfrozen inputs")
        for role, parent in meta.frozen_parents.items():
            print(f"  {role:<16} {parent}")

    lineage = resolve_lineage(path)
    if len(lineage) > 1:
        print("\nlineage (newest first)")
        for entry in lineage:
            print(f"  {entry['component']:<16} {entry['stage']:<24} step {entry['step']}")

    if args.keys and "model" in state:
        print("\nstate_dict")
        for k, v in state["model"].items():
            if torch.is_tensor(v):
                print(f"  {k:<60} {tuple(v.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
