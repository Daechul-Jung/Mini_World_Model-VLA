"""Train a VLA policy.

    # from scratch on the pick-and-place subset
    python scripts/train/train_vla.py --stage bc --config octo_small.yaml

    # frozen pretrained backbone + trainable adapter and head
    python scripts/train/train_vla.py --stage bc --config openvla_frozen_adapter.yaml

    # insert a research module without touching any training code
    python scripts/train/train_vla.py --stage bc --config octo_small.yaml \
        --set 'policy.modules=[{"name": "gated_residual", "num_heads": 6}]'

Overrides use the same `key.path=value` syntax as the world-model trainer and
parse as YAML, so lists and dicts work inline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.checkpoint import resolve_ckpt
from common.config import apply_overrides, config_hash, load_config
from common.logging import setup_logging
from common.seeding import describe_device, get_device, seed_everything
from common.stages import STAGES, StageContext
from common.trainer import train_stage

import vla  # noqa: F401  -- registers policies/heads/modules/datasets
import vla.training  # noqa: F401  -- registers stages

STAGE_ALIASES = {"bc": "stage_bc", "rl": "stage_rl"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="bc", help="bc|rl or a full stage name")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="key.path=value")
    parser.add_argument("--init_ckpt", default=None, help="warm-start from a previous policy checkpoint")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--list_components", action="store_true", help="print every registered name and exit")
    args = parser.parse_args()

    setup_logging()
    if args.list_components:
        for registry in (vla.POLICIES, vla.HEADS, vla.MODULES, vla.VLA_DATASETS):
            print(f"\n{registry.category}:")
            for name, meta in registry.describe().items():
                extra = " ".join(f"{k}={v}" for k, v in meta.items())
                print(f"  {name:<24} {extra}")
        return 0

    stage_name = STAGE_ALIASES.get(args.stage, args.stage)
    if stage_name not in STAGES:
        parser.error(f"unknown stage {args.stage!r}; available: {STAGES.names()}")

    cfg = load_config(Path("vla") / args.config if not Path(args.config).exists() else args.config)
    cfg = apply_overrides(cfg, args.overrides)
    for key in ("epochs", "batch_size", "precision", "grad_accum"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["seed"] = args.seed

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")

    parent_ckpts = {}
    if args.init_ckpt:
        parent_ckpts["init"] = str(resolve_ckpt(args.init_ckpt, project="vla"))
        print(f"  init <- {parent_ckpts['init']}")

    run_name = args.run_name or f"{cfg.get('name', 'run')}-{config_hash(cfg, 6)}"
    ctx = StageContext(
        cfg=cfg, device=device, run_name=run_name, project="vla", parent_ckpts=parent_ckpts
    )
    stage = STAGES.get(stage_name)(ctx)

    metrics = train_stage(
        stage,
        epochs=cfg.get("epochs", 50),
        grad_accum=cfg.get("grad_accum", 1),
        clip_grad_norm=cfg.get("clip_grad_norm", 1.0),
        precision=cfg.get("precision", "bf16"),
        log_every=cfg.get("log_every", 20),
        eval_every_epochs=cfg.get("eval_every_epochs", 1),
        save_every_epochs=cfg.get("save_every_epochs", 5),
        ema_decay=cfg.get("ema_decay", 0.0),
        resume=args.resume,
    )
    print(f"\n{stage_name} done: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"checkpoints: checkpoints/vla/{stage_name}/{run_name}/")
    print(
        "\nNOTE: offline action error does not predict task success. "
        "Wire up a simulator (see src/vla/research/003_simulation_stack.md) "
        "before drawing conclusions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
