"""Train one world-model stage.

Components are trained one at a time and each writes its own checkpoint, so a
later stage loads an earlier one frozen. That is the whole point of the layout:
you can replace the tokenizer without retraining the dynamics model's *code*, and
you can inspect each component in isolation when something is wrong.

    # A -- tokenizer (LSUN stills + TUM frames both work here)
    python scripts/train/train_world_model.py --stage a --config genie_small.yaml

    # B -- latent action model (needs VIDEO; LSUN stills cannot train this)
    python scripts/train/train_world_model.py --stage b --config genie_small.yaml

    # C -- dynamics, on frozen tokens from A and actions from B
    python scripts/train/train_world_model.py --stage c --config genie_small.yaml \
        --tokenizer_ckpt stage_a_tokenizer:best \
        --latent_action_ckpt stage_b_latent_action:best

    # D -- diffusion decoder (optional; do this last, or never)
    python scripts/train/train_world_model.py --stage d --config genie_small.yaml \
        --tokenizer_ckpt stage_a_tokenizer:best

`stage_a_tokenizer:best` resolves to the newest run's `best.pt`, so day-to-day
commands stay free of timestamps. Override any config value inline:

    --set optim.lr=3e-4 --set data.clip_len=16 --set tokenizer.codebook_size=2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from common.checkpoint import resolve_ckpt
from common.config import apply_overrides, config_hash, load_config
from common.logging import setup_logging
from common.seeding import describe_device, get_device, seed_everything
from common.stages import STAGES, StageContext
from common.trainer import train_stage

import world_model  # noqa: F401  -- registers components
import world_model.training  # noqa: F401  -- registers stages

STAGE_ALIASES = {
    "a": "stage_a_tokenizer",
    "b": "stage_b_latent_action",
    "c": "stage_c_dynamics",
    "d": "stage_d_decoder",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, help="a|b|c|d or a full stage name")
    parser.add_argument("--config", required=True, help="path under configs/ or absolute")
    parser.add_argument("--run_name", default=None, help="checkpoint subdirectory; defaults to a config hash")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="key.path=value")

    parser.add_argument("--tokenizer_ckpt", default=None, help="stage A weights (stage_a_tokenizer:best)")
    parser.add_argument("--latent_action_ckpt", default=None, help="stage B weights")
    parser.add_argument("--decoder_ckpt", default=None)
    parser.add_argument("--resume", default=None, help="resume this stage from a checkpoint")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--list_stages", action="store_true")
    args = parser.parse_args()

    setup_logging()
    if args.list_stages:
        print("\n".join(STAGES.names()))
        return 0

    stage_name = STAGE_ALIASES.get(args.stage, args.stage)
    if stage_name not in STAGES:
        parser.error(f"unknown stage {args.stage!r}; available: {STAGES.names()}")

    cfg = load_config(Path("world_model") / args.config if not Path(args.config).exists() else args.config)
    cfg = apply_overrides(cfg, args.overrides)
    for key in ("epochs", "batch_size", "precision", "grad_accum"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["seed"] = args.seed

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")

    parent_ckpts = {}
    for role, spec in (
        ("tokenizer", args.tokenizer_ckpt),
        ("latent_action", args.latent_action_ckpt),
        ("decoder", args.decoder_ckpt),
    ):
        if spec:
            parent_ckpts[role] = str(resolve_ckpt(spec, project="world_model"))
            print(f"  {role:<14} <- {parent_ckpts[role]}")

    run_name = args.run_name or f"{cfg.get('name', 'run')}-{config_hash(cfg, 6)}"
    ctx = StageContext(
        cfg=cfg, device=device, run_name=run_name, project="world_model", parent_ckpts=parent_ckpts
    )
    stage = STAGES.get(stage_name)(ctx)

    missing = [r for r in stage.requires if r.split("_", 2)[-1] not in parent_ckpts and "tokenizer" not in parent_ckpts]
    if missing and not parent_ckpts:
        print(f"\nERROR: stage '{stage_name}' requires checkpoints from {list(stage.requires)}.", file=sys.stderr)
        return 1

    metrics = train_stage(
        stage,
        epochs=cfg.get("epochs", 50),
        grad_accum=cfg.get("grad_accum", 1),
        clip_grad_norm=cfg.get("clip_grad_norm", 1.0),
        precision=cfg.get("precision", "bf16"),
        log_every=cfg.get("log_every", 50),
        eval_every_epochs=cfg.get("eval_every_epochs", 1),
        save_every_epochs=cfg.get("save_every_epochs", 1),
        ema_decay=cfg.get("ema_decay", 0.0),
        resume=args.resume,
    )
    print(f"\n{stage_name} done: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"checkpoints: checkpoints/world_model/{stage_name}/{run_name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
