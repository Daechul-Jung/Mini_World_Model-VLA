"""Evaluate a world-model stage and write images you can actually look at.

    # stage A (tokenizer) -- reconstructions, codebook health, latent interpolation
    python scripts/eval/eval_world_model.py --stage a \
        --config genie_small_lsun.yaml --ckpt stage_a_tokenizer:best \
        --set data.roots=[data/lsun_rooms]

    # stage C (dynamics) -- rollout video, action sweep
    python scripts/eval/eval_world_model.py --stage c --config genie_small.yaml \
        --tokenizer_ckpt stage_a_tokenizer:best \
        --latent_action_ckpt stage_b_latent_action:best \
        --ckpt stage_c_dynamics:best

**On what "generated results" means for a stage-A-only model.** A VQ-VAE
tokenizer is not a generative model -- it has no prior over tokens, so it cannot
invent a new room. What it can show is:

  * reconstruction -- how much of a real room survives the token bottleneck
  * interpolation  -- whether the latent space is smooth between two rooms
  * random codes   -- what the decoder does with tokens no encoder produced,
                      which is the honest demonstration that the prior is missing

Novel-scene generation needs a prior over the token grid. In this repo that
prior is the stage-C dynamics model, which predicts the *next frame's* tokens and
therefore needs ordered video -- which LSUN does not have. See
`src/world_model/docs/PRD.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torchvision

from common.checkpoint import load_component, resolve_ckpt
from common.config import apply_overrides, load_config
from common.metrics import codebook_usage, perplexity, psnr, ssim
from common.seeding import describe_device, get_device, seed_everything

import world_model as wm
from world_model.training.data import build_loaders


def _save(tensor: torch.Tensor, path: Path, nrow: int = 8) -> Path:
    """Save a (N, 3, H, W) tensor in [-1, 1] as a PNG grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = torchvision.utils.make_grid((tensor.clamp(-1, 1) + 1) / 2, nrow=nrow, padding=2)
    torchvision.utils.save_image(grid, path)
    return path


# --------------------------------------------------------------------- stage A


@torch.no_grad()
def eval_tokenizer(tokenizer, loader, device, out_dir: Path, n_show: int = 8) -> dict:
    tokenizer.eval().to(device)
    spec = tokenizer.latent_spec
    vocab = spec.vocab_size or 1

    totals, count = {}, 0
    histogram = torch.zeros(vocab)
    first_batch = None

    for batch in loader:
        frames = batch["frames"].to(device)
        enc = tokenizer.encode(frames)
        recon = tokenizer.decode(enc["latents"])

        if first_batch is None:
            first_batch = (frames[:n_show].cpu(), recon[:n_show].cpu())

        step = {
            "psnr": psnr(recon, frames),
            "ssim": ssim(recon, frames),
            "codebook_use": codebook_usage(enc["indices"], vocab),
            "perplexity": perplexity(enc["indices"], vocab),
        }
        for k, v in step.items():
            totals[k] = totals.get(k, 0.0) + v
        histogram += torch.bincount(enc["indices"].flatten().cpu(), minlength=vocab).float()
        count += 1

    metrics = {k: v / max(count, 1) for k, v in totals.items()}
    # Usage over the WHOLE eval set, not averaged per batch -- a batch of 32
    # images cannot touch 1024 codes even with a perfectly healthy codebook, so
    # the per-batch number systematically understates it.
    metrics["codebook_use_total"] = float((histogram > 0).sum()) / vocab
    metrics["dead_codes"] = int((histogram == 0).sum())

    # ---- 1. reconstructions ------------------------------------------------
    real, recon = first_batch
    interleaved = torch.stack([real, recon], dim=1).flatten(0, 1)   # real, recon, real, ...
    _save(interleaved, out_dir / "01_reconstructions.png", nrow=4)

    # ---- 2. codebook usage -------------------------------------------------
    _plot_codebook(histogram, out_dir / "02_codebook_usage.png", metrics)

    # ---- 3. latent interpolation -------------------------------------------
    _interpolate(tokenizer, real[:2].to(device), out_dir / "03_interpolation.png")

    # ---- 4. random codes ---------------------------------------------------
    _random_codes(tokenizer, out_dir / "04_random_codes.png", device)

    return metrics


def _plot_codebook(histogram: torch.Tensor, path: Path, metrics: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = histogram.numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(len(counts)), sorted(counts, reverse=True), width=1.0)
    ax1.set_yscale("log")
    ax1.set_xlabel("codebook entry (sorted by use)")
    ax1.set_ylabel("times used (log)")
    ax1.set_title(
        f"codebook usage: {metrics['codebook_use_total']:.1%} alive, "
        f"{metrics['dead_codes']} dead"
    )

    share = sorted(counts, reverse=True)
    cumulative = (torch.tensor(share).cumsum(0) / max(counts.sum(), 1)).numpy()
    ax2.plot(cumulative)
    ax2.axhline(0.9, ls="--", c="r", lw=1)
    ax2.set_xlabel("number of codes")
    ax2.set_ylabel("cumulative share of tokens")
    ax2.set_title("concentration (steep = collapsed)")
    ax2.set_ylim(0, 1.02)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def _interpolate(tokenizer, pair: torch.Tensor, path: Path, steps: int = 8) -> None:
    """Interpolate in the tokenizer's CONTINUOUS latent space, then decode.

    Interpolating quantized codes is meaningless -- code index 5 is not "between"
    4 and 6. So this walks the pre-quantization latents, which is why the result
    shows whether the latent geometry is smooth rather than whether the codebook
    is ordered.
    """
    a, b = pair[0:1], pair[1:2]
    za = tokenizer.encode(a)["latents"]
    zb = tokenizer.encode(b)["latents"]

    frames = []
    for alpha in torch.linspace(0, 1, steps):
        z = (1 - alpha) * za + alpha * zb
        frames.append(tokenizer.decode(z)[0].cpu())
    _save(torch.stack(frames), path, nrow=steps)


@torch.no_grad()
def _random_codes(tokenizer, path: Path, device: torch.device, n: int = 8) -> None:
    """Decode uniformly random token grids.

    Expect noise. That is the point: without a learned prior over token grids,
    the tokenizer cannot generate a room -- it can only round-trip one. This
    image is the visual argument for why stage C exists.
    """
    spec = tokenizer.latent_spec
    indices = torch.randint(0, spec.vocab_size, (n, 1, *spec.grid), device=device)
    _save(tokenizer.decode_indices(indices)[:, 0].cpu(), path, nrow=4)


# --------------------------------------------------------------------- stage C


@torch.no_grad()
def eval_dynamics(model, loader, device, out_dir: Path, n_steps: int = 8) -> dict:
    from world_model.eval import action_sweep, delta_psnr, rollout_metrics, save_rollout_video

    batch = next(iter(loader))
    frames = batch["frames"].to(device)

    metrics = rollout_metrics(model, frames, context_len=2)
    metrics.update(delta_psnr(model, frames))

    result = model.imagine(frames[:1, :2], actions=0, n_steps=n_steps)
    save_rollout_video(result.frames[0], out_dir / "05_rollout", fps=5)
    _save(result.frames[0].cpu(), out_dir / "05_rollout_strip.png", nrow=n_steps)

    if model.action_spec.kind == "latent":
        sweep = action_sweep(model, frames[:1, :2], n_steps=n_steps)
        _save(sweep.flatten(0, 1).cpu(), out_dir / "06_action_sweep.png", nrow=n_steps)
    return metrics


# ------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="a", choices=["a", "c"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, help="path or stage:best shorthand")
    parser.add_argument("--tokenizer_ckpt", default=None)
    parser.add_argument("--latent_action_ckpt", default=None)
    parser.add_argument("--out", default=None, help="output directory for images")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device("auto")
    print(f"device: {describe_device(device)}")

    cfg = load_config(Path("world_model") / args.config
                      if not Path(args.config).exists() else args.config)
    cfg = apply_overrides(cfg, args.overrides)
    out_dir = Path(args.out or f"outputs/eval_{args.stage}")

    _, val_loader = build_loaders(cfg, args.batch_size, cfg.get("seed", 0))
    if val_loader is None:
        raise SystemExit("no validation split -- lower data.val_fraction is not the issue; "
                         "check that the dataset has enough samples")
    loader = _limit(val_loader, args.max_batches)

    if args.stage == "a":
        tokenizer = wm.TOKENIZERS.build(cfg["tokenizer"])
        path = resolve_ckpt(args.ckpt, project="world_model")
        load_component(tokenizer, path, expect_component="tokenizer")
        print(f"tokenizer <- {path}")
        metrics = eval_tokenizer(tokenizer, loader, device, out_dir)
    else:
        model = wm.GenieWorldModel.from_checkpoints(
            cfg,
            tokenizer_ckpt=args.tokenizer_ckpt,
            latent_action_ckpt=args.latent_action_ckpt,
            dynamics_ckpt=args.ckpt,
        ).to(device)
        print(model.describe())
        metrics = eval_dynamics(model, loader, device, out_dir)

    print(f"\nmetrics")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<22} {v:.4f}" if isinstance(v, float) else f"  {k:<22} {v}")

    print(f"\nimages written to {out_dir}/")
    for p in sorted(out_dir.glob("*")):
        print(f"  {p.name}")
    return 0


def _limit(loader, n):
    """Yield at most n batches -- keeps eval fast on a large val split."""
    class _Limited:
        def __iter__(self):
            for i, b in enumerate(loader):
                if i >= n:
                    break
                yield b

        def __len__(self):
            return min(n, len(loader))

    return _Limited()


if __name__ == "__main__":
    raise SystemExit(main())
