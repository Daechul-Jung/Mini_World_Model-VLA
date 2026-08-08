"""Measure what a config actually costs before committing a training run to it.

    python scripts/tools/vram_probe.py --project vla --config octo_small.yaml
    python scripts/tools/vram_probe.py --project world_model --config genie_small.yaml --stage c

Builds the model, runs a few forward/backward steps on synthetic data, and
reports peak VRAM and step time. On a 7.7 GiB card this is the difference between
finding out now and finding out forty minutes into an epoch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from common.config import apply_overrides, load_config
from common.seeding import describe_device, get_device


def probe_vla(cfg: dict, device: torch.device, batch_size: int, steps: int,
              precision: str = "bf16") -> None:
    import vla  # noqa: F401
    from common.types import Observation

    policy = vla.POLICIES.build(cfg["policy"]).to(device)
    if cfg.get("freeze_backbone"):
        policy.freeze_backbone(True)
    print(policy.param_summary())

    spec = policy.spec
    obs = Observation(
        image=torch.randn(batch_size, spec.obs_horizon, 3, spec.image_size, spec.image_size,
                          device=device),
        instruction=["pick up the red object"] * batch_size,
    )
    target = torch.randn(batch_size, spec.action_chunk, spec.action_dim, device=device).tanh()
    opt = torch.optim.AdamW(policy.trainable_parameters(), lr=1e-4)

    _run(steps, device, lambda: policy.loss(obs, target)[0], opt, precision)


def probe_world_model(cfg: dict, device: torch.device, batch_size: int, steps: int,
                      stage: str, precision: str = "bf16") -> None:
    import world_model as wm

    image_size = cfg.get("image_size", 128)
    data_cfg = cfg.get("data", {})
    # A stills dataset yields (B, 3, H, W); a clip dataset yields (B, T, 3, H, W).
    # Assuming clips for a stills config inflates the effective batch by `clip_len`
    # and reports a spurious OOM.
    clip_len = 1 if data_cfg.get("name") == "images" else data_cfg.get("clip_len", 8)
    frames = torch.randn(batch_size, clip_len, 3, image_size, image_size, device=device)
    print(f"input shape {tuple(frames.shape)}"
          + ("  (stills dataset -> T=1)" if clip_len == 1 else ""))

    if stage == "a":
        model = wm.TOKENIZERS.build(cfg["tokenizer"]).to(device)
        fn = lambda: model(frames)[1]
    elif stage == "b":
        lam_cfg = dict(cfg["latent_action"]); lam_cfg.pop("freeze", None)
        model = wm.LATENT_ACTIONS.build(lam_cfg).to(device)
        fn = lambda: model(frames)[1]
    else:
        tokenizer = wm.TOKENIZERS.build(cfg["tokenizer"]).to(device).eval()
        model = wm.DYNAMICS.build(cfg["dynamics"]).to(device)
        with torch.no_grad():
            tokens = tokenizer.encode(frames)["indices"]
        actions = torch.randint(0, cfg["dynamics"].get("num_actions", 8),
                                (batch_size, clip_len - 1), device=device)
        fn = lambda: model(tokens, actions)["loss"]

    print(f"{type(model).__name__}: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    _run(steps, device, fn, opt, precision)


def _step(loss: torch.Tensor, opt: torch.optim.Optimizer) -> None:
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def _run(steps: int, device: torch.device, fn, opt: torch.optim.Optimizer,
         precision: str = "bf16") -> None:
    """Time and measure `fn` under the SAME autocast the trainer will use.

    Probing in fp32 while `common/trainer.py` trains under bf16 autocast
    over-reports peak VRAM by roughly the activation memory -- enough to report
    an OOM for a config that trains fine.
    """
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    use_amp = amp_dtype is not None and device.type == "cuda"

    def _once() -> None:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            loss = fn()
        _step(loss, opt)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _once()                                         # warm up allocator + kernels
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(steps):
        _once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    total = (
        torch.cuda.get_device_properties(device).total_memory / 2**30
        if device.type == "cuda" else 0.0
    )
    print(f"\nprecision   {precision}")
    print(f"peak VRAM   {peak:.2f} GiB of {total:.1f} GiB ({100*peak/max(total,1e-9):.0f}%)")
    print(f"step time   {1000*elapsed/steps:.0f} ms")
    if total and peak > 0.85 * total:
        print("\nWARNING: over 85% of VRAM. Lower batch_size and raise grad_accum "
              "-- the effective batch is what matters, not the per-step batch.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, choices=["vla", "world_model"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="a", help="world_model only: a|b|c")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--precision", default=None, choices=["bf16", "fp16", "fp32"],
                        help="defaults to the config's precision, matching the trainer")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    path = Path(args.project) / args.config
    cfg = apply_overrides(load_config(path if not Path(args.config).exists() else args.config),
                          args.overrides)
    batch_size = args.batch_size or cfg.get("batch_size", 8)
    precision = args.precision or cfg.get("precision", "bf16")

    device = get_device("auto")
    print(f"device: {describe_device(device)}")
    print(f"config: {args.config}   batch_size: {batch_size}\n")

    if args.project == "vla":
        probe_vla(cfg, device, batch_size, args.steps, precision)
    else:
        probe_world_model(cfg, device, batch_size, args.steps, args.stage, precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
