"""One training loop, shared by every stage of both projects.

Deliberately small: AMP, gradient accumulation, grad clipping, EMA, periodic
validation, checkpointing, and logging. Anything model-specific belongs in a
`Stage`, not here -- that separation is what lets a new research idea reuse the
whole harness by implementing three methods.

Sized for a single 8 GB RTX 4070 Laptop: bf16 autocast by default, gradient
accumulation instead of large batches, and `torch.cuda.max_memory_allocated`
reported every epoch so a config that will OOM shows up in the first epoch.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .checkpoint import CheckpointManager, CheckpointMeta
from .config import config_hash
from .logging import RunLogger
from .stages import Stage, move_to_device


class EMA:
    """Exponential moving average of model weights, kept on CPU to save VRAM."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float().cpu()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach().float().cpu(), alpha=1.0 - self.decay
                )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow


def train_stage(
    stage: Stage,
    *,
    epochs: int,
    grad_accum: int = 1,
    clip_grad_norm: float = 1.0,
    precision: str = "bf16",
    log_every: int = 50,
    eval_every_epochs: int = 1,
    save_every_epochs: int = 1,
    ema_decay: float = 0.0,
    resume: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one stage to completion and return its final metrics."""
    device = stage.device
    model = stage.build().to(device)
    train_loader, val_loader = stage.build_dataloaders()

    steps_per_epoch = max(len(train_loader) // grad_accum, 1)
    total_steps = steps_per_epoch * epochs

    optimizer = stage.build_optimizer(model)
    scheduler = stage.build_scheduler(optimizer, total_steps)
    ema = EMA(model, ema_decay) if ema_decay > 0 else None

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and use_amp)

    ckpt = CheckpointManager(
        stage=stage.name,
        run_name=stage.ctx.run_name,
        project=stage.ctx.project,
        monitor=stage.monitor,
        mode=stage.monitor_mode,
    )
    ckpt.save_config(stage.cfg)
    logger = RunLogger(ckpt.dir, stage.cfg)
    logger.info(
        f"stage={stage.name} component={stage.component} "
        f"params={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M "
        f"steps/epoch={steps_per_epoch} total={total_steps}"
    )

    start_epoch, global_step = 0, 0
    if resume:
        from .checkpoint import load_checkpoint

        state, meta = load_checkpoint(resume)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch, global_step = meta.epoch + 1, meta.step
        logger.info(f"resumed from {resume} at epoch {start_epoch} step {global_step}")

    model.train()
    metrics: Dict[str, float] = {}

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        running: Dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            batch = move_to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, batch_metrics = stage.loss(model, batch)
            scaler.scale(loss / grad_accum).backward()

            if (i + 1) % grad_accum == 0:
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip_grad_norm
                    )
                    batch_metrics["grad_norm"] = float(grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)
                global_step += 1

                for k, v in {"loss": float(loss.detach()), **batch_metrics}.items():
                    running[k] = running.get(k, 0.0) + float(v)

                if global_step % log_every == 0:
                    avg = {k: v / log_every for k, v in running.items()}
                    avg["lr"] = optimizer.param_groups[0]["lr"]
                    logger.log_scalars("train", avg, global_step)
                    running = {}

        metrics = {}
        if val_loader is not None and (epoch + 1) % eval_every_epochs == 0:
            metrics = stage.evaluate(model, val_loader)
            logger.log_scalars("", metrics, global_step)

        stage.on_epoch_end(model, epoch, metrics)

        peak = (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        )
        logger.info(
            f"epoch {epoch+1}/{epochs} "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            + f" | {time.time()-epoch_start:.0f}s | peak {peak:.2f} GiB"
        )

        if (epoch + 1) % save_every_epochs == 0 or epoch == epochs - 1:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            if ema is not None:
                state["model_ema"] = ema.state_dict()
            state.update(stage.extra_state())
            ckpt.save(
                state,
                CheckpointMeta(
                    component=stage.component,
                    stage=stage.name,
                    step=global_step,
                    epoch=epoch,
                    metrics={k: float(v) for k, v in metrics.items()},
                    config_hash=config_hash(stage.cfg),
                    parent=stage.ctx.parent_ckpts.get("init"),
                    frozen_parents=dict(stage.ctx.parent_ckpts),
                ),
            )
            ckpt.prune(keep_last=stage.cfg.get("keep_last_checkpoints", 3))

    logger.close()
    return metrics
