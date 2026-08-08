"""The `Stage` contract -- one trainable component, one stage, one checkpoint.

Everything in this repo is trained stage by stage rather than end to end, so the
stage is the unit the CLI, the checkpoint manager and the docs all agree on.

A stage declares:
  * `name`                 -- also its checkpoint directory
  * `component`            -- what it trains ("tokenizer", "dynamics", "octo", ...)
  * `requires`             -- stages whose checkpoints must be loaded first
  * `build()`              -- construct model + frozen dependencies
  * `build_dataloaders()`  -- train/val loaders
  * `loss()`               -- one batch -> (scalar loss, metrics dict)
  * `evaluate()`           -- optional, richer periodic validation

`STAGES` is the registry the CLIs dispatch through, so adding a research idea that
needs its own training procedure means adding one file, not editing any script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .registry import Registry

STAGES = Registry("stage")


@dataclass
class StageContext:
    """Runtime handles a stage needs but should not construct itself."""

    cfg: Dict[str, Any]
    device: torch.device
    run_name: str
    project: str
    parent_ckpts: Dict[str, str] = field(default_factory=dict)  # role -> path
    output_dir: Optional[str] = None


class Stage(ABC):
    """Base class for a single trainable component."""

    #: stage identifier; must match the registry key
    name: str = "unnamed_stage"
    #: which component this stage produces a checkpoint for
    component: str = "unknown"
    #: stage names whose checkpoints must be supplied via `ctx.parent_ckpts`
    requires: Sequence[str] = ()
    #: metric the checkpoint manager tracks for `best.pt`
    monitor: str = "val/loss"
    monitor_mode: str = "min"

    def __init__(self, ctx: StageContext) -> None:
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.device = ctx.device

    # ------------------------------------------------------------------ required

    @abstractmethod
    def build(self) -> nn.Module:
        """Return the module to be trained.

        Frozen dependencies (a tokenizer for a dynamics stage, say) are loaded
        here from `self.ctx.parent_ckpts` and stored on `self`, *not* returned --
        only the returned module's parameters are optimised and checkpointed.
        """

    @abstractmethod
    def build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Return `(train_loader, val_loader_or_None)`."""

    @abstractmethod
    def loss(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the training loss for one batch.

        Returns the scalar to backprop and a dict of scalars to log. The dict must
        not contain tensors that hold graph references.
        """

    # ------------------------------------------------------------------ optional

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        """Default validation: mean of `loss()` over the loader.

        Override to add stage-specific metrics (PSNR for the tokenizer,
        controllability for the dynamics model, success rate for a policy).
        """
        model.eval()
        totals: Dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = move_to_device(batch, self.device)
            loss, metrics = self.loss(model, batch)
            metrics = {"loss": float(loss.detach()), **metrics}
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            count += 1
        model.train()
        return {f"val/{k}": v / max(count, 1) for k, v in totals.items()}

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        opt_cfg = self.cfg.get("optim", {})
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": opt_cfg.get("weight_decay", 1e-2)},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=opt_cfg.get("lr", 1e-4),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        )

    def build_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        sched_cfg = self.cfg.get("optim", {}).get("schedule", "cosine")
        if sched_cfg in (None, "none", "constant"):
            return None
        warmup = int(self.cfg.get("optim", {}).get("warmup_steps", 0))

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return step / max(warmup, 1)
            if sched_cfg != "cosine":
                return 1.0
            progress = (step - warmup) / max(total_steps - warmup, 1)
            import math

            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def on_epoch_end(self, model: nn.Module, epoch: int, metrics: Mapping[str, float]) -> None:
        """Hook for sampling images/rollouts. Default: nothing."""

    def extra_state(self) -> Dict[str, Any]:
        """Extra things to store in the checkpoint (e.g. dataset normalisation)."""
        return {}


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a nested container onto `device`."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [move_to_device(v, device) for v in batch]
        return type(batch)(moved) if not isinstance(batch, tuple) else tuple(moved)
    return batch
