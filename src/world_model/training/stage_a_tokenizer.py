"""Stage A -- train the video tokenizer.

Trains on single frames or clips, whichever the dataset yields, so LSUN rooms and
TUM sequences can both feed it.

**How to know stage A is done.** Not by reconstruction loss alone. Watch three
numbers together:

| Metric            | Healthy        | What a bad value means                     |
|-------------------|----------------|--------------------------------------------|
| `val/psnr`        | rising, > ~22  | reconstruction is usable                    |
| `val/codebook_use`| > 0.5          | below ~0.2 the codebook has collapsed and   |
|                   |                | stage C will have almost no vocabulary      |
| `val/perplexity`  | -> vocab_size  | near 1 means every patch maps to one code   |

A collapsed codebook with good PSNR is the trap: the encoder has learned to route
everything through a handful of codes and let the decoder memorise, and the
dynamics model then has nothing to predict. Fix it with a different quantizer
(`tokenizer.quantizer.name: fsq`) rather than more epochs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.metrics import codebook_usage, perplexity, psnr, ssim
from common.stages import STAGES, Stage, move_to_device
from world_model.core.registry import TOKENIZERS

from .data import build_loaders


@STAGES.register("stage_a_tokenizer")
class TokenizerStage(Stage):
    name = "stage_a_tokenizer"
    component = "tokenizer"
    requires = ()
    monitor = "val/psnr"
    monitor_mode = "max"

    def build(self) -> nn.Module:
        return TOKENIZERS.build(self.cfg["tokenizer"])

    def build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        return build_loaders(self.cfg, self.cfg["batch_size"], self.cfg.get("seed", 0))

    def loss(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, loss, metrics = model(batch["frames"])
        return loss, metrics

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        totals: Dict[str, float] = {}
        count = 0
        vocab = model.latent_spec.vocab_size or 1
        for batch in loader:
            batch = move_to_device(batch, self.device)
            frames = batch["frames"]
            enc = model.encode(frames)
            recon = model.decode(enc["latents"])
            step = {
                "loss": float(torch.nn.functional.mse_loss(recon, frames)),
                "psnr": psnr(recon, frames),
                "ssim": ssim(recon, frames),
                "codebook_use": codebook_usage(enc["indices"], vocab),
                "perplexity": perplexity(enc["indices"], vocab),
            }
            for k, v in step.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1
        model.train()
        return {f"val/{k}": v / max(count, 1) for k, v in totals.items()}
