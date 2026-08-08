"""Stage D -- train the diffusion decoder (optional).

Learns to render pixels from *frozen stage-A latents*, not from dynamics
predictions. That separation is deliberate: stage D's job is rendering quality,
and asking it to also compensate for dynamics error would make its loss depend on
stage C and destroy the "one stage, one responsibility" property.

Skip this stage entirely until stages A-C produce sensible rollouts. It is the
most expensive stage per unit of research value: it makes videos prettier without
making the world model more correct or more steerable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.checkpoint import load_component, resolve_ckpt
from common.stages import STAGES, Stage
from world_model.core.registry import DECODERS, TOKENIZERS

from .data import build_loaders


@STAGES.register("stage_d_decoder")
class DecoderStage(Stage):
    name = "stage_d_decoder"
    component = "decoder"
    requires = ("stage_a_tokenizer",)
    monitor = "val/loss"
    monitor_mode = "min"

    def build(self) -> nn.Module:
        self.tokenizer = TOKENIZERS.build(self.cfg["tokenizer"]).to(self.device)
        tok_ckpt = self.ctx.parent_ckpts.get("tokenizer")
        if tok_ckpt is None:
            raise ValueError("stage D needs --tokenizer_ckpt (or stage_a_tokenizer:best)")
        load_component(
            self.tokenizer, resolve_ckpt(tok_ckpt), freeze=True, expect_component="tokenizer"
        )

        dec_cfg = dict(self.cfg["decoder"])
        dec_cfg.setdefault("context_dim", self.tokenizer.latent_spec.dim)
        return DECODERS.build(dec_cfg)

    def build_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        return build_loaders(self.cfg, self.cfg["batch_size"], self.cfg.get("seed", 0))

    def loss(self, model: nn.Module, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        frames = batch["frames"]
        with torch.no_grad():
            latents = self.tokenizer.encode(frames)["latents"]
        return model(frames, latents)
