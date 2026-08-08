"""Stage-A evaluation: reconstruction quality and codebook health."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from common.metrics import codebook_usage, lpips, perplexity, psnr, ssim


@torch.no_grad()
def evaluate_reconstruction(
    tokenizer, loader: DataLoader, device: torch.device, max_batches: int = 50
) -> Dict[str, float]:
    """Reconstruct held-out frames and report both image and codebook metrics.

    Report all of them. A tokenizer with PSNR 28 and `codebook_use` 0.05 is worse
    for this project than one with PSNR 24 and `codebook_use` 0.7, because the
    dynamics model can only work with the vocabulary the tokenizer actually uses.
    """
    tokenizer.eval().to(device)
    vocab = tokenizer.latent_spec.vocab_size or 1
    totals: Dict[str, float] = {}
    count = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        frames = batch["frames"].to(device)
        enc = tokenizer.encode(frames)
        recon = tokenizer.decode(enc["latents"])

        step = {
            "psnr": psnr(recon, frames),
            "ssim": ssim(recon, frames),
            "codebook_use": codebook_usage(enc["indices"], vocab),
            "perplexity": perplexity(enc["indices"], vocab),
        }
        perceptual = lpips(recon.flatten(0, 1) if recon.ndim == 5 else recon,
                          frames.flatten(0, 1) if frames.ndim == 5 else frames)
        if perceptual is not None:
            step["lpips"] = perceptual

        for k, v in step.items():
            totals[k] = totals.get(k, 0.0) + v
        count += 1

    return {k: v / max(count, 1) for k, v in totals.items()}
