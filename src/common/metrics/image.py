"""Reconstruction and tokenizer-health metrics.

`codebook_usage` and `perplexity` matter more than PSNR when debugging stage A:
a VQ-VAE that reaches good PSNR while using 40 of 1024 codes has collapsed, and
the dynamics model trained on those tokens will never recover.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

_LPIPS_NET = None


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Accept [-1, 1] or [0, 1] and return [0, 1]."""
    return (x + 1) / 2 if x.min() < -0.01 else x


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    pred, target = _to_01(pred), _to_01(target)
    mse = F.mse_loss(pred, target).clamp_min(1e-10)
    return float(10 * torch.log10(max_val**2 / mse))


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5) -> float:
    """Gaussian-window SSIM averaged over channels. Inputs (..., C, H, W)."""
    pred, target = _to_01(pred), _to_01(target)
    if pred.ndim > 4:
        pred = pred.flatten(0, pred.ndim - 4)
        target = target.flatten(0, target.ndim - 4)

    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    kernel = (g.t() @ g).expand(pred.shape[1], 1, window, window).contiguous()

    def _blur(x):
        return F.conv2d(x, kernel, padding=window // 2, groups=x.shape[1])

    mu_x, mu_y = _blur(pred), _blur(target)
    mu_xx, mu_yy, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x = _blur(pred**2) - mu_xx
    sigma_y = _blur(target**2) - mu_yy
    sigma_xy = _blur(pred * target) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_xx + mu_yy + c1) * (sigma_x + sigma_y + c2)
    )
    return float(s.mean())


def lpips(pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    """Perceptual distance. Returns None if the `lpips` package is unavailable."""
    global _LPIPS_NET
    if _LPIPS_NET is None:
        try:
            import lpips as _lpips_pkg

            _LPIPS_NET = _lpips_pkg.LPIPS(net="alex").eval()
        except Exception:
            _LPIPS_NET = False
    if _LPIPS_NET is False:
        return None
    net = _LPIPS_NET.to(pred.device)
    with torch.no_grad():
        return float(net(pred.clamp(-1, 1), target.clamp(-1, 1)).mean())


def codebook_usage(indices: torch.Tensor, codebook_size: int) -> float:
    """Fraction of codebook entries used in this batch. Healthy VQ-VAE: > 0.5."""
    return float(torch.unique(indices).numel()) / codebook_size


def perplexity(indices: torch.Tensor, codebook_size: int) -> float:
    """exp(entropy) of the code histogram. Equals codebook_size when uniform."""
    counts = torch.bincount(indices.flatten(), minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1)
    entropy = -(probs * (probs + 1e-10).log()).sum()
    return float(entropy.exp())
