"""Long-horizon rollout quality.

Two failure modes to measure separately, because they need different fixes:

* **Drift** -- image statistics degrade step by step until the frame is mush.
  Caused by compounding one-step error; fixed by better dynamics, scheduled
  sampling, or diffusion forcing.
* **Forgetting** -- the model is still producing sharp frames, but of the wrong
  room. Caused by finite context; fixed by the `memory/` slot, not by a better
  dynamics model.

`rollout_metrics` reports per-step PSNR against ground truth (drift) and
`revisit_consistency` compares the frame after returning to a viewpoint against
the frame from the first visit (forgetting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from common.metrics import psnr, ssim


@torch.no_grad()
def rollout_metrics(
    world_model,
    frames: torch.Tensor,
    context_len: int = 4,
    actions: Optional[torch.Tensor] = None,
    render: str = "tokenizer",
) -> Dict[str, float]:
    """Open-loop rollout against a real clip.

    frames: (B, T, 3, H, W). The first `context_len` frames prompt the model; the
    remainder are ground truth for the imagined steps.

    Per-step PSNR is expected to fall -- video prediction is stochastic and exact
    pixel match is not the goal past a few frames. What matters is the *shape* of
    the curve: a cliff in the first two steps means the dynamics model is broken;
    a slow decay is normal.
    """
    n_steps = frames.shape[1] - context_len
    if n_steps < 1:
        raise ValueError("clip is shorter than context_len + 1")

    if actions is None and world_model.action_spec.kind == "latent":
        lam = world_model.latent_action
        if lam is None:
            raise ValueError("pass `actions`, or load the LAM so true actions can be inferred")
        actions = lam.infer_actions(frames)["indices"][:, context_len - 1 :]

    result = world_model.imagine(
        frames[:, :context_len], actions=actions, n_steps=n_steps, render=render
    )
    target = frames[:, context_len:]

    per_step = [psnr(result.frames[:, i : i + 1], target[:, i : i + 1]) for i in range(n_steps)]
    return {
        "psnr_mean": sum(per_step) / n_steps,
        "psnr_step1": per_step[0],
        "psnr_final": per_step[-1],
        "psnr_decay": per_step[0] - per_step[-1],
        "ssim_mean": ssim(result.frames, target),
        **{f"psnr_step{i+1}": v for i, v in enumerate(per_step)},
    }


@torch.no_grad()
def revisit_consistency(
    world_model,
    context_frames: torch.Tensor,
    out_and_back: List[int],
    render: str = "tokenizer",
) -> Dict[str, float]:
    """Walk away and come back: is the scene the same?

    `out_and_back` is an action sequence that should return the camera to its
    starting viewpoint (e.g. `[1]*8 + [2]*8` if code 1 pans left and code 2 pans
    right). Compares the final imagined frame against the last context frame.

    This is the direct test of the Genie 3 consistency claim, and the metric the
    `memory/` slot exists to move. Expect a low score from a plain causal
    transformer whose context window is shorter than the excursion.
    """
    result = world_model.imagine(
        context_frames, actions=out_and_back, n_steps=len(out_and_back), render=render
    )
    start = context_frames[:, -1:]
    end = result.frames[:, -1:]
    return {"revisit_psnr": psnr(end, start), "revisit_ssim": ssim(end, start)}


def save_rollout_video(frames: torch.Tensor, path: str | Path, fps: int = 10) -> Path:
    """Write (T, 3, H, W) in [-1, 1] to an mp4 (falls back to a PNG contact sheet)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    video = ((frames.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu()

    try:
        import imageio.v3 as iio

        iio.imwrite(path.with_suffix(".mp4"), video.numpy(), fps=fps)
        return path.with_suffix(".mp4")
    except Exception:
        import torchvision

        grid = torchvision.utils.make_grid(frames.clamp(-1, 1) * 0.5 + 0.5, nrow=8)
        torchvision.utils.save_image(grid, path.with_suffix(".png"))
        return path.with_suffix(".png")
