"""Do actions actually control the world model?

This is the question that separates a *world model* from a *video predictor*, and
it is not answered by any loss curve. A dynamics model that ignores its action
input entirely can have a perfectly healthy training loss, because next-frame
prediction is dominated by "most of the scene stays the same".

Two probes:

* `delta_psnr` -- Genie's Delta_t PSNR. Predict frame t+1 conditioned on the true
  latent action, and again on a random one. Report the PSNR gap. Genie reports
  1.91 for the pixel-input LAM vs 1.33 for token-input, so single-digit values
  are the expected scale. A value near 0 means the action channel is dead.
* `action_sweep` -- qualitative. Hold a context clip fixed, roll forward under
  each of the |A| codes, and lay the results out as a grid. This is how you find
  out what the codes *mean*: with room video you typically see codes specialise
  into pan-left / pan-right / move-forward / no-op.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from common.metrics import psnr


@torch.no_grad()
def delta_psnr(
    world_model,
    frames: torch.Tensor,
    latent_action_model=None,
    n_random: int = 1,
) -> Dict[str, float]:
    """PSNR(prediction | true action) - PSNR(prediction | random action).

    Args:
        frames: (B, T, 3, H, W) real clip. The last frame is the target.
        latent_action_model: supplies the true action. Defaults to the world
            model's own LAM; required because the LAM is not part of the
            inference path and may not be loaded.
    """
    lam = latent_action_model or world_model.latent_action
    if lam is None:
        raise ValueError("delta_psnr needs a latent action model to label the true action")

    tokenizer, dynamics = world_model.tokenizer, world_model.dynamics
    tokens = tokenizer.encode(frames)["indices"]
    true_actions = lam.infer_actions(frames)["indices"]

    context, target = tokens[:, :-1], frames[:, -1:]
    a_true = true_actions[:, -1]
    num_actions = lam.action_spec.num_actions

    pred_true = dynamics.predict_next(context, action=a_true)
    psnr_true = psnr(tokenizer.decode_indices(pred_true), target)

    psnr_rand = 0.0
    for _ in range(n_random):
        a_rand = torch.randint_like(a_true, num_actions)
        pred_rand = dynamics.predict_next(context, action=a_rand)
        psnr_rand += psnr(tokenizer.decode_indices(pred_rand), target)
    psnr_rand /= max(n_random, 1)

    return {
        "delta_psnr": psnr_true - psnr_rand,
        "psnr_true_action": psnr_true,
        "psnr_random_action": psnr_rand,
    }


@torch.no_grad()
def action_sweep(
    world_model,
    context_frames: torch.Tensor,
    n_steps: int = 8,
    actions: Optional[List[int]] = None,
    render: str = "tokenizer",
) -> torch.Tensor:
    """Roll the same context forward under every action code.

    Returns (|A|, n_steps, 3, H, W) for a batch of one -- ready to write out as a
    grid of videos, one row per code.
    """
    spec = world_model.action_spec
    if spec.kind != "latent":
        raise ValueError(f"action_sweep is for latent action spaces, got {spec.kind}")
    codes = actions if actions is not None else list(range(spec.num_actions or 0))

    if context_frames.shape[0] != 1:
        context_frames = context_frames[:1]

    rows = []
    for code in codes:
        result = world_model.imagine(
            context_frames, actions=code, n_steps=n_steps, render=render
        )
        rows.append(result.frames[0])
    return torch.stack(rows)
