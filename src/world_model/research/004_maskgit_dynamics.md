# 004 -- MaskGIT dynamics

**Status**: idea
**Slot**: `dynamics/` (register as `maskgit_st`)
**Cost**: ~3 days, stage-C rerun
**Depends on**: W3 (causal_gpt clearly beating the copy baseline)

## Claim

Iterative parallel decoding produces frames that are coherent *within* the frame,
which the causal GPT baseline structurally cannot, and it does so with far fewer
forward passes per frame.

## Why now

`causal_gpt.predict_next` draws all `h*w` tokens of the next frame from a single
forward pass. They are therefore conditionally independent given the history --
the model has no way to make the left half of the frame agree with the right half
beyond what the history already determines. Expect visibly incoherent frames, and
expect that no amount of training fixes it, because it is a sampling-procedure
problem rather than a model-capacity problem.

Genie's numbers: mask rate sampled uniformly from [0.5, 1] during training, 25
MaskGIT steps at inference, temperature 2.0.

## Design

Training: bidirectional attention within a frame, causal across frames (ST
blocks). Mask a Bernoulli fraction of the target frame's tokens; predict the
masked ones from the unmasked ones plus the history plus the action.

Inference, per frame: start fully masked; over 25 rounds, predict all tokens,
commit the most confident `k` according to a cosine schedule, re-mask the rest,
repeat. Each round conditions on what is already committed -- that is where
within-frame coherence comes from.

At a 16x16 grid this is 25 forwards per frame vs. 1 for the flat baseline but
with far better samples; an autoregressive-within-frame model would need 256.

## Measurement

Stage C, same tokenizer and LAM checkpoints, 3 seeds.

| Metric | `causal_gpt` | target |
|--------|-------------|--------|
| `val/delta_psnr` | TBD | **+0.3 absolute** |
| `val/psnr_step1` (rollout) | TBD | +1.0 dB |
| `val/psnr_decay` over 8 steps | TBD | lower |
| wall-clock per imagined frame | ~1 forward | <= 25 forwards |
| qualitative within-frame coherence | judged by eye | clearly better |

## Kill criterion

If `delta_psnr` and rollout PSNR are both within noise of the baseline across 3
seeds, the bottleneck is elsewhere (probably the tokenizer or the LAM) -- go fix
that first and revisit.

## Result

*(pending)*
