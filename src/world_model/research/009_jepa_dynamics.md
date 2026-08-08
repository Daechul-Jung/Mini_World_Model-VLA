# 009 -- JEPA-style latent dynamics

**Status**: idea
**Slot**: `dynamics/` (register as `jepa`), plus a continuous tokenizer
**Cost**: ~1 week; it is a different training objective, not a layer swap
**Depends on**: W4 with the discrete baseline, for a fair comparison

## Claim

Predicting in a continuous latent space with a joint-embedding objective avoids
codebook collapse entirely and may suit the small-data regime better than
discrete next-token prediction.

## Why now

Discrete tokenisation is Genie's choice and it enables MaskGIT, but it brings
codebook collapse, quantisation error, and a vocabulary that has to be learned
before dynamics can be learned. JEPA (LeCun; I-JEPA, V-JEPA) sidesteps all three:
predict the *representation* of the next frame rather than its tokens or its
pixels, with a stop-gradient target encoder.

It also removes the "predict every pixel/token equally" problem. Most tokens in a
video do not change, so cross-entropy is dominated by the easy ones -- exactly
what `copy_baseline_acc` measures.

## Design

The contracts already allow this. `LatentSpec(discrete=False)` and
`Dynamics.accepts_discrete = False` are in `core/base.py` precisely so a
continuous path fits without touching `GenieWorldModel`.

* **Tokenizer**: `continuous_vae` -- same conv encoder, no quantizer, KL or plain
  bottleneck. `LatentSpec(discrete=False)`.
* **Dynamics**: predict `z_{t+1}` from `z_{<=t}` and `a_t`, trained against a
  stop-gradient EMA target encoder, cosine or smooth-L1 loss. Add VICReg-style
  variance/covariance terms -- without them, representation collapse (everything
  maps to one vector) is the expected failure, and it is the direct analogue of
  codebook collapse.
* **Rendering**: the VAE decoder, or the stage-D diffusion decoder.

## The catch, stated up front

The world model is meant to serve as an RL environment, which needs *pixels* for
the VLA to see. A JEPA model predicts representations, so it needs a decoder
anyway -- and a decoder trained on VAE latents may render predicted latents
poorly, since predicted and encoded latents are not the same distribution. Budget
for this; it is the main risk.

## Measurement

Against the discrete baseline, same data, 3 seeds.

| Metric | discrete | JEPA |
|--------|----------|------|
| `val/delta_psnr` | | |
| `psnr_step1` / `psnr_final` | | |
| representation variance (collapse check) | n/a | must stay > threshold |
| render quality of *predicted* latents | | |

## Kill criterion

If representation variance collapses despite VICReg terms, or if rendered
predicted latents are clearly worse than the discrete path's, stop -- the RL-env
requirement makes render quality non-negotiable here.

## Result

*(pending)*
