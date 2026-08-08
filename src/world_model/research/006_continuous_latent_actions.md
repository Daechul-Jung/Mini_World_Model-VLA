# 006 -- Continuous or larger latent action spaces

**Status**: idea
**Slot**: `latent_action/`
**Cost**: ~2 days
**Depends on**: W2, and the bridge's action-space problem being live

## Claim

|A| = 8 discrete codes cannot express a 7-DoF continuous robot action, so any
attempt to drive the world model from a VLA through latent codes is lossy by
construction. A larger or continuous action bottleneck reduces that loss.

## Why now

Genie uses |A| = 8 because its actions are game-like (up/down/left/right/jump)
and because a small set is interpretable -- you can enumerate all eight and see
what each does. For robotics that same smallness is the problem.

Note the tension, and do not resolve it by reflex: a bigger bottleneck lets more
information through, including *appearance* information that is not action at
all. The LAM's entire value comes from the bottleneck being narrow enough that
only the controllable part survives. Widen it too far and the "action" becomes a
compressed next frame, the dynamics model becomes a copier, and `delta_psnr`
paradoxically rises while the model becomes useless.

## Design

Three variants, all satisfying `LatentActionModel`:

1. **Larger |A|** (32, 64, 256). One config line. Do this first -- it is free and
   it maps the trade-off curve.
2. **Continuous action** -- drop the VQ, keep a low-dimensional continuous
   bottleneck with a KL or variance penalty. `ActionSpaceSpec(kind="robot",
   dim=d)` already covers the downstream side.
3. **Residual VQ** -- multiple quantization stages, so the action is a short
   sequence of codes rather than one. Expressive and still discrete.

## Measurement

Sweep |A| in {8, 16, 32, 64, 256} plus the continuous variant, 3 seeds.

| \|A\| | `actions_used` | `action_perplexity` | stage-C `delta_psnr` | `copy_baseline_acc` gap |
|------|---------------|--------------------|--------------------|------------------------|
| 8 | | | | |
| 32 | | | | |
| ... | | | | |

The fourth column is the guard against the failure above: if the dynamics model's
advantage over copying *shrinks* as |A| grows, the bottleneck is leaking
appearance.

Also relevant, and the reason this idea exists: `LearnedActionProjector.fit()`'s
`lift_over_chance` on OpenX data, as a function of |A|.

## Kill criterion

If `lift_over_chance` stays near zero at every |A|, robot actions and latent
actions are simply not alignable on this data -- abandon the translation route
entirely and commit to `action_kind="robot"` dynamics (bridge ADR-B02).

## Result

*(pending)*
