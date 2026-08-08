# 008 -- Physics and geometry auxiliary heads

**Status**: idea
**Slot**: `physics/` (contract exists, no implementation)
**Cost**: ~2 days for depth; ~3 for pose
**Depends on**: W4

## Claim

Predicting depth and relative camera pose from the dynamics model's hidden states
forces those states to carry geometry, which improves rollout consistency -- at
zero inference cost, since the heads are dropped at rollout time.

## Why now

Genie 3 uses no physics engine; physical plausibility is learned from
internet-scale video. At 5000 stills and two desk sequences, next-token
prediction will learn texture statistics long before it learns that objects are
solid. If the data cannot supply the structure, inject it.

The supervision is **already on disk**: TUM RGB-D ships per-frame depth maps and
ground-truth camera poses. `VideoFolderDataset` already loads both when
`with_depth` / `with_pose` are set. This is the cheapest structured signal
available in this project.

## Design

`PhysicsHead` takes dynamics hidden states `(B, T, N, D)` and a batch, returns
`(loss, metrics)` already scaled by `loss_weight`. `required_keys` lets the
dataset refuse a stage whose supervision is missing, instead of silently training
on zeros.

Two heads to build, in order:

* **`depth_aux`** -- linear probe per spatial token to a depth value, scale-
  invariant loss. `required_keys = ("depth",)`.
* **`pose_aux`** -- pool over tokens, predict the relative pose between frame t
  and t+1 (translation + quaternion). `required_keys = ("pose",)`. This one has a
  second payoff: it grounds the latent actions in actual camera motion, which is
  directly useful when latent actions later have to be related to robot actions.

Setting `loss_weight: 0` must be exactly equivalent to not having the head.

## Measurement

Stage C, same checkpoints, 3 seeds, `loss_weight` in {0, 0.1, 1.0}.

| Metric | no aux | + depth | + depth + pose |
|--------|--------|---------|----------------|
| `val/delta_psnr` | | | |
| `psnr_decay` over 16 steps | | | |
| `revisit_psnr` | | | |
| depth probe error (diagnostic) | -- | | |

## Kill criterion

If `delta_psnr` and `psnr_decay` are unchanged at every `loss_weight`, the
features already carry the geometry (or the dynamics model is too small to use
it) -- close the idea and record the depth-probe error as a standalone
observation about what the features contain.

## Honest framing

These do not make the model "know physics". They make its features carry
geometry, which is a prerequisite. Say it that way.

## Result

*(pending)*
