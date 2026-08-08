# 005 -- Diffusion forcing / scheduled sampling

**Status**: idea
**Slot**: `dynamics/`
**Cost**: ~2 days
**Depends on**: an observed large `psnr_decay`

## Claim

Training the dynamics model on its own noisy predictions, rather than only on
ground-truth history, reduces the compounding error that makes long rollouts
drift to mush.

## Why now

Teacher forcing trains the model on perfect history and then asks it, at rollout
time, to consume its own imperfect output. The distributions differ, the error
compounds, and `psnr_decay` grows superlinearly with horizon. Classic exposure
bias.

Two remedies, cheapest first:

1. **Scheduled sampling** -- with probability `p` (annealed up), replace a history
   frame's tokens with the model's own sample. Trivial to add to stage C.
2. **Diffusion forcing** (Chen et al., 2024) -- give each frame in the context an
   independent noise level, so the model learns to predict from partially
   corrupted history at every corruption level. Strictly more general.

Run scheduled sampling first. If it closes most of the gap, diffusion forcing is
not worth the complexity.

## Measurement

`rollout_metrics` over 8, 16 and 64 steps, 3 seeds.

| Metric | teacher forcing | + scheduled sampling | + diffusion forcing |
|--------|----------------|---------------------|--------------------|
| `psnr_step1` | | | |
| `psnr_final` (64) | | | |
| `psnr_decay` | | | |
| `delta_psnr` | | | |

Watch `psnr_step1`: these methods typically trade a little one-step accuracy for
a lot of long-horizon stability. That trade is the point, but it should be
measured rather than assumed.

## Kill criterion

If `psnr_decay` over 64 steps does not improve by 20% relative, the failure is
forgetting rather than drift -- go to idea 007 instead.

## Result

*(pending)*
