# 007 -- Long-horizon memory

**Status**: idea
**Slot**: `memory/` (contract exists, no implementation)
**Cost**: 1-4 weeks depending on the variant
**Depends on**: W5, and an *observed* failure at W7

## Claim

A bounded, retrievable memory of past frames keeps a generated room consistent
across excursions longer than the context window -- which is the capability gap
between Genie 1 and Genie 3.

## Why now

Genie 3's headline claim is minute-long consistency with visual memory reaching
about a minute back. A causal transformer cannot get there by extending context:
one minute at 10 fps and 256 tokens/frame is 150k positions. So consistency has
to come from a mechanism, and DeepMind states it is *emergent* rather than backed
by an explicit 3D representation -- which makes it an open question rather than a
recipe.

**Do not build this until `revisit_consistency` has actually been measured and
found bad.** `eval/rollout.py` has the probe. Building a memory module against an
imagined failure is the most likely way to waste a month here.

## Design space

Three families, increasing cost:

1. **Compression.** Bounded KV cache; summarise evicted frames into a small set
   of learned register tokens. No new losses, framework-only. Try first.
2. **Retrieval.** Store past frame latents keyed by an estimated pose or an
   embedding; retrieve the top-k when the view returns. This is the closest fit
   for "walk away and come back" and needs no explicit geometry. TUM ships
   ground-truth poses, so the retrieval key can be supervised for free.
3. **Explicit spatial state.** A persistent voxel grid / 3D Gaussian scene the
   dynamics model reads and writes. `splatting/` is the starting point. This is a
   deliberate departure from Genie's implicit approach -- interesting precisely
   because it is a different bet.

All three fit `WorldMemory`: `reset()`, `write(latents, info)`, `read(query, k)
-> {tokens}` with a constant `token_budget`.

## Measurement

`revisit_consistency` on TUM sequences with a scripted out-and-back action
sequence, plus per-step `psnr_decay` over 64 steps.

| Metric | no memory | target |
|--------|-----------|--------|
| `revisit_psnr` | TBD | +3 dB |
| `psnr_decay` over 64 steps | TBD | lower |
| tokens added per step | 0 | <= 64 |
| VRAM at 64 steps | TBD | fits 7.7 GiB |

## Kill criterion

If compression (family 1) closes most of the `revisit_psnr` gap, stop -- do not
build retrieval or explicit geometry. If none of the three moves `revisit_psnr`
by 1 dB, the problem is probably drift (dynamics quality) rather than forgetting;
switch to idea 005.

## Result

*(pending)*
