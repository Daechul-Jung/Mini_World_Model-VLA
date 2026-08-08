# 002 -- ST-transformer video tokenizer

**Status**: idea
**Slot**: `tokenizer/` (register as `st_vqvae`)
**Cost**: ~2 days to implement, stage-A rerun (~4 GPU-hours)
**Depends on**: W1 (a healthy `conv_vqvae` baseline to compare against)

## Claim

Tokens that see neighbouring frames are more predictable than tokens computed
per-frame, so stage C's job gets easier without stage C changing.

## Why now

`conv_vqvae` tokenises each frame independently. Two adjacent frames of a
slow-panning camera can therefore land on unrelated codes for the same physical
surface, and the dynamics model has to learn that they are the same thing. Genie
uses ST blocks for exactly this. The observable symptom is a large gap between
`token_acc` and `copy_baseline_acc` in the *wrong* direction -- tokens that flip
more than the pixels do.

## Design

Replace the conv encoder/decoder with interleaved spatiotemporal blocks:
spatial attention over the `h*w` tokens within a frame, temporal attention over
`T` frames at a fixed spatial position. `STBlock` in
`latent_action/vq_lam.py` is already written and can be lifted directly.

Encoder input `(B, T, 3, H, W)` -> patch embed -> L x STBlock -> quantize.
Decoder mirrors it. Keep the `VideoTokenizer` contract unchanged, so nothing
downstream moves.

One decision to make: causal or bidirectional temporal attention in the
tokenizer. Genie's tokenizer is causal, which matters if tokens are ever computed
online. Bidirectional would compress better. Start causal, matching the paper.

## Measurement

Same stage-A config, same data, 3 seeds. Then stage C on both tokenizers.

| Metric | `conv_vqvae` baseline | target |
|--------|----------------------|--------|
| `val/psnr` | TBD | >= baseline |
| `val/codebook_use` | TBD | >= baseline |
| stage C `token_acc - copy_baseline_acc` | TBD | **+0.05 absolute** |

The third row is the one that matters. Better reconstruction alone does not
justify the cost.

## Kill criterion

If stage-C `token_acc - copy_baseline_acc` does not improve by at least 0.03
across 3 seeds, the temporal context is not buying predictability here -- keep
`conv_vqvae` and spend the time on MaskGIT (004) instead.

## Result

*(pending)*
