# 003 -- FSQ / LFQ instead of VQ

**Status**: idea
**Slot**: `tokenizer/quantizers/` (register as `fsq`, `lfq`)
**Cost**: ~half a day each; they are small
**Depends on**: an observed codebook-collapse failure

## Claim

Finite Scalar Quantization (Mentzer et al., 2023) and Lookup-Free Quantization
(MAGVIT-v2) remove the codebook-collapse failure mode entirely, because they have
no learned codebook to collapse.

## Why now

The very first smoke run showed `codebook_use = 0.30` after one epoch. If a real
stage-A run plateaus below 0.5, this is the fix -- not more epochs, and not a
bigger encoder.

FSQ: project to `d` channels (d ~ 5), round each to one of `L` levels. The
"codebook" is the implicit product grid, always fully reachable. No commitment
loss, no EMA updates, no dead codes.

LFQ: binarise each of `log2(K)` dimensions. Same idea, binary levels, plus an
entropy bonus.

## Why now (the honest version)

Do **not** implement this pre-emptively. Implement it when a stage-A run has
actually plateaued with low codebook usage. Building both quantizers before
seeing the failure is optimising against an imagined problem, and VQ may well be
fine at this scale.

## Design

Both satisfy the existing quantizer interface -- `forward(z) -> (z_q, aux_loss,
indices)` and `decode_indices(indices) -> z_q` -- so `ConvVQVAETokenizer` needs
no change. `tokenizer.quantizer.name: fsq` in the config is the whole switch.

FSQ note: `num_embeddings` becomes `prod(levels)` rather than a free parameter,
so the tokenizer's `LatentSpec.vocab_size` must be derived from `levels`.

## Measurement

Stage A, same config, 3 seeds.

| Metric | `vq` | `fsq` | `lfq` |
|--------|------|-------|-------|
| `val/psnr` | | | |
| `val/codebook_use` | | | |
| `val/perplexity` | | | |
| epochs to `psnr > 22` | | | |

## Kill criterion

If `vq` reaches `codebook_use > 0.5` on a real run, this idea is unnecessary --
close it and note that.

## Result

*(pending)*
