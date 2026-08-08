# Improvements Log -- World Model Track

*Private. Gitignored. One entry per change that was measured.*

## How to use this file

Append an entry whenever you change something and measure the effect. **Negative
results get entries too.** Template:

```markdown
### YYYY-MM-DD -- <short title>

**Change**: what was modified, which stage, which files/config.
**Hypothesis**: what you expected and why.
**Setup**: config, data source, epochs, seed, parent checkpoints.
**Result**:

| Metric | Before | After |
|--------|--------|-------|
| ...    | ...    | ...   |

**Verdict**: kept / reverted / needs more data.
**Why**: mechanism, if understood.
**Next**: what this implies.
```

Rules:

- **Name the parent checkpoints.** A stage-C number is meaningless without which
  stage-A tokenizer produced its tokens. `resolve_lineage()` prints them.
- **Report `codebook_use` with every stage-A PSNR.** PSNR alone hides collapse.
- **Report `delta_psnr` with every stage-C loss.** Loss alone hides a dead action
  channel.
- **Never compare across resolutions or frame rates** without saying so. PSNR at
  128 px and 256 px are different numbers.

---

## Entries

### 2026-08-05 -- Restructured into four staged components with contracts

**Change**: replaced the monolithic `GenieWorldModel` with four ABCs
(`VideoTokenizer`, `LatentActionModel`, `Dynamics`, `Decoder`), each behind a
registry with its own training stage and checkpoint. Added the latent action
model, which did not exist before -- the previous code had a tokenizer, a
dynamics transformer and a diffusion decoder but no way to discover actions, so
it was a video predictor rather than a world model.

**Hypothesis**: not a performance change. The claims are that a component can be
swapped by one config line and that stage checkpoints chain correctly.

**Setup**: RTX 4070 Laptop 7.7 GiB. Stage A on 600 LSUN+TUM stills, stages B/C on
TUM `fr1_desk`/`fr2_desk`. 1 epoch each, tiny model overrides, seed 0. Smoke run
only -- these are *not* training results.

**Result**: all four stages ran and chained.

| Stage | Metric | Value |
|-------|--------|-------|
| A | `val/psnr` | 11.68 |
| A | `val/codebook_use` | 0.300 |
| A | `val/perplexity` | 127.0 |
| B | `val/action_perplexity` | 1.75 |
| B | `val/actions_used` | 2 / 8 |
| C | `val/token_acc` | 0.0495 |
| C | `val/copy_baseline_acc` | 0.195 |
| C | `val/delta_psnr` | -0.0002 |

Peak VRAM: A 1.47 GiB, B 0.13 GiB, C 0.72 GiB.

**Verdict**: kept. The infrastructure works.

**Why the numbers are bad, and why that is expected**: one epoch on a few hundred
samples. Two of the diagnostics fired exactly as designed and are worth noting:

- Stage B used **2 of 8** action codes. The bottleneck collapsed. On a real run
  this is the first thing to fix -- likely by raising `data.frame_skip` so there
  is real motion between frames.
- Stage C's `token_acc` (0.05) is **below** `copy_baseline_acc` (0.19), i.e. the
  model is worse than copying the previous frame. And `delta_psnr` ≈ 0 confirms
  the action channel carries nothing yet, which follows directly from stage B.

Both would have been invisible from the loss curves alone. That is the argument
for these metrics existing.

**Next**: W1 -- a real stage-A run to `psnr > 22` and `codebook_use > 0.5` before
anything else. Stage C cannot be better than the tokens it is given.

---

<!-- new entries above this line -->
