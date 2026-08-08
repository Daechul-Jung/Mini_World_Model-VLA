# Improvements Log -- VLA Track

*Private. Gitignored. One entry per change that was measured.*

## How to use this file

Append an entry whenever you change something and measure the effect. **Negative
results get entries too** -- an idea that did not work, recorded with its numbers,
is worth more than an idea silently abandoned, because it stops you retrying it
in three months.

Template:

```markdown
### YYYY-MM-DD -- <short title>

**Change**: what was modified, and which files/config.
**Hypothesis**: what you expected and why.
**Setup**: config, dataset, epochs, seed(s), GPU.
**Result**:

| Metric | Before | After |
|--------|--------|-------|
| ...    | ...    | ...   |

**Verdict**: kept / reverted / needs more data.
**Why**: the mechanism, if you understand it. "Unclear" is an acceptable answer.
**Next**: what this implies for the next experiment.
```

Rules that keep this file honest:

- **One change per entry.** Two changes measured together teach you nothing about
  either.
- **Report the seed.** With 100 episodes, seed-to-seed variance is large enough to
  swamp real effects. Prefer 3 seeds before believing a difference.
- **Report the baseline you beat.** "L1 improved to 0.31" is not a result without
  what it was before, under the same config.
- **Say which environment produced a success rate.** Simulator and world-model
  numbers are not comparable; label every one.

---

## Entries

### 2026-08-05 -- Repository restructured around swappable contracts

**Change**: replaced the flat `src/vla/` layout with
`core / backbones / modules / heads / data / training / rl / eval`, all behind
registries. Added `common/` (registry, config, checkpoint lineage, stage runner,
trainer). Ported the Octo recipe to a working PyTorch policy; the partial JAX
port is retained but unused.

**Hypothesis**: not a performance change. The claim is that adding a research
idea should touch one file plus one config line, and that staged checkpoints
should be independently loadable.

**Setup**: RTX 4070 Laptop 7.7 GiB, `data/openx` (100 episodes, 4-DoF).

**Result**: `octo_small` (12.2M at `depth=4`) trains end-to-end, 1 epoch in 7 s,
peak 0.32 GiB. Baseline to beat, 1 epoch only, seed 0:

| Metric | Value |
|--------|-------|
| `val/loss` | 0.3167 |
| `val/action_l1` | 0.5216 |
| `val/gripper_acc` | 0.4948 |

`gripper_acc` at 0.49 is chance. Expected after one epoch; it is the number to
watch first on a real run.

**Verdict**: kept.

**Next**: M1 -- wire up a simulator. Every number above is an offline proxy and
none of them predict task success.

---

<!-- new entries above this line -->
