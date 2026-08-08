# Research Notes -- VLA

*Private. Gitignored.*

One file per idea. **Write the note before the code.** The kill criterion is the
point: deciding in advance what result would make you stop is what prevents a
dead idea consuming three weeks.

## Template

```markdown
# NNN -- Title

**Status**: idea | in progress | done | abandoned
**Slot**: modules / heads / backbones / rl / eval
**Cost**: rough GPU-hours + implementation days
**Depends on**: which milestones must be met first

## Claim
One sentence: what improves, by what mechanism.

## Why now
Which current failing metric motivates this. "It sounds good" means not ready.

## Design
Shapes, losses, where it hooks in. Remember: modules must be identity at init.

## Measurement
Exact metric, exact baseline, exact config, number of seeds. Filled BEFORE running.

## Kill criterion
Be specific. "If success rate does not improve by 5 points over 3 seeds, drop it."

## Result
Filled after. Copy the summary into docs/IMPROVEMENTS.md.
```

## Index

| # | Idea | Slot | Status |
|---|------|------|--------|
| 001 | Finish the JAX->PyTorch Octo port | backbones | idea |
| 002 | pi0 vs OpenVLA, frozen | backbones | idea |
| 003 | Simulation stack (SimplerEnv / LIBERO) | eval | **blocking** |
| 004 | Frozen backbone vs from-scratch | -- | idea |
| 010 | World-model-conditioned VLA | modules | idea |
| 013 | Diffusion action head | heads | idea |
| 014 | Flow-matching action head | heads | idea |
| 020 | RL post-training | rl | idea |
