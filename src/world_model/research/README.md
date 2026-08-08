# Research Notes -- World Model

*Private. Gitignored.*

One file per idea. **Write the note before writing the code.** The point is the
kill criterion: deciding in advance what result would make you abandon the idea
is what stops a dead idea consuming three weeks.

## Template

```markdown
# NNN -- Title

**Status**: idea | in progress | done | abandoned
**Slot**: which contract this plugs into (tokenizer / latent_action / dynamics /
          decoder / memory / physics)
**Cost**: rough GPU-hours and implementation days
**Depends on**: which milestones must be met first

## Claim
One sentence. What will be better, and by what mechanism.

## Why now
What in the current results motivates this. If the answer is "it sounds good",
the idea is not ready -- go find the failing metric first.

## Design
The actual change. Shapes, losses, where it hooks in.

## Measurement
Exact metric, exact baseline, exact config. Filled in BEFORE running.

## Kill criterion
The result that means stop. Be specific: "if delta_psnr does not improve by 0.3
over three seeds, drop it."

## Result
Filled in after. Copy the summary to docs/IMPROVEMENTS.md.
```

## Index

| # | Idea | Slot | Status |
|---|------|------|--------|
| 002 | ST-transformer tokenizer | tokenizer | idea |
| 003 | FSQ / LFQ quantizers | quantizer | idea |
| 004 | MaskGIT dynamics | dynamics | idea |
| 005 | Diffusion forcing | dynamics | idea |
| 006 | Continuous latent actions | latent_action | idea |
| 007 | Long-horizon memory | memory | idea |
| 008 | Physics auxiliary heads | physics | idea |
| 009 | JEPA dynamics | dynamics | idea |
| 011 | Unified action conditioning | dynamics | idea |
| 012 | Reward on imagined frames | (eval) | idea |
