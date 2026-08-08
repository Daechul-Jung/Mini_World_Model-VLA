# 012 -- Does anything trained on real frames work on imagined ones?

**Status**: idea (**measure this early -- it gates the whole bridge**)
**Slot**: evaluation, not a component
**Cost**: ~1 day
**Depends on**: W5

## Claim

There is a distribution gap between real camera frames and world-model
reconstructions, and everything downstream -- the reward model, the VLA policy --
was trained on the real side.

## Why this is first, not last

The bridge plan is: a VLA acts on imagined frames, and a reward model scores
imagined frames. Both were built for real images. If either degrades sharply on
tokenizer output, then every in-dream number is measuring the gap rather than the
policy, and no amount of RL machinery fixes it.

This costs a day to measure and can invalidate weeks of work. Measure it before
building `stage_rl_in_dream.py`.

## Design

Take held-out **real** clips. Produce three versions of each frame:

1. the real frame
2. `tokenizer.decode(tokenizer.encode(frame))` -- reconstruction only, no dynamics
3. `world_model.imagine(...)` -- a genuinely predicted frame

Then run the same three things on each and compare:

* **Policy agreement**: `policy(real)` vs `policy(recon)` vs `policy(imagined)` --
  action L1 between them. This isolates whether the *policy* survives the gap.
* **Reward agreement**: same for `goal_image` reward.
* **A linear probe** trained on real frames, evaluated on each -- a
  representation-level measure independent of both models.

Comparing 2 against 3 separates two different gaps: tokenizer fidelity, and
dynamics error. They have different fixes.

## Measurement

| Input | policy action L1 vs real | reward vs real | probe accuracy |
|-------|-------------------------|----------------|----------------|
| real | 0 | 0 | baseline |
| reconstruction | | | |
| 1-step imagined | | | |
| 8-step imagined | | | |
| 16-step imagined | | | |

## Decision rule

* Gap small at reconstruction, grows with horizon -> dynamics error dominates.
  Cap `env.max_steps` at the horizon where agreement is still acceptable, and
  work on idea 005.
* Gap already large at reconstruction -> the tokenizer is the problem. Either
  improve stage A, or fine-tune the policy and reward on reconstructions so both
  sides live in the same distribution.
* Gap small everywhere -> proceed to the RL loop with confidence.

## Result

*(pending)*
