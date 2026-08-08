# 015 -- Automatic reward design space

**Status**: idea
**Slot**: `bridge/rewards/`
**Depends on**: M1 (a simulator, to validate any reward against ground truth)

## The problem

Inside a world model there is no object pose, no contact sensor, no success flag
-- only generated pixels. Every reward here is an approximation. This note maps
the space; individual notes cover the ones worth building.

## The design space

| Family | Signal | Needs | Fails when |
|--------|--------|-------|-----------|
| **Goal image** | embedding distance to a target frame | a goal image, a frozen encoder | policy makes the *scene* look right without acting; background dominates the embedding |
| **Progress / value** | time-contrastive value learned from demos (VIP, LIV) | demonstrations | imagined frames are off-distribution from training frames |
| **VLM judge** | a VLM answers "did it pick up the object?" | a VLM, latency budget | VLMs are unreliable on blurry generated frames; too slow for per-step reward |
| **Dynamics prior** | world-model likelihood of the transition | nothing extra | rewards *predictable* behaviour, not *successful* behaviour -- standing still scores well |
| **Learned classifier** | success/failure classifier on real episodes | labels | most hackable; the policy finds its blind spot fast |
| **Preference / RLHF** | pairwise human comparisons | your time | does not scale to a single researcher |

## The order to try them

1. **`goal_image` in `delta` mode.** No training. It is the baseline everything
   else must beat, and the delta formulation resists the standing-still failure.
2. **Validate it against ground truth.** Before optimising anything: does the
   reward rise on successful *real* held-out episodes and stay flat on failures?
   If not, stop -- no RL machinery repairs an uncorrelated reward.
3. **Ensemble it with a second, structurally different reward.** Two rewards that
   fail differently make hacking visible via `disagreement`.
4. Only then consider a learned reward.

## The non-obvious variant worth trying

Use the **world model's own tokenizer** as the reward encoder, rather than a
ResNet or CLIP. It removes the real-vs-imagined distribution gap entirely, since
the reward then lives in the same space the dynamics model operates in. Cheap to
test -- `GoalImageReward` already accepts any encoder.

## Measurement (for any reward)

On **real** held-out episodes, before any RL:

| Metric | What it tells you |
|--------|------------------|
| correlation with final success | is the reward measuring the task at all |
| reward at t=0 vs t=T on successes | does it increase monotonically |
| reward on failures | does it stay flat -- if it also rises, it is measuring time |
| reward on a "do nothing" trajectory | the standing-still check |
| reward on real vs reconstructed frames | the distribution-gap check (idea 012) |

## Kill criterion (for the whole reward-based approach)

If no reward correlates with success on real episodes at above ~0.5, then
automatic reward from pixels is not viable at this scale -- and the honest
alternative is to use the world model for data augmentation rather than for RL.
