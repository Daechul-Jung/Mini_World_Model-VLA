# 016 -- Progress reward (VIP / LIV style)

**Status**: idea
**Slot**: `bridge/rewards/` (register as `progress`)
**Depends on**: `goal_image` baseline measured (idea 015 step 2)

## Claim

A time-contrastive value function learned from demonstrations gives a denser and
less hackable progress signal than raw goal-image similarity.

## Why now

`goal_image` has one clear failure: it rewards *resembling* the goal, so a policy
that moves the camera to make the scene look right scores well without acting.
VIP (Ma et al., 2023) instead learns `V(s)` such that value increases along
demonstration trajectories, using only the demonstrations -- no reward labels, no
action labels.

## Design

Train on `data/openx` episodes. For frames `i < j` in the same episode,
`V(s_j) > V(s_i)`; add a goal-conditioning term. The reward at each step is
`V(s_{t+1}) - V(s_t)`.

Two properties that matter here: it is dense (a signal every step, not only at
success), and it is grounded in *trajectories* rather than in a single goal image,
so it is harder to satisfy by holding a pose.

## The risk, stated up front

VIP trained on real frames, evaluated on world-model reconstructions, is exactly
the distribution-gap problem in idea 012. Two options: measure the gap and cap the
rollout horizon accordingly, or train VIP on reconstructions in the first place so
both sides live in the same distribution. The second is cheap and probably right.

## Measurement

The idea-015 validation table, on real held-out episodes, versus `goal_image`.
Critically including the "do nothing" trajectory check.

## Kill criterion

If it does not beat `goal_image` on correlation-with-success on real episodes, do
not put it in the ensemble -- extra learned components mean extra things to hack.
