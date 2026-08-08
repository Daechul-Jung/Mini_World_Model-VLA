# 014 -- Flow-matching action head

**Status**: idea
**Slot**: `heads/` (register as `flow_matching`)
**Cost**: ~3 days
**Depends on**: idea 013 (same problem, compare directly)

## Claim

Flow matching gives diffusion's multi-modal expressiveness with far fewer
inference steps, which matters when the head runs inside an RL loop.

## Why now

pi0 uses flow matching for exactly this reason: continuous chunked actions at
control rates that diffusion sampling would not meet. If pi0 is chosen as the
backbone (idea 002), matching its action parameterisation also means the adapter
head is not fighting the backbone's native representation.

## Design

Learn a velocity field `v(a_t, t, features)` transporting noise to the action
distribution. Train by regressing the straight-line velocity between a noise
sample and a data sample. Sample by integrating the ODE -- typically 1-10 Euler
steps versus diffusion's 10-50.

Same `ActionHead` contract. Same open question as idea 013: log-probability for
RL is not free. Note that flow matching's ODE is deterministic given the initial
noise, which makes an exact likelihood *more* tractable than diffusion's -- worth
checking, because it would make this the head that serves both BC and RL.

## Measurement

| Head | offline `action_l1` | **sim success** | ms/action | RL-capable |
|------|--------------------|-----------------|-----------|------------|
| `continuous_mse` | | | | no |
| `gaussian` | | | | yes |
| `discrete_bins` | | | | yes |
| `diffusion` (10 steps) | | | | unclear |
| `flow_matching` (1 step) | | | | ? |
| `flow_matching` (10 steps) | | | | ? |

The 1-step row is the interesting one: if it is competitive, this is strictly
better than diffusion for this project.

## Kill criterion

If 1-step flow matching is no better than `gaussian` on sim success, the extra
machinery is not earning its place at this data scale.

## Result

*(pending)*
