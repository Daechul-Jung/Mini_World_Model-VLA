# 013 -- Diffusion action head

**Status**: idea
**Slot**: `heads/` (register as `diffusion`)
**Cost**: ~3 days
**Depends on**: M2, and an observed multi-modality failure

## Claim

A diffusion head represents multi-modal action distributions that `continuous_mse`
structurally cannot, fixing the specific failure where a policy with good offline
error acts wrongly at decision points.

## Why now

MSE regression has one failure mode that matters here. When demonstrations contain
two valid actions for the same observation -- two ways to approach a cup -- the MSE
optimum is their average, which is often invalid. The classic symptom is a gripper
that reaches *between* two grasp points.

**The diagnostic**: good `action_l1`, poor `gripper_transition_acc`, and rollouts
that fail at grasp. If that pattern is not present, this idea is not needed -- do
not build it pre-emptively.

Octo-Base uses a diffusion head; Octo-Small uses regression. `discrete_bins` is
the cheaper fix for the same problem and should be tried first.

## Design

Standard conditional DDPM over the action chunk. Condition on policy features,
predict noise, ~10-50 denoising steps at inference. `ActionHead.loss()` becomes
the denoising MSE; `forward()` runs the sampling loop.

Two consequences to plan for:

* Inference cost goes from 1 forward to ~10-50. In an RL loop with imagined
  rollouts that compounds with the world-model cost.
* `sample()` needs a tractable log-probability for RL, and diffusion does not
  give one directly. Either keep `gaussian` for RL post-training, or use one of
  the diffusion-policy RL formulations. **Decide this before building**, or the
  head will be unusable at the stage it is most wanted.

## Measurement

3 seeds, same backbone.

| Head | offline `action_l1` | `gripper_transition_acc` | **sim success** | ms/action |
|------|--------------------|-------------------------|-----------------|-----------|
| `continuous_mse` | | | | |
| `discrete_bins` | | | | |
| `diffusion` (10 steps) | | | | |
| `diffusion` (50 steps) | | | | |

## Kill criterion

If `discrete_bins` closes the gap, stop -- it is cheaper, RL-compatible, and it
addresses the same failure.

## Result

*(pending)*
