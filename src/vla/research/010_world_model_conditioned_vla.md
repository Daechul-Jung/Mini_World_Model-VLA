# 010 -- World-model-conditioned VLA

**Status**: idea
**Slot**: `modules/` (`wm_conditioning`, implemented but untested)
**Cost**: ~1 week
**Depends on**: M1 (simulator), W5 (usable world-model rollouts)

## Claim

A policy with access to what the world model predicts will happen acts on
consequences rather than on appearance alone, and generalises better from few
demonstrations.

## Why now

This is the "world model helps the VLA" half of the project thesis. With 100
episodes, a BC policy has very little signal about *why* an action is right. A
short lookahead adds signal at no data cost.

## Three variants, and the order to run them

Run 3 first. It is the honest baseline and it may already capture the gain.

**3. Representation transfer** (cheapest). Skip rollouts. Use the world model's
*tokenizer* as a frozen visual encoder for the policy. Tests whether world-model
training produces better visual features at all, with no imagination cost. If
this alone matches variants 1 and 2, the imagination machinery is not what helps.

**1. Passive lookahead** (implemented). Roll the world model forward `k` steps
under the policy's current action; cross-attend policy features to the imagined
latents. Cost: `k` extra world-model forwards per training step. Keep
`render="tokenizer"` -- diffusion rendering would make this ~100x.

**2. Counterfactual lookahead**. Roll forward under several candidate actions and
attend to all of them: a learned, differentiable one-step planner. Strongest
claim, highest cost, and it needs care about gradients flowing into the world
model (they should not -- keep it frozen).

## Design

`WorldModelConditioning` reads `context["wm_latents"]` rather than holding the
world model as a submodule, so it never trains it and works with any checkpoint.
`src/bridge/` populates that key. Zero-initialised output projection plus a gate,
per the identity-at-init rule.

## Measurement

Against the same policy without the module, 3 seeds.

| Setup | offline `action_l1` | `gripper_transition_acc` | **sim success rate** | gate value |
|-------|--------------------|-------------------------|---------------------|------------|
| baseline (no module) | | | | -- |
| + tokenizer-as-encoder (v3) | | | | -- |
| + passive lookahead k=4 (v1) | | | | |
| + counterfactual (v2) | | | | |

**The gate value is the cheapest diagnostic in this project.** If it stays near
zero, the module is inert and the idea is dead -- no further analysis needed.

## Kill criterion

If sim success rate does not improve by 5 points over 3 seeds for any variant, or
if variant 3 matches variants 1-2, drop the imagination path and report variant 3
as the finding.

## Result

*(pending)*
