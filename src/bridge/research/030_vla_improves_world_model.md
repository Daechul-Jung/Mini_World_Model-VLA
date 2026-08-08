# 030 -- Using the VLA to improve the world model

**Status**: idea (no concrete mechanism yet -- this note is for developing one)
**Slot**: unclear; probably `world_model/physics/` or a new loss in stage C

## The stated intuition

"Using a VLA for better performance of the world model is possible, but I do not
have a solid idea yet."

The intuition is reasonable. A VLA trained on robot data has learned what matters
about a manipulation scene -- where the gripper is, where the object is, what is
graspable. A world model trained on next-token prediction has learned what
*changes*, which is dominated by texture and lighting. Those are different, and
the first is closer to what a world model should care about.

## Four candidate mechanisms, weakest to strongest

**1. VLA features as auxiliary supervision.** Add a head on the dynamics model
predicting the frozen VLA's features for the next frame. Cheap, fits the existing
`PhysicsHead` contract exactly, no new infrastructure. *Weakness*: if the VLA's
features are not better than the tokenizer's for this purpose, it does nothing --
and that is a one-line ablation.

**2. VLA-weighted reconstruction loss.** Weight the tokenizer's and dynamics
model's loss by how much each spatial region affects the VLA's action. Gets
gradient from `d(action)/d(pixel)`, which is a saliency map over what matters for
control. Directly targets the "world models waste capacity on texture" problem.
*Weakness*: saliency maps are noisy, and this couples two models' training.

**3. VLA-guided data collection.** Use the VLA to generate action sequences that
visit states worth modelling, rather than training on whatever video exists.
Closes a loop -- the world model gets better where the policy actually goes.
*Weakness*: needs a simulator to execute the policy, so it is not available until
M1, and it risks the world model becoming good only where the policy already is.

**4. Joint training.** Both models, one objective. *Weakness*: two learned models,
no ground truth, no way to attribute failure. Explicitly out of scope in the
bridge PRD, and mechanism 4 should stay there until 1-3 have been tried.

## Recommended first move

**Mechanism 1**, because it needs no new infrastructure -- `PhysicsHead` already
takes dynamics features and returns an auxiliary loss, so this is one file. And
its ablation is clean: compare against predicting the *tokenizer's* features for
the next frame. If VLA features do not beat tokenizer features as a target, the
whole direction is answered cheaply and negatively.

## Measurement

| Aux target | `delta_psnr` | `psnr_decay` | rollout usefulness for RL |
|-----------|-------------|-------------|--------------------------|
| none | | | |
| tokenizer features of t+1 (control) | | | |
| VLA features of t+1 | | | |

## Kill criterion

If VLA features do not beat the tokenizer-feature control, close the idea and
record the negative -- it would say the VLA's representation carries nothing extra
about scene dynamics, which is worth knowing.
