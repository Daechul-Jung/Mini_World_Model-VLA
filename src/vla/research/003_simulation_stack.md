# 003 -- Simulation stack

**Status**: idea -- **this is the blocking item for the whole VLA track**
**Slot**: `eval/`
**Cost**: 3-7 days, mostly dependency wrangling
**Depends on**: nothing. Do this first.

## Claim

Without a simulator there is no success rate, and without a success rate every
number in this project is an offline proxy that is known to correlate weakly with
what we care about.

## Why now

Offline action error measures whether the policy reproduces demonstrator actions
on states the demonstrator visited. Manipulation policies fail on states the
demonstrator never entered -- after a slightly-off grasp, after a slip. Offline
error cannot see any of that.

Concretely: `gripper_acc` at 0.95 and `gripper_transition_acc` at 0.55 describe a
policy that will fail every episode, and both are compatible with a good-looking
`action_l1`.

## Why plain MuJoCo is the wrong target

Octo, OpenVLA and pi0 are trained on real-robot data (Open X-Embodiment, Bridge,
RT-1). A Gymnasium MuJoCo scene shares neither the visual distribution nor the
action semantics. A zero-shot number there is not a weak result -- it is not a
result.

## The two options that make sense

**SimplerEnv** -- built specifically to evaluate real-robot-trained VLAs in
simulation, with visual matching to the Bridge and Google-Robot setups, and
published correlation with real-robot performance. This is the correct target for
zero-shot pretrained-VLA evaluation.
Cost: SAPIEN dependency, which is heavier than Gymnasium.

**LIBERO** -- a manipulation benchmark with official OpenVLA fine-tuned
checkpoints, so there is a published reference number to reproduce. Better suited
to *fine-tuned* evaluation than zero-shot.
Cost: robosuite/MuJoCo, lighter than SAPIEN.

## Recommendation

Start with **LIBERO**. Lighter dependencies, a reproducible reference number, and
an environment where a policy trained on this project's data has a plausible
chance. Add SimplerEnv when the OpenVLA/pi0 zero-shot baseline is needed.

Both go behind `bridge/envs/base.py::BaseEnv`, so the RL loop and the world-model
env stay interchangeable.

## Design

```python
@ENVS.register("libero")
class LiberoEnv(BaseEnv):
    is_imagined = False
    def reset(self, batch_size=1, seed=None) -> Observation: ...
    def step(self, action) -> StepResult: ...
```

Two things that must be right, and are the usual sources of a silently broken
evaluation:

1. **Action space.** The sim expects specific units and a specific gripper
   convention (is +1 open or closed?). `ActionSpec.denormalize` must be applied,
   and the gripper convention checked by hand on one episode.
2. **Observation format.** Camera name, resolution, and whether images arrive
   already normalised. A policy fed [0, 255] when it expects [-1, 1] produces
   confident nonsense.

Write a scripted expert that solves at least one task first. If the scripted
expert cannot solve it, the environment is misconfigured, and no policy result
from it means anything.

## Measurement

| Policy | LIBERO success rate |
|--------|--------------------|
| scripted expert | should be ~1.0 -- **this validates the harness** |
| random actions | ~0 |
| repeat-previous-action | ~0 |
| `octo_small` BC | ? |
| pretrained backbone, zero-shot | ? |

## Kill criterion

None -- this is infrastructure, not a hypothesis. But if the scripted expert
cannot solve a task, stop and fix the environment before running any policy.

## Result

*(pending)*
