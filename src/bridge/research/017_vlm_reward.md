# 017 -- VLM-as-judge reward

**Status**: idea (low priority)
**Slot**: `bridge/rewards/` (register as `vlm_judge`)

## Claim

A vision-language model can answer "has the robot picked up the red object?"
directly from a frame, giving a semantic reward with no reward-model training.

## Why it is low priority

Three problems, and the third is the one that decides it.

1. **Latency.** A VLM call per step per environment is 100-1000x the cost of an
   embedding distance. With 32 parallel imagined episodes at 16 steps, that is 512
   VLM calls per iteration. Usable only as a sparse terminal reward, not per step.
2. **VRAM.** A local VLM competes with the world model and the policy for 7.7 GiB.
   An API is possible but adds cost and network latency to the inner loop.
3. **Reliability on generated frames.** VLMs are trained on natural images.
   Asked about a blurry VQ-VAE reconstruction of a robot arm, their failure mode
   is confident and arbitrary -- which is the worst possible property for
   something an RL policy will optimise against.

Problem 3 is the real blocker, and it is measurable before building anything.

## If pursued

Use it as a **terminal** reward on the final imagined frame only, combined in an
ensemble with a dense shaped reward. Never per step.

Structure the prompt for a binary answer plus a confidence, and map confidence to
the `uncertainty` the contract requires.

## Measurement, and the cheap pre-check

Before writing any code: take 50 real frames and 50 reconstructions from the same
episodes, hand-label success/failure, and ask the VLM. If agreement on
reconstructions is much worse than on real frames, problem 3 is confirmed and the
idea is closed for a day's work.

## Kill criterion

VLM agreement with hand labels below ~0.8 on reconstructions.
