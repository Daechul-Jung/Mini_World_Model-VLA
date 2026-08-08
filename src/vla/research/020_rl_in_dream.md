# 020 -- RL post-training inside the world model

**Status**: idea -- **the project's headline claim**
**Slot**: `rl/` + `bridge/training/`
**Cost**: 2-4 weeks
**Depends on**: M1, M3, W8, B2, and idea 012 measured

## Claim

RL post-training in world-model-generated environments improves a BC-pretrained
VLA more than additional BC does, measured in a real simulator.

## Why now

BC on 100 episodes gives a policy that imitates and cannot recover. RL needs
environment interaction, and a real simulator is slow. A world model gives cheap
on-policy experience. That is the argument; it has three known holes, all in
`bridge/docs/PRD.md`.

## Prerequisites, and why each is non-negotiable

| # | Prerequisite | Why skipping it invalidates the result |
|---|-------------|---------------------------------------|
| 1 | Simulator (idea 003) | there is no way to measure the claim |
| 2 | Reward validated on **real** episodes (B2) | optimising an uncorrelated reward |
| 3 | `action_kind=robot` world model (W8) | latent-code translation is lossy |
| 4 | Real-vs-imagined gap measured (idea 012) | in-dream results may be pure gap |
| 5 | Stochastic head (`gaussian`) | no policy gradient exists otherwise |

Item 5 is enforced in code: `RLAlgorithm.__init__` raises on a deterministic head,
because the alternative failure -- a run that trains and optimises nothing -- is
the most expensive kind.

## Design

PPO first: on-policy, well understood, and it does not accumulate world-model
error in a replay buffer (bridge ADR-B05). GRPO is worth trying after, since it
drops the critic -- and a critic trained on imagined rollouts is a third learned
model that can be wrong.

Non-negotiable settings, each with a reason:

* `kl_coef > 0` against a frozen BC copy. Two exploitable learned models; the BC
  policy is the only anchor to behaviour known to be physically real.
* `max_steps: 16`, not 200. Reward quality decays with `psnr_decay`.
* Reward ensemble with `disagreement_penalty > 0`.
* `validate_every: 20` in the simulator. **In-dream return is not a result.**

## The baseline that must be beaten first

Cheaper and less interesting, and it must be run: use the world model to
**generate more BC data** rather than for RL. If augmented BC matches RL-in-dream,
the RL machinery is not what is helping, and that is the finding.

| Setup | sim success rate |
|-------|-----------------|
| BC only | |
| BC + world-model-augmented BC data | **the baseline to beat** |
| BC + RL in simulator (upper bound, slow) | |
| BC + RL in dream | |

## Measurement

Report in-dream return and simulator success side by side, every time. The *gap*
between them is itself a finding about how far a small world model can be trusted.

## Kill criterion

If RL-in-dream does not beat world-model-augmented BC in the simulator across 3
seeds, report that. It is a clean negative result about world models at this
scale and is worth writing up as such.

## Result

*(pending)*
