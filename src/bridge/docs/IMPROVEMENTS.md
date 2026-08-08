# Improvements Log -- Bridge

*Private. Gitignored.*

Same template as the other two logs. Two extra rules specific to this track:

- **Label every success rate with its environment.** `is_imagined=True` numbers
  and simulator numbers are different quantities (ADR-B04).
- **Report `uncertainty` and `disagreement` alongside return.** A return that
  rose while uncertainty rose is most likely reward hacking, not learning.

---

## Entries

### 2026-08-05 -- Bridge package created

**Change**: new `src/bridge/` holding `ActionTranslator`, `BaseEnv`,
`WorldModelEnv`, `RewardModel` + `goal_image` + `ensemble`, and the
`rl_in_dream` config. Nothing trained.

**Why it exists**: the plan to post-train a VLA inside world-model-generated
environments has three unsolved sub-problems (action-space mismatch, reward
without ground truth, two exploitable learned models). This package names each
one, gives it a contract, and records the recommended first move for each rather
than leaving them to be discovered mid-implementation.

**Verdict**: kept, unmeasured.

**Next**: B1 -- but only after a real simulator exists (`envs/sim_env.py`) and
`goal_image` has been validated on real held-out episodes. Building the RL loop
before those two is building something that cannot be evaluated.

---

<!-- new entries above this line -->
