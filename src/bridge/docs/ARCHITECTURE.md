# Architecture -- Bridge

*Private. Gitignored.*

```
src/bridge/
├── docs/                    # PRD / ADR / ARCHITECTURE / IMPROVEMENTS  (gitignored)
├── research/                # idea specs                               (gitignored)
├── action_space.py          # ActionTranslator: VLA action <-> world-model action
├── envs/
│   ├── base.py              # BaseEnv, StepResult, is_imagined
│   ├── world_model_env.py   # rollouts inside GenieWorldModel
│   └── sim_env.py           # SimplerEnv / LIBERO / MuJoCo wrappers   [not built]
├── rewards/
│   ├── base.py              # RewardModel ABC -- must report `uncertainty`
│   ├── goal_image.py        # embedding distance to a goal frame
│   └── ensemble.py          # weighted combination + disagreement penalty
└── training/
    └── stage_rl_in_dream.py                                           [not built]
```

## The loop

```
   ┌──────────────────────────────────────────────────────────────┐
   │  reset(): real context frames from a dataset                 │
   │           (a world model rolled from noise is useless)       │
   └───────────────────────────┬──────────────────────────────────┘
                               v
              Observation ──> VLAPolicy.sample() ──> action, logp
                               |
                               v
                    ActionSpec.denormalize  (physical units)
                               |
                               v
                    ActionTranslator  ──> world-model action
                               |
                               v
                 Dynamics.predict_next() ──> next latents
                               |
                    ┌──────────┴──────────┐
                    v                     v
          tokenizer.decode        RewardModel(obs) ──> reward, uncertainty
           (imagined frame)                |
                    └──────────┬───────────┘
                               v
                          Rollout(T, N)
                               |
                               v
                    RLAlgorithm.update()  + KL to BC reference
                               |
                    every N iters: validate in a REAL simulator
```

## Contracts

```python
class ActionTranslator(nn.Module, ABC):
    target_kind: str                          # "latent" | "robot"
    def forward(robot_actions) -> Tensor

class BaseEnv(ABC):
    is_imagined: bool                         # log this next to every metric
    def reset(batch_size, seed) -> Observation
    def step(action) -> StepResult            # batched, tensor-native

class RewardModel(nn.Module, ABC):
    instruction: str | None
    def reset(context_frames) -> None
    def forward(obs, latents, step) -> (reward, info)   # info["uncertainty"] required

class RLAlgorithm(ABC):
    needs_critic: bool
    def update(rollout) -> dict               # must return kl_to_ref
    def kl_penalty(obs, actions) -> Tensor
```

## Registries

| Registry | Names |
|----------|-------|
| `ACTION_TRANSLATORS` | `identity`, `learned_projector` |
| `REWARDS` | `goal_image`, `ensemble` |
| `ENVS` | (world-model env is constructed directly) |
| `RL_ALGORITHMS` | none yet -- `ppo`, `grpo`, `awr` planned |

## Order of work

1. Wire up a **real simulator** first (`envs/sim_env.py`). Without it there is no
   way to tell whether anything here works.
2. Validate `goal_image` reward on **real** held-out episodes: does it rise on
   successful ones and not on failures? If not, no amount of RL will help.
3. Train stage C with `action_kind=robot` so no action translation is needed.
4. Only then build `stage_rl_in_dream.py`.

Doing these out of order produces a system that runs and cannot be evaluated.
