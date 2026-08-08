"""The world model as a Gymnasium-style environment.

    policy action --> [translator] --> world-model action
                                              |
    world model.predict_next --> imagined latents --> render --> observation
                                              |
                                     [reward model] --> reward

This is the piece that makes "post-train the VLA with RL inside generated
environments" concrete. Everything it needs is already a contract: the world
model is a `GenieWorldModel`, the reward is a `RewardModel`, the action mapping is
an `ActionTranslator`. Swapping any of the three is a config change.

**Three things this environment cannot do, and pretending otherwise wastes GPU
time:**

1. *It has no ground truth about success.* There is no simulator state, no object
   pose, no contact. Reward has to be inferred from imagined pixels, which is why
   `rewards/` is its own package and why every reward there is approximate.

2. *It is exploitable.* The dynamics model was trained on the data distribution;
   a policy optimising against it will find actions that produce high reward and
   are physically impossible. Mitigations, all of them partial: KL to the BC
   policy, short rollouts (`max_steps` 10-30, not 200), an ensemble reward, and
   periodic re-validation in a real simulator. `uncertainty` is reported in
   `info` for exactly this reason.

3. *Its frames are tokenizer-quality.* At `render="tokenizer"` the policy sees a
   VQ-VAE reconstruction, which is blurrier than a real camera image. A VLA
   trained on sharp real frames may behave differently on them. Measure that gap
   before trusting any in-dream number -- `eval/dream_vs_real.py`.

Used honestly, this is a fast way to generate cheap on-policy experience for
shaping, followed by validation elsewhere. It is not a replacement for a
simulator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from common.types import Observation

from .base import BaseEnv, StepResult


class WorldModelEnv(BaseEnv):
    """Batched rollouts inside a `GenieWorldModel`.

    Args:
        world_model: a `GenieWorldModel` with tokenizer + dynamics loaded.
        reward_model: a `RewardModel` scoring imagined observations.
        translator: maps policy actions into the dynamics model's action space.
        context_frames: real frames used to prompt each episode. Sampled from a
            dataset by `reset()`, because a world model rolled out from noise
            produces nothing useful.
        max_steps: episode length. Keep short -- compounding dynamics error makes
            late steps unreliable, and the reward signal degrades with them.
        render: `"tokenizer"` (fast) or `"decoder"` (sharp, ~25x cost).
    """

    def __init__(
        self,
        world_model: Any,
        reward_model: Any,
        translator: Any,
        context_provider: Any,
        max_steps: int = 16,
        render: str = "tokenizer",
        device: torch.device | str = "cuda",
        temperature: float = 1.0,
    ):
        self.world_model = world_model
        self.reward_model = reward_model
        self.translator = translator
        self.context_provider = context_provider
        self.max_steps = max_steps
        self.render_mode = render
        self.device = torch.device(device)
        self.temperature = temperature

        self._history: Optional[torch.Tensor] = None      # token history
        self._frames: List[torch.Tensor] = []
        self._step = 0
        self._batch = 0

    # -------------------------------------------------------------------- gym

    def reset(self, batch_size: int = 1, seed: Optional[int] = None) -> Observation:
        """Prompt the world model with real context frames."""
        if seed is not None:
            torch.manual_seed(seed)
        self._batch = batch_size
        self._step = 0

        context = self.context_provider(batch_size).to(self.device)   # (B, T_ctx, 3, H, W)
        with torch.no_grad():
            enc = self.world_model.tokenizer.encode(context)
        self._history = (
            enc["indices"] if self.world_model.latent_spec.discrete else enc["latents"]
        )
        self._frames = [context[:, -1]]
        self.world_model.dynamics.reset_cache()
        self.reward_model.reset(context)

        return self._observation(context[:, -1], context)

    def step(self, action: torch.Tensor) -> StepResult:
        """Advance one imagined frame.

        `action`: (B, action_dim) in the *policy's* action space; the translator
        converts it. Physical de-normalisation must already have happened --
        `VLAPolicy.act` does it.
        """
        wm_action = self.translator(action.to(self.device))

        with torch.no_grad():
            nxt = self.world_model.dynamics.predict_next(
                self._history, action=wm_action, temperature=self.temperature
            )
            self._history = torch.cat([self._history, nxt], dim=1)

            if self.world_model.latent_spec.discrete:
                latents = self.world_model.tokenizer.indices_to_latents(nxt)
            else:
                latents = nxt

            if self.render_mode == "decoder" and self.world_model.decoder is not None:
                frame = self.world_model.decoder.render(latents, steps=25)[:, 0]
            else:
                frame = self.world_model.tokenizer.decode(latents)[:, 0]

        self._frames.append(frame)
        self._step += 1

        obs = self._observation(frame, torch.stack(self._frames[-2:], dim=1))
        reward, reward_info = self.reward_model(obs, latents=latents, step=self._step)
        done = torch.full((self._batch,), self._step >= self.max_steps, device=self.device)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": self._step,
                # How far outside the training distribution this rollout has
                # drifted. Rising uncertainty means the reward is less trustworthy.
                "uncertainty": reward_info.get("uncertainty", 0.0),
                **reward_info,
            },
        )

    def _observation(self, frame: torch.Tensor, history: torch.Tensor) -> Observation:
        return Observation(
            image=history,
            instruction=[self.reward_model.instruction] * self._batch
            if getattr(self.reward_model, "instruction", None)
            else None,
        )

    def rollout_frames(self) -> torch.Tensor:
        """(B, T, 3, H, W) of everything imagined this episode -- for videos."""
        return torch.stack(self._frames, dim=1)

    def close(self) -> None:
        self._history = None
        self._frames = []
