"""Layer 3 -- the bridge: action translation, rewards, world-model env.

These are the pieces that make "RL inside a generated environment" run at all.
The contract properties tested here are the ones whose violation produces a run
that trains happily and means nothing.
"""

from __future__ import annotations

import pytest
import torch

import world_model as wm
from bridge.action_space import ACTION_TRANSLATORS, build_translator
from bridge.envs import WorldModelEnv
from bridge.rewards import REWARDS


class TestActionTranslation:
    def test_identity_passes_robot_actions_through(self):
        t = ACTION_TRANSLATORS.build({"name": "identity", "action_dim": 4})
        actions = torch.randn(3, 4)
        assert torch.equal(t(actions), actions)
        assert t.target_kind == "robot"

    def test_learned_projector_emits_valid_codes(self):
        t = ACTION_TRANSLATORS.build(
            {"name": "learned_projector", "action_dim": 4, "num_actions": 8}
        )
        codes = t(torch.randn(5, 4))
        assert codes.shape == (5,)
        assert codes.max() < 8 and codes.min() >= 0
        assert t.target_kind == "latent"

    def test_fit_reports_lift_over_chance(self):
        """`lift_over_chance` is the number that decides whether translation is
        viable at all -- near zero means the two action spaces do not align."""
        t = ACTION_TRANSLATORS.build(
            {"name": "learned_projector", "action_dim": 2, "num_actions": 4}
        )
        # A perfectly learnable mapping: the code is the sign quadrant.
        actions = torch.randn(512, 2)
        codes = ((actions[:, 0] > 0).long() * 2 + (actions[:, 1] > 0).long())
        stats = t.fit(actions, codes, epochs=300)
        assert stats["code_accuracy"] > 0.9
        assert stats["lift_over_chance"] > 0.6

    def test_soft_forward_is_differentiable(self):
        t = ACTION_TRANSLATORS.build(
            {"name": "learned_projector", "action_dim": 4, "num_actions": 8}
        )
        onehot = t.soft_forward(torch.randn(3, 4, requires_grad=True))
        assert onehot.shape == (3, 8)
        onehot.sum().backward()

    def test_kind_mismatch_with_world_model_is_rejected(self, tiny_wm_config):
        """A latent-conditioned dynamics model must not be handed robot actions."""
        model = wm.GenieWorldModel.from_config(tiny_wm_config)   # action_kind = latent
        with pytest.raises(ValueError, match="dynamics model expects 'latent'"):
            build_translator({"name": "identity", "action_dim": 4}, model)


class TestRewards:
    @pytest.fixture
    def reward(self):
        # A trivial encoder keeps the test offline -- no torchvision download.
        encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 64 * 64, 16))
        return REWARDS.build({"name": "goal_image", "encoder": encoder, "mode": "delta"})

    def test_reports_uncertainty(self, reward, tiny_frames):
        """ADR-B03: every reward must say when it is guessing."""
        from common.types import Observation

        reward.reset(tiny_frames)
        r, info = reward(Observation(image=tiny_frames[:, -2:]))
        assert r.shape == (2,)
        assert "uncertainty" in info

    def test_delta_mode_is_zero_on_the_first_step(self, reward, tiny_frames):
        from common.types import Observation

        reward.reset(tiny_frames)
        r, _ = reward(Observation(image=tiny_frames[:, -2:]), step=1)
        assert torch.allclose(r, torch.zeros_like(r))

    def test_ensemble_reports_disagreement(self, tiny_frames):
        from common.types import Observation

        def enc():
            return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 64 * 64, 16))

        ens = REWARDS.build({
            "name": "ensemble",
            "components": [
                {"name": "goal_image", "encoder": enc(), "mode": "absolute"},
                {"name": "goal_image", "encoder": enc(), "mode": "absolute"},
            ],
            "disagreement_penalty": 0.5,
        })
        ens.reset(tiny_frames)
        r, info = ens(Observation(image=tiny_frames[:, -2:]))
        assert r.shape == (2,)
        assert "disagreement" in info and "uncertainty" in info


class TestWorldModelEnv:
    def _env(self, tiny_wm_config, tiny_frames):
        cfg = dict(tiny_wm_config)
        cfg["dynamics"] = dict(cfg["dynamics"], action_kind="robot", robot_action_dim=4)
        cfg.pop("latent_action")
        model = wm.GenieWorldModel.from_config(cfg)

        encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 64 * 64, 16))
        reward = REWARDS.build({"name": "goal_image", "encoder": encoder, "mode": "delta"})
        translator = build_translator({"name": "identity", "action_dim": 4}, model)

        return WorldModelEnv(
            world_model=model,
            reward_model=reward,
            translator=translator,
            context_provider=lambda b: tiny_frames[:b, :2],
            max_steps=3,
            device="cpu",
        )

    def test_reset_and_step(self, tiny_wm_config, tiny_frames):
        env = self._env(tiny_wm_config, tiny_frames)
        obs = env.reset(batch_size=2)
        assert obs.image.shape[0] == 2

        result = env.step(torch.randn(2, 4))
        assert result.reward.shape == (2,)
        assert result.done.shape == (2,)
        assert "uncertainty" in result.info

    def test_episode_terminates_at_max_steps(self, tiny_wm_config, tiny_frames):
        env = self._env(tiny_wm_config, tiny_frames)
        env.reset(batch_size=2)
        for step in range(3):
            result = env.step(torch.randn(2, 4))
        assert bool(result.done.all())

    def test_is_imagined_is_declared(self, tiny_wm_config, tiny_frames):
        """ADR-B04: in-dream numbers must be labelled as such."""
        from bridge.envs.base import BaseEnv

        env = self._env(tiny_wm_config, tiny_frames)
        assert isinstance(env, BaseEnv)
        assert hasattr(env, "is_imagined")

    def test_rollout_frames_collected(self, tiny_wm_config, tiny_frames):
        env = self._env(tiny_wm_config, tiny_frames)
        env.reset(batch_size=2)
        env.step(torch.randn(2, 4))
        env.step(torch.randn(2, 4))
        assert env.rollout_frames().shape == (2, 3, 3, 64, 64)   # context + 2 imagined


class TestRLContract:
    def test_rl_rejects_deterministic_head(self, tiny_policy_config):
        """ADR-008 again, at the algorithm level: a run that optimises nothing is
        the most expensive possible bug."""
        import vla
        from vla.rl.base import RLAlgorithm

        class Dummy(RLAlgorithm):
            def update(self, rollout):
                return {}

        policy = vla.POLICIES.build(dict(tiny_policy_config, head={"name": "continuous_mse"}))
        with pytest.raises(ValueError, match="no action distribution"):
            Dummy(policy)

        ok = vla.POLICIES.build(dict(tiny_policy_config, head={"name": "gaussian"}))
        Dummy(ok)   # must not raise
