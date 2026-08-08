"""Layer 2 -- VLA contract conformance.

The most important test in this file is `test_module_is_identity_at_init`. That
property is what makes attaching a new idea to frozen pretrained weights safe
(ADR-002), and it is easy to break by accident when writing a new module.
"""

from __future__ import annotations

import pytest
import torch

import vla
from common.types import ActionSpec, Observation
from vla.core.base import VLAPolicy
from vla.modules import build_modules
from vla.modules.base import PolicyModule


class TestRegistrationIsComplete:
    def test_expected_components(self):
        for name in ("octo_torch", "octo_small", "octo_medium"):
            assert name in vla.POLICIES
        for name in ("continuous_mse", "gaussian", "discrete_bins"):
            assert name in vla.HEADS
        for name in ("bottleneck_adapter", "gated_residual", "wm_conditioning"):
            assert name in vla.MODULES
        assert "openx_npz" in vla.VLA_DATASETS


class TestPolicyContract:
    @pytest.fixture
    def policy(self, tiny_policy_config):
        return vla.POLICIES.build(tiny_policy_config)

    def test_is_a_vla_policy(self, policy):
        assert isinstance(policy, VLAPolicy)

    def test_encode_shape(self, policy, tiny_observation):
        features = policy.encode(tiny_observation)
        assert features.shape == (2, policy.spec.obs_horizon, policy.dim)

    def test_forward_returns_action_chunk(self, policy, tiny_observation):
        action = policy(tiny_observation)
        assert action.continuous.shape == (2, policy.spec.action_chunk, policy.spec.action_dim)
        assert action.first.shape == (2, policy.spec.action_dim)
        assert action.continuous.abs().max() <= 1.0, "actions must stay in normalised space"

    def test_loss_is_differentiable(self, policy, tiny_observation):
        target = torch.randn(2, policy.spec.action_chunk, policy.spec.action_dim).tanh()
        loss, metrics = policy.loss(tiny_observation, target)
        assert loss.ndim == 0 and loss.requires_grad
        loss.backward()
        assert any(p.grad is not None for p in policy.parameters())

    def test_short_history_is_padded_not_crashed(self, policy):
        """An env at t=0 has one frame; the policy must handle it."""
        obs = Observation(image=torch.randn(2, 1, 3, 64, 64), instruction=["a", "b"])
        assert policy(obs).continuous.shape[0] == 2

    def test_freeze_backbone_leaves_head_trainable(self, policy):
        before = len(policy.trainable_parameters())
        policy.freeze_backbone(True)
        after = len(policy.trainable_parameters())
        assert 0 < after < before, "freezing must leave the head trainable, not everything"

    def test_act_denormalizes(self, policy, tiny_observation):
        spec = ActionSpec(
            dim=4,
            q01=torch.tensor([-2.0, -2.0, -2.0, -1.0]),
            q99=torch.tensor([2.0, 2.0, 2.0, 1.0]),
        )
        raw = policy(tiny_observation).first
        physical = policy.act(tiny_observation, action_spec=spec)
        assert physical.shape == raw.shape
        assert physical.abs().max() > 0


class TestHeads:
    @pytest.mark.parametrize("name", ["continuous_mse", "gaussian", "discrete_bins"])
    def test_shapes_and_loss(self, name):
        head = vla.HEADS.build({"name": name, "dim": 32, "action_dim": 4, "action_chunk": 2})
        features = torch.randn(3, 2, 32)
        target = torch.randn(3, 2, 4).tanh()

        assert head(features).shape == (3, 2, 4)
        loss, metrics = head.loss(features, target)
        assert loss.ndim == 0 and loss.requires_grad
        assert "action_l1" in metrics

    def test_rl_capability_is_declared_honestly(self):
        """A head claiming RL support must return a real log-prob (ADR-008)."""
        for name, expected in [("continuous_mse", False), ("gaussian", True),
                               ("discrete_bins", True)]:
            head = vla.HEADS.build({"name": name, "dim": 32, "action_dim": 4})
            assert head.supports_rl is expected

            action, logp = head.sample(torch.randn(3, 2, 32))
            assert action.shape == (3, 1, 4)
            if expected:
                assert logp is not None and logp.shape == (3,)
            else:
                assert logp is None

    def test_discrete_bin_roundtrip(self):
        head = vla.HEADS.build({"name": "discrete_bins", "dim": 32, "action_dim": 4,
                                "n_bins": 256})
        actions = torch.tensor([[[-1.0, 0.0, 0.5, 1.0]]])
        recovered = head.from_bins(head.to_bins(actions))
        assert torch.allclose(recovered, actions, atol=1.0 / 255)


class TestModules:
    ALL = ["bottleneck_adapter", "gated_residual"]

    @pytest.mark.parametrize("name", ALL)
    def test_module_is_identity_at_init(self, name):
        """ADR-002. Without this, attaching a module to frozen pretrained weights
        destroys the pretrained representation in the first few hundred steps."""
        stack = build_modules([{"name": name}], 32)
        features = torch.randn(2, 4, 32)
        assert torch.allclose(stack(features), features, atol=1e-5), (
            f"{name} is not identity at init -- zero-initialise its output "
            "projection or start its gate closed"
        )

    @pytest.mark.parametrize("name", ALL)
    def test_module_preserves_shape(self, name):
        stack = build_modules([{"name": name}], 32)
        assert stack(torch.randn(2, 4, 32)).shape == (2, 4, 32)

    @pytest.mark.parametrize("name", ALL)
    def test_module_is_trainable(self, name):
        """Identity at init must not mean permanently identity."""
        stack = build_modules([{"name": name}], 32)
        out = stack(torch.randn(2, 4, 32))
        out.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stack.parameters())

    def test_modules_stack(self):
        stack = build_modules(
            [{"name": "bottleneck_adapter"}, {"name": "gated_residual"}], 32
        )
        assert len(stack) == 2
        features = torch.randn(2, 4, 32)
        assert torch.allclose(stack(features), features, atol=1e-5)

    def test_missing_context_is_a_clear_error(self):
        stack = build_modules([{"name": "wm_conditioning", "latent_dim": 16}], 32)
        with pytest.raises(KeyError, match="wm_latents"):
            stack(torch.randn(2, 4, 32), context={})

    def test_wm_conditioning_is_identity_at_init(self):
        stack = build_modules([{"name": "wm_conditioning", "latent_dim": 16}], 32)
        features = torch.randn(2, 4, 32)
        context = {"wm_latents": torch.randn(2, 3, 16, 4, 4)}
        assert torch.allclose(stack(features, context), features, atol=1e-5)

    def test_dim_mismatch_in_stack_is_rejected(self):
        from vla.modules.base import ModuleStack

        a = vla.MODULES.build({"name": "bottleneck_adapter", "dim": 32})
        b = vla.MODULES.build({"name": "bottleneck_adapter", "dim": 64})
        with pytest.raises(ValueError, match="expects dim"):
            ModuleStack([a, b], 32)


class TestPolicyWithModules:
    def test_module_config_flows_through(self, tiny_policy_config, tiny_observation):
        cfg = dict(tiny_policy_config,
                   modules=[{"name": "bottleneck_adapter", "bottleneck": 8}],
                   head={"name": "gaussian"})
        policy = vla.POLICIES.build(cfg)
        assert len(policy.modules_stack) == 1

        action = policy.sample(tiny_observation)
        assert action.logp is not None

    def test_rl_sample_refuses_a_deterministic_head(self, tiny_policy_config, tiny_observation):
        """ADR-008: fail loudly rather than optimise nothing."""
        policy = vla.POLICIES.build(dict(tiny_policy_config, head={"name": "continuous_mse"}))
        with pytest.raises(RuntimeError, match="no action distribution"):
            policy.sample(tiny_observation)


class TestActionSpec:
    def test_normalize_denormalize_roundtrip(self):
        spec = ActionSpec(
            dim=3,
            q01=torch.tensor([-1.0, -0.5, 0.0]),
            q99=torch.tensor([1.0, 0.5, 2.0]),
        )
        actions = torch.tensor([[0.0, 0.25, 1.0]])
        assert torch.allclose(spec.denormalize(spec.normalize(actions)), actions, atol=1e-5)

    def test_normalized_range_is_clamped(self):
        spec = ActionSpec(dim=1, q01=torch.tensor([-1.0]), q99=torch.tensor([1.0]))
        assert spec.normalize(torch.tensor([[99.0]])).max() <= 1.0
