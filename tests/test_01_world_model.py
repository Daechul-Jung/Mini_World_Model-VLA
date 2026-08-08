"""Layer 1 -- world-model contract conformance.

Every registered component is checked against its ABC's shape contract. A new
tokenizer / LAM / dynamics / decoder passes these tests or it is not swappable,
which is the only property the rest of the repo relies on.
"""

from __future__ import annotations

import pytest
import torch

import world_model as wm
from world_model.core.base import Decoder, Dynamics, LatentActionModel, VideoTokenizer


class TestRegistrationIsComplete:
    def test_expected_components_are_registered(self):
        assert "conv_vqvae" in wm.TOKENIZERS
        assert "vq" in wm.QUANTIZERS
        assert "vq_lam" in wm.LATENT_ACTIONS
        assert "causal_gpt" in wm.DYNAMICS
        assert "diffusion_unet" in wm.DECODERS

    def test_stages_are_registered(self):
        import world_model.training  # noqa: F401
        from common.stages import STAGES

        for stage in ("stage_a_tokenizer", "stage_b_latent_action",
                      "stage_c_dynamics", "stage_d_decoder"):
            assert stage in STAGES


class TestTokenizerContract:
    @pytest.fixture
    def tokenizer(self, tiny_wm_config):
        return wm.TOKENIZERS.build(tiny_wm_config["tokenizer"])

    def test_is_a_video_tokenizer(self, tokenizer):
        assert isinstance(tokenizer, VideoTokenizer)

    def test_encode_shapes(self, tokenizer, tiny_frames):
        spec = tokenizer.latent_spec
        out = tokenizer.encode(tiny_frames)
        b, t = tiny_frames.shape[:2]
        assert out["latents"].shape == (b, t, spec.dim, *spec.grid)
        assert out["indices"].shape == (b, t, *spec.grid)
        assert out["indices"].max() < spec.vocab_size

    def test_decode_roundtrips_shape(self, tokenizer, tiny_frames):
        out = tokenizer.encode(tiny_frames)
        assert tokenizer.decode(out["latents"]).shape == tiny_frames.shape

    def test_indices_to_latents_matches_encode(self, tokenizer, tiny_frames):
        """The codebook lookup used at rollout must match the one used at encode.

        Asserted in eval mode: an EMA quantizer updates its codebook inside
        `forward`, so in train mode the lookup legitimately reflects the
        post-update codebook while the returned latents reflect the pre-update
        one. Rollout always runs in eval mode, which is where this must hold.
        """
        tokenizer.eval()
        out = tokenizer.encode(tiny_frames)
        assert torch.allclose(
            tokenizer.indices_to_latents(out["indices"]), out["latents"], atol=1e-5
        )

    def test_accepts_single_frames_too(self, tokenizer):
        """Stage A trains on stills; (B, 3, H, W) must work without a time axis."""
        frames = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        out = tokenizer.encode(frames)
        assert out["latents"].ndim == 4
        assert tokenizer.decode(out["latents"]).shape == frames.shape

    def test_forward_returns_recon_loss_metrics(self, tokenizer, tiny_frames):
        recon, loss, metrics = tokenizer(tiny_frames)
        assert recon.shape == tiny_frames.shape
        assert loss.ndim == 0 and loss.requires_grad
        assert not any(torch.is_tensor(v) for v in metrics.values()), \
            "metrics must be plain floats -- tensors here leak the graph into logging"


class TestLatentActionContract:
    @pytest.fixture
    def lam(self, tiny_wm_config):
        return wm.LATENT_ACTIONS.build(tiny_wm_config["latent_action"])

    def test_is_a_latent_action_model(self, lam):
        assert isinstance(lam, LatentActionModel)

    def test_one_action_per_transition(self, lam, tiny_frames):
        """T frames -> T-1 actions. Off-by-one here trains a model on the past."""
        out = lam.infer_actions(tiny_frames)
        b, t = tiny_frames.shape[:2]
        assert out["indices"].shape == (b, t - 1)
        assert out["indices"].max() < lam.action_spec.num_actions

    def test_forward_predicts_next_frames(self, lam, tiny_frames):
        pred, loss, metrics = lam(tiny_frames)
        assert pred.shape == (tiny_frames.shape[0], tiny_frames.shape[1] - 1, *tiny_frames.shape[2:])
        assert "action_perplexity" in metrics, "collapse must be observable"
        assert "actions_used" in metrics

    def test_rejects_single_frame(self, lam):
        with pytest.raises(ValueError, match="at least 2 frames"):
            lam(torch.randn(2, 1, 3, 64, 64))


class TestDynamicsContract:
    @pytest.fixture
    def dynamics(self, tiny_wm_config):
        return wm.DYNAMICS.build(tiny_wm_config["dynamics"])

    def test_is_dynamics(self, dynamics):
        assert isinstance(dynamics, Dynamics)

    def test_forward_teacher_forced(self, dynamics):
        tokens = torch.randint(0, 128, (2, 4, 8, 8))
        actions = torch.randint(0, 8, (2, 3))
        out = dynamics(tokens, actions)
        assert out["loss"].ndim == 0 and out["loss"].requires_grad
        assert 0.0 <= float(out["token_acc"]) <= 1.0

    def test_predict_next_returns_one_frame(self, dynamics):
        tokens = torch.randint(0, 128, (2, 4, 8, 8))
        nxt = dynamics.predict_next(tokens, action=torch.randint(0, 8, (2,)))
        assert nxt.shape == (2, 1, 8, 8)
        assert nxt.max() < 128

    def test_missing_action_is_an_error_not_a_silent_zero(self, dynamics):
        tokens = torch.randint(0, 128, (2, 4, 8, 8))
        with pytest.raises(ValueError, match="requires an action"):
            dynamics.predict_next(tokens)

    def test_robot_action_kind(self, tiny_wm_config):
        cfg = dict(tiny_wm_config["dynamics"], action_kind="robot", robot_action_dim=4)
        dyn = wm.DYNAMICS.build(cfg)
        tokens = torch.randint(0, 128, (2, 4, 8, 8))
        out = dyn(tokens, torch.randn(2, 3, 4))
        assert out["loss"].ndim == 0

    def test_rejects_single_frame_clip(self, dynamics):
        with pytest.raises(ValueError, match="at least 2 frames"):
            dynamics(torch.randint(0, 128, (2, 1, 8, 8)), torch.randint(0, 8, (2, 0)))


class TestComposition:
    def test_builds_and_reports(self, tiny_wm_config):
        model = wm.GenieWorldModel.from_config(tiny_wm_config)
        assert isinstance(model.tokenizer, VideoTokenizer)
        assert isinstance(model.dynamics, Dynamics)
        assert isinstance(model.decoder, Decoder)
        assert "GenieWorldModel" in model.describe()

    def test_spec_mismatch_fails_at_construction(self, tiny_wm_config):
        """The whole point of LatentSpec: fail now, not three hours into stage C."""
        cfg = dict(tiny_wm_config)
        cfg["dynamics"] = dict(cfg["dynamics"], vocab_size=999)
        with pytest.raises(ValueError, match="vocab mismatch"):
            wm.GenieWorldModel.from_config(cfg)

    def test_lam_dynamics_action_mismatch_fails(self, tiny_wm_config):
        cfg = dict(tiny_wm_config)
        cfg["latent_action"] = dict(cfg["latent_action"], num_actions=16)
        with pytest.raises(ValueError, match="codebook size"):
            wm.GenieWorldModel.from_config(cfg)

    def test_imagine_tokenizer_render(self, tiny_wm_config, tiny_frames):
        model = wm.GenieWorldModel.from_config(tiny_wm_config)
        result = model.imagine(tiny_frames[:, :2], actions=3, n_steps=3, render="tokenizer")
        assert result.frames.shape == (2, 3, 3, 64, 64)
        assert result.indices.shape == (2, 3, 8, 8)

    def test_imagine_decoder_render(self, tiny_wm_config, tiny_frames):
        model = wm.GenieWorldModel.from_config(tiny_wm_config)
        result = model.imagine(tiny_frames[:, :2], actions=[1, 2], n_steps=2,
                               render="decoder", decoder_steps=2)
        assert result.frames.shape == (2, 2, 3, 64, 64)

    def test_imagine_restores_training_mode(self, tiny_wm_config, tiny_frames):
        model = wm.GenieWorldModel.from_config(tiny_wm_config)
        model.train()
        model.imagine(tiny_frames[:, :2], actions=0, n_steps=1)
        assert model.training, "imagine() must not silently leave the model in eval mode"

    def test_wrong_action_count_is_rejected(self, tiny_wm_config, tiny_frames):
        model = wm.GenieWorldModel.from_config(tiny_wm_config)
        with pytest.raises(ValueError, match="actions for"):
            model.imagine(tiny_frames[:, :2], actions=torch.zeros(2, 5, dtype=torch.long),
                          n_steps=3)


class TestMetrics:
    def test_codebook_usage_detects_collapse(self):
        from common.metrics import codebook_usage, perplexity

        collapsed = torch.zeros(1000, dtype=torch.long)
        healthy = torch.randint(0, 128, (1000,))
        assert codebook_usage(collapsed, 128) < 0.02
        assert codebook_usage(healthy, 128) > 0.5
        assert perplexity(collapsed, 128) < 1.1
        assert perplexity(healthy, 128) > 50

    def test_psnr_is_high_for_identical_images(self):
        from common.metrics import psnr

        x = torch.rand(2, 3, 32, 32)
        assert psnr(x, x) > 60
        assert psnr(x, torch.rand(2, 3, 32, 32)) < 20


class TestQuantizers:
    """Every quantizer must satisfy the same interface, so `tokenizer.quantizer.name`
    is a genuine one-line swap. Also pins the property that motivated `vq_ema` and
    `fsq` existing: plain `vq` collapses on this data (see IMPROVEMENTS.md)."""

    CONFIGS = [
        {"name": "vq", "beta": 0.25},
        {"name": "vq_ema", "restart_every": 5},
        {"name": "fsq", "levels": [4, 4, 4, 4, 4]},
    ]

    @pytest.mark.parametrize("qcfg", CONFIGS, ids=lambda c: c["name"])
    def test_interface(self, qcfg):
        q = wm.QUANTIZERS.build(dict(qcfg, num_embeddings=1024, embedding_dim=32))
        z = torch.randn(2, 32, 8, 8)
        z_q, loss, indices = q(z)

        assert z_q.shape == z.shape
        assert loss.ndim == 0
        assert indices.shape == (2, 8, 8)
        assert indices.max() < q.num_embeddings and indices.min() >= 0
        assert q.decode_indices(indices).shape == z.shape

    @pytest.mark.parametrize("qcfg", CONFIGS, ids=lambda c: c["name"])
    def test_gradients_reach_the_encoder(self, qcfg):
        """Straight-through must pass gradient back through the bottleneck."""
        q = wm.QUANTIZERS.build(dict(qcfg, num_embeddings=1024, embedding_dim=32))
        z = torch.randn(2, 32, 8, 8, requires_grad=True)
        q(z)[0].sum().backward()
        assert z.grad is not None and z.grad.abs().sum() > 0

    @pytest.mark.parametrize("qcfg", CONFIGS[1:], ids=lambda c: c["name"])
    def test_collapse_resistant_quantizers_spread_codes(self, qcfg):
        """The property `vq` fails on real data: use many codes, not a handful.

        With 2048 distinct input vectors a healthy quantizer should reach far
        more than the ~5 effective codes plain VQ collapses to.
        """
        from common.metrics import perplexity

        q = wm.QUANTIZERS.build(dict(qcfg, num_embeddings=1024, embedding_dim=32))
        q.train()
        torch.manual_seed(0)
        for _ in range(20):
            _, _, indices = q(torch.randn(8, 32, 16, 16))
        assert perplexity(indices, q.num_embeddings) > 20

    def test_fsq_vocab_is_the_product_of_levels(self):
        q = wm.QUANTIZERS.build({"name": "fsq", "embedding_dim": 32, "levels": [8, 5, 5, 5]})
        assert q.num_embeddings == 8 * 5 * 5 * 5

    def test_fsq_index_roundtrip_is_exact(self):
        """Indices are a mixed-radix encoding of the level grid; the decode must
        invert it exactly or rollout frames will not match training tokens."""
        q = wm.QUANTIZERS.build({"name": "fsq", "embedding_dim": 32, "levels": [8, 5, 5, 5]})
        codes = q._quantize(torch.randn(2, 4, 8, 8))
        indices = q.codes_to_indices(codes)
        assert torch.allclose(q.indices_to_codes(indices), codes, atol=1e-5)

    @pytest.mark.parametrize("qcfg", CONFIGS, ids=lambda c: c["name"])
    def test_tokenizer_accepts_any_quantizer(self, qcfg):
        """The actual swappability claim, end to end through the tokenizer."""
        tok = wm.TOKENIZERS.build({
            "name": "conv_vqvae", "base_channels": 16, "channel_mults": [1, 2],
            "latent_dim": 32, "n_res_blocks": 1, "image_size": 64,
            "codebook_size": 1024, "quantizer": qcfg,
        })
        frames = torch.randn(2, 2, 3, 64, 64).clamp(-1, 1)
        recon, loss, _ = tok(frames)
        assert recon.shape == frames.shape
        assert loss.ndim == 0

        tok.eval()
        enc = tok.encode(frames)
        assert torch.allclose(tok.indices_to_latents(enc["indices"]), enc["latents"], atol=1e-5)


class TestQuantizersUnderAutocast:
    """The training loop runs under bf16 autocast; these tests originally did not.

    That gap let a real bug ship: `vq_ema` keeps its codebook in fp32 (an EMA over
    hundreds of steps needs the mantissa) while the encoder output arrives in
    bf16, and `codebook[dead] = fresh` is an index-put, which raises on a dtype
    mismatch instead of promoting. It crashed several minutes into a real run.
    """

    @pytest.mark.parametrize("qcfg", TestQuantizers.CONFIGS, ids=lambda c: c["name"])
    def test_forward_and_backward_under_bf16(self, qcfg):
        q = wm.QUANTIZERS.build(dict(qcfg, num_embeddings=256, embedding_dim=32))
        q.train()
        for _ in range(3):
            z = torch.randn(2, 32, 8, 8, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                z_q, loss, indices = q(z)
            (z_q.float().sum() + loss.float()).backward()
            assert indices.max() < q.num_embeddings

    def test_vq_ema_restart_path_under_bf16(self):
        """Force the dead-code restart to fire while autocast is active."""
        q = wm.QUANTIZERS.build({
            "name": "vq_ema", "num_embeddings": 256, "embedding_dim": 32,
            "restart_every": 1, "restart_threshold": 1e9,   # every code counts as dead
        })
        q.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            q(torch.randn(2, 32, 8, 8))
            q(torch.randn(2, 32, 8, 8))
        assert q.codebook.dtype == torch.float32, "EMA state must stay fp32"

    @pytest.mark.parametrize("qcfg", TestQuantizers.CONFIGS, ids=lambda c: c["name"])
    def test_tokenizer_trains_under_bf16(self, qcfg):
        """End to end through the tokenizer, matching what `train_stage` does."""
        tok = wm.TOKENIZERS.build({
            "name": "conv_vqvae", "base_channels": 16, "channel_mults": [1, 2],
            "latent_dim": 32, "n_res_blocks": 1, "image_size": 32,
            "codebook_size": 256, "quantizer": qcfg,
        })
        opt = torch.optim.AdamW(tok.parameters(), lr=1e-4)
        for _ in range(3):
            frames = torch.randn(2, 3, 32, 32).clamp(-1, 1)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, loss, _ = tok(frames)
            loss.float().backward()
            opt.step()
            opt.zero_grad()


class TestLatentActionCodebookHealth:
    """The LAM's action codebook is the component with the least margin.

    |A| = 8 means losing six codes leaves a world model that cannot be steered --
    and it fails silently, because `rec_loss` keeps improving while the decoder
    learns to predict the next frame from the past alone. A measured 33-epoch run
    on TUM collapsed from 4.3 to 2.0 live codes exactly this way, because the LAM
    had its own hard-coded `vq` default that the tokenizer's config fix did not
    cover.
    """

    def test_default_quantizer_is_collapse_resistant(self):
        lam = wm.LATENT_ACTIONS.build({
            "name": "vq_lam", "image_size": 64, "patch_size": 16,
            "num_actions": 8, "dim": 64, "depth": 1, "num_heads": 2, "action_dim": 32,
        })
        assert type(lam.quantizer).__name__ != "VectorQuantizer", (
            "the LAM must not default to plain `vq` -- it has no dead-code revival "
            "and an 8-entry codebook cannot afford to lose entries"
        )

    def test_action_codebook_works_for_every_quantizer(self):
        """`action_codebook()` is how you find out what the codes mean; it must
        not assume a particular quantizer's internal attribute name."""
        for qcfg in ({"name": "vq"}, {"name": "vq_ema"}):
            lam = wm.LATENT_ACTIONS.build({
                "name": "vq_lam", "image_size": 64, "patch_size": 16,
                "num_actions": 8, "dim": 64, "depth": 1, "num_heads": 2,
                "action_dim": 32, "quantizer": qcfg,
            })
            book = lam.action_codebook()
            assert book.shape == (8, 32), f"{qcfg['name']}: got {tuple(book.shape)}"

    def test_dead_action_codes_are_revived(self):
        """The mechanism that prevents the collapse, tested directly.

        Rather than hoping a short synthetic training run rediscovers the codes
        (flaky), force every code to count as dead and assert the quantizer
        re-seeds them. Plain `vq` has no such path, which is exactly why it is
        no longer the default here.
        """
        lam = wm.LATENT_ACTIONS.build({
            "name": "vq_lam", "image_size": 64, "patch_size": 16,
            "num_actions": 8, "dim": 64, "depth": 1, "num_heads": 2, "action_dim": 32,
            "quantizer": {"name": "vq_ema", "restart_every": 1, "restart_threshold": 1e9},
        })
        lam.train()
        frames = torch.randn(2, 4, 3, 64, 64).clamp(-1, 1)
        lam(frames)
        before = lam.action_codebook().clone()
        lam(frames)
        after = lam.action_codebook()

        assert not torch.allclose(before, after), "dead codes were never re-seeded"
        assert (lam.quantizer.cluster_size > 0).all(), "some code left permanently dead"
